//! End-to-end media-to-prediction pipeline.
//!
//! Mirrors the Python `TribeModel.get_events_dataframe()` + `predict()` workflow:
//! 1. Accept raw media (video path, audio path, or text path/string)
//! 2. Extract audio from video (ffmpeg)
//! 3. Transcribe speech to words (whisperX)
//! 4. Extract features for each modality (LLaMA, Wav2Vec-BERT, V-JEPA2)
//! 5. Run TRIBE v2 forward pass
//! 6. Return predictions + segment metadata
//!
//! For text-only input without audio:
//! - Text → TTS (gtts-cli or system `say`) → audio file
//! - Audio → whisperX → word events → LLaMA features
//!
//! Feature extraction backends:
//! - Text: LLaMA 3.2-3B via llama-cpp (always available)
//! - Audio: Wav2Vec-BERT 2.0 via tribev2-audio burn crate (optional)
//! - Video: V-JEPA2 ViT-G via tribev2-video burn crate (optional)
//! - All modalities: pre-extracted binary features (always available)

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use anyhow::{Context, Result};

use crate::events::{self, EventList};
use crate::features::{self, LlamaFeatureConfig};
use crate::model::tribe::TribeV2;
use crate::tensor::Tensor;

/// Input specification for the pipeline.
#[derive(Debug, Clone, Default)]
pub struct PipelineInput {
    /// Path to video file (.mp4, .avi, .mkv, .mov, .webm)
    pub video_path: Option<String>,
    /// Path to audio file (.wav, .mp3, .flac, .ogg)
    pub audio_path: Option<String>,
    /// Path to text file (.txt)
    pub text_path: Option<String>,
    /// Raw text string (alternative to text_path)
    pub text: Option<String>,
    /// Path to LLaMA GGUF model (required for text feature extraction)
    pub llama_model: Option<String>,
    /// Pre-extracted feature files (bypass extraction)
    pub text_features_path: Option<String>,
    pub audio_features_path: Option<String>,
    pub video_features_path: Option<String>,
    /// Cache directory for intermediate files (TTS audio, extracted audio, etc.)
    pub cache_dir: String,
}

/// Pipeline configuration.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Layer positions for feature extraction (default: [0.5, 0.75, 1.0])
    pub layer_positions: Vec<f64>,
    /// Feature frequency in Hz (default: 2.0)
    pub frequency: f64,
    /// Whether to remove empty segments
    pub remove_empty_segments: bool,
    /// Segment duration in TRs
    pub segment_duration: usize,
    /// Verbose logging
    pub verbose: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            layer_positions: vec![0.5, 0.75, 1.0],
            frequency: 2.0,
            remove_empty_segments: true,
            segment_duration: 100,
            verbose: false,
        }
    }
}

/// Pipeline output.
#[derive(Debug)]
pub struct PipelineOutput {
    /// Per-timestep predictions: [n_timesteps][n_vertices]
    pub predictions: Vec<Vec<f32>>,
    /// Events extracted from media
    pub events: EventList,
    /// Duration of input media in seconds
    pub duration_secs: f64,
    /// Features per modality (for inspection/saving)
    pub features: BTreeMap<String, Tensor>,
    /// Number of active (non-zero) modalities
    pub n_active_modalities: usize,
}

/// Text-to-speech: convert text to audio file.
///
/// Tries (in order):
/// 1. `gtts-cli` (pip install gTTS)
/// 2. macOS `say` command
/// 3. `espeak` (Linux)
pub fn text_to_speech(text: &str, output_path: &Path) -> Result<()> {
    // Try gtts-cli first
    let gtts = std::process::Command::new("gtts-cli")
        .args(["--output", &output_path.to_string_lossy()])
        .arg(text)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();

    if let Ok(status) = gtts {
        if status.success() {
            eprintln!("  TTS: generated audio via gtts-cli");
            return Ok(());
        }
    }

    // Try macOS say
    let aiff_path = output_path.with_extension("aiff");
    let say = std::process::Command::new("say")
        .args(["-o", &aiff_path.to_string_lossy()])
        .arg(text)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();

    if let Ok(status) = say {
        if status.success() {
            // Convert AIFF to WAV via ffmpeg
            let ffmpeg = std::process::Command::new("ffmpeg")
                .args(["-y", "-i", &aiff_path.to_string_lossy()])
                .args(["-ar", "16000", "-ac", "1"])
                .arg(&output_path.to_string_lossy().to_string())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status();
            let _ = std::fs::remove_file(&aiff_path);
            if let Ok(s) = ffmpeg {
                if s.success() {
                    eprintln!("  TTS: generated audio via macOS say");
                    return Ok(());
                }
            }
        }
    }

    // Try espeak
    let espeak = std::process::Command::new("espeak")
        .args(["-w", &output_path.to_string_lossy()])
        .arg(text)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();

    if let Ok(status) = espeak {
        if status.success() {
            eprintln!("  TTS: generated audio via espeak");
            return Ok(());
        }
    }

    anyhow::bail!(
        "No TTS engine found. Install one of:\n\
         - pip install gTTS  (then gtts-cli is available)\n\
         - macOS: 'say' command (built-in)\n\
         - Linux: apt install espeak"
    )
}

/// Extract audio track from a video file.
pub fn extract_audio_from_video(video_path: &str, output_path: &Path) -> Result<()> {
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-i", video_path])
        .args(["-vn", "-ar", "16000", "-ac", "1", "-f", "wav"])
        .arg(&output_path.to_string_lossy().to_string())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .with_context(|| "ffmpeg not found")?;

    if !status.success() {
        anyhow::bail!("ffmpeg failed to extract audio from {}", video_path);
    }
    Ok(())
}

/// Get media duration in seconds.
pub fn get_duration(path: &str) -> Result<f64> {
    let output = std::process::Command::new("ffprobe")
        .args(["-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0"])
        .arg(path)
        .output()
        .with_context(|| "ffprobe not found")?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout.trim().parse::<f64>()
        .with_context(|| format!("failed to parse duration from '{}'", stdout.trim()))
}

/// Run the full media-to-prediction pipeline.
///
/// This is the main entry point — give it media files and get brain predictions.
pub fn predict_from_media(
    model: &TribeV2,
    input: &PipelineInput,
    config: &PipelineConfig,
) -> Result<PipelineOutput> {
    let cache = Path::new(&input.cache_dir);
    std::fs::create_dir_all(cache)?;

    let mut events = EventList::new();
    let mut audio_path: Option<PathBuf> = None;
    let mut video_path: Option<PathBuf> = None;
    let mut duration_secs: f64 = 0.0;

    // ── Step 1: Resolve input media ───────────────────────────────────
    if let Some(ref vp) = input.video_path {
        eprintln!("Pipeline: video input {}", vp);
        duration_secs = get_duration(vp)?;
        video_path = Some(PathBuf::from(vp));

        // Extract audio from video
        let audio_out = cache.join("extracted_audio.wav");
        extract_audio_from_video(vp, &audio_out)?;
        audio_path = Some(audio_out);

        events.push(crate::events::Event::video(vp, 0.0, duration_secs));
    }

    if let Some(ref ap) = input.audio_path {
        eprintln!("Pipeline: audio input {}", ap);
        if duration_secs == 0.0 {
            duration_secs = get_duration(ap)?;
        }
        audio_path = Some(PathBuf::from(ap));
        events.push(crate::events::Event::audio(ap, 0.0, duration_secs));
    }

    if let Some(ref tp) = input.text_path {
        eprintln!("Pipeline: text file input {}", tp);
        let text = std::fs::read_to_string(tp)
            .with_context(|| format!("failed to read text: {}", tp))?;
        return predict_from_text_string(model, &text, input, config);
    }

    if let Some(ref text) = input.text {
        eprintln!("Pipeline: text string input ({} chars)", text.len());
        return predict_from_text_string(model, text, input, config);
    }

    if audio_path.is_none() && video_path.is_none()
        && input.text_features_path.is_none()
        && input.audio_features_path.is_none()
        && input.video_features_path.is_none()
    {
        anyhow::bail!("No input specified. Provide --video-path, --audio-path, --text-path, or pre-extracted features.");
    }

    // ── Step 2: Transcribe audio → word events ────────────────────────
    if let Some(ref ap) = audio_path {
        let ap_str = ap.to_string_lossy();
        eprintln!("Pipeline: transcribing audio...");
        match events::transcribe_audio(&ap_str, "english", 0.0) {
            Ok(word_events) => {
                for e in word_events.events {
                    events.push(e);
                }
                events.add_sentence_context();
                events.add_context(1024);
                eprintln!("  {} word events transcribed", events.words().len());
            }
            Err(e) => {
                eprintln!("  Warning: transcription failed ({}), continuing without text", e);
            }
        }
    }

    let n_timesteps = (duration_secs * config.frequency).ceil() as usize;
    if n_timesteps == 0 {
        anyhow::bail!("Duration is 0 — cannot produce predictions");
    }
    eprintln!("Pipeline: duration={:.1}s, timesteps={}", duration_secs, n_timesteps);

    // ── Step 3: Extract features for each modality ────────────────────
    let mut features = BTreeMap::new();

    // Text features
    if let Some(ref path) = input.text_features_path {
        let t = load_preextracted(path, 2, 3072, n_timesteps)?;
        features.insert("text".to_string(), t);
    } else if let Some(ref llama_path) = input.llama_model {
        if !events.words().is_empty() {
            eprintln!("Pipeline: extracting LLaMA text features...");
            let llama_cfg = LlamaFeatureConfig {
                model_path: llama_path.clone(),
                layer_positions: config.layer_positions.clone(),
                n_layers: 28,
                n_ctx: 2048,
                frequency: config.frequency,
            };
            // Build prompt from word events
            let prompt: String = events.words().iter()
                .map(|e| e.text.as_deref().unwrap_or(""))
                .collect::<Vec<_>>()
                .join(" ");
            let text_feats = features::extract_llama_features(&llama_cfg, &prompt, config.verbose)?;
            let text_feats = features::resample_features(&text_feats, n_timesteps);
            let total_dim = text_feats.n_layers * text_feats.feature_dim;
            features.insert("text".to_string(), Tensor::from_vec(
                text_feats.data.data.clone(),
                vec![1, total_dim, text_feats.n_timesteps],
            ));
        }
    }

    // Audio features
    if let Some(ref path) = input.audio_features_path {
        let t = load_preextracted(path, 2, 1024, n_timesteps)?;
        features.insert("audio".to_string(), t);
    }
    // Note: live Wav2Vec-BERT extraction requires tribev2-audio crate
    // which is optional. See extract_audio_features_burn() below.

    // Video features
    if let Some(ref path) = input.video_features_path {
        let t = load_preextracted(path, 2, 1408, n_timesteps)?;
        features.insert("video".to_string(), t);
    }

    // Fill missing modalities with zeros
    for md in &model.feature_dims {
        if !features.contains_key(&md.name) {
            if let Some((n_l, dim)) = md.dims {
                features.insert(md.name.clone(), Tensor::zeros(&[1, n_l * dim, n_timesteps]));
            }
        }
    }

    let n_active = features.iter()
        .filter(|(_, t)| t.data.iter().any(|&v| v != 0.0))
        .count();
    eprintln!("Pipeline: {} / {} active modalities", n_active, features.len());

    // ── Step 4: Run inference ─────────────────────────────────────────
    eprintln!("Pipeline: running inference...");
    let output = model.forward(&features, None, true);
    let n_out = output.shape[1];
    let n_t = output.shape[2];
    let predictions: Vec<Vec<f32>> = (0..n_t)
        .map(|ti| (0..n_out).map(|di| output.data[di * n_t + ti]).collect())
        .collect();

    Ok(PipelineOutput {
        predictions,
        events,
        duration_secs,
        features,
        n_active_modalities: n_active,
    })
}

/// Pipeline for text-only input: text → TTS → audio → transcribe → features → predict.
fn predict_from_text_string(
    model: &TribeV2,
    text: &str,
    input: &PipelineInput,
    config: &PipelineConfig,
) -> Result<PipelineOutput> {
    let cache = Path::new(&input.cache_dir);
    std::fs::create_dir_all(cache)?;

    // Step 1: TTS
    let tts_audio = cache.join("tts_audio.wav");
    eprintln!("Pipeline: converting text to speech...");
    text_to_speech(text, &tts_audio)?;

    // Step 2: Get duration
    let duration_secs = get_duration(&tts_audio.to_string_lossy())?;
    eprintln!("Pipeline: TTS audio duration={:.1}s", duration_secs);

    // Step 3: Transcribe
    let mut events = EventList::new();
    events.push(crate::events::Event::audio(
        &tts_audio.to_string_lossy(), 0.0, duration_secs,
    ));

    match events::transcribe_audio(&tts_audio.to_string_lossy(), "english", 0.0) {
        Ok(word_events) => {
            for e in word_events.events {
                events.push(e);
            }
            events.add_sentence_context();
            events.add_context(1024);
            eprintln!("  {} words transcribed", events.words().len());
        }
        Err(e) => {
            eprintln!("  Transcription failed ({}), using uniform word timing", e);
            let word_events = events::text_to_events(text, duration_secs);
            for e in word_events.events {
                events.push(e);
            }
        }
    }

    // Step 4: Now run the normal pipeline with audio_path set
    let mut new_input = input.clone();
    new_input.audio_path = Some(tts_audio.to_string_lossy().to_string());
    new_input.text_path = None;
    new_input.text = None;

    // Recurse with the audio path
    predict_from_media(model, &new_input, config)
}

/// Load pre-extracted features from a binary f32 file.
fn load_preextracted(path: &str, n_layers: usize, feature_dim: usize, n_timesteps: usize) -> Result<Tensor> {
    // Try JSON sidecar for metadata
    let json_path = Path::new(path).with_extension("json");
    let (n_l, dim, n_t) = if json_path.exists() {
        let meta: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&json_path)?
        )?;
        let nl = meta["n_layers"].as_u64().unwrap_or(n_layers as u64) as usize;
        let d = meta["hidden_dim"].as_u64().unwrap_or(feature_dim as u64) as usize;
        let nt = meta["n_timesteps"].as_u64().unwrap_or(n_timesteps as u64) as usize;
        (nl, d, nt)
    } else {
        (n_layers, feature_dim, n_timesteps)
    };

    let bytes = std::fs::read(path)
        .with_context(|| format!("failed to read features: {}", path))?;
    let data: Vec<f32> = bytes.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let expected = n_l * dim * n_t;
    if data.len() != expected {
        anyhow::bail!("Feature file has {} floats, expected {} ({}×{}×{})",
            data.len(), expected, n_l, dim, n_t);
    }

    Ok(Tensor::from_vec(data, vec![1, n_l * dim, n_t]))
}
