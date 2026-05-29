//! Wav2Vec-BERT 2.0 feature extraction via RLX (`rlx-wav2vec2-bert`).

use anyhow::{Context, Result};
use rlx::Device;
use rlx_wav2vec2_bert::{
    W2vLayerStop, Wav2Vec2BertConfig, Wav2Vec2BertPreprocessConfig, build_wav2vec2_bert_graph_probe,
};
use rlx_models_core::{
    flow_bridge::compile_options_for_profile, load_weight_map, W2V_BERT_GGUF_ARCHES,
};
use rlx_flow::CompileProfile;
use std::path::Path;

use crate::config::AudioFeatureConfig;
use crate::ExtractedAudioFeatures;

fn parse_device(s: &str) -> Result<Device> {
    Ok(match s.trim().to_ascii_lowercase().as_str() {
        "cpu" => Device::Cpu,
        "metal" | "mps" => Device::Metal,
        "mlx" => Device::Mlx,
        "cuda" => Device::Cuda,
        "rocm" | "hip" => Device::Rocm,
        "gpu" | "wgpu" => Device::Gpu,
        "vulkan" => Device::Vulkan,
        other => anyhow::bail!(
            "unknown device {other} (cpu|metal|mlx|cuda|rocm|gpu|vulkan)"
        ),
    })
}

fn mel_seq_for_waveform(
    pre_cfg: &Wav2Vec2BertPreprocessConfig,
    waveform: &[f32],
    seq: usize,
) -> (Vec<f32>, Vec<f32>) {
    use rlx_wav2vec2_bert::LogMelExtractor;
    let extractor = LogMelExtractor::new(pre_cfg.clone());
    let mel = extractor.extract(waveform);
    let padded = extractor.pad_to_seq(mel, seq);
    (padded.features, padded.attention_mask)
}

/// Extract per-layer hidden states and resample to [`AudioFeatureConfig::frequency`] Hz.
pub fn extract_audio_features(
    config: &AudioFeatureConfig,
    waveform: &[f32],
    duration_secs: f64,
    verbose: bool,
) -> Result<ExtractedAudioFeatures> {
    let weights_path = Path::new(&config.weights_path);
    let weights_dir = weights_path
        .parent()
        .context("weights path has no parent directory")?;
    let cfg_path: std::path::PathBuf = config
        .config_path
        .as_ref()
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| weights_dir.join("config.json"));
    let rlx_cfg = Wav2Vec2BertConfig::from_file(&cfg_path)
        .with_context(|| format!("reading {cfg_path:?}"))?;

    let pre_cfg_path: std::path::PathBuf = config
        .preprocessor_config_path
        .as_ref()
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| weights_dir.join("preprocessor_config.json"));
    let pre_cfg = if pre_cfg_path.exists() {
        Wav2Vec2BertPreprocessConfig::from_file(&pre_cfg_path)?
    } else {
        Wav2Vec2BertPreprocessConfig::w2v_bert_2_0()
    };

    let n_total_layers = config.n_layers.unwrap_or(rlx_cfg.num_hidden_layers);
    let layer_indices = config.layer_indices(n_total_layers);
    let hidden_dim = rlx_cfg.hidden_size;
    let n_layer_groups = layer_indices.len();
    let n_timesteps = (duration_secs * config.frequency).ceil() as usize;

    let batch = 1;
    let seq = config
        .max_seq
        .unwrap_or_else(|| estimate_seq(&pre_cfg, waveform.len()));

    if verbose {
        eprintln!(
            "Wav2Vec-BERT (RLX): {} layers, hidden_dim={}, seq={}",
            n_total_layers, hidden_dim, seq
        );
        eprintln!("Extracting layers: {:?}", layer_indices);
    }

    let device = parse_device(&config.device)?;
    let path_str = weights_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("non-UTF-8 weights path"))?;
    let compile_opts = compile_options_for_profile(&CompileProfile::encoder(), device);

    let (mel, mask) = mel_seq_for_waveform(&pre_cfg, waveform, seq);
    let mut layer_outputs: Vec<Vec<f32>> = Vec::with_capacity(n_layer_groups);

    for &layer_idx in &layer_indices {
        let mut wm = load_weight_map(path_str, W2V_BERT_GGUF_ARCHES)?;
        let (graph, params) = build_wav2vec2_bert_graph_probe(
            &rlx_cfg,
            &mut wm,
            batch,
            seq,
            layer_idx,
            W2vLayerStop::AfterFfn2,
        )?;
        let mut compiled = rlx::Session::new(device).compile_with(graph, &compile_opts);
        for (name, data) in &params {
            compiled.set_param(name, data);
        }
        let hidden = compiled
            .run(&[("input_features", mel.as_slice()), ("attention_mask", mask.as_slice())])
            .into_iter()
            .next()
            .context("wav2vec2_bert probe returned no output")?;
        layer_outputs.push(hidden);
    }

    let mut data = vec![0.0f32; n_layer_groups * hidden_dim * n_timesteps.max(1)];
    let t_model = seq;

    for (li, hidden) in layer_outputs.iter().enumerate() {
        for ti in 0..n_timesteps {
            let src_t = ((ti as f64 / n_timesteps.max(1) as f64) * t_model as f64).floor() as usize;
            let src_t = src_t.min(t_model.saturating_sub(1));
            for di in 0..hidden_dim {
                let src = src_t * hidden_dim + di;
                if src < hidden.len() {
                    data[li * hidden_dim * n_timesteps + di * n_timesteps + ti] = hidden[src];
                }
            }
        }
    }

    Ok(ExtractedAudioFeatures {
        data,
        shape: vec![n_layer_groups, hidden_dim, n_timesteps],
        n_layers: n_layer_groups,
        feature_dim: hidden_dim,
        n_timesteps,
    })
}

fn estimate_seq(pre_cfg: &Wav2Vec2BertPreprocessConfig, n_samples: usize) -> usize {
    // ~50 Hz frame rate (320-sample hop at 16 kHz), capped by preprocessor `num_frames`.
    let hop = (pre_cfg.sampling_rate / 50).max(1);
    let frames = n_samples.saturating_div(hop).max(1);
    frames.min(pre_cfg.num_frames.max(1))
}
