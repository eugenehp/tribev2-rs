//! TRIBE v2 inference CLI.
//!
//! Loads a pretrained TRIBE v2 model (safetensors) and runs inference
//! with support for multi-modal features (text, audio, video) and
//! segment-based batching.
//!
//! Usage:
//!   tribev2-infer --config config.yaml --weights model.safetensors \
//!                 --llama-model llama-3.2-3b.gguf --prompt "The quick brown fox"
//!
//! Multi-modal:
//!   tribev2-infer --config config.yaml --weights model.safetensors \
//!                 --llama-model llama.gguf --prompt "Hello world" \
//!                 --n-timesteps 200 --segment
//!
//! With brain visualization:
//!   tribev2-infer --config config.yaml --weights model.safetensors \
//!                 --llama-model llama.gguf --prompt "The quick brown fox" \
//!                 --plot-dir ./plots --view left

use std::collections::BTreeMap;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;

use tribev2::config::{ModalityDims, TribeV2Config};
use tribev2::features::{self, LlamaFeatureConfig};
use tribev2::model::tribe::TribeV2;
use tribev2::plotting;
use tribev2::segments::{self, SegmentConfig};
use tribev2::tensor::Tensor;
use tribev2::weights::{WeightMap, load_weights};

#[derive(Parser, Debug)]
#[command(about = "TRIBE v2 fMRI brain encoding model inference")]
struct Args {
    /// Path to config.yaml
    #[arg(long)]
    config: String,

    /// Path to model.safetensors
    #[arg(long)]
    weights: String,

    /// Path to build_args.json (optional; uses pretrained defaults if not given)
    #[arg(long)]
    build_args: Option<String>,

    /// Path to LLaMA GGUF model for text feature extraction
    #[arg(long)]
    llama_model: Option<String>,

    /// Text prompt for LLaMA feature extraction
    #[arg(long, short = 'p')]
    prompt: Option<String>,

    /// Layer positions to extract (fractional, comma-separated).
    /// Default: 0.5,0.75,1.0 (matching Python pretrained config)
    #[arg(long, default_value = "0.5,0.75,1.0")]
    layer_positions: String,

    /// Path to pre-extracted text features (binary f32, [n_timesteps, feature_dim])
    #[arg(long)]
    text_features: Option<String>,

    /// Path to pre-extracted audio features (binary f32, [n_timesteps, feature_dim])
    #[arg(long)]
    audio_features: Option<String>,

    /// Path to pre-extracted video features (binary f32, [n_timesteps, feature_dim])
    #[arg(long)]
    video_features: Option<String>,

    /// Number of output timesteps to predict
    #[arg(long, default_value = "100")]
    n_timesteps: usize,

    /// Feature dimension for pre-extracted features
    #[arg(long)]
    feature_dim: Option<usize>,

    /// Number of layers in pre-extracted features
    #[arg(long)]
    n_layers: Option<usize>,

    /// Subject index for per-subject prediction (0-indexed).
    /// If not specified, predictions are averaged across all subjects.
    #[arg(long)]
    subject: Option<usize>,

    /// Enable segment-based batching
    #[arg(long)]
    segment: bool,

    /// Segment duration in TRs (timesteps)
    #[arg(long, default_value = "100")]
    segment_duration: usize,

    /// Segment overlap in TRs
    #[arg(long, default_value = "0")]
    segment_overlap: usize,

    /// Remove empty segments (segments with all-zero features)
    #[arg(long)]
    remove_empty: bool,

    /// Output file for predictions (binary f32)
    #[arg(long)]
    output: Option<String>,

    /// Output directory for brain surface plots (SVG)
    #[arg(long)]
    plot_dir: Option<String>,

    /// View for brain plots: left, right, medial_left, medial_right, dorsal, ventral
    #[arg(long, default_value = "left")]
    view: String,

    /// Color map for plots: hot, coolwarm, viridis, seismic
    #[arg(long, default_value = "hot")]
    cmap: String,

    /// Include colorbar in plots
    #[arg(long)]
    colorbar: bool,

    /// FreeSurfer subjects directory (for real brain meshes)
    #[arg(long)]
    subjects_dir: Option<String>,

    /// fsaverage mesh resolution (fsaverage3-6)
    #[arg(long, default_value = "fsaverage5")]
    mesh: String,

    /// Print verbose info
    #[arg(long, short = 'v')]
    verbose: bool,
}

/// Metadata sidecar for pre-extracted features (from extract_llama_features.py).
#[derive(serde::Deserialize, Default)]
#[allow(dead_code)]
struct FeatureMeta {
    #[serde(default)]
    shape: Vec<usize>,
    #[serde(default)]
    n_layers: usize,
    #[serde(default)]
    hidden_dim: usize,
    #[serde(default)]
    n_timesteps: usize,
}

/// Load pre-extracted features from a binary f32 file.
///
/// File format: flat f32 array, shape [n_layers, feature_dim, n_timesteps].
/// If a `.json` sidecar exists, reads shape metadata from it.
///
/// Returns Tensor of shape [1, n_layers * feature_dim, n_timesteps] for model input.
fn load_preextracted_features(
    path: &str,
    n_layers: usize,
    feature_dim: usize,
    n_timesteps: usize,
) -> Result<Tensor> {
    // Try to read sidecar metadata
    let json_path = std::path::Path::new(path).with_extension("json");
    let (n_l, dim, n_t) = if json_path.exists() {
        let meta: FeatureMeta = serde_json::from_str(
            &std::fs::read_to_string(&json_path)?
        ).unwrap_or_default();
        if meta.n_layers > 0 && meta.hidden_dim > 0 && meta.n_timesteps > 0 {
            eprintln!("  Loaded metadata from {}: [{}, {}, {}]",
                json_path.display(), meta.n_layers, meta.hidden_dim, meta.n_timesteps);
            (meta.n_layers, meta.hidden_dim, meta.n_timesteps)
        } else {
            (n_layers, feature_dim, n_timesteps)
        }
    } else {
        (n_layers, feature_dim, n_timesteps)
    };

    let bytes = std::fs::read(path)
        .with_context(|| format!("failed to read features: {}", path))?;
    let data: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let expected = n_l * dim * n_t;
    if data.len() != expected {
        anyhow::bail!(
            "Feature file {} has {} floats, expected {} ({} layers × {} dim × {} timesteps)",
            path, data.len(), expected, n_l, dim, n_t
        );
    }

    // Reshape to [1, n_layers * feature_dim, n_timesteps]
    // Features are stored as [n_layers, feature_dim, n_timesteps] and we concatenate layers
    Ok(Tensor::from_vec(data, vec![1, n_l * dim, n_t]))
}

fn parse_cmap(s: &str) -> plotting::ColorMap {
    match s {
        "hot" => plotting::ColorMap::Hot,
        "coolwarm" | "cool_warm" => plotting::ColorMap::CoolWarm,
        "viridis" => plotting::ColorMap::Viridis,
        "seismic" => plotting::ColorMap::Seismic,
        "bluered" | "blue_red" => plotting::ColorMap::BlueRed,
        "gray" | "grey" => plotting::ColorMap::GrayScale,
        _ => {
            eprintln!("Unknown cmap '{}', using 'hot'", s);
            plotting::ColorMap::Hot
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let t0 = Instant::now();

    // ── Parse layer positions ─────────────────────────────────────────
    let layer_positions: Vec<f64> = args
        .layer_positions
        .split(',')
        .map(|s| s.trim().parse::<f64>().unwrap_or(0.5))
        .collect();

    // ── Load config ───────────────────────────────────────────────────
    let config_str = std::fs::read_to_string(&args.config)
        .with_context(|| format!("failed to read config: {}", args.config))?;
    let mut config: TribeV2Config = serde_yaml::from_str(&config_str)
        .with_context(|| "failed to parse config.yaml")?;

    // Set average_subjects for inference when no specific subject is requested
    if args.subject.is_none() {
        if let Some(ref mut sl) = config.brain_model_config.subject_layers {
            sl.average_subjects = true;
        }
    }

    if args.verbose {
        eprintln!("Config loaded: hidden={}, depth={}, heads={}",
            config.brain_model_config.hidden,
            config.brain_model_config.encoder.as_ref().map_or(0, |e| e.depth),
            config.brain_model_config.encoder.as_ref().map_or(0, |e| e.heads),
        );
    }

    // ── Build model ───────────────────────────────────────────────────
    let feature_dims = if let Some(ref ba_path) = args.build_args {
        let ba = tribev2::ModelBuildArgs::from_json(ba_path)?;
        ba.to_modality_dims()
    } else {
        ModalityDims::pretrained()
    };
    let n_outputs = if let Some(ref ba_path) = args.build_args {
        tribev2::ModelBuildArgs::from_json(ba_path)
            .map(|ba| ba.n_outputs)
            .unwrap_or(20484)
    } else {
        20484
    };
    let n_output_timesteps = config.data.duration_trs;

    let mut model = TribeV2::new(
        feature_dims.clone(),
        n_outputs,
        n_output_timesteps,
        &config.brain_model_config,
    );

    eprintln!("Model built ({:.0} ms)", t0.elapsed().as_secs_f64() * 1000.0);

    // ── Load weights ──────────────────────────────────────────────────
    let t1 = Instant::now();
    let mut wm = WeightMap::from_safetensors(&args.weights)
        .with_context(|| format!("failed to load weights: {}", args.weights))?;

    if args.verbose {
        let keys = wm.remaining_keys();
        eprintln!("Weight keys ({}):", keys.len());
        for k in &keys {
            eprintln!("  {}", k);
        }
    }

    load_weights(&mut wm, &mut model)
        .with_context(|| "failed to load weights into model")?;

    let remaining = wm.remaining_keys();
    if !remaining.is_empty() && args.verbose {
        eprintln!("Unused weight keys: {:?}", remaining);
    }

    eprintln!("Weights loaded ({:.0} ms)", t1.elapsed().as_secs_f64() * 1000.0);

    // ── Extract / load features ───────────────────────────────────────
    let n_timesteps = args.n_timesteps;
    let mut features = BTreeMap::new();

    // Text features
    if let (Some(ref llama_path), Some(ref prompt)) = (&args.llama_model, &args.prompt) {
        eprintln!("Extracting LLaMA text features (with intermediate layers)...");
        let t2 = Instant::now();
        let llama_config = LlamaFeatureConfig {
            model_path: llama_path.clone(),
            layer_positions: layer_positions.clone(),
            n_layers: 28, // LLaMA-3.2-3B
            n_ctx: 2048,
            frequency: 2.0,
        };
        let text_feats = features::extract_llama_features(&llama_config, prompt, args.verbose)?;

        // Resample to n_timesteps if needed
        let text_feats = features::resample_features(&text_feats, n_timesteps);

        if args.verbose {
            eprintln!("Text features: [{}×{}, {}] → {} timesteps",
                text_feats.n_layers, text_feats.feature_dim,
                text_feats.n_timesteps, n_timesteps);
        }

        // Reshape to [1, n_layers*feature_dim, n_timesteps] for model input
        let total_dim = text_feats.n_layers * text_feats.feature_dim;
        let flat = Tensor::from_vec(
            text_feats.data.data.clone(),
            vec![1, total_dim, text_feats.n_timesteps],
        );
        features.insert("text".to_string(), flat);
        eprintln!("Text features extracted ({:.0} ms)", t2.elapsed().as_secs_f64() * 1000.0);

    } else if let Some(ref path) = args.text_features {
        eprintln!("Loading pre-extracted text features...");
        // Pretrained model: 2 layer groups × 3072 dim = 6144
        let n_l = args.n_layers.unwrap_or(2);
        let dim = args.feature_dim.unwrap_or(3072);
        let t = load_preextracted_features(path, n_l, dim, n_timesteps)?;
        features.insert("text".to_string(), t);
    }

    // Audio features
    if let Some(ref path) = args.audio_features {
        eprintln!("Loading pre-extracted audio features...");
        // Pretrained model: 2 layer groups × 1024 dim = 2048
        let n_l = args.n_layers.unwrap_or(2);
        let dim = args.feature_dim.unwrap_or(1024);
        let t = load_preextracted_features(path, n_l, dim, n_timesteps)?;
        features.insert("audio".to_string(), t);
    }

    // Video features
    if let Some(ref path) = args.video_features {
        eprintln!("Loading pre-extracted video features...");
        // Pretrained model: 2 layer groups × 1408 dim = 2816
        let n_l = args.n_layers.unwrap_or(2);
        let dim = args.feature_dim.unwrap_or(1408);
        let t = load_preextracted_features(path, n_l, dim, n_timesteps)?;
        features.insert("video".to_string(), t);
    }

    // Fill missing modalities with zeros using build_args dims (or pretrained defaults)
    for md in &feature_dims {
        if !features.contains_key(&md.name) {
            if let Some((n_l, dim)) = md.dims {
                features.insert(
                    md.name.clone(),
                    Tensor::zeros(&[1, n_l * dim, n_timesteps]),
                );
            }
        }
    }

    let n_modalities = features.iter()
        .filter(|(_, t)| t.data.iter().any(|&v| v != 0.0))
        .count();
    eprintln!("Active modalities: {} / {}", n_modalities, features.len());

    // ── Run inference ─────────────────────────────────────────────────
    eprintln!("Running inference...");
    let t3 = Instant::now();

    let predictions: Vec<Vec<f32>>;
    let n_pred_timesteps: usize;

    let subject_ids: Option<Vec<usize>> = args.subject.map(|s| vec![s]);

    if args.segment {
        // Segment-based inference
        let seg_config = SegmentConfig {
            duration_trs: args.segment_duration,
            overlap_trs: args.segment_overlap,
            tr: 0.5,
            remove_empty_segments: args.remove_empty,
            feature_frequency: 2.0,
            stride_drop_incomplete: false,
        };

        // TODO: per-subject segmented inference (currently averages subjects)
        let _ = &subject_ids;
        let result = segments::predict_segmented(&model, &features, &seg_config);
        eprintln!(
            "Segments: {} total TRs, {} kept ({:.1}%)",
            result.total_segments,
            result.kept_segments,
            100.0 * result.kept_segments as f64 / result.total_segments.max(1) as f64
        );
        predictions = result.predictions;
        n_pred_timesteps = predictions.len();
    } else {
        // Single forward pass
        let output = model.forward(&features, subject_ids.as_deref(), true);
        // output: [1, n_outputs, n_output_timesteps]
        let n_out = output.shape[1];
        let n_t = output.shape[2];
        n_pred_timesteps = n_t;
        predictions = (0..n_t)
            .map(|ti| {
                (0..n_out)
                    .map(|di| output.data[di * n_t + ti])
                    .collect()
            })
            .collect();
    }

    let infer_ms = t3.elapsed().as_secs_f64() * 1000.0;
    eprintln!("Inference complete ({:.0} ms)", infer_ms);
    eprintln!(
        "Output: {} timesteps × {} vertices",
        n_pred_timesteps,
        predictions.first().map_or(0, |v| v.len())
    );

    // ── Save predictions ──────────────────────────────────────────────
    if let Some(ref out_path) = args.output {
        let n_vertices = predictions.first().map_or(0, |v| v.len());
        let mut flat: Vec<f32> = Vec::with_capacity(n_pred_timesteps * n_vertices);
        for row in &predictions {
            flat.extend_from_slice(row);
        }
        let bytes: Vec<u8> = flat.iter().flat_map(|f| f.to_le_bytes()).collect();
        std::fs::write(out_path, &bytes)?;
        eprintln!(
            "Predictions saved to {} ({} timesteps × {} vertices)",
            out_path, n_pred_timesteps, n_vertices
        );
    }

    // ── Generate brain plots ──────────────────────────────────────────
    if let Some(ref plot_dir) = args.plot_dir {
        eprintln!("Generating brain surface plots...");
        let t4 = Instant::now();

        let view = plotting::View::from_str(&args.view)
            .unwrap_or(plotting::View::Left);

        // Try to load real fsaverage mesh; fall back to synthetic
        let brain = match tribev2::fsaverage::load_fsaverage(
            &args.mesh, "half", "sulcal", args.subjects_dir.as_deref(),
        ) {
            Ok(b) => {
                eprintln!("Loaded {} mesh ({} + {} vertices)",
                    args.mesh, b.left.mesh.n_vertices, b.right.mesh.n_vertices);
                b
            }
            Err(e) => {
                eprintln!("Could not load {} mesh ({}), using synthetic sphere", args.mesh, e);
                plotting::generate_test_mesh(5000)
            }
        };

        let plot_config = plotting::PlotConfig {
            width: 800,
            height: 600,
            cmap: parse_cmap(&args.cmap),
            view,
            colorbar: args.colorbar,
            symmetric_cbar: false,
            ..Default::default()
        };

        let paths = plotting::render_timesteps(
            &predictions,
            &brain,
            &plot_config,
            plot_dir,
        )?;

        eprintln!(
            "Generated {} SVG plots in {} ({:.0} ms)",
            paths.len(),
            plot_dir,
            t4.elapsed().as_secs_f64() * 1000.0
        );

        // Also generate multi-view for the first timestep
        if !predictions.is_empty() {
            let views = vec![
                plotting::View::Left,
                plotting::View::Right,
                plotting::View::Dorsal,
            ];
            let multi_paths = plotting::render_multi_view(
                &predictions[0],
                &brain,
                &views,
                &plot_config,
                plot_dir,
                "overview_t0",
            )?;
            eprintln!("Multi-view overview: {:?}", multi_paths);
        }
    }

    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!("Total: {:.0} ms", total_ms);

    Ok(())
}
