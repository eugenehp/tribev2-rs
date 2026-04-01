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
use tribev2::metrics;
use tribev2::model::tribe::TribeV2;
use tribev2::model_burn::tribe::TribeV2Burn;
use tribev2::model_burn::weights::{BurnWeightStore, load_burn_weights};
use tribev2::nifti::{self, NiftiConfig};
use tribev2::plotting;
use tribev2::roi;
use tribev2::segments::{self, SegmentConfig};
use tribev2::subcortical;
use tribev2::tensor::Tensor;
use tribev2::video_output::{self, VideoConfig};
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

    /// Output NIfTI file (.nii or .nii.gz) — projects surface predictions to volume
    #[arg(long)]
    nifti: Option<String>,

    /// NIfTI volume dimensions (isotropic). Default: 96
    #[arg(long, default_value = "96")]
    nifti_dim: usize,

    /// NIfTI voxel size in mm. Default: 2.0
    #[arg(long, default_value = "2.0")]
    nifti_voxel_size: f32,

    /// Print top-k ROI summary (HCP-MMP1 parcellation)
    #[arg(long)]
    roi_summary: Option<usize>,

    /// Save per-ROI averages as JSON
    #[arg(long)]
    roi_output: Option<String>,

    /// Path to HCP annotation directory (for exact ROI labels)
    #[arg(long)]
    hcp_annot_dir: Option<String>,

    /// Save segment metadata as JSON
    #[arg(long)]
    segments_output: Option<String>,

    /// Ground-truth fMRI data (binary f32) for evaluation
    #[arg(long)]
    ground_truth: Option<String>,

    /// Top-k for retrieval accuracy metric
    #[arg(long, default_value = "1")]
    topk: usize,

    /// Save per-vertex correlation map (binary f32)
    #[arg(long)]
    correlation_map: Option<String>,

    /// Show subcortical structure activations (requires subcortical model)
    #[arg(long)]
    subcortical: bool,

    /// Output MP4 video of brain activity over time
    #[arg(long)]
    mp4: Option<String>,

    /// Video FPS (default: 2)
    #[arg(long, default_value = "2")]
    video_fps: u32,

    /// Compute per-modality contribution maps via ablation
    #[arg(long)]
    modality_maps: Option<String>,

    /// Resample output to a different fsaverage resolution
    #[arg(long)]
    output_mesh: Option<String>,

    /// Backend: "cpu" (pure-Rust), "burn-cpu" (Burn NdArray), "burn-gpu" (Burn wgpu Metal/Vulkan)
    #[arg(long, default_value = "cpu")]
    backend: String,

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

/// Run forward pass using the Burn backend.
///
/// Builds a `TribeV2Burn<B>`, loads weights, converts features from the pure-Rust
/// `Tensor` format to burn tensors, runs the forward pass, and converts back.
fn run_burn_forward(
    backend_name: &str,
    weights_path: &str,
    feature_dims: &[ModalityDims],
    n_outputs: usize,
    n_output_timesteps: usize,
    brain_config: &tribev2::config::BrainModelConfig,
    features: &BTreeMap<String, Tensor>,
) -> Result<(Vec<Vec<f32>>, usize)> {
    match backend_name {
        #[cfg(feature = "wgpu")]
        "burn-gpu" => run_burn_forward_impl::<burn::backend::Wgpu>(
            burn::backend::wgpu::WgpuDevice::DefaultDevice,
            weights_path, feature_dims, n_outputs, n_output_timesteps,
            brain_config, features,
        ),
        #[cfg(not(feature = "wgpu"))]
        "burn-gpu" => anyhow::bail!(
            "burn-gpu backend requires the 'wgpu' feature.\n\
             Rebuild with: cargo run --release --features wgpu-metal -- ..."
        ),
        "burn-cpu" | _ if backend_name.starts_with("burn") => {
            #[cfg(feature = "ndarray")]
            {
                run_burn_forward_impl::<burn::backend::NdArray>(
                    Default::default(),
                    weights_path, feature_dims, n_outputs, n_output_timesteps,
                    brain_config, features,
                )
            }
            #[cfg(not(feature = "ndarray"))]
            {
                anyhow::bail!(
                    "burn-cpu backend requires the 'ndarray' feature.\n\
                     Rebuild with: cargo run --release --features ndarray -- ..."
                )
            }
        }
        _ => anyhow::bail!("Unknown backend: {}", backend_name),
    }
}

fn run_burn_forward_impl<B: burn::prelude::Backend>(
    device: B::Device,
    weights_path: &str,
    feature_dims: &[ModalityDims],
    n_outputs: usize,
    n_output_timesteps: usize,
    brain_config: &tribev2::config::BrainModelConfig,
    features: &BTreeMap<String, Tensor>,
) -> Result<(Vec<Vec<f32>>, usize)> {
    use burn::prelude::*;

    eprintln!("  Building Burn model...");
    let tb = Instant::now();
    let mut burn_model = TribeV2Burn::<B>::new(
        feature_dims, n_outputs, n_output_timesteps, brain_config, &device,
    );
    eprintln!("  Burn model built ({:.0} ms)", tb.elapsed().as_secs_f64() * 1000.0);

    eprintln!("  Loading Burn weights...");
    let tw = Instant::now();
    let mut ws = BurnWeightStore::from_safetensors(weights_path)
        .with_context(|| "failed to load burn weights")?;
    load_burn_weights(&mut ws, &mut burn_model, &device)
        .with_context(|| "failed to load burn weights into model")?;
    eprintln!("  Burn weights loaded ({:.0} ms)", tw.elapsed().as_secs_f64() * 1000.0);

    // Convert pure-Rust Tensors → Burn tensors
    let mut burn_features: Vec<(&str, burn::tensor::Tensor<B, 3>)> = Vec::new();
    for (name, tensor) in features {
        let shape = &tensor.shape;
        let burn_tensor = burn::tensor::Tensor::<B, 3>::from_data(
            TensorData::new(tensor.data.clone(), [shape[0], shape[1], shape[2]]),
            &device,
        );
        // Leak the name string so we can get a &str with appropriate lifetime
        // (the Vec lives for the duration of forward())
        burn_features.push((string_to_static_str(name.clone()), burn_tensor));
    }

    eprintln!("  Running Burn forward pass...");
    let tf = Instant::now();
    let output = burn_model.forward(burn_features);
    let [b, d, t] = output.dims();
    eprintln!("  Burn forward: [{}, {}, {}] ({:.0} ms)",
        b, d, t, tf.elapsed().as_secs_f64() * 1000.0);

    // Convert back to Vec<Vec<f32>>
    let output_data: Vec<f32> = output.into_data().to_vec().unwrap();
    // output_data layout: [B, D, T] row-major
    let n_out = d;
    let n_t = t;
    let predictions: Vec<Vec<f32>> = (0..n_t)
        .map(|ti| {
            (0..n_out)
                .map(|di| output_data[di * n_t + ti])
                .collect()
        })
        .collect();

    Ok((predictions, n_t))
}

/// Helper: convert String to &'static str (leaked, but fine for CLI lifetime).
fn string_to_static_str(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
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
    // Mirrors Python from_pretrained: n_subjects=0, average_subjects=true
    // The saved checkpoint has averaged weights → 1 subject row (the dropout row)
    if args.subject.is_none() {
        if let Some(ref mut sl) = config.brain_model_config.subject_layers {
            sl.average_subjects = true;
            sl.n_subjects = 0;
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
    let use_burn = args.backend.starts_with("burn");

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

    eprintln!("Weights loaded ({:.0} ms), backend={}", t1.elapsed().as_secs_f64() * 1000.0, args.backend);

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
    eprintln!("Running inference (backend={})...", args.backend);
    let t3 = Instant::now();

    let predictions: Vec<Vec<f32>>;
    let n_pred_timesteps: usize;

    let subject_ids: Option<Vec<usize>> = args.subject.map(|s| vec![s]);

    if args.segment {
        // Segment-based inference (always uses pure-Rust backend)
        let seg_config = SegmentConfig {
            duration_trs: args.segment_duration,
            overlap_trs: args.segment_overlap,
            tr: 0.5,
            remove_empty_segments: args.remove_empty,
            feature_frequency: 2.0,
            stride_drop_incomplete: false,
        };

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
    } else if use_burn {
        // ── Burn backend forward pass ──────────────────────────────────
        let (preds, n_t) = run_burn_forward(
            &args.backend,
            &args.weights,
            &feature_dims,
            n_outputs,
            n_output_timesteps,
            &config.brain_model_config,
            &features,
        )?;
        predictions = preds;
        n_pred_timesteps = n_t;
    } else {
        // ── Pure-Rust forward pass ────────────────────────────────────
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

    // ── Generate NIfTI output ─────────────────────────────────────────
    if let Some(ref nifti_path) = args.nifti {
        eprintln!("Generating NIfTI volume output...");
        let t5 = Instant::now();

        let nifti_config = NiftiConfig {
            dims: (args.nifti_dim, args.nifti_dim, args.nifti_dim),
            voxel_size: args.nifti_voxel_size,
            compress: nifti_path.ends_with(".gz"),
            ..Default::default()
        };

        // Load original pial coordinates in MNI space
        let vertex_coords = nifti::load_pial_coords_mni(
            &args.mesh,
            args.subjects_dir.as_deref(),
        ).unwrap_or_else(|e| {
            eprintln!("Warning: could not load pial coords ({}), using visualization mesh coords", e);
            match tribev2::fsaverage::load_fsaverage(
                &args.mesh, "pial", "sulcal", args.subjects_dir.as_deref(),
            ) {
                Ok(b) => nifti::get_mesh_coords(&b),
                Err(_) => {
                    eprintln!("Error: no mesh available for NIfTI projection");
                    Vec::new()
                }
            }
        });

        if vertex_coords.is_empty() {
            eprintln!("Skipping NIfTI output: no vertex coordinates available");
        } else {
            let path = std::path::Path::new(nifti_path);
            if predictions.len() == 1 {
                // Single timestep → 3D NIfTI
                let vol = nifti::surface_to_volume(&predictions[0], &vertex_coords, &nifti_config);
                nifti::write_nifti(path, &vol, &nifti_config)?;
            } else {
                // Multiple timesteps → 4D NIfTI
                nifti::write_nifti_4d(path, &predictions, &vertex_coords, &nifti_config)?;
            }
            eprintln!(
                "NIfTI saved to {} ({}×{}×{}, {} timesteps, {:.0} ms)",
                nifti_path, args.nifti_dim, args.nifti_dim, args.nifti_dim,
                predictions.len(),
                t5.elapsed().as_secs_f64() * 1000.0
            );
        }
    }

    // ── ROI summary ───────────────────────────────────────────────────
    if args.roi_summary.is_some() || args.roi_output.is_some() {
        let annot_dir = args.hcp_annot_dir.as_ref().map(|s| std::path::Path::new(s.as_str()));

        // Average predictions across timesteps for ROI analysis
        let n_vertices = predictions.first().map_or(0, |v| v.len());
        let mut avg_pred = vec![0.0f32; n_vertices];
        for row in &predictions {
            for (i, &v) in row.iter().enumerate() {
                avg_pred[i] += v;
            }
        }
        if !predictions.is_empty() {
            let scale = 1.0 / predictions.len() as f32;
            for v in &mut avg_pred {
                *v *= scale;
            }
        }

        if let Some(k) = args.roi_summary {
            let topk = roi::get_topk_rois(&avg_pred, k, annot_dir);
            eprintln!("\n{}", roi::topk_to_table(&topk));
        }

        if let Some(ref path) = args.roi_output {
            let summary = roi::summarize_by_roi(&avg_pred, annot_dir);
            let json = roi::roi_summary_to_json(&summary);
            std::fs::write(path, &json)?;
            eprintln!("ROI summary saved to {}", path);
        }
    }

    // ── Segment metadata output ───────────────────────────────────────
    if let Some(ref seg_path) = args.segments_output {
        // Build segment metadata from predictions
        let seg_entries: Vec<serde_json::Value> = predictions.iter().enumerate()
            .map(|(i, _)| {
                serde_json::json!({
                    "timestep_index": i,
                    "start": i as f64 * 0.5,
                    "duration": 0.5,
                })
            })
            .collect();
        let json = serde_json::to_string_pretty(&seg_entries)?;
        std::fs::write(seg_path, &json)?;
        eprintln!("Segment metadata saved to {} ({} segments)", seg_path, seg_entries.len());
    }

    // ── Evaluation metrics ────────────────────────────────────────────
    if let Some(ref gt_path) = args.ground_truth {
        eprintln!("\nEvaluating against ground truth...");
        let n_vertices = predictions.first().map_or(0, |v| v.len());
        let truth = metrics::load_ground_truth(gt_path, n_vertices)?;

        let n_eval = predictions.len().min(truth.len());
        let pred_slice = &predictions[..n_eval];
        let truth_slice = &truth[..n_eval];

        let mean_r = metrics::mean_pearson(pred_slice, truth_slice);
        let median_r = metrics::median_pearson(pred_slice, truth_slice);
        let mse_val = metrics::mse(pred_slice, truth_slice);
        let topk_acc = metrics::topk_accuracy(pred_slice, truth_slice, args.topk);

        let report = metrics::format_metrics_report(
            mean_r, median_r, mse_val,
            Some((args.topk, topk_acc)),
            n_eval, n_vertices,
        );
        eprintln!("\n{}", report);

        // Optionally save correlation map
        if let Some(ref corr_path) = args.correlation_map {
            let corr = metrics::pearson_per_vertex(pred_slice, truth_slice);
            let bytes: Vec<u8> = corr.iter().flat_map(|f| f.to_le_bytes()).collect();
            std::fs::write(corr_path, &bytes)?;
            eprintln!("Correlation map saved to {} ({} vertices)", corr_path, corr.len());
        }
    }

    // ── Subcortical summary ───────────────────────────────────────────
    if args.subcortical {
        eprintln!("\nSubcortical structure analysis:");
        let n_vertices = predictions.first().map_or(0, |v| v.len());
        let mut avg_pred = vec![0.0f32; n_vertices];
        for row in &predictions {
            for (i, &v) in row.iter().enumerate() {
                avg_pred[i] += v;
            }
        }
        if !predictions.is_empty() {
            let scale = 1.0 / predictions.len() as f32;
            for v in &mut avg_pred { *v *= scale; }
        }

        let config = subcortical::SubcorticalConfig::default();
        let summary = subcortical::summarize_subcortical(&avg_pred, &config);
        eprintln!("\n{}", subcortical::format_subcortical_table(&summary));
        eprintln!("\nNote: Subcortical analysis requires a model trained with MaskProjector.");
        eprintln!("The cortical model\'s vertex indices do not directly map to subcortical voxels.");
    }

    // ── MP4 video output ──────────────────────────────────────────────
    if let Some(ref mp4_path) = args.mp4 {
        eprintln!("Generating MP4 video...");

        let view = plotting::View::from_str(&args.view)
            .unwrap_or(plotting::View::Left);

        let brain = match tribev2::fsaverage::load_fsaverage(
            &args.mesh, "half", "sulcal", args.subjects_dir.as_deref(),
        ) {
            Ok(b) => b,
            Err(_) => plotting::generate_test_mesh(5000),
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

        let video_config = VideoConfig {
            fps: args.video_fps,
            ..Default::default()
        };

        video_output::render_mp4(
            &predictions,
            &brain,
            &plot_config,
            &video_config,
            std::path::Path::new(mp4_path),
        )?;
    }

    // ── Per-modality contribution maps ────────────────────────────────
    if let Some(ref maps_dir) = args.modality_maps {
        eprintln!("Computing per-modality contribution maps...");
        std::fs::create_dir_all(maps_dir)?;

        let contributions = model.modality_ablation(
            &features,
            subject_ids.as_deref(),
        );

        for (modality, contrib) in &contributions {
            // Save as binary f32
            let bin_path = format!("{}/{}_contribution.bin", maps_dir, modality);
            let bytes: Vec<u8> = contrib.iter().flat_map(|f| f.to_le_bytes()).collect();
            std::fs::write(&bin_path, &bytes)?;

            // Summary stats
            let max_val = contrib.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mean_val: f32 = contrib.iter().sum::<f32>() / contrib.len() as f32;
            eprintln!("  {}: mean={:.6}, max={:.6} → {}",
                modality, mean_val, max_val, bin_path);
        }

        // Also save as SVG if we can load a brain mesh
        if let Ok(brain) = tribev2::fsaverage::load_fsaverage(
            &args.mesh, "half", "sulcal", args.subjects_dir.as_deref(),
        ) {
            let view = plotting::View::from_str(&args.view)
                .unwrap_or(plotting::View::Left);
            for (modality, contrib) in &contributions {
                let plot_config = plotting::PlotConfig {
                    width: 800,
                    height: 600,
                    cmap: plotting::ColorMap::Hot,
                    view,
                    colorbar: true,
                    title: Some(format!("{} contribution", modality)),
                    ..Default::default()
                };
                let svg = plotting::render_brain_svg(contrib, &brain, &plot_config);
                let svg_path = format!("{}/{}_contribution.svg", maps_dir, modality);
                std::fs::write(&svg_path, &svg)?;
                eprintln!("  SVG: {}", svg_path);
            }
        }
    }

    // ── Resample output ───────────────────────────────────────────────
    if let Some(ref target_mesh) = args.output_mesh {
        eprintln!("Resampling predictions from {} to {}...", args.mesh, target_mesh);
        let target_size = tribev2::fsaverage::fsaverage_size(target_mesh)
            .ok_or_else(|| anyhow::anyhow!("Unknown target mesh: {}", target_mesh))?;

        match tribev2::resample::resample_surface(
            &predictions[0], &args.mesh, target_mesh,
            args.subjects_dir.as_deref(), 5,
        ) {
            Ok(resampled) => {
                let out_path = format!("predictions_{}.bin", target_mesh);
                let bytes: Vec<u8> = resampled.iter().flat_map(|f| f.to_le_bytes()).collect();
                std::fs::write(&out_path, &bytes)?;
                eprintln!("Resampled predictions saved to {} ({} vertices)",
                    out_path, 2 * target_size);
            }
            Err(e) => {
                eprintln!("Resampling failed: {}. Requires FreeSurfer mesh data.", e);
            }
        }
    }

    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!("Total: {:.0} ms", total_ms);

    Ok(())
}
