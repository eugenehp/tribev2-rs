//! TRIBE v2 inference CLI.
//!
//! Loads a pretrained TRIBE v2 model (safetensors) and runs inference
//! using LLaMA text features extracted via llama-cpp-4.
//!
//! Usage:
//!   tribev2-infer --config config.yaml --weights model.safetensors \
//!                 --llama-model llama-3.2-3b.gguf --prompt "The quick brown fox"
//!
//! Or text-only with pre-extracted features:
//!   tribev2-infer --config config.yaml --weights model.safetensors \
//!                 --features features.bin --n-timesteps 100

use std::num::NonZeroU32;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;

use tribev2_rs::config::{ModalityDims, TribeV2Config};
use tribev2_rs::model::tribe::TribeV2;
use tribev2_rs::weights::{WeightMap, load_weights};

#[derive(Parser, Debug)]
#[command(about = "TRIBE v2 fMRI brain encoding model inference")]
struct Args {
    /// Path to config.yaml
    #[arg(long)]
    config: String,

    /// Path to model.safetensors
    #[arg(long)]
    weights: String,

    /// Path to LLaMA GGUF model for text feature extraction
    #[arg(long)]
    llama_model: Option<String>,

    /// Text prompt for LLaMA feature extraction
    #[arg(long, short = 'p')]
    prompt: Option<String>,

    /// Number of output timesteps to predict
    #[arg(long, default_value = "100")]
    n_timesteps: usize,

    /// Output file for predictions (binary f32)
    #[arg(long)]
    output: Option<String>,

    /// Print verbose info
    #[arg(long, short = 'v')]
    verbose: bool,
}

/// Extract text embeddings from LLaMA using llama-cpp-4.
///
/// Returns hidden state embeddings from intermediate layers.
/// The pretrained TRIBE v2 uses layers at positions [0.5, 0.75, 1.0] of
/// LLaMA-3.2-3B (28 layers total → layers 14, 21, 27).
/// With layer_aggregation="group_mean", features are 3 groups × 3072 = 9216 dims (concatenated).
fn extract_llama_features(
    model_path: &str,
    prompt: &str,
    verbose: bool,
) -> Result<Vec<Vec<f32>>> {
    use llama_cpp_4::context::params::LlamaContextParams;
    use llama_cpp_4::llama_backend::LlamaBackend;
    use llama_cpp_4::llama_batch::LlamaBatch;
    use llama_cpp_4::model::params::LlamaModelParams;
    use llama_cpp_4::model::{AddBos, LlamaModel};

    let backend = LlamaBackend::init()?;

    let model_params = {
        #[cfg(any(feature = "cuda", feature = "vulkan", feature = "metal"))]
        { LlamaModelParams::default().with_n_gpu_layers(1000) }
        #[cfg(not(any(feature = "cuda", feature = "vulkan", feature = "metal")))]
        { LlamaModelParams::default() }
    };

    let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
        .with_context(|| "unable to load LLaMA model")?;

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(NonZeroU32::new(2048).unwrap()))
        .with_embeddings(true);

    let mut ctx = model
        .new_context(&backend, ctx_params)
        .with_context(|| "unable to create LLaMA context")?;

    let tokens = model
        .str_to_token(prompt, AddBos::Always)
        .with_context(|| "failed to tokenize prompt")?;

    if verbose {
        eprintln!("Prompt tokens: {}", tokens.len());
    }

    let mut batch = LlamaBatch::new(2048, 1);
    let last_idx = (tokens.len() - 1) as i32;
    for (i, token) in tokens.iter().enumerate() {
        batch.add(*token, i as i32, &[0], i as i32 == last_idx)?;
    }

    ctx.decode(&mut batch)
        .with_context(|| "llama_decode() failed")?;

    // Get embeddings for the last token
    let n_embd = model.n_embd() as usize;
    if verbose {
        eprintln!("LLaMA embedding dim: {}", n_embd);
    }

    // For TRIBE v2 pretrained: LLaMA-3.2-3B has hidden_size=3072, 28 layers
    // layers [0.5, 0.75, 1.0] with group_mean → 3 groups
    // Each group has dim 3072, concatenated = 9216
    // Since llama-cpp gives us the final embedding, we'll use it 3× as a proxy
    // (for exact parity, you'd need to extract intermediate layer activations)

    let embedding = ctx.embeddings_ith(last_idx)
        .with_context(|| "failed to get embeddings")?;

    if verbose {
        eprintln!("Got embedding of size {}", embedding.len());
    }

    // Simulate 3-layer group concatenation: repeat the embedding 3 times
    // This matches the expected input dimension: 3 × 3072 = 9216
    let mut features = Vec::new();
    let concat_emb: Vec<f32> = [embedding, embedding, embedding].concat();

    // Create one feature vector per timestep (at 2Hz, 100 timesteps = 50 seconds)
    // For a single prompt, we'll repeat the embedding across all timesteps
    for _ in 0..100 {
        features.push(concat_emb.clone());
    }

    Ok(features)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let t0 = Instant::now();

    // ── Load config ───────────────────────────────────────────────────
    let config_str = std::fs::read_to_string(&args.config)
        .with_context(|| format!("failed to read config: {}", args.config))?;
    let config: TribeV2Config = serde_yaml::from_str(&config_str)
        .with_context(|| "failed to parse config.yaml")?;

    if args.verbose {
        eprintln!("Config loaded: hidden={}, depth={}, heads={}",
            config.brain_model_config.hidden,
            config.brain_model_config.encoder.as_ref().map_or(0, |e| e.depth),
            config.brain_model_config.encoder.as_ref().map_or(0, |e| e.heads),
        );
    }

    // ── Build model ───────────────────────────────────────────────────
    let feature_dims = ModalityDims::pretrained();
    let n_outputs = 20484; // fsaverage5: 10242 * 2
    let n_output_timesteps = config.data.duration_trs;

    let mut model = TribeV2::new(
        feature_dims,
        n_outputs,
        n_output_timesteps,
        &config.brain_model_config,
    );

    // Override average_subjects for inference
    if config.average_subjects {
        model.predictor.config.average_subjects = true;
    }

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

    // ── Extract features ──────────────────────────────────────────────
    let features = if let (Some(ref llama_path), Some(ref prompt)) = (&args.llama_model, &args.prompt) {
        eprintln!("Extracting LLaMA features...");
        let t2 = Instant::now();
        let feats = extract_llama_features(llama_path, prompt, args.verbose)?;
        eprintln!("Features extracted ({:.0} ms)", t2.elapsed().as_secs_f64() * 1000.0);
        feats
    } else {
        eprintln!("No LLaMA model/prompt specified; using zero features");
        // Zero features: 100 timesteps × 9216 dims
        vec![vec![0.0f32; 9216]; args.n_timesteps]
    };

    // ── Run inference ─────────────────────────────────────────────────
    eprintln!("Running inference...");
    let t3 = Instant::now();
    let predictions = model.predict_from_text_features(&features, args.n_timesteps)?;
    let infer_ms = t3.elapsed().as_secs_f64() * 1000.0;
    eprintln!("Inference complete ({:.0} ms)", infer_ms);
    eprintln!("Output shape: [{}, {}]", predictions.len(), predictions.first().map_or(0, |v| v.len()));

    // ── Save output ───────────────────────────────────────────────────
    if let Some(ref out_path) = args.output {
        let n_timesteps = predictions.len();
        let n_vertices = predictions.first().map_or(0, |v| v.len());
        let mut flat: Vec<f32> = Vec::with_capacity(n_timesteps * n_vertices);
        for row in &predictions {
            flat.extend_from_slice(row);
        }
        let bytes: Vec<u8> = flat.iter().flat_map(|f| f.to_le_bytes()).collect();
        std::fs::write(out_path, &bytes)?;
        eprintln!("Predictions saved to {} ({} timesteps × {} vertices)", out_path, n_timesteps, n_vertices);
    }

    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!("Total: {:.0} ms", total_ms);

    Ok(())
}
