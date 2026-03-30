//! Feature extraction for TRIBE v2 — text, audio, and video.
//!
//! The pretrained TRIBE v2 model uses:
//! - **Text**: LLaMA-3.2-3B → 3 layer groups × 3072 dims = 9216 concat
//! - **Audio**: Wav2Vec-BERT 2.0 → 3 layer groups × 1024 dims = 3072 concat
//! - **Video**: V-JEPA2 ViT-G → 3 layer groups × 1408 dims = 4224 concat
//!
//! Layer selection:
//!   layers = [0.5, 0.75, 1.0] → for a model with N layers, pick layers at
//!   positions floor(0.5*N), floor(0.75*N), N-1 (0-indexed).
//!   With layer_aggregation="group_mean" and n_layers_to_use=3, each of the
//!   3 groups is a single layer (no averaging needed).
//!
//! Feature frequency: 2 Hz (one feature vector per 0.5s)
//!
//! ## Per-layer hidden state extraction
//!
//! Uses [`TensorCapture`](llama_cpp_4::context::tensor_capture::TensorCapture)
//! from llama-cpp-4 to intercept intermediate layer outputs (`"l_out-{N}"`)
//! during graph evaluation. This gives true per-layer activations matching
//! the Python `output_hidden_states=True` behavior from HuggingFace
//! transformers.

use std::num::NonZeroU32;
use anyhow::{Context, Result};
use crate::tensor::Tensor;
use llama_cpp_4::context::tensor_capture::TensorCapture;

/// Information about extracted features for one modality.
#[derive(Debug, Clone)]
pub struct ExtractedFeatures {
    /// Feature tensor: [n_layers, feature_dim, n_timesteps]
    pub data: Tensor,
    /// Number of layer groups.
    pub n_layers: usize,
    /// Feature dimension per layer.
    pub feature_dim: usize,
    /// Number of timesteps.
    pub n_timesteps: usize,
}

/// Configuration for LLaMA feature extraction.
#[derive(Debug, Clone)]
pub struct LlamaFeatureConfig {
    /// Path to the GGUF model file.
    pub model_path: String,
    /// Layer positions to extract (fractional, 0.0-1.0).
    /// Default: [0.5, 0.75, 1.0]
    pub layer_positions: Vec<f64>,
    /// Total number of transformer layers in the model.
    /// LLaMA-3.2-3B = 28, LLaMA-3.2-1B = 16.
    /// Used to compute which layers to extract.
    pub n_layers: usize,
    /// Context window size.
    pub n_ctx: u32,
    /// Feature extraction frequency in Hz.
    pub frequency: f64,
}

impl Default for LlamaFeatureConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            layer_positions: vec![0.5, 0.75, 1.0],
            n_layers: 28, // LLaMA-3.2-3B default
            n_ctx: 2048,
            frequency: 2.0,
        }
    }
}

/// Compute which layer indices to extract given fractional positions and total layer count.
///
/// Python: `layers = np.linspace(0, 1, n_layers_to_use)` → [0.5, 0.75, 1.0]
/// Maps each fraction f to layer index `floor(f * (n_layers - 1))`.
pub fn compute_layer_indices(layer_positions: &[f64], n_total_layers: usize) -> Vec<usize> {
    layer_positions
        .iter()
        .map(|&f| {
            let idx = (f * (n_total_layers as f64 - 1.0)).floor() as usize;
            idx.min(n_total_layers - 1)
        })
        .collect()
}

// ── Helper: copy captured layer data into output tensor ──────────────────

/// Copy captured per-layer data from a `TensorCapture` into the output
/// tensor layout `[n_layers, hidden_dim, n_timesteps]`.
///
/// For each layer group, reads from `capture.get_layer(layer_idx)` using
/// the `CapturedTensor::token_embedding()` API. Falls back to `fallback_embs`
/// (final-layer embeddings) for any layer not captured.
fn fill_layer_data(
    data: &mut [f32],
    layer_indices: &[usize],
    capture: &TensorCapture,
    fallback_embs: Option<&[Vec<f32>]>,
    hidden_dim: usize,
    n_timesteps: usize,
) {
    for (li, &layer_idx) in layer_indices.iter().enumerate() {
        if let Some(ct) = capture.get_layer(layer_idx) {
            let tokens_to_copy = ct.n_tokens().min(n_timesteps);
            let dims_to_copy = ct.n_embd().min(hidden_dim);
            for ti in 0..tokens_to_copy {
                let emb = ct.token_embedding(ti).unwrap();
                for di in 0..dims_to_copy {
                    data[li * hidden_dim * n_timesteps + di * n_timesteps + ti] = emb[di];
                }
            }
        } else if let Some(embs) = fallback_embs {
            for ti in 0..n_timesteps {
                for di in 0..hidden_dim {
                    data[li * hidden_dim * n_timesteps + di * n_timesteps + ti] = embs[ti][di];
                }
            }
        }
    }
}

/// Extract text features from a prompt using LLaMA with **true per-layer**
/// hidden state extraction.
///
/// Returns features as [n_layers, hidden_dim, n_timesteps].
///
/// For LLaMA-3.2-3B: 28 layers, hidden_size=3072.
/// With layer_positions=[0.5, 0.75, 1.0] → layers [13, 20, 27] (0-indexed).
///
/// Uses [`TensorCapture`] to intercept intermediate layer outputs
/// (`"l_out-{N}"` tensors) during graph evaluation, providing true
/// per-layer activations matching the Python HuggingFace
/// `output_hidden_states=True` behavior.
pub fn extract_llama_features(
    config: &LlamaFeatureConfig,
    prompt: &str,
    verbose: bool,
) -> Result<ExtractedFeatures> {
    use llama_cpp_4::context::params::LlamaContextParams;
    use llama_cpp_4::llama_backend::LlamaBackend;
    use llama_cpp_4::llama_batch::LlamaBatch;
    use llama_cpp_4::model::params::LlamaModelParams;
    use llama_cpp_4::model::{AddBos, LlamaModel};

    let backend = LlamaBackend::init()?;

    let model_params = {
        #[cfg(any(feature = "llama-cuda", feature = "llama-vulkan", feature = "llama-metal"))]
        { LlamaModelParams::default().with_n_gpu_layers(1000) }
        #[cfg(not(any(feature = "llama-cuda", feature = "llama-vulkan", feature = "llama-metal")))]
        { LlamaModelParams::default() }
    };

    let model = LlamaModel::load_from_file(&backend, &config.model_path, &model_params)
        .with_context(|| format!("unable to load LLaMA model: {}", config.model_path))?;

    let n_total_layers = model.n_layer() as usize;
    let hidden_dim = model.n_embd() as usize;
    let layer_indices = compute_layer_indices(&config.layer_positions, n_total_layers);
    let n_layer_groups = layer_indices.len();

    if verbose {
        eprintln!("LLaMA: {} layers, hidden_dim={}", n_total_layers, hidden_dim);
        eprintln!("Extracting layers: {:?} (from positions {:?})",
            layer_indices, config.layer_positions);
    }

    // Tokenize
    let tokens = model
        .str_to_token(prompt, AddBos::Always)
        .with_context(|| "failed to tokenize prompt")?;
    let n_tokens = tokens.len();

    if verbose {
        eprintln!("Tokens: {}", n_tokens);
    }

    // Set up per-layer capture via TensorCapture
    let mut capture = TensorCapture::for_layers(&layer_indices);

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(NonZeroU32::new(config.n_ctx.max(n_tokens as u32 + 16)).unwrap()))
        .with_embeddings(true)
        .with_tensor_capture(&mut capture);

    let mut ctx = model
        .new_context(&backend, ctx_params)
        .with_context(|| "unable to create LLaMA context")?;

    // Process all tokens in one batch
    let mut batch = LlamaBatch::new(n_tokens + 16, 1);
    for (i, token) in tokens.iter().enumerate() {
        batch.add(*token, i as i32, &[0], true)?;
    }

    ctx.decode(&mut batch)
        .with_context(|| "llama_decode() failed")?;

    // Check what we captured
    let captured = capture.captured_layers();
    if verbose {
        eprintln!("Captured {}/{} layers: {:?}", captured.len(), n_layer_groups, captured);
    }

    let n_timesteps = n_tokens;
    let total = n_layer_groups * hidden_dim * n_timesteps;
    let mut data = vec![0.0f32; total];

    // Collect final-layer embeddings as fallback only if some layers are missing
    let fallback_embs: Option<Vec<Vec<f32>>> = if captured.len() < n_layer_groups {
        if verbose {
            eprintln!("WARNING: using final-layer fallback for {}/{} missing layers",
                n_layer_groups - captured.len(), n_layer_groups);
        }
        let mut embs = Vec::with_capacity(n_tokens);
        for i in 0..n_tokens {
            let emb = ctx.embeddings_ith(i as i32)
                .with_context(|| format!("failed to get embedding for token {}", i))?;
            embs.push(emb.to_vec());
        }
        Some(embs)
    } else {
        None
    };

    fill_layer_data(
        &mut data,
        &layer_indices,
        &capture,
        fallback_embs.as_deref(),
        hidden_dim,
        n_timesteps,
    );

    Ok(ExtractedFeatures {
        data: Tensor::from_vec(data, vec![n_layer_groups, hidden_dim, n_timesteps]),
        n_layers: n_layer_groups,
        feature_dim: hidden_dim,
        n_timesteps,
    })
}

/// Extract text features using LLaMA with a word-level event list and
/// **true per-layer** hidden state extraction.
///
/// Each word event has a start time and optional duration. Features are
/// extracted per-word (one token group per word), then temporally aligned
/// to produce features at the specified frequency.
///
/// Uses [`TensorCapture`] to intercept intermediate layer outputs,
/// matching the Python HuggingFace `output_hidden_states=True` behavior.
///
/// `words`: list of (text, start_time_seconds)
/// `total_duration`: total duration in seconds
///
/// Returns features as [n_layers, hidden_dim, n_timesteps] where
/// n_timesteps = ceil(total_duration * frequency).
pub fn extract_llama_features_timed(
    config: &LlamaFeatureConfig,
    words: &[(String, f64)],
    total_duration: f64,
    verbose: bool,
) -> Result<ExtractedFeatures> {
    use llama_cpp_4::context::params::LlamaContextParams;
    use llama_cpp_4::llama_backend::LlamaBackend;
    use llama_cpp_4::llama_batch::LlamaBatch;
    use llama_cpp_4::model::params::LlamaModelParams;
    use llama_cpp_4::model::{AddBos, LlamaModel};

    let backend = LlamaBackend::init()?;

    let model_params = {
        #[cfg(any(feature = "llama-cuda", feature = "llama-vulkan", feature = "llama-metal"))]
        { LlamaModelParams::default().with_n_gpu_layers(1000) }
        #[cfg(not(any(feature = "llama-cuda", feature = "llama-vulkan", feature = "llama-metal")))]
        { LlamaModelParams::default() }
    };

    let model = LlamaModel::load_from_file(&backend, &config.model_path, &model_params)
        .with_context(|| format!("unable to load LLaMA model: {}", config.model_path))?;

    let n_total_layers = model.n_layer() as usize;
    let hidden_dim = model.n_embd() as usize;
    let layer_indices = compute_layer_indices(&config.layer_positions, n_total_layers);
    let n_layer_groups = layer_indices.len();

    // Build full text for tokenization with context
    let full_text: String = words.iter().map(|(w, _)| w.as_str()).collect::<Vec<_>>().join(" ");
    let tokens = model
        .str_to_token(&full_text, AddBos::Always)
        .with_context(|| "failed to tokenize")?;
    let n_tokens = tokens.len();

    if verbose {
        eprintln!("LLaMA timed: {} words, {} tokens, {} layers, dim={}",
            words.len(), n_tokens, n_total_layers, hidden_dim);
        eprintln!("Extracting layers: {:?}", layer_indices);
    }

    // Set up per-layer capture via TensorCapture
    let mut capture = TensorCapture::for_layers(&layer_indices);

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(NonZeroU32::new(config.n_ctx.max(n_tokens as u32 + 16)).unwrap()))
        .with_embeddings(true)
        .with_tensor_capture(&mut capture);

    let mut ctx = model
        .new_context(&backend, ctx_params)
        .with_context(|| "unable to create LLaMA context")?;

    let mut batch = LlamaBatch::new(n_tokens + 16, 1);
    for (i, token) in tokens.iter().enumerate() {
        batch.add(*token, i as i32, &[0], true)?;
    }
    ctx.decode(&mut batch)
        .with_context(|| "llama_decode() failed")?;

    // Check what we captured
    let captured = capture.captured_layers();
    let all_captured = captured.len() == n_layer_groups;

    if verbose {
        eprintln!("Captured {}/{} layers: {:?}", captured.len(), n_layer_groups, captured);
    }

    // Collect final-layer embeddings as fallback
    let final_embeddings: Option<Vec<Vec<f32>>> = if !all_captured {
        let mut embs = Vec::with_capacity(n_tokens);
        for i in 0..n_tokens {
            let emb = ctx.embeddings_ith(i as i32)
                .with_context(|| format!("failed to get embedding for token {}", i))?;
            embs.push(emb.to_vec());
        }
        Some(embs)
    } else {
        None
    };

    // Map words to token ranges (approximate: even distribution)
    let n_words = words.len();
    let tokens_per_word = if n_words > 0 {
        (n_tokens - 1).max(1) as f64 / n_words as f64
    } else {
        1.0
    };

    // Create per-layer word embeddings: [layer_group][word] = (avg_embedding, start_time)
    let mut layer_word_embeddings: Vec<Vec<(Vec<f32>, f64)>> = Vec::with_capacity(n_layer_groups);

    for &layer_idx in &layer_indices {
        let ct = capture.get_layer(layer_idx);

        let mut word_embs: Vec<(Vec<f32>, f64)> = Vec::with_capacity(n_words);

        for (wi, (_, start_time)) in words.iter().enumerate() {
            let tok_start = 1 + (wi as f64 * tokens_per_word).floor() as usize;
            let tok_end = (1 + ((wi + 1) as f64 * tokens_per_word).floor() as usize).min(n_tokens);
            let tok_end = tok_end.max(tok_start + 1).min(n_tokens);

            let mut avg = vec![0.0f32; hidden_dim];
            let count = (tok_end - tok_start) as f32;

            for ti in tok_start..tok_end {
                if let Some(ct) = ct {
                    // Per-layer data via TensorCapture
                    let ti_clamped = ti.min(ct.n_tokens() - 1);
                    if let Some(emb) = ct.token_embedding(ti_clamped) {
                        for di in 0..hidden_dim.min(ct.n_embd()) {
                            avg[di] += emb[di];
                        }
                    }
                } else if let Some(ref embs) = final_embeddings {
                    // Fallback to final-layer embeddings
                    for di in 0..hidden_dim {
                        avg[di] += embs[ti][di];
                    }
                }
            }
            if count > 0.0 {
                for v in avg.iter_mut() {
                    *v /= count;
                }
            }
            word_embs.push((avg, *start_time));
        }
        layer_word_embeddings.push(word_embs);
    }

    // Temporally align to output grid at `frequency` Hz
    let n_timesteps = (total_duration * config.frequency).ceil() as usize;
    let dt = 1.0 / config.frequency;

    let total = n_layer_groups * hidden_dim * n_timesteps;
    let mut data = vec![0.0f32; total];

    for ti in 0..n_timesteps {
        let t = ti as f64 * dt;

        for li in 0..n_layer_groups {
            let word_embs = &layer_word_embeddings[li];
            // Find the last word that started at or before time t
            let emb = if let Some(pos) = word_embs.iter().rposition(|(_, st)| *st <= t) {
                &word_embs[pos].0
            } else if !word_embs.is_empty() {
                &word_embs[0].0
            } else {
                continue;
            };

            for di in 0..hidden_dim {
                data[li * hidden_dim * n_timesteps + di * n_timesteps + ti] = emb[di];
            }
        }
    }

    Ok(ExtractedFeatures {
        data: Tensor::from_vec(data, vec![n_layer_groups, hidden_dim, n_timesteps]),
        n_layers: n_layer_groups,
        feature_dim: hidden_dim,
        n_timesteps,
    })
}

/// Create zero features for a missing modality.
///
/// Returns [n_layers, feature_dim, n_timesteps] of zeros.
pub fn zero_features(n_layers: usize, feature_dim: usize, n_timesteps: usize) -> ExtractedFeatures {
    ExtractedFeatures {
        data: Tensor::zeros(&[n_layers, feature_dim, n_timesteps]),
        n_layers,
        feature_dim,
        n_timesteps,
    }
}

/// Resample features from one temporal resolution to another using nearest-neighbor.
///
/// Input: [n_layers, feature_dim, n_timesteps_in]
/// Output: [n_layers, feature_dim, n_timesteps_out]
pub fn resample_features(features: &ExtractedFeatures, n_timesteps_out: usize) -> ExtractedFeatures {
    let n_layers = features.n_layers;
    let feature_dim = features.feature_dim;
    let n_in = features.n_timesteps;

    if n_in == n_timesteps_out {
        return features.clone();
    }

    let mut data = vec![0.0f32; n_layers * feature_dim * n_timesteps_out];
    for li in 0..n_layers {
        for di in 0..feature_dim {
            for to in 0..n_timesteps_out {
                let ti = (to as f64 * n_in as f64 / n_timesteps_out as f64).floor() as usize;
                let ti = ti.min(n_in - 1);
                data[li * feature_dim * n_timesteps_out + di * n_timesteps_out + to] =
                    features.data.data[li * feature_dim * n_in + di * n_in + ti];
            }
        }
    }

    ExtractedFeatures {
        data: Tensor::from_vec(data, vec![n_layers, feature_dim, n_timesteps_out]),
        n_layers,
        feature_dim,
        n_timesteps: n_timesteps_out,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_layer_indices() {
        // LLaMA-3.2-3B: 28 layers
        let indices = compute_layer_indices(&[0.5, 0.75, 1.0], 28);
        assert_eq!(indices, vec![13, 20, 27]);
    }

    #[test]
    fn test_compute_layer_indices_small() {
        let indices = compute_layer_indices(&[0.5, 0.75, 1.0], 4);
        assert_eq!(indices, vec![1, 2, 3]);
    }

    #[test]
    fn test_zero_features() {
        let f = zero_features(3, 1024, 100);
        assert_eq!(f.data.shape, vec![3, 1024, 100]);
        assert!(f.data.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_resample_features_identity() {
        let f = zero_features(2, 4, 10);
        let r = resample_features(&f, 10);
        assert_eq!(r.n_timesteps, 10);
    }

    #[test]
    fn test_resample_features_upsample() {
        let mut f = zero_features(1, 2, 4);
        f.data.data = vec![
            1.0, 2.0, 3.0, 4.0, // layer 0, dim 0, t=[0,1,2,3]
            5.0, 6.0, 7.0, 8.0, // layer 0, dim 1, t=[0,1,2,3]
        ];
        let r = resample_features(&f, 8);
        assert_eq!(r.n_timesteps, 8);
        assert_eq!(r.data.shape, vec![1, 2, 8]);
        // Each original timestep maps to 2 output timesteps (nearest neighbor)
        assert_eq!(r.data.data[0], 1.0); // t_out=0 → t_in=0
        assert_eq!(r.data.data[1], 1.0); // t_out=1 → t_in=0
        assert_eq!(r.data.data[2], 2.0); // t_out=2 → t_in=1
    }
}
