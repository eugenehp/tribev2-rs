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
//!
//! Feature frequency: 2 Hz (one feature vector per 0.5s)
//!
//! ## Per-layer hidden state extraction
//!
//! Uses [`Llama32Flow`] prefill with per-layer hidden taps to emit full-sequence
//! hidden states at selected transformer layers (post-block, pre-final-norm),
//! matching HuggingFace `output_hidden_states=True` behavior.

use anyhow::{Context, Result};
use std::path::Path;

use crate::tensor::Tensor;

#[cfg(feature = "rlx-text")]
use rlx::Session;
#[cfg(feature = "rlx-text")]
use rlx_models_core::flow_bridge::compile_options_for_profile;
#[cfg(feature = "rlx-text")]
use rlx_models_core::gguf_support::{assert_gguf_family, GgufModelFamily};
#[cfg(feature = "rlx-text")]
use rlx_models_core::weight_loader::GgufLoader;
#[cfg(feature = "rlx-text")]
use rlx_flow::CompileProfile;
#[cfg(feature = "rlx-text")]
use rlx_flow::blocks::CustomStage;
#[cfg(feature = "rlx-text")]
use rlx_flow::{FlowStage, SideOutputs};
#[cfg(feature = "rlx-text")]
use rlx_llama32::{encode_prompt_auto, llama32_cfg_from_gguf, Llama32Flow, LlamaLayerCtx};
#[cfg(feature = "rlx-text")]
use std::collections::HashSet;

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

/// Configuration for LLaMA feature extraction via RLX.
#[derive(Debug, Clone)]
pub struct LlamaFeatureConfig {
    /// Path to the GGUF model file.
    pub model_path: String,
    /// Layer positions to extract (fractional, 0.0-1.0).
    /// Default: [0.5, 0.75, 1.0]
    pub layer_positions: Vec<f64>,
    /// Total number of transformer layers in the model.
    /// When unset, read from the GGUF metadata after load.
    pub n_layers: Option<usize>,
    /// RLX device string: `cpu`, `metal`, `cuda`, `gpu`, `vulkan`, …
    pub device: String,
    /// Feature extraction frequency in Hz (timed extraction only).
    pub frequency: f64,
}

impl Default for LlamaFeatureConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            layer_positions: vec![0.5, 0.75, 1.0],
            n_layers: None,
            device: {
                #[cfg(feature = "rlx-encoder")]
                {
                    crate::rlx_device::default_rlx_device_label().to_string()
                }
                #[cfg(not(feature = "rlx-encoder"))]
                {
                    "cpu".to_string()
                }
            },
            frequency: 2.0,
        }
    }
}

#[cfg(feature = "rlx-text")]
fn parse_rlx_device(s: &str) -> Result<rlx::Device> {
    crate::rlx_device::parse_rlx_device(s)
}

/// Compute which layer indices to extract given fractional positions and total layer count.
pub fn compute_layer_indices(layer_positions: &[f64], n_total_layers: usize) -> Vec<usize> {
    layer_positions
        .iter()
        .map(|&f| {
            let idx = (f * (n_total_layers as f64 - 1.0)).floor() as usize;
            idx.min(n_total_layers - 1)
        })
        .collect()
}

fn fill_layer_outputs(
    data: &mut [f32],
    layer_outputs: &[Vec<f32>],
    hidden_dim: usize,
    n_timesteps: usize,
) {
    for (li, hidden) in layer_outputs.iter().enumerate() {
        let seq = n_timesteps.min(hidden.len() / hidden_dim.max(1));
        for ti in 0..seq {
            for di in 0..hidden_dim {
                let src = ti * hidden_dim + di;
                if src < hidden.len() {
                    data[li * hidden_dim * n_timesteps + di * n_timesteps + ti] = hidden[src];
                }
            }
        }
    }
}

#[cfg(feature = "rlx-text")]
fn run_llama_layer_export(
    config: &LlamaFeatureConfig,
    tokens: &[u32],
    verbose: bool,
) -> Result<(Vec<Vec<f32>>, usize, usize)> {
    let weights_path = Path::new(&config.model_path);
    let path_str = weights_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("non-UTF-8 model path"))?;
    let raw = assert_gguf_family(weights_path, GgufModelFamily::Llama32)
        .with_context(|| format!("unable to load GGUF: {}", config.model_path))?;
    let cfg = llama32_cfg_from_gguf(&raw)?;
    let mut loader = GgufLoader::from_file(path_str)
        .with_context(|| format!("unable to open GGUF weights: {}", config.model_path))?;
    let n_total_layers = config.n_layers.unwrap_or(cfg.num_hidden_layers);
    let hidden_dim = cfg.hidden_size;
    let layer_indices = compute_layer_indices(&config.layer_positions, n_total_layers);
    let n_layer_groups = layer_indices.len();
    let seq = tokens.len().max(1);
    let batch = 1;

    if verbose {
        eprintln!(
            "LLaMA (RLX): {} layers, hidden_dim={}",
            n_total_layers, hidden_dim
        );
        eprintln!(
            "Extracting layers: {:?} (from positions {:?})",
            layer_indices, config.layer_positions
        );
        eprintln!("Tokens: {}", tokens.len());
    }

    let export_layers: HashSet<usize> = layer_indices.iter().copied().collect();
    let hidden_sink = SideOutputs::new();
    let sink_inner = hidden_sink.inner();

    let built = Llama32Flow::for_prefill(&cfg, batch, seq)
        .hidden_only()
        .layer(move |ctx| {
            let mut stages = vec![ctx.default_stage()];
            if let LlamaLayerCtx::Prefill { index, .. } = ctx {
                if export_layers.contains(&index) {
                    let tap = sink_inner.clone();
                    stages.push(FlowStage::Custom(CustomStage::named(
                        format!("tribev2_hidden_tap_L{index}"),
                        move |_emit, input| {
                            let Some(v) = input else {
                                anyhow::bail!("hidden tap requires layer output");
                            };
                            tap.lock().expect("hidden tap").push(v.hir_id());
                            Ok(Some(v))
                        },
                    )));
                }
            }
            if stages.len() == 1 {
                stages.into_iter().next().unwrap()
            } else {
                FlowStage::Sequence(stages)
            }
        })
        .build(&mut loader)
        .context("build LLaMA layer-hidden export flow")?;

    let taps = hidden_sink.drain();
    let (mut hir, params) = built.into_parts()?;
    hir.outputs = taps;

    let device = parse_rlx_device(&config.device)?;
    let session = Session::new(device);
    let compile_opts =
        compile_options_for_profile(&CompileProfile::llama32_prefill(), device);
    let mut compiled = session
        .compile_hir_with(hir, &compile_opts)
        .context("compile LLaMA prefill graph")?;
    for (name, data) in &params {
        compiled.set_param(name, data);
    }

    let ids_f32: Vec<f32> = tokens.iter().map(|&t| t as f32).collect();
    let outputs = compiled.run(&[("input_ids", ids_f32.as_slice())]);

    anyhow::ensure!(
        outputs.len() == n_layer_groups,
        "expected {} layer hidden outputs, got {}",
        n_layer_groups,
        outputs.len()
    );

    let per_layer: Vec<Vec<f32>> = outputs;
    Ok((per_layer, hidden_dim, tokens.len()))
}

/// Extract text features from a prompt using LLaMA with per-layer hidden states.
///
/// Returns features as [n_layers, hidden_dim, n_timesteps].
#[cfg(feature = "rlx-text")]
pub fn extract_llama_features(
    config: &LlamaFeatureConfig,
    prompt: &str,
    verbose: bool,
) -> Result<ExtractedFeatures> {
    let weights_path = Path::new(&config.model_path);
    let tokens = encode_prompt_auto(weights_path, None, prompt)
        .with_context(|| "failed to tokenize prompt")?;
    let (layer_outputs, hidden_dim, n_timesteps) =
        run_llama_layer_export(config, &tokens, verbose)?;
    let n_layer_groups = layer_outputs.len();
    let total = n_layer_groups * hidden_dim * n_timesteps;
    let mut data = vec![0.0f32; total];
    fill_layer_outputs(&mut data, &layer_outputs, hidden_dim, n_timesteps);

    Ok(ExtractedFeatures {
        data: Tensor::from_vec(data, vec![n_layer_groups, hidden_dim, n_timesteps]),
        n_layers: n_layer_groups,
        feature_dim: hidden_dim,
        n_timesteps,
    })
}

/// Extract text features using LLaMA with a word-level event list and temporal alignment.
#[cfg(feature = "rlx-text")]
pub fn extract_llama_features_timed(
    config: &LlamaFeatureConfig,
    words: &[(String, f64)],
    total_duration: f64,
    verbose: bool,
) -> Result<ExtractedFeatures> {
    let weights_path = Path::new(&config.model_path);
    let full_text: String = words
        .iter()
        .map(|(w, _)| w.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let tokens = encode_prompt_auto(weights_path, None, &full_text)
        .with_context(|| "failed to tokenize")?;
    let (layer_outputs, hidden_dim, n_tokens) =
        run_llama_layer_export(config, &tokens, verbose)?;
    let n_layer_groups = layer_outputs.len();

    let n_words = words.len();
    let tokens_per_word = if n_words > 0 {
        (n_tokens - 1).max(1) as f64 / n_words as f64
    } else {
        1.0
    };

    let mut layer_word_embeddings: Vec<Vec<(Vec<f32>, f64)>> = Vec::with_capacity(n_layer_groups);

    for hidden in &layer_outputs {
        let mut word_embs: Vec<(Vec<f32>, f64)> = Vec::with_capacity(n_words);
        for (wi, (_, start_time)) in words.iter().enumerate() {
            let tok_start = 1 + (wi as f64 * tokens_per_word).floor() as usize;
            let tok_end = (1 + ((wi + 1) as f64 * tokens_per_word).floor() as usize).min(n_tokens);
            let tok_end = tok_end.max(tok_start + 1).min(n_tokens);

            let mut avg = vec![0.0f32; hidden_dim];
            let count = (tok_end - tok_start) as f32;
            for ti in tok_start..tok_end {
                for di in 0..hidden_dim {
                    avg[di] += hidden[ti * hidden_dim + di];
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

    let n_timesteps = (total_duration * config.frequency).ceil() as usize;
    let dt = 1.0 / config.frequency;
    let total = n_layer_groups * hidden_dim * n_timesteps;
    let mut data = vec![0.0f32; total];

    for ti in 0..n_timesteps {
        let t = ti as f64 * dt;
        for li in 0..n_layer_groups {
            let word_embs = &layer_word_embeddings[li];
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

#[cfg(not(feature = "rlx-text"))]
pub fn extract_llama_features(
    _config: &LlamaFeatureConfig,
    _prompt: &str,
    _verbose: bool,
) -> Result<ExtractedFeatures> {
    anyhow::bail!("LLaMA feature extraction requires the `rlx-text` Cargo feature")
}

#[cfg(not(feature = "rlx-text"))]
pub fn extract_llama_features_timed(
    _config: &LlamaFeatureConfig,
    _words: &[(String, f64)],
    _total_duration: f64,
    _verbose: bool,
) -> Result<ExtractedFeatures> {
    anyhow::bail!("LLaMA feature extraction requires the `rlx-text` Cargo feature")
}

/// Create zero features for a missing modality.
pub fn zero_features(n_layers: usize, feature_dim: usize, n_timesteps: usize) -> ExtractedFeatures {
    ExtractedFeatures {
        data: Tensor::zeros(&[n_layers, feature_dim, n_timesteps]),
        n_layers,
        feature_dim,
        n_timesteps,
    }
}

/// Resample features from one temporal resolution to another using nearest-neighbor.
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
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        ];
        let r = resample_features(&f, 8);
        assert_eq!(r.n_timesteps, 8);
        assert_eq!(r.data.data[0], 1.0);
        assert_eq!(r.data.data[2], 2.0);
    }
}
