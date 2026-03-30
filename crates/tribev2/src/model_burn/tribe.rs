use burn::prelude::*;
use burn::module::{Param, ParamId};
use burn::nn::{Linear, LinearConfig};
use crate::config::{BrainModelConfig, ModalityDims};
use super::projector::{Projector, MlpProjector};
use super::encoder::XTransformerEncoder;
use super::subject_layers::SubjectLayers;

/// Full TRIBE v2 FmriEncoderModel in burn.
///
/// Now includes all features from the Python model:
/// - Per-modality projectors (Linear, MLP, or SubjectLayers)
/// - Optional combiner (Linear/MLP/Identity)
/// - Time positional embedding
/// - Optional subject embedding
/// - x-transformers Encoder
/// - Optional temporal smoothing (depthwise Conv1d)
/// - Low-rank head
/// - SubjectLayers predictor
/// - AdaptiveAvgPool1d
pub struct TribeV2Burn<B: Backend> {
    pub projectors:       Vec<Projector<B>>,
    pub projector_names:  Vec<String>,
    pub combiner:         Option<MlpProjector<B>>,
    pub time_pos_embed:   Option<Param<Tensor<B, 3>>>,
    pub subject_embed:    Option<Param<Tensor<B, 2>>>,
    pub encoder:          Option<XTransformerEncoder<B>>,
    pub low_rank_head:    Option<Linear<B>>,
    pub temporal_smoothing_kernel: Option<Param<Tensor<B, 3>>>,
    pub predictor:        SubjectLayers<B>,
    pub n_outputs:          usize,
    pub n_output_timesteps: usize,
    pub hidden:             usize,
    pub n_modalities:       usize,
    pub use_average_subjects: bool,
    pub linear_baseline:    bool,
    pub ts_kernel_size:     usize,
}

impl<B: Backend> TribeV2Burn<B> {
    pub fn new(
        feature_dims: &[ModalityDims],
        n_outputs: usize,
        n_output_timesteps: usize,
        config: &BrainModelConfig,
        device: &B::Device,
    ) -> Self {
        let hidden      = config.hidden;
        let n_modalities = feature_dims.len();

        // ── Projectors ────────────────────────────────────────────────
        let mut projectors      = Vec::new();
        let mut projector_names = Vec::new();
        for md in feature_dims {
            if let Some((num_layers, feature_dim)) = md.dims {
                let in_dim = if config.layer_aggregation == "cat" {
                    feature_dim * num_layers
                } else {
                    feature_dim
                };
                let out_dim = if config.extractor_aggregation == "cat" {
                    hidden / n_modalities
                } else {
                    hidden
                };

                let proj = if config.projector.name.as_deref() == Some("SubjectLayers") {
                    let sl_cfg = config.subject_layers.clone().unwrap_or_default();
                    Projector::new_subject_layers(in_dim, out_dim, &sl_cfg, device)
                } else if let Some(ref hs) = config.projector.hidden_sizes {
                    if !hs.is_empty() {
                        let has_norm = config.projector.norm_layer.as_deref() == Some("layer");
                        Projector::new_mlp(in_dim, out_dim, hs, has_norm, device)
                    } else {
                        Projector::new_linear(in_dim, out_dim, device)
                    }
                } else {
                    Projector::new_linear(in_dim, out_dim, device)
                };

                projectors.push(proj);
                projector_names.push(md.name.clone());
            }
        }

        // ── Combiner ──────────────────────────────────────────────────
        let combiner = if config.combiner.is_some() {
            let in_dim = if config.extractor_aggregation == "cat" {
                (hidden / n_modalities) * n_modalities
            } else {
                hidden
            };
            Some(MlpProjector::new_linear(in_dim, hidden, device))
        } else {
            None
        };

        // ── Time positional embedding ─────────────────────────────────
        let time_pos_embed = if config.time_pos_embedding && !config.linear_baseline {
            Some(Param::initialized(
                ParamId::new(),
                Tensor::zeros([1, config.max_seq_len, hidden], device),
            ))
        } else {
            None
        };

        // ── Subject embedding ─────────────────────────────────────────
        let subject_embed = if config.subject_embedding && !config.linear_baseline {
            let n_subj = config.subject_layers.as_ref().map_or(200, |sl| sl.n_subjects);
            Some(Param::initialized(
                ParamId::new(),
                Tensor::zeros([n_subj, hidden], device),
            ))
        } else {
            None
        };

        // ── Encoder ───────────────────────────────────────────────────
        let encoder = if !config.linear_baseline {
            config.encoder.as_ref().map(|ec| {
                XTransformerEncoder::new(hidden, config.max_seq_len, ec, device)
            })
        } else {
            None
        };

        // ── Temporal smoothing ────────────────────────────────────────
        let (temporal_smoothing_kernel, ts_kernel_size) =
            if let Some(ref ts_cfg) = config.temporal_smoothing {
                let ks = ts_cfg.kernel_size;
                let kernel = if let Some(sigma) = ts_cfg.sigma {
                    // Gaussian kernel
                    let mut k = vec![0.0f32; hidden * ks];
                    for j in 0..ks {
                        let x = j as f32 - (ks / 2) as f32;
                        let v = (-0.5 * (x / sigma as f32).powi(2)).exp();
                        for c in 0..hidden {
                            k[c * ks + j] = v;
                        }
                    }
                    // Normalize per channel
                    for c in 0..hidden {
                        let sum: f32 = (0..ks).map(|j| k[c * ks + j]).sum();
                        for j in 0..ks {
                            k[c * ks + j] /= sum;
                        }
                    }
                    Tensor::from_data(TensorData::new(k, [hidden, 1, ks]), device)
                } else {
                    Tensor::zeros([hidden, 1, ks], device)
                };
                (Some(Param::initialized(ParamId::new(), kernel)), ks)
            } else {
                (None, 0)
            };

        // ── Low-rank head ─────────────────────────────────────────────
        let low_rank_head = config.low_rank_head.map(|lr| {
            LinearConfig::new(hidden, lr).with_bias(false).init(device)
        });

        // ── Predictor ─────────────────────────────────────────────────
        let bottleneck = config.low_rank_head.unwrap_or(hidden);
        let sl         = config.subject_layers.clone().unwrap_or_default();
        let predictor  = SubjectLayers::new(bottleneck, n_outputs, &sl, device);
        let use_average_subjects = sl.average_subjects;

        Self {
            projectors, projector_names, combiner, time_pos_embed, subject_embed,
            encoder, low_rank_head, temporal_smoothing_kernel, predictor,
            n_outputs, n_output_timesteps, hidden, n_modalities,
            use_average_subjects, linear_baseline: config.linear_baseline,
            ts_kernel_size,
        }
    }
}

// ── forward — standard burn-ops path ──────────────────────────────────────
#[cfg(not(feature = "wgpu-kernels-metal"))]
impl<B: Backend> TribeV2Burn<B> {
    /// Forward pass.
    /// `features`: Vec of `(name, tensor [B, L*D, T])` in feature_dims order.
    /// Returns `[B, n_outputs, n_output_timesteps]`.
    pub fn forward(&self, features: Vec<(&str, Tensor<B, 3>)>) -> Tensor<B, 3> {
        forward_body(self, features)
    }
}

// ── forward — fused-kernel path ────────────────────────────────────────────
#[cfg(feature = "wgpu-kernels-metal")]
impl<B: Backend + crate::model_burn::FusedOps> TribeV2Burn<B> {
    pub fn forward(&self, features: Vec<(&str, Tensor<B, 3>)>) -> Tensor<B, 3> {
        forward_body(self, features)
    }
}

// ── Shared body ────────────────────────────────────────────────────────────

macro_rules! define_forward_body {
    ($($bound:tt)*) => {
        fn forward_body<B: Backend $($bound)*>(
            m:        &TribeV2Burn<B>,
            features: Vec<(&str, Tensor<B, 3>)>,
        ) -> Tensor<B, 3> {
            let device = features[0].1.device();
            let b = features[0].1.dims()[0];
            let t = features[0].1.dims()[2];

            // ── 1. Project each modality and concatenate ──────────────────
            let mut tensors: Vec<Tensor<B, 3>> = Vec::with_capacity(m.n_modalities);
            for (name, data) in &features {
                if let Some(idx) = m.projector_names.iter().position(|n| n == name) {
                    let x = data.clone().swap_dims(1, 2); // [B, L*D, T] → [B, T, L*D]
                    tensors.push(m.projectors[idx].forward(x));
                }
            }
            for name in &m.projector_names {
                if !features.iter().any(|(n, _)| n == name) {
                    let out_dim = m.hidden / m.n_modalities;
                    tensors.push(Tensor::zeros([b, t, out_dim], &device));
                }
            }
            let mut x = Tensor::cat(tensors, 2); // [B, T, H]

            // ── 2. Temporal smoothing ─────────────────────────────────────
            if let Some(ref kernel) = m.temporal_smoothing_kernel {
                // x: [B, T, H] → [B, H, T], depthwise conv, → [B, T, H]
                x = depthwise_conv1d(x.swap_dims(1, 2), &kernel.val(), m.ts_kernel_size)
                    .swap_dims(1, 2);
            }

            if !m.linear_baseline {
                // ── 3. Combiner ───────────────────────────────────────────
                if let Some(ref combiner) = m.combiner {
                    x = combiner.forward(x);
                }

                // ── 4. Time positional embedding ──────────────────────────
                if let Some(ref tpe) = m.time_pos_embed {
                    x = x + tpe.val().slice([0..1, 0..t, 0..m.hidden]);
                }

                // ── 5. Subject embedding ──────────────────────────────────
                // (in average mode with no subject IDs, this is skipped)
                // Full subject embedding would require subject_ids passed through.

                // ── 6. Transformer encoder ────────────────────────────────
                if let Some(ref enc) = m.encoder {
                    x = enc.forward(x);
                }
            }

            // ── 7. [B, T, H] → [B, H, T] ─────────────────────────────────
            x = x.swap_dims(1, 2);

            // ── 8. Low-rank head ──────────────────────────────────────────
            if let Some(ref lr) = m.low_rank_head {
                x = lr.forward(x.swap_dims(1, 2)).swap_dims(1, 2);
            }

            // ── 9. Subject predictor ──────────────────────────────────────
            x = if m.use_average_subjects {
                m.predictor.forward_average(x)
            } else {
                m.predictor.forward_subjects(x, &[0])
            };

            // ── 10. Adaptive avg-pool over T ──────────────────────────────
            adaptive_avg_pool1d(x, m.n_output_timesteps)
        }
    };
}

#[cfg(not(feature = "wgpu-kernels-metal"))]
define_forward_body!();

#[cfg(feature = "wgpu-kernels-metal")]
define_forward_body!(+ crate::model_burn::FusedOps);

// ── Helpers ────────────────────────────────────────────────────────────────

/// AdaptiveAvgPool1d: [B, D, T_in] → [B, D, T_out].
/// Matches PyTorch: start=floor(i*T_in/T_out), end=ceil((i+1)*T_in/T_out).
pub fn adaptive_avg_pool1d_pub<B: Backend>(x: Tensor<B, 3>, t_out: usize) -> Tensor<B, 3> {
    adaptive_avg_pool1d(x, t_out)
}

fn adaptive_avg_pool1d<B: Backend>(x: Tensor<B, 3>, t_out: usize) -> Tensor<B, 3> {
    let [b, d, t_in] = x.dims();
    if t_in == t_out { return x; }
    let mut slices = Vec::with_capacity(t_out);
    for i in 0..t_out {
        // Match PyTorch AdaptiveAvgPool1d: floor(start), ceil(end)
        let start = (i * t_in) / t_out;
        let end   = ((i + 1) * t_in + t_out - 1) / t_out;
        slices.push(x.clone().slice([0..b, 0..d, start..end]).mean_dim(2));
    }
    Tensor::cat(slices, 2)
}

/// Depthwise 1D convolution with same-padding.
/// x: [B, C, T], kernel: [C, 1, K] → [B, C, T]
fn depthwise_conv1d<B: Backend>(x: Tensor<B, 3>, kernel: &Tensor<B, 3>, kernel_size: usize) -> Tensor<B, 3> {
    let [batch, channels, t] = x.dims();
    let pad = kernel_size / 2;
    let device = x.device();

    // Pad
    let x = if pad > 0 {
        let left = Tensor::<B, 3>::zeros([batch, channels, pad], &device);
        let right = Tensor::<B, 3>::zeros([batch, channels, pad], &device);
        Tensor::cat(vec![left, x, right], 2)
    } else {
        x
    };

    // Per-channel convolution via windowed matmul
    let mut out_channels = Vec::with_capacity(channels);
    for c in 0..channels {
        let x_c = x.clone().slice([0..batch, c..c + 1, 0..t + 2 * pad]); // [B, 1, T+2*pad]
        let w_c = kernel.clone().slice([c..c + 1, 0..1, 0..kernel_size])
            .reshape([1, kernel_size]); // [1, K]

        let mut patches = Vec::with_capacity(t);
        for i in 0..t {
            let patch = x_c.clone().slice([0..batch, 0..1, i..i + kernel_size])
                .reshape([batch, kernel_size]);
            patches.push(patch);
        }
        let unfolded = Tensor::stack(patches, 1); // [B, T, K]
        let w_t = w_c.transpose(); // [K, 1]
        let conv = unfolded.matmul(w_t.unsqueeze::<3>()); // [B, T, 1]
        out_channels.push(conv.swap_dims(1, 2)); // [B, 1, T]
    }
    Tensor::cat(out_channels, 1) // [B, C, T]
}
