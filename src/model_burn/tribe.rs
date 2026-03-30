use burn::prelude::*;
use burn::module::{Param, ParamId};
use burn::nn::{Linear, LinearConfig};
use crate::config::{BrainModelConfig, ModalityDims};
use super::projector::Projector;
use super::encoder::XTransformerEncoder;
use super::subject_layers::SubjectLayers;

/// Full TRIBE v2 FmriEncoderModel in burn.
#[derive(Module, Debug)]
pub struct TribeV2Burn<B: Backend> {
    pub projectors:       Vec<Projector<B>>,
    pub projector_names:  Vec<String>,
    pub time_pos_embed:   Option<Param<Tensor<B, 3>>>,
    pub encoder:          Option<XTransformerEncoder<B>>,
    pub low_rank_head:    Option<Linear<B>>,
    pub predictor:        SubjectLayers<B>,
    pub n_outputs:          usize,
    pub n_output_timesteps: usize,
    pub hidden:             usize,
    pub n_modalities:       usize,
    pub use_average_subjects: bool,
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
                projectors.push(Projector::new(in_dim, out_dim, device));
                projector_names.push(md.name.clone());
            }
        }

        let time_pos_embed = if config.time_pos_embedding && !config.linear_baseline {
            Some(Param::initialized(
                ParamId::new(),
                Tensor::zeros([1, config.max_seq_len, hidden], device),
            ))
        } else {
            None
        };

        let encoder = if !config.linear_baseline {
            config.encoder.as_ref().map(|ec| {
                XTransformerEncoder::new(hidden, config.max_seq_len, ec, device)
            })
        } else {
            None
        };

        let low_rank_head = config.low_rank_head.map(|lr| {
            LinearConfig::new(hidden, lr).with_bias(false).init(device)
        });

        let bottleneck = config.low_rank_head.unwrap_or(hidden);
        let sl         = config.subject_layers.clone().unwrap_or_default();
        let predictor  = SubjectLayers::new(bottleneck, n_outputs, &sl, device);
        let use_average_subjects = sl.average_subjects;

        Self {
            projectors, projector_names, time_pos_embed, encoder, low_rank_head,
            predictor, n_outputs, n_output_timesteps, hidden, n_modalities,
            use_average_subjects,
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

// ── Shared body (macro-duplicated to carry the right where-bound) ──────────

/// Macro to define `forward_body` for a given trait bound.
/// The macro emits two identical definitions — the only difference is
/// whether the `FusedOps` bound is present on `B`.
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

            // ── 2. Time positional embedding ──────────────────────────────
            if let Some(ref tpe) = m.time_pos_embed {
                x = x + tpe.val().slice([0..1, 0..t, 0..m.hidden]);
            }

            // ── 3. Transformer encoder ────────────────────────────────────
            if let Some(ref enc) = m.encoder {
                x = enc.forward(x);
            }

            // ── 4. [B, T, H] → [B, H, T] ─────────────────────────────────
            x = x.swap_dims(1, 2);

            // ── 5. Low-rank head ──────────────────────────────────────────
            if let Some(ref lr) = m.low_rank_head {
                x = lr.forward(x.swap_dims(1, 2)).swap_dims(1, 2);
            }

            // ── 6. Subject predictor ──────────────────────────────────────
            x = if m.use_average_subjects {
                m.predictor.forward_average(x)
            } else {
                m.predictor.forward_subjects(x, &[0])
            };

            // ── 7. Adaptive avg-pool over T ───────────────────────────────
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
/// Identity when T_in == T_out (the common inference case).
pub fn adaptive_avg_pool1d_pub<B: Backend>(x: Tensor<B, 3>, t_out: usize) -> Tensor<B, 3> {
    adaptive_avg_pool1d(x, t_out)
}

fn adaptive_avg_pool1d<B: Backend>(x: Tensor<B, 3>, t_out: usize) -> Tensor<B, 3> {
    let [b, d, t_in] = x.dims();
    if t_in == t_out { return x; }
    let mut slices = Vec::with_capacity(t_out);
    for i in 0..t_out {
        let start = (i * t_in) / t_out;
        let end   = ((i + 1) * t_in) / t_out;
        slices.push(x.clone().slice([0..b, 0..d, start..end]).mean_dim(2));
    }
    Tensor::cat(slices, 2)
}
