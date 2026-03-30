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
    pub projectors: Vec<Projector<B>>,
    pub projector_names: Vec<String>,
    pub time_pos_embed: Option<Param<Tensor<B, 3>>>,
    pub encoder: Option<XTransformerEncoder<B>>,
    pub low_rank_head: Option<Linear<B>>,
    pub predictor: SubjectLayers<B>,
    pub n_outputs: usize,
    pub n_output_timesteps: usize,
    pub hidden: usize,
    pub n_modalities: usize,
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
        let hidden = config.hidden;
        let n_modalities = feature_dims.len();

        let mut projectors = Vec::new();
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
            config.encoder.as_ref().map(|ec| XTransformerEncoder::new(hidden, ec, device))
        } else {
            None
        };

        let low_rank_head = config.low_rank_head.map(|lr| {
            LinearConfig::new(hidden, lr).with_bias(false).init(device)
        });

        let bottleneck = config.low_rank_head.unwrap_or(hidden);
        let sl = config.subject_layers.clone().unwrap_or_default();
        let predictor = SubjectLayers::new(bottleneck, n_outputs, &sl, device);
        let use_average_subjects = sl.average_subjects;

        Self {
            projectors, projector_names, time_pos_embed, encoder, low_rank_head,
            predictor, n_outputs, n_output_timesteps, hidden, n_modalities,
            use_average_subjects,
        }
    }

    /// Forward pass.
    /// features: Vec of (name, tensor [B, L*D, T]) in feature_dims order.
    /// Returns: [B, n_outputs, n_output_timesteps]
    pub fn forward(&self, features: Vec<(&str, Tensor<B, 3>)>) -> Tensor<B, 3> {
        let device = features[0].1.device();
        let b = features[0].1.dims()[0];
        let t = features[0].1.dims()[2];

        // Aggregate: project each modality and cat on last dim
        let mut tensors: Vec<Tensor<B, 3>> = Vec::with_capacity(self.n_modalities);

        // Process provided features
        for (name, data) in &features {
            if let Some(idx) = self.projector_names.iter().position(|n| n == name) {
                // [B, L*D, T] → swap to [B, T, L*D] → project → [B, T, out_dim]
                let x = data.clone().swap_dims(1, 2);
                tensors.push(self.projectors[idx].forward(x));
            }
        }
        // Zero-fill for modalities not in features
        for name in &self.projector_names {
            if !features.iter().any(|(n, _)| n == name) {
                let out_dim = self.hidden / self.n_modalities;
                tensors.push(Tensor::zeros([b, t, out_dim], &device));
            }
        }

        // Cat on last dim → [B, T, H]
        let mut x = Tensor::cat(tensors, 2);

        // Time pos embed
        if let Some(ref tpe) = self.time_pos_embed {
            let tpe_slice = tpe.val().slice([0..1, 0..t, 0..self.hidden]);
            x = x + tpe_slice;
        }

        // Encoder
        if let Some(ref enc) = self.encoder {
            x = enc.forward(x);
        }

        // [B, T, H] → [B, H, T]
        x = x.swap_dims(1, 2);

        // Low-rank head: [B, H, T] → [B, T, H] → Linear → [B, T, LR] → [B, LR, T]
        if let Some(ref lr) = self.low_rank_head {
            x = lr.forward(x.swap_dims(1, 2)).swap_dims(1, 2);
        }

        // Predictor: [B, C, T] → [B, D, T]
        x = if self.use_average_subjects {
            self.predictor.forward_average(x)
        } else {
            self.predictor.forward_subjects(x, &[0])
        };

        // Adaptive avg pool over T: [B, D, T] → [B, D, T']
        adaptive_avg_pool1d(x, self.n_output_timesteps)
    }
}

/// AdaptiveAvgPool1d: [B, D, T_in] → [B, D, T_out]
/// When T_in == T_out (common case for pretrained model), returns identity.
pub fn adaptive_avg_pool1d_pub<B: Backend>(x: Tensor<B, 3>, t_out: usize) -> Tensor<B, 3> {
    adaptive_avg_pool1d(x, t_out)
}

fn adaptive_avg_pool1d<B: Backend>(x: Tensor<B, 3>, t_out: usize) -> Tensor<B, 3> {
    let [b, d, t_in] = x.dims();
    if t_in == t_out {
        return x;
    }
    // General case: compute bin boundaries and use slice + mean
    // For the pretrained model this branch is never taken (T=100, T'=100).
    let mut slices = Vec::with_capacity(t_out);
    for i in 0..t_out {
        let start = (i * t_in) / t_out;
        let end = ((i + 1) * t_in) / t_out;
        let chunk = x.clone().slice([0..b, 0..d, start..end]);
        slices.push(chunk.mean_dim(2));
    }
    Tensor::cat(slices, 2)
}
