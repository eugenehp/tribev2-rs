//! Projector — single Linear, multi-layer MLP, or SubjectLayers (burn).

use burn::prelude::*;
use burn::nn::{Linear, LinearConfig};
use burn::module::{Param, ParamId};
use burn::tensor::activation::gelu;
use super::subject_layers::SubjectLayers;
use crate::config::SubjectLayersConfig;

/// A single MLP layer: Linear + optional LayerNorm + optional GELU.
#[derive(Module, Debug)]
pub struct MlpLayer<B: Backend> {
    pub linear: Linear<B>,
    pub ln_weight: Option<Param<Tensor<B, 1>>>,
    pub ln_bias: Option<Param<Tensor<B, 1>>>,
    pub has_activation: bool,
}

impl<B: Backend> MlpLayer<B> {
    pub fn new(in_dim: usize, out_dim: usize, has_norm: bool, has_activation: bool, device: &B::Device) -> Self {
        Self {
            linear: LinearConfig::new(in_dim, out_dim).with_bias(true).init(device),
            ln_weight: if has_norm {
                Some(Param::initialized(ParamId::new(), Tensor::ones([out_dim], device)))
            } else {
                None
            },
            ln_bias: if has_norm {
                Some(Param::initialized(ParamId::new(), Tensor::zeros([out_dim], device)))
            } else {
                None
            },
            has_activation,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut x = self.linear.forward(x);

        // LayerNorm over last dim
        if let (Some(ref w), Some(ref b)) = (&self.ln_weight, &self.ln_bias) {
            let mean = x.clone().mean_dim(2);
            let diff = x.clone() - mean;
            let var = diff.clone().powf_scalar(2.0).mean_dim(2);
            let x_norm = diff / (var + 1e-5).sqrt();
            x = x_norm * w.val().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(1)
                + b.val().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(1);
        }

        if self.has_activation {
            x = gelu(x);
        }
        x
    }
}

/// Projector: Linear, MLP, or SubjectLayers.
///
/// This enum cannot derive `Module` (mixed generic contents), so we
/// wrap each variant in its own `Module`-deriving struct and dispatch manually.
pub enum Projector<B: Backend> {
    /// Single linear or multi-layer MLP.
    Mlp(MlpProjector<B>),
    /// Per-subject linear projection.
    SubjectLayers(SubjectLayers<B>),
}

// Implement Debug manually since we can't derive it on the enum easily
impl<B: Backend> std::fmt::Debug for Projector<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Mlp(m) => write!(f, "Projector::Mlp({:?})", m),
            Self::SubjectLayers(s) => write!(f, "Projector::SubjectLayers({:?})", s),
        }
    }
}

/// Multi-layer MLP projector (one or more layers).
#[derive(Module, Debug)]
pub struct MlpProjector<B: Backend> {
    pub layers: Vec<MlpLayer<B>>,
}

impl<B: Backend> MlpProjector<B> {
    /// Single linear layer (no norm, no activation).
    pub fn new_linear(in_dim: usize, out_dim: usize, device: &B::Device) -> Self {
        Self {
            layers: vec![MlpLayer::new(in_dim, out_dim, false, false, device)],
        }
    }

    /// Multi-layer MLP: intermediate layers have norm+GELU, final layer is plain Linear.
    pub fn new_mlp(in_dim: usize, out_dim: usize, hidden_sizes: &[usize], has_norm: bool, device: &B::Device) -> Self {
        let mut layers = Vec::new();
        let mut prev = in_dim;
        for &h in hidden_sizes {
            layers.push(MlpLayer::new(prev, h, has_norm, true, device));
            prev = h;
        }
        layers.push(MlpLayer::new(prev, out_dim, false, false, device));
        Self { layers }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(x);
        }
        x
    }
}

impl<B: Backend> Projector<B> {
    pub fn new_linear(in_dim: usize, out_dim: usize, device: &B::Device) -> Self {
        Self::Mlp(MlpProjector::new_linear(in_dim, out_dim, device))
    }

    pub fn new_mlp(in_dim: usize, out_dim: usize, hidden_sizes: &[usize], has_norm: bool, device: &B::Device) -> Self {
        Self::Mlp(MlpProjector::new_mlp(in_dim, out_dim, hidden_sizes, has_norm, device))
    }

    pub fn new_subject_layers(in_ch: usize, out_ch: usize, config: &SubjectLayersConfig, device: &B::Device) -> Self {
        Self::SubjectLayers(SubjectLayers::new(in_ch, out_ch, config, device))
    }

    /// x: [B, T, D] → [B, T, out]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        match self {
            Self::Mlp(mlp) => mlp.forward(x),
            Self::SubjectLayers(sl) => {
                // x: [B, T, D] → [B, D, T] for SubjectLayers, then back
                let x = x.swap_dims(1, 2);
                sl.forward_average(x).swap_dims(1, 2)
            }
        }
    }

    /// Forward with explicit subject IDs.
    pub fn forward_with_subjects(&self, x: Tensor<B, 3>, subject_ids: &[usize]) -> Tensor<B, 3> {
        match self {
            Self::Mlp(mlp) => mlp.forward(x),
            Self::SubjectLayers(sl) => {
                let x = x.swap_dims(1, 2);
                sl.forward_subjects(x, subject_ids).swap_dims(1, 2)
            }
        }
    }
}
