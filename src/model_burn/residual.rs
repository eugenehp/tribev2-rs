use burn::prelude::*;
use burn::module::{Param, ParamId};

/// Residual with optional per-dim scale (from x_transformers).
#[derive(Module, Debug)]
pub struct Residual<B: Backend> {
    pub residual_scale: Option<Param<Tensor<B, 1>>>,
}

impl<B: Backend> Residual<B> {
    pub fn new(dim: usize, scale_residual: bool, device: &B::Device) -> Self {
        Self {
            residual_scale: if scale_residual {
                Some(Param::initialized(ParamId::new(), Tensor::ones([dim], device)))
            } else {
                None
            },
        }
    }

    /// x + residual * scale. Consumes both inputs to avoid copies.
    pub fn forward(&self, x: Tensor<B, 3>, residual: Tensor<B, 3>) -> Tensor<B, 3> {
        match &self.residual_scale {
            Some(scale) => {
                // scale: [D] → [1, 1, D] broadcast
                let s = scale.val().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0);
                x + residual * s
            }
            None => x + residual,
        }
    }
}
