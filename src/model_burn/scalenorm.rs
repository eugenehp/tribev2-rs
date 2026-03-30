use burn::prelude::*;
use burn::module::{Param, ParamId};

/// ScaleNorm from x_transformers: F.normalize(x, dim=-1) * sqrt(dim) * g
#[derive(Module, Debug)]
pub struct ScaleNorm<B: Backend> {
    pub g: Param<Tensor<B, 1>>,
    pub scale: f32,
}

impl<B: Backend> ScaleNorm<B> {
    pub fn new(dim: usize, device: &B::Device) -> Self {
        Self {
            g: Param::initialized(ParamId::new(), Tensor::ones([1], device)),
            scale: (dim as f32).sqrt(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // L2 norm: sqrt(sum(x^2, dim=-1, keepdim=True)).clamp(min=1e-12)
        let norm = (x.clone() * x.clone()).sum_dim(2).sqrt().clamp_min(1e-12);
        // normalize then scale
        let g_val = self.g.val().reshape([1, 1, 1]); // [1,1,1] broadcast
        (x / norm).mul_scalar(self.scale) * g_val
    }
}
