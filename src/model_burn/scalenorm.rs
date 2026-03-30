use burn::prelude::*;
use burn::module::{Param, ParamId};

/// ScaleNorm from x_transformers: F.normalize(x, dim=-1) * sqrt(dim) * g
///
/// Optimized: uses single-pass norm computation.
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

    /// F.normalize(x, dim=-1) * scale * g
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // L2 normalize: x / max(||x||, eps)
        // Compute ||x||^2 = sum(x*x, dim=-1), then rsqrt for division
        let x_sq = x.clone().powf_scalar(2.0);
        let norm_sq = x_sq.sum_dim(2).clamp_min(1e-24);
        let inv_norm = norm_sq.powf_scalar(-0.5); // 1/||x||

        let g_val = self.g.val().reshape([1, 1, 1]);
        x * inv_norm * g_val.mul_scalar(self.scale)
    }
}
