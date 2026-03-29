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
        let [_b, _n, _d] = x.dims();
        // L2 norm over last dim
        let norm = x.clone().powf_scalar(2.0).sum_dim(2).sqrt().clamp_min(1e-12);
        let normalized = x / norm;
        let g_val = self.g.val();
        normalized.mul_scalar(self.scale) * g_val.unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0)
    }
}
