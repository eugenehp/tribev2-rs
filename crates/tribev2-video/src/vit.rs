//! Vision Transformer layers for V-JEPA2.
//!
//! Standard ViT encoder block:
//! ```text
//! x → LayerNorm → Attention → Residual → LayerNorm → MLP → Residual
//! ```

use burn::prelude::*;
use burn::module::{Param, ParamId};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::softmax;

/// Layer normalization over the last dimension.
#[derive(Module, Debug)]
pub struct LayerNorm<B: Backend> {
    pub weight: Param<Tensor<B, 1>>,
    pub bias: Param<Tensor<B, 1>>,
    pub eps: f64,
    pub dim: usize,
}

impl<B: Backend> LayerNorm<B> {
    pub fn new(dim: usize, eps: f64, device: &B::Device) -> Self {
        Self {
            weight: Param::initialized(ParamId::new(), Tensor::ones([dim], device)),
            bias: Param::initialized(ParamId::new(), Tensor::zeros([dim], device)),
            eps,
            dim,
        }
    }

    /// x: [B, N, D] → [B, N, D]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mean = x.clone().mean_dim(2);
        let diff = x - mean;
        let var = diff.clone().powf_scalar(2.0).mean_dim(2);
        let x_norm = diff / (var + self.eps).sqrt();
        x_norm * self.weight.val().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(1)
            + self.bias.val().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(1)
    }
}

/// Multi-head self-attention (no causal mask).
#[derive(Module, Debug)]
pub struct ViTAttention<B: Backend> {
    pub qkv: Linear<B>,     // Fused QKV projection: D → 3D
    pub proj: Linear<B>,     // Output projection: D → D
    pub heads: usize,
    pub dim_head: usize,
    pub scale: f32,
}

impl<B: Backend> ViTAttention<B> {
    pub fn new(dim: usize, heads: usize, device: &B::Device) -> Self {
        let dim_head = dim / heads;
        Self {
            qkv: LinearConfig::new(dim, 3 * dim).with_bias(true).init(device),
            proj: LinearConfig::new(dim, dim).with_bias(true).init(device),
            heads,
            dim_head,
            scale: (dim_head as f32).powf(-0.5),
        }
    }

    /// x: [B, N, D] → [B, N, D]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, n, _d] = x.dims();
        let (h, dh) = (self.heads, self.dim_head);
        let dim = h * dh;

        let qkv = self.qkv.forward(x);
        let q = qkv.clone().narrow(2, 0, dim).reshape([b, n, h, dh]).swap_dims(1, 2);
        let k = qkv.clone().narrow(2, dim, dim).reshape([b, n, h, dh]).swap_dims(1, 2);
        let v = qkv.narrow(2, 2 * dim, dim).reshape([b, n, h, dh]).swap_dims(1, 2);

        let q = q.mul_scalar(self.scale);
        let attn = softmax(q.matmul(k.swap_dims(2, 3)), 3);
        let out = attn.matmul(v);
        let out = out.swap_dims(1, 2).flatten(2, 3);
        self.proj.forward(out)
    }
}

/// MLP: Linear → GELU → Linear
#[derive(Module, Debug)]
pub struct ViTMlp<B: Backend> {
    pub fc1: Linear<B>,
    pub fc2: Linear<B>,
}

impl<B: Backend> ViTMlp<B> {
    pub fn new(dim: usize, mlp_dim: usize, device: &B::Device) -> Self {
        Self {
            fc1: LinearConfig::new(dim, mlp_dim).with_bias(true).init(device),
            fc2: LinearConfig::new(mlp_dim, dim).with_bias(true).init(device),
        }
    }

    /// x: [B, N, D] → [B, N, D]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.fc2.forward(burn::tensor::activation::gelu(self.fc1.forward(x)))
    }
}

/// A single ViT encoder block: LayerNorm → Attn → Res → LayerNorm → MLP → Res
#[derive(Module, Debug)]
pub struct ViTBlock<B: Backend> {
    pub norm1: LayerNorm<B>,
    pub attn: ViTAttention<B>,
    pub norm2: LayerNorm<B>,
    pub mlp: ViTMlp<B>,
}

impl<B: Backend> ViTBlock<B> {
    pub fn new(
        dim: usize,
        heads: usize,
        mlp_dim: usize,
        eps: f64,
        device: &B::Device,
    ) -> Self {
        Self {
            norm1: LayerNorm::new(dim, eps, device),
            attn: ViTAttention::new(dim, heads, device),
            norm2: LayerNorm::new(dim, eps, device),
            mlp: ViTMlp::new(dim, mlp_dim, device),
        }
    }

    /// x: [B, N, D] → [B, N, D]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Attention with pre-norm + residual
        let residual = x.clone();
        let x = self.attn.forward(self.norm1.forward(x));
        let x = x + residual;

        // MLP with pre-norm + residual
        let residual = x.clone();
        let x = self.mlp.forward(self.norm2.forward(x));
        x + residual
    }
}
