use burn::prelude::*;
use burn::nn::{Linear, LinearConfig};
use super::rotary::apply_rotary;

/// Multi-head self-attention from x_transformers.
/// Q/K/V/Out: Linear(dim, dim, bias=False), scale = dim_head^-0.5
#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    pub to_q: Linear<B>,
    pub to_k: Linear<B>,
    pub to_v: Linear<B>,
    pub to_out: Linear<B>,
    pub heads: usize,
    pub dim_head: usize,
    pub scale: f32,
}

impl<B: Backend> Attention<B> {
    pub fn new(dim: usize, heads: usize, device: &B::Device) -> Self {
        let dim_head = dim / heads;
        Self {
            to_q:   LinearConfig::new(dim, dim).with_bias(false).init(device),
            to_k:   LinearConfig::new(dim, dim).with_bias(false).init(device),
            to_v:   LinearConfig::new(dim, dim).with_bias(false).init(device),
            to_out: LinearConfig::new(dim, dim).with_bias(false).init(device),
            heads,
            dim_head,
            scale: (dim_head as f32).powf(-0.5),
        }
    }

    /// x: [B, N, D], rotary_freqs: optional [N, rot_dim] → [B, N, D]
    pub fn forward(&self, x: Tensor<B, 3>, rotary_freqs: Option<&Tensor<B, 2>>) -> Tensor<B, 3> {
        let [b, n, d] = x.dims();
        let (h, dh) = (self.heads, self.dim_head);

        let q = self.to_q.forward(x.clone()).reshape([b, n, h, dh]).swap_dims(1, 2);
        let k = self.to_k.forward(x.clone()).reshape([b, n, h, dh]).swap_dims(1, 2);
        let v = self.to_v.forward(x).reshape([b, n, h, dh]).swap_dims(1, 2);

        let (q, k) = if let Some(freqs) = rotary_freqs {
            (apply_rotary(q, freqs), apply_rotary(k, freqs))
        } else {
            (q, k)
        };

        // Scaled dot-product: [B,H,N,Dh] @ [B,H,Dh,N] → [B,H,N,N]
        let scores = q.matmul(k.swap_dims(2, 3)).mul_scalar(self.scale);
        let attn = burn::tensor::activation::softmax(scores, 3);
        let out = attn.matmul(v); // [B, H, N, Dh]

        // Merge heads: [B, H, N, Dh] → [B, N, D]
        let out = out.swap_dims(1, 2).reshape([b, n, d]);
        self.to_out.forward(out)
    }
}
