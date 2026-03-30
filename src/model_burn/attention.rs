use burn::prelude::*;
use burn::nn::{Linear, LinearConfig};

/// Fused multi-head self-attention with flash attention kernel.
///
/// Optimizations:
/// 1. Fused QKV projection: 1 matmul instead of 3
/// 2. burn::tensor::module::attention() → cubecl flash attention on GPU
/// 3. Pre-computed RoPE cos/sin (passed in, not recomputed)
#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    pub to_qkv: Linear<B>,
    pub to_out: Linear<B>,
    pub heads: usize,
    pub dim_head: usize,
    pub scale: f32,
}

impl<B: Backend> Attention<B> {
    pub fn new(dim: usize, heads: usize, device: &B::Device) -> Self {
        let dim_head = dim / heads;
        Self {
            to_qkv: LinearConfig::new(dim, 3 * dim).with_bias(false).init(device),
            to_out: LinearConfig::new(dim, dim).with_bias(false).init(device),
            heads,
            dim_head,
            scale: (dim_head as f32).powf(-0.5),
        }
    }

    /// x: [B, N, D], rotary_cos/sin: optional pre-computed [N, half_rot] → [B, N, D]
    pub fn forward(
        &self, x: Tensor<B, 3>,
        rotary_cos: Option<&Tensor<B, 2>>,
        rotary_sin: Option<&Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        let [b, n, _d] = x.dims();
        let (h, dh) = (self.heads, self.dim_head);
        let dim = h * dh;

        // Fused QKV: [B, N, D] → [B, N, 3D]
        let qkv = self.to_qkv.forward(x);

        // Split + reshape to [B, H, N, Dh]
        let q = qkv.clone().slice([0..b, 0..n, 0..dim])
            .reshape([b, n, h, dh]).swap_dims(1, 2);
        let k = qkv.clone().slice([0..b, 0..n, dim..2*dim])
            .reshape([b, n, h, dh]).swap_dims(1, 2);
        let v = qkv.slice([0..b, 0..n, 2*dim..3*dim])
            .reshape([b, n, h, dh]).swap_dims(1, 2);

        // Apply RoPE with pre-computed cos/sin
        let (q, k) = match (rotary_cos, rotary_sin) {
            (Some(cos), Some(sin)) => {
                (apply_rotary_precomputed(q, cos, sin),
                 apply_rotary_precomputed(k, cos, sin))
            }
            _ => (q, k),
        };

        // Scale Q (flash attention doesn't do internal scaling)
        let q = q.mul_scalar(self.scale);

        // Flash attention: fused QK^T + softmax + @V in one GPU kernel
        let out = burn::tensor::module::attention(q, k, v, None);

        // Merge heads + output projection
        let out = out.swap_dims(1, 2).flatten(2, 3);
        self.to_out.forward(out)
    }
}

/// Apply rotary with pre-computed cos/sin.
/// x: [B, H, N, D], cos: [N, half_rot], sin: [N, half_rot]
fn apply_rotary_precomputed<B: Backend>(
    x: Tensor<B, 4>,
    cos: &Tensor<B, 2>,
    sin: &Tensor<B, 2>,
) -> Tensor<B, 4> {
    let [b, h, n, d] = x.dims();
    let half = cos.dims()[1];
    let rot_dim = half * 2;

    let x_rot = x.clone().slice([0..b, 0..h, 0..n, 0..rot_dim]);
    let x1 = x_rot.clone().slice([0..b, 0..h, 0..n, 0..half]);
    let x2 = x_rot.slice([0..b, 0..h, 0..n, half..rot_dim]);

    let c = cos.clone().unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0);
    let s = sin.clone().unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0);

    let r1 = x1.clone() * c.clone() - x2.clone() * s.clone();
    let r2 = x2 * c + x1 * s;
    let rotated = Tensor::cat(vec![r1, r2], 3);

    if rot_dim < d {
        let x_pass = x.slice([0..b, 0..h, 0..n, rot_dim..d]);
        Tensor::cat(vec![rotated, x_pass], 3)
    } else {
        rotated
    }
}
