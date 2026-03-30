use burn::prelude::*;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::softmax;

/// Multi-head self-attention matching x_transformers' non-causal encoder.
///
/// 1. **Manual QK^T + softmax + V** — the cubecl flash-attention kernel
///    hardcodes `causal: true` (wrong for a bidirectional encoder) and is
///    slower than plain matmuls for the small T=100 sequence used here.
///
/// 2. **Fused QKV projection + `narrow` split** — one [D→3D] matmul,
///    then view-level splits; avoids 3 separate GPU buffer copies.
///
/// 3. **Fused RoPE** (when `wgpu-kernels-metal`) — a single CubeCL kernel
///    replaces the slice×3 + mul×4 + sub + add + cat chain (~10 dispatches,
///    4 intermediate allocations per apply).
#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    pub to_qkv: Linear<B>,
    pub to_out:  Linear<B>,
    pub heads:    usize,
    pub dim_head: usize,
    pub scale:    f32,
}

impl<B: Backend> Attention<B> {
    pub fn new(dim: usize, heads: usize, device: &B::Device) -> Self {
        let dim_head = dim / heads;
        Self {
            to_qkv: LinearConfig::new(dim, 3 * dim).with_bias(false).init(device),
            to_out:  LinearConfig::new(dim,     dim).with_bias(false).init(device),
            heads,
            dim_head,
            scale: (dim_head as f32).powf(-0.5),
        }
    }
}

// ── forward — standard burn-ops path ──────────────────────────────────────
#[cfg(not(feature = "wgpu-kernels-metal"))]
impl<B: Backend> Attention<B> {
    pub fn forward(
        &self,
        x:          Tensor<B, 3>,
        rotary_cos: Option<&Tensor<B, 2>>,
        rotary_sin: Option<&Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        attention_forward(self, x, rotary_cos, rotary_sin)
    }
}

// ── forward — fused-kernel path ────────────────────────────────────────────
#[cfg(feature = "wgpu-kernels-metal")]
impl<B: Backend + crate::model_burn::FusedOps> Attention<B> {
    pub fn forward(
        &self,
        x:          Tensor<B, 3>,
        rotary_cos: Option<&Tensor<B, 2>>,
        rotary_sin: Option<&Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        attention_forward(self, x, rotary_cos, rotary_sin)
    }
}

// ── Shared implementation ──────────────────────────────────────────────────

/// Inner attention logic shared between both cfg branches.
/// The `where` bound differs per cfg but the body is identical.
#[cfg(not(feature = "wgpu-kernels-metal"))]
fn attention_forward<B: Backend>(
    this:       &Attention<B>,
    x:          Tensor<B, 3>,
    rotary_cos: Option<&Tensor<B, 2>>,
    rotary_sin: Option<&Tensor<B, 2>>,
) -> Tensor<B, 3> {
    _attention_forward_body(this, x, rotary_cos, rotary_sin)
}

#[cfg(feature = "wgpu-kernels-metal")]
fn attention_forward<B: Backend + crate::model_burn::FusedOps>(
    this:       &Attention<B>,
    x:          Tensor<B, 3>,
    rotary_cos: Option<&Tensor<B, 2>>,
    rotary_sin: Option<&Tensor<B, 2>>,
) -> Tensor<B, 3> {
    _attention_forward_body(this, x, rotary_cos, rotary_sin)
}

// Both forward-body variants are written out in full below.
// (They're identical — the only difference is the where-bound, which
//  Rust requires to be part of the function signature.)

#[cfg(not(feature = "wgpu-kernels-metal"))]
fn _attention_forward_body<B: Backend>(
    this:       &Attention<B>,
    x:          Tensor<B, 3>,
    rotary_cos: Option<&Tensor<B, 2>>,
    rotary_sin: Option<&Tensor<B, 2>>,
) -> Tensor<B, 3> {
    let [b, n, _d] = x.dims();
    let (h, dh) = (this.heads, this.dim_head);
    let dim     = h * dh;

    let qkv = this.to_qkv.forward(x);
    let q = qkv.clone().narrow(2, 0,     dim)
                .reshape([b, n, h, dh]).swap_dims(1, 2);
    let k = qkv.clone().narrow(2, dim,   dim)
                .reshape([b, n, h, dh]).swap_dims(1, 2);
    let v = qkv       .narrow(2, 2*dim,  dim)
                .reshape([b, n, h, dh]).swap_dims(1, 2);

    let (q, k) = match (rotary_cos, rotary_sin) {
        (Some(cos), Some(sin)) => (
            apply_rotary_precomputed(q, cos, sin),
            apply_rotary_precomputed(k, cos, sin),
        ),
        _ => (q, k),
    };

    let q    = q.mul_scalar(this.scale);
    let attn = softmax(q.matmul(k.swap_dims(2, 3)), 3);
    let out  = attn.matmul(v);
    let out  = out.swap_dims(1, 2).flatten(2, 3);
    this.to_out.forward(out)
}

#[cfg(feature = "wgpu-kernels-metal")]
fn _attention_forward_body<B: Backend + crate::model_burn::FusedOps>(
    this:       &Attention<B>,
    x:          Tensor<B, 3>,
    rotary_cos: Option<&Tensor<B, 2>>,
    rotary_sin: Option<&Tensor<B, 2>>,
) -> Tensor<B, 3> {
    let [b, n, _d] = x.dims();
    let (h, dh) = (this.heads, this.dim_head);
    let dim     = h * dh;

    let qkv = this.to_qkv.forward(x);
    let q = qkv.clone().narrow(2, 0,     dim)
                .reshape([b, n, h, dh]).swap_dims(1, 2);
    let k = qkv.clone().narrow(2, dim,   dim)
                .reshape([b, n, h, dh]).swap_dims(1, 2);
    let v = qkv       .narrow(2, 2*dim,  dim)
                .reshape([b, n, h, dh]).swap_dims(1, 2);

    let (q, k) = match (rotary_cos, rotary_sin) {
        (Some(cos), Some(sin)) => (
            // Fused CubeCL kernel: 1 dispatch vs ~10, no intermediate allocs
            rotate_qk::<B>(q, cos, sin),
            rotate_qk::<B>(k, cos, sin),
        ),
        _ => (q, k),
    };

    let q    = q.mul_scalar(this.scale);
    let attn = softmax(q.matmul(k.swap_dims(2, 3)), 3);
    let out  = attn.matmul(v);
    let out  = out.swap_dims(1, 2).flatten(2, 3);
    this.to_out.forward(out)
}

// ── RoPE dispatch ─────────────────────────────────────────────────────────

/// Standard burn-ops RoPE (used when wgpu-kernels-metal is off).
#[cfg(not(feature = "wgpu-kernels-metal"))]
fn apply_rotary_precomputed<B: Backend>(
    x:   Tensor<B, 4>,
    cos: &Tensor<B, 2>,
    sin: &Tensor<B, 2>,
) -> Tensor<B, 4> {
    let [b, h, n, d] = x.dims();
    let half    = cos.dims()[1];
    let rot_dim = half * 2;

    let x_rot = x.clone().slice([0..b, 0..h, 0..n, 0..rot_dim]);
    let x1    = x_rot.clone().slice([0..b, 0..h, 0..n, 0..half]);
    let x2    = x_rot        .slice([0..b, 0..h, 0..n, half..rot_dim]);

    let c = cos.clone().unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0);
    let s = sin.clone().unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0);

    let r1 = x1.clone() * c.clone() - x2.clone() * s.clone();
    let r2 = x2          * c         + x1          * s;
    let rotated = Tensor::cat(vec![r1, r2], 3);

    if rot_dim < d {
        let x_pass = x.slice([0..b, 0..h, 0..n, rot_dim..d]);
        Tensor::cat(vec![rotated, x_pass], 3)
    } else {
        rotated
    }
}

/// Fused CubeCL RoPE (used when wgpu-kernels-metal is on).
#[cfg(feature = "wgpu-kernels-metal")]
fn rotate_qk<B: Backend + crate::model_burn::FusedOps>(
    x:   Tensor<B, 4>,
    cos: &Tensor<B, 2>,
    sin: &Tensor<B, 2>,
) -> Tensor<B, 4> {
    B::rope_rotate(x, cos.clone(), sin.clone())
}
