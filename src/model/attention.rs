//! Multi-head self-attention — x_transformers implementation.
//!
//! Python (x_transformers `Attention`):
//! - to_q, to_k, to_v: Linear(dim, dim, bias=False)
//! - to_out: Linear(dim, dim, bias=False)
//! - scale = dim_head ** -0.5
//! - Softmax attention with optional rotary embeddings.

use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct Attention {
    /// to_q weight: [dim, q_dim] where q_dim = dim_head * heads = dim
    pub w_q: Tensor,
    /// to_k weight: [dim, k_dim]
    pub w_k: Tensor,
    /// to_v weight: [dim, v_dim]
    pub w_v: Tensor,
    /// to_out weight: [dim, dim]
    pub w_out: Tensor,
    /// Number of attention heads.
    pub heads: usize,
    /// Dimension per head.
    pub dim_head: usize,
    /// Scale factor = dim_head ** -0.5
    pub scale: f32,
}

impl Attention {
    pub fn new(dim: usize, heads: usize) -> Self {
        let dim_head = dim / heads;
        Self {
            w_q: Tensor::zeros(&[dim, dim]),
            w_k: Tensor::zeros(&[dim, dim]),
            w_v: Tensor::zeros(&[dim, dim]),
            w_out: Tensor::zeros(&[dim, dim]),
            heads,
            dim_head,
            scale: (dim_head as f32).powf(-0.5),
        }
    }

    /// Forward pass: x [B, N, D] → [B, N, D]
    /// `rotary_freqs`: optional [N, rot_dim] raw angle frequencies.
    pub fn forward(&self, x: &Tensor, rotary_freqs: Option<&Tensor>) -> Tensor {
        let (b, n, d) = (x.shape[0], x.shape[1], x.shape[2]);
        let h = self.heads;
        let dh = self.dim_head;

        // Project Q, K, V: [B, N, D] × [D, D]^T = [B, N, D]
        // Note: x_transformers Linear(dim, q_dim, bias=False)
        // PyTorch Linear stores weight as [out, in], forward = x @ W^T
        // But we store W as [in, out] (transposed from PyTorch), so x @ W.
        let q = x.reshape(&[b * n, d]).matmul(&self.w_q).reshape(&[b, n, d]);
        let k = x.reshape(&[b * n, d]).matmul(&self.w_k).reshape(&[b, n, d]);
        let v = x.reshape(&[b * n, d]).matmul(&self.w_v).reshape(&[b, n, d]);

        // Reshape to [B, H, N, D_head]
        let q = q.reshape(&[b, n, h, dh]).permute(&[0, 2, 1, 3]);
        let k = k.reshape(&[b, n, h, dh]).permute(&[0, 2, 1, 3]);
        let v = v.reshape(&[b, n, h, dh]).permute(&[0, 2, 1, 3]);

        // Apply rotary embeddings
        let (q, k) = if let Some(freqs) = rotary_freqs {
            (q.apply_rotary_pos_emb(freqs), k.apply_rotary_pos_emb(freqs))
        } else {
            (q, k)
        };

        // Scaled dot-product attention
        // QK^T: [B, H, N, Dh] × [B, H, Dh, N] → [B, H, N, N]
        let kt = k.permute(&[0, 1, 3, 2]); // transpose last 2
        let scores = q.matmul(&kt).mul_scalar(self.scale);

        // Softmax (cast to f32 as in x_transformers: partial(F.softmax, dtype=torch.float32))
        let attn = scores.softmax_last();

        // Attn × V: [B, H, N, N] × [B, H, N, Dh] → [B, H, N, Dh]
        let out = attn.matmul(&v);

        // Merge heads: [B, H, N, Dh] → [B, N, H*Dh] = [B, N, D]
        let out = out.permute(&[0, 2, 1, 3]).reshape(&[b, n, d]);

        // Output projection
        out.reshape(&[b * n, d]).matmul(&self.w_out).reshape(&[b, n, d])
    }
}
