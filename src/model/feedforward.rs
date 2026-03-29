//! FeedForward — x_transformers implementation.
//!
//! Python (x_transformers):
//! ```python
//! class FeedForward(nn.Module):
//!     def __init__(self, dim, mult=4, dropout=0., no_bias=False):
//!         inner_dim = int(dim * mult)
//!         self.ff = Sequential(
//!             Linear(dim, inner_dim, bias=not no_bias),
//!             GELU(),
//!             Dropout(dropout),
//!             Linear(inner_dim, dim, bias=not no_bias),
//!         )
//! ```
//!
//! For the pretrained model: no_bias=False (default), so bias=True.

use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct FeedForward {
    /// First linear weight: [dim, inner_dim] (stored as [in, out])
    pub w1: Tensor,
    /// First linear bias: [inner_dim]
    pub b1: Tensor,
    /// Second linear weight: [inner_dim, dim]
    pub w2: Tensor,
    /// Second linear bias: [dim]
    pub b2: Tensor,
    pub dim: usize,
    pub inner_dim: usize,
}

impl FeedForward {
    pub fn new(dim: usize, mult: usize) -> Self {
        let inner_dim = dim * mult;
        Self {
            w1: Tensor::zeros(&[dim, inner_dim]),
            b1: Tensor::zeros(&[inner_dim]),
            w2: Tensor::zeros(&[inner_dim, dim]),
            b2: Tensor::zeros(&[dim]),
            dim,
            inner_dim,
        }
    }

    /// Forward: x [B, N, D] → [B, N, D]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let (b, n, d) = (x.shape[0], x.shape[1], x.shape[2]);
        // Linear 1 + GELU
        let h = x.reshape(&[b * n, d])
            .matmul(&self.w1)
            .add_bias(&self.b1)
            .gelu();
        // Linear 2
        h.matmul(&self.w2)
            .add_bias(&self.b2)
            .reshape(&[b, n, d])
    }
}
