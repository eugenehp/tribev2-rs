//! Residual connection — x_transformers implementation.
//!
//! Python (x_transformers):
//! ```python
//! class Residual(nn.Module):
//!     def __init__(self, dim, scale_residual=False):
//!         self.residual_scale = nn.Parameter(torch.ones(dim)) if scale_residual else None
//!     def forward(self, x, residual):
//!         if self.residual_scale is not None:
//!             residual = residual * self.residual_scale
//!         return x + residual
//! ```

use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct Residual {
    /// Per-dimension residual scale. `None` if `scale_residual=False`.
    pub residual_scale: Option<Tensor>,
}

impl Residual {
    pub fn new(dim: usize, scale_residual: bool) -> Self {
        let residual_scale = if scale_residual {
            Some(Tensor::from_vec(vec![1.0; dim], vec![dim]))
        } else {
            None
        };
        Self { residual_scale }
    }

    /// x + residual (optionally scaled).
    pub fn forward(&self, x: &Tensor, residual: &Tensor) -> Tensor {
        match &self.residual_scale {
            Some(scale) => {
                // residual * scale (broadcast over batch dims)
                let nd = residual.ndim();
                let d = *residual.shape.last().unwrap();
                assert_eq!(scale.shape[0], d);
                let batch: usize = residual.shape[..nd - 1].iter().product();
                let mut scaled = residual.data.clone();
                for b in 0..batch {
                    let base = b * d;
                    for i in 0..d {
                        scaled[base + i] *= scale.data[i];
                    }
                }
                let scaled = Tensor::from_vec(scaled, residual.shape.clone());
                x.add(&scaled)
            }
            None => x.add(residual),
        }
    }
}
