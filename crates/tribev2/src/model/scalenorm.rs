//! ScaleNorm — x_transformers normalization.
//!
//! Python (x_transformers):
//! ```python
//! class ScaleNorm(nn.Module):
//!     def __init__(self, dim):
//!         self.scale = dim ** 0.5
//!         self.g = nn.Parameter(torch.ones(1))
//!     def forward(self, x):
//!         return F.normalize(x, dim=-1) * self.scale * self.g
//! ```

use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct ScaleNorm {
    /// Learned scalar parameter `g`.
    pub g: f32,
    /// Fixed scale = dim ** 0.5.
    pub scale: f32,
    /// The dimension this norm operates on.
    pub dim: usize,
}

impl ScaleNorm {
    pub fn new(dim: usize) -> Self {
        Self {
            g: 1.0,
            scale: (dim as f32).sqrt(),
            dim,
        }
    }

    /// F.normalize(x, dim=-1) * scale * g
    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.scale_norm(self.g, self.dim)
    }
}
