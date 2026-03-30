//! Temporal smoothing — depthwise Conv1d with optional Gaussian kernel.
//!
//! Python (model.py `TemporalSmoothing`):
//! ```python
//! class TemporalSmoothing(BaseModelConfig):
//!     kernel_size: int = 9
//!     sigma: float | None = None
//!     def build(self, dim):
//!         conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2, bias=False, groups=dim)
//!         if sigma is not None:
//!             kernel = gaussian_kernel_1d(kernel_size, sigma).repeat(dim, 1, 1)
//!             conv.weight.data = kernel; conv.requires_grad = False
//!         return conv
//! ```

use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct TemporalSmoothing {
    /// Kernel: [dim, 1, kernel_size]
    pub kernel: Tensor,
}

impl TemporalSmoothing {
    /// Create with Gaussian kernel.
    pub fn new_gaussian(dim: usize, kernel_size: usize, sigma: f64) -> Self {
        let mut kernel_1d = vec![0.0f32; kernel_size];
        let _half = kernel_size as f64 / 2.0;
        let mut sum = 0.0f64;
        for i in 0..kernel_size {
            let x = i as f64 - (kernel_size / 2) as f64;
            let v = (-0.5 * (x / sigma) * (x / sigma)).exp();
            kernel_1d[i] = v as f32;
            sum += v;
        }
        for v in kernel_1d.iter_mut() {
            *v /= sum as f32;
        }
        // Repeat for each channel
        let mut data = Vec::with_capacity(dim * kernel_size);
        for _ in 0..dim {
            data.extend_from_slice(&kernel_1d);
        }
        Self {
            kernel: Tensor::from_vec(data, vec![dim, 1, kernel_size]),
        }
    }

    /// Create with learnable kernel (weights loaded from checkpoint).
    pub fn new_learnable(dim: usize, kernel_size: usize) -> Self {
        Self {
            kernel: Tensor::zeros(&[dim, 1, kernel_size]),
        }
    }

    /// Forward: x [B, D, T] → [B, D, T] (depthwise conv with groups=dim)
    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.depthwise_conv1d(&self.kernel)
    }
}
