//! Rotary Position Embedding — x_transformers implementation.
//!
//! Python (x_transformers):
//! ```python
//! class RotaryEmbedding(nn.Module):
//!     def __init__(self, dim, base=10000, base_rescale_factor=1.):
//!         base *= base_rescale_factor ** (dim / (dim - 2))
//!         inv_freq = 1. / (base ** (arange(0, dim, 2).float() / dim))
//!         self.register_buffer('inv_freq', inv_freq)
//!     def forward(self, t):  # t: positions [N]
//!         freqs = einsum('i , j -> i j', t, self.inv_freq)
//!         freqs = cat((freqs, freqs), dim=-1)
//!         return freqs, 1.
//! ```
//!
//! apply_rotary_pos_emb:
//!   t_rot, t_pass = t[..., :rot_dim], t[..., rot_dim:]
//!   t_rot = t_rot * cos(freqs) + rotate_half(t_rot) * sin(freqs)
//!   return cat(t_rot, t_pass)

use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    /// inv_freq: [dim/2]
    pub inv_freq: Vec<f32>,
    /// Total rotary dimension (= dim parameter, which is rot_dim from config)
    pub dim: usize,
}

impl RotaryEmbedding {
    /// Create rotary embedding with `dim` (= rotary_emb_dim from config).
    /// base_rescale_factor = 1.0 for the pretrained model.
    pub fn new(dim: usize) -> Self {
        let base: f64 = 10000.0;
        // base *= base_rescale_factor ** (dim / (dim - 2))
        // With factor = 1.0, this is just base = 10000.0
        let half = dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| {
                let exp = (2 * i) as f64 / dim as f64;
                (1.0 / base.powf(exp)) as f32
            })
            .collect();
        Self { inv_freq, dim }
    }

    /// Compute rotary frequencies for positions [0..seq_len).
    /// Returns freqs: [seq_len, dim] where dim = rot_dim.
    /// The freqs are raw angle values (not yet cos/sin applied).
    /// In x_transformers, freqs = cat((pos @ inv_freq, pos @ inv_freq), dim=-1)
    pub fn forward(&self, seq_len: usize) -> Tensor {
        let half = self.inv_freq.len();
        let rot_dim = half * 2;
        let mut data = vec![0.0f32; seq_len * rot_dim];
        for pos in 0..seq_len {
            for j in 0..half {
                let freq = pos as f32 * self.inv_freq[j];
                // cat((freqs, freqs), dim=-1): first half and second half are identical
                data[pos * rot_dim + j] = freq;
                data[pos * rot_dim + half + j] = freq;
            }
        }
        Tensor::from_vec(data, vec![seq_len, rot_dim])
    }
}
