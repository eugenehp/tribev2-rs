//! SubjectLayers — per-subject linear prediction head.
//!
//! Python (neuraltrain/models/common.py `SubjectLayersModel`):
//! ```python
//! class SubjectLayersModel(nn.Module):
//!     def __init__(self, in_channels, out_channels, n_subjects=200,
//!                  bias=True, subject_dropout=None, average_subjects=False, ...):
//!         num_weight_subjects = n_subjects + 1 if subject_dropout else n_subjects
//!         self.weights = nn.Parameter(empty(num_weight_subjects, in_channels, out_channels))
//!         self.bias = nn.Parameter(empty(num_weight_subjects, out_channels)) if bias else None
//!
//!     def forward(self, x, subjects):
//!         # x: [B, C, T] or [B, C], subjects: [B]
//!         if self.average_subjects:
//!             weights = self.weights[self.n_subjects]  # [C, D]
//!             out = einsum('bct,cd->bdt', x, weights)
//!             if self.bias is not None:
//!                 out += self.bias[self.n_subjects].view(1, D, 1)
//!         else:
//!             # gather mode (default):
//!             weights = self.weights.index_select(0, subjects)  # [B, C, D]
//!             out = einsum('bct,bcd->bdt', x, weights)
//!             if self.bias is not None:
//!                 out += self.bias.index_select(0, subjects).view(B, D, 1)
//! ```

use crate::tensor::Tensor;
use crate::config::SubjectLayersConfig;

#[derive(Debug, Clone)]
pub struct SubjectLayers {
    /// Weights: [num_weight_subjects, in_channels, out_channels]
    pub weights: Tensor,
    /// Bias: [num_weight_subjects, out_channels] (optional)
    pub bias: Option<Tensor>,
    pub config: SubjectLayersConfig,
}

impl SubjectLayers {
    pub fn new(in_channels: usize, out_channels: usize, config: &SubjectLayersConfig) -> Self {
        let n = config.num_weight_subjects();
        Self {
            weights: Tensor::zeros(&[n, in_channels, out_channels]),
            bias: if config.bias {
                Some(Tensor::zeros(&[n, out_channels]))
            } else {
                None
            },
            config: config.clone(),
        }
    }

    /// Forward pass.
    ///
    /// - `x`: [B, C, T]
    /// - `subjects`: per-batch subject indices [B]. In average_subjects mode this is ignored.
    ///
    /// Returns: [B, D, T]
    pub fn forward(&self, x: &Tensor, subjects: Option<&[usize]>) -> Tensor {
        let (b, c, t) = (x.shape[0], x.shape[1], x.shape[2]);
        let d = self.weights.shape[2];

        if self.config.average_subjects {
            // Use the dropout subject (last row = n_subjects index)
            let idx = self.config.n_subjects;

            // Extract weights[idx]: [in_channels, out_channels]
            let w_offset = idx * c * d;
            let w_slice = Tensor::from_vec(
                self.weights.data[w_offset..w_offset + c * d].to_vec(),
                vec![c, d],
            );

            // einsum('bct,cd->bdt', x, w)
            let out = x.einsum_bct_cd_bdt(&w_slice);

            // Add bias if present: bias[idx].view(1, D, 1) broadcast
            if let Some(ref bias) = self.bias {
                let b_offset = idx * d;
                let b_data: Vec<f32> = bias.data[b_offset..b_offset + d].to_vec();
                return self.add_bias_3d(&out, &b_data);
            }
            return out;
        }

        // Per-subject mode (gather or for_loop — we implement the equivalent)
        let subj = subjects.unwrap_or(&[0]);

        if b == 1 || (subj.len() == b && subj.windows(2).all(|w| w[0] == w[1])) {
            // All same subject — single matmul (equivalent to for_loop fast path)
            let idx = if subj.is_empty() { 0 } else { subj[0] };
            let w_offset = idx * c * d;
            let w_slice = Tensor::from_vec(
                self.weights.data[w_offset..w_offset + c * d].to_vec(),
                vec![c, d],
            );
            let out = x.einsum_bct_cd_bdt(&w_slice);
            if let Some(ref bias) = self.bias {
                let b_offset = idx * d;
                let b_data: Vec<f32> = bias.data[b_offset..b_offset + d].to_vec();
                return self.add_bias_3d(&out, &b_data);
            }
            return out;
        }

        // Gather mode: per-batch subject weights
        // einsum('bct,bcd->bdt') — one matmul per batch item
        let mut out_data = vec![0.0f32; b * d * t];
        for bi in 0..b {
            let idx = if bi < subj.len() { subj[bi] } else { 0 };
            let w_offset = idx * c * d;
            for di in 0..d {
                for ti in 0..t {
                    let mut sum = 0.0f32;
                    for ci in 0..c {
                        sum += x.data[bi * c * t + ci * t + ti]
                            * self.weights.data[w_offset + ci * d + di];
                    }
                    out_data[bi * d * t + di * t + ti] = sum;
                }
            }
            // Add bias
            if let Some(ref bias) = self.bias {
                let b_off = idx * d;
                for di in 0..d {
                    let bv = bias.data[b_off + di];
                    for ti in 0..t {
                        out_data[bi * d * t + di * t + ti] += bv;
                    }
                }
            }
        }
        Tensor::from_vec(out_data, vec![b, d, t])
    }

    /// Helper: add bias [D] broadcast as [1, D, 1] to a [B, D, T] tensor.
    fn add_bias_3d(&self, x: &Tensor, bias_data: &[f32]) -> Tensor {
        let (b, d, t) = (x.shape[0], x.shape[1], x.shape[2]);
        let mut data = x.data.clone();
        for bi in 0..b {
            for di in 0..d {
                let bv = bias_data[di];
                for ti in 0..t {
                    data[bi * d * t + di * t + ti] += bv;
                }
            }
        }
        Tensor::from_vec(data, x.shape.clone())
    }
}
