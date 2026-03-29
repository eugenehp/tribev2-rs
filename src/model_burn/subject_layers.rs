use burn::prelude::*;
use burn::module::{Param, ParamId};
use crate::config::SubjectLayersConfig;

/// SubjectLayersModel: per-subject linear + average_subjects inference mode.
/// weights: [num_weight_subjects, C, D], bias: [num_weight_subjects, D]
#[derive(Module, Debug)]
pub struct SubjectLayers<B: Backend> {
    pub weights: Param<Tensor<B, 3>>,
    pub bias: Option<Param<Tensor<B, 2>>>,
    pub n_subjects: usize,
    pub has_dropout: bool,
}

impl<B: Backend> SubjectLayers<B> {
    pub fn new(in_ch: usize, out_ch: usize, config: &SubjectLayersConfig, device: &B::Device) -> Self {
        let n = config.num_weight_subjects();
        Self {
            weights: Param::initialized(ParamId::new(), Tensor::zeros([n, in_ch, out_ch], device)),
            bias: if config.bias {
                Some(Param::initialized(ParamId::new(), Tensor::zeros([n, out_ch], device)))
            } else {
                None
            },
            n_subjects: config.n_subjects,
            has_dropout: config.subject_dropout.is_some(),
        }
    }

    /// Average-subjects mode: use the dropout-subject row.
    /// x: [B, C, T] → [B, D, T]
    pub fn forward_average(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let idx = self.n_subjects; // dropout subject row
        let w = self.weights.val(); // [N, C, D]

        // Extract w[idx]: [C, D]
        let w_row: Tensor<B, 2> = w.slice([idx..idx + 1]).squeeze(); // [C, D]

        // einsum('bct,cd->bdt'): [B,C,T] → x^T @ w → need to do matmul per batch
        // Transpose x: [B, T, C], matmul w: [C, D] → [B, T, D], transpose → [B, D, T]
        let [b, c, t] = x.dims();
        let xt = x.swap_dims(1, 2); // [B, T, C]
        let out = xt.matmul(w_row.unsqueeze_dim::<3>(0)); // [B, T, D]
        let out = out.swap_dims(1, 2); // [B, D, T]

        if let Some(ref bias) = self.bias {
            let bv: Tensor<B, 1> = bias.val().slice([idx..idx + 1]).squeeze(); // [D]
            out + bv.unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(2) // [1, D, 1]
        } else {
            out
        }
    }

    /// Per-subject gather mode: subjects [B] selects per-batch weights.
    /// x: [B, C, T] → [B, D, T]
    pub fn forward_subjects(&self, x: Tensor<B, 3>, subject_ids: &[usize]) -> Tensor<B, 3> {
        let [b, _c, _t] = x.dims();
        let w = self.weights.val(); // [N, C, D]
        let device = x.device();

        // Gather weights for each batch item
        let indices: Vec<usize> = subject_ids.iter().copied().collect();
        let idx_t = Tensor::<B, 1, Int>::from_data(
            TensorData::new(indices.iter().map(|&i| i as i64).collect::<Vec<_>>(), [b]),
            &device,
        );

        // Select: [B, C, D]
        let w_sel = w.select(0, idx_t.clone());

        // einsum('bct,bcd->bdt')
        let xt = x.swap_dims(1, 2); // [B, T, C]
        let out = xt.matmul(w_sel); // [B, T, D]
        let out = out.swap_dims(1, 2); // [B, D, T]

        if let Some(ref bias) = self.bias {
            let b_sel = bias.val().select(0, idx_t); // [B, D]
            out + b_sel.unsqueeze_dim::<3>(2) // [B, D, 1]
        } else {
            out
        }
    }
}
