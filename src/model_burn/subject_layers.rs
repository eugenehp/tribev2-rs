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
    ///
    /// einsum('bct,cd->bdt') = (w^T @ x_per_batch)
    /// Rewritten as: reshape x to [B*C, T], do [D, C] @ [C, B*T] = [D, B*T], reshape
    /// But more efficient: x is [B, C, T]. We want out[b,d,t] = sum_c x[b,c,t] * w[c,d]
    /// = for each t: out[:,d,t] = w^T @ x[:,:,t]
    /// Best approach: flatten x to [B, C, T], w to [C, D]
    /// out = einsum via: x.permute(0,2,1) @ w = [B, T, C] @ [C, D] = [B, T, D], then permute
    /// But permute is expensive on large tensors. Instead use reshape tricks.
    ///
    /// Actually the fastest: x is [B, C, T], treat as [B, C, T].
    /// matmul expects [..., M, K] @ [..., K, N].
    /// w^T is [D, C]. We want w^T @ x = [D, C] @ [C, T] per batch.
    /// So: w_t.unsqueeze(0) @ x = [1, D, C] @ [B, C, T] = [B, D, T]. Done!
    pub fn forward_average(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let idx = self.n_subjects;
        let w = self.weights.val(); // [N, C, D]

        // w_row: [C, D] → transpose → [D, C] → unsqueeze → [1, D, C]
        let w_row: Tensor<B, 2> = w.slice([idx..idx + 1]).squeeze();
        let w_t = w_row.transpose().unsqueeze::<3>(); // [1, D, C]

        // [1, D, C] @ [B, C, T] = [B, D, T]
        let out = w_t.matmul(x);

        if let Some(ref bias) = self.bias {
            let bv: Tensor<B, 1> = bias.val().slice([idx..idx + 1]).squeeze();
            out + bv.unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(2)
        } else {
            out
        }
    }

    /// Per-subject gather mode.
    pub fn forward_subjects(&self, x: Tensor<B, 3>, subject_ids: &[usize]) -> Tensor<B, 3> {
        let [b, _c, _t] = x.dims();
        let w = self.weights.val();
        let device = x.device();

        let idx_t = Tensor::<B, 1, Int>::from_data(
            TensorData::new(subject_ids.iter().map(|&i| i as i64).collect::<Vec<_>>(), [b]),
            &device,
        );

        // w_sel: [B, C, D] → transpose last two → [B, D, C]
        let w_sel = w.select(0, idx_t.clone());
        let w_sel_t = w_sel.swap_dims(1, 2); // [B, D, C]

        // [B, D, C] @ [B, C, T] = [B, D, T]
        let out = w_sel_t.matmul(x);

        if let Some(ref bias) = self.bias {
            let b_sel = bias.val().select(0, idx_t); // [B, D]
            out + b_sel.unsqueeze_dim::<3>(2)
        } else {
            out
        }
    }
}
