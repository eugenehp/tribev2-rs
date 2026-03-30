use burn::prelude::*;
use burn::module::{Param, ParamId};
use crate::config::SubjectLayersConfig;

/// SubjectLayersModel: per-subject linear + average_subjects inference mode.
/// weights: [num_weight_subjects, C, D], bias: [num_weight_subjects, D]
#[derive(Module, Debug)]
pub struct SubjectLayers<B: Backend> {
    pub weights: Param<Tensor<B, 3>>,
    pub bias: Option<Param<Tensor<B, 2>>>,
    /// Pre-transposed average-subject weight: [1, D, C].
    /// Stored ready for `w_avg_t.matmul(x)` so every forward skips the
    /// runtime slice → squeeze → transpose → unsqueeze chain on the full
    /// [N, C, D] weight tensor.
    pub w_avg_t: Param<Tensor<B, 3>>,
    pub n_subjects: usize,
    pub has_dropout: bool,
}

impl<B: Backend> SubjectLayers<B> {
    pub fn new(in_ch: usize, out_ch: usize, config: &SubjectLayersConfig, device: &B::Device) -> Self {
        let n = config.num_weight_subjects();
        let weights_data = Tensor::zeros([n, in_ch, out_ch], device);
        // Pre-compute and store the average-subject weight transposed [1, D, C].
        // avg subject is always the last row (index n_subjects).
        let idx = config.n_subjects;
        let w_avg_t = weights_data
            .clone()
            .slice([idx..idx + 1]) // [1, C, D]
            .squeeze::<2>()        // [C, D]
            .transpose()           // [D, C]
            .unsqueeze::<3>();     // [1, D, C]
        Self {
            weights: Param::initialized(ParamId::new(), weights_data),
            bias: if config.bias {
                Some(Param::initialized(ParamId::new(), Tensor::zeros([n, out_ch], device)))
            } else {
                None
            },
            w_avg_t: Param::initialized(ParamId::new(), w_avg_t),
            n_subjects: config.n_subjects,
            has_dropout: config.subject_dropout.is_some(),
        }
    }

    /// Average-subjects mode: use the pre-transposed dropout-subject weight.
    /// x: [B, C, T] → [B, D, T]
    ///
    /// `w_avg_t` is stored as [1, D, C] and updated in `set_weights_for_inference`.
    /// matmul: [1, D, C] @ [B, C, T] = [B, D, T]
    pub fn forward_average(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // [1, D, C] @ [B, C, T] = [B, D, T]
        let out = self.w_avg_t.val().matmul(x);

        if let Some(ref bias) = self.bias {
            let idx = self.n_subjects;
            let bv: Tensor<B, 1> = bias.val().slice([idx..idx + 1]).squeeze();
            out + bv.unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(2)
        } else {
            out
        }
    }

    /// Call this after loading pretrained weights to re-sync `w_avg_t`
    /// with `weights[n_subjects]`.  During benchmarking with random weights
    /// this is not needed, but it must be called in the real inference path
    /// after `model = model.load_record(record)`.
    pub fn rebuild_w_avg_t(mut self) -> Self {
        let idx = self.n_subjects;
        let w = self.weights.val();
        let w_t = w
            .slice([idx..idx + 1])
            .squeeze::<2>()
            .transpose()
            .unsqueeze::<3>();
        self.w_avg_t = Param::initialized(ParamId::new(), w_t);
        self
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
