//! Weight loading from safetensors into the burn TribeV2Burn model.
//!
//! Mirrors `src/weights.rs` but targets burn tensors instead of the
//! pure-Rust `Tensor` type.

use std::collections::HashMap;
use anyhow::{Context, Result};
use burn::prelude::*;
use burn::module::{Param, ParamId};
use burn::nn::LinearRecord;

use super::tribe::TribeV2Burn;
use super::projector::{Projector, MlpProjector};

/// Raw weight store for burn models.
pub struct BurnWeightStore {
    pub tensors: HashMap<String, (Vec<f32>, Vec<usize>)>,
}

impl BurnWeightStore {
    /// Load from a safetensors file, stripping `model.` prefix.
    pub fn from_safetensors(path: &str) -> Result<Self> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("failed to read: {}", path))?;
        let st = safetensors::SafeTensors::deserialize(&bytes)?;
        let mut tensors = HashMap::with_capacity(st.len());
        for (key, view) in st.tensors() {
            let key = key.strip_prefix("model.").unwrap_or(&key).to_string();
            let shape: Vec<usize> = view.shape().to_vec();
            let data = view.data();
            let f32s: Vec<f32> = match view.dtype() {
                safetensors::Dtype::BF16 => {
                    data.chunks_exact(2)
                        .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
                        .collect()
                }
                safetensors::Dtype::F16 => {
                    data.chunks_exact(2)
                        .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
                        .collect()
                }
                safetensors::Dtype::F32 => {
                    data.chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect()
                }
                other => anyhow::bail!("unsupported dtype {:?}", other),
            };
            tensors.insert(key, (f32s, shape));
        }
        Ok(Self { tensors })
    }

    fn take(&mut self, key: &str) -> Option<(Vec<f32>, Vec<usize>)> {
        self.tensors.remove(key)
    }

    fn take_1d<B: Backend>(&mut self, key: &str, device: &B::Device) -> Option<Tensor<B, 1>> {
        self.take(key).map(|(data, shape)| {
            Tensor::from_data(TensorData::new(data, [shape[0]]), device)
        })
    }

    fn take_2d<B: Backend>(&mut self, key: &str, device: &B::Device) -> Option<Tensor<B, 2>> {
        self.take(key).map(|(data, shape)| {
            Tensor::from_data(TensorData::new(data, [shape[0], shape[1]]), device)
        })
    }

    fn take_3d<B: Backend>(&mut self, key: &str, device: &B::Device) -> Option<Tensor<B, 3>> {
        self.take(key).map(|(data, shape)| {
            Tensor::from_data(TensorData::new(data, [shape[0], shape[1], shape[2]]), device)
        })
    }

    pub fn remaining_keys(&self) -> Vec<String> {
        let mut keys: Vec<_> = self.tensors.keys().cloned().collect();
        keys.sort();
        keys
    }
}

/// Load safetensors weights into a burn TribeV2Burn model.
pub fn load_burn_weights<B: Backend>(
    ws: &mut BurnWeightStore,
    model: &mut TribeV2Burn<B>,
    device: &B::Device,
) -> Result<()> {
    // ── Projectors ────────────────────────────────────────────────────
    for (idx, name) in model.projector_names.clone().iter().enumerate() {
        let proj = &mut model.projectors[idx];
        match proj {
            Projector::SubjectLayers(ref mut sl) => {
                if let Some(t) = ws.take_3d::<B>(&format!("projectors.{name}.weights"), device) {
                    sl.weights = Param::initialized(ParamId::new(), t);
                }
                if let Some(t) = ws.take_2d::<B>(&format!("projectors.{name}.bias"), device) {
                    sl.bias = Some(Param::initialized(ParamId::new(), t));
                }
            }
            Projector::Mlp(ref mut mlp) => {
                load_mlp_weights(ws, mlp, &format!("projectors.{name}"), device)?;
            }
        }
    }

    // ── Combiner ──────────────────────────────────────────────────────
    if let Some(ref mut combiner) = model.combiner {
        load_mlp_weights(ws, combiner, "combiner", device)?;
    }

    // ── Time positional embedding ─────────────────────────────────────
    if let Some(ref mut tpe) = model.time_pos_embed {
        if let Some(t) = ws.take_3d::<B>("time_pos_embed", device) {
            *tpe = Param::initialized(ParamId::new(), t);
        }
    }

    // ── Subject embedding ─────────────────────────────────────────────
    if let Some(ref mut se) = model.subject_embed {
        if let Some(t) = ws.take_2d::<B>("subject_embed.weight", device) {
            *se = Param::initialized(ParamId::new(), t);
        }
    }

    // ── Encoder ───────────────────────────────────────────────────────
    if let Some(ref mut encoder) = model.encoder {
        let depth = encoder.attns.len();
        for i in 0..depth {
            let attn_prefix = format!("encoder.layers.{}", i * 2);
            let ff_prefix = format!("encoder.layers.{}", i * 2 + 1);

            // Attention pre-norm
            if let Some(g) = ws.take_1d::<B>(&format!("{attn_prefix}.0.0.g"), device) {
                encoder.attn_norms[i].g = Param::initialized(ParamId::new(), g);
            }

            // Attention QKV — Python has separate to_q/to_k/to_v, burn fuses into to_qkv
            // We need to concatenate [to_q.weight; to_k.weight; to_v.weight] → to_qkv.weight
            let q_w = ws.take_2d::<B>(&format!("{attn_prefix}.1.to_q.weight"), device);
            let k_w = ws.take_2d::<B>(&format!("{attn_prefix}.1.to_k.weight"), device);
            let v_w = ws.take_2d::<B>(&format!("{attn_prefix}.1.to_v.weight"), device);
            if let (Some(q), Some(k), Some(v)) = (q_w, k_w, v_w) {
                // PyTorch: [out, in], concat along out dim → [3*out, in] → transpose → [in, 3*out]
                let qkv = Tensor::cat(vec![q, k, v], 0).transpose();
                let record = LinearRecord {
                    weight: Param::initialized(ParamId::new(), qkv),
                    bias: None,
                };
                encoder.attns[i].to_qkv = encoder.attns[i].to_qkv.clone().load_record(record);
            }

            // Attention output projection
            if let Some(w) = ws.take_2d::<B>(&format!("{attn_prefix}.1.to_out.weight"), device) {
                let record = LinearRecord {
                    weight: Param::initialized(ParamId::new(), w.transpose()),
                    bias: None,
                };
                encoder.attns[i].to_out = encoder.attns[i].to_out.clone().load_record(record);
            }

            // Attention residual scale
            if let Some(ref mut rs) = encoder.attn_residuals[i].residual_scale {
                if let Some(s) = ws.take_1d::<B>(&format!("{attn_prefix}.2.residual_scale"), device) {
                    *rs = Param::initialized(ParamId::new(), s);
                }
            }

            // FF pre-norm
            if let Some(g) = ws.take_1d::<B>(&format!("{ff_prefix}.0.0.g"), device) {
                encoder.ff_norms[i].g = Param::initialized(ParamId::new(), g);
            }

            // FF fc1
            load_burn_linear(ws, &mut encoder.ffs[i].fc1, &format!("{ff_prefix}.1.ff.0.0"), device);
            // FF fc2
            load_burn_linear(ws, &mut encoder.ffs[i].fc2, &format!("{ff_prefix}.1.ff.2"), device);

            // FF residual scale
            if let Some(ref mut rs) = encoder.ff_residuals[i].residual_scale {
                if let Some(s) = ws.take_1d::<B>(&format!("{ff_prefix}.2.residual_scale"), device) {
                    *rs = Param::initialized(ParamId::new(), s);
                }
            }
        }

        // Final norm
        if let Some(g) = ws.take_1d::<B>("encoder.final_norm.g", device) {
            encoder.final_norm.g = Param::initialized(ParamId::new(), g);
        }

        // Consume rotary inv_freq buffer
        ws.take("encoder.rotary_pos_emb.inv_freq");
    }

    // ── Low-rank head ─────────────────────────────────────────────────
    if let Some(ref mut lr) = model.low_rank_head {
        if let Some(w) = ws.take_2d::<B>("low_rank_head.weight", device) {
            let record = LinearRecord {
                weight: Param::initialized(ParamId::new(), w.transpose()),
                bias: None,
            };
            *lr = lr.clone().load_record(record);
        }
    }

    // ── Predictor ─────────────────────────────────────────────────────
    if let Some(w) = ws.take_3d::<B>("predictor.weights", device) {
        model.predictor.weights = Param::initialized(ParamId::new(), w);
        // Rebuild w_avg_t cache
        model.predictor = model.predictor.clone().rebuild_w_avg_t();
    }
    if let Some(b) = ws.take_2d::<B>("predictor.bias", device) {
        model.predictor.bias = Some(Param::initialized(ParamId::new(), b));
    }

    // ── Temporal smoothing ────────────────────────────────────────────
    if let Some(ref mut k) = model.temporal_smoothing_kernel {
        if let Some(t) = ws.take_3d::<B>("temporal_smoothing.weight", device) {
            *k = Param::initialized(ParamId::new(), t);
        }
    }

    Ok(())
}

/// Load weights into a burn Linear from PyTorch format (weight transposed).
fn load_burn_linear<B: Backend>(
    ws: &mut BurnWeightStore,
    linear: &mut burn::nn::Linear<B>,
    prefix: &str,
    device: &B::Device,
) {
    let w = ws.take_2d::<B>(&format!("{prefix}.weight"), device);
    let b = ws.take_1d::<B>(&format!("{prefix}.bias"), device);
    if let Some(w) = w {
        let record = LinearRecord {
            weight: Param::initialized(ParamId::new(), w.transpose()),
            bias: b.map(|b| Param::initialized(ParamId::new(), b)),
        };
        *linear = linear.clone().load_record(record);
    }
}

/// Load weights into MlpProjector layers.
fn load_mlp_weights<B: Backend>(
    ws: &mut BurnWeightStore,
    mlp: &mut MlpProjector<B>,
    prefix: &str,
    device: &B::Device,
) -> Result<()> {
    let n_layers = mlp.layers.len();

    if n_layers == 1 {
        // Single linear: try prefix.0.weight then prefix.weight
        let w = ws.take_2d::<B>(&format!("{prefix}.0.weight"), device)
            .or_else(|| ws.take_2d::<B>(&format!("{prefix}.weight"), device));
        let b = ws.take_1d::<B>(&format!("{prefix}.0.bias"), device)
            .or_else(|| ws.take_1d::<B>(&format!("{prefix}.bias"), device));
        if let Some(w) = w {
            let record = LinearRecord {
                weight: Param::initialized(ParamId::new(), w.transpose()),
                bias: b.map(|b| Param::initialized(ParamId::new(), b)),
            };
            mlp.layers[0].linear = mlp.layers[0].linear.clone().load_record(record);
        }
    } else {
        // Multi-layer: torchvision stride-4 layout
        for (li, layer) in mlp.layers.iter_mut().enumerate() {
            let pytorch_idx = if li < n_layers - 1 { li * 4 } else { (n_layers - 1) * 4 };

            let w = ws.take_2d::<B>(&format!("{prefix}.{pytorch_idx}.0.weight"), device)
                .or_else(|| ws.take_2d::<B>(&format!("{prefix}.{pytorch_idx}.weight"), device));
            let b = ws.take_1d::<B>(&format!("{prefix}.{pytorch_idx}.0.bias"), device)
                .or_else(|| ws.take_1d::<B>(&format!("{prefix}.{pytorch_idx}.bias"), device));
            if let Some(w) = w {
                let record = LinearRecord {
                    weight: Param::initialized(ParamId::new(), w.transpose()),
                    bias: b.map(|b| Param::initialized(ParamId::new(), b)),
                };
                layer.linear = layer.linear.clone().load_record(record);
            }

            // LayerNorm
            if let Some(ref mut ln_w) = layer.ln_weight {
                if let Some(w) = ws.take_1d::<B>(&format!("{prefix}.{pytorch_idx}.1.weight"), device) {
                    *ln_w = Param::initialized(ParamId::new(), w);
                }
            }
            if let Some(ref mut ln_b) = layer.ln_bias {
                if let Some(b) = ws.take_1d::<B>(&format!("{prefix}.{pytorch_idx}.1.bias"), device) {
                    *ln_b = Param::initialized(ParamId::new(), b);
                }
            }
        }
    }
    Ok(())
}
