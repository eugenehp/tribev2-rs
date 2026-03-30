//! Weight loading from PyTorch Lightning checkpoint (.ckpt) or safetensors.
//!
//! The TRIBE v2 checkpoint is a PyTorch Lightning .ckpt file containing:
//! - `state_dict`: model weights with prefix `model.`
//! - `model_build_args`: dict with feature_dims, n_outputs, n_output_timesteps
//!
//! State dict keys for the pretrained model (after `model.` prefix is stripped):
//! ```text
//! projectors.text.weight              [384, 6144]   (2 layers × 3072 dim)
//! projectors.text.bias                [384]
//! projectors.audio.weight             [384, 2048]   (2 layers × 1024 dim)
//! projectors.audio.bias               [384]
//! projectors.video.weight             [384, 2816]   (2 layers × 1408 dim)
//! projectors.video.bias               [384]
//! time_pos_embed                      [1, 1024, 1152]
//! encoder.rotary_pos_emb.inv_freq     [36]          (buffer, recomputed — consumed but not used)
//! encoder.layers.{i}.0.0.g           [1]            (ScaleNorm pre-norm; i=0,2,4,...14 = attn)
//! encoder.layers.{i}.1.to_q.weight   [1152, 1152]
//! encoder.layers.{i}.1.to_k.weight   [1152, 1152]
//! encoder.layers.{i}.1.to_v.weight   [1152, 1152]
//! encoder.layers.{i}.1.to_out.weight [1152, 1152]
//! encoder.layers.{i}.2.residual_scale [1152]
//! encoder.layers.{j}.0.0.g           [1]            (ScaleNorm pre-norm; j=1,3,5,...15 = FF)
//! encoder.layers.{j}.1.ff.0.0.weight [4608, 1152]
//! encoder.layers.{j}.1.ff.0.0.bias   [4608]
//! encoder.layers.{j}.1.ff.2.weight   [1152, 4608]
//! encoder.layers.{j}.1.ff.2.bias     [1152]
//! encoder.layers.{j}.2.residual_scale [1152]
//! encoder.final_norm.g               [1]
//! low_rank_head.weight               [2048, 1152]
//! predictor.weights                  [n_subjects, 2048, 20484]
//! predictor.bias                     [n_subjects, 20484]
//! ```
//!
//! Note: the released public checkpoint has `n_subjects=1`. The key names have
//! no `.0.` infix (unlike older checkpoints); the loader tries both forms.

use std::collections::HashMap;
use crate::tensor::Tensor;
use crate::model::tribe::TribeV2;
use crate::model::encoder::{LayerBlock, XTransformerEncoder};

/// A map of weight name → (f32 data, shape).
pub struct WeightMap {
    pub tensors: HashMap<String, (Vec<f32>, Vec<usize>)>,
}

impl WeightMap {
    /// Load from a safetensors file.
    pub fn from_safetensors(path: &str) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path)?;
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

    /// Take a tensor by key, removing it from the map.
    pub fn take(&mut self, key: &str) -> anyhow::Result<Tensor> {
        let (data, shape) = self.tensors.remove(key)
            .ok_or_else(|| anyhow::anyhow!("weight key not found: {key}"))?;
        Ok(Tensor::from_vec(data, shape))
    }

    /// Try to take a tensor; return None if not found.
    pub fn try_take(&mut self, key: &str) -> Option<Tensor> {
        self.tensors.remove(key).map(|(data, shape)| Tensor::from_vec(data, shape))
    }

    /// List all remaining keys (useful for debugging).
    pub fn remaining_keys(&self) -> Vec<String> {
        let mut keys: Vec<_> = self.tensors.keys().cloned().collect();
        keys.sort();
        keys
    }
}

/// Load model weights from a safetensors file.
///
/// Despite the name, this function only reads safetensors format — not raw
/// PyTorch Lightning `.ckpt` files. Convert a `.ckpt` first:
///
/// ```bash
/// python3 scripts/convert_checkpoint.py best.ckpt model.safetensors
/// ```
///
/// The conversion strips the `model.` prefix from all weight keys and saves
/// a `model_build_args.json` sidecar with feature dims and output shape.
pub fn load_checkpoint(path: &str) -> anyhow::Result<WeightMap> {
    WeightMap::from_safetensors(path)
}

/// Load weights from a WeightMap into a TribeV2 model.
///
/// Key mapping follows the PyTorch state_dict with `model.` prefix stripped.
pub fn load_weights(wm: &mut WeightMap, model: &mut TribeV2) -> anyhow::Result<()> {
    // ── Projectors ────────────────────────────────────────────────────
    for np in &mut model.projectors {
        let name = &np.name;
        let projector = &mut np.projector;
        // MLP/Linear projector: projectors.<name>.0.weight, projectors.<name>.0.bias
        // PyTorch Linear weight is [out, in]; we need [in, out] for our matmul
        if let Ok(w) = wm.take(&format!("projectors.{name}.0.weight")) {
            projector.layers[0].weight = w.transpose_last2(); // [out, in] → [in, out]
        } else if let Ok(w) = wm.take(&format!("projectors.{name}.weight")) {
            projector.layers[0].weight = w.transpose_last2();
        }
        if let Ok(b) = wm.take(&format!("projectors.{name}.0.bias")) {
            projector.layers[0].bias = b;
        } else if let Ok(b) = wm.take(&format!("projectors.{name}.bias")) {
            projector.layers[0].bias = b;
        }

        // LayerNorm (if MLP with hidden layers; for the pretrained model this is absent)
        if let Some(ref mut ln_w) = projector.layers[0].ln_weight {
            if let Ok(w) = wm.take(&format!("projectors.{name}.0.1.weight")) {
                *ln_w = w;
            }
        }
    }

    // ── Combiner ──────────────────────────────────────────────────────
    if let Some(ref mut combiner) = model.combiner {
        if let Ok(w) = wm.take("combiner.0.weight") {
            combiner.layers[0].weight = w.transpose_last2();
        } else if let Ok(w) = wm.take("combiner.weight") {
            combiner.layers[0].weight = w.transpose_last2();
        }
        if let Ok(b) = wm.take("combiner.0.bias") {
            combiner.layers[0].bias = b;
        } else if let Ok(b) = wm.take("combiner.bias") {
            combiner.layers[0].bias = b;
        }
    }

    // ── Time positional embedding ─────────────────────────────────────
    if let Some(ref mut tpe) = model.time_pos_embed {
        if let Ok(t) = wm.take("time_pos_embed") {
            *tpe = t;
        }
    }

    // ── Subject embedding ────────────────────────────────────────────
    if let Some(ref mut se) = model.subject_embed {
        if let Ok(t) = wm.take("subject_embed.weight") {
            *se = t;
        }
    }

    // ── Encoder ───────────────────────────────────────────────────────
    if let Some(ref mut encoder) = model.encoder {
        load_encoder_weights(wm, encoder)?;
    }

    // ── Low-rank head ─────────────────────────────────────────────────
    if let Some(ref mut lr) = model.low_rank_head {
        if let Ok(w) = wm.take("low_rank_head.weight") {
            // PyTorch: Linear(hidden, lr, bias=False) → weight [lr, hidden]
            // We need [hidden, lr] for our matmul
            *lr = w.transpose_last2();
        }
    }

    // ── Predictor ─────────────────────────────────────────────────────
    if let Ok(w) = wm.take("predictor.weights") {
        model.predictor.weights = w;
    }
    if let Ok(b) = wm.take("predictor.bias") {
        model.predictor.bias = Some(b);
    }

    // ── Temporal smoothing ────────────────────────────────────────────
    if let Some(ref mut ts) = model.temporal_smoothing {
        if let Ok(k) = wm.take("temporal_smoothing.weight") {
            ts.kernel = k;
        }
    }

    Ok(())
}

/// Load weights into the x-transformers encoder.
///
/// Layer ordering: ('a', 'f') * depth
/// encoder.layers.{i} = [norms, block, residual]
fn load_encoder_weights(wm: &mut WeightMap, encoder: &mut XTransformerEncoder) -> anyhow::Result<()> {
    for (i, layer) in encoder.layers.iter_mut().enumerate() {
        let prefix = format!("encoder.layers.{i}");

        // Pre-norm: ScaleNorm.g → .0.0.g
        if let Ok(g) = wm.take(&format!("{prefix}.0.0.g")) {
            layer.pre_norm.g = g.data[0];
        }

        // Block
        match &mut layer.block {
            LayerBlock::Attn(attn) => {
                // to_q.weight, to_k.weight, to_v.weight, to_out.weight
                // PyTorch Linear(dim, dim, bias=False) → weight [dim, dim]
                // In x_transformers, these are [out_dim, in_dim] (standard PyTorch)
                // We store as [in, out] for x @ W
                if let Ok(w) = wm.take(&format!("{prefix}.1.to_q.weight")) {
                    attn.w_q = w.transpose_last2();
                }
                if let Ok(w) = wm.take(&format!("{prefix}.1.to_k.weight")) {
                    attn.w_k = w.transpose_last2();
                }
                if let Ok(w) = wm.take(&format!("{prefix}.1.to_v.weight")) {
                    attn.w_v = w.transpose_last2();
                }
                if let Ok(w) = wm.take(&format!("{prefix}.1.to_out.weight")) {
                    attn.w_out = w.transpose_last2();
                }
            }
            LayerBlock::FF(ff) => {
                // ff.0.0.weight [inner, dim], ff.0.0.bias [inner]
                // ff.2.weight [dim, inner], ff.2.bias [dim]
                if let Ok(w) = wm.take(&format!("{prefix}.1.ff.0.0.weight")) {
                    ff.w1 = w.transpose_last2(); // [inner, dim] → [dim, inner]
                }
                if let Ok(b) = wm.take(&format!("{prefix}.1.ff.0.0.bias")) {
                    ff.b1 = b;
                }
                if let Ok(w) = wm.take(&format!("{prefix}.1.ff.2.weight")) {
                    ff.w2 = w.transpose_last2(); // [dim, inner] → [inner, dim]
                }
                if let Ok(b) = wm.take(&format!("{prefix}.1.ff.2.bias")) {
                    ff.b2 = b;
                }
            }
        }

        // Residual scale
        if let Some(ref mut rs) = layer.residual.residual_scale {
            if let Ok(s) = wm.take(&format!("{prefix}.2.residual_scale")) {
                *rs = s;
            }
        }
    }

    // Final norm
    if let Ok(g) = wm.take("encoder.final_norm.g") {
        encoder.final_norm.g = g.data[0];
    }

    // Rotary embedding inv_freq is saved as a buffer in the checkpoint but we
    // recompute it from the config (values match to < 1e-7). Consume the key
    // so it doesn't show up in the "unused weights" warning.
    wm.try_take("encoder.rotary_pos_emb.inv_freq");

    Ok(())
}
