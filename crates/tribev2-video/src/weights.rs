//! Weight loading for V-JEPA2 ViT-G from HuggingFace safetensors.
//!
//! Weight key mapping from `facebook/vjepa2-vitg-fpc64-256`:
//! ```text
//! encoder.patch_embed.proj.weight       [D, C, t_p, p, p]
//! encoder.patch_embed.proj.bias         [D]
//! encoder.pos_embed                     [1, N, D]  (or as a buffer)
//! encoder.blocks.{i}.norm1.weight       [D]
//! encoder.blocks.{i}.norm1.bias         [D]
//! encoder.blocks.{i}.attn.qkv.weight    [3D, D]
//! encoder.blocks.{i}.attn.qkv.bias      [3D]
//! encoder.blocks.{i}.attn.proj.weight   [D, D]
//! encoder.blocks.{i}.attn.proj.bias     [D]
//! encoder.blocks.{i}.norm2.weight       [D]
//! encoder.blocks.{i}.norm2.bias         [D]
//! encoder.blocks.{i}.mlp.fc1.weight     [MLP, D]
//! encoder.blocks.{i}.mlp.fc1.bias       [MLP]
//! encoder.blocks.{i}.mlp.fc2.weight     [D, MLP]
//! encoder.blocks.{i}.mlp.fc2.bias       [D]
//! encoder.norm.weight                   [D]
//! encoder.norm.bias                     [D]
//! ```

use std::collections::HashMap;
use anyhow::{Context, Result};
use burn::prelude::*;
use burn::module::{Param, ParamId};

/// Raw weight store.
pub struct WeightStore {
    pub tensors: HashMap<String, (Vec<f32>, Vec<usize>)>,
}

impl WeightStore {
    /// Load from a safetensors file.
    pub fn from_safetensors(path: &str) -> Result<Self> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("failed to read: {}", path))?;
        let st = safetensors::SafeTensors::deserialize(&bytes)?;
        let mut tensors = HashMap::with_capacity(st.len());

        for (key, view) in st.tensors() {
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
            tensors.insert(key.to_string(), (f32s, shape));
        }
        Ok(Self { tensors })
    }

    pub fn take_1d<B: Backend>(&mut self, key: &str, device: &B::Device) -> Result<Tensor<B, 1>> {
        let (data, shape) = self.tensors.remove(key)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {}", key))?;
        assert_eq!(shape.len(), 1, "expected 1D for {}", key);
        Ok(Tensor::from_data(TensorData::new(data, [shape[0]]), device))
    }

    pub fn take_2d<B: Backend>(&mut self, key: &str, device: &B::Device) -> Result<Tensor<B, 2>> {
        let (data, shape) = self.tensors.remove(key)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {}", key))?;
        if shape.len() != 2 {
            // Flatten higher-dim weights to 2D
            let total: usize = shape.iter().product();
            let first = shape[0];
            let rest = total / first;
            Ok(Tensor::from_data(TensorData::new(data, [first, rest]), device))
        } else {
            Ok(Tensor::from_data(TensorData::new(data, [shape[0], shape[1]]), device))
        }
    }

    pub fn take_3d<B: Backend>(&mut self, key: &str, device: &B::Device) -> Result<Tensor<B, 3>> {
        let (data, shape) = self.tensors.remove(key)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {}", key))?;
        if shape.len() == 3 {
            Ok(Tensor::from_data(TensorData::new(data, [shape[0], shape[1], shape[2]]), device))
        } else {
            anyhow::bail!("expected 3D for {}, got {:?}", key, shape)
        }
    }

    pub fn remaining_keys(&self) -> Vec<String> {
        let mut keys: Vec<_> = self.tensors.keys().cloned().collect();
        keys.sort();
        keys
    }
}

/// Load V-JEPA2 weights into the model.
pub fn load_vjepa2_weights<B: Backend>(
    ws: &mut WeightStore,
    model: &mut crate::model::VJepa2<B>,
    device: &B::Device,
) -> Result<()> {
    let prefix = "encoder";

    // ── Patch embedding ──────────────────────────────────────────
    // proj.weight: [D, C, t_p, p, p] → flatten to [D, C*t_p*p*p]
    if let Ok(w) = ws.take_2d::<B>(&format!("{prefix}.patch_embed.proj.weight"), device) {
        model.patch_embed.proj_weight = Param::initialized(ParamId::new(), w);
    }
    if let Ok(b) = ws.take_1d::<B>(&format!("{prefix}.patch_embed.proj.bias"), device) {
        model.patch_embed.proj_bias = Param::initialized(ParamId::new(), b);
    }

    // ── Positional embedding ─────────────────────────────────────
    if let Ok(pe) = ws.take_3d::<B>(&format!("{prefix}.pos_embed"), device) {
        model.pos_embed.pos_embed = Param::initialized(ParamId::new(), pe);
    }

    // ── ViT blocks ───────────────────────────────────────────────
    for (i, block) in model.blocks.iter_mut().enumerate() {
        let p = format!("{prefix}.blocks.{i}");

        // norm1
        if let Ok(w) = ws.take_1d::<B>(&format!("{p}.norm1.weight"), device) {
            block.norm1.weight = Param::initialized(ParamId::new(), w);
        }
        if let Ok(b) = ws.take_1d::<B>(&format!("{p}.norm1.bias"), device) {
            block.norm1.bias = Param::initialized(ParamId::new(), b);
        }

        // Attention QKV
        load_linear(ws, &mut block.attn.qkv, &format!("{p}.attn.qkv"), device)?;
        // Attention proj
        load_linear(ws, &mut block.attn.proj, &format!("{p}.attn.proj"), device)?;

        // norm2
        if let Ok(w) = ws.take_1d::<B>(&format!("{p}.norm2.weight"), device) {
            block.norm2.weight = Param::initialized(ParamId::new(), w);
        }
        if let Ok(b) = ws.take_1d::<B>(&format!("{p}.norm2.bias"), device) {
            block.norm2.bias = Param::initialized(ParamId::new(), b);
        }

        // MLP
        load_linear(ws, &mut block.mlp.fc1, &format!("{p}.mlp.fc1"), device)?;
        load_linear(ws, &mut block.mlp.fc2, &format!("{p}.mlp.fc2"), device)?;
    }

    // ── Final norm ───────────────────────────────────────────────
    if let Ok(w) = ws.take_1d::<B>(&format!("{prefix}.norm.weight"), device) {
        model.norm.weight = Param::initialized(ParamId::new(), w);
    }
    if let Ok(b) = ws.take_1d::<B>(&format!("{prefix}.norm.bias"), device) {
        model.norm.bias = Param::initialized(ParamId::new(), b);
    }

    Ok(())
}

/// Load PyTorch Linear weights into burn Linear.
fn load_linear<B: Backend>(
    ws: &mut WeightStore,
    linear: &mut burn::nn::Linear<B>,
    prefix: &str,
    device: &B::Device,
) -> Result<()> {
    if let Ok(w) = ws.take_2d::<B>(&format!("{prefix}.weight"), device) {
        let record = burn::nn::LinearRecord {
            weight: Param::initialized(ParamId::new(), w.transpose()),
            bias: if let Ok(b) = ws.take_1d::<B>(&format!("{prefix}.bias"), device) {
                Some(Param::initialized(ParamId::new(), b))
            } else {
                linear.clone().into_record().bias
            },
        };
        *linear = linear.clone().load_record(record);
    }
    Ok(())
}
