//! Weight loading for Wav2Vec-BERT 2.0 from HuggingFace safetensors.
//!
//! Weight key mapping from HuggingFace `model.safetensors`:
//! ```text
//! wav2vec2_bert.feature_extractor.conv_layers.{i}.conv.weight    [out, in, K]
//! wav2vec2_bert.feature_extractor.conv_layers.{i}.conv.bias      [out]
//! wav2vec2_bert.feature_extractor.conv_layers.0.layer_norm.weight [out]
//! wav2vec2_bert.feature_extractor.conv_layers.0.layer_norm.bias   [out]
//! wav2vec2_bert.feature_projection.projection.weight              [out, in]
//! wav2vec2_bert.feature_projection.projection.bias                [out]
//! wav2vec2_bert.feature_projection.layer_norm.weight              [D]
//! wav2vec2_bert.feature_projection.layer_norm.bias                [D]
//! wav2vec2_bert.adapter.layers.{i}.conv.weight                    [out, in, K]
//! wav2vec2_bert.adapter.layers.{i}.conv.bias                      [out]
//! wav2vec2_bert.adapter.layers.{i}.layer_norm.weight              [D]
//! wav2vec2_bert.adapter.layers.{i}.layer_norm.bias                [D]
//! wav2vec2_bert.encoder.layers.{i}.self_attn.{q,k,v}_proj.weight [out, in]
//! wav2vec2_bert.encoder.layers.{i}.self_attn.{q,k,v}_proj.bias   [out]
//! wav2vec2_bert.encoder.layers.{i}.self_attn.linear_out.weight   [out, in]
//! wav2vec2_bert.encoder.layers.{i}.self_attn.linear_out.bias     [out]
//! wav2vec2_bert.encoder.layers.{i}.self_attn_layer_norm.weight   [D]
//! wav2vec2_bert.encoder.layers.{i}.self_attn_layer_norm.bias     [D]
//! wav2vec2_bert.encoder.layers.{i}.conv_module.pointwise_conv1.{weight,bias}
//! wav2vec2_bert.encoder.layers.{i}.conv_module.depthwise_conv.{weight,bias}
//! wav2vec2_bert.encoder.layers.{i}.conv_module.batch_norm.{weight,bias,running_mean,running_var}
//! wav2vec2_bert.encoder.layers.{i}.conv_module.pointwise_conv2.{weight,bias}
//! wav2vec2_bert.encoder.layers.{i}.conv_module.layer_norm.{weight,bias}
//! wav2vec2_bert.encoder.layers.{i}.ffn.intermediate_dense.{weight,bias}
//! wav2vec2_bert.encoder.layers.{i}.ffn.output_dense.{weight,bias}
//! wav2vec2_bert.encoder.layers.{i}.ffn.layer_norm.{weight,bias}
//! wav2vec2_bert.encoder.layers.{i}.final_layer_norm.{weight,bias}
//! ```

use std::collections::HashMap;
use anyhow::{Context, Result};
use burn::prelude::*;
use burn::module::{Param, ParamId};

/// Raw weight store — loads safetensors and provides typed accessors.
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

    /// Take a weight as a burn Tensor<B, 1>.
    pub fn take_1d<B: Backend>(&mut self, key: &str, device: &B::Device) -> Result<Tensor<B, 1>> {
        let (data, shape) = self.tensors.remove(key)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {}", key))?;
        assert_eq!(shape.len(), 1, "expected 1D for {}", key);
        Ok(Tensor::from_data(TensorData::new(data, [shape[0]]), device))
    }

    /// Take a weight as a burn Tensor<B, 2>.
    pub fn take_2d<B: Backend>(&mut self, key: &str, device: &B::Device) -> Result<Tensor<B, 2>> {
        let (data, shape) = self.tensors.remove(key)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {}", key))?;
        assert_eq!(shape.len(), 2, "expected 2D for {}", key);
        Ok(Tensor::from_data(TensorData::new(data, [shape[0], shape[1]]), device))
    }

    /// Take a weight as a burn Tensor<B, 3>.
    pub fn take_3d<B: Backend>(&mut self, key: &str, device: &B::Device) -> Result<Tensor<B, 3>> {
        let (data, shape) = self.tensors.remove(key)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {}", key))?;
        assert_eq!(shape.len(), 3, "expected 3D for {}", key);
        Ok(Tensor::from_data(TensorData::new(data, [shape[0], shape[1], shape[2]]), device))
    }

    /// Try to take a 1D weight, return None if missing.
    pub fn try_take_1d<B: Backend>(&mut self, key: &str, device: &B::Device) -> Option<Tensor<B, 1>> {
        self.take_1d(key, device).ok()
    }

    /// List remaining keys.
    pub fn remaining_keys(&self) -> Vec<String> {
        let mut keys: Vec<_> = self.tensors.keys().cloned().collect();
        keys.sort();
        keys
    }
}

/// Load weights from a HuggingFace safetensors checkpoint into the model.
pub fn load_wav2vec_bert_weights<B: Backend>(
    ws: &mut WeightStore,
    model: &mut crate::model::Wav2VecBertWithConfig<B>,
    device: &B::Device,
) -> Result<()> {
    let model = &mut model.model;
    let prefix = "wav2vec2_bert";

    // ── Feature encoder CNN ──────────────────────────────────────────
    for (i, layer) in model.feature_encoder.layers.iter_mut().enumerate() {
        let p = format!("{prefix}.feature_extractor.conv_layers.{i}");

        if let Ok(w) = ws.take_3d::<B>(&format!("{p}.conv.weight"), device) {
            layer.weight = Param::initialized(ParamId::new(), w);
        }
        if let Ok(b) = ws.take_1d::<B>(&format!("{p}.conv.bias"), device) {
            layer.bias = Some(Param::initialized(ParamId::new(), b));
        }
        if layer.has_group_norm {
            if let Ok(w) = ws.take_1d::<B>(&format!("{p}.layer_norm.weight"), device) {
                layer.group_norm_weight = Some(Param::initialized(ParamId::new(), w));
            }
            if let Ok(b) = ws.take_1d::<B>(&format!("{p}.layer_norm.bias"), device) {
                layer.group_norm_bias = Some(Param::initialized(ParamId::new(), b));
            }
        }
    }

    // ── Feature projection ───────────────────────────────────────────
    {
        let p = format!("{prefix}.feature_projection");
        if let Ok(w) = ws.take_2d::<B>(&format!("{p}.projection.weight"), device) {
            // PyTorch Linear: [out, in] → transpose for x @ W^T
            model.feature_projection.projection_weight = Param::initialized(ParamId::new(), w.transpose());
        }
        if let Ok(b) = ws.take_1d::<B>(&format!("{p}.projection.bias"), device) {
            model.feature_projection.projection_bias = Param::initialized(ParamId::new(), b);
        }
        if let Ok(w) = ws.take_1d::<B>(&format!("{p}.layer_norm.weight"), device) {
            model.feature_projection.layer_norm_weight = Param::initialized(ParamId::new(), w);
        }
        if let Ok(b) = ws.take_1d::<B>(&format!("{p}.layer_norm.bias"), device) {
            model.feature_projection.layer_norm_bias = Param::initialized(ParamId::new(), b);
        }
    }

    // ── Adapter ──────────────────────────────────────────────────────
    for (i, layer) in model.adapter.layers.iter_mut().enumerate() {
        let p = format!("{prefix}.adapter.layers.{i}");
        if let Ok(w) = ws.take_3d::<B>(&format!("{p}.conv.weight"), device) {
            layer.weight = Param::initialized(ParamId::new(), w);
        }
        if let Ok(b) = ws.take_1d::<B>(&format!("{p}.conv.bias"), device) {
            layer.bias = Param::initialized(ParamId::new(), b);
        }
        if let Ok(w) = ws.take_1d::<B>(&format!("{p}.layer_norm.weight"), device) {
            layer.layer_norm_weight = Param::initialized(ParamId::new(), w);
        }
        if let Ok(b) = ws.take_1d::<B>(&format!("{p}.layer_norm.bias"), device) {
            layer.layer_norm_bias = Param::initialized(ParamId::new(), b);
        }
    }

    // ── Conformer encoder layers ─────────────────────────────────────
    for (i, layer) in model.encoder_layers.iter_mut().enumerate() {
        let p = format!("{prefix}.encoder.layers.{i}");

        // Self-attention
        load_linear_weights(ws, &mut layer.self_attn.q_proj, &format!("{p}.self_attn.q_proj"), device)?;
        load_linear_weights(ws, &mut layer.self_attn.k_proj, &format!("{p}.self_attn.k_proj"), device)?;
        load_linear_weights(ws, &mut layer.self_attn.v_proj, &format!("{p}.self_attn.v_proj"), device)?;
        load_linear_weights(ws, &mut layer.self_attn.out_proj, &format!("{p}.self_attn.linear_out"), device)?;

        // Self-attention layer norm
        if let Ok(w) = ws.take_1d::<B>(&format!("{p}.self_attn_layer_norm.weight"), device) {
            layer.self_attn_layer_norm_weight = Param::initialized(ParamId::new(), w);
        }
        if let Ok(b) = ws.take_1d::<B>(&format!("{p}.self_attn_layer_norm.bias"), device) {
            layer.self_attn_layer_norm_bias = Param::initialized(ParamId::new(), b);
        }

        // Conv module
        let cm = &mut layer.conv_module;
        let cp = format!("{p}.conv_module");

        // Pointwise conv1 (stored as Conv1d weight [2D, D, 1])
        if let Ok(w) = ws.take_3d::<B>(&format!("{cp}.pointwise_conv1.weight"), device) {
            let [out_ch, in_ch, _one] = w.dims();
            cm.pointwise_conv1_weight = Param::initialized(ParamId::new(),
                w.reshape([out_ch, in_ch]).transpose()); // [in, out] for matmul
        }
        if let Ok(b) = ws.take_1d::<B>(&format!("{cp}.pointwise_conv1.bias"), device) {
            cm.pointwise_conv1_bias = Param::initialized(ParamId::new(), b);
        }

        // Depthwise conv
        if let Ok(w) = ws.take_3d::<B>(&format!("{cp}.depthwise_conv.weight"), device) {
            cm.depthwise_weight = Param::initialized(ParamId::new(), w);
        }
        if let Ok(b) = ws.take_1d::<B>(&format!("{cp}.depthwise_conv.bias"), device) {
            cm.depthwise_bias = Param::initialized(ParamId::new(), b);
        }

        // Batch norm
        if let Ok(w) = ws.take_1d::<B>(&format!("{cp}.batch_norm.weight"), device) {
            cm.batch_norm_weight = Param::initialized(ParamId::new(), w);
        }
        if let Ok(b) = ws.take_1d::<B>(&format!("{cp}.batch_norm.bias"), device) {
            cm.batch_norm_bias = Param::initialized(ParamId::new(), b);
        }
        if let Ok(m) = ws.take_1d::<B>(&format!("{cp}.batch_norm.running_mean"), device) {
            cm.batch_norm_mean = Param::initialized(ParamId::new(), m);
        }
        if let Ok(v) = ws.take_1d::<B>(&format!("{cp}.batch_norm.running_var"), device) {
            cm.batch_norm_var = Param::initialized(ParamId::new(), v);
        }
        // Consume num_batches_tracked
        ws.tensors.remove(&format!("{cp}.batch_norm.num_batches_tracked"));

        // Pointwise conv2
        if let Ok(w) = ws.take_3d::<B>(&format!("{cp}.pointwise_conv2.weight"), device) {
            let [out_ch, in_ch, _one] = w.dims();
            cm.pointwise_conv2_weight = Param::initialized(ParamId::new(),
                w.reshape([out_ch, in_ch]).transpose());
        }
        if let Ok(b) = ws.take_1d::<B>(&format!("{cp}.pointwise_conv2.bias"), device) {
            cm.pointwise_conv2_bias = Param::initialized(ParamId::new(), b);
        }

        // Conv module layer norm
        if let Ok(w) = ws.take_1d::<B>(&format!("{cp}.layer_norm.weight"), device) {
            cm.layer_norm_weight = Param::initialized(ParamId::new(), w);
        }
        if let Ok(b) = ws.take_1d::<B>(&format!("{cp}.layer_norm.bias"), device) {
            cm.layer_norm_bias = Param::initialized(ParamId::new(), b);
        }

        // Feed-forward
        load_linear_weights(ws, &mut layer.ffn.fc1, &format!("{p}.ffn.intermediate_dense"), device)?;
        load_linear_weights(ws, &mut layer.ffn.fc2, &format!("{p}.ffn.output_dense"), device)?;
        if let Ok(w) = ws.take_1d::<B>(&format!("{p}.ffn.layer_norm.weight"), device) {
            layer.ffn.layer_norm_weight = Param::initialized(ParamId::new(), w);
        }
        if let Ok(b) = ws.take_1d::<B>(&format!("{p}.ffn.layer_norm.bias"), device) {
            layer.ffn.layer_norm_bias = Param::initialized(ParamId::new(), b);
        }

        // Final layer norm
        if let Ok(w) = ws.take_1d::<B>(&format!("{p}.final_layer_norm.weight"), device) {
            layer.final_layer_norm_weight = Param::initialized(ParamId::new(), w);
        }
        if let Ok(b) = ws.take_1d::<B>(&format!("{p}.final_layer_norm.bias"), device) {
            layer.final_layer_norm_bias = Param::initialized(ParamId::new(), b);
        }
    }

    Ok(())
}

/// Load weights into a burn Linear module from PyTorch format.
fn load_linear_weights<B: Backend>(
    ws: &mut WeightStore,
    linear: &mut burn::nn::Linear<B>,
    prefix: &str,
    device: &B::Device,
) -> Result<()> {
    if let Ok(w) = ws.take_2d::<B>(&format!("{prefix}.weight"), device) {
        // PyTorch Linear weight: [out, in] → burn needs [in, out]
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
