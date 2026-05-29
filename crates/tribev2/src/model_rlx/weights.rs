//! Safetensors → RLX parameter map for the TRIBE encoder graph.

use std::collections::HashMap;

use half::bf16;
use safetensors::SafeTensors;

use crate::model_rlx::graph::TribeSpec;

#[derive(Clone, Debug)]
pub struct ParamBuf {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

pub type ParamMap = HashMap<String, ParamBuf>;

pub fn load_safetensors(path: &str) -> anyhow::Result<HashMap<String, ParamBuf>> {
    let bytes = std::fs::read(path)?;
    let st = SafeTensors::deserialize(&bytes)?;
    let mut out = HashMap::with_capacity(st.len());
    for (raw_key, view) in st.tensors() {
        let key = raw_key
            .strip_prefix("model.")
            .unwrap_or(raw_key.as_str())
            .to_string();
        let shape: Vec<usize> = view.shape().to_vec();
        let data = match view.dtype() {
            safetensors::Dtype::BF16 => view
                .data()
                .chunks_exact(2)
                .map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect(),
            safetensors::Dtype::F16 => view
                .data()
                .chunks_exact(2)
                .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect(),
            safetensors::Dtype::F32 => view
                .data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            other => anyhow::bail!("unsupported dtype {:?} for key {}", other, key),
        };
        out.insert(key, ParamBuf { data, shape });
    }
    Ok(out)
}

fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; data.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

fn take_linear_w(raw: &mut HashMap<String, ParamBuf>, key: &str) -> anyhow::Result<ParamBuf> {
    let p = raw
        .remove(key)
        .ok_or_else(|| anyhow::anyhow!("missing weight key: {key}"))?;
    anyhow::ensure!(
        p.shape.len() == 2,
        "Linear weight {key} must be 2-D, got {:?}",
        p.shape
    );
    let (out_d, in_d) = (p.shape[0], p.shape[1]);
    Ok(ParamBuf {
        data: transpose(&p.data, out_d, in_d),
        shape: vec![in_d, out_d],
    })
}

fn take(raw: &mut HashMap<String, ParamBuf>, key: &str) -> anyhow::Result<ParamBuf> {
    raw.remove(key)
        .ok_or_else(|| anyhow::anyhow!("missing weight key: {key}"))
}

fn insert(params: &mut ParamMap, key: impl AsRef<str>, buf: ParamBuf) {
    params.insert(key.as_ref().to_string(), buf);
}

pub fn build_tribe_params(
    raw: &mut HashMap<String, ParamBuf>,
    _spec: &TribeSpec,
) -> anyhow::Result<ParamMap> {
    let hidden = 1152usize;
    let mut params = ParamMap::new();

    for name in ["text", "audio", "video"] {
        let wkey = format!("projectors.{name}.weight");
        let bkey = format!("projectors.{name}.bias");
        insert(&mut params, &wkey, take_linear_w(raw, &wkey)?);
        insert(&mut params, &bkey, take(raw, &bkey)?);
    }

    if let Ok(tpe) = take(raw, "time_pos_embed") {
        insert(&mut params, "time_pos_embed", tpe);
    }

    for layer in 0..16 {
        let prefix = format!("encoder.layers.{layer}");
        let gkey = format!("{prefix}.0.0.g");
        if let Ok(p) = take(raw, &gkey) {
            let g = p.data[0] * (hidden as f32).sqrt();
            insert(
                &mut params,
                format!("{prefix}.0.0.g_scale"),
                ParamBuf {
                    data: vec![g],
                    shape: vec![1],
                },
            );
        }
        if layer % 2 == 0 {
            for w in ["to_q", "to_k", "to_v", "to_out"] {
                let k = format!("{prefix}.1.{w}.weight");
                insert(&mut params, &k, take_linear_w(raw, &k)?);
            }
        } else {
            insert(
                &mut params,
                format!("{prefix}.1.ff.0.0.weight"),
                take_linear_w(raw, &format!("{prefix}.1.ff.0.0.weight"))?,
            );
            insert(
                &mut params,
                format!("{prefix}.1.ff.0.0.bias"),
                take(raw, &format!("{prefix}.1.ff.0.0.bias"))?,
            );
            insert(
                &mut params,
                format!("{prefix}.1.ff.2.weight"),
                take_linear_w(raw, &format!("{prefix}.1.ff.2.weight"))?,
            );
            insert(
                &mut params,
                format!("{prefix}.1.ff.2.bias"),
                take(raw, &format!("{prefix}.1.ff.2.bias"))?,
            );
        }
        let rskey = format!("{prefix}.2.residual_scale");
        if let Ok(rs) = take(raw, &rskey) {
            insert(&mut params, &rskey, rs);
        }
    }

    if let Ok(p) = take(raw, "encoder.final_norm.g") {
        let g = p.data[0] * (hidden as f32).sqrt();
        insert(
            &mut params,
            "encoder.final_norm.g_scale",
            ParamBuf {
                data: vec![g],
                shape: vec![1],
            },
        );
    }

    insert(
        &mut params,
        "low_rank_head.weight",
        take_linear_w(raw, "low_rank_head.weight")?,
    );

    // average_subjects: use dropout row (index 0 when n_subjects=0)
    let pw = take(raw, "predictor.weights")?;
    anyhow::ensure!(pw.shape.len() == 3, "predictor.weights must be 3-D");
    let (_ns, c, d) = (pw.shape[0], pw.shape[1], pw.shape[2]);
    let idx = 0usize;
    let mut w2d = vec![0f32; c * d];
    let off = idx * c * d;
    w2d.copy_from_slice(&pw.data[off..off + c * d]);
    insert(
        &mut params,
        "predictor.weights",
        ParamBuf {
            data: w2d,
            shape: vec![c, d],
        },
    );

    if let Ok(pb) = take(raw, "predictor.bias") {
        let d = pb.shape[pb.shape.len() - 1];
        let idx = 0usize;
        let off = idx * d;
        insert(
            &mut params,
            "predictor.bias",
            ParamBuf {
                data: pb.data[off..off + d].to_vec(),
                shape: vec![d],
            },
        );
    }

    let _ = raw.remove("encoder.rotary_pos_emb.inv_freq");
    Ok(params)
}

pub fn apply_params(compiled: &mut rlx::CompiledGraph, params: &ParamMap) {
    for (name, buf) in params {
        compiled.set_param(name, &buf.data);
    }
}
