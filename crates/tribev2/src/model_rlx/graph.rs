//! TRIBE v2 FmriEncoderModel as an RLX graph.

use rlx::ir::shape;
use rlx::ir::GraphExt;
use rlx::ops::MaskKind;
use rlx::prelude::*;

/// Fixed-shape spec for one `(batch, timesteps)` compile.
#[derive(Clone, Copy, Debug)]
pub struct TribeSpec {
    pub b: usize,
    pub t: usize,
    pub text_in: usize,
    pub audio_in: usize,
    pub video_in: usize,
    pub hidden: usize,
    pub proj_out: usize,
    pub n_heads: usize,
    pub dim_head: usize,
    pub rot_dim: usize,
    pub ff_inner: usize,
    pub lr_dim: usize,
    pub n_outputs: usize,
    pub depth: usize,
}

impl TribeSpec {
    /// Build spec from modality layer counts (e.g. text: 2 layers × 3072 dim → 6144).
    pub fn new(
        b: usize,
        t: usize,
        text_in: usize,
        audio_in: usize,
        video_in: usize,
        n_outputs: usize,
    ) -> Self {
        let hidden = 1152;
        let n_modalities = 3;
        Self {
            b,
            t,
            text_in,
            audio_in,
            video_in,
            hidden,
            proj_out: hidden / n_modalities,
            n_heads: 8,
            dim_head: hidden / 8,
            rot_dim: (hidden / 8 / 2).max(32),
            ff_inner: hidden * 4,
            lr_dim: 2048,
            n_outputs,
            depth: 8,
        }
    }

    pub fn with_depth(mut self, depth: usize) -> Self {
        self.depth = depth;
        self
    }
}

fn s1(d: usize) -> Shape {
    Shape::new(&[d], DType::F32)
}
fn s2(a: usize, b: usize) -> Shape {
    Shape::new(&[a, b], DType::F32)
}
fn s3(a: usize, b: usize, c: usize) -> Shape {
    Shape::new(&[a, b, c], DType::F32)
}
fn s4(a: usize, b: usize, c: usize, d: usize) -> Shape {
    Shape::new(&[a, b, c, d], DType::F32)
}

/// ScaleNorm: L2-normalize over last axis, multiply by learned scale `g` (already `g * sqrt(dim)`).
fn scale_norm(g: &mut Graph, x: NodeId, g_param: NodeId, norm_axis: usize) -> NodeId {
    let sq = g.mul(x, x);
    let norm_sq = g.sum(sq, vec![norm_axis], true);
    // Match pure-Rust `Tensor::scale_norm` / PyTorch `F.normalize` eps.
    let eps_data = 1e-12f32.to_le_bytes().to_vec();
    let eps = g.append_node(Op::Constant { data: eps_data }, vec![], s1(1), None);
    let norm_sq_safe = g.add(norm_sq, eps);
    let norm = g.sqrt(norm_sq_safe);
    let normed = g.div(x, norm);
    g.mul(normed, g_param)
}

fn linear_bias(g: &mut Graph, x: NodeId, w: NodeId, b: NodeId) -> NodeId {
    let y = g.mm(x, w);
    g.add(y, b)
}

fn gelu_ffn(
    g: &mut Graph,
    x: NodeId,
    w1: NodeId,
    b1: NodeId,
    w2: NodeId,
    b2: NodeId,
) -> NodeId {
    let mm1 = g.mm(x, w1);
    let a = g.add(mm1, b1);
    let h = g.gelu(a);
    let mm2 = g.mm(h, w2);
    g.add(mm2, b2)
}

/// RoPE `rotate_half` matching pure-Rust `apply_rotary_pos_emb` (split first/second half).
fn rotate_half(
    g: &mut Graph,
    x: NodeId,
    cos: NodeId,
    sin: NodeId,
    _b: usize,
    _h: usize,
    _s: usize,
    d: usize,
) -> NodeId {
    let half = d / 2;
    let x0 = g.narrow_(x, 3, 0, half);
    let x1 = g.narrow_(x, 3, half, half);
    let ec = g.mul(x0, cos);
    let os = g.mul(x1, sin);
    let out0 = g.sub(ec, os);
    let es = g.mul(x0, sin);
    let oc = g.mul(x1, cos);
    let out1 = g.add(es, oc);
    g.concat_(vec![out0, out1], 3)
}

fn self_attention(
    g: &mut Graph,
    x: NodeId,
    wq: NodeId,
    wk: NodeId,
    wv: NodeId,
    wo: NodeId,
    cos: NodeId,
    sin: NodeId,
    spec: &TribeSpec,
) -> NodeId {
    let b = spec.b;
    let s = spec.t;
    let d = spec.hidden;
    let nh = spec.n_heads;
    let dh = spec.dim_head;
    let h_total = nh * dh;

    let q = g.mm(x, wq);
    let k = g.mm(x, wk);
    let v = g.mm(x, wv);

    // Match pure-Rust / Burn layout: [B, S, D] → [B, H, S, D_head].
    let q4 = g.reshape_(q, vec![b as i64, s as i64, nh as i64, dh as i64]);
    let k4 = g.reshape_(k, vec![b as i64, s as i64, nh as i64, dh as i64]);
    let v4 = g.reshape_(v, vec![b as i64, s as i64, nh as i64, dh as i64]);
    let q_bhs = g.transpose_(q4, vec![0, 2, 1, 3]);
    let k_bhs = g.transpose_(k4, vec![0, 2, 1, 3]);
    let v_bhs = g.transpose_(v4, vec![0, 2, 1, 3]);

    let rot_d = spec.rot_dim;
    let q_rot_part = g.narrow_(q_bhs, 3, 0, rot_d);
    let q_pass = g.narrow_(q_bhs, 3, rot_d, dh - rot_d);
    let q_rot = rotate_half(g, q_rot_part, cos, sin, b, nh, s, rot_d);
    let q_full = g.concat_(vec![q_rot, q_pass], 3);

    let k_rot_part = g.narrow_(k_bhs, 3, 0, rot_d);
    let k_pass = g.narrow_(k_bhs, 3, rot_d, dh - rot_d);
    let k_rot = rotate_half(g, k_rot_part, cos, sin, b, nh, s, rot_d);
    let k_full = g.concat_(vec![k_rot, k_pass], 3);

    let attn_shape = shape::attention_shape(g.shape(q_full));
    let attn = g.attention_kind(q_full, k_full, v_bhs, nh, dh, MaskKind::None, attn_shape);
    let attn_bsh = g.transpose_(attn, vec![0, 2, 1, 3]);
    let attn_3 = g.reshape_(attn_bsh, vec![b as i64, s as i64, h_total as i64]);
    g.mm(attn_3, wo)
}

fn attn_layer(g: &mut Graph, x: NodeId, layer: usize, cos: NodeId, sin: NodeId, spec: &TribeSpec) -> NodeId {
    let d = spec.hidden;
    let p = format!("encoder.layers.{layer}");
    let residual = x;
    let g_param = g.param(format!("{p}.0.0.g_scale"), s1(1));
    let x_norm = scale_norm(g, x, g_param, 2);

    let wq = g.param(format!("{p}.1.to_q.weight"), s2(d, d));
    let wk = g.param(format!("{p}.1.to_k.weight"), s2(d, d));
    let wv = g.param(format!("{p}.1.to_v.weight"), s2(d, d));
    let wo = g.param(format!("{p}.1.to_out.weight"), s2(d, d));
    let branch = self_attention(g, x_norm, wq, wk, wv, wo, cos, sin, spec);

    let rs = g.param(format!("{p}.2.residual_scale"), s1(d));
    let rs3 = g.reshape_(rs, vec![1, 1, d as i64]);
    let scaled_res = g.mul(residual, rs3);
    g.add(branch, scaled_res)
}

fn ff_layer(g: &mut Graph, x: NodeId, layer: usize, spec: &TribeSpec) -> NodeId {
    let d = spec.hidden;
    let inner = spec.ff_inner;
    let p = format!("encoder.layers.{layer}");
    let residual = x;
    let g_param = g.param(format!("{p}.0.0.g_scale"), s1(1));
    let x_norm = scale_norm(g, x, g_param, 2);

    let w1 = g.param(format!("{p}.1.ff.0.0.weight"), s2(d, inner));
    let b1 = g.param(format!("{p}.1.ff.0.0.bias"), s1(inner));
    let w2 = g.param(format!("{p}.1.ff.2.weight"), s2(inner, d));
    let b2 = g.param(format!("{p}.1.ff.2.bias"), s1(d));
    let branch = gelu_ffn(g, x_norm, w1, b1, w2, b2);

    let rs = g.param(format!("{p}.2.residual_scale"), s1(d));
    let rs3 = g.reshape_(rs, vec![1, 1, d as i64]);
    let scaled_res = g.mul(residual, rs3);
    g.add(branch, scaled_res)
}

/// Modality input [B, D_in, T] → projected [B, T, proj_out].
fn projector_branch(
    g: &mut Graph,
    feat: NodeId,
    name: &str,
    in_dim: usize,
    spec: &TribeSpec,
) -> NodeId {
    let b = spec.b;
    let t = spec.t;
    let out = spec.proj_out;
    // [B, D, T] → [B, T, D]
    let x = g.transpose_(feat, vec![0, 2, 1]);
    let x2 = g.reshape_(x, vec![(b * t) as i64, in_dim as i64]);
    let w = g.param(format!("projectors.{name}.weight"), s2(in_dim, out));
    let bias = g.param(format!("projectors.{name}.bias"), s1(out));
    let y = linear_bias(g, x2, w, bias);
    g.reshape_(y, vec![b as i64, t as i64, out as i64])
}

/// Projectors + concat + time embedding only (for parity debugging).
pub fn build_tribe_cat_graph(spec: &TribeSpec, mod_order: &[String]) -> Graph {
    let mut g = Graph::new("tribe_v2_cat");
    let t = spec.t;
    let h = spec.hidden;

    let x = concat_modalities(&mut g, spec, mod_order);
    g.set_outputs(vec![x]);
    g
}

fn modality_input(
    g: &mut Graph,
    name: &str,
    spec: &TribeSpec,
) -> (NodeId, usize) {
    let b = spec.b;
    let t = spec.t;
    let (node, in_dim) = match name {
        "text" => (g.input("text", s3(b, spec.text_in, t)), spec.text_in),
        "audio" => (g.input("audio", s3(b, spec.audio_in, t)), spec.audio_in),
        "video" => (g.input("video", s3(b, spec.video_in, t)), spec.video_in),
        other => panic!("unknown modality {other}"),
    };
    (node, in_dim)
}

fn concat_modalities(g: &mut Graph, spec: &TribeSpec, mod_order: &[String]) -> NodeId {
    let branches: Vec<NodeId> = mod_order
        .iter()
        .map(|name| {
            let (feat, in_dim) = modality_input(g, name, spec);
            projector_branch(g, feat, name, in_dim, spec)
        })
        .collect();
    g.concat_(branches, 2)
}

fn tribe_encoder_prologue(g: &mut Graph, spec: &TribeSpec, mod_order: &[String]) -> (NodeId, NodeId, NodeId) {
    let t = spec.t;
    let h = spec.hidden;
    let cos_in = g.input("rope_cos", s4(1, 1, t, spec.rot_dim / 2));
    let sin_in = g.input("rope_sin", s4(1, 1, t, spec.rot_dim / 2));
    let mut x = concat_modalities(g, spec, mod_order);
    let tpe = g.param("time_pos_embed", s3(1, 1024, h));
    let tpe_slice = g.narrow_(tpe, 1, 0, t);
    let tpe_b = g.reshape_(tpe_slice, vec![1, t as i64, h as i64]);
    x = g.add(x, tpe_b);
    (x, cos_in, sin_in)
}

/// Modalities → concat → time embed → encoder (output `[B, T, H]`).
pub fn build_tribe_encoder_graph(spec: &TribeSpec, mod_order: &[String]) -> Graph {
    let mut g = Graph::new("tribe_v2_encoder");
    let (mut x, cos_in, sin_in) = tribe_encoder_prologue(&mut g, spec, mod_order);
    for i in 0..(spec.depth * 2) {
        if i % 2 == 0 {
            x = attn_layer(&mut g, x, i, cos_in, sin_in, spec);
        } else {
            x = ff_layer(&mut g, x, i, spec);
        }
    }
    let fn_g = g.param("encoder.final_norm.g_scale", s1(1));
    x = scale_norm(&mut g, x, fn_g, 2);
    g.set_outputs(vec![x]);
    g
}

/// Encoder through the first `n_attn` attention blocks only (no FF, no final norm).
pub fn build_tribe_encoder_attn_only_graph(spec: &TribeSpec, mod_order: &[String], n_attn: usize) -> Graph {
    let mut g = Graph::new("tribe_v2_encoder_attn_only");
    let (mut x, cos_in, sin_in) = tribe_encoder_prologue(&mut g, spec, mod_order);
    for layer in 0..n_attn {
        x = attn_layer(&mut g, x, layer * 2, cos_in, sin_in, spec);
    }
    g.set_outputs(vec![x]);
    g
}

/// Apply encoder `final_norm` to a `[B, T, H]` tensor.
pub fn build_tribe_final_norm_graph(spec: &TribeSpec) -> Graph {
    let mut g = Graph::new("tribe_v2_final_norm");
    let x = g.input("x", s3(spec.b, spec.t, spec.hidden));
    let fn_g = g.param("encoder.final_norm.g_scale", s1(1));
    let y = scale_norm(&mut g, x, fn_g, 2);
    g.set_outputs(vec![y]);
    g
}

/// First attention block + first FF block (encoder layers 0–1).
pub fn build_tribe_encoder_first_block_graph(spec: &TribeSpec, mod_order: &[String]) -> Graph {
    let mut g = Graph::new("tribe_v2_encoder_block0");
    let (mut x, cos_in, sin_in) = tribe_encoder_prologue(&mut g, spec, mod_order);
    x = attn_layer(&mut g, x, 0, cos_in, sin_in, spec);
    x = ff_layer(&mut g, x, 1, spec);
    g.set_outputs(vec![x]);
    g
}

/// Through low-rank head only (output `[B, lr_dim, T]`).
pub fn build_tribe_lowrank_graph(spec: &TribeSpec, mod_order: &[String]) -> Graph {
    let mut g = Graph::new("tribe_v2_lowrank");
    let b = spec.b;
    let t = spec.t;
    let h = spec.hidden;
    let cos_in = g.input("rope_cos", s4(1, 1, t, spec.rot_dim / 2));
    let sin_in = g.input("rope_sin", s4(1, 1, t, spec.rot_dim / 2));
    let mut x = concat_modalities(&mut g, spec, mod_order);
    let tpe = g.param("time_pos_embed", s3(1, 1024, h));
    let tpe_slice = g.narrow_(tpe, 1, 0, t);
    let tpe_b = g.reshape_(tpe_slice, vec![1, t as i64, h as i64]);
    x = g.add(x, tpe_b);
    for i in 0..(spec.depth * 2) {
        if i % 2 == 0 {
            x = attn_layer(&mut g, x, i, cos_in, sin_in, spec);
        } else {
            x = ff_layer(&mut g, x, i, spec);
        }
    }
    let fn_g = g.param("encoder.final_norm.g_scale", s1(1));
    x = scale_norm(&mut g, x, fn_g, 2);
    x = g.transpose_(x, vec![0, 2, 1]);
    let x_bt = g.transpose_(x, vec![0, 2, 1]);
    let x2 = g.reshape_(x_bt, vec![(b * t) as i64, h as i64]);
    let lr_w = g.param("low_rank_head.weight", s2(h, spec.lr_dim));
    let x_lr = g.mm(x2, lr_w);
    let x_lr3 = g.reshape_(x_lr, vec![b as i64, t as i64, spec.lr_dim as i64]);
    let out = g.transpose_(x_lr3, vec![0, 2, 1]);
    g.set_outputs(vec![out]);
    g
}

/// Low-rank + predictor from encoder output `[B, T, H]` → `[B, n_outputs, T]`.
pub fn build_tribe_tail_graph(spec: &TribeSpec) -> Graph {
    let mut g = Graph::new("tribe_v2_tail");
    let b = spec.b;
    let t = spec.t;
    let h = spec.hidden;
    let mut x = g.input("enc", s3(b, t, h));
    x = g.transpose_(x, vec![0, 2, 1]);
    let x_bt = g.transpose_(x, vec![0, 2, 1]);
    let x2 = g.reshape_(x_bt, vec![(b * t) as i64, h as i64]);
    let lr_w = g.param("low_rank_head.weight", s2(h, spec.lr_dim));
    let x_lr = g.mm(x2, lr_w);
    let x_lr3 = g.reshape_(x_lr, vec![b as i64, t as i64, spec.lr_dim as i64]);
    x = g.transpose_(x_lr3, vec![0, 2, 1]);
    let c = spec.lr_dim;
    let d_out = spec.n_outputs;
    let x_btc = g.transpose_(x, vec![0, 2, 1]);
    let x_flat = g.reshape_(x_btc, vec![(b * t) as i64, c as i64]);
    let pred_w = g.param("predictor.weights", s2(c, d_out));
    let pred_b = g.param("predictor.bias", s1(d_out));
    let y = linear_bias(&mut g, x_flat, pred_w, pred_b);
    let y_btd = g.reshape_(y, vec![b as i64, t as i64, d_out as i64]);
    let out = g.transpose_(y_btd, vec![0, 2, 1]);
    g.set_outputs(vec![out]);
    g
}

/// Predictor head only; input `x` is `[B, lr_dim, T]`.
pub fn build_tribe_predictor_graph(spec: &TribeSpec) -> Graph {
    let mut g = Graph::new("tribe_v2_predictor");
    let b = spec.b;
    let t = spec.t;
    let c = spec.lr_dim;
    let d_out = spec.n_outputs;
    let x = g.input("x", s3(b, c, t));
    let x_btc = g.transpose_(x, vec![0, 2, 1]);
    let x_flat = g.reshape_(x_btc, vec![(b * t) as i64, c as i64]);
    let pred_w = g.param("predictor.weights", s2(c, d_out));
    let pred_b = g.param("predictor.bias", s1(d_out));
    let y = linear_bias(&mut g, x_flat, pred_w, pred_b);
    let y_btd = g.reshape_(y, vec![b as i64, t as i64, d_out as i64]);
    let out = g.transpose_(y_btd, vec![0, 2, 1]);
    g.set_outputs(vec![out]);
    g
}

/// Build the full encoder graph for one shape.
pub fn build_tribe_graph(spec: &TribeSpec, mod_order: &[String]) -> Graph {
    let mut g = Graph::new("tribe_v2");
    let b = spec.b;
    let t = spec.t;
    let h = spec.hidden;

    // RoPE tables shaped for [B, H, S, D] attention (sequence on axis 2).
    let cos_in = g.input("rope_cos", s4(1, 1, t, spec.rot_dim / 2));
    let sin_in = g.input("rope_sin", s4(1, 1, t, spec.rot_dim / 2));

    let mut x = concat_modalities(&mut g, spec, mod_order);

    let tpe = g.param("time_pos_embed", s3(1, 1024, h));
    let tpe_slice = g.narrow_(tpe, 1, 0, t);
    let tpe_b = g.reshape_(tpe_slice, vec![1, t as i64, h as i64]);
    x = g.add(x, tpe_b);

    for i in 0..(spec.depth * 2) {
        if i % 2 == 0 {
            x = attn_layer(&mut g, x, i, cos_in, sin_in, spec);
        } else {
            x = ff_layer(&mut g, x, i, spec);
        }
    }

    let fn_g = g.param("encoder.final_norm.g_scale", s1(1));
    x = scale_norm(&mut g, x, fn_g, 2);

    // [B, T, H] → [B, H, T]
    x = g.transpose_(x, vec![0, 2, 1]);

    // low_rank_head: [B, H, T] → [B, T, H] → mm → [B, T, lr] → [B, lr, T]
    let x_bt = g.transpose_(x, vec![0, 2, 1]);
    let x2 = g.reshape_(x_bt, vec![(b * t) as i64, h as i64]);
    let lr_w = g.param("low_rank_head.weight", s2(h, spec.lr_dim));
    let x_lr = g.mm(x2, lr_w);
    let x_lr3 = g.reshape_(x_lr, vec![b as i64, t as i64, spec.lr_dim as i64]);
    x = g.transpose_(x_lr3, vec![0, 2, 1]);

    // predictor: x is [B, C, T]; matmul per (b,t) over C.
    let c = spec.lr_dim;
    let d_out = spec.n_outputs;
    let x_btc = g.transpose_(x, vec![0, 2, 1]);
    let x_flat = g.reshape_(x_btc, vec![(b * t) as i64, c as i64]);
    let pred_w = g.param("predictor.weights", s2(c, d_out));
    let pred_b = g.param("predictor.bias", s1(d_out));
    let y = linear_bias(&mut g, x_flat, pred_w, pred_b);
    let y_btd = g.reshape_(y, vec![b as i64, t as i64, d_out as i64]);
    let out = g.transpose_(y_btd, vec![0, 2, 1]);

    g.set_outputs(vec![out]);
    g
}
