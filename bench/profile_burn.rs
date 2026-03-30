//! Profile individual components of the TRIBE v2 burn model.
//!
//! Usage:
//!   cargo run --release --example profile_burn --features blas-accelerate
//!   cargo run --release --example profile_burn --no-default-features --features wgpu-metal,llama-metal

use std::time::Instant;
use burn::prelude::*;
use tribev2_rs::config::*;
use tribev2_rs::model_burn::*;

// ── Backend ───────────────────────────────────────────────────────────────
#[cfg(all(feature = "wgpu", not(feature = "ndarray")))]
mod backend {
    pub use burn::backend::Wgpu as B;
    pub fn device() -> burn::backend::wgpu::WgpuDevice {
        burn::backend::wgpu::WgpuDevice::DefaultDevice
    }
    pub const NAME: &str = "wgpu";
}

#[cfg(feature = "ndarray")]
mod backend {
    pub use burn::backend::NdArray as B;
    pub fn device() -> burn::backend::ndarray::NdArrayDevice {
        burn::backend::ndarray::NdArrayDevice::Cpu
    }
    #[cfg(feature = "blas-accelerate")]
    pub const NAME: &str = "NdArray + Accelerate";
    #[cfg(not(feature = "blas-accelerate"))]
    pub const NAME: &str = "NdArray";
}

use backend::{B, device};

fn sync(x: Tensor<B, 3>) -> Tensor<B, 3> {
    // Force GPU sync by reading one element
    let _ = x.clone().slice([0..1, 0..1, 0..1]).into_data();
    x
}

fn time_op(name: &str, iters: usize, mut f: impl FnMut()) -> f64 {
    // Warmup
    for _ in 0..2 { f(); }
    let t0 = Instant::now();
    for _ in 0..iters { f(); }
    let total = t0.elapsed().as_secs_f64() * 1000.0;
    let per = total / iters as f64;
    println!("  {name:<30} {per:>8.2} ms  (×{iters} = {total:.0}ms)");
    per
}

fn main() {
    let dev = device();
    println!("=== TRIBE v2 Component Profile [{}] ===\n", backend::NAME);

    let dim = 1152;
    let t_len = 100;
    let heads = 8;

    // ── Projectors ────────────────────────────────────────────────────
    println!("Projectors:");
    let proj_text = projector::Projector::<B>::new(6144, 384, &dev);
    let proj_audio = projector::Projector::<B>::new(2048, 384, &dev);
    let proj_video = projector::Projector::<B>::new(2816, 384, &dev);
    let txt_in = Tensor::<B,3>::ones([1, t_len, 6144], &dev).mul_scalar(0.01);
    let aud_in = Tensor::<B,3>::ones([1, t_len, 2048], &dev).mul_scalar(0.01);
    let vid_in = Tensor::<B,3>::ones([1, t_len, 2816], &dev).mul_scalar(0.01);

    time_op("text  [1,100,6144]→[1,100,384]", 20, || {
        let _ = sync(proj_text.forward(txt_in.clone()));
    });
    time_op("audio [1,100,2048]→[1,100,384]", 20, || {
        let _ = sync(proj_audio.forward(aud_in.clone()));
    });
    time_op("video [1,100,2816]→[1,100,384]", 20, || {
        let _ = sync(proj_video.forward(vid_in.clone()));
    });

    // ── ScaleNorm ─────────────────────────────────────────────────────
    println!("\nScaleNorm:");
    let sn = scalenorm::ScaleNorm::<B>::new(dim, &dev);
    let x = Tensor::<B,3>::ones([1, t_len, dim], &dev).mul_scalar(0.1);
    time_op("[1,100,1152]", 20, || {
        let _ = sync(sn.forward(x.clone()));
    });

    // ── Attention ─────────────────────────────────────────────────────
    println!("\nAttention (dim={dim}, heads={heads}):");
    let attn = attention::Attention::<B>::new(dim, heads, &dev);
    let rot_dim = 72;
    let half_rot = rot_dim / 2;
    let freqs = rotary::build_rotary_freqs::<B>(rot_dim, t_len, &dev);
    let f_half = freqs.clone().slice([0..t_len, 0..half_rot]);
    let rot_cos = f_half.clone().cos();
    let rot_sin = f_half.sin();
    time_op("[1,100,1152] + RoPE (precomp)", 5, || {
        let _ = sync(attn.forward(x.clone(), Some(&rot_cos), Some(&rot_sin)));
    });
    time_op("[1,100,1152] no RoPE", 5, || {
        let _ = sync(attn.forward(x.clone(), None, None));
    });

    // ── FeedForward ───────────────────────────────────────────────────
    println!("\nFeedForward (dim={dim}, mult=4):");
    let ff = feedforward::FeedForward::<B>::new(dim, 4, &dev);
    time_op("[1,100,1152]→[1,100,4608]→[1,100,1152]", 5, || {
        let _ = sync(ff.forward(x.clone()));
    });

    // ── Full encoder (8 layers) ───────────────────────────────────────
    println!("\nFull encoder (depth=8):");
    let enc_cfg = EncoderConfig { depth: 8, heads: 8, ff_mult: 4,
        use_scalenorm: true, rotary_pos_emb: true, scale_residual: true,
        ..Default::default() };
    let enc = encoder::XTransformerEncoder::<B>::new(dim, &enc_cfg, &dev);
    time_op("[1,100,1152] 8 layers", 3, || {
        let _ = sync(enc.forward(x.clone()));
    });

    // ── Low-rank head ─────────────────────────────────────────────────
    println!("\nLow-rank head:");
    let lr = burn::nn::LinearConfig::new(dim, 2048).with_bias(false).init::<B>(&dev);
    let x_ht = Tensor::<B,3>::ones([1, t_len, dim], &dev).mul_scalar(0.01);
    time_op("[1,100,1152]→[1,100,2048]", 20, || {
        let _ = sync(lr.forward(x_ht.clone()));
    });

    // ── SubjectLayers ─────────────────────────────────────────────────
    println!("\nSubjectLayers (avg mode, 1 subject):");
    let slc = SubjectLayersConfig { n_subjects: 0, bias: true,
        subject_dropout: Some(0.1), average_subjects: true, ..Default::default() };
    let sl = subject_layers::SubjectLayers::<B>::new(2048, 20484, &slc, &dev);
    let sx = Tensor::<B,3>::ones([1, 2048, t_len], &dev).mul_scalar(0.01);
    time_op("[1,2048,100]→[1,20484,100]", 3, || {
        let _ = sync(sl.forward_average(sx.clone()));
    });

    // ── AdaptiveAvgPool1d ─────────────────────────────────────────────
    println!("\nAdaptiveAvgPool1d:");
    let big = Tensor::<B,3>::ones([1, 20484, t_len], &dev);
    time_op("[1,20484,100]→[1,20484,100]", 20, || {
        let _ = sync(tribe::adaptive_avg_pool1d_pub::<B>(big.clone(), 100));
    });

    // ── Full forward ──────────────────────────────────────────────────
    println!("\nFull forward pass:");
    let feature_dims = vec![
        ModalityDims::new("text", 2, 3072),
        ModalityDims::new("audio", 2, 1024),
        ModalityDims::new("video", 2, 1408),
    ];
    let config = BrainModelConfig {
        hidden: 1152, max_seq_len: 1024,
        extractor_aggregation: "cat".into(), layer_aggregation: "cat".into(),
        linear_baseline: false, time_pos_embedding: true, subject_embedding: false,
        dropout: 0.0, modality_dropout: 0.0, temporal_dropout: 0.0,
        low_rank_head: Some(2048), combiner: None, temporal_smoothing: None,
        projector: Default::default(),
        encoder: Some(EncoderConfig { heads: 8, depth: 8, ff_mult: 4,
            use_scalenorm: true, rotary_pos_emb: true, scale_residual: true,
            ..Default::default() }),
        subject_layers: Some(SubjectLayersConfig {
            n_subjects: 0, bias: true, subject_dropout: Some(0.1),
            average_subjects: true, ..Default::default() }),
    };
    let model = tribe::TribeV2Burn::<B>::new(&feature_dims, 20484, 100, &config, &dev);
    let text = Tensor::<B,3>::ones([1, 6144, 100], &dev).mul_scalar(0.01);
    let audio = Tensor::<B,3>::ones([1, 2048, 100], &dev).mul_scalar(0.01);
    let video = Tensor::<B,3>::ones([1, 2816, 100], &dev).mul_scalar(0.01);

    time_op("complete [1,20484,100]", 5, || {
        let feats = vec![
            ("text", text.clone()), ("audio", audio.clone()), ("video", video.clone()),
        ];
        let out = model.forward(feats);
        let _ = out.into_data();
    });
}
