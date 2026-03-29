//! Burn-based benchmark for TRIBE v2 forward pass.
//! Supports NdArray (CPU), NdArray+Accelerate, and wgpu backends.
//!
//! cargo run --release --example bench_burn                             # ndarray CPU
//! cargo run --release --example bench_burn --features blas-accelerate  # ndarray + Accelerate
//! cargo run --release --example bench_burn --no-default-features --features wgpu           # wgpu (WGSL)
//! cargo run --release --example bench_burn --no-default-features --features wgpu-metal     # wgpu Metal

use std::time::Instant;
use tribev2_rs::config::*;
use tribev2_rs::model_burn::tribe::TribeV2Burn;

#[cfg(feature = "ndarray")]
mod backend {
    pub use burn::backend::NdArray as B;
    pub fn device() -> burn::backend::ndarray::NdArrayDevice {
        burn::backend::ndarray::NdArrayDevice::Cpu
    }
    #[cfg(feature = "blas-accelerate")]
    pub const NAME: &str = "NdArray + Accelerate";
    #[cfg(not(feature = "blas-accelerate"))]
    pub const NAME: &str = "NdArray CPU";
}

#[cfg(all(feature = "wgpu", not(feature = "ndarray")))]
mod backend {
    pub use burn::backend::Wgpu as B;
    pub fn device() -> burn::backend::wgpu::WgpuDevice {
        burn::backend::wgpu::WgpuDevice::DefaultDevice
    }
    #[cfg(feature = "wgpu-metal")]
    pub const NAME: &str = "wgpu (Metal)";
    #[cfg(feature = "wgpu-vulkan")]
    pub const NAME: &str = "wgpu (Vulkan)";
    #[cfg(not(any(feature = "wgpu-metal", feature = "wgpu-vulkan")))]
    pub const NAME: &str = "wgpu (WGSL)";
}

use backend::{B, device};

fn main() {
    let dev = device();
    println!("=== Rust Burn [{}] ===", backend::NAME);

    let config = BrainModelConfig {
        hidden: 1152,
        max_seq_len: 1024,
        extractor_aggregation: "cat".into(),
        layer_aggregation: "cat".into(),
        linear_baseline: false,
        time_pos_embedding: true,
        subject_embedding: false,
        dropout: 0.0,
        modality_dropout: 0.0,
        temporal_dropout: 0.0,
        low_rank_head: Some(2048),
        combiner: None,
        temporal_smoothing: None,
        projector: Default::default(),
        encoder: Some(EncoderConfig {
            heads: 8, depth: 8, ff_mult: 4,
            use_scalenorm: true, rotary_pos_emb: true, scale_residual: true,
            ..Default::default()
        }),
        subject_layers: Some(SubjectLayersConfig {
            n_subjects: 25, bias: true,
            subject_dropout: Some(0.1), average_subjects: true,
            ..Default::default()
        }),
    };

    let feature_dims = ModalityDims::pretrained();

    // For wgpu: SubjectLayers [26, 2048, 20484] is ~4GB, too large for single GPU buffer.
    let n_outputs = if cfg!(all(feature = "wgpu", not(feature = "ndarray"))) { 2048 } else { 20484 };
    let model = TribeV2Burn::<B>::new(&feature_dims, n_outputs, 100, &config, &dev);

    // Also run a reduced-output version for fair wgpu comparison
    let run_reduced = cfg!(feature = "ndarray") && n_outputs > 2048;

    let t = 100;
    use burn::prelude::*;
    let text  = Tensor::<B, 3>::ones([1, 9216, t], &dev).mul_scalar(0.01);
    let audio = Tensor::<B, 3>::ones([1, 3072, t], &dev).mul_scalar(0.01);
    let video = Tensor::<B, 3>::ones([1, 4224, t], &dev).mul_scalar(0.01);

    let n_warmup = 3;
    let n_runs = 5;

    // Warmup
    for _ in 0..n_warmup {
        let feats = vec![
            ("text", text.clone()), ("audio", audio.clone()), ("video", video.clone()),
        ];
        let _ = model.forward(feats);
    }

    // Timed
    let mut times = Vec::with_capacity(n_runs);
    for _ in 0..n_runs {
        let feats = vec![
            ("text", text.clone()), ("audio", audio.clone()), ("video", video.clone()),
        ];
        let t0 = Instant::now();
        let out = model.forward(feats);
        // Force sync for GPU backends
        let _ = out.into_data();
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        times.push(ms);
    }

    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let std = (times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / times.len() as f64).sqrt();

    println!("  Mean: {mean:.1} ms, Min: {min:.1} ms, Max: {max:.1} ms, Std: {std:.1} ms");

    let key = if cfg!(all(feature = "wgpu", not(feature = "ndarray"))) {
        if cfg!(feature = "wgpu-metal") { "rust_burn_wgpu_metal" }
        else { "rust_burn_wgpu" }
    } else if cfg!(feature = "blas-accelerate") {
        "rust_burn_ndarray_accelerate"
    } else {
        "rust_burn_ndarray"
    };

    let json = format!(
        r#"{{"{key}":{{"mean_ms":{mean:.1},"min_ms":{min:.1},"max_ms":{max:.1},"std_ms":{std:.1},"n_runs":{n_runs},"output_shape":[1,20484,100]}}}}"#
    );
    let path = format!("bench/results_{key}.json");
    std::fs::write(&path, &json).unwrap();
    println!("Results saved to {path}");
}
