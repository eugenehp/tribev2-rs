//! Burn-based benchmark for TRIBE v2 forward pass.
//!
//! CPU:
//!   cargo run --release --example bench_burn
//!   cargo run --release --example bench_burn --features blas-accelerate
//!
//! GPU (macOS Metal, f32):
//!   cargo run --release --example bench_burn --no-default-features --features wgpu-metal,llama-metal
//!
//! GPU (macOS Metal, f16 — recommended, uses Metal WMMA for ~2x faster matmuls):
//!   cargo run --release --example bench_burn --no-default-features --features wgpu-metal-f16,llama-metal
//!
//! GPU (macOS Metal, fused CubeCL kernels — fused RoPE + ScaleNorm, no fusion wrapper):
//!   cargo run --release --example bench_burn --no-default-features --features wgpu-kernels-metal,llama-metal
//!
//! GPU (Linux/Windows Vulkan):
//!   cargo run --release --example bench_burn --no-default-features --features wgpu-vulkan
//!
//! GPU (generic wgpu — auto-detects Metal/Vulkan/DX12):
//!   cargo run --release --example bench_burn --no-default-features --features wgpu

use std::time::Instant;
use tribev2::config::*;
use tribev2::model_burn::tribe::TribeV2Burn;

// ── Backend dispatch (compile-time) ───────────────────────────────────────

#[cfg(all(feature = "wgpu", not(feature = "ndarray")))]  
mod backend {
    // wgpu-kernels-metal: no-fusion CubeBackend + fused RoPE/ScaleNorm kernels.
    // Uses CubeBackend<WgpuRuntime, f32> directly (no Fusion wrapper),
    // so custom #[cube] kernels can access the underlying CubeTensor.
    #[cfg(feature = "wgpu-kernels-metal")]
    pub type B = burn::backend::wgpu::CubeBackend<
        cubecl::wgpu::WgpuRuntime, f32, i32, u32
    >;

    // f16 path: Metal WMMA gives 2x matmul throughput over f32.
    #[cfg(all(feature = "wgpu-metal-f16", not(feature = "wgpu-kernels-metal")))]
    pub type B = burn::backend::Wgpu<half::f16, i32>;

    // Default f32 fusion path.
    #[cfg(not(any(feature = "wgpu-metal-f16", feature = "wgpu-kernels-metal")))]
    pub use burn::backend::Wgpu as B;

    pub fn device() -> burn::backend::wgpu::WgpuDevice {
        burn::backend::wgpu::WgpuDevice::DefaultDevice
    }

    #[cfg(feature = "wgpu-kernels-metal")]
    pub const NAME: &str = "wgpu (Metal / MSL, f32, fused CubeCL kernels)";
    #[cfg(all(feature = "wgpu-metal", feature = "wgpu-metal-f16", not(feature = "wgpu-kernels-metal")))]
    pub const NAME: &str = "wgpu (Metal / MSL, f16)";
    #[cfg(all(feature = "wgpu-metal", not(feature = "wgpu-metal-f16"), not(feature = "wgpu-kernels-metal")))]
    pub const NAME: &str = "wgpu (Metal / MSL, f32)";
    #[cfg(all(feature = "wgpu-vulkan", not(feature = "wgpu-kernels-metal")))]
    pub const NAME: &str = "wgpu (Vulkan / SPIR-V)";
    #[cfg(not(any(feature = "wgpu-metal", feature = "wgpu-vulkan", feature = "wgpu-kernels-metal")))]
    pub const NAME: &str = "wgpu (WGSL — auto backend)";

    pub const KEY: &str = if cfg!(feature = "wgpu-kernels-metal") { "rust_burn_wgpu_metal_kernels" }
        else if cfg!(feature = "wgpu-metal-f16") { "rust_burn_wgpu_metal_f16" }
        else if cfg!(feature = "wgpu-metal")     { "rust_burn_wgpu_metal" }
        else if cfg!(feature = "wgpu-vulkan")    { "rust_burn_wgpu_vulkan" }
        else                                     { "rust_burn_wgpu" };
}

#[cfg(feature = "ndarray")]
mod backend {
    pub use burn::backend::NdArray as B;
    pub type Device = burn::backend::ndarray::NdArrayDevice;
    pub fn device() -> Device { Device::Cpu }
    #[cfg(feature = "blas-accelerate")]
    pub const NAME: &str = "NdArray + Apple Accelerate";
    #[cfg(not(feature = "blas-accelerate"))]
    pub const NAME: &str = "NdArray CPU (Rayon)";
    pub const KEY: &str = if cfg!(feature = "blas-accelerate") { "rust_burn_ndarray_accelerate" }
        else { "rust_burn_ndarray" };
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
            n_subjects: 0, bias: true,
            subject_dropout: Some(0.1), average_subjects: true,
            ..Default::default()
        }),
    };

    // Use actual pretrained feature dims
    let feature_dims = vec![
        ModalityDims::new("text", 2, 3072),
        ModalityDims::new("audio", 2, 1024),
        ModalityDims::new("video", 2, 1408),
    ];

    // For wgpu: SubjectLayers [1, 2048, 20484] may be large; use full size
    // since the pretrained model has only 1 subject (averaged)
    let n_outputs = 20484;
    let n_output_timesteps = 100;

    eprintln!("Building model...");
    let t_build = Instant::now();
    let model = TribeV2Burn::<B>::new(&feature_dims, n_outputs, n_output_timesteps, &config, &dev);
    eprintln!("  Built in {:.0}ms", t_build.elapsed().as_millis());

    let t = 100;
    use burn::prelude::*;
    let text  = Tensor::<B, 3>::ones([1, 6144, t], &dev).mul_scalar(0.01);
    let audio = Tensor::<B, 3>::ones([1, 2048, t], &dev).mul_scalar(0.01);
    let video = Tensor::<B, 3>::ones([1, 2816, t], &dev).mul_scalar(0.01);

    let n_warmup = 5;
    let n_runs = 10;

    // Warmup
    eprintln!("Warmup ({n_warmup} runs)...");
    for _ in 0..n_warmup {
        let feats = vec![
            ("text", text.clone()), ("audio", audio.clone()), ("video", video.clone()),
        ];
        let out = model.forward(feats);
        let _ = out.into_data(); // force sync
    }

    // Timed
    eprintln!("Benchmarking ({n_runs} runs)...");
    let mut times = Vec::with_capacity(n_runs);
    for i in 0..n_runs {
        let feats = vec![
            ("text", text.clone()), ("audio", audio.clone()), ("video", video.clone()),
        ];
        let t0 = Instant::now();
        let out = model.forward(feats);
        let _ = out.into_data(); // force GPU sync
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        times.push(ms);
        eprintln!("  Run {}: {ms:.1} ms", i + 1);
    }

    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let std = (times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / times.len() as f64).sqrt();

    println!("\n  Mean: {mean:.1} ms, Min: {min:.1} ms, Max: {max:.1} ms, Std: {std:.1} ms");
    println!("  Backend: {}", backend::NAME);
    println!("  Output: [1, {n_outputs}, {n_output_timesteps}]");

    let key = backend::KEY;
    let json = format!(
        r#"{{"{key}":{{"mean_ms":{mean:.1},"min_ms":{min:.1},"max_ms":{max:.1},"std_ms":{std:.1},"n_runs":{n_runs},"output_shape":[1,{n_outputs},{n_output_timesteps}],"backend":"{}"}}}}
"#,
        backend::NAME,
    );
    let path = format!("bench/results_{key}.json");
    std::fs::write(&path, &json).unwrap();
    println!("Results saved to {path}");
}
