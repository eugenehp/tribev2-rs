//! Burn backend parity test: verify burn NdArray forward pass matches pure-Rust.
//!
//! This loads the same model + inputs with both backends and compares outputs.

use std::collections::BTreeMap;
use std::path::Path;
use tribev2::config::*;
use tribev2::model::tribe::TribeV2;
use tribev2::model_burn::tribe::TribeV2Burn;
use tribev2::model_burn::weights::{BurnWeightStore, load_burn_weights};
use tribev2::tensor::Tensor as RustTensor;

const DATA_DIR: &str = "/Users/Shared/tribev2-rs/data";

fn refs_exist() -> bool {
    Path::new(&format!("{}/model.safetensors", DATA_DIR)).exists()
        && Path::new(&format!("{}/parity_refs/input_text.bin", DATA_DIR)).exists()
}

fn load_ref(name: &str) -> RustTensor {
    let path = format!("{}/parity_refs/{}", DATA_DIR, name);
    let bytes = std::fs::read(&path).unwrap();
    let ndims = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let mut shape = Vec::with_capacity(ndims);
    let mut offset = 4;
    for _ in 0..ndims {
        let d = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        shape.push(d);
        offset += 4;
    }
    let n_floats: usize = shape.iter().product();
    let data: Vec<f32> = (0..n_floats)
        .map(|i| f32::from_le_bytes(bytes[offset + i * 4..offset + i * 4 + 4].try_into().unwrap()))
        .collect();
    RustTensor::from_vec(data, shape)
}

fn pearson(x: &[f32], y: &[f32]) -> f64 {
    let n = x.len();
    let mx: f64 = x.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let my: f64 = y.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let mut cov = 0.0f64;
    let mut vx = 0.0f64;
    let mut vy = 0.0f64;
    for i in 0..n {
        let dx = x[i] as f64 - mx;
        let dy = y[i] as f64 - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    cov / (vx * vy).sqrt()
}

#[cfg(feature = "ndarray")]
#[test]
fn test_burn_ndarray_vs_pure_rust() {
    if !refs_exist() {
        eprintln!("SKIP: reference files not found");
        return;
    }
    eprintln!("\n══ Burn NdArray vs Pure-Rust Parity Test ══\n");

    // Load inputs
    let input_text = load_ref("input_text.bin");
    let input_audio = load_ref("input_audio.bin");
    let input_video = load_ref("input_video.bin");

    let config_str = std::fs::read_to_string(format!("{}/config.yaml", DATA_DIR)).unwrap();
    let mut config: TribeV2Config = serde_yaml::from_str(&config_str).unwrap();
    if let Some(ref mut sl) = config.brain_model_config.subject_layers {
        sl.average_subjects = true;
        sl.n_subjects = 0;
    }

    let build_args = ModelBuildArgs::from_json(&format!("{}/build_args.json", DATA_DIR)).unwrap();
    let feature_dims = build_args.to_modality_dims();
    let n_outputs = build_args.n_outputs;
    let n_output_timesteps = build_args.n_output_timesteps;

    // ── Pure-Rust forward pass ────────────────────────────────────────
    eprintln!("  Running pure-Rust forward...");
    let rust_model = TribeV2::from_pretrained(
        &format!("{}/config.yaml", DATA_DIR),
        &format!("{}/model.safetensors", DATA_DIR),
        Some(&format!("{}/build_args.json", DATA_DIR)),
    ).unwrap();

    let mut features = BTreeMap::new();
    features.insert("text".to_string(), input_text.clone());
    features.insert("audio".to_string(), input_audio.clone());
    features.insert("video".to_string(), input_video.clone());

    let t0 = std::time::Instant::now();
    let rust_output = rust_model.forward(&features, None, true);
    let rust_ms = t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!("  Pure-Rust: {:?} in {:.0}ms", rust_output.shape, rust_ms);

    // ── Burn NdArray forward pass ─────────────────────────────────────
    eprintln!("  Running Burn NdArray forward...");
    type B = burn::backend::NdArray;
    let device: <B as burn::prelude::Backend>::Device = Default::default();

    let mut burn_model = TribeV2Burn::<B>::new(
        &feature_dims, n_outputs, n_output_timesteps,
        &config.brain_model_config, &device,
    );

    let mut ws = BurnWeightStore::from_safetensors(
        &format!("{}/model.safetensors", DATA_DIR)
    ).unwrap();
    load_burn_weights(&mut ws, &mut burn_model, &device).unwrap();

    // Convert inputs to burn tensors
    let to_burn = |t: &RustTensor| -> burn::tensor::Tensor<B, 3> {
        burn::tensor::Tensor::from_data(
            burn::tensor::TensorData::new(t.data.clone(), [t.shape[0], t.shape[1], t.shape[2]]),
            &device,
        )
    };

    let burn_features = vec![
        ("audio", to_burn(&input_audio)),
        ("text", to_burn(&input_text)),
        ("video", to_burn(&input_video)),
    ];

    let t1 = std::time::Instant::now();
    let burn_output = burn_model.forward(burn_features);
    let burn_ms = t1.elapsed().as_secs_f64() * 1000.0;
    let [b, d, t] = burn_output.dims();
    eprintln!("  Burn NdArray: [{}, {}, {}] in {:.0}ms", b, d, t, burn_ms);

    // ── Compare outputs ───────────────────────────────────────────────
    let burn_data: Vec<f32> = burn_output.into_data().to_vec().unwrap();

    assert_eq!(rust_output.data.len(), burn_data.len(),
        "Output size mismatch: rust={} burn={}", rust_output.data.len(), burn_data.len());

    let r = pearson(&rust_output.data, &burn_data);
    let max_diff: f32 = rust_output.data.iter().zip(burn_data.iter())
        .map(|(&a, &b)| (a - b).abs()).fold(0.0f32, f32::max);
    let rmse: f64 = {
        let sum: f64 = rust_output.data.iter().zip(burn_data.iter())
            .map(|(&a, &b)| { let d = a as f64 - b as f64; d * d }).sum();
        (sum / rust_output.data.len() as f64).sqrt()
    };

    eprintln!("\n  ── Results ──");
    eprintln!("  Pearson:    {:.10}", r);
    eprintln!("  Max abs:    {:.2e}", max_diff);
    eprintln!("  RMSE:       {:.2e}", rmse);
    eprintln!("  Speedup:    {:.1}× (Burn {:.0}ms vs Rust {:.0}ms)",
        rust_ms / burn_ms, burn_ms, rust_ms);

    assert!(r > 0.9999,
        "Pearson {:.10} < 0.9999 — Burn NdArray diverges from pure-Rust", r);
    eprintln!("  ✅ PASS — Burn NdArray matches pure-Rust (Pearson={:.10})", r);
}

#[cfg(feature = "wgpu")]
#[test]
fn test_burn_wgpu_vs_pure_rust() {
    #[allow(unused_imports)]
    use burn::prelude::*;
    if !refs_exist() {
        eprintln!("SKIP: reference files not found");
        return;
    }
    eprintln!("\n══ Burn wgpu (Metal/Vulkan) vs Pure-Rust Parity Test ══\n");

    let input_text = load_ref("input_text.bin");
    let input_audio = load_ref("input_audio.bin");
    let input_video = load_ref("input_video.bin");

    let config_str = std::fs::read_to_string(format!("{}/config.yaml", DATA_DIR)).unwrap();
    let mut config: TribeV2Config = serde_yaml::from_str(&config_str).unwrap();
    if let Some(ref mut sl) = config.brain_model_config.subject_layers {
        sl.average_subjects = true;
        sl.n_subjects = 0;
    }

    let build_args = ModelBuildArgs::from_json(&format!("{}/build_args.json", DATA_DIR)).unwrap();
    let feature_dims = build_args.to_modality_dims();

    // ── Pure-Rust ─────────────────────────────────────────────────────
    let rust_model = TribeV2::from_pretrained(
        &format!("{}/config.yaml", DATA_DIR),
        &format!("{}/model.safetensors", DATA_DIR),
        Some(&format!("{}/build_args.json", DATA_DIR)),
    ).unwrap();

    let mut features = BTreeMap::new();
    features.insert("text".to_string(), input_text.clone());
    features.insert("audio".to_string(), input_audio.clone());
    features.insert("video".to_string(), input_video.clone());

    let t0 = std::time::Instant::now();
    let rust_output = rust_model.forward(&features, None, true);
    let rust_ms = t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!("  Pure-Rust: {:?} in {:.0}ms", rust_output.shape, rust_ms);

    // ── Burn wgpu ─────────────────────────────────────────────────────
    type B = burn::backend::Wgpu;
    let device = burn::backend::wgpu::WgpuDevice::DefaultDevice;

    let mut burn_model = TribeV2Burn::<B>::new(
        &feature_dims, build_args.n_outputs, build_args.n_output_timesteps,
        &config.brain_model_config, &device,
    );

    let mut ws = BurnWeightStore::from_safetensors(
        &format!("{}/model.safetensors", DATA_DIR)
    ).unwrap();
    load_burn_weights(&mut ws, &mut burn_model, &device).unwrap();

    let to_burn = |t: &RustTensor| -> burn::tensor::Tensor<B, 3> {
        burn::tensor::Tensor::from_data(
            burn::tensor::TensorData::new(t.data.clone(), [t.shape[0], t.shape[1], t.shape[2]]),
            &device,
        )
    };

    let burn_features = vec![
        ("audio", to_burn(&input_audio)),
        ("text", to_burn(&input_text)),
        ("video", to_burn(&input_video)),
    ];

    // Warmup
    eprintln!("  Warmup...");
    let _ = burn_model.forward(vec![
        ("audio", to_burn(&input_audio)),
        ("text", to_burn(&input_text)),
        ("video", to_burn(&input_video)),
    ]);

    let t1 = std::time::Instant::now();
    let burn_output = burn_model.forward(burn_features);
    let burn_ms = t1.elapsed().as_secs_f64() * 1000.0;
    let [b, d, t] = burn_output.dims();
    eprintln!("  Burn wgpu: [{}, {}, {}] in {:.0}ms", b, d, t, burn_ms);

    let burn_data: Vec<f32> = burn_output.into_data().to_vec().unwrap();

    let r = pearson(&rust_output.data, &burn_data);
    let max_diff: f32 = rust_output.data.iter().zip(burn_data.iter())
        .map(|(&a, &b)| (a - b).abs()).fold(0.0f32, f32::max);

    eprintln!("\n  ── Results ──");
    eprintln!("  Pearson:    {:.10}", r);
    eprintln!("  Max abs:    {:.2e}", max_diff);
    eprintln!("  Speedup:    {:.1}× (Burn wgpu {:.0}ms vs Rust CPU {:.0}ms)",
        rust_ms / burn_ms, burn_ms, rust_ms);

    // wgpu may have slightly lower precision due to GPU f32 differences
    assert!(r > 0.999,
        "Pearson {:.10} < 0.999 — Burn wgpu diverges from pure-Rust", r);
    eprintln!("  ✅ PASS — Burn wgpu matches pure-Rust (Pearson={:.10})", r);
}

#[cfg(not(any(feature = "ndarray", feature = "wgpu")))]
#[test]
fn test_burn_no_backend() {
    eprintln!("SKIP: no burn backend feature enabled");
}
