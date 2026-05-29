//! Compare TRIBE encoder forward passes: pure-Rust vs Burn vs RLX.
//!
//! Engines are **compile-time** optional (`burn`, `rlx-encoder`). Default crate
//! features are empty — enable backends explicitly or use
//! `bench/run_all_backends.sh` to sweep builds. See `bench/README.md`.
//!
//! ```text
//! # Single build: whatever features are enabled
//! cargo run --release -p tribev2 --example bench_encoder --no-default-features \
//!   --features pure-rust,burn,rlx-encoder,rlx-cpu -- --engines all
//!
//! # Full multi-backend sweep (separate cargo builds per GPU stack)
//! ./bench/run_all_backends.sh
//! ```

use std::collections::BTreeMap;
#[cfg(any(feature = "pure-rust", feature = "burn", feature = "rlx-encoder"))]
use std::time::Instant;

use clap::Parser;
use tribev2::config::*;
use tribev2::tensor::Tensor;

#[derive(Parser, Debug)]
#[command(about = "Benchmark TRIBE v2 encoder: rust / burn / rlx")]
struct Args {
    /// Engines: rust, burn, rlx, rlx-metal, … or `all` (every engine compiled into this binary)
    #[arg(long, default_value = "rust")]
    engines: String,

    /// RLX device when engine is `rlx` (cpu|metal|mps|mlx|cuda|rocm|wgpu|vulkan)
    #[arg(long, default_value = "cpu")]
    device: String,

    /// Input timesteps
    #[arg(long, default_value_t = 100)]
    timesteps: usize,

    /// Warmup iterations
    #[arg(long, default_value_t = 2)]
    warmup: usize,

    /// Timed iterations
    #[arg(long, default_value_t = 5)]
    runs: usize,

    /// Optional safetensors for RLX (must match pretrained arch)
    #[arg(long)]
    weights: Option<String>,

    /// Write JSON to bench/results_<key>.json
    #[arg(long, default_value_t = true)]
    json: bool,

    /// Print engines/backends compiled into this binary and exit
    #[arg(long)]
    list_engines: bool,
}

#[cfg(any(feature = "pure-rust", feature = "burn"))]
fn pretrained_config() -> BrainModelConfig {
    BrainModelConfig {
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
            heads: 8,
            depth: 8,
            ff_mult: 4,
            use_scalenorm: true,
            rotary_pos_emb: true,
            scale_residual: true,
            ..Default::default()
        }),
        subject_layers: Some(SubjectLayersConfig {
            n_subjects: 0,
            bias: true,
            subject_dropout: Some(0.1),
            average_subjects: true,
            ..Default::default()
        }),
    }
}

/// Feature layout for benchmarks: `build_args.json` next to weights, else `TRIBEV2_DATA_DIR`.
fn resolve_bench_modality_dims(weights: Option<&str>) -> Vec<ModalityDims> {
    let candidates: Vec<std::path::PathBuf> = weights
        .and_then(|w| {
            std::path::Path::new(w)
                .parent()
                .map(|p| p.join("build_args.json"))
        })
        .into_iter()
        .chain(std::iter::once(tribev2::data_paths::build_args_path()))
        .collect();

    for path in candidates {
        if path.exists() {
            if let Ok(ba) = ModelBuildArgs::from_json(path.to_str().unwrap_or_default()) {
                return ba.to_modality_dims();
            }
        }
    }
    ModalityDims::pretrained()
}

fn make_features(t: usize, modality_dims: &[ModalityDims]) -> BTreeMap<String, Tensor> {
    let mut features = BTreeMap::new();
    for md in modality_dims {
        if let Some((num_layers, feature_dim)) = md.dims {
            let flat = num_layers * feature_dim;
            features.insert(
                md.name.clone(),
                Tensor::from_vec(vec![0.01f32; flat * t], vec![1, flat, t]),
            );
        }
    }
    features
}

#[cfg(any(feature = "pure-rust", feature = "burn", feature = "rlx-encoder"))]
fn stats(times: &[f64]) -> (f64, f64, f64, f64) {
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let std = (times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / times.len() as f64).sqrt();
    (mean, min, max, std)
}

#[cfg(any(feature = "pure-rust", feature = "burn", feature = "rlx-encoder"))]
fn write_json(
    key: &str,
    engine: &str,
    device: &str,
    mean: f64,
    min: f64,
    max: f64,
    std: f64,
    runs: usize,
    t: usize,
) {
    let json = format!(
        r#"{{"{key}":{{"engine":"{engine}","device":"{device}","mean_ms":{mean:.1},"min_ms":{min:.1},"max_ms":{max:.1},"std_ms":{std:.1},"n_runs":{runs},"timesteps":{t},"output_shape":[1,20484,100]}}}}"#
    );
    let path = format!("bench/results_{key}.json");
    std::fs::create_dir_all("bench").ok();
    std::fs::write(&path, json).expect("write bench json");
    println!("  → {path}");
}

fn print_compiled_engines() {
    print!("Compiled engines:");
    #[cfg(feature = "pure-rust")]
    print!(" rust");
    #[cfg(feature = "burn")]
    {
        print!(" burn({})", burn_bench::backend::NAME);
    }
    #[cfg(feature = "rlx-encoder")]
    {
        print!(" rlx[");
        let labels = tribev2::rlx_device::available_rlx_device_labels();
        print!("{}", labels.join(","));
        print!("]");
    }
    println!();
}

/// Expand `all` to every engine enabled in this binary.
fn resolve_engine_tokens(spec: &str) -> Vec<String> {
    if spec.trim() != "all" {
        return spec
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
    }

    #[cfg(any(feature = "pure-rust", feature = "burn", feature = "rlx-encoder"))]
    {
        let mut out = Vec::new();
        #[cfg(feature = "pure-rust")]
        {
            out.push("rust".into());
        }
        #[cfg(feature = "burn")]
        {
            out.push("burn".into());
        }
        #[cfg(feature = "rlx-encoder")]
        {
            for label in tribev2::rlx_device::available_rlx_device_labels() {
                out.push(format!("rlx-{label}"));
            }
        }
        out
    }
    #[cfg(not(any(feature = "pure-rust", feature = "burn", feature = "rlx-encoder")))]
    {
        Vec::new()
    }
}

#[cfg(feature = "pure-rust")]
fn bench_rust(
    features: &BTreeMap<String, Tensor>,
    weights: Option<&str>,
    warmup: usize,
    runs: usize,
) -> Option<(f64, f64, f64, f64)> {
    use tribev2::model::tribe::TribeV2;
    let model = if let Some(w) = weights {
        let cfg = tribev2::data_paths::config_path();
        let ba = tribev2::data_paths::build_args_path();
        let ba = if ba.exists() {
            Some(ba.to_string_lossy().into_owned())
        } else {
            None
        };
        TribeV2::from_pretrained(cfg.to_str()?, w, ba.as_deref()).ok()?
    } else {
        let config = pretrained_config();
        TribeV2::new(ModalityDims::pretrained(), 20484, 100, &config)
    };
    let model = model;
    for _ in 0..warmup {
        let _ = model.forward(features, None, true);
    }
    let mut times = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t0 = Instant::now();
        let _ = model.forward(features, None, true);
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    Some(stats(&times))
}

#[cfg(feature = "burn")]
mod burn_bench {
    use super::*;

    #[cfg(feature = "wgpu")]
    pub mod backend {
        pub use burn::backend::Wgpu as B;
        pub fn device() -> burn::backend::wgpu::WgpuDevice {
            burn::backend::wgpu::WgpuDevice::DefaultDevice
        }
        #[cfg(feature = "wgpu-kernels-metal")]
        pub const NAME: &str = "burn/wgpu-metal-kernels";
        #[cfg(all(feature = "wgpu-metal-f16", not(feature = "wgpu-kernels-metal")))]
        pub const NAME: &str = "burn/wgpu-metal-f16";
        #[cfg(all(
            feature = "wgpu-metal",
            not(feature = "wgpu-metal-f16"),
            not(feature = "wgpu-kernels-metal")
        ))]
        pub const NAME: &str = "burn/wgpu-metal";
        #[cfg(feature = "wgpu-vulkan")]
        pub const NAME: &str = "burn/wgpu-vulkan";
        #[cfg(not(any(
            feature = "wgpu-kernels-metal",
            feature = "wgpu-metal-f16",
            feature = "wgpu-metal",
            feature = "wgpu-vulkan"
        )))]
        pub const NAME: &str = "burn/wgpu";
        pub const JSON_KEY: &str = if cfg!(feature = "wgpu-kernels-metal") {
            "burn_wgpu_metal_kernels"
        } else if cfg!(feature = "wgpu-metal-f16") {
            "burn_wgpu_metal_f16"
        } else if cfg!(feature = "wgpu-metal") {
            "burn_wgpu_metal"
        } else if cfg!(feature = "wgpu-vulkan") {
            "burn_wgpu_vulkan"
        } else {
            "burn_wgpu"
        };
    }

    #[cfg(not(feature = "wgpu"))]
    pub mod backend {
        pub use burn::backend::NdArray as B;
        pub fn device() -> burn::backend::ndarray::NdArrayDevice {
            burn::backend::ndarray::NdArrayDevice::Cpu
        }
        #[cfg(feature = "blas-accelerate")]
        pub const NAME: &str = "burn/ndarray-accelerate";
        #[cfg(not(feature = "blas-accelerate"))]
        pub const NAME: &str = "burn/ndarray";
        pub const JSON_KEY: &str = if cfg!(feature = "blas-accelerate") {
            "burn_ndarray_accelerate"
        } else {
            "burn_ndarray"
        };
    }

    pub fn run(
        modality_dims: &[ModalityDims],
        t: usize,
        warmup: usize,
        runs: usize,
    ) -> Option<(f64, f64, f64, f64)> {
        use burn::prelude::*;
        use tribev2::model_burn::tribe::TribeV2Burn;

        let dev = backend::device();
        let config = super::pretrained_config();
        let model = TribeV2Burn::<backend::B>::new(modality_dims, 20484, 100, &config, &dev);

        let mut tensors: Vec<(&str, Tensor<backend::B, 3>)> = Vec::new();
        for md in modality_dims {
            let (num_layers, feature_dim) = md.dims?;
            let flat = num_layers * feature_dim;
            let name: &'static str = match md.name.as_str() {
                "text" => "text",
                "audio" => "audio",
                "video" => "video",
                other => {
                    eprintln!("  SKIP burn: unknown modality {other}");
                    return None;
                }
            };
            tensors.push((
                name,
                Tensor::<backend::B, 3>::ones([1, flat, t], &dev).mul_scalar(0.01),
            ));
        }

        for _ in 0..warmup {
            let feats: Vec<_> = tensors.iter().map(|(n, x)| (*n, x.clone())).collect();
            let _ = model.forward(feats).into_data();
        }

        let mut times = Vec::with_capacity(runs);
        for _ in 0..runs {
            let feats: Vec<_> = tensors.iter().map(|(n, x)| (*n, x.clone())).collect();
            let t0 = Instant::now();
            let _ = model.forward(feats).into_data();
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        Some(super::stats(&times))
    }
}

#[cfg(feature = "rlx-encoder")]
fn bench_rlx(
    features: &BTreeMap<String, Tensor>,
    device_label: &str,
    weights: Option<&str>,
    warmup: usize,
    runs: usize,
) -> Option<(f64, f64, f64, f64, String)> {
    use tribev2::model_rlx::TribeRlx;
    use tribev2::rlx_device::parse_rlx_device_available;

    let weights = weights?;
    if !std::path::Path::new(weights).exists() {
        eprintln!("  SKIP rlx: weights not found at {weights}");
        return None;
    }

    let device = parse_rlx_device_available(device_label).ok()?;
    let config_path = std::path::Path::new(weights)
        .parent()
        .map(|p| p.join("config.yaml"))
        .filter(|p| p.exists())
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|| "data/config.yaml".into());
    let build_args = std::path::Path::new(weights)
        .parent()
        .map(|p| p.join("build_args.json"))
        .filter(|p| p.exists())
        .map(|p| p.to_string_lossy().into_owned());

    let mut model = TribeRlx::from_pretrained(&config_path, weights, build_args.as_deref()).ok()?;
    model = model.with_device(device);

    for _ in 0..warmup {
        let _ = model.forward(features, None, true);
    }
    let mut times = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t0 = Instant::now();
        let _ = model.forward(features, None, true);
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let (mean, min, max, std) = stats(&times);
    let key = format!("rlx_{}", device_label.replace('/', "_"));
    Some((mean, min, max, std, key))
}

#[allow(unused_variables)]
fn run_engine(
    engine: &str,
    args: &Args,
    features: &BTreeMap<String, Tensor>,
    modality_dims: &[ModalityDims],
    weights: Option<&str>,
) {
    let is_rust = matches!(engine, "rust" | "cpu" | "pure-rust");
    let is_burn = engine == "burn" || engine.starts_with("burn-");
    let is_rlx = engine == "rlx" || engine.starts_with("rlx-");

    if is_rust {
        #[cfg(feature = "pure-rust")]
        {
            println!("[rust] pure-Rust CPU");
            if let Some((mean, min, max, std)) = bench_rust(features, weights, args.warmup, args.runs)
            {
                println!("  mean={mean:.1} ms  min={min:.1}  max={max:.1}  std={std:.1}");
                if args.json {
                    write_json(
                        "rust_cpu",
                        "rust",
                        "cpu",
                        mean,
                        min,
                        max,
                        std,
                        args.runs,
                        args.timesteps,
                    );
                }
            }
        }
        #[cfg(not(feature = "pure-rust"))]
        eprintln!("  SKIP rust: rebuild with `--features pure-rust`");
    } else if is_burn {
        #[cfg(feature = "burn")]
        {
            println!("[burn] {}", burn_bench::backend::NAME);
            if let Some((mean, min, max, std)) =
                burn_bench::run(modality_dims, args.timesteps, args.warmup, args.runs)
            {
                println!("  mean={mean:.1} ms  min={min:.1}  max={max:.1}  std={std:.1}");
                if args.json {
                    write_json(
                        burn_bench::backend::JSON_KEY,
                        "burn",
                        burn_bench::backend::NAME,
                        mean,
                        min,
                        max,
                        std,
                        args.runs,
                        args.timesteps,
                    );
                }
            }
        }
        #[cfg(not(feature = "burn"))]
        eprintln!("  SKIP burn: rebuild with `--features burn` (+ `wgpu-metal` or `ndarray`)");
    } else if is_rlx {
        #[cfg(feature = "rlx-encoder")]
        {
            let dev = if engine == "rlx" {
                args.device.as_str()
            } else {
                engine.strip_prefix("rlx-").unwrap_or(args.device.as_str())
            };
            println!("[rlx] device={dev}");
            if let Some((mean, min, max, std, key)) =
                bench_rlx(features, dev, weights, args.warmup, args.runs)
            {
                println!("  mean={mean:.1} ms  min={min:.1}  max={max:.1}  std={std:.1}");
                if args.json {
                    write_json(&key, "rlx", dev, mean, min, max, std, args.runs, args.timesteps);
                }
            }
        }
        #[cfg(not(feature = "rlx-encoder"))]
        eprintln!("  SKIP rlx: rebuild with `--features rlx-encoder` (+ `rlx-metal`, …)");
    } else {
        eprintln!("  SKIP unknown engine '{engine}'");
    }
}

fn main() {
    let args = Args::parse();

    if args.list_engines {
        print_compiled_engines();
        return;
    }

    let default_weights = tribev2::data_paths::weights_path();
    let default_weights = default_weights.to_string_lossy().into_owned();
    let weights = args
        .weights
        .as_deref()
        .or(Some(default_weights.as_str()));
    let modality_dims = resolve_bench_modality_dims(weights);
    let features = make_features(args.timesteps, &modality_dims);

    println!("=== TRIBE v2 encoder benchmark (T={}) ===\n", args.timesteps);
    print_compiled_engines();
    println!();

    let engines = resolve_engine_tokens(&args.engines);
    if engines.is_empty() {
        eprintln!("No engines enabled in this build. Use `--features` or `--list-engines`.");
        std::process::exit(1);
    }

    for engine in &engines {
        run_engine(engine, &args, &features, &modality_dims, weights);
        println!();
    }
}
