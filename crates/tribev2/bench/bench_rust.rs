//! Rust benchmark for TRIBE v2 forward pass.
//! Run with: cargo run --release --example bench_rust
//!       or: cargo run --release --features accelerate --example bench_rust

use std::collections::BTreeMap;
use std::time::Instant;
use tribev2::config::*;
use tribev2::model::tribe::TribeV2;
use tribev2::tensor::Tensor;

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
            n_subjects: 25,
            bias: true,
            subject_dropout: Some(0.1),
            average_subjects: true,
            ..Default::default()
        }),
    }
}

fn make_input(t: usize) -> BTreeMap<String, Tensor> {
    let mut features = BTreeMap::new();
    features.insert("text".into(), Tensor::from_vec(vec![0.01f32; 9216 * t], vec![1, 9216, t]));
    features.insert("audio".into(), Tensor::from_vec(vec![0.01f32; 3072 * t], vec![1, 3072, t]));
    features.insert("video".into(), Tensor::from_vec(vec![0.01f32; 4224 * t], vec![1, 4224, t]));
    features
}

fn benchmark(
    model: &TribeV2,
    features: &BTreeMap<String, Tensor>,
    n_warmup: usize,
    n_runs: usize,
) -> (f64, f64, f64, f64) {
    for _ in 0..n_warmup {
        let _ = model.forward(features, None, true);
    }

    let mut times = Vec::with_capacity(n_runs);
    for _ in 0..n_runs {
        let t0 = Instant::now();
        let _ = model.forward(features, None, true);
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let std = (times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / times.len() as f64).sqrt();
    (mean, min, max, std)
}

fn main() {
    let backend = if cfg!(feature = "accelerate") {
        "Accelerate BLAS"
    } else {
        "naive loops"
    };

    println!("=== Rust CPU ({backend}) ===");

    let config = pretrained_config();
    let feature_dims = ModalityDims::pretrained();
    let model = TribeV2::new(feature_dims, 20484, 100, &config);
    let features = make_input(100);

    let n_warmup = 2;
    let n_runs = 5;

    let (mean, min, max, std) = benchmark(&model, &features, n_warmup, n_runs);
    println!("  Mean: {mean:.1} ms, Min: {min:.1} ms, Max: {max:.1} ms, Std: {std:.1} ms");

    let key = if cfg!(feature = "accelerate") {
        "rust_cpu_accelerate"
    } else {
        "rust_cpu"
    };

    let json = format!(
        r#"{{"{key}":{{"mean_ms":{mean:.1},"min_ms":{min:.1},"max_ms":{max:.1},"std_ms":{std:.1},"n_runs":{n_runs},"output_shape":[1,20484,100]}}}}"#
    );
    let path = format!("bench/results_{key}.json");
    std::fs::write(&path, &json).unwrap();
    println!("Results saved to {path}");
}
