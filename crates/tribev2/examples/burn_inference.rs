//! Example: Burn backend inference for TRIBE v2.
//!
//! Demonstrates building and running the burn-based model with the
//! NdArray CPU backend. Shows both random-weight and pretrained paths.
//!
//! ```bash
//! # CPU (NdArray)
//! cargo run --example burn_inference
//!
//! # GPU (Metal)
//! cargo run --example burn_inference --no-default-features --features wgpu-metal,llama-metal
//! ```

use tribev2::config::*;
use tribev2::model_burn::tribe::TribeV2Burn;

// ── Backend selection ─────────────────────────────────────────────────────
#[cfg(all(feature = "ndarray", not(feature = "wgpu")))]
mod backend {
    pub type B = burn::backend::NdArray;
    pub fn device() -> <B as burn::prelude::Backend>::Device { Default::default() }
    pub const NAME: &str = "NdArray";
}

#[cfg(all(feature = "wgpu", not(feature = "ndarray")))]
mod backend {
    pub type B = burn::backend::Wgpu;
    pub fn device() -> burn::backend::wgpu::WgpuDevice {
        burn::backend::wgpu::WgpuDevice::DefaultDevice
    }
    pub const NAME: &str = "wgpu";
}

#[cfg(all(feature = "ndarray", feature = "wgpu"))]
mod backend {
    pub type B = burn::backend::NdArray;
    pub fn device() -> <B as burn::prelude::Backend>::Device { Default::default() }
    pub const NAME: &str = "NdArray (wgpu also enabled; using NdArray)";
}

use backend::{B, device, NAME};
use burn::prelude::*;

fn main() -> anyhow::Result<()> {
    println!("TRIBE v2 — Burn Backend Inference");
    println!("=================================");
    println!("Backend: {}\n", NAME);

    let dev = device();

    // ── Build model from config ───────────────────────────────────────
    let config = BrainModelConfig {
        hidden: 256,
        max_seq_len: 128,
        extractor_aggregation: "cat".into(),
        layer_aggregation: "cat".into(),
        linear_baseline: false,
        time_pos_embedding: true,
        subject_embedding: false,
        low_rank_head: Some(64),
        combiner: None,
        temporal_smoothing: None,
        projector: Default::default(),
        encoder: Some(EncoderConfig {
            heads: 4,
            depth: 2,
            ff_mult: 4,
            ..Default::default()
        }),
        subject_layers: Some(SubjectLayersConfig {
            n_subjects: 0,
            bias: true,
            subject_dropout: Some(0.1),
            average_subjects: true,
            ..Default::default()
        }),
        ..Default::default()
    };

    let feature_dims = vec![
        ModalityDims::new("text", 2, 128),
        ModalityDims::new("audio", 2, 64),
    ];
    let n_outputs = 500;
    let n_output_timesteps = 20;

    let model = TribeV2Burn::<B>::new(&feature_dims, n_outputs, n_output_timesteps, &config, &dev);

    println!("Model built:");
    println!("  Hidden: {}", config.hidden);
    println!("  Encoder depth: {}", config.encoder.as_ref().map_or(0, |e| e.depth));
    println!("  Low-rank head: {:?}", config.low_rank_head);
    println!("  Output: {} vertices × {} timesteps", n_outputs, n_output_timesteps);

    // ── Create synthetic features ─────────────────────────────────────
    let t = 40;
    let text  = Tensor::<B, 3>::zeros([1, 2 * 128, t], &dev);
    let audio = Tensor::<B, 3>::zeros([1, 2 * 64, t], &dev);

    let features = vec![("text", text), ("audio", audio)];

    // ── Forward pass ──────────────────────────────────────────────────
    let t0 = std::time::Instant::now();
    let output = model.forward(features);
    let elapsed = t0.elapsed();

    let [b, d, t_out] = output.dims();
    println!("\nForward pass:");
    println!("  Output: [{}, {}, {}]", b, d, t_out);
    println!("  Time: {:.1} ms", elapsed.as_secs_f64() * 1000.0);

    // ── Load pretrained weights (if available) ────────────────────────
    // Uncomment to load real weights:
    //
    // use tribev2::model_burn::weights::{BurnWeightStore, load_burn_weights};
    // let mut ws = BurnWeightStore::from_safetensors("data/model.safetensors")?;
    // let mut model = TribeV2Burn::<B>::new(&ModalityDims::pretrained(), 20484, 100,
    //     &pretrained_config, &dev);
    // load_burn_weights(&mut ws, &mut model, &dev)?;

    println!("\nDone!");
    Ok(())
}
