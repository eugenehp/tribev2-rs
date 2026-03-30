//! Example: Text-based brain activity prediction using TRIBE v2.
//!
//! Demonstrates building a model from config and running a forward pass
//! with synthetic text features. No pretrained weights needed.
//!
//! ```bash
//! cargo run --example text_predict
//! ```

use std::collections::BTreeMap;
use tribev2::config::{BrainModelConfig, EncoderConfig, ModalityDims, SubjectLayersConfig};
use tribev2::model::tribe::TribeV2;
use tribev2::tensor::Tensor;

fn main() -> anyhow::Result<()> {
    println!("TRIBE v2 — Text Prediction Example");
    println!("===================================\n");

    // Build a small model (random weights, for demonstration)
    let hidden = 128;
    let n_outputs = 100;
    let n_output_timesteps = 10;

    let feature_dims = vec![
        ModalityDims::new("text", 1, 128),
    ];

    let config = BrainModelConfig {
        hidden,
        max_seq_len: 128,
        extractor_aggregation: "cat".into(),
        layer_aggregation: "cat".into(),
        linear_baseline: false,
        time_pos_embedding: true,
        subject_embedding: false,
        dropout: 0.0,
        modality_dropout: 0.0,
        temporal_dropout: 0.0,
        low_rank_head: None,
        combiner: None,
        temporal_smoothing: None,
        projector: Default::default(),
        encoder: Some(EncoderConfig {
            heads: 4,
            depth: 2,
            ff_mult: 4,
            use_scalenorm: true,
            rotary_pos_emb: true,
            scale_residual: true,
            ..Default::default()
        }),
        subject_layers: Some(SubjectLayersConfig {
            n_subjects: 1,
            bias: true,
            subject_dropout: None,
            average_subjects: false,
            ..Default::default()
        }),
    };

    let model = TribeV2::new(feature_dims, n_outputs, n_output_timesteps, &config);

    println!("Model built:");
    println!("  Hidden dim: {}", hidden);
    println!("  Output vertices: {}", n_outputs);
    println!("  Output timesteps: {}", n_output_timesteps);

    // Create synthetic features: [1, 128, 20] (B=1, D=128, T=20)
    let t = 20;
    let d = 128;
    let data: Vec<f32> = (0..d * t).map(|i| (i as f32 * 0.01).sin()).collect();

    let mut features = BTreeMap::new();
    features.insert("text".to_string(), Tensor::from_vec(data, vec![1, d, t]));

    let t0 = std::time::Instant::now();
    let output = model.forward(&features, Some(&[0]), true);
    let elapsed = t0.elapsed();

    println!("\nForward pass:");
    println!("  Input: text [1, {}, {}]", d, t);
    println!("  Output shape: {:?}", output.shape);
    println!("  Time: {:.1} ms", elapsed.as_secs_f64() * 1000.0);

    let mean: f32 = output.data.iter().sum::<f32>() / output.data.len() as f32;
    let max: f32 = output.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min: f32 = output.data.iter().cloned().fold(f32::INFINITY, f32::min);
    println!("  Stats: mean={:.6}, min={:.6}, max={:.6}", mean, min, max);

    println!("\nDone!");
    Ok(())
}
