//! Example: Multi-modal brain prediction with all 3 modalities.
//!
//! Demonstrates building a full-size TRIBE v2 model with text, audio, and
//! video modalities, running a forward pass, and using segment-based batching.
//! Uses synthetic features (no pretrained weights needed).
//!
//! ```bash
//! cargo run --example multimodal_predict
//! ```

use std::collections::BTreeMap;
use tribev2::config::*;
use tribev2::model::tribe::TribeV2;
use tribev2::segments::{SegmentConfig, predict_segmented};
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
        encoder: Some(EncoderConfig::default()),
        subject_layers: Some(SubjectLayersConfig {
            n_subjects: 0,
            bias: true,
            subject_dropout: Some(0.1),
            average_subjects: true,
            ..Default::default()
        }),
    }
}

fn main() -> anyhow::Result<()> {
    println!("TRIBE v2 — Multi-Modal Prediction Example");
    println!("==========================================\n");

    let config = pretrained_config();
    let feature_dims = ModalityDims::pretrained();
    let n_outputs = 20484;       // fsaverage5
    let n_output_timesteps = 100;

    let model = TribeV2::new(feature_dims, n_outputs, n_output_timesteps, &config);

    // Synthetic features for 200 timesteps (100 seconds at 2 Hz)
    let n_timesteps = 200;
    let mut features: BTreeMap<String, Tensor> = BTreeMap::new();
    features.insert("text".into(),  Tensor::rand(&[1, 3 * 3072, n_timesteps]));
    features.insert("audio".into(), Tensor::rand(&[1, 3 * 1024, n_timesteps]));
    features.insert("video".into(), Tensor::rand(&[1, 3 * 1408, n_timesteps]));

    // ── Single forward pass ──────────────────────────────────────────
    println!("1) Single forward pass (first 100 timesteps):");
    let short_features: BTreeMap<String, Tensor> = features.iter()
        .map(|(k, v)| {
            let d = v.shape[1];
            let data = v.data[..d * 100].to_vec();
            (k.clone(), Tensor::from_vec(data, vec![1, d, 100]))
        })
        .collect();

    let t0 = std::time::Instant::now();
    let output = model.forward(&short_features, None, true);
    println!("   Output: {:?}  ({:.0} ms)", output.shape, t0.elapsed().as_secs_f64() * 1000.0);

    // ── Segment-based inference ──────────────────────────────────────
    println!("\n2) Segment-based inference (200 timesteps, 100-TR segments):");
    let seg_config = SegmentConfig {
        duration_trs: 100,
        overlap_trs: 0,
        remove_empty_segments: false,
        ..Default::default()
    };

    let t0 = std::time::Instant::now();
    let result = predict_segmented(&model, &features, &seg_config);
    let elapsed = t0.elapsed();

    println!("   Total TRs: {}", result.total_segments);
    println!("   Kept TRs:  {}", result.kept_segments);
    println!("   Predictions: {} × {} vertices", result.predictions.len(),
        result.predictions.first().map_or(0, |v| v.len()));
    println!("   Time: {:.0} ms", elapsed.as_secs_f64() * 1000.0);

    println!("\nDone!");
    Ok(())
}
