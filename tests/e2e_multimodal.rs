//! End-to-end multi-modal inference test with real pretrained weights.
//!
//! Loads the pretrained TRIBE v2 model, generates synthetic multi-modal features
//! (text + audio + video), runs segmented inference, and produces brain maps + PNG.

use std::collections::BTreeMap;
use tribev2::model::tribe::TribeV2;
use tribev2::tensor::Tensor;
use tribev2::segments::{SegmentConfig, predict_segmented};
use tribev2::plotting;

const DATA_DIR: &str = "/Users/Shared/tribev2-rs/data";

fn model_files_exist() -> bool {
    std::path::Path::new(&format!("{}/model.safetensors", DATA_DIR)).exists()
        && std::path::Path::new(&format!("{}/config.yaml", DATA_DIR)).exists()
        && std::path::Path::new(&format!("{}/build_args.json", DATA_DIR)).exists()
}

/// Generate synthetic features mimicking real extractor output.
///
/// For text (LLaMA-3.2-3B, 2 layer groups, dim=3072): [1, 6144, T]
/// For audio (Wav2Vec-BERT, 2 layer groups, dim=1024): [1, 2048, T]
/// For video (V-JEPA2, 2 layer groups, dim=1408): [1, 2816, T]
fn generate_synthetic_features(n_timesteps: usize) -> BTreeMap<String, Tensor> {
    let mut features = BTreeMap::new();

    // Text: sinusoidal pattern at different frequencies per dimension
    let text_dim = 6144;
    let mut text_data = vec![0.0f32; text_dim * n_timesteps];
    for d in 0..text_dim {
        let freq = 0.01 + (d as f32) * 0.001;
        for t in 0..n_timesteps {
            text_data[d * n_timesteps + t] = (freq * t as f32).sin() * 0.5;
        }
    }
    features.insert("text".to_string(), Tensor::from_vec(text_data, vec![1, text_dim, n_timesteps]));

    // Audio: different sinusoidal pattern
    let audio_dim = 2048;
    let mut audio_data = vec![0.0f32; audio_dim * n_timesteps];
    for d in 0..audio_dim {
        let freq = 0.02 + (d as f32) * 0.002;
        for t in 0..n_timesteps {
            audio_data[d * n_timesteps + t] = (freq * t as f32).cos() * 0.3;
        }
    }
    features.insert("audio".to_string(), Tensor::from_vec(audio_data, vec![1, audio_dim, n_timesteps]));

    // Video: pulse pattern
    let video_dim = 2816;
    let mut video_data = vec![0.0f32; video_dim * n_timesteps];
    for d in 0..video_dim {
        for t in 0..n_timesteps {
            let phase = (d as f32 * 0.003 + t as f32 * 0.05).sin();
            video_data[d * n_timesteps + t] = if phase > 0.0 { 0.4 } else { -0.1 };
        }
    }
    features.insert("video".to_string(), Tensor::from_vec(video_data, vec![1, video_dim, n_timesteps]));

    features
}

#[test]
fn test_e2e_multimodal_inference() {
    if !model_files_exist() {
        eprintln!("Skipping e2e test: model files not found in {}", DATA_DIR);
        return;
    }

    let output_dir = format!("{}/e2e_output", DATA_DIR);
    std::fs::create_dir_all(&output_dir).unwrap();

    eprintln!("=== TRIBE v2 End-to-End Multi-Modal Test ===\n");

    // ── 1. Load model ─────────────────────────────────────────────
    eprintln!("1. Loading model...");
    let t0 = std::time::Instant::now();

    let model = TribeV2::from_pretrained(
        &format!("{}/config.yaml", DATA_DIR),
        &format!("{}/model.safetensors", DATA_DIR),
        Some(&format!("{}/build_args.json", DATA_DIR)),
    ).unwrap();

    eprintln!("   Model loaded in {:.0}ms", t0.elapsed().as_millis());
    eprintln!("   Hidden: {}", model.config.hidden);
    eprintln!("   Projectors: {}", model.projectors.len());
    eprintln!("   Encoder layers: {}", model.encoder.as_ref().map_or(0, |e| e.layers.len()));
    eprintln!("   N outputs: {}", model.n_outputs);
    eprintln!("   N output timesteps: {}", model.n_output_timesteps);
    eprintln!("   Predictor weights: {:?}", model.predictor.weights.shape);

    assert_eq!(model.config.hidden, 1152);
    assert_eq!(model.n_outputs, 20484);
    assert_eq!(model.n_output_timesteps, 100);
    assert_eq!(model.projectors.len(), 3);

    // ── 2. Generate multi-modal features ──────────────────────────
    eprintln!("\n2. Generating synthetic multi-modal features...");
    let n_timesteps = 200; // 200 timesteps at 2Hz = 100 seconds
    let features = generate_synthetic_features(n_timesteps);

    for (name, tensor) in &features {
        eprintln!("   {}: {:?}", name, tensor.shape);
    }

    // ── 3. Single forward pass ────────────────────────────────────
    eprintln!("\n3. Running single forward pass (first 100 timesteps)...");
    let t1 = std::time::Instant::now();

    // Slice to first 100 timesteps for direct forward
    let mut features_100 = BTreeMap::new();
    for (name, tensor) in &features {
        let dim = tensor.shape[1];
        let t = 100;
        let mut data = vec![0.0f32; dim * t];
        for d in 0..dim {
            for ti in 0..t {
                data[d * t + ti] = tensor.data[d * n_timesteps + ti];
            }
        }
        features_100.insert(name.clone(), Tensor::from_vec(data, vec![1, dim, t]));
    }

    let output = model.forward(&features_100, None, true);
    let fwd_ms = t1.elapsed().as_millis();

    eprintln!("   Output shape: {:?}", output.shape);
    eprintln!("   Forward pass: {}ms", fwd_ms);

    assert_eq!(output.shape, vec![1, 20484, 100]);

    // Stats
    let mean: f32 = output.data.iter().sum::<f32>() / output.data.len() as f32;
    let min: f32 = output.data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max: f32 = output.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let std_dev = (output.data.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / output.data.len() as f32).sqrt();
    eprintln!("   Stats: mean={:.6}, std={:.6}, min={:.6}, max={:.6}", mean, std_dev, min, max);

    // Verify output is non-trivial (not all zeros, has variation)
    assert!(std_dev > 1e-8, "output has no variation — likely broken");
    assert!(min != max, "all values identical");

    // ── 4. Save raw embeddings ────────────────────────────────────
    eprintln!("\n4. Saving raw predictions...");

    // Save as binary f32: [n_timesteps, n_vertices]
    let n_out = output.shape[1]; // 20484
    let n_t = output.shape[2];   // 100
    let mut flat = Vec::with_capacity(n_t * n_out);
    for ti in 0..n_t {
        for di in 0..n_out {
            flat.push(output.data[di * n_t + ti]);
        }
    }
    let bytes: Vec<u8> = flat.iter().flat_map(|f| f.to_le_bytes()).collect();
    let pred_path = format!("{}/predictions.bin", output_dir);
    std::fs::write(&pred_path, &bytes).unwrap();
    eprintln!("   Saved {} ({} timesteps × {} vertices, {:.1} MB)",
        pred_path, n_t, n_out, bytes.len() as f64 / 1e6);

    // ── 5. Segment-based inference ────────────────────────────────
    eprintln!("\n5. Running segment-based inference (200 timesteps, segments of 100)...");
    let t2 = std::time::Instant::now();

    let seg_config = SegmentConfig {
        duration_trs: 100,
        overlap_trs: 0,
        tr: 0.5,
        remove_empty_segments: false,
        feature_frequency: 2.0,
        stride_drop_incomplete: false,
    };

    let seg_result = predict_segmented(&model, &features, &seg_config);
    let seg_ms = t2.elapsed().as_millis();

    eprintln!("   Segmented inference: {}ms", seg_ms);
    eprintln!("   Total TRs: {}, Kept: {}", seg_result.total_segments, seg_result.kept_segments);
    eprintln!("   Prediction rows: {}", seg_result.predictions.len());

    assert!(seg_result.predictions.len() > 0);
    assert_eq!(seg_result.predictions[0].len(), 20484);

    // Save segmented predictions
    let seg_flat: Vec<f32> = seg_result.predictions.iter().flatten().copied().collect();
    let seg_bytes: Vec<u8> = seg_flat.iter().flat_map(|f| f.to_le_bytes()).collect();
    let seg_path = format!("{}/predictions_segmented.bin", output_dir);
    std::fs::write(&seg_path, &seg_bytes).unwrap();
    eprintln!("   Saved {} ({} TRs × {} vertices, {:.1} MB)",
        seg_path, seg_result.predictions.len(), 20484, seg_bytes.len() as f64 / 1e6);

    // ── 6. Generate brain surface plots (SVG) ─────────────────────
    eprintln!("\n6. Generating brain surface visualizations...");
    let t3 = std::time::Instant::now();

    // Load real fsaverage5 mesh (10242 vertices per hemisphere = 20484 total)
    let brain = tribev2::fsaverage::load_fsaverage(
        "fsaverage5", "half", "sulcal",
        Some(DATA_DIR),
    ).expect("failed to load fsaverage5 — ensure data/fsaverage5/surf/ exists");

    eprintln!("   Loaded fsaverage5: {} + {} vertices",
        brain.left.mesh.n_vertices, brain.right.mesh.n_vertices);

    // Predictions are [20484] which matches fsaverage5 exactly
    let first_pred = &seg_result.predictions[0];
    let n_mesh_verts = brain.left.mesh.n_vertices + brain.right.mesh.n_vertices;

    let downsampled: Vec<f32> = if first_pred.len() == n_mesh_verts {
        first_pred.clone()
    } else {
        (0..n_mesh_verts)
            .map(|i| {
                let src_idx = i * first_pred.len() / n_mesh_verts;
                first_pred[src_idx.min(first_pred.len() - 1)]
            })
            .collect()
    };

    // Left view
    let config_left = plotting::PlotConfig {
        width: 800,
        height: 600,
        cmap: plotting::ColorMap::Hot,
        view: plotting::View::Left,
        colorbar: true,
        title: Some("TRIBE v2 — Left hemisphere (t=0)".into()),
        ..Default::default()
    };
    let svg_left = plotting::render_brain_svg(&downsampled, &brain, &config_left);
    let svg_left_path = format!("{}/brain_left.svg", output_dir);
    std::fs::write(&svg_left_path, &svg_left).unwrap();
    eprintln!("   Saved {}", svg_left_path);

    // Right view
    let config_right = plotting::PlotConfig {
        view: plotting::View::Right,
        title: Some("TRIBE v2 — Right hemisphere (t=0)".into()),
        ..config_left.clone()
    };
    let svg_right = plotting::render_brain_svg(&downsampled, &brain, &config_right);
    let svg_right_path = format!("{}/brain_right.svg", output_dir);
    std::fs::write(&svg_right_path, &svg_right).unwrap();
    eprintln!("   Saved {}", svg_right_path);

    // Dorsal view
    let config_dorsal = plotting::PlotConfig {
        view: plotting::View::Dorsal,
        title: Some("TRIBE v2 — Dorsal (t=0)".into()),
        ..config_left.clone()
    };
    let svg_dorsal = plotting::render_brain_svg(&downsampled, &brain, &config_dorsal);
    let svg_dorsal_path = format!("{}/brain_dorsal.svg", output_dir);
    std::fs::write(&svg_dorsal_path, &svg_dorsal).unwrap();
    eprintln!("   Saved {}", svg_dorsal_path);

    // CoolWarm colormap
    let config_cw = plotting::PlotConfig {
        cmap: plotting::ColorMap::CoolWarm,
        symmetric_cbar: true,
        view: plotting::View::Left,
        title: Some("TRIBE v2 — CoolWarm (t=0)".into()),
        ..config_left.clone()
    };
    let svg_cw = plotting::render_brain_svg(&downsampled, &brain, &config_cw);
    std::fs::write(format!("{}/brain_coolwarm.svg", output_dir), &svg_cw).unwrap();
    eprintln!("   Saved brain_coolwarm.svg");

    // RGB multi-modal overlay — simulate 3 modality contributions
    let text_signal: Vec<f32> = (0..n_mesh_verts).map(|i| {
        first_pred[i.min(first_pred.len() - 1)].abs()
    }).collect();
    let audio_signal: Vec<f32> = (0..n_mesh_verts).map(|i| {
        let src = (i + 5000) % first_pred.len();
        first_pred[src].abs() * 0.8
    }).collect();
    let video_signal: Vec<f32> = (0..n_mesh_verts).map(|i| {
        let src = (i + 10000) % first_pred.len();
        first_pred[src].abs() * 0.6
    }).collect();

    let rgb_colors = plotting::rgb_overlay(
        &[&text_signal, &audio_signal, &video_signal],
        95.0, 0.3, None,
    );

    let left_colors: Vec<(u8,u8,u8)> = rgb_colors[..brain.left.mesh.n_vertices].to_vec();
    let rgb_svg = plotting::render_hemisphere_rgb_svg(
        &left_colors,
        &brain.left.mesh,
        &plotting::PlotConfig { width: 800, height: 600, view: plotting::View::Left, ..Default::default() },
    );
    std::fs::write(format!("{}/brain_rgb_overlay.svg", output_dir), &rgb_svg).unwrap();
    eprintln!("   Saved brain_rgb_overlay.svg");

    // Mosaic of multiple views
    let views = vec![plotting::View::Left, plotting::View::Right, plotting::View::Dorsal];
    let svgs: Vec<String> = views.iter().map(|&view| {
        let cfg = plotting::PlotConfig {
            width: 400, height: 300, view, cmap: plotting::ColorMap::Hot,
            title: Some(format!("{}", view.name())),
            ..Default::default()
        };
        plotting::render_brain_svg(&downsampled, &brain, &cfg)
    }).collect();
    let mosaic = plotting::combine_svgs(&svgs, 400, 300, 3, 10);
    std::fs::write(format!("{}/brain_mosaic.svg", output_dir), &mosaic).unwrap();
    eprintln!("   Saved brain_mosaic.svg");

    // Time series: first 5 timesteps
    let ts_preds: Vec<Vec<f32>> = seg_result.predictions.iter().take(5).map(|pred| {
        (0..n_mesh_verts).map(|i| {
            let src = i * pred.len() / n_mesh_verts;
            pred[src.min(pred.len() - 1)]
        }).collect()
    }).collect();
    let ts_paths = plotting::render_timesteps(
        &ts_preds, &brain,
        &plotting::PlotConfig {
            width: 400, height: 300,
            cmap: plotting::ColorMap::Hot,
            view: plotting::View::Left,
            ..Default::default()
        },
        &format!("{}/timesteps", output_dir),
    ).unwrap();
    eprintln!("   Saved {} timestep frames", ts_paths.len());

    let plot_ms = t3.elapsed().as_millis();
    eprintln!("   Plotting: {}ms", plot_ms);

    // ── 7. Summary statistics ─────────────────────────────────────
    eprintln!("\n7. Summary statistics per timestep:");
    for (ti, pred) in seg_result.predictions.iter().take(5).enumerate() {
        let mean: f32 = pred.iter().sum::<f32>() / pred.len() as f32;
        let max: f32 = pred.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min: f32 = pred.iter().cloned().fold(f32::INFINITY, f32::min);
        eprintln!("   t={}: mean={:.6}, min={:.6}, max={:.6}", ti, mean, min, max);
    }

    // ── 8. ROI analysis ───────────────────────────────────────────
    eprintln!("\n8. ROI analysis (synthetic labels)...");
    let n_rois = 10;
    let labels: Vec<String> = (0..20484).map(|i| format!("ROI_{}", i % n_rois)).collect();
    let roi_summary = plotting::summarize_by_roi(first_pred, &labels);
    let mut roi_vec: Vec<(&String, &f32)> = roi_summary.iter().collect();
    roi_vec.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    for (name, val) in roi_vec.iter().take(5) {
        eprintln!("   {}: {:.6}", name, val);
    }

    // ── Final ─────────────────────────────────────────────────────
    let total_ms = t0.elapsed().as_millis();
    eprintln!("\n=== COMPLETE ===");
    eprintln!("Total time: {}ms", total_ms);
    eprintln!("Output directory: {}", output_dir);

    // List all output files
    let entries: Vec<_> = std::fs::read_dir(&output_dir).unwrap()
        .filter_map(|e| e.ok())
        .collect();
    eprintln!("Output files ({}):", entries.len());
    for entry in &entries {
        let meta = entry.metadata().unwrap();
        eprintln!("   {} ({:.1} KB)", entry.file_name().to_string_lossy(), meta.len() as f64 / 1024.0);
    }
}
