//! Full numeric parity test: validates every output path in Rust matches Python.
//!
//! Tests:
//! 1. Forward pass output (existing — Pearson = 1.0 against Python reference)
//! 2. Per-timestep prediction unraveling matches Python layout
//! 3. Average prediction across time matches Python
//! 4. Evaluation metrics (Pearson, MSE) computed on same data match Python
//! 5. Correlation map matches Python per-vertex Pearson
//! 6. ROI summaries are consistent (sums match vertex data)
//! 7. Modality ablation produces distinct contribution maps
//!
//! Prerequisites:
//!   python3 scripts/generate_parity_refs.py
//!   python3 scripts/generate_full_parity_refs.py

use std::collections::BTreeMap;
use std::path::Path;
use tribev2::model::tribe::TribeV2;
use tribev2::tensor::Tensor;

const DATA_DIR: &str = "/Users/Shared/tribev2-rs/data";
const REFS_DIR: &str = "/Users/Shared/tribev2-rs/data/parity_refs";

fn refs_exist() -> bool {
    Path::new(&format!("{}/final_output.bin", REFS_DIR)).exists()
        && Path::new(&format!("{}/model.safetensors", DATA_DIR)).exists()
        && Path::new(&format!("{}/full_parity_stats.json", REFS_DIR)).exists()
}

fn load_ref_with_header(name: &str) -> Tensor {
    let path = format!("{}/{}", REFS_DIR, name);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("failed to read {}: {}", path, e));
    let ndims = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let mut shape = Vec::with_capacity(ndims);
    let mut offset = 4;
    for _ in 0..ndims {
        let d = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        shape.push(d);
        offset += 4;
    }
    let n_floats: usize = shape.iter().product();
    let mut data = Vec::with_capacity(n_floats);
    for i in 0..n_floats {
        let start = offset + i * 4;
        let v = f32::from_le_bytes(bytes[start..start + 4].try_into().unwrap());
        data.push(v);
    }
    Tensor::from_vec(data, shape)
}

fn load_flat_f32(name: &str) -> Vec<f32> {
    let path = format!("{}/{}", REFS_DIR, name);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("failed to read {}: {}", path, e));
    bytes.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn load_json(name: &str) -> serde_json::Value {
    let path = format!("{}/{}", REFS_DIR, name);
    let s = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("failed to read {}: {}", path, e));
    serde_json::from_str(&s).unwrap()
}

fn pearson(x: &[f32], y: &[f32]) -> f64 {
    let n = x.len().min(y.len());
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
    let denom = (vx * vy).sqrt();
    if denom < 1e-15 { 0.0 } else { cov / denom }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).abs()).fold(0.0f32, f32::max)
}

fn rmse(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let sum: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| {
        let d = x as f64 - y as f64;
        d * d
    }).sum();
    (sum / n as f64).sqrt()
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 1: Forward pass output matches Python reference
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_1_forward_pass_parity() {
    if !refs_exist() {
        eprintln!("SKIP: reference files not found");
        return;
    }
    eprintln!("\n══ TEST 1: Forward pass numeric parity ══\n");

    let model = TribeV2::from_pretrained(
        &format!("{}/config.yaml", DATA_DIR),
        &format!("{}/model.safetensors", DATA_DIR),
        Some(&format!("{}/build_args.json", DATA_DIR)),
    ).expect("failed to load model");

    let input_text = load_ref_with_header("input_text.bin");
    let input_audio = load_ref_with_header("input_audio.bin");
    let input_video = load_ref_with_header("input_video.bin");

    let mut features = BTreeMap::new();
    features.insert("text".to_string(), input_text);
    features.insert("audio".to_string(), input_audio);
    features.insert("video".to_string(), input_video);

    let rust_output = model.forward(&features, None, true);
    let ref_final = load_ref_with_header("final_output.bin");

    assert_eq!(rust_output.shape, ref_final.shape,
        "Shape mismatch: rust={:?} vs python={:?}", rust_output.shape, ref_final.shape);

    let r = pearson(&rust_output.data, &ref_final.data);
    let mad = max_abs_diff(&rust_output.data, &ref_final.data);
    let rms = rmse(&rust_output.data, &ref_final.data);

    eprintln!("  Pearson:  {:.10}", r);
    eprintln!("  Max abs:  {:.2e}", mad);
    eprintln!("  RMSE:     {:.2e}", rms);

    assert!(r > 0.999999, "Pearson {:.10} < 0.999999", r);
    assert!(mad < 1e-4, "Max abs diff {:.2e} >= 1e-4", mad);
    eprintln!("  ✅ PASS");
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 2: Per-timestep prediction layout matches Python
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_2_prediction_layout_parity() {
    if !refs_exist() {
        eprintln!("SKIP: reference files not found");
        return;
    }
    eprintln!("\n══ TEST 2: Per-timestep prediction layout ══\n");

    let model = TribeV2::from_pretrained(
        &format!("{}/config.yaml", DATA_DIR),
        &format!("{}/model.safetensors", DATA_DIR),
        Some(&format!("{}/build_args.json", DATA_DIR)),
    ).unwrap();

    let input_text = load_ref_with_header("input_text.bin");
    let input_audio = load_ref_with_header("input_audio.bin");
    let input_video = load_ref_with_header("input_video.bin");

    let mut features = BTreeMap::new();
    features.insert("text".to_string(), input_text);
    features.insert("audio".to_string(), input_audio);
    features.insert("video".to_string(), input_video);

    // Forward pass and unravel (same logic as CLI)
    let output = model.forward(&features, None, true);
    let n_out = output.shape[1];
    let n_t = output.shape[2];

    let predictions: Vec<Vec<f32>> = (0..n_t)
        .map(|ti| {
            (0..n_out).map(|di| output.data[di * n_t + ti]).collect()
        })
        .collect();

    // Load Python reference (flat [T*D])
    let ref_flat = load_flat_f32("predictions_flat.bin");
    let ref_n_t = 100;
    let ref_n_v = 20484;
    assert_eq!(ref_flat.len(), ref_n_t * ref_n_v);

    // Compare each timestep
    let mut total_err = 0.0f64;
    let mut max_err = 0.0f32;
    let mut count = 0usize;

    for ti in 0..ref_n_t.min(n_t) {
        for vi in 0..ref_n_v.min(n_out) {
            let rust_val = predictions[ti][vi];
            let py_val = ref_flat[ti * ref_n_v + vi];
            let err = (rust_val - py_val).abs();
            max_err = max_err.max(err);
            total_err += (err as f64) * (err as f64);
            count += 1;
        }
    }

    let rms = (total_err / count as f64).sqrt();
    eprintln!("  Timesteps: {}", n_t.min(ref_n_t));
    eprintln!("  Vertices: {}", n_out.min(ref_n_v));
    eprintln!("  Max abs diff: {:.2e}", max_err);
    eprintln!("  RMSE: {:.2e}", rms);

    assert!(max_err < 1e-4, "Max error {:.2e} >= 1e-4 — layout mismatch", max_err);
    eprintln!("  ✅ PASS");
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 3: Average prediction matches Python
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_3_average_prediction_parity() {
    if !refs_exist() {
        eprintln!("SKIP: reference files not found");
        return;
    }
    eprintln!("\n══ TEST 3: Average prediction parity ══\n");

    let model = TribeV2::from_pretrained(
        &format!("{}/config.yaml", DATA_DIR),
        &format!("{}/model.safetensors", DATA_DIR),
        Some(&format!("{}/build_args.json", DATA_DIR)),
    ).unwrap();

    let input_text = load_ref_with_header("input_text.bin");
    let input_audio = load_ref_with_header("input_audio.bin");
    let input_video = load_ref_with_header("input_video.bin");

    let mut features = BTreeMap::new();
    features.insert("text".to_string(), input_text);
    features.insert("audio".to_string(), input_audio);
    features.insert("video".to_string(), input_video);

    let output = model.forward(&features, None, true);
    let n_out = output.shape[1];
    let n_t = output.shape[2];

    // Average across timesteps: for each vertex, mean over T
    let mut avg_pred = vec![0.0f32; n_out];
    for di in 0..n_out {
        let base = di * n_t;
        let sum: f32 = output.data[base..base + n_t].iter().sum();
        avg_pred[di] = sum / n_t as f32;
    }

    let ref_avg = load_flat_f32("avg_prediction.bin");

    let r = pearson(&avg_pred, &ref_avg);
    let mad = max_abs_diff(&avg_pred, &ref_avg);

    eprintln!("  Pearson:  {:.10}", r);
    eprintln!("  Max abs:  {:.2e}", mad);

    // Load stats for cross-check
    let stats = load_json("full_parity_stats.json");
    let py_mean = stats["avg_prediction"]["mean"].as_f64().unwrap();
    let rust_mean: f64 = avg_pred.iter().map(|&v| v as f64).sum::<f64>() / avg_pred.len() as f64;
    eprintln!("  Python mean:  {:.8}", py_mean);
    eprintln!("  Rust mean:    {:.8}", rust_mean);
    eprintln!("  Mean diff:    {:.2e}", (py_mean - rust_mean).abs());

    assert!(r > 0.999999, "Pearson {:.10} < 0.999999", r);
    assert!(mad < 1e-4, "Max abs diff {:.2e} >= 1e-4", mad);
    assert!((py_mean - rust_mean).abs() < 1e-5,
        "Mean prediction diff {:.2e} >= 1e-5", (py_mean - rust_mean).abs());
    eprintln!("  ✅ PASS");
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 4: Evaluation metrics match Python
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_4_metrics_parity() {
    if !refs_exist() {
        eprintln!("SKIP: reference files not found");
        return;
    }
    eprintln!("\n══ TEST 4: Evaluation metrics parity ══\n");

    // Load predictions and ground truth
    let pred_flat = load_flat_f32("predictions_flat.bin");
    let gt_flat = load_flat_f32("ground_truth.bin");
    let py_metrics = load_json("metrics.json");

    let n_t = py_metrics["n_timesteps"].as_u64().unwrap() as usize;
    let n_v = py_metrics["n_vertices"].as_u64().unwrap() as usize;

    // Reshape to Vec<Vec<f32>>
    let predictions: Vec<Vec<f32>> = (0..n_t)
        .map(|ti| pred_flat[ti * n_v..(ti + 1) * n_v].to_vec())
        .collect();
    let truth: Vec<Vec<f32>> = (0..n_t)
        .map(|ti| gt_flat[ti * n_v..(ti + 1) * n_v].to_vec())
        .collect();

    // Rust metrics
    let rust_mean_r = tribev2::metrics::mean_pearson(&predictions, &truth);
    let rust_median_r = tribev2::metrics::median_pearson(&predictions, &truth);
    let rust_mse = tribev2::metrics::mse(&predictions, &truth);

    let py_mean_r = py_metrics["mean_pearson"].as_f64().unwrap() as f32;
    let py_median_r = py_metrics["median_pearson"].as_f64().unwrap() as f32;
    let py_mse = py_metrics["mse"].as_f64().unwrap() as f32;

    eprintln!("  Mean Pearson r:   Rust={:.8} Python={:.8} diff={:.2e}",
        rust_mean_r, py_mean_r, (rust_mean_r - py_mean_r).abs());
    eprintln!("  Median Pearson r: Rust={:.8} Python={:.8} diff={:.2e}",
        rust_median_r, py_median_r, (rust_median_r - py_median_r).abs());
    eprintln!("  MSE:              Rust={:.8} Python={:.8} diff={:.2e}",
        rust_mse, py_mse, (rust_mse - py_mse).abs());

    assert!((rust_mean_r - py_mean_r).abs() < 1e-4,
        "Mean Pearson diff {:.2e}", (rust_mean_r - py_mean_r).abs());
    assert!((rust_median_r - py_median_r).abs() < 1e-4,
        "Median Pearson diff {:.2e}", (rust_median_r - py_median_r).abs());
    assert!((rust_mse - py_mse).abs() < 1e-6,
        "MSE diff {:.2e}", (rust_mse - py_mse).abs());
    eprintln!("  ✅ PASS");
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 5: Correlation map matches Python
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_5_correlation_map_parity() {
    if !refs_exist() {
        eprintln!("SKIP: reference files not found");
        return;
    }
    eprintln!("\n══ TEST 5: Per-vertex correlation map parity ══\n");

    let pred_flat = load_flat_f32("predictions_flat.bin");
    let gt_flat = load_flat_f32("ground_truth.bin");
    let ref_corr = load_flat_f32("correlation_map.bin");
    let py_metrics = load_json("metrics.json");

    let n_t = py_metrics["n_timesteps"].as_u64().unwrap() as usize;
    let n_v = py_metrics["n_vertices"].as_u64().unwrap() as usize;

    let predictions: Vec<Vec<f32>> = (0..n_t)
        .map(|ti| pred_flat[ti * n_v..(ti + 1) * n_v].to_vec())
        .collect();
    let truth: Vec<Vec<f32>> = (0..n_t)
        .map(|ti| gt_flat[ti * n_v..(ti + 1) * n_v].to_vec())
        .collect();

    let rust_corr = tribev2::metrics::pearson_per_vertex(&predictions, &truth);

    assert_eq!(rust_corr.len(), ref_corr.len(),
        "Length mismatch: rust={} python={}", rust_corr.len(), ref_corr.len());

    let r = pearson(&rust_corr, &ref_corr);
    let mad = max_abs_diff(&rust_corr, &ref_corr);
    let rms = rmse(&rust_corr, &ref_corr);

    eprintln!("  Correlation map Pearson:  {:.10}", r);
    eprintln!("  Max abs diff:            {:.2e}", mad);
    eprintln!("  RMSE:                    {:.2e}", rms);

    assert!(r > 0.9999, "Correlation map Pearson {:.10} < 0.9999", r);
    assert!(mad < 1e-3, "Max abs diff {:.2e} >= 1e-3", mad);
    eprintln!("  ✅ PASS");
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 6: ROI summaries are consistent with vertex data
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_6_roi_consistency() {
    if !refs_exist() {
        eprintln!("SKIP: reference files not found");
        return;
    }
    eprintln!("\n══ TEST 6: ROI summary consistency ══\n");

    let ref_avg = load_flat_f32("avg_prediction.bin");

    // Compute ROI summary using our module
    let roi_summary = tribev2::roi::summarize_by_roi(&ref_avg, None);

    eprintln!("  Number of ROIs: {}", roi_summary.len());

    // Verify: for each ROI, the mean matches manual computation from vertices
    let labels = tribev2::roi::get_hcp_labels(None);
    let mut max_roi_diff = 0.0f32;

    for (name, vertices) in &labels {
        if vertices.is_empty() { continue; }
        let manual_mean: f32 = vertices.iter()
            .filter_map(|&vi| ref_avg.get(vi))
            .sum::<f32>() / vertices.iter().filter(|&&vi| vi < ref_avg.len()).count() as f32;

        if let Some(&roi_mean) = roi_summary.get(name) {
            let diff = (manual_mean - roi_mean).abs();
            max_roi_diff = max_roi_diff.max(diff);
            if diff > 1e-6 {
                eprintln!("  WARNING: ROI {} manual={:.8} roi={:.8} diff={:.2e}",
                    name, manual_mean, roi_mean, diff);
            }
        }
    }

    eprintln!("  Max ROI mean diff: {:.2e}", max_roi_diff);
    assert!(max_roi_diff < 1e-6, "ROI mean computation has errors");

    // Top-k should be sorted
    let topk = tribev2::roi::get_topk_rois(&ref_avg, 10, None);
    eprintln!("  Top-10 ROIs:");
    for (i, (name, val)) in topk.iter().enumerate() {
        eprintln!("    {}: {} = {:.6}", i + 1, name, val);
    }

    // Verify descending order
    for w in topk.windows(2) {
        assert!(w[0].1 >= w[1].1, "Top-k not sorted: {} ({}) < {} ({})",
            w[0].0, w[0].1, w[1].0, w[1].1);
    }

    eprintln!("  ✅ PASS");
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 7: Modality ablation produces distinct maps
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_7_modality_ablation() {
    if !refs_exist() {
        eprintln!("SKIP: reference files not found");
        return;
    }
    eprintln!("\n══ TEST 7: Modality ablation ══\n");

    let model = TribeV2::from_pretrained(
        &format!("{}/config.yaml", DATA_DIR),
        &format!("{}/model.safetensors", DATA_DIR),
        Some(&format!("{}/build_args.json", DATA_DIR)),
    ).unwrap();

    let input_text = load_ref_with_header("input_text.bin");
    let input_audio = load_ref_with_header("input_audio.bin");
    let input_video = load_ref_with_header("input_video.bin");

    let mut features = BTreeMap::new();
    features.insert("text".to_string(), input_text);
    features.insert("audio".to_string(), input_audio);
    features.insert("video".to_string(), input_video);

    let contributions = model.modality_ablation(&features, None);

    eprintln!("  Modality contributions:");
    let mut all_norms = Vec::new();
    for (name, contrib) in &contributions {
        let mean: f32 = contrib.iter().sum::<f32>() / contrib.len() as f32;
        let max: f32 = contrib.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let norm: f32 = contrib.iter().map(|v| v * v).sum::<f32>().sqrt();
        eprintln!("    {}: mean={:.6}, max={:.6}, norm={:.4}", name, mean, max, norm);
        all_norms.push((name.clone(), norm));
    }

    // Verify: each modality should have nonzero contribution
    for (name, norm) in &all_norms {
        assert!(*norm > 0.0, "Modality {} has zero contribution", name);
    }

    // Verify: contributions should differ between modalities
    if all_norms.len() >= 2 {
        let r = pearson(
            &contributions[&all_norms[0].0],
            &contributions[&all_norms[1].0],
        );
        eprintln!("  Correlation between {} and {}: {:.6}",
            all_norms[0].0, all_norms[1].0, r);
        // They shouldn't be identical (r < 1.0)
        assert!(r < 0.999, "Modality contributions are too similar (r={:.6})", r);
    }

    eprintln!("  ✅ PASS");
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 8: Intermediate stage parity (projectors + concatenation)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_8_intermediate_stages() {
    if !refs_exist() {
        eprintln!("SKIP: reference files not found");
        return;
    }
    eprintln!("\n══ TEST 8: Intermediate stage parity ══\n");

    let model = TribeV2::from_pretrained(
        &format!("{}/config.yaml", DATA_DIR),
        &format!("{}/model.safetensors", DATA_DIR),
        Some(&format!("{}/build_args.json", DATA_DIR)),
    ).unwrap();

    let input_text = load_ref_with_header("input_text.bin");
    let input_audio = load_ref_with_header("input_audio.bin");
    let input_video = load_ref_with_header("input_video.bin");

    let mut features = BTreeMap::new();
    features.insert("text".to_string(), input_text);
    features.insert("audio".to_string(), input_audio);
    features.insert("video".to_string(), input_video);

    // Test aggregate_features (after projectors + concat)
    let agg = model.aggregate_features(&features);
    let ref_cat = load_ref_with_header("after_cat.bin");

    let r = pearson(&agg.data, &ref_cat.data);
    let mad = max_abs_diff(&agg.data, &ref_cat.data);

    eprintln!("  after_cat shape: rust={:?} python={:?}", agg.shape, ref_cat.shape);
    eprintln!("  Pearson:  {:.10}", r);
    eprintln!("  Max abs:  {:.2e}", mad);

    assert_eq!(agg.shape, ref_cat.shape);
    assert!(r > 0.999999, "after_cat Pearson {:.10} < 0.999999", r);
    assert!(mad < 1e-4, "after_cat max abs diff {:.2e} >= 1e-4", mad);
    eprintln!("  ✅ PASS");
}
