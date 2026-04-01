//! Evaluation metrics for TRIBE v2 predictions.
//!
//! Mirrors the Python metrics:
//! - **Pearson correlation** (per-vertex and averaged)
//! - **Retrieval top-k accuracy**
//! - **MSE** (mean squared error)
//!
//! These metrics compare predicted brain activity against ground-truth fMRI data.
//! Predictions and ground truth should both have shape `[n_timesteps, n_vertices]`.

/// Compute Pearson correlation coefficient between two vectors.
///
/// Returns `None` if either vector has zero variance or they have different lengths.
pub fn pearson_r(x: &[f32], y: &[f32]) -> Option<f32> {
    if x.len() != y.len() || x.is_empty() {
        return None;
    }
    let n = x.len() as f64;
    let mean_x = x.iter().map(|&v| v as f64).sum::<f64>() / n;
    let mean_y = y.iter().map(|&v| v as f64).sum::<f64>() / n;

    let mut cov = 0.0f64;
    let mut var_x = 0.0f64;
    let mut var_y = 0.0f64;

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi as f64 - mean_x;
        let dy = yi as f64 - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < 1e-15 {
        return None;
    }
    Some((cov / denom) as f32)
}

/// Compute per-vertex Pearson correlation between predictions and ground truth.
///
/// `pred`: `[n_timesteps][n_vertices]` — predicted brain activity.
/// `truth`: `[n_timesteps][n_vertices]` — ground-truth fMRI data.
///
/// Returns a Vec of length `n_vertices` with the Pearson r for each vertex
/// across timesteps. Vertices with zero variance get r = 0.0.
pub fn pearson_per_vertex(pred: &[Vec<f32>], truth: &[Vec<f32>]) -> Vec<f32> {
    if pred.is_empty() || truth.is_empty() {
        return Vec::new();
    }
    let n_t = pred.len().min(truth.len());
    let n_v = pred[0].len().min(truth[0].len());

    let mut result = vec![0.0f32; n_v];

    for vi in 0..n_v {
        let x: Vec<f32> = (0..n_t).map(|ti| pred[ti][vi]).collect();
        let y: Vec<f32> = (0..n_t).map(|ti| truth[ti][vi]).collect();
        result[vi] = pearson_r(&x, &y).unwrap_or(0.0);
    }

    result
}

/// Compute mean Pearson correlation across all vertices.
///
/// This is the primary evaluation metric for brain encoding models.
pub fn mean_pearson(pred: &[Vec<f32>], truth: &[Vec<f32>]) -> f32 {
    let per_vertex = pearson_per_vertex(pred, truth);
    if per_vertex.is_empty() {
        return 0.0;
    }
    let valid: Vec<f32> = per_vertex.iter().filter(|&&v| v.is_finite()).copied().collect();
    if valid.is_empty() {
        return 0.0;
    }
    valid.iter().sum::<f32>() / valid.len() as f32
}

/// Compute median Pearson correlation across all vertices.
pub fn median_pearson(pred: &[Vec<f32>], truth: &[Vec<f32>]) -> f32 {
    let mut per_vertex: Vec<f32> = pearson_per_vertex(pred, truth)
        .into_iter()
        .filter(|v| v.is_finite())
        .collect();
    if per_vertex.is_empty() {
        return 0.0;
    }
    per_vertex.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = per_vertex.len() / 2;
    if per_vertex.len() % 2 == 0 {
        (per_vertex[mid - 1] + per_vertex[mid]) / 2.0
    } else {
        per_vertex[mid]
    }
}

/// Compute mean squared error between predictions and ground truth.
///
/// Averaged across all timesteps and vertices.
pub fn mse(pred: &[Vec<f32>], truth: &[Vec<f32>]) -> f32 {
    if pred.is_empty() || truth.is_empty() {
        return 0.0;
    }
    let n_t = pred.len().min(truth.len());
    let n_v = pred[0].len().min(truth[0].len());
    let mut total = 0.0f64;
    let mut count = 0usize;

    for ti in 0..n_t {
        for vi in 0..n_v {
            let diff = pred[ti][vi] as f64 - truth[ti][vi] as f64;
            total += diff * diff;
            count += 1;
        }
    }

    if count == 0 { 0.0 } else { (total / count as f64) as f32 }
}

/// Retrieval top-k accuracy.
///
/// For each timestep, check whether the correct timestep is in the top-k
/// most similar predictions (by cosine similarity across vertices).
///
/// `pred`: `[n_timesteps][n_vertices]`
/// `truth`: `[n_timesteps][n_vertices]`
/// `k`: number of top candidates to check.
///
/// Returns accuracy in [0, 1].
pub fn topk_accuracy(pred: &[Vec<f32>], truth: &[Vec<f32>], k: usize) -> f32 {
    let n = pred.len().min(truth.len());
    if n == 0 || k == 0 {
        return 0.0;
    }

    let mut correct = 0usize;

    // For each ground-truth timestep, find the top-k most similar predictions
    for ti in 0..n {
        let mut sims: Vec<(usize, f32)> = (0..n)
            .map(|pi| {
                let sim = cosine_similarity(&pred[pi], &truth[ti]);
                (pi, sim)
            })
            .collect();
        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if sims.iter().take(k).any(|(idx, _)| *idx == ti) {
            correct += 1;
        }
    }

    correct as f32 / n as f32
}

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;

    for i in 0..n {
        dot += a[i] as f64 * b[i] as f64;
        norm_a += (a[i] as f64) * (a[i] as f64);
        norm_b += (b[i] as f64) * (b[i] as f64);
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom < 1e-15 { 0.0 } else { (dot / denom) as f32 }
}

/// Format metrics as a readable report string.
pub fn format_metrics_report(
    mean_r: f32,
    median_r: f32,
    mse_val: f32,
    topk_acc: Option<(usize, f32)>,
    n_timesteps: usize,
    n_vertices: usize,
) -> String {
    let mut lines = vec![
        format!("Evaluation Metrics"),
        format!("{}", "=".repeat(45)),
        format!("  Timesteps:          {}", n_timesteps),
        format!("  Vertices:           {}", n_vertices),
        format!("  Mean Pearson r:     {:.6}", mean_r),
        format!("  Median Pearson r:   {:.6}", median_r),
        format!("  MSE:                {:.6}", mse_val),
    ];
    if let Some((k, acc)) = topk_acc {
        lines.push(format!("  Top-{} accuracy:    {:.4} ({:.1}%)", k, acc, acc * 100.0));
    }
    lines.join("\n")
}

/// Load ground-truth fMRI predictions from a binary f32 file.
///
/// File format: flat little-endian f32, shape [n_timesteps, n_vertices].
/// `n_vertices`: expected number of vertices per timestep (e.g., 20484).
///
/// Returns Vec of Vec<f32> — one inner Vec per timestep.
pub fn load_ground_truth(path: &str, n_vertices: usize) -> anyhow::Result<Vec<Vec<f32>>> {
    let bytes = std::fs::read(path)
        .map_err(|e| anyhow::anyhow!("failed to read ground truth: {}: {}", path, e))?;

    let data: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    if data.len() % n_vertices != 0 {
        anyhow::bail!(
            "Ground truth file has {} floats, not divisible by {} vertices",
            data.len(), n_vertices
        );
    }

    let n_timesteps = data.len() / n_vertices;
    let mut result = Vec::with_capacity(n_timesteps);
    for ti in 0..n_timesteps {
        let start = ti * n_vertices;
        result.push(data[start..start + n_vertices].to_vec());
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pearson_perfect_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let r = pearson_r(&x, &y).unwrap();
        assert!((r - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_pearson_negative_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let r = pearson_r(&x, &y).unwrap();
        assert!((r - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn test_pearson_zero_variance() {
        let x = vec![1.0, 1.0, 1.0];
        let y = vec![1.0, 2.0, 3.0];
        assert!(pearson_r(&x, &y).is_none());
    }

    #[test]
    fn test_mse_zero() {
        let pred = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let truth = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert!((mse(&pred, &truth)).abs() < 1e-6);
    }

    #[test]
    fn test_mse_nonzero() {
        let pred = vec![vec![1.0, 2.0]];
        let truth = vec![vec![2.0, 4.0]];
        // MSE = ((1)^2 + (2)^2) / 2 = 2.5
        assert!((mse(&pred, &truth) - 2.5).abs() < 1e-5);
    }

    #[test]
    fn test_topk_accuracy_perfect() {
        // Identical predictions → top-1 should be 100%
        let data = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let acc = topk_accuracy(&data, &data, 1);
        assert!((acc - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_mean_pearson() {
        // Same data → r = 1.0 for each vertex
        let pred = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let truth = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let r = mean_pearson(&pred, &truth);
        assert!((r - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-5);
    }
}
