//! Shared numeric parity metrics for RLX / Burn / Rust comparisons.

/// Tolerance for "100%" parity (matches Python f32 reference pipeline).
pub const PARITY_PEARSON_MIN: f64 = 0.999999;
pub const PARITY_RMSE_MAX: f64 = 1e-5;
pub const PARITY_MAX_ABS_MAX: f32 = 1e-4;
pub const PARITY_COSINE_DIST_MAX: f64 = 1e-6;

#[derive(Debug, Clone, Copy)]
pub struct ParityReport {
    pub pearson: f64,
    pub rmse: f64,
    pub max_abs: f32,
    pub cosine_sim: f64,
    /// `1 - cosine_sim` (0 = identical direction, scale-invariant).
    pub cosine_dist: f64,
}

pub fn pearson(x: &[f32], y: &[f32]) -> f64 {
    let n = x.len().min(y.len());
    if n == 0 {
        return 1.0;
    }
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
    if denom < 1e-30 {
        1.0
    } else {
        cov / denom
    }
}

pub fn rmse(x: &[f32], y: &[f32]) -> f64 {
    let n = x.len().min(y.len());
    if n == 0 {
        return 0.0;
    }
    let sum: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(&a, &b)| {
            let d = a as f64 - b as f64;
            d * d
        })
        .sum();
    (sum / n as f64).sqrt()
}

pub fn cosine_similarity(x: &[f32], y: &[f32]) -> f64 {
    let n = x.len().min(y.len());
    if n == 0 {
        return 1.0;
    }
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..n {
        let a = x[i] as f64;
        let b = y[i] as f64;
        dot += a * b;
        na += a * a;
        nb += b * b;
    }
    let denom = (na * nb).sqrt();
    if denom < 1e-30 {
        1.0
    } else {
        dot / denom
    }
}

pub fn compare_slices(x: &[f32], y: &[f32]) -> ParityReport {
    assert_eq!(x.len(), y.len(), "length mismatch: {} vs {}", x.len(), y.len());
    let pearson = pearson(x, y);
    let cosine_sim = cosine_similarity(x, y);
    ParityReport {
        pearson,
        rmse: rmse(x, y),
        max_abs: x
            .iter()
            .zip(y.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max),
        cosine_sim,
        cosine_dist: 1.0 - cosine_sim,
    }
}

/// Log the largest absolute diffs (for debugging GPU divergence).
pub fn log_top_diffs(x: &[f32], y: &[f32], label: &str, n: usize) {
    let mut errs: Vec<(usize, f32)> = x
        .iter()
        .zip(y.iter())
        .enumerate()
        .map(|(i, (&a, &b))| (i, (a - b).abs()))
        .collect();
    errs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    eprintln!("{label}: top {n} diffs");
    for (i, e) in errs.into_iter().take(n) {
        eprintln!("  idx={i} a={:.6} b={:.6} |a-b|={e:.6}", x[i], y[i]);
    }
}

impl ParityReport {
    pub fn log(&self, label: &str) {
        eprintln!(
            "{label}: pearson={:.10} rmse={:.2e} max_abs={:.2e} cosine_sim={:.10} cosine_dist={:.2e}",
            self.pearson, self.rmse, self.max_abs, self.cosine_sim, self.cosine_dist
        );
    }

    pub fn assert_full_parity(&self, label: &str) {
        self.log(label);
        assert!(
            self.pearson >= PARITY_PEARSON_MIN,
            "{label}: pearson {:.10} < {PARITY_PEARSON_MIN}",
            self.pearson
        );
        assert!(
            self.rmse <= PARITY_RMSE_MAX,
            "{label}: rmse {:.2e} > {PARITY_RMSE_MAX:.2e}",
            self.rmse
        );
        assert!(
            self.max_abs <= PARITY_MAX_ABS_MAX,
            "{label}: max_abs {:.2e} > {PARITY_MAX_ABS_MAX:.2e}",
            self.max_abs
        );
        assert!(
            self.cosine_dist <= PARITY_COSINE_DIST_MAX,
            "{label}: cosine_dist {:.2e} > {PARITY_COSINE_DIST_MAX:.2e}",
            self.cosine_dist
        );
    }
}
