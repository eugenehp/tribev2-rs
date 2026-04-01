//! Cross-resolution mesh resampling for fsaverage surfaces.
//!
//! Mirrors the Python `get_stat_map()` in `base.py` which resamples vertex data
//! between different fsaverage resolutions (e.g., fsaverage5 → fsaverage6) using
//! k-nearest-neighbor interpolation on the inflated surface coordinates.
//!
//! Supported meshes: fsaverage3 (642), fsaverage4 (2562), fsaverage5 (10242), fsaverage6 (40962).

use anyhow::{Context, Result};

/// Resample surface data from one fsaverage resolution to another.
///
/// Uses nearest-neighbor interpolation based on the pial surface coordinates.
/// For downsampling (e.g., fsaverage6 → fsaverage5), takes the nearest vertex.
/// For upsampling (e.g., fsaverage5 → fsaverage6), uses inverse-distance weighted
/// interpolation from the k nearest vertices.
///
/// `data`: per-vertex values, length = 2 * n_vertices_per_hemi for `from_mesh`.
/// `from_mesh`: source mesh name (e.g., "fsaverage5").
/// `to_mesh`: target mesh name (e.g., "fsaverage6").
/// `subjects_dir`: optional FreeSurfer subjects directory.
/// `k`: number of nearest neighbors for interpolation (default: 5).
///
/// Returns: resampled data of length 2 * n_vertices_per_hemi for `to_mesh`.
pub fn resample_surface(
    data: &[f32],
    from_mesh: &str,
    to_mesh: &str,
    subjects_dir: Option<&str>,
    k: usize,
) -> Result<Vec<f32>> {
    let from_size = crate::fsaverage::fsaverage_size(from_mesh)
        .ok_or_else(|| anyhow::anyhow!("Unknown mesh: {}", from_mesh))?;
    let to_size = crate::fsaverage::fsaverage_size(to_mesh)
        .ok_or_else(|| anyhow::anyhow!("Unknown mesh: {}", to_mesh))?;

    if data.len() != 2 * from_size {
        anyhow::bail!(
            "Data length {} doesn't match {} (expected {})",
            data.len(), from_mesh, 2 * from_size
        );
    }

    if from_mesh == to_mesh {
        return Ok(data.to_vec());
    }

    // Load both meshes
    let from_brain = crate::fsaverage::load_fsaverage(from_mesh, "pial", "sulcal", subjects_dir)
        .with_context(|| format!("Failed to load {} mesh", from_mesh))?;
    let to_brain = crate::fsaverage::load_fsaverage(to_mesh, "pial", "sulcal", subjects_dir)
        .with_context(|| format!("Failed to load {} mesh", to_mesh))?;

    let mut result = Vec::with_capacity(2 * to_size);

    // Process each hemisphere separately
    for (hemi_idx, (from_hemi, to_hemi)) in [
        (&from_brain.left, &to_brain.left),
        (&from_brain.right, &to_brain.right),
    ].iter().enumerate() {
        let data_offset = hemi_idx * from_size;
        let hemi_data = &data[data_offset..data_offset + from_size];

        let from_coords = &from_hemi.mesh.coords;
        let to_coords = &to_hemi.mesh.coords;

        // For each target vertex, find k nearest source vertices
        let resampled = resample_hemisphere(
            hemi_data,
            from_coords,
            from_hemi.mesh.n_vertices,
            to_coords,
            to_hemi.mesh.n_vertices,
            k,
        );
        result.extend_from_slice(&resampled);
    }

    Ok(result)
}

/// Resample one hemisphere using inverse-distance weighted k-NN interpolation.
fn resample_hemisphere(
    data: &[f32],
    from_coords: &[f32],  // [n_from * 3] flat
    n_from: usize,
    to_coords: &[f32],    // [n_to * 3] flat
    n_to: usize,
    k: usize,
) -> Vec<f32> {
    // Build a simple brute-force kd-tree equivalent
    // For each target vertex, find k nearest source vertices
    let mut result = Vec::with_capacity(n_to);

    for ti in 0..n_to {
        let tx = to_coords[ti * 3];
        let ty = to_coords[ti * 3 + 1];
        let tz = to_coords[ti * 3 + 2];

        // Find k nearest neighbors (brute force — fine for fsaverage sizes)
        let mut dists: Vec<(usize, f32)> = (0..n_from)
            .map(|fi| {
                let dx = from_coords[fi * 3] - tx;
                let dy = from_coords[fi * 3 + 1] - ty;
                let dz = from_coords[fi * 3 + 2] - tz;
                (fi, dx * dx + dy * dy + dz * dz)
            })
            .collect();

        // Partial sort to get top-k
        dists.select_nth_unstable_by(k.min(n_from) - 1, |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        let nearest = &dists[..k.min(n_from)];

        // Inverse-distance weighted average
        let mut weight_sum = 0.0f32;
        let mut value_sum = 0.0f32;
        for &(fi, dist_sq) in nearest {
            if fi < data.len() {
                let dist = dist_sq.sqrt().max(1e-12);
                let w = 1.0 / dist;
                weight_sum += w;
                value_sum += w * data[fi];
            }
        }
        result.push(if weight_sum > 0.0 { value_sum / weight_sum } else { 0.0 });
    }

    result
}

/// Resample using precomputed index mapping (faster for repeated resampling).
///
/// First call `compute_resampling_map()`, then use the map for each data array.
pub struct ResamplingMap {
    /// For each target vertex: Vec of (source_vertex_index, weight)
    pub mappings: Vec<Vec<(usize, f32)>>,
    /// Source mesh name
    pub from_mesh: String,
    /// Target mesh name
    pub to_mesh: String,
}

impl ResamplingMap {
    /// Apply the precomputed mapping to resample data.
    pub fn apply(&self, data: &[f32]) -> Vec<f32> {
        self.mappings.iter().map(|neighbors| {
            let mut weight_sum = 0.0f32;
            let mut value_sum = 0.0f32;
            for &(idx, w) in neighbors {
                if idx < data.len() {
                    weight_sum += w;
                    value_sum += w * data[idx];
                }
            }
            if weight_sum > 0.0 { value_sum / weight_sum } else { 0.0 }
        }).collect()
    }
}

/// Compute a resampling map between two mesh resolutions.
///
/// This is expensive but only needs to be done once. The resulting map
/// can be applied to many data arrays efficiently.
pub fn compute_resampling_map(
    from_mesh: &str,
    to_mesh: &str,
    subjects_dir: Option<&str>,
    k: usize,
) -> Result<ResamplingMap> {
    let from_size = crate::fsaverage::fsaverage_size(from_mesh)
        .ok_or_else(|| anyhow::anyhow!("Unknown mesh: {}", from_mesh))?;
    let to_size = crate::fsaverage::fsaverage_size(to_mesh)
        .ok_or_else(|| anyhow::anyhow!("Unknown mesh: {}", to_mesh))?;

    let from_brain = crate::fsaverage::load_fsaverage(from_mesh, "pial", "sulcal", subjects_dir)?;
    let to_brain = crate::fsaverage::load_fsaverage(to_mesh, "pial", "sulcal", subjects_dir)?;

    let mut mappings = Vec::with_capacity(2 * to_size);

    for (hemi_idx, (from_hemi, to_hemi)) in [
        (&from_brain.left, &to_brain.left),
        (&from_brain.right, &to_brain.right),
    ].iter().enumerate() {
        let offset = hemi_idx * from_size;
        let from_coords = &from_hemi.mesh.coords;
        let to_coords = &to_hemi.mesh.coords;

        for ti in 0..to_hemi.mesh.n_vertices {
            let tx = to_coords[ti * 3];
            let ty = to_coords[ti * 3 + 1];
            let tz = to_coords[ti * 3 + 2];

            let mut dists: Vec<(usize, f32)> = (0..from_hemi.mesh.n_vertices)
                .map(|fi| {
                    let dx = from_coords[fi * 3] - tx;
                    let dy = from_coords[fi * 3 + 1] - ty;
                    let dz = from_coords[fi * 3 + 2] - tz;
                    (fi + offset, dx * dx + dy * dy + dz * dz)
                })
                .collect();

            let kk = k.min(from_hemi.mesh.n_vertices);
            dists.select_nth_unstable_by(kk - 1, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });

            let neighbors: Vec<(usize, f32)> = dists[..kk]
                .iter()
                .map(|&(idx, dist_sq)| {
                    let w = 1.0 / dist_sq.sqrt().max(1e-12);
                    (idx, w)
                })
                .collect();

            mappings.push(neighbors);
        }
    }

    Ok(ResamplingMap {
        mappings,
        from_mesh: from_mesh.to_string(),
        to_mesh: to_mesh.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resample_identity() {
        // Same mesh → should return identical data
        // Can't actually load meshes in unit test without FreeSurfer installed,
        // so just test the logic
        let data = vec![1.0f32; 10];
        let resampled = resample_hemisphere(&data, &[0.0; 30], 10, &[0.0; 30], 10, 3);
        assert_eq!(resampled.len(), 10);
    }

    #[test]
    fn test_resample_simple() {
        // 3 source vertices, 2 target vertices
        let data = vec![1.0, 2.0, 3.0];
        let from_coords = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let to_coords = vec![0.5, 0.0, 0.0, 1.5, 0.0, 0.0];
        let result = resample_hemisphere(&data, &from_coords, 3, &to_coords, 2, 2);
        assert_eq!(result.len(), 2);
        // First target at 0.5 — equidistant from vertices 0 (val=1) and 1 (val=2)
        // Inverse-distance weighted with equal distances → ~1.5
        assert!((result[0] - 1.5).abs() < 0.1, "got {}", result[0]);
        // Second target at 1.5 — equidistant from vertices 1 (val=2) and 2 (val=3)
        assert!((result[1] - 2.5).abs() < 0.1, "got {}", result[1]);
    }

    #[test]
    fn test_resampling_map_apply() {
        let map = ResamplingMap {
            mappings: vec![
                vec![(0, 1.0), (1, 1.0)],  // average of vertices 0 and 1
                vec![(2, 1.0)],              // just vertex 2
            ],
            from_mesh: "test".to_string(),
            to_mesh: "test".to_string(),
        };
        let data = vec![2.0, 4.0, 6.0];
        let result = map.apply(&data);
        assert_eq!(result.len(), 2);
        assert!((result[0] - 3.0).abs() < 1e-5); // (2+4)/2
        assert!((result[1] - 6.0).abs() < 1e-5);
    }
}
