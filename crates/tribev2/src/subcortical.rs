//! Subcortical brain structure analysis for TRIBE v2.
//!
//! Mirrors the Python `subcortical.py` module. The Python version uses the
//! Harvard-Oxford subcortical atlas to define structures like hippocampus,
//! amygdala, thalamus, caudate, putamen, etc.
//!
//! **Important:** Subcortical predictions require a model trained with a
//! `MaskProjector` (subcortical mask) instead of the default `SurfaceProjector`.
//! The pretrained cortical model predicts fsaverage5 surface vertices, not
//! subcortical voxels. A separate subcortical checkpoint is needed.
//!
//! This module provides:
//! - Subcortical structure definitions (Harvard-Oxford atlas labels)
//! - Mapping between structure names and prediction indices
//! - Summary statistics per structure
//! - Utilities for identifying subcortical regions

use std::collections::HashMap;

/// Harvard-Oxford subcortical structure labels (bilateral).
///
/// These are the standard subcortical ROIs from the Harvard-Oxford atlas,
/// excluding cortex, white matter, and brain stem.
pub const SUBCORTICAL_LABELS: &[&str] = &[
    "Left Cerebral White Matter",
    "Left Cerebral Cortex",
    "Left Lateral Ventricle",
    "Left Thalamus",
    "Left Caudate",
    "Left Putamen",
    "Left Pallidum",
    "Brain-Stem",
    "Left Hippocampus",
    "Left Amygdala",
    "Left Accumbens",
    "Right Cerebral White Matter",
    "Right Cerebral Cortex",
    "Right Lateral Ventricle",
    "Right Thalamus",
    "Right Caudate",
    "Right Putamen",
    "Right Pallidum",
    "Right Hippocampus",
    "Right Amygdala",
    "Right Accumbens",
];

/// Subcortical structure labels (excluding cortex, white matter, stem, ventricles).
/// These are the structures typically of interest for subcortical analysis.
pub const SUBCORTICAL_STRUCTURES: &[&str] = &[
    "Thalamus",
    "Caudate",
    "Putamen",
    "Pallidum",
    "Hippocampus",
    "Amygdala",
    "Accumbens",
];

/// Configuration for subcortical analysis.
#[derive(Debug, Clone)]
pub struct SubcorticalConfig {
    /// Atlas resolution: "1mm" or "2mm".
    pub resolution: String,
    /// Whether to merge left/right hemispheres.
    pub merge_hemispheres: bool,
    /// Structures to include (None = all).
    pub structures: Option<Vec<String>>,
}

impl Default for SubcorticalConfig {
    fn default() -> Self {
        Self {
            resolution: "2mm".to_string(),
            merge_hemispheres: true,
            structures: None,
        }
    }
}

/// Get the list of subcortical structure labels.
///
/// If `with_hemi` is true, returns "Left Thalamus", "Right Thalamus", etc.
/// If false, returns just "Thalamus", "Caudate", etc.
pub fn get_subcortical_labels(with_hemi: bool) -> Vec<String> {
    if with_hemi {
        let mut labels = Vec::new();
        for &structure in SUBCORTICAL_STRUCTURES {
            labels.push(format!("Left {}", structure));
            labels.push(format!("Right {}", structure));
        }
        labels
    } else {
        SUBCORTICAL_STRUCTURES.iter().map(|s| s.to_string()).collect()
    }
}

/// Approximate voxel index ranges for subcortical structures.
///
/// When using the Harvard-Oxford atlas at 2mm resolution, the subcortical mask
/// has approximately 5,500 voxels. This provides approximate index ranges
/// for each structure.
///
/// For exact indices, the Harvard-Oxford atlas NIfTI file must be loaded.
///
/// Returns: HashMap<label, (start_index, end_index)>
pub fn get_subcortical_roi_ranges() -> HashMap<String, (usize, usize)> {
    // Approximate voxel counts from Harvard-Oxford subcortical atlas at 2mm
    // Total subcortical voxels ≈ 5500
    let structure_sizes: Vec<(&str, usize)> = vec![
        ("Left Thalamus", 550),
        ("Left Caudate", 300),
        ("Left Putamen", 350),
        ("Left Pallidum", 100),
        ("Left Hippocampus", 300),
        ("Left Amygdala", 120),
        ("Left Accumbens", 60),
        ("Right Thalamus", 550),
        ("Right Caudate", 300),
        ("Right Putamen", 350),
        ("Right Pallidum", 100),
        ("Right Hippocampus", 300),
        ("Right Amygdala", 120),
        ("Right Accumbens", 60),
    ];

    let mut ranges = HashMap::new();
    let mut offset = 0;
    for (name, size) in structure_sizes {
        ranges.insert(name.to_string(), (offset, offset + size));
        offset += size;
    }
    ranges
}

/// Get voxel indices for a subcortical structure.
///
/// `roi`: structure name, e.g., "Thalamus" (merges left+right),
///        "Left Hippocampus", or "Right Amygdala".
pub fn get_subcortical_roi_indices(roi: &str) -> Vec<usize> {
    let ranges = get_subcortical_roi_ranges();

    // Check for exact match first
    if let Some(&(start, end)) = ranges.get(roi) {
        return (start..end).collect();
    }

    // Check if it's a merged (no hemisphere) name
    let left_key = format!("Left {}", roi);
    let right_key = format!("Right {}", roi);
    let mut indices = Vec::new();
    if let Some(&(start, end)) = ranges.get(&left_key) {
        indices.extend(start..end);
    }
    if let Some(&(start, end)) = ranges.get(&right_key) {
        indices.extend(start..end);
    }

    indices
}

/// Summarize subcortical predictions per structure.
///
/// `voxel_scores`: per-voxel prediction values from a subcortical model.
/// `config`: subcortical configuration.
///
/// Returns: HashMap<structure_name, mean_value>
pub fn summarize_subcortical(
    voxel_scores: &[f32],
    config: &SubcorticalConfig,
) -> HashMap<String, f32> {
    let mut summary = HashMap::new();

    let structures = if let Some(ref s) = config.structures {
        s.clone()
    } else if config.merge_hemispheres {
        get_subcortical_labels(false)
    } else {
        get_subcortical_labels(true)
    };

    for structure in &structures {
        let indices = get_subcortical_roi_indices(structure);
        if indices.is_empty() {
            continue;
        }
        let sum: f32 = indices.iter()
            .filter_map(|&i| voxel_scores.get(i))
            .sum();
        let count = indices.iter().filter(|&&i| i < voxel_scores.len()).count();
        if count > 0 {
            summary.insert(structure.clone(), sum / count as f32);
        }
    }

    summary
}

/// Format subcortical summary as a readable table.
pub fn format_subcortical_table(summary: &HashMap<String, f32>) -> String {
    let mut sorted: Vec<(&String, &f32)> = summary.iter().collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut lines = vec![
        format!("{:<25} {:>12}", "Structure", "Activation"),
        format!("{}", "-".repeat(39)),
    ];
    for (name, val) in sorted {
        lines.push(format!("{:<25} {:>12.6}", name, val));
    }
    lines.join("\n")
}

/// Serialize subcortical summary to JSON.
pub fn subcortical_to_json(summary: &HashMap<String, f32>) -> String {
    let mut sorted: Vec<(&String, &f32)> = summary.iter().collect();
    sorted.sort_by(|(a, _), (b, _)| a.cmp(b));

    let entries: Vec<String> = sorted.iter()
        .map(|(k, v)| format!("  \"{}\": {:.6}", k, v))
        .collect();
    format!("{{\n{}\n}}", entries.join(",\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subcortical_labels() {
        let labels = get_subcortical_labels(false);
        assert_eq!(labels.len(), 7);
        assert!(labels.contains(&"Thalamus".to_string()));

        let labels_hemi = get_subcortical_labels(true);
        assert_eq!(labels_hemi.len(), 14);
        assert!(labels_hemi.contains(&"Left Thalamus".to_string()));
        assert!(labels_hemi.contains(&"Right Thalamus".to_string()));
    }

    #[test]
    fn test_subcortical_roi_ranges() {
        let ranges = get_subcortical_roi_ranges();
        assert!(ranges.contains_key("Left Thalamus"));
        assert!(ranges.contains_key("Right Hippocampus"));

        // Verify non-overlapping
        let mut all_indices: Vec<(usize, usize)> = ranges.values().copied().collect();
        all_indices.sort_by_key(|r| r.0);
        for w in all_indices.windows(2) {
            assert!(w[0].1 <= w[1].0, "Overlapping ranges: {:?} and {:?}", w[0], w[1]);
        }
    }

    #[test]
    fn test_get_roi_indices_merged() {
        let indices = get_subcortical_roi_indices("Thalamus");
        let left = get_subcortical_roi_indices("Left Thalamus");
        let right = get_subcortical_roi_indices("Right Thalamus");
        assert_eq!(indices.len(), left.len() + right.len());
    }

    #[test]
    fn test_summarize_subcortical() {
        let scores = vec![1.0f32; 5000];
        let config = SubcorticalConfig::default();
        let summary = summarize_subcortical(&scores, &config);
        assert!(!summary.is_empty());
        for (_, val) in &summary {
            assert!((val - 1.0).abs() < 1e-5);
        }
    }
}
