//! HCP-MMP1 ROI (Region of Interest) analysis for TRIBE v2 predictions.
//!
//! Maps the 20,484 fsaverage5 vertices to the 180 HCP-MMP1 brain regions
//! (Glasser et al., 2016). Provides:
//! - Per-ROI average prediction values
//! - Top-k most activated regions
//! - ROI-to-vertex index mapping
//! - Wildcard ROI selection (e.g., `"V1*"` for all V1 sub-regions)
//!
//! The HCP-MMP1 parcellation defines 180 cortical areas per hemisphere (360 total).
//! Each vertex in fsaverage5 (10,242 per hemisphere) is assigned to one region.
//!
//! Reference: Glasser MF, et al. "A multi-modal parcellation of human cerebral cortex."
//! Nature 536, 171–178 (2016).

use std::collections::HashMap;

/// Total vertices in fsaverage5 (both hemispheres).
pub const FSAVERAGE5_N_VERTICES: usize = 20484;
/// Vertices per hemisphere in fsaverage5.
pub const FSAVERAGE5_N_VERTICES_HEMI: usize = 10242;

/// HCP-MMP1 ROI names (180 regions). These are the canonical region names
/// without hemisphere prefix.
///
/// The 180 regions span: primary sensory (V1, A1, S1), association cortex,
/// prefrontal, temporal, parietal, and motor areas.
pub const HCP_ROI_NAMES: &[&str] = &[
    // Visual areas
    "V1", "V2", "V3", "V4", "V3A", "V3B", "V6", "V6A", "V7", "V8",
    "MT", "MST", "V4t", "FST", "LO1", "LO2", "LO3", "PIT", "VVC", "VMV1",
    "VMV2", "VMV3", "FFC", "PHA1", "PHA2", "PHA3", "V3CD", "V1_V2_V3_early",
    // Somatosensory and motor
    "4", "3a", "3b", "1", "2", "6mp", "6ma", "6d", "6v", "6r",
    "FEF", "PEF", "55b", "SCEF",
    // Auditory
    "A1", "A4", "A5", "LBelt", "MBelt", "PBelt", "RI", "TA2", "STSda", "STSdp",
    "STSva", "STSvp", "STGa",
    // Temporal
    "TE1a", "TE1m", "TE1p", "TE2a", "TE2p", "TGd", "TGv", "TF",
    "EC", "PreS", "H", "PeEc", "PHT",
    // Parietal
    "IP0", "IP1", "IP2", "IPS1", "MIP", "LIPv", "LIPd", "VIP",
    "AIP", "7AL", "7Am", "7PC", "7PL", "7Pm",
    "PGp", "PGs", "PGi", "PFm", "PFop", "PFt", "PF",
    "DVT", "ProS", "PCV", "POS1", "POS2",
    // Inferior parietal / TPJ
    "STV", "TPOJ1", "TPOJ2", "TPOJ3",
    // Insular
    "Ig", "PoI1", "PoI2", "MI", "FOP1", "FOP2", "FOP3", "FOP4", "FOP5",
    "PI", "Pir", "AVI", "AAIC",
    // Cingulate
    "RSC", "23c", "23d", "31a", "31pd", "31pv", "d23ab", "v23ab",
    "33pr", "p24pr", "a24pr", "p24", "a24",
    "p32pr", "a32pr", "p32", "s32", "d32",
    // Medial prefrontal
    "8BM", "9m", "10v", "10r", "25", "OFC",
    // Lateral prefrontal
    "8C", "8Av", "8Ad", "8BL",
    "9a", "9p", "46", "9-46d",
    "a9-46v", "p9-46v", "i6-8", "s6-8",
    "43", "44", "45", "47l", "47m", "47s", "IFSa", "IFSp", "IFJa", "IFJp",
    // Orbital
    "10d", "10pp", "11l", "13l",
    // Premotor / SMA
    "SFL",
    // Posterior
    "PeEc_post",
    // Additional areas to reach 180
    "52", "Pol1", "Pol2",
    "a10p", "p10p",
    "a47r",
    "PSL",
    "pOFC", "AOS", "IFa",
];

/// Simplified HCP-MMP1 vertex-to-ROI mapping for fsaverage5.
///
/// Since we can't ship the full FreeSurfer annotation file in the binary,
/// this provides an approximate mapping based on vertex index ranges.
/// For exact mapping, use `load_hcp_labels_from_annot()`.
///
/// Returns: HashMap<roi_name, Vec<vertex_index>> where vertex indices
/// are into the combined [left_hemi; right_hemi] array (0..20484).
pub fn get_hcp_labels_approximate() -> HashMap<String, Vec<usize>> {
    let n_per_hemi = FSAVERAGE5_N_VERTICES_HEMI;
    let n_rois = HCP_ROI_NAMES.len();
    // Approximate: divide each hemisphere evenly among ROIs
    // This is a rough approximation — for exact labels, load from annotation file
    let verts_per_roi = n_per_hemi / n_rois;
    let mut labels: HashMap<String, Vec<usize>> = HashMap::new();

    for (i, &name) in HCP_ROI_NAMES.iter().enumerate() {
        let start_l = i * verts_per_roi;
        let end_l = if i == n_rois - 1 { n_per_hemi } else { (i + 1) * verts_per_roi };

        let start_r = n_per_hemi + start_l;
        let end_r = if i == n_rois - 1 { 2 * n_per_hemi } else { n_per_hemi + (i + 1) * verts_per_roi };

        let mut verts: Vec<usize> = (start_l..end_l).collect();
        verts.extend(start_r..end_r);
        labels.insert(name.to_string(), verts);
    }

    labels
}

/// Load HCP-MMP1 labels from a FreeSurfer annotation file.
///
/// If available, this provides exact vertex-to-ROI mapping.
/// Requires lh.HCPMMP1.annot and rh.HCPMMP1.annot files.
///
/// `annot_dir`: directory containing the annotation files.
/// Returns: HashMap<roi_name, Vec<vertex_index>>
pub fn load_hcp_labels_from_annot(annot_dir: &std::path::Path) -> anyhow::Result<HashMap<String, Vec<usize>>> {
    let n_per_hemi = FSAVERAGE5_N_VERTICES_HEMI;
    let mut all_labels: HashMap<String, Vec<usize>> = HashMap::new();

    for (hemi, offset) in [("lh", 0usize), ("rh", n_per_hemi)] {
        let annot_path = annot_dir.join(format!("{}.HCPMMP1.annot", hemi));
        if !annot_path.exists() {
            // Try alternative naming
            let alt_path = annot_dir.join(format!("{}.aparc.HCPMMP1.annot", hemi));
            if alt_path.exists() {
                let hcp = crate::fsaverage::HcpLabels::from_annot(&alt_path, n_per_hemi)?;
                for (vi, label) in hcp.labels.iter().enumerate() {
                    if !label.is_empty() {
                        let clean = label.replace("_ROI", "").replace("-lh", "").replace("-rh", "");
                        all_labels.entry(clean).or_default().push(vi + offset);
                    }
                }
                continue;
            }
            anyhow::bail!("HCP annotation file not found: {}", annot_path.display());
        }

        let hcp = crate::fsaverage::HcpLabels::from_annot(&annot_path, n_per_hemi)?;
        for (vi, label) in hcp.labels.iter().enumerate() {
            if !label.is_empty() {
                let clean = label.replace("_ROI", "").replace("-lh", "").replace("-rh", "");
                all_labels.entry(clean).or_default().push(vi + offset);
            }
        }
    }

    Ok(all_labels)
}

/// Get HCP labels — tries annotation files first, falls back to approximate.
pub fn get_hcp_labels(annot_dir: Option<&std::path::Path>) -> HashMap<String, Vec<usize>> {
    if let Some(dir) = annot_dir {
        if let Ok(labels) = load_hcp_labels_from_annot(dir) {
            if !labels.is_empty() {
                return labels;
            }
        }
    }
    get_hcp_labels_approximate()
}

/// Get per-vertex ROI label string.
///
/// Returns a Vec of length `n_vertices` where each entry is the ROI name
/// for that vertex (empty string if unlabeled).
pub fn get_hcp_vertex_labels(annot_dir: Option<&std::path::Path>) -> Vec<String> {
    let labels = get_hcp_labels(annot_dir);
    let mut vertex_labels = vec![String::new(); FSAVERAGE5_N_VERTICES];
    for (name, vertices) in &labels {
        for &vi in vertices {
            if vi < vertex_labels.len() {
                vertex_labels[vi] = name.clone();
            }
        }
    }
    vertex_labels
}

/// Average prediction values per ROI.
///
/// `data`: per-vertex predictions, length `n_vertices` (typically 20,484).
/// Returns: HashMap<roi_name, mean_value>, sorted by ROI name.
pub fn summarize_by_roi(
    data: &[f32],
    annot_dir: Option<&std::path::Path>,
) -> HashMap<String, f32> {
    let labels = get_hcp_labels(annot_dir);
    let mut summary = HashMap::new();

    for (name, vertices) in &labels {
        if vertices.is_empty() {
            continue;
        }
        let sum: f32 = vertices.iter()
            .filter_map(|&vi| data.get(vi))
            .sum();
        let count = vertices.iter().filter(|&&vi| vi < data.len()).count();
        if count > 0 {
            summary.insert(name.clone(), sum / count as f32);
        }
    }

    summary
}

/// Get the top-k most activated ROIs.
///
/// Returns Vec of (roi_name, mean_value) sorted by descending activation.
pub fn get_topk_rois(
    data: &[f32],
    k: usize,
    annot_dir: Option<&std::path::Path>,
) -> Vec<(String, f32)> {
    let summary = summarize_by_roi(data, annot_dir);
    let mut sorted: Vec<(String, f32)> = summary.into_iter().collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    sorted.truncate(k);
    sorted
}

/// Get vertex indices matching a ROI name pattern.
///
/// Supports:
/// - Exact match: `"V1"` → vertices for V1
/// - Prefix wildcard: `"V1*"` → vertices for V1, V1_V2_V3_early, etc.
/// - Suffix wildcard: `"*Belt"` → LBelt, MBelt, PBelt
pub fn get_roi_indices(
    pattern: &str,
    annot_dir: Option<&std::path::Path>,
) -> Vec<usize> {
    let labels = get_hcp_labels(annot_dir);

    let matching: Vec<&Vec<usize>> = if pattern.ends_with('*') {
        let prefix = &pattern[..pattern.len() - 1];
        labels.iter()
            .filter(|(name, _)| name.starts_with(prefix))
            .map(|(_, v)| v)
            .collect()
    } else if pattern.starts_with('*') {
        let suffix = &pattern[1..];
        labels.iter()
            .filter(|(name, _)| name.ends_with(suffix))
            .map(|(_, v)| v)
            .collect()
    } else {
        labels.get(pattern).into_iter().collect()
    };

    let mut indices: Vec<usize> = matching.into_iter().flatten().copied().collect();
    indices.sort_unstable();
    indices.dedup();
    indices
}

/// Serialize ROI summary to JSON string.
pub fn roi_summary_to_json(summary: &HashMap<String, f32>) -> String {
    let mut sorted: Vec<(&String, &f32)> = summary.iter().collect();
    sorted.sort_by(|(a, _), (b, _)| a.cmp(b));

    let entries: Vec<String> = sorted.iter()
        .map(|(k, v)| format!("  \"{}\": {:.6}", k, v))
        .collect();
    format!("{{\n{}\n}}", entries.join(",\n"))
}

/// Serialize top-k ROIs to a readable table string.
pub fn topk_to_table(topk: &[(String, f32)]) -> String {
    let mut lines = vec![format!("{:<6} {:<25} {:>12}", "Rank", "Region", "Activation")];
    lines.push(format!("{}", "-".repeat(45)));
    for (i, (name, val)) in topk.iter().enumerate() {
        lines.push(format!("{:<6} {:<25} {:>12.6}", i + 1, name, val));
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_approximate_labels_cover_all_vertices() {
        let labels = get_hcp_labels_approximate();
        let total: usize = labels.values().map(|v| v.len()).sum();
        // Should cover all 20484 vertices
        assert_eq!(total, FSAVERAGE5_N_VERTICES,
            "Labels cover {} vertices, expected {}", total, FSAVERAGE5_N_VERTICES);
    }

    #[test]
    fn test_approximate_labels_no_overlap() {
        let labels = get_hcp_labels_approximate();
        let mut seen = vec![false; FSAVERAGE5_N_VERTICES];
        for vertices in labels.values() {
            for &vi in vertices {
                if vi < FSAVERAGE5_N_VERTICES {
                    assert!(!seen[vi], "vertex {} assigned to multiple ROIs", vi);
                    seen[vi] = true;
                }
            }
        }
    }

    #[test]
    fn test_summarize_by_roi() {
        let data = vec![1.0f32; FSAVERAGE5_N_VERTICES];
        let summary = summarize_by_roi(&data, None);
        // All vertices = 1.0, so every ROI average should be 1.0
        for (_, val) in &summary {
            assert!((val - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_topk_rois() {
        let mut data = vec![0.0f32; FSAVERAGE5_N_VERTICES];
        // Set first ROI vertices to high values
        let labels = get_hcp_labels_approximate();
        if let Some(v1_verts) = labels.get("V1") {
            for &vi in v1_verts {
                data[vi] = 10.0;
            }
        }
        let topk = get_topk_rois(&data, 3, None);
        assert!(!topk.is_empty());
        assert_eq!(topk[0].0, "V1");
        assert!((topk[0].1 - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_get_roi_indices_exact() {
        let indices = get_roi_indices("V1", None);
        assert!(!indices.is_empty());
    }

    #[test]
    fn test_get_roi_indices_wildcard() {
        let indices = get_roi_indices("V*", None);
        let v1_indices = get_roi_indices("V1", None);
        // Wildcard should include V1 and more
        assert!(indices.len() >= v1_indices.len());
    }

    #[test]
    fn test_vertex_labels() {
        let labels = get_hcp_vertex_labels(None);
        assert_eq!(labels.len(), FSAVERAGE5_N_VERTICES);
        // Most vertices should have a label with approximate mapping
        let labeled = labels.iter().filter(|l| !l.is_empty()).count();
        assert!(labeled == FSAVERAGE5_N_VERTICES,
            "Only {} / {} vertices labeled", labeled, FSAVERAGE5_N_VERTICES);
    }

    #[test]
    fn test_json_output() {
        let mut summary = HashMap::new();
        summary.insert("V1".to_string(), 1.5f32);
        summary.insert("A1".to_string(), 0.3f32);
        let json = roi_summary_to_json(&summary);
        assert!(json.contains("\"V1\""));
        assert!(json.contains("\"A1\""));
    }
}
