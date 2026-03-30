//! FreeSurfer fsaverage mesh loading.
//!
//! Loads fsaverage surface meshes from FreeSurfer's standard directory structure.
//! Supports loading from:
//! - FreeSurfer subjects directory (`$SUBJECTS_DIR/fsaverage5/surf/`)
//! - nilearn's cached data (`~/.nilearn/data/`)
//! - Explicit paths
//!
//! FreeSurfer surface file format (.pial, .inflated, .white, .sulc, .curv):
//! - Binary "triangle" format with magic number 0xFF_FF_FE
//! - Header: 2 lines of comments, then n_vertices (i32 BE), n_faces (i32 BE)
//! - Vertex data: n_vertices × (x: f32 BE, y: f32 BE, z: f32 BE)
//! - Face data: n_faces × (v0: i32 BE, v1: i32 BE, v2: i32 BE)
//!
//! Curvature file format (.sulc, .curv):
//! - Magic: 3 bytes (0xFF 0xFF 0xFF)
//! - n_vertices: i32 BE, n_faces: i32 BE, vals_per_vertex: i32 BE
//! - Data: n_vertices × f32 BE

use std::path::{Path, PathBuf};
use anyhow::{Context, Result};
use crate::plotting::{BrainMesh, HemisphereMesh, SurfaceMesh};

/// Standard fsaverage sizes: vertices per hemisphere.
pub const FSAVERAGE_SIZES: &[(&str, usize)] = &[
    ("fsaverage3", 642),
    ("fsaverage4", 2562),
    ("fsaverage5", 10242),
    ("fsaverage6", 40962),
    ("fsaverage", 163842),
];

/// Get the number of vertices per hemisphere for a given mesh name.
pub fn fsaverage_size(mesh: &str) -> Option<usize> {
    FSAVERAGE_SIZES.iter().find(|(n, _)| *n == mesh).map(|(_, s)| *s)
}

/// Read a FreeSurfer surface file (binary triangle format).
///
/// Returns (coords_flat [n_vertices * 3], faces_flat [n_faces * 3]).
pub fn read_freesurfer_surface(path: &Path) -> Result<(Vec<f32>, Vec<u32>, usize, usize)> {
    let data = std::fs::read(path)
        .with_context(|| format!("failed to read surface: {}", path.display()))?;

    if data.len() < 3 {
        anyhow::bail!("surface file too small: {}", path.display());
    }

    // Check magic number: 0xFF 0xFF 0xFE for triangle format
    if data[0] != 0xFF || data[1] != 0xFF || data[2] != 0xFE {
        anyhow::bail!("not a FreeSurfer triangle surface file: {}", path.display());
    }

    let mut pos = 3;

    // Skip two comment lines (terminated by \n)
    let mut newlines = 0;
    while pos < data.len() && newlines < 2 {
        if data[pos] == b'\n' {
            newlines += 1;
        }
        pos += 1;
    }

    if pos + 8 > data.len() {
        anyhow::bail!("surface file truncated at header: {}", path.display());
    }

    let n_vertices = i32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
    pos += 4;
    let n_faces = i32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
    pos += 4;

    // Read vertices: n_vertices × 3 × f32 (big-endian)
    let vertex_bytes = n_vertices * 3 * 4;
    if pos + vertex_bytes > data.len() {
        anyhow::bail!("surface file truncated at vertices: {} (need {} bytes at offset {}, have {})",
            path.display(), vertex_bytes, pos, data.len());
    }

    let mut coords = Vec::with_capacity(n_vertices * 3);
    for _ in 0..n_vertices * 3 {
        let v = f32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
        coords.push(v);
        pos += 4;
    }

    // Read faces: n_faces × 3 × i32 (big-endian)
    let face_bytes = n_faces * 3 * 4;
    if pos + face_bytes > data.len() {
        anyhow::bail!("surface file truncated at faces: {}", path.display());
    }

    let mut faces = Vec::with_capacity(n_faces * 3);
    for _ in 0..n_faces * 3 {
        let v = i32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as u32;
        faces.push(v);
        pos += 4;
    }

    Ok((coords, faces, n_vertices, n_faces))
}

/// Read a FreeSurfer curvature/sulcal-depth file (.sulc, .curv).
///
/// Returns per-vertex scalar values.
pub fn read_freesurfer_curv(path: &Path) -> Result<Vec<f32>> {
    let data = std::fs::read(path)
        .with_context(|| format!("failed to read curvature: {}", path.display()))?;

    if data.len() < 3 {
        anyhow::bail!("curv file too small: {}", path.display());
    }

    // New binary format: magic = 0xFF 0xFF 0xFF
    if data[0] == 0xFF && data[1] == 0xFF && data[2] == 0xFF {
        if data.len() < 15 {
            anyhow::bail!("curv file truncated: {}", path.display());
        }
        let pos = 3;
        let n_vertices = i32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        let _n_faces = i32::from_be_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]]) as usize;
        let _vals_per_vertex = i32::from_be_bytes([data[pos + 8], data[pos + 9], data[pos + 10], data[pos + 11]]) as usize;
        let mut offset = pos + 12;

        let mut values = Vec::with_capacity(n_vertices);
        for _ in 0..n_vertices {
            if offset + 4 > data.len() {
                break;
            }
            let v = f32::from_be_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]]);
            values.push(v);
            offset += 4;
        }
        return Ok(values);
    }

    // Old format: first 3 bytes = n_vertices as i24 BE (big endian), then n_faces as i24 BE
    let n_vertices = ((data[0] as usize) << 16) | ((data[1] as usize) << 8) | (data[2] as usize);
    let _n_faces = ((data[3] as usize) << 16) | ((data[4] as usize) << 8) | (data[5] as usize);
    let mut offset = 6;
    let mut values = Vec::with_capacity(n_vertices);
    for _ in 0..n_vertices {
        if offset + 2 > data.len() {
            break;
        }
        // Old format stores as i16 / 100
        let v = i16::from_be_bytes([data[offset], data[offset + 1]]) as f32 / 100.0;
        values.push(v);
        offset += 2;
    }
    Ok(values)
}

/// Discover the fsaverage subjects directory.
///
/// Searches in order:
/// 1. Explicit `base_path` if provided
/// 2. `$FREESURFER_SUBJECTS_DIR`
/// 3. `$SUBJECTS_DIR`
/// 4. nilearn cache: `~/.nilearn/data/`
/// 5. `/usr/local/freesurfer/subjects/`
/// 6. `/opt/freesurfer/subjects/`
pub fn find_fsaverage_dir(mesh: &str, base_path: Option<&str>) -> Option<PathBuf> {
    let candidates: Vec<PathBuf> = if let Some(bp) = base_path {
        vec![PathBuf::from(bp).join(mesh)]
    } else {
        let mut dirs = Vec::new();

        if let Ok(d) = std::env::var("FREESURFER_SUBJECTS_DIR") {
            dirs.push(PathBuf::from(d).join(mesh));
        }
        if let Ok(d) = std::env::var("SUBJECTS_DIR") {
            dirs.push(PathBuf::from(d).join(mesh));
        }

        // nilearn cache
        if let Ok(home) = std::env::var("HOME") {
            // nilearn stores fetched data like:
            // ~/.nilearn/data/fsaverage5/surf/lh.pial
            dirs.push(PathBuf::from(&home).join(".nilearn/data").join(mesh));
            // Also check for the nilearn datasets format
            dirs.push(PathBuf::from(&home).join("nilearn_data").join(mesh));
        }

        dirs.push(PathBuf::from("/usr/local/freesurfer/subjects").join(mesh));
        dirs.push(PathBuf::from("/opt/freesurfer/subjects").join(mesh));

        dirs
    };

    for dir in candidates {
        if dir.join("surf").exists() || dir.join("lh.pial").exists() {
            return Some(dir);
        }
    }
    None
}

/// Load a brain mesh from FreeSurfer fsaverage files.
///
/// `mesh`: e.g. "fsaverage5"
/// `inflate`: "half" (half-inflated), "full" (inflated), "pial" (original pial)
/// `bg_map`: "sulcal" (sulcal depth) or "curvature"
/// `base_path`: optional FreeSurfer subjects directory
pub fn load_fsaverage(
    mesh: &str,
    inflate: &str,
    bg_map_type: &str,
    base_path: Option<&str>,
) -> Result<BrainMesh> {
    let mesh_dir = find_fsaverage_dir(mesh, base_path)
        .ok_or_else(|| anyhow::anyhow!(
            "Could not find {} mesh. Set FREESURFER_SUBJECTS_DIR or provide base_path.\n\
             Searched standard locations. Run Python: \
             `from nilearn.datasets import fetch_surf_fsaverage; fetch_surf_fsaverage('{}')`\n\
             to download the mesh data.",
            mesh, mesh
        ))?;

    let surf_dir = if mesh_dir.join("surf").exists() {
        mesh_dir.join("surf")
    } else {
        mesh_dir.clone()
    };

    let hemisphere_gap = 0.0;

    let load_hemi = |hemi: &str| -> Result<HemisphereMesh> {
        let h = if hemi == "left" { "lh" } else { "rh" };

        // Load pial surface
        let pial_path = surf_dir.join(format!("{}.pial", h));
        let (pial_coords, faces, n_vertices, n_faces) = read_freesurfer_surface(&pial_path)?;

        // Load inflated surface (if available)
        let coords = match inflate {
            "half" => {
                let infl_path = surf_dir.join(format!("{}.inflated", h));
                if infl_path.exists() {
                    let (infl_coords, _, _, _) = read_freesurfer_surface(&infl_path)?;
                    // Half inflated: 0.5 * inflated + 0.5 * pial
                    pial_coords.iter().zip(infl_coords.iter())
                        .map(|(&p, &i)| 0.5 * p + 0.5 * i)
                        .collect()
                } else {
                    pial_coords.clone()
                }
            }
            "full" | "inflated" => {
                let infl_path = surf_dir.join(format!("{}.inflated", h));
                if infl_path.exists() {
                    let (infl_coords, _, _, _) = read_freesurfer_surface(&infl_path)?;
                    infl_coords
                } else {
                    pial_coords.clone()
                }
            }
            _ => pial_coords.clone(), // "pial" or default
        };

        // Apply hemisphere gap (shift x-coordinates)
        let mut coords = coords;
        if hemi == "left" {
            // Shift left hemisphere to the left
            let max_x = coords.iter().step_by(3).cloned().fold(f32::MIN, f32::max);
            for i in (0..coords.len()).step_by(3) {
                coords[i] -= max_x + hemisphere_gap;
            }
        } else {
            let min_x = coords.iter().step_by(3).cloned().fold(f32::MAX, f32::min);
            for i in (0..coords.len()).step_by(3) {
                coords[i] -= min_x - hemisphere_gap;
            }
        }

        // Load background map
        let bg_ext = if bg_map_type == "curvature" { "curv" } else { "sulc" };
        let bg_path = surf_dir.join(format!("{}.{}", h, bg_ext));
        let bg_map = if bg_path.exists() {
            read_freesurfer_curv(&bg_path)?
        } else {
            vec![0.0; n_vertices]
        };

        Ok(HemisphereMesh {
            mesh: SurfaceMesh {
                coords,
                faces,
                n_vertices,
                n_faces,
            },
            bg_map,
        })
    };

    let left = load_hemi("left").with_context(|| "failed to load left hemisphere")?;
    let right = load_hemi("right").with_context(|| "failed to load right hemisphere")?;

    Ok(BrainMesh { left, right })
}

/// HCP-MMP1 ROI label for each vertex.
///
/// This is a placeholder — actual HCP labels require the parcellation atlas.
/// Use `load_hcp_labels()` with a label file to populate this.
pub struct HcpLabels {
    /// Per-vertex label strings. Empty string = no label.
    pub labels: Vec<String>,
}

impl HcpLabels {
    /// Create empty labels for n_vertices.
    pub fn empty(n_vertices: usize) -> Self {
        Self {
            labels: vec![String::new(); n_vertices],
        }
    }

    /// Load HCP-MMP1 labels from a FreeSurfer annotation file (.annot).
    ///
    /// Annotation file format:
    /// - n_vertices (i32 LE)
    /// - vertex_data: n_vertices × (vertex_index: i32 LE, label: i32 LE)
    /// - tag (i32 LE) — should be 1
    /// - ctab_n_entries (i32 LE)
    /// - For each entry: name_len (i32 LE), name (bytes), r,g,b,a,label (all i32 LE)
    pub fn from_annot(path: &Path, n_vertices_total: usize) -> Result<Self> {
        let data = std::fs::read(path)
            .with_context(|| format!("failed to read annot: {}", path.display()))?;

        if data.len() < 4 {
            anyhow::bail!("annot file too small");
        }

        let mut pos = 0;

        // n_vertices
        let n = i32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]) as usize;
        pos += 4;

        // Read vertex → label mapping
        let mut vertex_labels = vec![0i32; n];
        for _ in 0..n {
            if pos + 8 > data.len() { break; }
            let vidx = i32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]) as usize;
            pos += 4;
            let label = i32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]);
            pos += 4;
            if vidx < n {
                vertex_labels[vidx] = label;
            }
        }

        // Read color table
        // Skip to tag
        if pos + 4 > data.len() {
            // No color table — return numeric labels
            let mut labels = vec![String::new(); n_vertices_total];
            for (i, &l) in vertex_labels.iter().enumerate() {
                if i < labels.len() && l != 0 {
                    labels[i] = format!("ROI_{}", l);
                }
            }
            return Ok(Self { labels });
        }

        let _tag = i32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]);
        pos += 4;

        if pos + 4 > data.len() {
            let mut labels = vec![String::new(); n_vertices_total];
            for (i, &l) in vertex_labels.iter().enumerate() {
                if i < labels.len() && l != 0 {
                    labels[i] = format!("ROI_{}", l);
                }
            }
            return Ok(Self { labels });
        }

        let n_entries = i32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]) as usize;
        pos += 4;

        // Original format: first read "orig_tab" string length
        if pos + 4 > data.len() {
            let mut labels = vec![String::new(); n_vertices_total];
            for (i, &l) in vertex_labels.iter().enumerate() {
                if i < labels.len() && l != 0 {
                    labels[i] = format!("ROI_{}", l);
                }
            }
            return Ok(Self { labels });
        }

        // Read entries: build label_code → name map
        let mut code_to_name: std::collections::HashMap<i32, String> = std::collections::HashMap::new();

        // Try to parse color table entries
        // Format varies; simplified parsing:
        for _ in 0..n_entries {
            if pos + 4 > data.len() { break; }
            let name_len = i32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]) as usize;
            pos += 4;
            if pos + name_len > data.len() { break; }
            let name = String::from_utf8_lossy(&data[pos..pos + name_len]).trim_end_matches('\0').to_string();
            pos += name_len;
            if pos + 16 > data.len() { break; }
            let r = i32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]);
            let g = i32::from_le_bytes([data[pos+4], data[pos+5], data[pos+6], data[pos+7]]);
            let b = i32::from_le_bytes([data[pos+8], data[pos+9], data[pos+10], data[pos+11]]);
            let a = i32::from_le_bytes([data[pos+12], data[pos+13], data[pos+14], data[pos+15]]);
            pos += 16;
            // label code = r + g*256 + b*65536 + a*16777216
            let code = r + g * 256 + b * 65536 + a * 16777216;
            code_to_name.insert(code, name);
        }

        let mut labels = vec![String::new(); n_vertices_total];
        for (i, &l) in vertex_labels.iter().enumerate() {
            if i < labels.len() {
                if let Some(name) = code_to_name.get(&l) {
                    labels[i] = name.clone();
                }
            }
        }

        Ok(Self { labels })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fsaverage_sizes() {
        assert_eq!(fsaverage_size("fsaverage5"), Some(10242));
        assert_eq!(fsaverage_size("fsaverage"), Some(163842));
        assert_eq!(fsaverage_size("unknown"), None);
    }

    #[test]
    fn test_find_fsaverage_dir_nonexistent() {
        let result = find_fsaverage_dir("fsaverage5", Some("/nonexistent/path"));
        assert!(result.is_none());
    }

    #[test]
    fn test_hcp_labels_empty() {
        let labels = HcpLabels::empty(100);
        assert_eq!(labels.labels.len(), 100);
        assert!(labels.labels.iter().all(|l| l.is_empty()));
    }
}
