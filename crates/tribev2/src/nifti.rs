//! NIfTI-1 (.nii / .nii.gz) writer for volumetric brain output.
//!
//! Converts surface-based predictions (fsaverage5 vertices) to volumetric
//! NIfTI images by projecting vertex coordinates into a 3D voxel grid.
//!
//! Output format:
//! - **NIfTI-1** (.nii.gz with gzip compression, or .nii uncompressed)
//! - **Image size**: configurable, default 96×96×96 voxels
//! - **Voxel size**: 2mm isotropic (MNI152 space)
//! - **Data type**: float32
//!
//! The surface-to-volume projection uses nearest-neighbor assignment:
//! each vertex's MNI coordinate is mapped to the nearest voxel, and
//! if multiple vertices map to the same voxel, values are averaged.

use std::io::Write;
use std::path::Path;
use anyhow::{Context, Result};

/// Configuration for NIfTI volume output.
#[derive(Debug, Clone)]
pub struct NiftiConfig {
    /// Volume dimensions (x, y, z). Default: (96, 96, 96).
    pub dims: (usize, usize, usize),
    /// Voxel size in mm (isotropic). Default: 2.0.
    pub voxel_size: f32,
    /// Origin offset in mm (center of volume in MNI space).
    /// Default: (90.0, 126.0, 72.0) — centers the MNI152 brain.
    pub origin_mm: (f32, f32, f32),
    /// Whether to gzip compress the output (.nii.gz vs .nii).
    pub compress: bool,
    /// Gaussian smoothing FWHM in mm (0 = no smoothing). Default: 6.0.
    /// Applied after surface-to-volume scatter to fill the cortical ribbon.
    pub smooth_fwhm_mm: f32,
}

impl Default for NiftiConfig {
    fn default() -> Self {
        Self {
            dims: (96, 96, 96),
            voxel_size: 2.0,
            // MNI152 origin: the volume spans roughly [-90,90] x [-126,90] x [-72,108]
            // With 96 voxels at 2mm, the volume spans 192mm.
            // We offset so that MNI (0,0,0) is near the center.
            origin_mm: (90.0, 126.0, 72.0),
            compress: true,
            smooth_fwhm_mm: 6.0,
        }
    }
}

/// NIfTI-1 header (348 bytes).
///
/// Minimal header for a float32 3D volume in MNI space.
fn build_nifti1_header(config: &NiftiConfig) -> [u8; 348] {
    let mut hdr = [0u8; 348];

    let (nx, ny, nz) = config.dims;
    let vs = config.voxel_size;

    // sizeof_hdr = 348
    hdr[0..4].copy_from_slice(&348i32.to_le_bytes());

    // dim[0] = 3 (3D), dim[1..3] = nx, ny, nz
    // dim field starts at offset 40, 8 × i16
    let dim_off = 40;
    hdr[dim_off..dim_off + 2].copy_from_slice(&3i16.to_le_bytes());         // ndim
    hdr[dim_off + 2..dim_off + 4].copy_from_slice(&(nx as i16).to_le_bytes());
    hdr[dim_off + 4..dim_off + 6].copy_from_slice(&(ny as i16).to_le_bytes());
    hdr[dim_off + 6..dim_off + 8].copy_from_slice(&(nz as i16).to_le_bytes());
    hdr[dim_off + 8..dim_off + 10].copy_from_slice(&1i16.to_le_bytes());    // dim[4]=1
    hdr[dim_off + 10..dim_off + 12].copy_from_slice(&1i16.to_le_bytes());   // dim[5]=1
    hdr[dim_off + 12..dim_off + 14].copy_from_slice(&1i16.to_le_bytes());   // dim[6]=1
    hdr[dim_off + 14..dim_off + 16].copy_from_slice(&1i16.to_le_bytes());   // dim[7]=1

    // datatype = 16 (FLOAT32), bitpix = 32
    let datatype_off = 70;
    hdr[datatype_off..datatype_off + 2].copy_from_slice(&16i16.to_le_bytes());
    let bitpix_off = 72;
    hdr[bitpix_off..bitpix_off + 2].copy_from_slice(&32i16.to_le_bytes());

    // pixdim: pixdim[0]=1 (qfac), pixdim[1..3] = voxel sizes
    let pixdim_off = 76;
    hdr[pixdim_off..pixdim_off + 4].copy_from_slice(&1.0f32.to_le_bytes());     // qfac
    hdr[pixdim_off + 4..pixdim_off + 8].copy_from_slice(&vs.to_le_bytes());     // x
    hdr[pixdim_off + 8..pixdim_off + 12].copy_from_slice(&vs.to_le_bytes());    // y
    hdr[pixdim_off + 12..pixdim_off + 16].copy_from_slice(&vs.to_le_bytes());   // z

    // vox_offset = 352.0 (data starts at byte 352 in .nii single-file)
    let vox_offset_off = 108;
    hdr[vox_offset_off..vox_offset_off + 4].copy_from_slice(&352.0f32.to_le_bytes());

    // scl_slope = 1.0, scl_inter = 0.0
    let scl_slope_off = 112;
    hdr[scl_slope_off..scl_slope_off + 4].copy_from_slice(&1.0f32.to_le_bytes());
    let scl_inter_off = 116;
    hdr[scl_inter_off..scl_inter_off + 4].copy_from_slice(&0.0f32.to_le_bytes());

    // qform_code = 1 (Scanner Anat), sform_code = 4 (MNI)
    let qform_off = 252;
    hdr[qform_off..qform_off + 2].copy_from_slice(&1i16.to_le_bytes());
    let sform_off = 254;
    hdr[sform_off..sform_off + 2].copy_from_slice(&4i16.to_le_bytes());

    // srow_x, srow_y, srow_z — affine matrix rows (offset 280, 296, 312; 4×f32 each)
    // Maps voxel (i,j,k) → MNI (x,y,z):
    //   x = vs*i - origin_x
    //   y = vs*j - origin_y
    //   z = vs*k - origin_z
    let (ox, oy, oz) = config.origin_mm;
    let srow_x_off = 280;
    hdr[srow_x_off..srow_x_off + 4].copy_from_slice(&vs.to_le_bytes());
    hdr[srow_x_off + 4..srow_x_off + 8].copy_from_slice(&0.0f32.to_le_bytes());
    hdr[srow_x_off + 8..srow_x_off + 12].copy_from_slice(&0.0f32.to_le_bytes());
    hdr[srow_x_off + 12..srow_x_off + 16].copy_from_slice(&(-ox).to_le_bytes());

    let srow_y_off = 296;
    hdr[srow_y_off..srow_y_off + 4].copy_from_slice(&0.0f32.to_le_bytes());
    hdr[srow_y_off + 4..srow_y_off + 8].copy_from_slice(&vs.to_le_bytes());
    hdr[srow_y_off + 8..srow_y_off + 12].copy_from_slice(&0.0f32.to_le_bytes());
    hdr[srow_y_off + 12..srow_y_off + 16].copy_from_slice(&(-oy).to_le_bytes());

    let srow_z_off = 312;
    hdr[srow_z_off..srow_z_off + 4].copy_from_slice(&0.0f32.to_le_bytes());
    hdr[srow_z_off + 4..srow_z_off + 8].copy_from_slice(&0.0f32.to_le_bytes());
    hdr[srow_z_off + 8..srow_z_off + 12].copy_from_slice(&vs.to_le_bytes());
    hdr[srow_z_off + 12..srow_z_off + 16].copy_from_slice(&(-oz).to_le_bytes());

    // magic = "n+1\0" (single-file NIfTI)
    let magic_off = 344;
    hdr[magic_off..magic_off + 4].copy_from_slice(b"n+1\0");

    hdr
}

/// Map surface vertex coordinates (MNI space) to voxel indices.
///
/// `coords`: flat [n_vertices * 3] array of (x, y, z) in mm (MNI space).
/// Returns `(voxel_i, voxel_j, voxel_k)` for each vertex, or `None` if out of bounds.
pub fn mni_to_voxel(
    coords: &[f32],
    config: &NiftiConfig,
) -> Vec<Option<(usize, usize, usize)>> {
    let (nx, ny, nz) = config.dims;
    let vs = config.voxel_size;
    let (ox, oy, oz) = config.origin_mm;
    let n_vertices = coords.len() / 3;

    (0..n_vertices)
        .map(|i| {
            let x = coords[i * 3];
            let y = coords[i * 3 + 1];
            let z = coords[i * 3 + 2];

            // Inverse of sform: voxel = (mni + origin) / voxel_size
            let vi = ((x + ox) / vs).round() as isize;
            let vj = ((y + oy) / vs).round() as isize;
            let vk = ((z + oz) / vs).round() as isize;

            if vi >= 0 && vi < nx as isize && vj >= 0 && vj < ny as isize && vk >= 0 && vk < nz as isize {
                Some((vi as usize, vj as usize, vk as usize))
            } else {
                None
            }
        })
        .collect()
}

/// Project surface vertex values onto a 3D volume.
///
/// `vertex_values`: per-vertex scalar values (e.g., predicted BOLD).
/// `vertex_coords`: flat [n_vertices * 3] MNI coordinates from fsaverage mesh.
/// `config`: volume configuration.
///
/// Returns a flat f32 volume of size `nx * ny * nz` in row-major order.
///
/// The projection works in two stages:
/// 1. **Scatter**: each vertex is mapped to its nearest voxel (multiple vertices
///    in the same voxel are averaged).
/// 2. **Smooth + fill**: a 3D Gaussian kernel (FWHM from config) is applied to
///    spread each scattered point into a filled cortical ribbon. A distance mask
///    ensures only voxels within ~2× the smoothing radius of a vertex are filled.
pub fn surface_to_volume(
    vertex_values: &[f32],
    vertex_coords: &[f32],
    config: &NiftiConfig,
) -> Vec<f32> {
    let (nx, ny, nz) = config.dims;
    let n_voxels = nx * ny * nz;

    let mut volume = vec![0.0f32; n_voxels];
    let mut counts = vec![0u32; n_voxels];

    let voxels = mni_to_voxel(vertex_coords, config);

    for (vi, vox) in voxels.iter().enumerate() {
        if vi >= vertex_values.len() {
            break;
        }
        if let Some((i, j, k)) = *vox {
            let idx = i + j * nx + k * nx * ny;
            volume[idx] += vertex_values[vi];
            counts[idx] += 1;
        }
    }

    // Average where multiple vertices hit the same voxel
    for i in 0..n_voxels {
        if counts[i] > 1 {
            volume[i] /= counts[i] as f32;
        }
    }

    // Apply Gaussian smoothing to fill the cortical ribbon
    if config.smooth_fwhm_mm > 0.0 {
        let sigma_voxels = config.smooth_fwhm_mm / (2.355 * config.voxel_size);
        volume = gaussian_smooth_3d_masked(&volume, &counts, nx, ny, nz, sigma_voxels);
    }

    volume
}

/// 3D Gaussian smoothing via normalized convolution.
///
/// Smoothes both the signal and a binary indicator of where data exists,
/// then divides: `result = smooth(signal) / smooth(indicator)`. This
/// preserves signal amplitude near the cortical ribbon instead of
/// diluting it with surrounding zeros.
///
/// Only voxels within `3×sigma` of a scattered vertex receive values.
fn gaussian_smooth_3d_masked(
    volume: &[f32],
    scatter_counts: &[u32],
    nx: usize, ny: usize, nz: usize,
    sigma: f32,
) -> Vec<f32> {
    let radius = (3.0 * sigma).ceil() as isize;
    let n_voxels = nx * ny * nz;

    // indicator[i] = 1 where we have scattered data, 0 elsewhere
    let indicator: Vec<f32> = scatter_counts.iter()
        .map(|&c| if c > 0 { 1.0 } else { 0.0 })
        .collect();

    // Build 1D Gaussian kernel (unnormalized — normalization happens via division)
    let ksize = (2 * radius + 1) as usize;
    let mut kernel = vec![0.0f32; ksize];
    for i in 0..ksize {
        let x = i as f32 - radius as f32;
        kernel[i] = (-0.5 * (x / sigma) * (x / sigma)).exp();
    }

    // Smooth both signal and indicator with the same separable Gaussian
    let smooth_signal = separable_convolve_3d(volume, nx, ny, nz, &kernel, radius);
    let smooth_indicator = separable_convolve_3d(&indicator, nx, ny, nz, &kernel, radius);

    // Build distance mask: only fill voxels within radius of data
    let mut mask = vec![false; n_voxels];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                if scatter_counts[i + j * nx + k * nx * ny] > 0 {
                    let r = radius;
                    for dk in -r..=r {
                        let kk = k as isize + dk;
                        if kk < 0 || kk >= nz as isize { continue; }
                        for dj in -r..=r {
                            let jj = j as isize + dj;
                            if jj < 0 || jj >= ny as isize { continue; }
                            for di in -r..=r {
                                let ii = i as isize + di;
                                if ii < 0 || ii >= nx as isize { continue; }
                                if (di*di + dj*dj + dk*dk) as f32 <= (r*r) as f32 {
                                    mask[ii as usize + jj as usize * nx + kk as usize * nx * ny] = true;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Normalized convolution: signal / indicator, masked
    let mut result = vec![0.0f32; n_voxels];
    for i in 0..n_voxels {
        if mask[i] && smooth_indicator[i] > 1e-8 {
            result[i] = smooth_signal[i] / smooth_indicator[i];
        }
    }
    result
}

/// Separable 3-pass 1D Gaussian convolution on a 3D volume.
fn separable_convolve_3d(
    input: &[f32],
    nx: usize, ny: usize, nz: usize,
    kernel: &[f32],
    radius: isize,
) -> Vec<f32> {
    let n = nx * ny * nz;
    let ksize = kernel.len();
    let mut buf = input.to_vec();
    let mut tmp = vec![0.0f32; n];

    // Pass 1: X
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let mut sum = 0.0f32;
                for ki in 0..ksize {
                    let ii = i as isize + ki as isize - radius;
                    if ii >= 0 && ii < nx as isize {
                        sum += buf[ii as usize + j * nx + k * nx * ny] * kernel[ki];
                    }
                }
                tmp[i + j * nx + k * nx * ny] = sum;
            }
        }
    }
    std::mem::swap(&mut buf, &mut tmp);

    // Pass 2: Y
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let mut sum = 0.0f32;
                for ki in 0..ksize {
                    let jj = j as isize + ki as isize - radius;
                    if jj >= 0 && jj < ny as isize {
                        sum += buf[i + jj as usize * nx + k * nx * ny] * kernel[ki];
                    }
                }
                tmp[i + j * nx + k * nx * ny] = sum;
            }
        }
    }
    std::mem::swap(&mut buf, &mut tmp);

    // Pass 3: Z
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let mut sum = 0.0f32;
                for ki in 0..ksize {
                    let kk = k as isize + ki as isize - radius;
                    if kk >= 0 && kk < nz as isize {
                        sum += buf[i + j * nx + kk as usize * nx * ny] * kernel[ki];
                    }
                }
                tmp[i + j * nx + k * nx * ny] = sum;
            }
        }
    }

    tmp
}

/// Get combined vertex coordinates from a BrainMesh (left + right hemispheres).
///
/// Returns flat [n_total_vertices * 3] coordinate array.
/// Note: the mesh coordinates from `fsaverage.rs` may be shifted for visualization.
/// For NIfTI output, you should use the **original pial** coordinates (MNI space).
pub fn get_mesh_coords(brain: &crate::plotting::BrainMesh) -> Vec<f32> {
    let mut coords = brain.left.mesh.coords.clone();
    coords.extend_from_slice(&brain.right.mesh.coords);
    coords
}

/// Write a single 3D volume as NIfTI-1 (.nii or .nii.gz).
pub fn write_nifti(
    path: &Path,
    volume: &[f32],
    config: &NiftiConfig,
) -> Result<()> {
    let (nx, ny, nz) = config.dims;
    let expected = nx * ny * nz;
    if volume.len() != expected {
        anyhow::bail!(
            "Volume has {} voxels, expected {} ({}×{}×{})",
            volume.len(), expected, nx, ny, nz
        );
    }

    let header = build_nifti1_header(config);

    // 4-byte extension padding after header (bytes 348..352)
    let extension = [0u8; 4];

    // Convert volume to bytes
    let data_bytes: Vec<u8> = volume.iter().flat_map(|v| v.to_le_bytes()).collect();

    let is_gz = path.to_string_lossy().ends_with(".gz");

    if is_gz || config.compress {
        // Write gzip-compressed .nii.gz
        let file = std::fs::File::create(path)
            .with_context(|| format!("failed to create {}", path.display()))?;
        let mut gz = flate2::write::GzEncoder::new(file, flate2::Compression::default());
        gz.write_all(&header)?;
        gz.write_all(&extension)?;
        gz.write_all(&data_bytes)?;
        gz.finish()?;
    } else {
        // Write uncompressed .nii
        let mut file = std::fs::File::create(path)
            .with_context(|| format!("failed to create {}", path.display()))?;
        file.write_all(&header)?;
        file.write_all(&extension)?;
        file.write_all(&data_bytes)?;
    }

    Ok(())
}

/// Write per-timestep predictions as a 4D NIfTI (x, y, z, t).
///
/// `predictions`: Vec of per-timestep vertex values (each Vec<f32> has n_vertices entries).
/// `vertex_coords`: flat MNI coordinates [n_vertices * 3].
/// `config`: volume configuration.
/// `path`: output path (.nii or .nii.gz).
pub fn write_nifti_4d(
    path: &Path,
    predictions: &[Vec<f32>],
    vertex_coords: &[f32],
    config: &NiftiConfig,
) -> Result<()> {
    let (nx, ny, nz) = config.dims;
    let nt = predictions.len();

    if nt == 0 {
        anyhow::bail!("No timesteps to write");
    }

    // Build a 4D header
    let mut hdr = build_nifti1_header(config);

    // Override dim: ndim=4, dim[4]=nt
    let dim_off = 40;
    hdr[dim_off..dim_off + 2].copy_from_slice(&4i16.to_le_bytes());
    hdr[dim_off + 8..dim_off + 10].copy_from_slice(&(nt as i16).to_le_bytes());

    // pixdim[4] = TR (0.5s for TRIBE v2 at 2Hz)
    let pixdim_off = 76;
    hdr[pixdim_off + 16..pixdim_off + 20].copy_from_slice(&0.5f32.to_le_bytes());

    // xyzt_units: mm + sec (bits 0-2 = 2 for mm, bits 3-5 = 8 for sec → 10)
    let xyzt_off = 123;
    hdr[xyzt_off] = 10; // NIFTI_UNITS_MM | NIFTI_UNITS_SEC

    let extension = [0u8; 4];

    // Project all timesteps
    let mut all_data: Vec<u8> = Vec::with_capacity(nx * ny * nz * nt * 4);
    for (ti, pred) in predictions.iter().enumerate() {
        let vol = surface_to_volume(pred, vertex_coords, config);
        all_data.extend(vol.iter().flat_map(|v| v.to_le_bytes()));
        if (ti + 1) % 50 == 0 {
            eprintln!("  NIfTI: projected {}/{} timesteps", ti + 1, nt);
        }
    }

    let is_gz = path.to_string_lossy().ends_with(".gz");
    if is_gz || config.compress {
        let file = std::fs::File::create(path)
            .with_context(|| format!("failed to create {}", path.display()))?;
        let mut gz = flate2::write::GzEncoder::new(file, flate2::Compression::default());
        gz.write_all(&hdr)?;
        gz.write_all(&extension)?;
        gz.write_all(&all_data)?;
        gz.finish()?;
    } else {
        let mut file = std::fs::File::create(path)
            .with_context(|| format!("failed to create {}", path.display()))?;
        file.write_all(&hdr)?;
        file.write_all(&extension)?;
        file.write_all(&all_data)?;
    }

    eprintln!(
        "NIfTI written: {} ({}×{}×{}×{}, {:.1} MB)",
        path.display(), nx, ny, nz, nt,
        all_data.len() as f64 / 1e6
    );

    Ok(())
}

/// Load original (un-shifted) pial coordinates for surface-to-volume mapping.
///
/// Unlike the visualization mesh (which shifts hemispheres apart), these are
/// the raw FreeSurfer pial coordinates in MNI space.
pub fn load_pial_coords_mni(
    mesh: &str,
    base_path: Option<&str>,
) -> Result<Vec<f32>> {
    let mesh_dir = crate::fsaverage::find_fsaverage_dir(mesh, base_path)
        .ok_or_else(|| anyhow::anyhow!("Could not find {} mesh for NIfTI projection", mesh))?;

    let surf_dir = if mesh_dir.join("surf").exists() {
        mesh_dir.join("surf")
    } else {
        mesh_dir.clone()
    };

    let (lh_coords, _, _, _) = crate::fsaverage::read_freesurfer_surface(&surf_dir.join("lh.pial"))?;
    let (rh_coords, _, _, _) = crate::fsaverage::read_freesurfer_surface(&surf_dir.join("rh.pial"))?;

    let mut coords = lh_coords;
    coords.extend_from_slice(&rh_coords);
    Ok(coords)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nifti_header_size() {
        let config = NiftiConfig::default();
        let hdr = build_nifti1_header(&config);
        assert_eq!(hdr.len(), 348);
        // sizeof_hdr field
        assert_eq!(i32::from_le_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]), 348);
        // magic
        assert_eq!(&hdr[344..348], b"n+1\0");
    }

    #[test]
    fn test_mni_to_voxel() {
        let config = NiftiConfig::default();
        // MNI origin (0,0,0) should map to center-ish of volume
        let coords = vec![0.0, 0.0, 0.0];
        let voxels = mni_to_voxel(&coords, &config);
        assert_eq!(voxels.len(), 1);
        let (i, j, k) = voxels[0].unwrap();
        assert_eq!(i, 45); // 90/2 = 45
        assert_eq!(j, 63); // 126/2 = 63
        assert_eq!(k, 36); // 72/2 = 36
    }

    #[test]
    fn test_surface_to_volume_basic() {
        let config = NiftiConfig {
            dims: (10, 10, 10),
            voxel_size: 1.0,
            origin_mm: (5.0, 5.0, 5.0),
            compress: false,
            smooth_fwhm_mm: 0.0, // no smoothing for basic test
        };
        // Single vertex at MNI (0,0,0) → voxel (5,5,5)
        let coords = vec![0.0, 0.0, 0.0];
        let values = vec![42.0];
        let vol = surface_to_volume(&values, &coords, &config);
        assert_eq!(vol.len(), 1000);
        let idx = 5 + 5 * 10 + 5 * 100;
        assert_eq!(vol[idx], 42.0);
    }

    #[test]
    fn test_surface_to_volume_smoothed() {
        let config = NiftiConfig {
            dims: (20, 20, 20),
            voxel_size: 1.0,
            origin_mm: (10.0, 10.0, 10.0),
            compress: false,
            smooth_fwhm_mm: 3.0,
        };
        // Single vertex at MNI (0,0,0) → voxel (10,10,10)
        let coords = vec![0.0, 0.0, 0.0];
        let values = vec![42.0];
        let vol = surface_to_volume(&values, &coords, &config);
        let idx_center = 10 + 10 * 20 + 10 * 20 * 20;
        // Center voxel should have the strongest value
        assert!(vol[idx_center] > 0.0, "center voxel should be > 0");
        // Neighbor should also have nonzero value (smoothing spread)
        let idx_neighbor = 11 + 10 * 20 + 10 * 20 * 20;
        assert!(vol[idx_neighbor] > 0.0, "neighbor should be > 0 after smoothing");
        // Far-away voxel should be zero (masked)
        let idx_far = 0 + 0 * 20 + 0 * 20 * 20;
        assert_eq!(vol[idx_far], 0.0, "far voxel should be 0");
    }

    #[test]
    fn test_write_nifti_roundtrip() {
        let config = NiftiConfig {
            dims: (4, 4, 4),
            voxel_size: 2.0,
            origin_mm: (4.0, 4.0, 4.0),
            compress: false,
            smooth_fwhm_mm: 0.0,
        };
        let volume = vec![1.0f32; 64];
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.nii");
        write_nifti(&path, &volume, &config).unwrap();

        let data = std::fs::read(&path).unwrap();
        // 348 header + 4 extension + 64*4 data = 608 bytes
        assert_eq!(data.len(), 348 + 4 + 256);
    }
}
