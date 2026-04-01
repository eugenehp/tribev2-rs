//! Volume-to-surface projection for fMRI data.
//!
//! Mirrors the Python `nilearn.surface.vol_to_surf` functionality:
//! projects volumetric fMRI data (NIfTI) onto the fsaverage cortical surface
//! by sampling voxel values at surface vertex locations.
//!
//! Supports:
//! - **Ball** sampling: average voxels within a sphere around each vertex
//! - **Line** sampling: sample along the normal direction at multiple depths
//! - **Nearest** interpolation: take the single nearest voxel
//! - Arbitrary affine transforms (from NIfTI header)

use anyhow::{Context, Result};

/// Interpolation method for volume-to-surface projection.
#[derive(Debug, Clone, Copy)]
pub enum Interpolation {
    /// Nearest-neighbor: take the closest voxel value.
    Nearest,
    /// Trilinear interpolation between 8 surrounding voxels.
    Linear,
}

/// Sampling strategy for volume-to-surface projection.
#[derive(Debug, Clone)]
pub enum SamplingKind {
    /// Ball: average voxels within `radius` mm of each vertex.
    Ball { radius: f32 },
    /// Line: sample along vertex normal at multiple depths between surfaces.
    Line { n_samples: usize },
}

impl Default for SamplingKind {
    fn default() -> Self {
        SamplingKind::Ball { radius: 3.0 }
    }
}

/// Configuration for volume-to-surface projection.
#[derive(Debug, Clone)]
pub struct VolToSurfConfig {
    /// Sampling method.
    pub kind: SamplingKind,
    /// Interpolation for individual voxel lookups.
    pub interpolation: Interpolation,
    /// Depths along the surface normal to sample at (0.0 = white surface, 1.0 = pial).
    /// Only used with `SamplingKind::Line`.
    pub depths: Vec<f32>,
}

impl Default for VolToSurfConfig {
    fn default() -> Self {
        Self {
            kind: SamplingKind::Ball { radius: 3.0 },
            interpolation: Interpolation::Linear,
            depths: vec![0.0, 0.25, 0.5, 0.75, 1.0],
        }
    }
}

/// A loaded NIfTI volume with its affine transform.
#[derive(Debug, Clone)]
pub struct NiftiVolume {
    /// 3D or 4D volume data, flattened row-major.
    pub data: Vec<f32>,
    /// Shape: [nx, ny, nz] or [nx, ny, nz, nt].
    pub shape: Vec<usize>,
    /// 4×4 affine matrix (row-major, maps voxel → MNI coordinates).
    pub affine: [f32; 16],
    /// Inverse affine (MNI → voxel).
    pub inv_affine: [f32; 16],
}

impl NiftiVolume {
    /// Load a NIfTI-1 file (.nii or .nii.gz).
    pub fn load(path: &str) -> Result<Self> {
        let bytes = if path.ends_with(".gz") {
            use std::io::Read;
            let file = std::fs::File::open(path)
                .with_context(|| format!("failed to open {}", path))?;
            let mut decoder = flate2::read::GzDecoder::new(file);
            let mut buf = Vec::new();
            decoder.read_to_end(&mut buf)?;
            buf
        } else {
            std::fs::read(path)
                .with_context(|| format!("failed to read {}", path))?
        };

        if bytes.len() < 352 {
            anyhow::bail!("NIfTI file too small: {} bytes", bytes.len());
        }

        // Parse header
        let dim_off = 40;
        let ndims = i16::from_le_bytes([bytes[dim_off], bytes[dim_off + 1]]) as usize;
        let mut shape = Vec::new();
        for d in 0..ndims.min(7) {
            let off = dim_off + 2 + d * 2;
            let s = i16::from_le_bytes([bytes[off], bytes[off + 1]]) as usize;
            if s > 0 { shape.push(s); }
        }

        // Read sform affine (srow_x, srow_y, srow_z at offsets 280, 296, 312)
        let mut affine = [0.0f32; 16];
        for row in 0..3 {
            let off = 280 + row * 16;
            for col in 0..4 {
                affine[row * 4 + col] = f32::from_le_bytes([
                    bytes[off + col * 4],
                    bytes[off + col * 4 + 1],
                    bytes[off + col * 4 + 2],
                    bytes[off + col * 4 + 3],
                ]);
            }
        }
        affine[12] = 0.0;
        affine[13] = 0.0;
        affine[14] = 0.0;
        affine[15] = 1.0;

        let inv_affine = invert_affine_4x4(&affine);

        // Read vox_offset
        let vox_off_bytes = &bytes[108..112];
        let vox_offset = f32::from_le_bytes([vox_off_bytes[0], vox_off_bytes[1], vox_off_bytes[2], vox_off_bytes[3]]) as usize;

        // Read datatype
        let datatype = i16::from_le_bytes([bytes[70], bytes[71]]);

        let n_voxels: usize = shape.iter().product();
        let data_start = vox_offset.max(352);

        let data: Vec<f32> = match datatype {
            16 => {
                // FLOAT32
                bytes[data_start..].chunks_exact(4).take(n_voxels)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect()
            }
            4 => {
                // INT16
                bytes[data_start..].chunks_exact(2).take(n_voxels)
                    .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32)
                    .collect()
            }
            8 => {
                // INT32
                bytes[data_start..].chunks_exact(4).take(n_voxels)
                    .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32)
                    .collect()
            }
            _ => {
                anyhow::bail!("Unsupported NIfTI datatype: {}", datatype);
            }
        };

        // Apply scl_slope/scl_inter
        let scl_slope = f32::from_le_bytes([bytes[112], bytes[113], bytes[114], bytes[115]]);
        let scl_inter = f32::from_le_bytes([bytes[116], bytes[117], bytes[118], bytes[119]]);
        let data = if scl_slope != 0.0 && scl_slope != 1.0 {
            data.iter().map(|&v| v * scl_slope + scl_inter).collect()
        } else {
            data
        };

        Ok(Self { data, shape, affine, inv_affine })
    }

    /// Get voxel value at integer coordinates. Returns 0 if out of bounds.
    pub fn get_voxel(&self, i: isize, j: isize, k: isize) -> f32 {
        if self.shape.len() < 3 { return 0.0; }
        let (nx, ny, nz) = (self.shape[0], self.shape[1], self.shape[2]);
        if i < 0 || j < 0 || k < 0 || i >= nx as isize || j >= ny as isize || k >= nz as isize {
            return 0.0;
        }
        let idx = i as usize + j as usize * nx + k as usize * nx * ny;
        self.data.get(idx).copied().unwrap_or(0.0)
    }

    /// Get voxel value at integer coordinates for a specific timepoint.
    pub fn get_voxel_4d(&self, i: isize, j: isize, k: isize, t: usize) -> f32 {
        if self.shape.len() < 4 { return self.get_voxel(i, j, k); }
        let (nx, ny, nz) = (self.shape[0], self.shape[1], self.shape[2]);
        if i < 0 || j < 0 || k < 0 || i >= nx as isize || j >= ny as isize || k >= nz as isize {
            return 0.0;
        }
        let vol_size = nx * ny * nz;
        let idx = i as usize + j as usize * nx + k as usize * nx * ny + t * vol_size;
        self.data.get(idx).copied().unwrap_or(0.0)
    }

    /// Trilinear interpolation at continuous voxel coordinates.
    fn sample_linear(&self, vi: f32, vj: f32, vk: f32) -> f32 {
        let i0 = vi.floor() as isize;
        let j0 = vj.floor() as isize;
        let k0 = vk.floor() as isize;
        let fi = vi - i0 as f32;
        let fj = vj - j0 as f32;
        let fk = vk - k0 as f32;

        let c000 = self.get_voxel(i0, j0, k0);
        let c001 = self.get_voxel(i0, j0, k0 + 1);
        let c010 = self.get_voxel(i0, j0 + 1, k0);
        let c011 = self.get_voxel(i0, j0 + 1, k0 + 1);
        let c100 = self.get_voxel(i0 + 1, j0, k0);
        let c101 = self.get_voxel(i0 + 1, j0, k0 + 1);
        let c110 = self.get_voxel(i0 + 1, j0 + 1, k0);
        let c111 = self.get_voxel(i0 + 1, j0 + 1, k0 + 1);

        let c00 = c000 * (1.0 - fi) + c100 * fi;
        let c01 = c001 * (1.0 - fi) + c101 * fi;
        let c10 = c010 * (1.0 - fi) + c110 * fi;
        let c11 = c011 * (1.0 - fi) + c111 * fi;

        let c0 = c00 * (1.0 - fj) + c10 * fj;
        let c1 = c01 * (1.0 - fj) + c11 * fj;

        c0 * (1.0 - fk) + c1 * fk
    }

    /// Map MNI coordinate to voxel coordinate using inverse affine.
    fn mni_to_voxel(&self, x: f32, y: f32, z: f32) -> (f32, f32, f32) {
        let a = &self.inv_affine;
        let vi = a[0] * x + a[1] * y + a[2] * z + a[3];
        let vj = a[4] * x + a[5] * y + a[6] * z + a[7];
        let vk = a[8] * x + a[9] * y + a[10] * z + a[11];
        (vi, vj, vk)
    }
}

/// Project a 3D volume onto the cortical surface.
///
/// `volume`: the NIfTI volume to project.
/// `pial_coords`: vertex coordinates on the pial surface [n_vertices * 3] in MNI space.
/// `white_coords`: vertex coordinates on the white surface [n_vertices * 3] in MNI space (optional, for line sampling).
/// `config`: projection configuration.
///
/// Returns: per-vertex values [n_vertices].
pub fn vol_to_surf(
    volume: &NiftiVolume,
    pial_coords: &[f32],
    white_coords: Option<&[f32]>,
    config: &VolToSurfConfig,
) -> Vec<f32> {
    let n_vertices = pial_coords.len() / 3;
    let mut result = vec![0.0f32; n_vertices];

    match &config.kind {
        SamplingKind::Ball { radius } => {
            let radius_voxels = radius / volume.affine[0].abs().max(0.01);

            for vi in 0..n_vertices {
                let x = pial_coords[vi * 3];
                let y = pial_coords[vi * 3 + 1];
                let z = pial_coords[vi * 3 + 2];

                let (vx, vy, vz) = volume.mni_to_voxel(x, y, z);

                // Sample in a ball of given radius
                let r_int = radius_voxels.ceil() as isize;
                let mut sum = 0.0f32;
                let mut count = 0u32;

                for dk in -r_int..=r_int {
                    for dj in -r_int..=r_int {
                        for di in -r_int..=r_int {
                            let dist = ((di * di + dj * dj + dk * dk) as f32).sqrt();
                            if dist <= radius_voxels {
                                let ii = vx.round() as isize + di;
                                let jj = vy.round() as isize + dj;
                                let kk = vz.round() as isize + dk;
                                let val = volume.get_voxel(ii, jj, kk);
                                if val != 0.0 || count == 0 {
                                    sum += val;
                                    count += 1;
                                }
                            }
                        }
                    }
                }

                result[vi] = if count > 0 { sum / count as f32 } else { 0.0 };
            }
        }

        SamplingKind::Line { n_samples: _ } => {
            let depths = &config.depths;
            let wc = white_coords.unwrap_or(pial_coords);

            for vi in 0..n_vertices {
                let px = pial_coords[vi * 3];
                let py = pial_coords[vi * 3 + 1];
                let pz = pial_coords[vi * 3 + 2];
                let wx = wc[vi * 3];
                let wy = wc[vi * 3 + 1];
                let wz = wc[vi * 3 + 2];

                let mut sum = 0.0f32;
                let mut count = 0u32;

                for &depth in depths {
                    // Interpolate between white (depth=0) and pial (depth=1)
                    let x = wx + (px - wx) * depth;
                    let y = wy + (py - wy) * depth;
                    let z = wz + (pz - wz) * depth;

                    let (vx, vy, vz) = volume.mni_to_voxel(x, y, z);
                    let val = match config.interpolation {
                        Interpolation::Nearest => {
                            volume.get_voxel(vx.round() as isize, vy.round() as isize, vz.round() as isize)
                        }
                        Interpolation::Linear => {
                            volume.sample_linear(vx, vy, vz)
                        }
                    };
                    sum += val;
                    count += 1;
                }

                result[vi] = if count > 0 { sum / count as f32 } else { 0.0 };
            }
        }
    }

    result
}

/// Project a 4D volume (time series) onto the surface.
///
/// Returns: [n_timepoints][n_vertices]
pub fn vol_to_surf_4d(
    volume: &NiftiVolume,
    pial_coords: &[f32],
    white_coords: Option<&[f32]>,
    config: &VolToSurfConfig,
) -> Vec<Vec<f32>> {
    if volume.shape.len() < 4 {
        return vec![vol_to_surf(volume, pial_coords, white_coords, config)];
    }

    let n_t = volume.shape[3];
    let (nx, ny, nz) = (volume.shape[0], volume.shape[1], volume.shape[2]);
    let vol_size = nx * ny * nz;

    (0..n_t).map(|t| {
        // Create a single-volume slice
        let start = t * vol_size;
        let end = start + vol_size;
        let slice_data = volume.data[start..end.min(volume.data.len())].to_vec();
        let single_vol = NiftiVolume {
            data: slice_data,
            shape: vec![nx, ny, nz],
            affine: volume.affine,
            inv_affine: volume.inv_affine,
        };
        vol_to_surf(&single_vol, pial_coords, white_coords, config)
    }).collect()
}

/// Invert a 4×4 affine matrix (row-major).
fn invert_affine_4x4(m: &[f32; 16]) -> [f32; 16] {
    // For an affine matrix [R|t; 0 0 0 1], the inverse is [R^-1 | -R^-1*t; 0 0 0 1]
    // Extract 3×3 rotation/scale
    let a = m[0] as f64; let b = m[1] as f64; let c = m[2] as f64;
    let d = m[4] as f64; let e = m[5] as f64; let f = m[6] as f64;
    let g = m[8] as f64; let h = m[9] as f64; let i = m[10] as f64;

    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if det.abs() < 1e-15 {
        return [1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0];
    }
    let inv_det = 1.0 / det;

    let ri00 = (e * i - f * h) * inv_det;
    let ri01 = (c * h - b * i) * inv_det;
    let ri02 = (b * f - c * e) * inv_det;
    let ri10 = (f * g - d * i) * inv_det;
    let ri11 = (a * i - c * g) * inv_det;
    let ri12 = (c * d - a * f) * inv_det;
    let ri20 = (d * h - e * g) * inv_det;
    let ri21 = (b * g - a * h) * inv_det;
    let ri22 = (a * e - b * d) * inv_det;

    let tx = m[3] as f64;
    let ty = m[7] as f64;
    let tz = m[11] as f64;

    let it0 = -(ri00 * tx + ri01 * ty + ri02 * tz);
    let it1 = -(ri10 * tx + ri11 * ty + ri12 * tz);
    let it2 = -(ri20 * tx + ri21 * ty + ri22 * tz);

    [
        ri00 as f32, ri01 as f32, ri02 as f32, it0 as f32,
        ri10 as f32, ri11 as f32, ri12 as f32, it1 as f32,
        ri20 as f32, ri21 as f32, ri22 as f32, it2 as f32,
        0.0, 0.0, 0.0, 1.0,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invert_affine_identity() {
        let id = [1.0, 0.0, 0.0, 0.0,
                   0.0, 1.0, 0.0, 0.0,
                   0.0, 0.0, 1.0, 0.0,
                   0.0, 0.0, 0.0, 1.0];
        let inv = invert_affine_4x4(&id);
        for i in 0..16 { assert!((inv[i] - id[i]).abs() < 1e-6); }
    }

    #[test]
    fn test_invert_affine_translation() {
        let m = [1.0, 0.0, 0.0, 10.0,
                  0.0, 1.0, 0.0, 20.0,
                  0.0, 0.0, 1.0, 30.0,
                  0.0, 0.0, 0.0, 1.0];
        let inv = invert_affine_4x4(&m);
        assert!((inv[3] - (-10.0)).abs() < 1e-6);
        assert!((inv[7] - (-20.0)).abs() < 1e-6);
        assert!((inv[11] - (-30.0)).abs() < 1e-6);
    }

    #[test]
    fn test_invert_affine_scale() {
        let m = [2.0, 0.0, 0.0, -90.0,
                  0.0, 2.0, 0.0, -126.0,
                  0.0, 0.0, 2.0, -72.0,
                  0.0, 0.0, 0.0, 1.0];
        let inv = invert_affine_4x4(&m);
        // inv should be [0.5, 0, 0, 45; 0, 0.5, 0, 63; 0, 0, 0.5, 36; ...]
        assert!((inv[0] - 0.5).abs() < 1e-6);
        assert!((inv[3] - 45.0).abs() < 1e-6);
        assert!((inv[7] - 63.0).abs() < 1e-6);
        assert!((inv[11] - 36.0).abs() < 1e-6);
    }

    #[test]
    fn test_vol_to_surf_nearest() {
        // 4×4×4 volume with value 1.0 everywhere
        let vol = NiftiVolume {
            data: vec![1.0; 64],
            shape: vec![4, 4, 4],
            affine: [1.0, 0.0, 0.0, 0.0,
                     0.0, 1.0, 0.0, 0.0,
                     0.0, 0.0, 1.0, 0.0,
                     0.0, 0.0, 0.0, 1.0],
            inv_affine: [1.0, 0.0, 0.0, 0.0,
                         0.0, 1.0, 0.0, 0.0,
                         0.0, 0.0, 1.0, 0.0,
                         0.0, 0.0, 0.0, 1.0],
        };
        let coords = vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0];
        let config = VolToSurfConfig {
            kind: SamplingKind::Ball { radius: 0.5 },
            interpolation: Interpolation::Nearest,
            ..Default::default()
        };
        let result = vol_to_surf(&vol, &coords, None, &config);
        assert_eq!(result.len(), 2);
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 1.0).abs() < 1e-6);
    }
}
