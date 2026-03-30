//! Brain surface plotting for TRIBE v2 predictions.
//!
//! Mirrors the Python `plotting/` module for visualizing fMRI predictions
//! on the fsaverage cortical surface.
//!
//! Generates:
//! - **SVG** files with cortical surface renderings
//! - **PNG** files via the `resvg` feature (optional)
//! - Raw vertex data for use with external tools
//!
//! The fsaverage5 mesh has 10242 vertices per hemisphere (20484 total).
//!
//! Mesh data can be loaded from FreeSurfer or embedded (see `EmbeddedMesh`).
//!
//! Views supported: left, right, dorsal, ventral, medial_left, medial_right

use std::collections::HashMap;

/// A cortical surface mesh with vertex coordinates and triangle faces.
#[derive(Debug, Clone)]
pub struct SurfaceMesh {
    /// Vertex coordinates: [n_vertices, 3] flattened row-major.
    pub coords: Vec<f32>,
    /// Triangle faces: [n_faces, 3] flattened row-major (vertex indices).
    pub faces: Vec<u32>,
    /// Number of vertices.
    pub n_vertices: usize,
    /// Number of faces.
    pub n_faces: usize,
}

/// A hemisphere mesh with associated background map.
#[derive(Debug, Clone)]
pub struct HemisphereMesh {
    pub mesh: SurfaceMesh,
    /// Sulcal depth or curvature map [n_vertices] for shading.
    pub bg_map: Vec<f32>,
}

/// Brain mesh data for both hemispheres.
#[derive(Debug, Clone)]
pub struct BrainMesh {
    pub left: HemisphereMesh,
    pub right: HemisphereMesh,
}

/// Supported view angles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum View {
    Left,
    Right,
    MedLeft,
    MedRight,
    Dorsal,
    Ventral,
}

impl View {
    /// Camera direction (position) and up vector for each view.
    /// Returns (eye_x, eye_y, eye_z, up_x, up_y, up_z).
    pub fn camera(&self) -> (f32, f32, f32, f32, f32, f32) {
        match self {
            View::Left      => (-1.0,  0.0,  0.0,  0.0, 0.0, 1.0),
            View::Right     => ( 1.0,  0.0,  0.0,  0.0, 0.0, 1.0),
            View::MedLeft   => ( 1.0,  0.0,  0.0,  0.0, 0.0, 1.0),
            View::MedRight  => (-1.0,  0.0,  0.0,  0.0, 0.0, 1.0),
            View::Dorsal    => ( 0.0,  0.0,  1.0,  0.0, 1.0, 0.0),
            View::Ventral   => ( 0.0,  0.0, -1.0,  1.0, 0.0, 0.0),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            View::Left => "left",
            View::Right => "right",
            View::MedLeft => "medial_left",
            View::MedRight => "medial_right",
            View::Dorsal => "dorsal",
            View::Ventral => "ventral",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "left" => Some(View::Left),
            "right" => Some(View::Right),
            "medial_left" => Some(View::MedLeft),
            "medial_right" => Some(View::MedRight),
            "dorsal" => Some(View::Dorsal),
            "ventral" => Some(View::Ventral),
            _ => None,
        }
    }
}

/// Color map for brain surface visualization.
#[derive(Debug, Clone, Copy)]
pub enum ColorMap {
    Hot,
    CoolWarm,
    Viridis,
    Seismic,
    BlueRed,
    GrayScale,
}

impl ColorMap {
    /// Map a normalized value [0, 1] to (R, G, B) in [0, 255].
    pub fn map(&self, t: f32) -> (u8, u8, u8) {
        let t = t.clamp(0.0, 1.0);
        match self {
            ColorMap::Hot => {
                let r = (t * 3.0).min(1.0);
                let g = ((t - 0.333) * 3.0).clamp(0.0, 1.0);
                let b = ((t - 0.666) * 3.0).clamp(0.0, 1.0);
                ((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
            }
            ColorMap::CoolWarm | ColorMap::Seismic | ColorMap::BlueRed => {
                // Blue → White → Red
                if t < 0.5 {
                    let s = t * 2.0;
                    let r = (s * 255.0) as u8;
                    let g = (s * 255.0) as u8;
                    let b = 255;
                    (r, g, b)
                } else {
                    let s = (t - 0.5) * 2.0;
                    let r = 255;
                    let g = ((1.0 - s) * 255.0) as u8;
                    let b = ((1.0 - s) * 255.0) as u8;
                    (r, g, b)
                }
            }
            ColorMap::Viridis => {
                // Simplified viridis: purple → teal → yellow
                let r = ((t * 0.5 + t * t * 0.5) * 255.0).min(255.0) as u8;
                let g = ((0.1 + t * 0.8) * 255.0).min(255.0) as u8;
                let b = ((0.3 + (1.0 - t) * 0.5 + t * 0.1) * 255.0).min(255.0) as u8;
                (r, g, b)
            }
            ColorMap::GrayScale => {
                let v = (t * 255.0) as u8;
                (v, v, v)
            }
        }
    }
}

/// Configuration for brain surface plotting.
#[derive(Debug, Clone)]
pub struct PlotConfig {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Color map to use.
    pub cmap: ColorMap,
    /// Minimum value for color mapping. None = auto (data min).
    pub vmin: Option<f32>,
    /// Maximum value for color mapping. None = auto (data max).
    pub vmax: Option<f32>,
    /// Whether to use symmetric color bar (center at 0).
    pub symmetric_cbar: bool,
    /// Threshold: values with |v| < threshold shown as gray.
    pub threshold: Option<f32>,
    /// Background darkness (0 = white, 1 = black).
    pub bg_darkness: f32,
    /// View to render.
    pub view: View,
    /// Whether to include a color bar.
    pub colorbar: bool,
    /// Title text.
    pub title: Option<String>,
}

impl Default for PlotConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            cmap: ColorMap::Hot,
            vmin: None,
            vmax: None,
            symmetric_cbar: false,
            threshold: None,
            bg_darkness: 0.0,
            view: View::Left,
            colorbar: false,
            title: None,
        }
    }
}

/// Normalize data to [0, 1] range.
fn normalize_data(data: &[f32], vmin: f32, vmax: f32) -> Vec<f32> {
    let range = vmax - vmin;
    if range.abs() < 1e-12 {
        return vec![0.5; data.len()];
    }
    data.iter().map(|&v| ((v - vmin) / range).clamp(0.0, 1.0)).collect()
}

/// Robust normalize: use percentile-based min/max.
pub fn robust_normalize(data: &[f32], percentile: f32) -> Vec<f32> {
    if data.is_empty() {
        return vec![];
    }
    let mut sorted: Vec<f32> = data.iter().filter(|v| v.is_finite()).copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if sorted.is_empty() {
        return vec![0.5; data.len()];
    }
    let lo_idx = ((100.0 - percentile) / 100.0 * (sorted.len() - 1) as f32) as usize;
    let hi_idx = ((percentile) / 100.0 * (sorted.len() - 1) as f32) as usize;
    let lo = sorted[lo_idx.min(sorted.len() - 1)];
    let hi = sorted[hi_idx.min(sorted.len() - 1)];
    normalize_data(data, lo, hi)
}

/// Split vertex data into hemispheres.
///
/// Input: [n_total_vertices] where n_total_vertices = 2 * n_per_hemisphere.
/// Returns (left, right).
pub fn split_hemispheres(data: &[f32]) -> (&[f32], &[f32]) {
    let half = data.len() / 2;
    (&data[..half], &data[half..])
}

/// Simple 3D → 2D projection for a triangle mesh.
///
/// Uses orthographic projection along the view direction.
///
/// Returns projected 2D coordinates for each vertex: [(x, y)].
fn project_vertices(
    coords: &[f32],
    n_vertices: usize,
    view: View,
) -> Vec<(f32, f32)> {
    let (eye_x, eye_y, eye_z, up_x, up_y, up_z) = view.camera();

    // View direction = -eye (looking toward origin)
    let fwd = [-eye_x, -eye_y, -eye_z];

    // Right = fwd × up (normalized)
    let right = [
        fwd[1] * up_z - fwd[2] * up_y,
        fwd[2] * up_x - fwd[0] * up_z,
        fwd[0] * up_y - fwd[1] * up_x,
    ];
    let r_len = (right[0] * right[0] + right[1] * right[1] + right[2] * right[2]).sqrt();
    let right = [right[0] / r_len, right[1] / r_len, right[2] / r_len];

    // Recompute up = right × fwd
    let up = [
        right[1] * fwd[2] - right[2] * fwd[1],
        right[2] * fwd[0] - right[0] * fwd[2],
        right[0] * fwd[1] - right[1] * fwd[0],
    ];

    let mut proj = Vec::with_capacity(n_vertices);
    for i in 0..n_vertices {
        let x = coords[i * 3];
        let y = coords[i * 3 + 1];
        let z = coords[i * 3 + 2];

        // Orthographic projection: dot with right and up
        let px = x * right[0] + y * right[1] + z * right[2];
        let py = x * up[0] + y * up[1] + z * up[2];
        proj.push((px, py));
    }
    proj
}

/// Compute depth (distance along view direction) for each face.
fn compute_face_depths(
    coords: &[f32],
    faces: &[u32],
    n_faces: usize,
    view: View,
) -> Vec<f32> {
    let (eye_x, eye_y, eye_z, _, _, _) = view.camera();
    let fwd = [-eye_x, -eye_y, -eye_z];

    let mut depths = Vec::with_capacity(n_faces);
    for fi in 0..n_faces {
        let i0 = faces[fi * 3] as usize;
        let i1 = faces[fi * 3 + 1] as usize;
        let i2 = faces[fi * 3 + 2] as usize;

        let cx = (coords[i0 * 3] + coords[i1 * 3] + coords[i2 * 3]) / 3.0;
        let cy = (coords[i0 * 3 + 1] + coords[i1 * 3 + 1] + coords[i2 * 3 + 1]) / 3.0;
        let cz = (coords[i0 * 3 + 2] + coords[i1 * 3 + 2] + coords[i2 * 3 + 2]) / 3.0;

        let depth = cx * fwd[0] + cy * fwd[1] + cz * fwd[2];
        depths.push(depth);
    }
    depths
}

/// Render a brain hemisphere to SVG.
///
/// `data`: per-vertex scalar values [n_vertices] — normalized to [0, 1].
/// `mesh`: the hemisphere mesh.
/// `bg_map`: sulcal depth values [n_vertices] — used for background shading.
/// `config`: plot configuration.
///
/// Returns SVG string.
pub fn render_hemisphere_svg(
    data: &[f32],
    mesh: &SurfaceMesh,
    bg_map: &[f32],
    config: &PlotConfig,
) -> String {
    let n_v = mesh.n_vertices;
    let n_f = mesh.n_faces;

    // Project vertices to 2D
    let proj = project_vertices(&mesh.coords, n_v, config.view);

    // Find bounding box
    let (mut min_x, mut min_y) = (f32::MAX, f32::MAX);
    let (mut max_x, mut max_y) = (f32::MIN, f32::MIN);
    for &(px, py) in &proj {
        min_x = min_x.min(px);
        min_y = min_y.min(py);
        max_x = max_x.max(px);
        max_y = max_y.max(py);
    }

    let data_w = max_x - min_x;
    let data_h = max_y - min_y;

    // Reserve space for title (top) and colorbar (right)
    let margin_left = 15.0;
    let margin_top = if config.title.is_some() { 35.0 } else { 15.0 };
    let margin_right = if config.colorbar { 75.0 } else { 15.0 };
    let margin_bottom = 15.0;

    let avail_w = config.width as f32 - margin_left - margin_right;
    let avail_h = config.height as f32 - margin_top - margin_bottom;
    let scale = (avail_w / data_w).min(avail_h / data_h);

    let offset_x = margin_left + (avail_w - data_w * scale) / 2.0;
    let offset_y = margin_top + (avail_h - data_h * scale) / 2.0;

    // Transform vertices to screen coordinates
    let screen: Vec<(f32, f32)> = proj.iter().map(|&(px, py)| {
        let sx = (px - min_x) * scale + offset_x;
        let sy = config.height as f32 - ((py - min_y) * scale + offset_y); // flip Y
        (sx, sy)
    }).collect();

    // Compute face depths for painter's algorithm (back-to-front)
    let depths = compute_face_depths(&mesh.coords, &mesh.faces, n_f, config.view);
    let mut face_order: Vec<usize> = (0..n_f).collect();
    face_order.sort_by(|&a, &b| depths[a].partial_cmp(&depths[b]).unwrap_or(std::cmp::Ordering::Equal));

    // Determine color range
    let vmin = config.vmin.unwrap_or_else(|| data.iter().copied().fold(f32::MAX, f32::min));
    let vmax = config.vmax.unwrap_or_else(|| data.iter().copied().fold(f32::MIN, f32::max));
    let (vmin, vmax) = if config.symmetric_cbar {
        let absmax = vmin.abs().max(vmax.abs());
        (-absmax, absmax)
    } else {
        (vmin, vmax)
    };

    let normalized = normalize_data(data, vmin, vmax);

    // Normalize background map
    let bg_min = bg_map.iter().copied().fold(f32::MAX, f32::min);
    let bg_max = bg_map.iter().copied().fold(f32::MIN, f32::max);
    let bg_norm: Vec<f32> = if (bg_max - bg_min).abs() > 1e-12 {
        bg_map.iter().map(|&v| (v - bg_min) / (bg_max - bg_min)).collect()
    } else {
        vec![0.5; bg_map.len()]
    };

    // Build SVG
    let mut svg = String::with_capacity(n_f * 200);
    svg.push_str(&format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#,
        config.width, config.height, config.width, config.height
    ));
    svg.push('\n');

    // White background
    svg.push_str(&format!(
        r#"<rect width="{}" height="{}" fill="white"/>"#,
        config.width, config.height
    ));
    svg.push('\n');

    // Title — rendered in the reserved top margin
    if let Some(ref title) = config.title {
        svg.push_str(&format!(
            r#"<text x="{:.0}" y="22" text-anchor="middle" font-size="14" font-family="sans-serif">{}</text>"#,
            margin_left + avail_w / 2.0, title
        ));
        svg.push('\n');
    }

    // Draw faces (painter's algorithm: back to front)
    for &fi in &face_order {
        let i0 = mesh.faces[fi * 3] as usize;
        let i1 = mesh.faces[fi * 3 + 1] as usize;
        let i2 = mesh.faces[fi * 3 + 2] as usize;

        // Average vertex values for face color
        let face_val = (normalized[i0] + normalized[i1] + normalized[i2]) / 3.0;
        let face_bg = (bg_norm[i0] + bg_norm[i1] + bg_norm[i2]) / 3.0;

        let (mut r, mut g, mut b) = if let Some(thr) = config.threshold {
            let raw_val = (data[i0] + data[i1] + data[i2]) / 3.0;
            if raw_val.abs() < thr {
                // Below threshold → gray
                let gray = (128.0 + (1.0 - config.bg_darkness) * 64.0 * (1.0 - face_bg)) as u8;
                (gray, gray, gray)
            } else {
                config.cmap.map(face_val)
            }
        } else {
            config.cmap.map(face_val)
        };

        // Blend with background (sulcal depth shading)
        let bg_factor = 0.7 + 0.3 * (1.0 - face_bg);
        r = ((r as f32) * bg_factor).min(255.0) as u8;
        g = ((g as f32) * bg_factor).min(255.0) as u8;
        b = ((b as f32) * bg_factor).min(255.0) as u8;

        let (x0, y0) = screen[i0];
        let (x1, y1) = screen[i1];
        let (x2, y2) = screen[i2];

        svg.push_str(&format!(
            r#"<polygon points="{:.1},{:.1} {:.1},{:.1} {:.1},{:.1}" fill="rgb({},{},{})" stroke="rgb({},{},{})" stroke-width="0.3"/>"#,
            x0, y0, x1, y1, x2, y2, r, g, b, r, g, b
        ));
        svg.push('\n');
    }

    // Colorbar — drawn in the reserved right margin
    if config.colorbar {
        let cb_x = config.width as f32 - margin_right + 10.0;
        let cb_w = 15.0;
        let cb_y = margin_top + avail_h * 0.1;
        let cb_h = avail_h * 0.8;
        let n_steps = 64;

        for i in 0..n_steps {
            let t = 1.0 - (i as f32 / n_steps as f32);
            let (r, g, b) = config.cmap.map(t);
            let y = cb_y + (i as f32 / n_steps as f32) * cb_h;
            let h = cb_h / n_steps as f32 + 0.5;
            svg.push_str(&format!(
                r#"<rect x="{:.1}" y="{:.1}" width="{:.1}" height="{:.1}" fill="rgb({},{},{})"/>"#,
                cb_x, y, cb_w, h, r, g, b
            ));
            svg.push('\n');
        }

        // Labels
        svg.push_str(&format!(
            r#"<text x="{:.1}" y="{:.1}" font-size="10" font-family="sans-serif">{:.2}</text>"#,
            cb_x + cb_w + 4.0, cb_y + 10.0, vmax
        ));
        svg.push_str(&format!(
            r#"<text x="{:.1}" y="{:.1}" font-size="10" font-family="sans-serif">{:.2}</text>"#,
            cb_x + cb_w + 4.0, cb_y + cb_h, vmin
        ));
        svg.push('\n');
    }

    svg.push_str("</svg>\n");
    svg
}

/// Render both hemispheres to a single SVG.
///
/// `data`: per-vertex values [n_total_vertices] (left + right concatenated).
/// `brain`: brain mesh data.
/// `config`: plot configuration.
pub fn render_brain_svg(
    data: &[f32],
    brain: &BrainMesh,
    config: &PlotConfig,
) -> String {
    let (left_data, right_data) = split_hemispheres(data);

    // Determine which hemisphere to show based on view
    match config.view {
        View::Left | View::MedRight => {
            render_hemisphere_svg(left_data, &brain.left.mesh, &brain.left.bg_map, config)
        }
        View::Right | View::MedLeft => {
            render_hemisphere_svg(right_data, &brain.right.mesh, &brain.right.bg_map, config)
        }
        View::Dorsal | View::Ventral => {
            // Both hemispheres side by side.
            // Sub-renders have no title/colorbar; drawn once in outer SVG.
            let pad_top: f32 = if config.title.is_some() { 35.0 } else { 15.0 };
            let pad_right: f32 = if config.colorbar { 75.0 } else { 15.0 };
            let pad_left: f32 = 15.0;
            let pad_bottom: f32 = 15.0;
            let cw = config.width as f32 - pad_left - pad_right;
            let ch = config.height as f32 - pad_top - pad_bottom;
            let hw = (cw / 2.0) as u32;
            let chu = ch as u32;

            let mut sub_cfg = config.clone();
            sub_cfg.title = None;
            sub_cfg.colorbar = false;
            sub_cfg.width = hw;
            sub_cfg.height = chu;

            let left_svg = render_hemisphere_svg(left_data, &brain.left.mesh, &brain.left.bg_map, &sub_cfg);
            let right_svg = render_hemisphere_svg(right_data, &brain.right.mesh, &brain.right.bg_map, &sub_cfg);

            let mut c = format!(
                r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">
<rect width="{}" height="{}" fill="white"/>"#,
                config.width, config.height, config.width, config.height,
                config.width, config.height
            );

            if let Some(ref title) = config.title {
                c.push_str(&format!(
                    r#"
<text x="{:.0}" y="22" text-anchor="middle" font-size="14" font-family="sans-serif">{}</text>"#,
                    pad_left + cw / 2.0, title
                ));
            }

            c.push_str(&format!(r#"
<g transform="translate({:.0}, {:.0})">"#, pad_left, pad_top));
            if let Some(inner) = extract_svg_inner(&left_svg) { c.push_str(inner); }
            c.push_str("</g>");

            c.push_str(&format!(r#"
<g transform="translate({:.0}, {:.0})">"#, pad_left + hw as f32, pad_top));
            if let Some(inner) = extract_svg_inner(&right_svg) { c.push_str(inner); }
            c.push_str("</g>");

            if config.colorbar {
                let vmin_v = config.vmin.unwrap_or_else(|| data.iter().copied().fold(f32::MAX, f32::min));
                let vmax_v = config.vmax.unwrap_or_else(|| data.iter().copied().fold(f32::MIN, f32::max));
                let (vmin_v, vmax_v) = if config.symmetric_cbar {
                    let m = vmin_v.abs().max(vmax_v.abs()); (-m, m)
                } else { (vmin_v, vmax_v) };
                let cbx = config.width as f32 - pad_right + 10.0;
                let cby = pad_top + ch * 0.1;
                let cbh = ch * 0.8;
                for si in 0..64 {
                    let t = 1.0 - si as f32 / 64.0;
                    let (r, g, b) = config.cmap.map(t);
                    let y = cby + si as f32 / 64.0 * cbh;
                    c.push_str(&format!(
                        r#"
<rect x="{:.1}" y="{:.1}" width="15" height="{:.1}" fill="rgb({},{},{})"/>"#,
                        cbx, y, cbh / 64.0 + 0.5, r, g, b
                    ));
                }
                c.push_str(&format!(
                    r#"
<text x="{:.1}" y="{:.1}" font-size="10" font-family="sans-serif">{:.2}</text>
<text x="{:.1}" y="{:.1}" font-size="10" font-family="sans-serif">{:.2}</text>"#,
                    cbx + 19.0, cby + 10.0, vmax_v,
                    cbx + 19.0, cby + cbh, vmin_v
                ));
            }

            c.push_str("\n</svg>\n");
            c
        }
    }
}

fn extract_svg_inner(svg: &str) -> Option<&str> {
    let start = svg.find('>')?;
    let end = svg.rfind("</svg>")?;
    Some(&svg[start + 1..end])
}

/// Render multiple views to separate SVG files.
pub fn render_multi_view(
    data: &[f32],
    brain: &BrainMesh,
    views: &[View],
    base_config: &PlotConfig,
    output_dir: &str,
    prefix: &str,
) -> anyhow::Result<Vec<String>> {
    std::fs::create_dir_all(output_dir)?;
    let mut paths = Vec::new();

    for view in views {
        let mut config = base_config.clone();
        config.view = *view;
        config.title = Some(view.name().to_string());

        let svg = render_brain_svg(data, brain, &config);
        let path = format!("{}/{}_{}.svg", output_dir, prefix, view.name());
        std::fs::write(&path, &svg)?;
        paths.push(path);
    }

    Ok(paths)
}

/// Render a time series of brain maps to SVG files.
///
/// `predictions`: [n_timesteps, n_vertices] — one row per timestep.
/// `brain`: brain mesh.
/// `config`: base plot config.
/// `output_dir`: directory to save SVG files.
///
/// Returns paths to generated SVG files.
pub fn render_timesteps(
    predictions: &[Vec<f32>],
    brain: &BrainMesh,
    config: &PlotConfig,
    output_dir: &str,
) -> anyhow::Result<Vec<String>> {
    std::fs::create_dir_all(output_dir)?;
    let mut paths = Vec::new();

    // Compute global min/max for consistent color scale
    let global_min = predictions.iter()
        .flat_map(|row| row.iter())
        .copied()
        .fold(f32::MAX, f32::min);
    let global_max = predictions.iter()
        .flat_map(|row| row.iter())
        .copied()
        .fold(f32::MIN, f32::max);

    for (i, row) in predictions.iter().enumerate() {
        let mut ts_config = config.clone();
        ts_config.vmin = Some(global_min);
        ts_config.vmax = Some(global_max);
        ts_config.title = Some(format!("t = {}s", i));

        let svg = render_brain_svg(row, brain, &ts_config);
        let path = format!("{}/frame_{:04}.svg", output_dir, i);
        std::fs::write(&path, &svg)?;
        paths.push(path);
    }

    Ok(paths)
}

/// Generate a simple synthetic brain mesh for testing (low-res sphere).
///
/// This creates a rough sphere mesh that can be used for testing
/// without needing actual FreeSurfer data.
pub fn generate_test_mesh(n_per_hemisphere: usize) -> BrainMesh {
    let make_hemi = |offset_x: f32| -> HemisphereMesh {
        // Create a UV sphere
        let n_lat = (n_per_hemisphere as f32).sqrt() as usize;
        let n_lon = n_lat * 2;
        let mut coords = Vec::new();
        let mut faces = Vec::new();
        let mut bg_map = Vec::new();

        for i in 0..=n_lat {
            let theta = std::f32::consts::PI * i as f32 / n_lat as f32;
            for j in 0..n_lon {
                let phi = 2.0 * std::f32::consts::PI * j as f32 / n_lon as f32;
                let x = 50.0 * theta.sin() * phi.cos() + offset_x;
                let y = 50.0 * theta.sin() * phi.sin();
                let z = 50.0 * theta.cos();
                coords.extend_from_slice(&[x, y, z]);
                bg_map.push(theta.sin() * 0.5); // sulcal-like pattern
            }
        }

        let n_vertices = (n_lat + 1) * n_lon;
        for i in 0..n_lat {
            for j in 0..n_lon {
                let v0 = (i * n_lon + j) as u32;
                let v1 = (i * n_lon + (j + 1) % n_lon) as u32;
                let v2 = ((i + 1) * n_lon + j) as u32;
                let v3 = ((i + 1) * n_lon + (j + 1) % n_lon) as u32;
                faces.extend_from_slice(&[v0, v1, v2]);
                faces.extend_from_slice(&[v1, v3, v2]);
            }
        }

        let n_faces = faces.len() / 3;
        HemisphereMesh {
            mesh: SurfaceMesh {
                coords,
                faces,
                n_vertices,
                n_faces,
            },
            bg_map,
        }
    };

    BrainMesh {
        left: make_hemi(-60.0),
        right: make_hemi(60.0),
    }
}

/// Save per-vertex data to a binary f32 file.
///
/// Layout: [n_vertices] as little-endian f32.
pub fn save_vertex_data(data: &[f32], path: &str) -> anyhow::Result<()> {
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    std::fs::write(path, &bytes)?;
    Ok(())
}

/// Load per-vertex data from a binary f32 file.
pub fn load_vertex_data(path: &str) -> anyhow::Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    let data: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Ok(data)
}

/// Summarize predictions by HCP-MMP1 ROI names.
///
/// `data`: per-vertex values [n_total_vertices].
/// `roi_labels`: per-vertex ROI label strings [n_total_vertices].
///
/// Returns map of ROI name → mean value.
pub fn summarize_by_roi(
    data: &[f32],
    roi_labels: &[String],
) -> HashMap<String, f32> {
    assert_eq!(data.len(), roi_labels.len());
    let mut roi_sums: HashMap<String, (f64, usize)> = HashMap::new();

    for (i, label) in roi_labels.iter().enumerate() {
        if label.is_empty() {
            continue;
        }
        let entry = roi_sums.entry(label.clone()).or_insert((0.0, 0));
        entry.0 += data[i] as f64;
        entry.1 += 1;
    }

    roi_sums
        .into_iter()
        .map(|(name, (sum, count))| (name, (sum / count as f64) as f32))
        .collect()
}

/// Get the top-K ROIs by mean value.
pub fn top_k_rois(
    data: &[f32],
    roi_labels: &[String],
    k: usize,
) -> Vec<(String, f32)> {
    let roi_means = summarize_by_roi(data, roi_labels);
    let mut sorted: Vec<(String, f32)> = roi_means.into_iter().collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    sorted.truncate(k);
    sorted
}

// ── RGB multi-channel overlay ──────────────────────────────────────────────

/// Render an RGB brain overlay from 3 per-vertex signal arrays.
///
/// Mirrors `PlotBrainNilearn.plot_surf_rgb()`: each of the 3 signals
/// maps to R, G, B. The resulting per-vertex colour encodes multi-modal
/// activation (e.g. text=R, audio=G, video=B).
///
/// `signals`: exactly 3 arrays of [n_total_vertices], one per channel.
/// `norm_percentile`: percentile for per-channel normalization (e.g. 95).
/// `alpha_bg`: background blending factor (0 = full colour, 1 = full bg).
///
/// Returns per-vertex `(r, g, b)` in 0–255, length = n_total_vertices.
pub fn rgb_overlay(
    signals: &[&[f32]; 3],
    norm_percentile: f32,
    alpha_bg: f32,
    bg_map: Option<&[f32]>,
) -> Vec<(u8, u8, u8)> {
    let n = signals[0].len();
    assert_eq!(signals[1].len(), n);
    assert_eq!(signals[2].len(), n);

    let ch: Vec<Vec<f32>> = signals.iter()
        .map(|s| robust_normalize(s, norm_percentile))
        .collect();

    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let mut r = ch[0][i];
        let mut g = ch[1][i];
        let mut b = ch[2][i];

        // Background blending
        if let Some(bg) = bg_map {
            if i < bg.len() {
                let bg_val = bg[i].clamp(0.0, 1.0);
                let bg_gray = 0.7 + 0.3 * (1.0 - bg_val);
                r = r * (1.0 - alpha_bg) + bg_gray * alpha_bg;
                g = g * (1.0 - alpha_bg) + bg_gray * alpha_bg;
                b = b * (1.0 - alpha_bg) + bg_gray * alpha_bg;
            }
        }

        result.push((
            (r.clamp(0.0, 1.0) * 255.0) as u8,
            (g.clamp(0.0, 1.0) * 255.0) as u8,
            (b.clamp(0.0, 1.0) * 255.0) as u8,
        ));
    }
    result
}

/// Render an RGB overlay hemisphere to SVG.
///
/// `colors`: per-vertex (r, g, b) in 0–255.
pub fn render_hemisphere_rgb_svg(
    colors: &[(u8, u8, u8)],
    mesh: &SurfaceMesh,
    config: &PlotConfig,
) -> String {
    let n_v = mesh.n_vertices;
    let n_f = mesh.n_faces;

    let proj = project_vertices(&mesh.coords, n_v, config.view);

    let (mut min_x, mut min_y) = (f32::MAX, f32::MAX);
    let (mut max_x, mut max_y) = (f32::MIN, f32::MIN);
    for &(px, py) in &proj {
        min_x = min_x.min(px); min_y = min_y.min(py);
        max_x = max_x.max(px); max_y = max_y.max(py);
    }
    let data_w = max_x - min_x;
    let data_h = max_y - min_y;

    let margin_left = 15.0;
    let margin_top = if config.title.is_some() { 35.0 } else { 15.0 };
    let margin_right = 15.0;
    let margin_bottom = 15.0;
    let avail_w = config.width as f32 - margin_left - margin_right;
    let avail_h = config.height as f32 - margin_top - margin_bottom;
    let scale = (avail_w / data_w).min(avail_h / data_h);
    let offset_x = margin_left + (avail_w - data_w * scale) / 2.0;
    let offset_y = margin_top + (avail_h - data_h * scale) / 2.0;

    let screen: Vec<(f32, f32)> = proj.iter().map(|&(px, py)| {
        let sx = (px - min_x) * scale + offset_x;
        let sy = config.height as f32 - ((py - min_y) * scale + offset_y);
        (sx, sy)
    }).collect();

    let depths = compute_face_depths(&mesh.coords, &mesh.faces, n_f, config.view);
    let mut face_order: Vec<usize> = (0..n_f).collect();
    face_order.sort_by(|&a, &b| depths[a].partial_cmp(&depths[b]).unwrap_or(std::cmp::Ordering::Equal));

    let mut svg = String::with_capacity(n_f * 200);
    svg.push_str(&format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">
<rect width="{}" height="{}" fill="white"/>\n"#,
        config.width, config.height, config.width, config.height,
        config.width, config.height,
    ));

    for &fi in &face_order {
        let i0 = mesh.faces[fi * 3] as usize;
        let i1 = mesh.faces[fi * 3 + 1] as usize;
        let i2 = mesh.faces[fi * 3 + 2] as usize;

        let r = ((colors[i0].0 as u16 + colors[i1].0 as u16 + colors[i2].0 as u16) / 3) as u8;
        let g = ((colors[i0].1 as u16 + colors[i1].1 as u16 + colors[i2].1 as u16) / 3) as u8;
        let b = ((colors[i0].2 as u16 + colors[i1].2 as u16 + colors[i2].2 as u16) / 3) as u8;

        let (x0, y0) = screen[i0];
        let (x1, y1) = screen[i1];
        let (x2, y2) = screen[i2];

        svg.push_str(&format!(
            r#"<polygon points="{:.1},{:.1} {:.1},{:.1} {:.1},{:.1}" fill="rgb({},{},{})" stroke="rgb({},{},{})" stroke-width="0.3"/>\n"#,
            x0, y0, x1, y1, x2, y2, r, g, b, r, g, b
        ));
    }

    svg.push_str("</svg>\n");
    svg
}

// ── Saturate colors ───────────────────────────────────────────────────────

/// Boost or reduce colour saturation.
///
/// `factor > 1` boosts saturation, `1` leaves unchanged, `0` makes grayscale.
/// Uses Rec.709 luminance weights.
///
/// Mirrors `plotting/utils.py saturate_colors()`.
pub fn saturate_colors(rgb: &[(f32, f32, f32)], factor: f32) -> Vec<(f32, f32, f32)> {
    rgb.iter().map(|&(r, g, b)| {
        let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        let nr = (lum + factor * (r - lum)).clamp(0.0, 1.0);
        let ng = (lum + factor * (g - lum)).clamp(0.0, 1.0);
        let nb = (lum + factor * (b - lum)).clamp(0.0, 1.0);
        (nr, ng, nb)
    }).collect()
}

// ── Tight crop ────────────────────────────────────────────────────────────

/// Crop an image (row-major RGBA u8 buffer) to its non-background content.
///
/// `img`: pixel data, row-major, `width × height × 4` (RGBA).
/// `bg_color`: background colour to crop away `(r, g, b)`.
/// `tol`: tolerance for background matching.
///
/// Returns `(cropped_data, new_width, new_height)`.
///
/// Mirrors `plotting/utils.py tight_crop()`.
pub fn tight_crop(
    img: &[u8],
    width: usize,
    height: usize,
    channels: usize,
    bg_color: (u8, u8, u8),
    tol: u8,
) -> (Vec<u8>, usize, usize) {
    let mut min_x = width;
    let mut max_x = 0usize;
    let mut min_y = height;
    let mut max_y = 0usize;

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * channels;
            if idx + 2 >= img.len() { continue; }
            let r = img[idx];
            let g = img[idx + 1];
            let b = img[idx + 2];
            let dr = (r as i16 - bg_color.0 as i16).unsigned_abs() as u8;
            let dg = (g as i16 - bg_color.1 as i16).unsigned_abs() as u8;
            let db = (b as i16 - bg_color.2 as i16).unsigned_abs() as u8;
            if dr > tol || dg > tol || db > tol {
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
            }
        }
    }

    if max_x < min_x || max_y < min_y {
        return (img.to_vec(), width, height);
    }

    let new_w = max_x - min_x + 1;
    let new_h = max_y - min_y + 1;
    let mut cropped = Vec::with_capacity(new_w * new_h * channels);
    for y in min_y..=max_y {
        let start = (y * width + min_x) * channels;
        let end = start + new_w * channels;
        if end <= img.len() {
            cropped.extend_from_slice(&img[start..end]);
        }
    }
    (cropped, new_w, new_h)
}

// ── ROI annotation ────────────────────────────────────────────────────────

/// Annotate ROI labels on an SVG brain rendering.
///
/// `svg`: existing SVG content to add annotations to.
/// `roi_centers`: map of ROI name → (screen_x, screen_y) coordinates.
/// `font_size`: label font size in pixels.
///
/// Returns SVG with text labels added.
pub fn annotate_rois_svg(
    svg: &str,
    roi_centers: &[(String, f32, f32)],
    font_size: f32,
) -> String {
    // Insert labels before closing </svg>
    let insert_point = svg.rfind("</svg>").unwrap_or(svg.len());
    let mut result = svg[..insert_point].to_string();

    for (name, x, y) in roi_centers {
        result.push_str(&format!(
            r#"<text x="{:.1}" y="{:.1}" font-size="{:.0}" font-family="sans-serif" text-anchor="middle" fill="black" stroke="white" stroke-width="0.5">{}</text>\n"#,
            x, y, font_size, name
        ));
    }

    result.push_str("</svg>\n");
    result
}

/// Compute screen positions for ROI centers given vertex-to-ROI mapping.
///
/// `roi_labels`: per-vertex ROI name (empty = unlabelled).
/// `mesh`: the surface mesh.
/// `view`: camera view.
/// `width`, `height`: output image dimensions.
///
/// Returns (roi_name, screen_x, screen_y) for each unique ROI.
pub fn compute_roi_screen_positions(
    roi_labels: &[String],
    mesh: &SurfaceMesh,
    view: View,
    width: u32,
    height: u32,
) -> Vec<(String, f32, f32)> {
    let n_v = mesh.n_vertices;
    let proj = project_vertices(&mesh.coords, n_v, view);

    let (mut min_x, mut min_y) = (f32::MAX, f32::MAX);
    let (mut max_x, mut max_y) = (f32::MIN, f32::MIN);
    for &(px, py) in &proj {
        min_x = min_x.min(px); min_y = min_y.min(py);
        max_x = max_x.max(px); max_y = max_y.max(py);
    }
    let data_w = max_x - min_x;
    let data_h = max_y - min_y;
    let margin = 15.0;
    let avail_w = width as f32 - 2.0 * margin;
    let avail_h = height as f32 - 2.0 * margin;
    let scale = (avail_w / data_w).min(avail_h / data_h);
    let offset_x = margin + (avail_w - data_w * scale) / 2.0;
    let offset_y = margin + (avail_h - data_h * scale) / 2.0;

    // Group vertices by ROI and compute mean screen position
    let mut roi_sums: HashMap<String, (f64, f64, usize)> = HashMap::new();
    for (i, label) in roi_labels.iter().enumerate() {
        if label.is_empty() || i >= n_v { continue; }
        let sx = (proj[i].0 - min_x) * scale + offset_x;
        let sy = height as f32 - ((proj[i].1 - min_y) * scale + offset_y);
        let entry = roi_sums.entry(label.clone()).or_insert((0.0, 0.0, 0));
        entry.0 += sx as f64;
        entry.1 += sy as f64;
        entry.2 += 1;
    }

    roi_sums.into_iter().map(|(name, (sx, sy, count))| {
        (name, (sx / count as f64) as f32, (sy / count as f64) as f32)
    }).collect()
}

// ── MP4 generation ────────────────────────────────────────────────────────

/// Generate an MP4 video from a series of SVG frames using ffmpeg.
///
/// Requires `rsvg-convert` (from librsvg) and `ffmpeg` to be installed.
///
/// Mirrors `BasePlotBrain.plot_timesteps_mp4()`.
pub fn render_timesteps_mp4(
    predictions: &[Vec<f32>],
    brain: &BrainMesh,
    config: &PlotConfig,
    output_path: &str,
    fps: u32,
) -> anyhow::Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let temp_path = temp_dir.path();

    // Render SVG frames
    let global_min = predictions.iter()
        .flat_map(|r| r.iter()).copied().fold(f32::MAX, f32::min);
    let global_max = predictions.iter()
        .flat_map(|r| r.iter()).copied().fold(f32::MIN, f32::max);

    for (i, row) in predictions.iter().enumerate() {
        let mut frame_config = config.clone();
        frame_config.vmin = Some(global_min);
        frame_config.vmax = Some(global_max);
        frame_config.title = Some(format!("t = {}s", i));

        let svg = render_brain_svg(row, brain, &frame_config);
        let svg_path = temp_path.join(format!("frame_{:05}.svg", i));
        std::fs::write(&svg_path, &svg)?;

        // Convert SVG to PNG using rsvg-convert (if available)
        let png_path = temp_path.join(format!("frame_{:05}.png", i));
        let result = std::process::Command::new("rsvg-convert")
            .args(["-o", png_path.to_str().unwrap_or("")])
            .arg(svg_path.to_str().unwrap_or(""))
            .output();

        if result.is_err() || !result.as_ref().unwrap().status.success() {
            // Fallback: try Inkscape
            let _ = std::process::Command::new("inkscape")
                .args(["--export-type=png"])
                .args(["--export-filename", png_path.to_str().unwrap_or("")])
                .arg(svg_path.to_str().unwrap_or(""))
                .output();
        }
    }

    // Assemble MP4 with ffmpeg
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-framerate", &fps.to_string()])
        .args(["-i", &format!("{}/frame_%05d.png", temp_path.display())])
        .args(["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p"])
        .arg(output_path)
        .status()
        .map_err(|e| anyhow::anyhow!("ffmpeg not found: {}", e))?;

    if !status.success() {
        anyhow::bail!("ffmpeg failed with status {}", status);
    }

    Ok(())
}

// ── Combine mosaics ───────────────────────────────────────────────────────

/// Combine multiple SVG images into a mosaic layout.
///
/// Mirrors `plotting/utils.py combine_mosaics()`.
///
/// `svgs`: list of SVG strings to combine.
/// `cols`: number of columns in the mosaic.
/// `gap`: gap between cells in pixels.
pub fn combine_svgs(
    svgs: &[String],
    cell_width: u32,
    cell_height: u32,
    cols: usize,
    gap: u32,
) -> String {
    let n = svgs.len();
    let rows = (n + cols - 1) / cols;
    let total_w = cols as u32 * (cell_width + gap) - gap;
    let total_h = rows as u32 * (cell_height + gap) - gap;

    let mut combined = format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">
<rect width="{}" height="{}" fill="white"/>\n"#,
        total_w, total_h, total_w, total_h, total_w, total_h
    );

    for (i, svg) in svgs.iter().enumerate() {
        let col = i % cols;
        let row = i / cols;
        let x = col as u32 * (cell_width + gap);
        let y = row as u32 * (cell_height + gap);
        combined.push_str(&format!(r#"<g transform="translate({}, {})">
"#, x, y));
        if let Some(inner) = extract_svg_inner(svg) {
            combined.push_str(inner);
        }
        combined.push_str("</g>\n");
    }

    combined.push_str("</svg>\n");
    combined
}

// ── Rainbow brain ─────────────────────────────────────────────────────────

/// Generate a rainbow-coloured brain map based on spherical coordinates.
///
/// Each vertex is coloured by its angular position on the sphere,
/// producing a unique colour fingerprint for each cortical location.
///
/// Mirrors `plotting/utils.py get_rainbow_brain()`.
///
/// `coords`: vertex positions [n_vertices * 3] (flattened xyz).
/// Returns per-vertex (r, g, b) in 0–255.
pub fn rainbow_brain(coords: &[f32], n_vertices: usize) -> Vec<(u8, u8, u8)> {
    let mut result = Vec::with_capacity(n_vertices);
    for i in 0..n_vertices {
        let x = coords.get(i * 3).copied().unwrap_or(0.0);
        let y = coords.get(i * 3 + 1).copied().unwrap_or(0.0);
        let z = coords.get(i * 3 + 2).copied().unwrap_or(0.0);

        // Hue from longitude
        let phi = y.atan2(x);
        let hue = (phi + std::f32::consts::PI) / (2.0 * std::f32::consts::PI);

        // Value from elevation
        let r_dist = (x * x + y * y + z * z).sqrt();
        let z_norm = if r_dist > 1e-6 { (z / r_dist + 1.0) / 2.0 } else { 0.5 };
        let val = (0.8 + z_norm * 0.3).min(1.0);

        // HSV to RGB
        let sat = 0.9;
        let (r, g, b) = hsv_to_rgb(hue, sat, val);
        result.push((
            (r * 255.0) as u8,
            (g * 255.0) as u8,
            (b * 255.0) as u8,
        ));
    }
    result
}

/// P-value significance stars.
///
/// Mirrors `plotting/utils.py get_pval_stars()`.
pub fn pval_stars(pval: f64) -> &'static str {
    if pval < 0.0005 { "***" }
    else if pval < 0.005 { "**" }
    else if pval < 0.05 { "*" }
    else { "" }
}

/// Standalone colorbar as SVG.
///
/// Mirrors `plotting/utils.py plot_colorbar()`.
pub fn render_colorbar_svg(
    cmap: ColorMap,
    vmin: f32,
    vmax: f32,
    width: u32,
    height: u32,
    label: Option<&str>,
    orientation: &str, // "vertical" or "horizontal"
) -> String {
    let n_steps = 64;
    let mut svg = format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">
<rect width="{}" height="{}" fill="white"/>\n"#,
        width, height, width, height, width, height
    );

    if orientation == "horizontal" {
        let bar_w = width as f32 * 0.8;
        let bar_h = height as f32 * 0.3;
        let bar_x = (width as f32 - bar_w) / 2.0;
        let bar_y = height as f32 * 0.3;

        for i in 0..n_steps {
            let t = i as f32 / n_steps as f32;
            let (r, g, b) = cmap.map(t);
            let x = bar_x + t * bar_w;
            let w = bar_w / n_steps as f32 + 0.5;
            svg.push_str(&format!(
                r#"<rect x="{:.1}" y="{:.1}" width="{:.1}" height="{:.1}" fill="rgb({},{},{})"/>\n"#,
                x, bar_y, w, bar_h, r, g, b
            ));
        }
        svg.push_str(&format!(
            r#"<text x="{:.1}" y="{:.1}" font-size="10" font-family="sans-serif" text-anchor="start">{:.2}</text>\n"#,
            bar_x, bar_y + bar_h + 15.0, vmin
        ));
        svg.push_str(&format!(
            r#"<text x="{:.1}" y="{:.1}" font-size="10" font-family="sans-serif" text-anchor="end">{:.2}</text>\n"#,
            bar_x + bar_w, bar_y + bar_h + 15.0, vmax
        ));
    } else {
        let bar_w = width as f32 * 0.3;
        let bar_h = height as f32 * 0.8;
        let bar_x = (width as f32 - bar_w) / 2.0;
        let bar_y = (height as f32 - bar_h) / 2.0;

        for i in 0..n_steps {
            let t = 1.0 - i as f32 / n_steps as f32;
            let (r, g, b) = cmap.map(t);
            let y = bar_y + (i as f32 / n_steps as f32) * bar_h;
            let h = bar_h / n_steps as f32 + 0.5;
            svg.push_str(&format!(
                r#"<rect x="{:.1}" y="{:.1}" width="{:.1}" height="{:.1}" fill="rgb({},{},{})"/>\n"#,
                bar_x, y, bar_w, h, r, g, b
            ));
        }
        svg.push_str(&format!(
            r#"<text x="{:.1}" y="{:.1}" font-size="10" font-family="sans-serif">{:.2}</text>\n"#,
            bar_x + bar_w + 5.0, bar_y + 10.0, vmax
        ));
        svg.push_str(&format!(
            r#"<text x="{:.1}" y="{:.1}" font-size="10" font-family="sans-serif">{:.2}</text>\n"#,
            bar_x + bar_w + 5.0, bar_y + bar_h, vmin
        ));
    }

    if let Some(lbl) = label {
        svg.push_str(&format!(
            r#"<text x="{}" y="{}" font-size="12" font-family="sans-serif" text-anchor="middle">{}</text>\n"#,
            width / 2, height - 5, lbl
        ));
    }

    svg.push_str("</svg>\n");
    svg
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let h = h * 6.0;
    let i = h.floor() as i32;
    let f = h - i as f32;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    match i % 6 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_data() {
        let data = vec![0.0, 5.0, 10.0];
        let norm = normalize_data(&data, 0.0, 10.0);
        assert_eq!(norm, vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn test_robust_normalize() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let norm = robust_normalize(&data, 99.0);
        assert!(norm[0] <= 0.01);
        assert!(norm[10] >= 0.99);
    }

    #[test]
    fn test_split_hemispheres() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let (left, right) = split_hemispheres(&data);
        assert_eq!(left, &[1.0, 2.0]);
        assert_eq!(right, &[3.0, 4.0]);
    }

    #[test]
    fn test_generate_test_mesh() {
        let brain = generate_test_mesh(100);
        assert!(brain.left.mesh.n_vertices > 0);
        assert!(brain.right.mesh.n_vertices > 0);
        assert!(brain.left.mesh.n_faces > 0);
    }

    #[test]
    fn test_render_hemisphere_svg() {
        let brain = generate_test_mesh(100);
        let data: Vec<f32> = (0..brain.left.mesh.n_vertices)
            .map(|i| i as f32 / brain.left.mesh.n_vertices as f32)
            .collect();
        let config = PlotConfig::default();
        let svg = render_hemisphere_svg(
            &data,
            &brain.left.mesh,
            &brain.left.bg_map,
            &config,
        );
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("<polygon"));
    }

    #[test]
    fn test_render_brain_svg_left() {
        let brain = generate_test_mesh(100);
        let n_total = brain.left.mesh.n_vertices + brain.right.mesh.n_vertices;
        let data: Vec<f32> = (0..n_total).map(|i| (i as f32).sin()).collect();
        let config = PlotConfig { view: View::Left, ..Default::default() };
        let svg = render_brain_svg(&data, &brain, &config);
        assert!(svg.contains("<svg"));
    }

    #[test]
    fn test_colormap_hot() {
        let (r, g, b) = ColorMap::Hot.map(0.0);
        assert_eq!((r, g, b), (0, 0, 0));
        let (r, _, _) = ColorMap::Hot.map(1.0);
        assert_eq!(r, 255);
    }

    #[test]
    fn test_summarize_by_roi() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let labels = vec!["A".into(), "A".into(), "B".into(), "B".into()];
        let summary = summarize_by_roi(&data, &labels);
        assert!((summary["A"] - 1.5).abs() < 1e-6);
        assert!((summary["B"] - 3.5).abs() < 1e-6);
    }

    #[test]
    fn test_top_k_rois() {
        let data = vec![1.0, 2.0, 5.0, 6.0, 0.1, 0.2];
        let labels = vec!["A".into(), "A".into(), "B".into(), "B".into(), "C".into(), "C".into()];
        let top = top_k_rois(&data, &labels, 2);
        assert_eq!(top[0].0, "B");
        assert_eq!(top[1].0, "A");
    }

    #[test]
    fn test_render_with_colorbar() {
        let brain = generate_test_mesh(50);
        let data: Vec<f32> = (0..brain.left.mesh.n_vertices)
            .map(|i| i as f32 * 0.01)
            .collect();
        let config = PlotConfig {
            colorbar: true,
            cmap: ColorMap::CoolWarm,
            ..Default::default()
        };
        let svg = render_hemisphere_svg(&data, &brain.left.mesh, &brain.left.bg_map, &config);
        assert!(svg.contains("rgb("));
    }
}
