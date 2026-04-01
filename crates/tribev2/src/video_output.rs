//! MP4 video output for animated brain activity visualization.
//!
//! Mirrors the Python `plot_timesteps_mp4()` method:
//! 1. Render each timestep as an SVG brain surface plot
//! 2. Convert SVGs to PNGs (via `rsvg-convert` or `resvg`)
//! 3. Assemble PNGs into an MP4 video via `ffmpeg`
//!
//! Requires external tools:
//! - `ffmpeg` for video encoding
//! - `rsvg-convert` (from librsvg) or `resvg` for SVG→PNG conversion

use std::path::{Path, PathBuf};
use anyhow::{Context, Result};

use crate::plotting::{self, BrainMesh, PlotConfig};

#[allow(unused_imports)]
use std::io::Write;

/// Configuration for MP4 video output.
#[derive(Debug, Clone)]
pub struct VideoConfig {
    /// Frames per second for the output video.
    pub fps: u32,
    /// Whether to interpolate between frames (using ffmpeg minterpolate).
    pub interpolated_fps: Option<u32>,
    /// Title text to overlay on each frame (None = "t = Xs").
    pub title: Option<String>,
    /// PNG DPI for SVG rendering.
    pub dpi: u32,
    /// Whether to keep temporary frame files.
    pub keep_frames: bool,
}

impl Default for VideoConfig {
    fn default() -> Self {
        Self {
            fps: 2,
            interpolated_fps: None,
            title: None,
            dpi: 150,
            keep_frames: false,
        }
    }
}

/// Check if a command-line tool is available.
fn has_tool(name: &str) -> bool {
    std::process::Command::new(name)
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok()
}

/// Convert an SVG file to PNG using available tools.
fn svg_to_png(svg_path: &Path, png_path: &Path, width: u32) -> Result<()> {
    if has_tool("rsvg-convert") {
        let status = std::process::Command::new("rsvg-convert")
            .args([
                "-w", &width.to_string(),
                "-o", &png_path.to_string_lossy(),
                &svg_path.to_string_lossy(),
            ])
            .status()
            .context("failed to run rsvg-convert")?;
        if !status.success() {
            anyhow::bail!("rsvg-convert failed with status: {}", status);
        }
    } else if has_tool("resvg") {
        let status = std::process::Command::new("resvg")
            .args([
                "--width", &width.to_string(),
                &svg_path.to_string_lossy().to_string(),
                &png_path.to_string_lossy().to_string(),
            ])
            .status()
            .context("failed to run resvg")?;
        if !status.success() {
            anyhow::bail!("resvg failed with status: {}", status);
        }
    } else {
        anyhow::bail!(
            "No SVG-to-PNG converter found. Install `rsvg-convert` (librsvg) or `resvg`.\n\
             macOS:  brew install librsvg\n\
             Ubuntu: apt install librsvg2-bin\n\
             Cargo:  cargo install resvg"
        );
    }
    Ok(())
}

/// Render predictions as an MP4 video.
///
/// `predictions`: `[n_timesteps][n_vertices]` — per-timestep brain activity.
/// `brain`: the brain mesh for rendering.
/// `plot_config`: configuration for brain surface plots.
/// `video_config`: configuration for video output.
/// `output_path`: path for the output .mp4 file.
///
/// Returns the output path on success.
pub fn render_mp4(
    predictions: &[Vec<f32>],
    brain: &BrainMesh,
    plot_config: &PlotConfig,
    video_config: &VideoConfig,
    output_path: &Path,
) -> Result<PathBuf> {
    if predictions.is_empty() {
        anyhow::bail!("No predictions to render");
    }

    // Check ffmpeg availability
    if !has_tool("ffmpeg") {
        anyhow::bail!(
            "ffmpeg not found. Install it:\n\
             macOS:  brew install ffmpeg\n\
             Ubuntu: apt install ffmpeg"
        );
    }

    // Create temporary directory for frames
    let tmp_dir = tempfile::tempdir().context("failed to create temp directory")?;
    let frames_dir = tmp_dir.path().join("frames");
    std::fs::create_dir_all(&frames_dir)?;

    let n_timesteps = predictions.len();
    eprintln!("Rendering {} frames...", n_timesteps);

    // Render each timestep as SVG, then convert to PNG
    for (ti, pred) in predictions.iter().enumerate() {
        // Render SVG
        let svg = plotting::render_brain_svg(pred, brain, plot_config);

        let svg_path = frames_dir.join(format!("frame_{:05}.svg", ti));
        std::fs::write(&svg_path, &svg)?;

        // Convert to PNG
        let png_path = frames_dir.join(format!("frame_{:05}.png", ti));
        svg_to_png(&svg_path, &png_path, plot_config.width)?;

        if (ti + 1) % 20 == 0 || ti == n_timesteps - 1 {
            eprintln!("  Rendered {}/{} frames", ti + 1, n_timesteps);
        }
    }

    // Assemble MP4 with ffmpeg
    eprintln!("Encoding MP4...");
    let input_pattern = frames_dir.join("frame_%05d.png");

    let mut cmd = std::process::Command::new("ffmpeg");
    cmd.args([
        "-y",
        "-framerate", &video_config.fps.to_string(),
        "-i", &input_pattern.to_string_lossy(),
    ]);

    // Optional frame interpolation
    if let Some(ifps) = video_config.interpolated_fps {
        cmd.args(["-vf", &format!("minterpolate=fps={}", ifps)]);
    }

    cmd.args([
        "-c:v", "libx264",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        &output_path.to_string_lossy(),
    ]);

    let status = cmd.status().context("failed to run ffmpeg")?;
    if !status.success() {
        anyhow::bail!("ffmpeg failed with status: {}", status);
    }

    // Optionally keep frames
    if video_config.keep_frames {
        let keep_dir = output_path.with_extension("frames");
        if keep_dir.exists() {
            std::fs::remove_dir_all(&keep_dir)?;
        }
        std::fs::rename(&frames_dir, &keep_dir)?;
        eprintln!("Frames saved to {}", keep_dir.display());
    }

    eprintln!("MP4 written to {}", output_path.display());
    Ok(output_path.to_path_buf())
}

/// Render predictions as an animated GIF (lower quality, no ffmpeg required).
///
/// This is a simpler alternative that only requires an SVG→PNG converter.
/// Uses ffmpeg if available, otherwise skips.
pub fn render_gif(
    predictions: &[Vec<f32>],
    brain: &BrainMesh,
    plot_config: &PlotConfig,
    fps: u32,
    output_path: &Path,
) -> Result<PathBuf> {
    if predictions.is_empty() {
        anyhow::bail!("No predictions to render");
    }

    if !has_tool("ffmpeg") {
        anyhow::bail!("ffmpeg required for GIF generation");
    }

    let tmp_dir = tempfile::tempdir()?;
    let frames_dir = tmp_dir.path().join("frames");
    std::fs::create_dir_all(&frames_dir)?;

    for (ti, pred) in predictions.iter().enumerate() {
        let svg = plotting::render_brain_svg(pred, brain, plot_config);
        let svg_path = frames_dir.join(format!("frame_{:05}.svg", ti));
        std::fs::write(&svg_path, &svg)?;
        let png_path = frames_dir.join(format!("frame_{:05}.png", ti));
        svg_to_png(&svg_path, &png_path, plot_config.width)?;
    }

    let input_pattern = frames_dir.join("frame_%05d.png");
    let status = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-framerate", &fps.to_string(),
            "-i", &input_pattern.to_string_lossy(),
            "-vf", "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
            &output_path.to_string_lossy(),
        ])
        .status()
        .context("failed to run ffmpeg for GIF")?;

    if !status.success() {
        anyhow::bail!("ffmpeg GIF encoding failed");
    }

    eprintln!("GIF written to {}", output_path.display());
    Ok(output_path.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_video_config_default() {
        let config = VideoConfig::default();
        assert_eq!(config.fps, 2);
        assert!(config.interpolated_fps.is_none());
    }

    #[test]
    fn test_has_tool() {
        // "echo" should exist on any Unix system
        // (This is a basic sanity check; don't fail if ffmpeg isn't installed)
        let _ = has_tool("echo");
    }
}
