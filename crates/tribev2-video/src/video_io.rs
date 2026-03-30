//! Video frame extraction — extract and preprocess frames for V-JEPA2.
//!
//! Uses ffmpeg to extract frames, then loads and normalizes them.

use anyhow::{Context, Result};

/// Extract frames from a video file at the given FPS using ffmpeg.
///
/// Returns paths to extracted frame images in a temp directory.
/// The caller is responsible for keeping the tempdir alive.
pub fn extract_frames(
    video_path: &str,
    fps: f64,
    output_dir: &str,
) -> Result<Vec<String>> {
    std::fs::create_dir_all(output_dir)?;

    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-i", video_path])
        .args(["-vf", &format!("fps={}", fps)])
        .args(["-q:v", "2"])
        .arg(format!("{}/frame_%06d.png", output_dir))
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .with_context(|| "ffmpeg not found")?;

    if !status.success() {
        anyhow::bail!("ffmpeg failed extracting frames from {}", video_path);
    }

    let mut frames: Vec<String> = std::fs::read_dir(output_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map_or(false, |ext| ext == "png" || ext == "jpg")
        })
        .map(|e| e.path().to_string_lossy().to_string())
        .collect();
    frames.sort();
    Ok(frames)
}

/// Load a single frame as [H, W, 3] f32 normalized to [0, 1].
pub fn load_frame(path: &str) -> Result<(Vec<f32>, usize, usize)> {
    let img = image::open(path)
        .with_context(|| format!("failed to load frame: {}", path))?
        .to_rgb8();

    let (w, h) = (img.width() as usize, img.height() as usize);
    let data: Vec<f32> = img.into_raw().iter().map(|&v| v as f32 / 255.0).collect();

    Ok((data, h, w))
}

/// Resize a frame to target size using bilinear interpolation.
///
/// Input: [H, W, 3] f32 in [0, 1]
/// Output: [target_h, target_w, 3] f32 in [0, 1]
pub fn resize_frame(
    data: &[f32],
    src_h: usize,
    src_w: usize,
    target_h: usize,
    target_w: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; target_h * target_w * 3];

    for y in 0..target_h {
        for x in 0..target_w {
            let src_y = y as f64 * (src_h as f64 - 1.0) / (target_h as f64 - 1.0).max(1.0);
            let src_x = x as f64 * (src_w as f64 - 1.0) / (target_w as f64 - 1.0).max(1.0);

            let y0 = src_y.floor() as usize;
            let x0 = src_x.floor() as usize;
            let y1 = (y0 + 1).min(src_h - 1);
            let x1 = (x0 + 1).min(src_w - 1);

            let fy = src_y - y0 as f64;
            let fx = src_x - x0 as f64;

            for c in 0..3 {
                let v00 = data[(y0 * src_w + x0) * 3 + c] as f64;
                let v01 = data[(y0 * src_w + x1) * 3 + c] as f64;
                let v10 = data[(y1 * src_w + x0) * 3 + c] as f64;
                let v11 = data[(y1 * src_w + x1) * 3 + c] as f64;

                let v = v00 * (1.0 - fx) * (1.0 - fy)
                    + v01 * fx * (1.0 - fy)
                    + v10 * (1.0 - fx) * fy
                    + v11 * fx * fy;
                out[(y * target_w + x) * 3 + c] = v as f32;
            }
        }
    }
    out
}

/// Normalize a frame with ImageNet mean/std.
///
/// Input/output: [H, W, 3] in [0, 1] → normalized.
pub fn normalize_frame(data: &mut [f32], mean: &[f32; 3], std: &[f32; 3]) {
    let n_pixels = data.len() / 3;
    for i in 0..n_pixels {
        for c in 0..3 {
            data[i * 3 + c] = (data[i * 3 + c] - mean[c]) / std[c];
        }
    }
}

/// Get video duration in seconds using ffprobe.
pub fn video_duration(path: &str) -> Result<f64> {
    let output = std::process::Command::new("ffprobe")
        .args(["-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0"])
        .arg(path)
        .output()
        .with_context(|| "ffprobe not found")?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let dur: f64 = stdout.trim().parse()
        .with_context(|| format!("failed to parse duration: '{}'", stdout.trim()))?;
    Ok(dur)
}

/// Get video FPS using ffprobe.
pub fn video_fps(path: &str) -> Result<f64> {
    let output = std::process::Command::new("ffprobe")
        .args(["-v", "quiet", "-select_streams", "v:0"])
        .args(["-show_entries", "stream=r_frame_rate"])
        .args(["-of", "csv=p=0"])
        .arg(path)
        .output()
        .with_context(|| "ffprobe not found")?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let fps_str = stdout.trim();
    // Format can be "30/1" or "29.97"
    if fps_str.contains('/') {
        let parts: Vec<&str> = fps_str.split('/').collect();
        let num: f64 = parts[0].parse().unwrap_or(30.0);
        let den: f64 = parts[1].parse().unwrap_or(1.0);
        Ok(num / den)
    } else {
        Ok(fps_str.parse().unwrap_or(30.0))
    }
}
