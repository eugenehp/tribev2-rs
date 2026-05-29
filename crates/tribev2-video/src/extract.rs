//! V-JEPA2 ViT-G feature extraction via RLX (`rlx-vjepa2`).

use anyhow::{Context, Result};
use rlx::Device;
use rlx_vjepa2::{
    Vjepa2Config, encode_video_native_ext, extract_model_weights, normalize_video_hwc,
};
use rlx_models_core::{load_weight_map, VJEPA2_GGUF_ARCHES};
use std::path::Path;

use crate::config::VideoFeatureConfig;
use crate::video_io;
use crate::ExtractedVideoFeatures;

fn parse_device(s: &str) -> Result<Device> {
    Ok(match s.trim().to_ascii_lowercase().as_str() {
        "cpu" => Device::Cpu,
        "metal" | "mps" => Device::Metal,
        "mlx" => Device::Mlx,
        "cuda" => Device::Cuda,
        "rocm" | "hip" => Device::Rocm,
        "gpu" | "wgpu" => Device::Gpu,
        "vulkan" => Device::Vulkan,
        other => anyhow::bail!(
            "unknown device {other} (cpu|metal|mlx|cuda|rocm|gpu|vulkan)"
        ),
    })
}

/// Mean-pool spatial tokens: `[batch, seq, hidden]` → `[hidden]`.
fn mean_pool_tokens(tokens: &[f32], seq: usize, hidden: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; hidden];
    if seq == 0 {
        return out;
    }
    for t in 0..seq {
        for d in 0..hidden {
            out[d] += tokens[t * hidden + d];
        }
    }
    let n = seq as f32;
    for v in &mut out {
        *v /= n;
    }
    out
}

/// Extract features from a video file path.
pub fn extract_video_features(
    config: &VideoFeatureConfig,
    video_path: &str,
    duration_secs: f64,
    verbose: bool,
) -> Result<ExtractedVideoFeatures> {
    let tmp = tempfile::tempdir()?;
    let frames_dir = tmp.path().join("frames");
    let frame_paths =
        video_io::extract_frames(video_path, config.fps, frames_dir.to_str().unwrap())?;
    extract_video_features_from_frames(config, &frame_paths, duration_secs, verbose)
}

/// Extract from sorted frame image paths (PNG/JPEG).
pub fn extract_video_features_from_frames(
    config: &VideoFeatureConfig,
    frame_paths: &[String],
    duration_secs: f64,
    verbose: bool,
) -> Result<ExtractedVideoFeatures> {
    let weights_path = Path::new(&config.weights_path);
    let weights_dir = weights_path
        .parent()
        .context("weights path has no parent directory")?;
    let cfg_path: std::path::PathBuf = config
        .config_path
        .as_ref()
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| weights_dir.join("config.json"));

    let mut rlx_cfg = if cfg_path.exists() {
        Vjepa2Config::from_file(&cfg_path)?
    } else {
        Vjepa2Config::vit_g_384()
    };
    rlx_cfg.frames_per_clip = config.frames_per_clip;
    rlx_cfg.crop_size = config.img_size;

    let path_str = weights_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("non-UTF-8 weights path"))?;
    let mut wm = load_weight_map(path_str, VJEPA2_GGUF_ARCHES)?;
    let model = extract_model_weights(&mut wm, &rlx_cfg)?;

    let n_total_layers = config.n_layers.unwrap_or(rlx_cfg.num_hidden_layers);
    let layer_indices = config.layer_indices(n_total_layers);
    let hidden_dim = rlx_cfg.hidden_size;
    let n_layer_groups = layer_indices.len();
    let n_timesteps = (duration_secs * config.frequency).ceil() as usize;
    let seq = rlx_cfg.num_patches();

    let _device = parse_device(&config.device)?;

    if verbose {
        eprintln!(
            "V-JEPA2 (RLX): {} layers, hidden_dim={}, patches={}",
            n_total_layers, hidden_dim, seq
        );
        eprintln!("Extracting layers: {:?}", layer_indices);
        eprintln!("Frames: {}, clips of {}", frame_paths.len(), config.frames_per_clip);
    }

    let crop = config.img_size;
    let fpc = config.frames_per_clip;
    let clip_stride = fpc;

    let mut clip_features: Vec<Vec<Vec<f32>>> = Vec::new();

    let mut start = 0usize;
    while start + fpc <= frame_paths.len() {
        let mut hwc_u8 = Vec::with_capacity(fpc * crop * crop * 3);
        for path in &frame_paths[start..start + fpc] {
            let (data, h, w) = video_io::load_frame(path)?;
            let resized = if h != crop || w != crop {
                video_io::resize_frame(&data, h, w, crop, crop)
            } else {
                data
            };
            for y in 0..crop {
                for x in 0..crop {
                    for c in 0..3 {
                        let v = resized[(y * crop + x) * 3 + c].clamp(0.0, 1.0);
                        hwc_u8.push((v * 255.0).round() as u8);
                    }
                }
            }
        }

        let ncthw = normalize_video_hwc(&hwc_u8, fpc, crop);
        let mut per_layer = Vec::with_capacity(n_layer_groups);
        let last_block = n_total_layers.saturating_sub(1);

        for &layer_idx in &layer_indices {
            let out = encode_video_native_ext(
                &model.encoder,
                &rlx_cfg,
                &ncthw,
                1,
                Some(layer_idx),
            )?;
            let pooled = mean_pool_tokens(&out.tokens, out.seq, out.hidden);
            per_layer.push(pooled);
            let _ = last_block;
        }
        clip_features.push(per_layer);
        start += clip_stride;
    }

    if clip_features.is_empty() {
        return Ok(ExtractedVideoFeatures {
            data: vec![0.0; n_layer_groups * hidden_dim * n_timesteps.max(1)],
            shape: vec![n_layer_groups, hidden_dim, n_timesteps],
            n_layers: n_layer_groups,
            feature_dim: hidden_dim,
            n_timesteps,
        });
    }

    let clip_duration = config.clip_duration();
    let mut data = vec![0.0f32; n_layer_groups * hidden_dim * n_timesteps.max(1)];

    for ti in 0..n_timesteps {
        let t_sec = ti as f64 / config.frequency;
        let clip_idx =
            ((t_sec / clip_duration).floor() as usize).min(clip_features.len().saturating_sub(1));
        for li in 0..n_layer_groups {
            let feat = &clip_features[clip_idx][li];
            for di in 0..hidden_dim {
                data[li * hidden_dim * n_timesteps + di * n_timesteps + ti] = feat[di];
            }
        }
    }

    Ok(ExtractedVideoFeatures {
        data,
        shape: vec![n_layer_groups, hidden_dim, n_timesteps],
        n_layers: n_layer_groups,
        feature_dim: hidden_dim,
        n_timesteps,
    })
}
