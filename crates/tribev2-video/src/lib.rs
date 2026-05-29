//! # tribev2-video — V-JEPA2 ViT-G video feature extraction (RLX)
//!
//! Extracts intermediate-layer ViT hidden states for TRIBE v2 using
//! [`rlx-vjepa2`] (encoder trunk; per-layer probes via truncated blocks).

pub mod config;
pub mod extract;
pub mod video_io;

pub use config::VideoFeatureConfig;
pub use extract::{extract_video_features, extract_video_features_from_frames};

/// Extracted video features ready for TRIBE v2.
#[derive(Debug, Clone)]
pub struct ExtractedVideoFeatures {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub n_layers: usize,
    pub feature_dim: usize,
    pub n_timesteps: usize,
}
