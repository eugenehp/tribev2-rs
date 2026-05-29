//! TRIBE v2 video feature extraction configuration.

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct VideoFeatureConfig {
    /// Safetensors or GGUF weights (file or directory).
    pub weights_path: String,
    /// Optional `config.json` beside weights.
    pub config_path: Option<String>,
    #[serde(default = "default_layer_positions")]
    pub layer_positions: Vec<f64>,
    pub n_layers: Option<usize>,
    #[serde(default = "default_frequency")]
    pub frequency: f64,
    #[serde(default = "default_device")]
    pub device: String,
    #[serde(default = "default_img_size")]
    pub img_size: usize,
    #[serde(default = "default_fpc")]
    pub frames_per_clip: usize,
    #[serde(default = "default_fps")]
    pub fps: f64,
}

fn default_layer_positions() -> Vec<f64> {
    vec![0.5, 0.75, 1.0]
}
fn default_frequency() -> f64 {
    2.0
}
fn default_device() -> String {
    if cfg!(feature = "rlx-cuda") {
        "cuda".into()
    } else if cfg!(all(feature = "rlx-metal", not(feature = "rlx-cuda"))) {
        "metal".into()
    } else {
        "cpu".into()
    }
}
fn default_img_size() -> usize {
    256
}
fn default_fpc() -> usize {
    64
}
fn default_fps() -> f64 {
    16.0
}

impl Default for VideoFeatureConfig {
    fn default() -> Self {
        Self {
            weights_path: String::new(),
            config_path: None,
            layer_positions: default_layer_positions(),
            n_layers: None,
            frequency: default_frequency(),
            device: default_device(),
            img_size: default_img_size(),
            frames_per_clip: default_fpc(),
            fps: default_fps(),
        }
    }
}

impl VideoFeatureConfig {
    pub fn layer_indices(&self, n_total_layers: usize) -> Vec<usize> {
        self.layer_positions
            .iter()
            .map(|&f| {
                let idx = (f * (n_total_layers as f64 - 1.0)).floor() as usize;
                idx.min(n_total_layers - 1)
            })
            .collect()
    }

    pub fn clip_duration(&self) -> f64 {
        self.frames_per_clip as f64 / self.fps
    }
}
