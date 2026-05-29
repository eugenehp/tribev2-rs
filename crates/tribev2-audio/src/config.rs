//! TRIBE v2 audio feature extraction configuration.

use serde::Deserialize;

/// Extraction settings (paths + layer selection for TRIBE v2).
#[derive(Debug, Clone, Deserialize)]
pub struct AudioFeatureConfig {
    /// Safetensors or GGUF weights (file or directory).
    pub weights_path: String,
    /// Optional `config.json` (defaults to sibling of weights).
    pub config_path: Option<String>,
    /// Optional `preprocessor_config.json`.
    pub preprocessor_config_path: Option<String>,
    /// Layer positions to extract (fractional 0.0–1.0). Default: [0.5, 0.75, 1.0]
    #[serde(default = "default_layer_positions")]
    pub layer_positions: Vec<f64>,
    /// Override layer count (else read from model config).
    pub n_layers: Option<usize>,
    /// Output feature frequency in Hz.
    #[serde(default = "default_frequency")]
    pub frequency: f64,
    /// RLX device: `cpu`, `metal`, `cuda`, `gpu`, `vulkan`, …
    #[serde(default = "default_device")]
    pub device: String,
    /// Compiled prefill sequence length (log-mel frames). Auto-estimated when unset.
    pub max_seq: Option<usize>,
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

impl Default for AudioFeatureConfig {
    fn default() -> Self {
        Self {
            weights_path: String::new(),
            config_path: None,
            preprocessor_config_path: None,
            layer_positions: default_layer_positions(),
            n_layers: None,
            frequency: default_frequency(),
            device: default_device(),
            max_seq: None,
        }
    }
}

impl AudioFeatureConfig {
    pub fn layer_indices(&self, n_total_layers: usize) -> Vec<usize> {
        self.layer_positions
            .iter()
            .map(|&f| {
                let idx = (f * (n_total_layers as f64 - 1.0)).floor() as usize;
                idx.min(n_total_layers - 1)
            })
            .collect()
    }
}
