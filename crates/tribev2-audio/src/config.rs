//! Configuration for Wav2Vec-BERT 2.0.
//!
//! Default values match `facebook/w2v-bert-2.0` from HuggingFace.

use serde::Deserialize;

/// Full model configuration mirroring HuggingFace `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct Wav2VecBertConfig {
    // ── Feature encoder (CNN) ────────────────────────────────────
    /// Conv layer channel sizes.
    #[serde(default = "default_conv_dim")]
    pub conv_dim: Vec<usize>,
    /// Conv kernel sizes.
    #[serde(default = "default_conv_kernel")]
    pub conv_kernel: Vec<usize>,
    /// Conv strides.
    #[serde(default = "default_conv_stride")]
    pub conv_stride: Vec<usize>,

    // ── Adapter ──────────────────────────────────────────────────
    /// Adapter kernel sizes (strided conv subsampler).
    #[serde(default = "default_adapter_kernel")]
    pub adapter_kernel_size: Vec<usize>,
    /// Adapter strides.
    #[serde(default = "default_adapter_stride")]
    pub adapter_stride: Vec<usize>,

    // ── Transformer / Conformer ──────────────────────────────────
    /// Hidden dimension of the conformer.
    #[serde(default = "default_hidden")]
    pub hidden_size: usize,
    /// Number of conformer layers.
    #[serde(default = "default_num_layers")]
    pub num_hidden_layers: usize,
    /// Number of attention heads.
    #[serde(default = "default_num_heads")]
    pub num_attention_heads: usize,
    /// Intermediate (feed-forward) dimension.
    #[serde(default = "default_intermediate")]
    pub intermediate_size: usize,
    /// Hidden activation function.
    #[serde(default = "default_act")]
    pub hidden_act: String,
    /// Dropout probability.
    #[serde(default)]
    pub hidden_dropout: f64,
    /// Attention dropout.
    #[serde(default)]
    pub attention_dropout: f64,
    /// Layer norm epsilon.
    #[serde(default = "default_eps")]
    pub layer_norm_eps: f64,
    /// Feature projection input dim (output of CNN).
    #[serde(default = "default_conv_output")]
    pub output_hidden_size: usize,
    /// Conformer conv depthwise kernel size.
    #[serde(default = "default_conformer_kernel")]
    pub conformer_conv_kernel_size: usize,

    // ── Feature extraction config ────────────────────────────────
    /// Layer positions to extract (fractional 0.0–1.0).
    #[serde(default = "default_layer_positions")]
    pub layer_positions: Vec<f64>,
    /// Output feature frequency in Hz.
    #[serde(default = "default_frequency")]
    pub frequency: f64,
    /// Input audio sample rate.
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,
}

fn default_conv_dim() -> Vec<usize> {
    vec![512, 512, 512, 512, 512, 512, 512]
}
fn default_conv_kernel() -> Vec<usize> {
    vec![10, 3, 3, 3, 3, 2, 2]
}
fn default_conv_stride() -> Vec<usize> {
    vec![5, 2, 2, 2, 2, 2, 2]
}
fn default_adapter_kernel() -> Vec<usize> { vec![3, 3] }
fn default_adapter_stride() -> Vec<usize> { vec![2, 2] }
fn default_hidden() -> usize { 1024 }
fn default_num_layers() -> usize { 24 }
fn default_num_heads() -> usize { 16 }
fn default_intermediate() -> usize { 4096 }
fn default_act() -> String { "swish".into() }
fn default_eps() -> f64 { 1e-5 }
fn default_conv_output() -> usize { 512 }
fn default_conformer_kernel() -> usize { 31 }
fn default_layer_positions() -> Vec<f64> { vec![0.5, 0.75, 1.0] }
fn default_frequency() -> f64 { 2.0 }
fn default_sample_rate() -> u32 { 16000 }

impl Default for Wav2VecBertConfig {
    fn default() -> Self {
        Self {
            conv_dim: default_conv_dim(),
            conv_kernel: default_conv_kernel(),
            conv_stride: default_conv_stride(),
            adapter_kernel_size: default_adapter_kernel(),
            adapter_stride: default_adapter_stride(),
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4096,
            hidden_act: "swish".into(),
            hidden_dropout: 0.0,
            attention_dropout: 0.0,
            layer_norm_eps: 1e-5,
            output_hidden_size: 512,
            conformer_conv_kernel_size: 31,
            layer_positions: default_layer_positions(),
            frequency: 2.0,
            sample_rate: 16000,
        }
    }
}

impl Wav2VecBertConfig {
    /// Load from HuggingFace `config.json`.
    pub fn from_json(path: &str) -> anyhow::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&json)?)
    }

    /// Compute which layer indices to extract.
    pub fn layer_indices(&self) -> Vec<usize> {
        self.layer_positions
            .iter()
            .map(|&f| {
                let idx = (f * (self.num_hidden_layers as f64 - 1.0)).floor() as usize;
                idx.min(self.num_hidden_layers - 1)
            })
            .collect()
    }

    /// Total downsampling factor of the feature encoder CNN.
    pub fn cnn_downsample_factor(&self) -> usize {
        self.conv_stride.iter().product()
    }

    /// Total downsampling factor including adapter.
    pub fn total_downsample_factor(&self) -> usize {
        let cnn: usize = self.conv_stride.iter().product();
        let adapter: usize = self.adapter_stride.iter().product();
        cnn * adapter
    }
}
