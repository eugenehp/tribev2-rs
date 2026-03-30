//! Configuration for V-JEPA2 ViT-G.
//!
//! Default values match `facebook/vjepa2-vitg-fpc64-256`.

use serde::Deserialize;

/// Full model configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct VJepa2Config {
    // ── Patch embedding ──────────────────────────────────────────
    /// Input image size (spatial, square).
    #[serde(default = "default_img_size")]
    pub img_size: usize,
    /// Spatial patch size.
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    /// Temporal patch size (number of frames per temporal patch).
    #[serde(default = "default_temporal_patch_size")]
    pub temporal_patch_size: usize,
    /// Number of input channels (RGB = 3).
    #[serde(default = "default_in_chans")]
    pub in_chans: usize,

    // ── Transformer ──────────────────────────────────────────────
    /// Hidden dimension.
    #[serde(default = "default_embed_dim")]
    pub embed_dim: usize,
    /// Number of transformer layers.
    #[serde(default = "default_depth")]
    pub depth: usize,
    /// Number of attention heads.
    #[serde(default = "default_num_heads")]
    pub num_heads: usize,
    /// MLP ratio (inner_dim = embed_dim * mlp_ratio).
    #[serde(default = "default_mlp_ratio")]
    pub mlp_ratio: f64,
    /// Layer norm epsilon.
    #[serde(default = "default_eps")]
    pub layer_norm_eps: f64,

    // ── Video processing ─────────────────────────────────────────
    /// Frames per clip.
    #[serde(default = "default_fpc")]
    pub frames_per_clip: usize,
    /// Input FPS for frame extraction.
    #[serde(default = "default_fps")]
    pub fps: f64,
    /// Clip duration in seconds (frames_per_clip / fps).
    #[serde(default = "default_clip_duration")]
    pub clip_duration: f64,

    // ── Feature extraction ───────────────────────────────────────
    /// Layer positions to extract (fractional 0.0–1.0).
    #[serde(default = "default_layer_positions")]
    pub layer_positions: Vec<f64>,
    /// Output feature frequency in Hz.
    #[serde(default = "default_frequency")]
    pub frequency: f64,

    // ── Normalization ────────────────────────────────────────────
    /// ImageNet mean for normalization [R, G, B].
    #[serde(default = "default_mean")]
    pub normalize_mean: Vec<f32>,
    /// ImageNet std for normalization [R, G, B].
    #[serde(default = "default_std")]
    pub normalize_std: Vec<f32>,
}

fn default_img_size() -> usize { 256 }
fn default_patch_size() -> usize { 16 }
fn default_temporal_patch_size() -> usize { 2 }
fn default_in_chans() -> usize { 3 }
fn default_embed_dim() -> usize { 1408 }
fn default_depth() -> usize { 40 }
fn default_num_heads() -> usize { 16 }
fn default_mlp_ratio() -> f64 { 48.0 / 11.0 } // ≈ 4.36 for ViT-G
fn default_eps() -> f64 { 1e-6 }
fn default_fpc() -> usize { 64 }
fn default_fps() -> f64 { 16.0 }
fn default_clip_duration() -> f64 { 4.0 }
fn default_layer_positions() -> Vec<f64> { vec![0.5, 0.75, 1.0] }
fn default_frequency() -> f64 { 2.0 }
fn default_mean() -> Vec<f32> { vec![0.485, 0.456, 0.406] }
fn default_std() -> Vec<f32> { vec![0.229, 0.224, 0.225] }

impl Default for VJepa2Config {
    fn default() -> Self {
        Self {
            img_size: 256,
            patch_size: 16,
            temporal_patch_size: 2,
            in_chans: 3,
            embed_dim: 1408,
            depth: 40,
            num_heads: 16,
            mlp_ratio: 48.0 / 11.0,
            layer_norm_eps: 1e-6,
            frames_per_clip: 64,
            fps: 16.0,
            clip_duration: 4.0,
            layer_positions: default_layer_positions(),
            frequency: 2.0,
            normalize_mean: default_mean(),
            normalize_std: default_std(),
        }
    }
}

impl VJepa2Config {
    /// Load from a JSON config file.
    pub fn from_json(path: &str) -> anyhow::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&json)?)
    }

    /// Compute which layer indices to extract.
    pub fn layer_indices(&self) -> Vec<usize> {
        self.layer_positions
            .iter()
            .map(|&f| {
                let idx = (f * (self.depth as f64 - 1.0)).floor() as usize;
                idx.min(self.depth - 1)
            })
            .collect()
    }

    /// Number of spatial patches per frame.
    pub fn num_spatial_patches(&self) -> usize {
        (self.img_size / self.patch_size) * (self.img_size / self.patch_size)
    }

    /// Number of temporal patches per clip.
    pub fn num_temporal_patches(&self) -> usize {
        self.frames_per_clip / self.temporal_patch_size
    }

    /// Total number of patch tokens per clip.
    pub fn num_patches(&self) -> usize {
        self.num_spatial_patches() * self.num_temporal_patches()
    }

    /// MLP inner dimension.
    pub fn mlp_dim(&self) -> usize {
        (self.embed_dim as f64 * self.mlp_ratio) as usize
    }

    /// Attention head dimension.
    pub fn head_dim(&self) -> usize {
        self.embed_dim / self.num_heads
    }
}
