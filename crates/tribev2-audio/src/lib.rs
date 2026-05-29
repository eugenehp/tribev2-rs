//! # tribev2-audio — Wav2Vec-BERT 2.0 audio feature extraction (RLX)
//!
//! Extracts intermediate-layer hidden states from Wav2Vec-BERT 2.0
//! (`facebook/w2v-bert-2.0`) for TRIBE v2 using [`rlx-wav2vec2-bert`].
//!
//! ## Backends
//!
//! Enable at build time (same pattern as [zuna-rs](https://github.com/eugenehp/zuna-rs)):
//!
//! | Feature | Backend |
//! |---------|---------|
//! | `rlx-metal` *(default)* | Apple Metal |
//! | `rlx-cuda` | NVIDIA CUDA |
//! | `rlx-vulkan` | Vulkan |
//! | `rlx-cpu` | CPU |

pub mod audio_io;
pub mod config;
pub mod extract;

pub use config::AudioFeatureConfig;
pub use extract::extract_audio_features;

/// Extracted audio features ready for TRIBE v2.
#[derive(Debug, Clone)]
pub struct ExtractedAudioFeatures {
    /// Feature data: [n_layers, feature_dim, n_timesteps]
    pub data: Vec<f32>,
    /// Shape: [n_layers, feature_dim, n_timesteps]
    pub shape: Vec<usize>,
    pub n_layers: usize,
    pub feature_dim: usize,
    pub n_timesteps: usize,
}
