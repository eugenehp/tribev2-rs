//! # tribev2-video — V-JEPA2 ViT-G video feature extraction
//!
//! Extracts intermediate-layer hidden states from V-JEPA2 ViT-Giant
//! (`facebook/vjepa2-vitg-fpc64-256`) for use as video features in TRIBE v2.
//!
//! Architecture (ViT-Giant, ~1.1B parameters):
//! - 3D patch embedding: (2×16×16) patches from video clips (T×256×256)
//! - 40 transformer layers, hidden_dim=1408, 16 heads
//! - Each layer: LayerNorm → Attention → Residual → LayerNorm → MLP → Residual
//!
//! Video processing pipeline:
//! 1. Extract frames from video at target FPS (e.g. 16 FPS)
//! 2. Group frames into clips (e.g. 64 frames = 4s at 16 FPS)
//! 3. Resize to 256×256, normalize
//! 4. Create 3D patches → patch embeddings + positional embeddings
//! 5. Run through ViT layers, extract hidden states at selected layers
//! 6. Mean-pool spatial tokens → per-clip features
//! 7. Resample to output frequency (2 Hz)

pub mod config;
pub mod video_io;
pub mod patch_embed;
pub mod vit;
pub mod model;
pub mod weights;

pub use config::VJepa2Config;
pub use model::{VJepa2, VJepa2WithConfig, ExtractedVideoFeatures};
