//! Full V-JEPA2 ViT-G model for video feature extraction.
//!
//! Pipeline:
//! 1. Extract frames from video at target FPS
//! 2. Group into clips of `frames_per_clip` frames
//! 3. Resize to img_size × img_size, normalize
//! 4. 3D patch embedding → [B, N, D]
//! 5. Add positional embedding
//! 6. Run through ViT layers, extract hidden states at selected layers
//! 7. Mean-pool spatial tokens → per-clip feature vector [D]
//! 8. Resample clips to output frequency (2 Hz)
//! 9. Return [n_layer_groups, D, n_timesteps]

use burn::prelude::*;
use crate::config::VJepa2Config;
use crate::patch_embed::{PatchEmbed3d, PositionalEmbedding};
use crate::vit::{ViTBlock, LayerNorm};

/// Extracted video features ready for TRIBE v2.
#[derive(Debug, Clone)]
pub struct ExtractedVideoFeatures {
    /// Feature data: [n_layers, feature_dim, n_timesteps]
    pub data: Vec<f32>,
    /// Shape.
    pub shape: Vec<usize>,
    pub n_layers: usize,
    pub feature_dim: usize,
    pub n_timesteps: usize,
}

/// Full V-JEPA2 ViT-G model (encoder only, no predictor/decoder).
#[derive(Module, Debug)]
pub struct VJepa2<B: Backend> {
    pub patch_embed: PatchEmbed3d<B>,
    pub pos_embed: PositionalEmbedding<B>,
    pub blocks: Vec<ViTBlock<B>>,
    pub norm: LayerNorm<B>,
}

/// Wrapper pairing the burn Module with its config.
pub struct VJepa2WithConfig<B: Backend> {
    pub model: VJepa2<B>,
    pub config: VJepa2Config,
}

impl<B: Backend> VJepa2WithConfig<B> {
    /// Build model from config (weights initialized to zero).
    pub fn new(config: &VJepa2Config, device: &B::Device) -> Self {
        let num_patches = config.num_patches();
        let mlp_dim = config.mlp_dim();

        let patch_embed = PatchEmbed3d::new(
            config.in_chans,
            config.embed_dim,
            config.img_size,
            config.patch_size,
            config.temporal_patch_size,
            config.frames_per_clip,
            device,
        );

        let pos_embed = PositionalEmbedding::new(num_patches, config.embed_dim, device);

        let blocks: Vec<ViTBlock<B>> = (0..config.depth)
            .map(|_| {
                ViTBlock::new(
                    config.embed_dim,
                    config.num_heads,
                    mlp_dim,
                    config.layer_norm_eps,
                    device,
                )
            })
            .collect();

        let norm = LayerNorm::new(config.embed_dim, config.layer_norm_eps, device);

        Self {
            model: VJepa2 {
                patch_embed,
                pos_embed,
                blocks,
                norm,
            },
            config: config.clone(),
        }
    }

    /// Run encoder and extract hidden states from selected layers.
    pub fn extract_hidden_states(
        &self,
        video_clip: Tensor<B, 5>,
    ) -> Vec<(usize, Tensor<B, 3>)> {
        let layer_indices = self.config.layer_indices();

        let x = self.model.patch_embed.forward(video_clip);
        let mut x = self.model.pos_embed.forward(x);

        let mut hidden_states = Vec::new();
        for (i, block) in self.model.blocks.iter().enumerate() {
            x = block.forward(x);
            if layer_indices.contains(&i) {
                hidden_states.push((i, x.clone()));
            }
        }

        if let Some(last) = hidden_states.last_mut() {
            if last.0 == self.config.depth - 1 {
                last.1 = self.model.norm.forward(last.1.clone());
            }
        }

        hidden_states
    }

    /// Extract features from preprocessed video clips.
    ///
    /// `clips`: vec of video clips, each [T_clip, H, W, C] f32 normalized
    /// `duration_secs`: total video duration
    ///
    /// Returns `ExtractedVideoFeatures` with shape [n_layers, embed_dim, n_timesteps].
    pub fn extract_features(
        &self,
        clips: &[Vec<f32>],
        clip_shapes: &[(usize, usize, usize, usize)], // (T, H, W, C) per clip
        duration_secs: f64,
        device: &B::Device,
    ) -> ExtractedVideoFeatures {
        let n_timesteps = (duration_secs * self.config.frequency).ceil() as usize;
        let layer_indices = self.config.layer_indices();
        let n_layers = layer_indices.len();
        let embed_dim = self.config.embed_dim;
        let n_clips = clips.len();

        if n_clips == 0 {
            return ExtractedVideoFeatures {
                data: vec![0.0; n_layers * embed_dim * n_timesteps],
                shape: vec![n_layers, embed_dim, n_timesteps],
                n_layers,
                feature_dim: embed_dim,
                n_timesteps,
            };
        }

        // Process each clip individually (batch=1 to keep memory manageable)
        // Each clip produces a mean-pooled feature vector [D] per layer
        let mut clip_features: Vec<Vec<Vec<f32>>> = Vec::with_capacity(n_clips);
        // clip_features[clip_idx][layer_idx] = [D]

        for (ci, clip_data) in clips.iter().enumerate() {
            let (t, h, w, c) = clip_shapes[ci];
            let input = Tensor::<B, 5>::from_data(
                TensorData::new(clip_data.clone(), [1, t, h, w, c]),
                device,
            );

            let hidden_states = self.extract_hidden_states(input);

            let mut per_layer = Vec::with_capacity(n_layers);
            for (_layer_idx, hs) in &hidden_states {
                // hs: [1, N, D] → mean over N (spatial pooling) → [D]
                let pooled = hs.clone().mean_dim(1); // [1, 1, D]
                let feat: Vec<f32> = pooled.reshape([embed_dim]).to_data().to_vec().unwrap();
                per_layer.push(feat);
            }
            clip_features.push(per_layer);
        }

        // Resample from n_clips to n_timesteps
        // Each clip covers clip_duration seconds; assign to output timesteps
        let clip_duration = self.config.clip_duration;
        let mut data = vec![0.0f32; n_layers * embed_dim * n_timesteps];

        for ti in 0..n_timesteps {
            let t_sec = ti as f64 / self.config.frequency;
            // Find the clip that covers this time
            let clip_idx = ((t_sec / clip_duration).floor() as usize).min(n_clips - 1);

            for li in 0..n_layers {
                let feat = &clip_features[clip_idx][li];
                for di in 0..embed_dim {
                    data[li * embed_dim * n_timesteps + di * n_timesteps + ti] = feat[di];
                }
            }
        }

        ExtractedVideoFeatures {
            data,
            shape: vec![n_layers, embed_dim, n_timesteps],
            n_layers,
            feature_dim: embed_dim,
            n_timesteps,
        }
    }
}
