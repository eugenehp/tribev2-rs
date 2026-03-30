//! 3D patch embedding for V-JEPA2.
//!
//! Converts a video clip [B, C, T, H, W] into patch tokens [B, N, D]
//! via a 3D convolution (kernel = temporal_patch × patch × patch).
//!
//! For V-JEPA2 ViT-G with 256×256 input and 64 frames:
//! - Patch: 2×16×16 → spatial patches = 16×16 = 256, temporal patches = 32
//! - Total patches N = 256 × 32 = 8192
//! - Embed dim D = 1408

use burn::prelude::*;
use burn::module::{Param, ParamId};

/// 3D Patch Embedding: Conv3d(in_chans, embed_dim, kernel=(t_p, p, p), stride=(t_p, p, p))
#[derive(Module, Debug)]
pub struct PatchEmbed3d<B: Backend> {
    /// Projection weight: [embed_dim, in_chans, t_patch, patch, patch]
    pub proj_weight: Param<Tensor<B, 2>>,  // Stored flattened: [embed_dim, in_chans * t_p * p * p]
    pub proj_bias: Param<Tensor<B, 1>>,    // [embed_dim]
    pub embed_dim: usize,
    pub in_chans: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub num_spatial_patches: usize,
    pub num_temporal_patches: usize,
}

impl<B: Backend> PatchEmbed3d<B> {
    pub fn new(
        in_chans: usize,
        embed_dim: usize,
        img_size: usize,
        patch_size: usize,
        temporal_patch_size: usize,
        frames_per_clip: usize,
        device: &B::Device,
    ) -> Self {
        let kernel_vol = in_chans * temporal_patch_size * patch_size * patch_size;
        Self {
            proj_weight: Param::initialized(
                ParamId::new(),
                Tensor::zeros([embed_dim, kernel_vol], device),
            ),
            proj_bias: Param::initialized(
                ParamId::new(),
                Tensor::zeros([embed_dim], device),
            ),
            embed_dim,
            in_chans,
            patch_size,
            temporal_patch_size,
            num_spatial_patches: (img_size / patch_size) * (img_size / patch_size),
            num_temporal_patches: frames_per_clip / temporal_patch_size,
        }
    }

    /// Embed video patches.
    ///
    /// `video`: [B, T, H, W, C] — batch of video clips (f32 normalized).
    ///
    /// Returns [B, N, D] where N = num_temporal_patches × num_spatial_patches.
    pub fn forward(&self, video: Tensor<B, 5>) -> Tensor<B, 3> {
        let [batch, t, h, w, c] = video.dims();
        let (tp, sp) = (self.temporal_patch_size, self.patch_size);
        let nt = t / tp;
        let nh = h / sp;
        let nw = w / sp;
        let n_patches = nt * nh * nw;
        let kernel_vol = c * tp * sp * sp;

        // Extract patches by reshaping:
        // [B, T, H, W, C] → unfold into [B, nt, tp, nh, sp, nw, sp, C]
        // → flatten patch dims → [B, N, kernel_vol]
        // This is done via manual indexing for clarity.
        let video_data: Vec<f32> = video.to_data().to_vec().unwrap();
        let mut patches = vec![0.0f32; batch * n_patches * kernel_vol];

        for bi in 0..batch {
            for ti in 0..nt {
                for hi in 0..nh {
                    for wi in 0..nw {
                        let patch_idx = ti * nh * nw + hi * nw + wi;
                        for tt in 0..tp {
                            for hh in 0..sp {
                                for ww in 0..sp {
                                    for cc in 0..c {
                                        let src_t = ti * tp + tt;
                                        let src_h = hi * sp + hh;
                                        let src_w = wi * sp + ww;
                                        let src_idx = bi * t * h * w * c
                                            + src_t * h * w * c
                                            + src_h * w * c
                                            + src_w * c
                                            + cc;
                                        let dst_idx = bi * n_patches * kernel_vol
                                            + patch_idx * kernel_vol
                                            + cc * tp * sp * sp
                                            + tt * sp * sp
                                            + hh * sp
                                            + ww;
                                        patches[dst_idx] = video_data[src_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let device = self.proj_weight.val().device();
        let patches_tensor = Tensor::<B, 3>::from_data(
            TensorData::new(patches, [batch, n_patches, kernel_vol]),
            &device,
        );

        // Linear projection: [B, N, kernel_vol] @ [kernel_vol, embed_dim] = [B, N, embed_dim]
        let weight_t = self.proj_weight.val().transpose(); // [kernel_vol, embed_dim]
        let bias = self.proj_bias.val().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0);
        patches_tensor.matmul(weight_t.unsqueeze::<3>()) + bias
    }
}

/// Learned positional embedding: [1, N + 1, D] (with optional CLS token).
/// V-JEPA2 uses sinusoidal or learned pos embeddings.
#[derive(Module, Debug)]
pub struct PositionalEmbedding<B: Backend> {
    pub pos_embed: Param<Tensor<B, 3>>,  // [1, N, D]
    pub num_tokens: usize,
}

impl<B: Backend> PositionalEmbedding<B> {
    pub fn new(num_tokens: usize, embed_dim: usize, device: &B::Device) -> Self {
        Self {
            pos_embed: Param::initialized(
                ParamId::new(),
                Tensor::zeros([1, num_tokens, embed_dim], device),
            ),
            num_tokens,
        }
    }

    /// Add positional embedding to patch tokens.
    /// x: [B, N, D] → [B, N, D]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_b, n, _d] = x.dims();
        let n = n.min(self.num_tokens);
        let pe = self.pos_embed.val().slice([0..1, 0..n]);
        x + pe
    }
}
