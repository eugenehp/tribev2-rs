use burn::prelude::*;
use burn::module::{Param, ParamId};
use crate::config::EncoderConfig;
use super::scalenorm::ScaleNorm;
use super::attention::Attention;
use super::feedforward::FeedForward;
use super::residual::Residual;

/// x_transformers Encoder — optimised for GPU throughput.
///
/// Optimisations:
/// 1. RoPE cos/sin pre-computed once at construction and cached on-device.
/// 2. Fused QKV projection in attention (1 matmul vs 3).
/// 3. Fused RoPE kernel (wgpu-kernels-metal): 1 dispatch vs ~10.
#[derive(Module, Debug)]
pub struct XTransformerEncoder<B: Backend> {
    pub attn_norms:     Vec<ScaleNorm<B>>,
    pub attns:          Vec<Attention<B>>,
    pub attn_residuals: Vec<Residual<B>>,
    pub ff_norms:       Vec<ScaleNorm<B>>,
    pub ffs:            Vec<FeedForward<B>>,
    pub ff_residuals:   Vec<Residual<B>>,
    pub final_norm:     ScaleNorm<B>,
    /// Cached RoPE cos [max_seq_len, rot_half].
    pub rot_cos: Option<Param<Tensor<B, 2>>>,
    /// Cached RoPE sin [max_seq_len, rot_half].
    pub rot_sin: Option<Param<Tensor<B, 2>>>,
    pub rotary_dim: usize,
    pub dim:        usize,
}

impl<B: Backend> XTransformerEncoder<B> {
    pub fn new(
        dim: usize,
        max_seq_len: usize,
        config: &EncoderConfig,
        device: &B::Device,
    ) -> Self {
        let depth = config.depth;
        let mut attn_norms     = Vec::with_capacity(depth);
        let mut attns          = Vec::with_capacity(depth);
        let mut attn_residuals = Vec::with_capacity(depth);
        let mut ff_norms       = Vec::with_capacity(depth);
        let mut ffs            = Vec::with_capacity(depth);
        let mut ff_residuals   = Vec::with_capacity(depth);

        for _ in 0..depth {
            attn_norms    .push(ScaleNorm::new(dim, device));
            attns         .push(Attention::new(dim, config.heads, device));
            attn_residuals.push(Residual::new(dim, config.scale_residual, device));
            ff_norms      .push(ScaleNorm::new(dim, device));
            ffs           .push(FeedForward::new(dim, config.ff_mult, device));
            ff_residuals  .push(Residual::new(dim, config.scale_residual, device));
        }

        let rotary_dim = if config.rotary_pos_emb {
            config.rotary_emb_dim(dim)
        } else {
            0
        };

        // Pre-compute RoPE cos/sin for [0..max_seq_len] on-device once.
        // forward() slices to actual seq_len — zero CPU work per call.
        let (rot_cos, rot_sin) = if rotary_dim > 0 {
            let half = rotary_dim / 2;
            let mut freq_data = vec![0.0f32; max_seq_len * half];
            for pos in 0..max_seq_len {
                for j in 0..half {
                    let inv_freq =
                        1.0 / 10000.0f64.powf(2.0 * j as f64 / rotary_dim as f64) as f32;
                    freq_data[pos * half + j] = pos as f32 * inv_freq;
                }
            }
            let freqs = Tensor::<B, 2>::from_data(
                TensorData::new(freq_data, [max_seq_len, half]),
                device,
            );
            (
                Some(Param::initialized(ParamId::new(), freqs.clone().cos())),
                Some(Param::initialized(ParamId::new(), freqs.sin())),
            )
        } else {
            (None, None)
        };

        Self {
            attn_norms, attns, attn_residuals,
            ff_norms, ffs, ff_residuals,
            final_norm: ScaleNorm::new(dim, device),
            rot_cos, rot_sin, rotary_dim, dim,
        }
    }
}

// ── forward — standard burn-ops path ──────────────────────────────────────
#[cfg(not(feature = "wgpu-kernels-metal"))]
impl<B: Backend> XTransformerEncoder<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        encoder_forward(self, x)
    }
}

// ── forward — fused-kernel path ────────────────────────────────────────────
#[cfg(feature = "wgpu-kernels-metal")]
impl<B: Backend + crate::model_burn::FusedOps> XTransformerEncoder<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        encoder_forward(self, x)
    }
}

// ── Shared encoder body ────────────────────────────────────────────────────

#[cfg(not(feature = "wgpu-kernels-metal"))]
fn encoder_forward<B: Backend>(enc: &XTransformerEncoder<B>, x: Tensor<B, 3>) -> Tensor<B, 3> {
    run_encoder(enc, x)
}

#[cfg(feature = "wgpu-kernels-metal")]
fn encoder_forward<B: Backend + crate::model_burn::FusedOps>(
    enc: &XTransformerEncoder<B>,
    x:   Tensor<B, 3>,
) -> Tensor<B, 3> {
    run_encoder(enc, x)
}

/// The actual loop — identical for both cfg branches; the where-bound
/// just propagates through the call chain.
#[cfg(not(feature = "wgpu-kernels-metal"))]
fn run_encoder<B: Backend>(enc: &XTransformerEncoder<B>, x: Tensor<B, 3>) -> Tensor<B, 3> {
    encoder_body(enc, x)
}

#[cfg(feature = "wgpu-kernels-metal")]
fn run_encoder<B: Backend + crate::model_burn::FusedOps>(
    enc: &XTransformerEncoder<B>,
    x:   Tensor<B, 3>,
) -> Tensor<B, 3> {
    encoder_body(enc, x)
}

/// Body shared by both cfg variants — written once.
/// The macro duplicates the text so each variant sees its own where-bound.
macro_rules! define_encoder_body {
    ($($bound:tt)*) => {
        fn encoder_body<B: Backend $($bound)*>(
            enc: &XTransformerEncoder<B>,
            x:   Tensor<B, 3>,
        ) -> Tensor<B, 3> {
            let n     = x.dims()[1];
            let depth = enc.attns.len();

            // Slice cached RoPE to current seq_len: [max, half] → [n, half]
            let (rot_cos, rot_sin) = match (&enc.rot_cos, &enc.rot_sin) {
                (Some(cos), Some(sin)) => {
                    let half = cos.val().dims()[1];
                    (
                        Some(cos.val().slice([0..n, 0..half])),
                        Some(sin.val().slice([0..n, 0..half])),
                    )
                }
                _ => (None, None),
            };

            let mut x = x;
            for i in 0..depth {
                let inner = x.clone();
                x = enc.attn_norms[i].forward(x);
                x = enc.attns[i].forward(x, rot_cos.as_ref(), rot_sin.as_ref());
                x = enc.attn_residuals[i].forward(x, inner);

                let inner = x.clone();
                x = enc.ff_norms[i].forward(x);
                x = enc.ffs[i].forward(x);
                x = enc.ff_residuals[i].forward(x, inner);
            }
            enc.final_norm.forward(x)
        }
    };
}

#[cfg(not(feature = "wgpu-kernels-metal"))]
define_encoder_body!();

#[cfg(feature = "wgpu-kernels-metal")]
define_encoder_body!(+ crate::model_burn::FusedOps);
