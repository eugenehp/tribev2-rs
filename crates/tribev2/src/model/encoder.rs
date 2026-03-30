//! x_transformers Encoder — full transformer encoder stack.
//!
//! The encoder consists of repeating (Attention, FeedForward) blocks with:
//! - Pre-norm (ScaleNorm) on each branch
//! - Residual connections with optional `scale_residual`
//! - Rotary position embeddings
//! - Final ScaleNorm
//!
//! Layer structure for each depth iteration:
//!   layer_types = ('a', 'f') * depth
//!   Each layer = [norms(ModuleList[3]), block, residual]
//!
//! For pre_norm=True (default):
//!   norms[0] = ScaleNorm(dim)  (pre-branch norm)
//!   norms[1] = None            (post-branch norm, only for sandwich_norm)
//!   norms[2] = None            (post-main norm, only for post-norm)

use crate::tensor::Tensor;
use crate::config::EncoderConfig;
use super::scalenorm::ScaleNorm;
use super::rotary::RotaryEmbedding;
use super::attention::Attention;
use super::feedforward::FeedForward;
use super::residual::Residual;

/// One transformer layer: either an attention or feedforward block,
/// wrapped with pre-norm and residual connection.
#[derive(Debug, Clone)]
pub enum LayerBlock {
    Attn(Attention),
    FF(FeedForward),
}

/// A single layer entry: [pre_norm, block, residual].
#[derive(Debug, Clone)]
pub struct TransformerLayer {
    pub pre_norm: ScaleNorm,
    pub block: LayerBlock,
    pub residual: Residual,
}

/// Full x_transformers Encoder.
#[derive(Debug, Clone)]
pub struct XTransformerEncoder {
    pub layers: Vec<TransformerLayer>,
    pub final_norm: ScaleNorm,
    pub rotary: Option<RotaryEmbedding>,
    pub dim: usize,
    pub config: EncoderConfig,
}

impl XTransformerEncoder {
    /// Create encoder with randomly initialized weights.
    pub fn new(dim: usize, config: &EncoderConfig) -> Self {
        let mut layers = Vec::new();
        let depth = config.depth;

        // layer_types = ('a', 'f') * depth
        for _ in 0..depth {
            // Attention layer
            layers.push(TransformerLayer {
                pre_norm: ScaleNorm::new(dim),
                block: LayerBlock::Attn(Attention::new(dim, config.heads)),
                residual: Residual::new(dim, config.scale_residual),
            });
            // FeedForward layer
            layers.push(TransformerLayer {
                pre_norm: ScaleNorm::new(dim),
                block: LayerBlock::FF(FeedForward::new(dim, config.ff_mult)),
                residual: Residual::new(dim, config.scale_residual),
            });
        }

        let rotary = if config.rotary_pos_emb {
            let rot_dim = config.rotary_emb_dim(dim);
            Some(RotaryEmbedding::new(rot_dim))
        } else {
            None
        };

        Self {
            layers,
            final_norm: ScaleNorm::new(dim),
            rotary,
            dim,
            config: config.clone(),
        }
    }

    /// Forward pass: x [B, N, D] → [B, N, D]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let n = x.shape[1]; // sequence length

        // Compute rotary position embeddings once
        let rotary_freqs = self.rotary.as_ref().map(|r| r.forward(n));

        let mut x = x.clone();

        for layer in &self.layers {
            let inner_residual = x.clone();

            // Pre-norm
            x = layer.pre_norm.forward(&x);

            // Block forward
            match &layer.block {
                LayerBlock::Attn(attn) => {
                    x = attn.forward(&x, rotary_freqs.as_ref());
                }
                LayerBlock::FF(ff) => {
                    x = ff.forward(&x);
                }
            }

            // Residual
            x = layer.residual.forward(&x, &inner_residual);
        }

        // Final norm
        self.final_norm.forward(&x)
    }
}
