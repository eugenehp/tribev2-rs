use burn::prelude::*;
use crate::config::EncoderConfig;
use super::scalenorm::ScaleNorm;
use super::attention::Attention;
use super::feedforward::FeedForward;
use super::residual::Residual;
use super::rotary::build_rotary_freqs;

/// x_transformers Encoder: ('a','f') * depth + final_norm + RoPE.
#[derive(Module, Debug)]
pub struct XTransformerEncoder<B: Backend> {
    pub attn_norms: Vec<ScaleNorm<B>>,
    pub attns: Vec<Attention<B>>,
    pub attn_residuals: Vec<Residual<B>>,
    pub ff_norms: Vec<ScaleNorm<B>>,
    pub ffs: Vec<FeedForward<B>>,
    pub ff_residuals: Vec<Residual<B>>,
    pub final_norm: ScaleNorm<B>,
    pub rotary_dim: usize,
    pub dim: usize,
}

impl<B: Backend> XTransformerEncoder<B> {
    pub fn new(dim: usize, config: &EncoderConfig, device: &B::Device) -> Self {
        let depth = config.depth;
        let mut attn_norms = Vec::with_capacity(depth);
        let mut attns = Vec::with_capacity(depth);
        let mut attn_residuals = Vec::with_capacity(depth);
        let mut ff_norms = Vec::with_capacity(depth);
        let mut ffs = Vec::with_capacity(depth);
        let mut ff_residuals = Vec::with_capacity(depth);

        for _ in 0..depth {
            attn_norms.push(ScaleNorm::new(dim, device));
            attns.push(Attention::new(dim, config.heads, device));
            attn_residuals.push(Residual::new(dim, config.scale_residual, device));
            ff_norms.push(ScaleNorm::new(dim, device));
            ffs.push(FeedForward::new(dim, config.ff_mult, device));
            ff_residuals.push(Residual::new(dim, config.scale_residual, device));
        }

        let rotary_dim = if config.rotary_pos_emb {
            config.rotary_emb_dim(dim)
        } else {
            0
        };

        Self {
            attn_norms, attns, attn_residuals,
            ff_norms, ffs, ff_residuals,
            final_norm: ScaleNorm::new(dim, device),
            rotary_dim,
            dim,
        }
    }

    /// x: [B, N, D] → [B, N, D]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let n = x.dims()[1];
        let device = x.device();

        let rotary_freqs = if self.rotary_dim > 0 {
            Some(build_rotary_freqs::<B>(self.rotary_dim, n, &device))
        } else {
            None
        };

        let depth = self.attns.len();
        let mut x = x;

        for i in 0..depth {
            // Attention block
            let inner = x.clone();
            let normed = self.attn_norms[i].forward(x);
            let attn_out = self.attns[i].forward(normed, rotary_freqs.as_ref());
            x = self.attn_residuals[i].forward(attn_out, inner);

            // FF block
            let inner = x.clone();
            let normed = self.ff_norms[i].forward(x);
            let ff_out = self.ffs[i].forward(normed);
            x = self.ff_residuals[i].forward(ff_out, inner);
        }

        self.final_norm.forward(x)
    }
}
