use burn::prelude::*;
use crate::config::EncoderConfig;
use super::scalenorm::ScaleNorm;
use super::attention::Attention;
use super::feedforward::FeedForward;
use super::residual::Residual;

/// x_transformers Encoder — optimized for GPU throughput.
///
/// Optimizations over naive implementation:
/// 1. Pre-computed RoPE cos/sin (1 compute, reused across 8 layers)
/// 2. Fused QKV projection in attention (1 matmul vs 3)
/// 3. ScaleNorm uses powf_scalar(-0.5) instead of sqrt+div
/// 4. Minimal intermediate tensor allocations
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

    /// Forward pass with all optimizations.
    ///
    /// Key: compute RoPE cos/sin once, pass to all layers by reference.
    /// Each layer: norm→attn→residual, norm→ff→residual (6 ops per layer).
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let n = x.dims()[1];
        let device = x.device();
        let depth = self.attns.len();

        // Pre-compute RoPE cos/sin ONCE
        let (rot_cos, rot_sin) = if self.rotary_dim > 0 {
            let half = self.rotary_dim / 2;
            let mut freq_data = vec![0.0f32; n * half];
            for pos in 0..n {
                for j in 0..half {
                    let inv_freq = 1.0 / 10000.0f64.powf(2.0 * j as f64 / self.rotary_dim as f64) as f32;
                    freq_data[pos * half + j] = pos as f32 * inv_freq;
                }
            }
            let freqs = Tensor::<B, 2>::from_data(
                TensorData::new(freq_data, [n, half]), &device
            );
            (Some(freqs.clone().cos()), Some(freqs.sin()))
        } else {
            (None, None)
        };

        let mut x = x;

        for i in 0..depth {
            // ── Attention block ───────────────────────────────────
            let inner = x.clone();
            x = self.attn_norms[i].forward(x);
            x = self.attns[i].forward(x, rot_cos.as_ref(), rot_sin.as_ref());
            x = self.attn_residuals[i].forward(x, inner);

            // ── FeedForward block ─────────────────────────────────
            let inner = x.clone();
            x = self.ff_norms[i].forward(x);
            x = self.ffs[i].forward(x);
            x = self.ff_residuals[i].forward(x, inner);
        }

        self.final_norm.forward(x)
    }
}
