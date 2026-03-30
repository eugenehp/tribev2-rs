//! Conformer encoder layer for Wav2Vec-BERT 2.0.
//!
//! Each layer:
//! ```text
//! x → LayerNorm → SelfAttention → Residual
//!   → LayerNorm → ConvModule → Residual
//!   → LayerNorm → FeedForward → Residual
//!   → Final LayerNorm
//! ```
//!
//! The conformer uses pre-norm (LayerNorm before each sub-layer) with
//! residual connections.

use burn::prelude::*;
use burn::module::{Param, ParamId};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::softmax;
use crate::feature_encoder::layer_norm_last;

/// Self-attention with relative position bias.
#[derive(Module, Debug)]
pub struct ConformerAttention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub out_proj: Linear<B>,
    pub pos_bias: Param<Tensor<B, 3>>,  // [1, heads, 2*T-1] (relative position)
    pub heads: usize,
    pub dim_head: usize,
    pub scale: f32,
}

impl<B: Backend> ConformerAttention<B> {
    pub fn new(dim: usize, heads: usize, max_len: usize, device: &B::Device) -> Self {
        let dim_head = dim / heads;
        Self {
            q_proj: LinearConfig::new(dim, dim).with_bias(true).init(device),
            k_proj: LinearConfig::new(dim, dim).with_bias(true).init(device),
            v_proj: LinearConfig::new(dim, dim).with_bias(true).init(device),
            out_proj: LinearConfig::new(dim, dim).with_bias(true).init(device),
            pos_bias: Param::initialized(
                ParamId::new(),
                Tensor::zeros([1, heads, 2 * max_len - 1], device),
            ),
            heads,
            dim_head,
            scale: (dim_head as f32).powf(-0.5),
        }
    }

    /// x: [B, T, D] → [B, T, D]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, n, _d] = x.dims();
        let (h, dh) = (self.heads, self.dim_head);

        let q = self.q_proj.forward(x.clone()).reshape([b, n, h, dh]).swap_dims(1, 2);
        let k = self.k_proj.forward(x.clone()).reshape([b, n, h, dh]).swap_dims(1, 2);
        let v = self.v_proj.forward(x).reshape([b, n, h, dh]).swap_dims(1, 2);

        let q = q.mul_scalar(self.scale);
        let attn = softmax(q.matmul(k.swap_dims(2, 3)), 3);
        let out = attn.matmul(v);
        let out = out.swap_dims(1, 2).flatten(2, 3);
        self.out_proj.forward(out)
    }
}

/// Conformer convolution module:
/// ```text
/// LayerNorm → Pointwise Conv (expand) → GLU → Depthwise Conv → BatchNorm → SiLU → Pointwise Conv
/// ```
#[derive(Module, Debug)]
pub struct ConformerConvModule<B: Backend> {
    // Pointwise expand: Linear(D, 2D) (acts as 1×1 conv)
    pub pointwise_conv1_weight: Param<Tensor<B, 2>>,
    pub pointwise_conv1_bias: Param<Tensor<B, 1>>,
    // Depthwise conv: groups=D, kernel=31
    pub depthwise_weight: Param<Tensor<B, 3>>,  // [D, 1, K]
    pub depthwise_bias: Param<Tensor<B, 1>>,
    // Batch norm
    pub batch_norm_weight: Param<Tensor<B, 1>>,
    pub batch_norm_bias: Param<Tensor<B, 1>>,
    pub batch_norm_mean: Param<Tensor<B, 1>>,
    pub batch_norm_var: Param<Tensor<B, 1>>,
    // Pointwise contract: Linear(D, D)
    pub pointwise_conv2_weight: Param<Tensor<B, 2>>,
    pub pointwise_conv2_bias: Param<Tensor<B, 1>>,
    // Layer norm
    pub layer_norm_weight: Param<Tensor<B, 1>>,
    pub layer_norm_bias: Param<Tensor<B, 1>>,
    pub dim: usize,
    pub kernel_size: usize,
}

impl<B: Backend> ConformerConvModule<B> {
    pub fn new(dim: usize, kernel_size: usize, device: &B::Device) -> Self {
        Self {
            pointwise_conv1_weight: Param::initialized(ParamId::new(), Tensor::zeros([dim, 2 * dim], device)),
            pointwise_conv1_bias: Param::initialized(ParamId::new(), Tensor::zeros([2 * dim], device)),
            depthwise_weight: Param::initialized(ParamId::new(), Tensor::zeros([dim, 1, kernel_size], device)),
            depthwise_bias: Param::initialized(ParamId::new(), Tensor::zeros([dim], device)),
            batch_norm_weight: Param::initialized(ParamId::new(), Tensor::ones([dim], device)),
            batch_norm_bias: Param::initialized(ParamId::new(), Tensor::zeros([dim], device)),
            batch_norm_mean: Param::initialized(ParamId::new(), Tensor::zeros([dim], device)),
            batch_norm_var: Param::initialized(ParamId::new(), Tensor::ones([dim], device)),
            pointwise_conv2_weight: Param::initialized(ParamId::new(), Tensor::zeros([dim, dim], device)),
            pointwise_conv2_bias: Param::initialized(ParamId::new(), Tensor::zeros([dim], device)),
            layer_norm_weight: Param::initialized(ParamId::new(), Tensor::ones([dim], device)),
            layer_norm_bias: Param::initialized(ParamId::new(), Tensor::zeros([dim], device)),
            dim,
            kernel_size,
        }
    }

    /// x: [B, T, D] → [B, T, D]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, t, d] = x.dims();

        // Layer norm
        let x = layer_norm_last(x, &self.layer_norm_weight.val(), &self.layer_norm_bias.val(), 1e-5);

        // Pointwise conv 1: [B, T, D] → [B, T, 2D]
        let bias1 = self.pointwise_conv1_bias.val().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0);
        let x = x.matmul(self.pointwise_conv1_weight.val().unsqueeze::<3>()) + bias1;

        // GLU: split into two halves, sigmoid gate
        let x1 = x.clone().slice([0..b, 0..t, 0..d]);
        let x2 = x.slice([0..b, 0..t, d..2 * d]);
        let x = x1 * burn::tensor::activation::sigmoid(x2);

        // Depthwise conv: [B, T, D] → [B, D, T] → depthwise → [B, D, T] → [B, T, D]
        let x = x.swap_dims(1, 2); // [B, D, T]
        let x = depthwise_conv1d(x, &self.depthwise_weight.val(), &self.depthwise_bias.val(), self.kernel_size);

        // Batch norm (use running stats in eval mode)
        let x = batch_norm_1d(
            x,
            &self.batch_norm_weight.val(),
            &self.batch_norm_bias.val(),
            &self.batch_norm_mean.val(),
            &self.batch_norm_var.val(),
            1e-5,
        );

        // SiLU (swish)
        let x = x.clone() * burn::tensor::activation::sigmoid(x);

        // Pointwise conv 2: [B, D, T] → [B, T, D]
        let x = x.swap_dims(1, 2); // [B, T, D]
        let bias2 = self.pointwise_conv2_bias.val().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0);
        let x = x.matmul(self.pointwise_conv2_weight.val().unsqueeze::<3>()) + bias2;

        x
    }
}

/// Depthwise 1D convolution (groups = channels).
/// x: [B, D, T], weight: [D, 1, K] → [B, D, T] (same padding)
fn depthwise_conv1d<B: Backend>(
    x: Tensor<B, 3>,
    weight: &Tensor<B, 3>,
    bias: &Tensor<B, 1>,
    kernel_size: usize,
) -> Tensor<B, 3> {
    let [batch, channels, t] = x.dims();
    let pad = kernel_size / 2;

    // Pad input: [B, D, pad + T + pad]
    let device = x.device();
    let x = if pad > 0 {
        let left = Tensor::<B, 3>::zeros([batch, channels, pad], &device);
        let right = Tensor::<B, 3>::zeros([batch, channels, pad], &device);
        Tensor::cat(vec![left, x, right], 2)
    } else {
        x
    };

    // For each channel, apply its own 1D conv kernel
    // weight: [D, 1, K] → per-channel kernel
    let mut out_channels = Vec::with_capacity(channels);
    for c in 0..channels {
        let x_c = x.clone().slice([0..batch, c..c + 1, 0..t + 2 * pad]); // [B, 1, T+2*pad]
        let w_c = weight.clone().slice([c..c + 1, 0..1, 0..kernel_size])
            .reshape([1, kernel_size]); // [1, K]
        let b_c = bias.clone().slice([c..c + 1]); // [1]

        // Unfold: extract windows [B, T, K]
        let mut patches = Vec::with_capacity(t);
        for i in 0..t {
            let patch = x_c.clone().slice([0..batch, 0..1, i..i + kernel_size])
                .reshape([batch, kernel_size]);
            patches.push(patch);
        }
        let unfolded = Tensor::stack(patches, 1); // [B, T, K]

        // Dot product: [B, T, K] @ [K, 1] = [B, T, 1]
        let w_t = w_c.transpose(); // [K, 1]
        let b_c = b_c.unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0); // [1, 1, 1]
        let conv = unfolded.matmul(w_t.unsqueeze::<3>()) + b_c; // [B, T, 1]
        out_channels.push(conv.swap_dims(1, 2)); // [B, 1, T]
    }
    Tensor::cat(out_channels, 1) // [B, D, T]
}

/// Batch norm 1D (inference mode, using running stats).
/// x: [B, C, T]
fn batch_norm_1d<B: Backend>(
    x: Tensor<B, 3>,
    weight: &Tensor<B, 1>,
    bias: &Tensor<B, 1>,
    running_mean: &Tensor<B, 1>,
    running_var: &Tensor<B, 1>,
    eps: f64,
) -> Tensor<B, 3> {
    // Normalize using running stats
    let mean = running_mean.clone().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(2);
    let var = running_var.clone().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(2);
    let w = weight.clone().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(2);
    let b = bias.clone().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(2);

    (x - mean) / (var + eps).sqrt() * w + b
}

/// Feed-forward network: Linear → activation → Linear
#[derive(Module, Debug)]
pub struct ConformerFeedForward<B: Backend> {
    pub fc1: Linear<B>,
    pub fc2: Linear<B>,
    pub layer_norm_weight: Param<Tensor<B, 1>>,
    pub layer_norm_bias: Param<Tensor<B, 1>>,
}

impl<B: Backend> ConformerFeedForward<B> {
    pub fn new(dim: usize, intermediate: usize, device: &B::Device) -> Self {
        Self {
            fc1: LinearConfig::new(dim, intermediate).with_bias(true).init(device),
            fc2: LinearConfig::new(intermediate, dim).with_bias(true).init(device),
            layer_norm_weight: Param::initialized(ParamId::new(), Tensor::ones([dim], device)),
            layer_norm_bias: Param::initialized(ParamId::new(), Tensor::zeros([dim], device)),
        }
    }

    /// x: [B, T, D] → [B, T, D]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Pre-norm
        let x = layer_norm_last(x, &self.layer_norm_weight.val(), &self.layer_norm_bias.val(), 1e-5);
        // SiLU activation
        let h = self.fc1.forward(x);
        let h = h.clone() * burn::tensor::activation::sigmoid(h);
        self.fc2.forward(h)
    }
}

/// A single conformer encoder layer.
#[derive(Module, Debug)]
pub struct ConformerLayer<B: Backend> {
    pub self_attn: ConformerAttention<B>,
    pub self_attn_layer_norm_weight: Param<Tensor<B, 1>>,
    pub self_attn_layer_norm_bias: Param<Tensor<B, 1>>,
    pub conv_module: ConformerConvModule<B>,
    pub ffn: ConformerFeedForward<B>,
    pub final_layer_norm_weight: Param<Tensor<B, 1>>,
    pub final_layer_norm_bias: Param<Tensor<B, 1>>,
    pub dim: usize,
}

impl<B: Backend> ConformerLayer<B> {
    pub fn new(
        dim: usize,
        heads: usize,
        intermediate: usize,
        conv_kernel: usize,
        max_len: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            self_attn: ConformerAttention::new(dim, heads, max_len, device),
            self_attn_layer_norm_weight: Param::initialized(ParamId::new(), Tensor::ones([dim], device)),
            self_attn_layer_norm_bias: Param::initialized(ParamId::new(), Tensor::zeros([dim], device)),
            conv_module: ConformerConvModule::new(dim, conv_kernel, device),
            ffn: ConformerFeedForward::new(dim, intermediate, device),
            final_layer_norm_weight: Param::initialized(ParamId::new(), Tensor::ones([dim], device)),
            final_layer_norm_bias: Param::initialized(ParamId::new(), Tensor::zeros([dim], device)),
            dim,
        }
    }

    /// x: [B, T, D] → [B, T, D]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // 1. Feed-forward (first half-step in Macaron-style)
        let residual = x.clone();
        let x = self.ffn.forward(x);
        let x = residual + x.mul_scalar(0.5);

        // 2. Self-attention
        let residual = x.clone();
        let normed = layer_norm_last(
            x,
            &self.self_attn_layer_norm_weight.val(),
            &self.self_attn_layer_norm_bias.val(),
            1e-5,
        );
        let x = self.self_attn.forward(normed);
        let x = residual + x;

        // 3. Convolution module
        let residual = x.clone();
        let x = self.conv_module.forward(x);
        let x = residual + x;

        // 4. Final layer norm
        layer_norm_last(
            x,
            &self.final_layer_norm_weight.val(),
            &self.final_layer_norm_bias.val(),
            1e-5,
        )
    }
}
