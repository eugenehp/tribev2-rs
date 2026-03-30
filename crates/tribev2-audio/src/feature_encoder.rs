//! Wav2Vec2 feature encoder — 7-layer 1D CNN that converts raw 16 kHz
//! waveform to frame-level features.
//!
//! Architecture (from `facebook/w2v-bert-2.0`):
//! ```text
//! Conv1d(1, 512, kernel=10, stride=5) + GroupNorm(512) + GELU
//! Conv1d(512, 512, kernel=3, stride=2) + GELU     × 6 layers
//! ```
//!
//! Total stride = 5 × 2^6 = 320 → one frame per 20 ms at 16 kHz.

use burn::prelude::*;
use burn::module::{Param, ParamId};

/// A single 1D convolution layer with optional group norm.
#[derive(Module, Debug)]
pub struct ConvLayer<B: Backend> {
    pub weight: Param<Tensor<B, 3>>,  // [out_ch, in_ch, kernel]
    pub bias: Option<Param<Tensor<B, 1>>>,
    pub group_norm_weight: Option<Param<Tensor<B, 1>>>,  // [out_ch]
    pub group_norm_bias: Option<Param<Tensor<B, 1>>>,    // [out_ch]
    pub kernel_size: usize,
    pub stride: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub has_group_norm: bool,
    pub num_groups: usize,
}

impl<B: Backend> ConvLayer<B> {
    pub fn new(
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        has_group_norm: bool,
        device: &B::Device,
    ) -> Self {
        // Group norm uses out_ch groups (i.e., instance-norm-like) for first layer
        let num_groups = if has_group_norm { out_ch } else { 1 };
        Self {
            weight: Param::initialized(
                ParamId::new(),
                Tensor::zeros([out_ch, in_ch, kernel], device),
            ),
            bias: Some(Param::initialized(
                ParamId::new(),
                Tensor::zeros([out_ch], device),
            )),
            group_norm_weight: if has_group_norm {
                Some(Param::initialized(ParamId::new(), Tensor::ones([out_ch], device)))
            } else {
                None
            },
            group_norm_bias: if has_group_norm {
                Some(Param::initialized(ParamId::new(), Tensor::zeros([out_ch], device)))
            } else {
                None
            },
            kernel_size: kernel,
            stride,
            in_channels: in_ch,
            out_channels: out_ch,
            has_group_norm,
            num_groups,
        }
    }

    /// x: [B, C_in, T] → [B, C_out, T']
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // 1D convolution via unfold + matmul (burn doesn't have native Conv1d with stride)
        let x = conv1d_forward(
            x,
            &self.weight.val(),
            self.bias.as_ref().map(|b| b.val()),
            self.stride,
        );

        // Group norm
        let x = if self.has_group_norm {
            group_norm(
                x,
                self.num_groups,
                self.group_norm_weight.as_ref().map(|w| w.val()),
                self.group_norm_bias.as_ref().map(|b| b.val()),
                1e-5,
            )
        } else {
            x
        };

        // GELU activation
        burn::tensor::activation::gelu(x)
    }
}

/// 1D convolution: x [B, C_in, T], weight [C_out, C_in, K] → [B, C_out, T']
fn conv1d_forward<B: Backend>(
    x: Tensor<B, 3>,
    weight: &Tensor<B, 3>,
    bias: Option<Tensor<B, 1>>,
    stride: usize,
) -> Tensor<B, 3> {
    let [batch, _c_in, t_in] = x.dims();
    let [c_out, c_in, kernel] = weight.dims();
    let t_out = (t_in - kernel) / stride + 1;

    // Unfold: extract patches [B, C_in * K, T_out]
    // Then matmul with weight reshaped to [C_out, C_in * K]
    let mut patches = Vec::with_capacity(t_out);
    for i in 0..t_out {
        let start = i * stride;
        let end = start + kernel;
        // [B, C_in, K] → [B, C_in * K]
        let patch = x.clone().slice([0..batch, 0..c_in, start..end])
            .reshape([batch, c_in * kernel]);
        patches.push(patch);
    }
    // Stack: [B, T_out, C_in * K]
    let unfolded = Tensor::stack(patches, 1);

    // weight: [C_out, C_in * K] → transpose → [C_in * K, C_out]
    let w_flat = weight.clone().reshape([c_out, c_in * kernel]).transpose();

    // [B, T_out, C_in * K] @ [C_in * K, C_out] = [B, T_out, C_out]
    let out = unfolded.matmul(w_flat.unsqueeze::<3>());
    // → [B, C_out, T_out]
    let out = out.swap_dims(1, 2);

    if let Some(b) = bias {
        out + b.unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(2)
    } else {
        out
    }
}

/// Group normalization: [B, C, T] with G groups.
fn group_norm<B: Backend>(
    x: Tensor<B, 3>,
    num_groups: usize,
    weight: Option<Tensor<B, 1>>,
    bias: Option<Tensor<B, 1>>,
    eps: f64,
) -> Tensor<B, 3> {
    let [batch, channels, t] = x.dims();
    let group_size = channels / num_groups;

    // Reshape to [B, G, group_size * T]
    let x_grouped = x.clone().reshape([batch, num_groups, group_size * t]);

    // Mean and variance over last dim
    let mean = x_grouped.clone().mean_dim(2); // [B, G, 1]
    let var = x_grouped.clone()
        .sub(mean.clone())
        .powf_scalar(2.0)
        .mean_dim(2); // [B, G, 1]

    // Normalize
    let x_norm = (x_grouped - mean) / (var + eps).sqrt();

    // Reshape back to [B, C, T]
    let x_norm = x_norm.reshape([batch, channels, t]);

    // Scale and shift
    let x_norm = if let Some(w) = weight {
        x_norm * w.unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(2)
    } else {
        x_norm
    };

    if let Some(b) = bias {
        x_norm + b.unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(2)
    } else {
        x_norm
    }
}

/// The full 7-layer feature encoder CNN.
#[derive(Module, Debug)]
pub struct FeatureEncoder<B: Backend> {
    pub layers: Vec<ConvLayer<B>>,
}

impl<B: Backend> FeatureEncoder<B> {
    pub fn new(
        conv_dim: &[usize],
        conv_kernel: &[usize],
        conv_stride: &[usize],
        device: &B::Device,
    ) -> Self {
        let mut layers = Vec::new();
        for i in 0..conv_dim.len() {
            let in_ch = if i == 0 { 1 } else { conv_dim[i - 1] };
            let out_ch = conv_dim[i];
            let has_gn = i == 0; // Only first layer has GroupNorm
            layers.push(ConvLayer::new(in_ch, out_ch, conv_kernel[i], conv_stride[i], has_gn, device));
        }
        Self { layers }
    }

    /// waveform: [B, 1, samples] → [B, 512, T_frames]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(x);
        }
        x
    }
}

/// Feature projection: Linear(512, 1024) + LayerNorm(1024).
#[derive(Module, Debug)]
pub struct FeatureProjection<B: Backend> {
    pub projection_weight: Param<Tensor<B, 2>>,  // [in, out]
    pub projection_bias: Param<Tensor<B, 1>>,
    pub layer_norm_weight: Param<Tensor<B, 1>>,
    pub layer_norm_bias: Param<Tensor<B, 1>>,
    pub in_dim: usize,
    pub out_dim: usize,
}

impl<B: Backend> FeatureProjection<B> {
    pub fn new(in_dim: usize, out_dim: usize, device: &B::Device) -> Self {
        Self {
            projection_weight: Param::initialized(ParamId::new(), Tensor::zeros([in_dim, out_dim], device)),
            projection_bias: Param::initialized(ParamId::new(), Tensor::zeros([out_dim], device)),
            layer_norm_weight: Param::initialized(ParamId::new(), Tensor::ones([out_dim], device)),
            layer_norm_bias: Param::initialized(ParamId::new(), Tensor::zeros([out_dim], device)),
            in_dim,
            out_dim,
        }
    }

    /// x: [B, T, in_dim] → [B, T, out_dim]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Linear: [B, T, in] @ [1, in, out] + [1, 1, out] → [B, T, out]
        let bias = self.projection_bias.val().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0);
        let x = x.matmul(self.projection_weight.val().unsqueeze::<3>()) + bias;

        // Layer norm over last dim
        layer_norm_last(x, &self.layer_norm_weight.val(), &self.layer_norm_bias.val(), 1e-5)
    }
}

/// Adapter: strided Conv1d layers that downsample the sequence.
///
/// Wav2Vec-BERT 2.0 uses 2 adapter layers with kernel=3, stride=2 each,
/// reducing sequence length by 4×.
#[derive(Module, Debug)]
pub struct Adapter<B: Backend> {
    pub layers: Vec<AdapterConvLayer<B>>,
}

#[derive(Module, Debug)]
pub struct AdapterConvLayer<B: Backend> {
    pub weight: Param<Tensor<B, 3>>,  // [out_ch, in_ch, kernel]
    pub bias: Param<Tensor<B, 1>>,
    pub layer_norm_weight: Param<Tensor<B, 1>>,
    pub layer_norm_bias: Param<Tensor<B, 1>>,
    pub kernel_size: usize,
    pub stride: usize,
}

impl<B: Backend> AdapterConvLayer<B> {
    pub fn new(channels: usize, kernel: usize, stride: usize, device: &B::Device) -> Self {
        Self {
            weight: Param::initialized(ParamId::new(), Tensor::zeros([channels, channels, kernel], device)),
            bias: Param::initialized(ParamId::new(), Tensor::zeros([channels], device)),
            layer_norm_weight: Param::initialized(ParamId::new(), Tensor::ones([channels], device)),
            layer_norm_bias: Param::initialized(ParamId::new(), Tensor::zeros([channels], device)),
            kernel_size: kernel,
            stride,
        }
    }

    /// x: [B, C, T] → [B, C, T']
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_batch, _channels, _t] = x.dims();
        let x = conv1d_forward(x, &self.weight.val(), Some(self.bias.val()), self.stride);
        let x = burn::tensor::activation::gelu(x);

        // Layer norm: [B, C, T'] → [B, T', C] → norm → [B, C, T']
        let [_, _, _t_out] = x.dims();
        let x = x.swap_dims(1, 2); // [B, T', C]
        let x = layer_norm_last(x, &self.layer_norm_weight.val(), &self.layer_norm_bias.val(), 1e-5);
        x.swap_dims(1, 2) // [B, C, T']
    }
}

impl<B: Backend> Adapter<B> {
    pub fn new(
        channels: usize,
        kernel_sizes: &[usize],
        strides: &[usize],
        device: &B::Device,
    ) -> Self {
        let layers = kernel_sizes.iter().zip(strides)
            .map(|(&k, &s)| AdapterConvLayer::new(channels, k, s, device))
            .collect();
        Self { layers }
    }

    /// x: [B, C, T] → [B, C, T'] (downsampled)
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(x);
        }
        x
    }
}

/// Layer norm over the last dimension of a 3D tensor.
pub fn layer_norm_last<B: Backend>(
    x: Tensor<B, 3>,
    weight: &Tensor<B, 1>,
    bias: &Tensor<B, 1>,
    eps: f64,
) -> Tensor<B, 3> {
    let mean = x.clone().mean_dim(2);         // [B, T, 1]
    let diff = x.clone() - mean.clone();
    let var = diff.clone().powf_scalar(2.0).mean_dim(2); // [B, T, 1]
    let x_norm = diff / (var + eps).sqrt();

    x_norm * weight.clone().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(1)
        + bias.clone().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(1)
}
