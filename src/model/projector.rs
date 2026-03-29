//! Projector �� single Linear or MLP (from torchvision.ops.MLP).
//!
//! For the pretrained TRIBE v2 model, the projector config has no
//! `hidden_sizes`, so it builds to a single `nn.Linear(input_dim, output_dim)`.
//!
//! The MLP config specifies `norm_layer="layer"`, `activation_layer="gelu"`,
//! but since `hidden_sizes` is empty/None and `output_size` is given,
//! `build()` returns `nn.Linear(input_size, output_size)`.
//!
//! For the general case (with hidden_sizes), MLP uses torchvision's MLP:
//!   for each hidden size: Linear + LayerNorm + GELU + Dropout
//!   final layer: Linear (no norm, no activation)

use crate::tensor::Tensor;

/// A projector: either a single linear layer or a multi-layer MLP.
#[derive(Debug, Clone)]
pub struct Projector {
    /// Sequence of (weight, bias, has_norm, has_activation) layers.
    /// Weight: [in, out], bias: [out].
    /// For a single linear: one entry with has_norm=false, has_activation=false.
    /// For MLP: intermediate layers have norm+activation, last layer doesn't.
    pub layers: Vec<ProjectorLayer>,
}

#[derive(Debug, Clone)]
pub struct ProjectorLayer {
    /// Weight [in_dim, out_dim] — stored transposed from PyTorch [out, in].
    pub weight: Tensor,
    /// Bias [out_dim].
    pub bias: Tensor,
    /// LayerNorm weight [out_dim] (if has_norm).
    pub ln_weight: Option<Tensor>,
    /// LayerNorm bias [out_dim] (if has_norm).
    pub ln_bias: Option<Tensor>,
    /// Whether to apply GELU after this layer.
    pub has_activation: bool,
}

impl Projector {
    /// Create a single linear projector.
    pub fn new_linear(in_dim: usize, out_dim: usize) -> Self {
        Self {
            layers: vec![ProjectorLayer {
                weight: Tensor::zeros(&[in_dim, out_dim]),
                bias: Tensor::zeros(&[out_dim]),
                ln_weight: None,
                ln_bias: None,
                has_activation: false,
            }],
        }
    }

    /// Forward: x [..., in_dim] → [..., out_dim]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let nd = x.ndim();
        let d = *x.shape.last().unwrap();
        let batch: usize = x.shape[..nd - 1].iter().product();
        let batch_shape = x.shape[..nd - 1].to_vec();

        let mut current = x.reshape(&[batch, d]);

        for layer in &self.layers {
            let out_dim = layer.bias.shape[0];
            // Linear: x @ W + b
            current = current.matmul(&layer.weight).add_bias(&layer.bias);

            // LayerNorm (if present)
            if let (Some(w), Some(b)) = (&layer.ln_weight, &layer.ln_bias) {
                current = current.layer_norm(w, b, 1e-5);
            }

            // GELU activation
            if layer.has_activation {
                current = current.gelu();
            }
            let _ = out_dim;
        }

        let out_dim = *current.shape.last().unwrap();
        let mut out_shape = batch_shape;
        out_shape.push(out_dim);
        current.reshape(&out_shape)
    }
}
