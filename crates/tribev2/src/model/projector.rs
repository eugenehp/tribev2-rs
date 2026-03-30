//! Projector — single Linear, multi-layer MLP, or SubjectLayers.
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
//!
//! When `projector = SubjectLayers(...)` in the config, each modality gets
//! a per-subject projection matrix instead of a shared linear.

use crate::tensor::Tensor;
use crate::config::SubjectLayersConfig;
use super::subject_layers::SubjectLayers;

/// A projector: either a standard MLP or a per-subject SubjectLayers.
#[derive(Debug, Clone)]
pub enum Projector {
    /// Standard Linear / MLP projector.
    Mlp(MlpProjector),
    /// Per-subject linear projector (when config uses `SubjectLayers` as projector).
    SubjectLayers(SubjectLayers),
}

/// Standard MLP projector (one or more linear layers).
#[derive(Debug, Clone)]
pub struct MlpProjector {
    /// Sequence of layers.
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
        Self::Mlp(MlpProjector::new_linear(in_dim, out_dim))
    }

    /// Create a multi-layer MLP projector.
    ///
    /// `hidden_sizes`: intermediate layer dimensions.
    /// `has_norm`: whether intermediate layers have LayerNorm.
    ///
    /// Architecture: for each hidden size `h`:
    ///   Linear(prev_dim, h) + LayerNorm(h) + GELU
    /// Final layer: Linear(last_hidden, out_dim)  (no norm, no activation)
    pub fn new_mlp(
        in_dim: usize,
        out_dim: usize,
        hidden_sizes: &[usize],
        has_norm: bool,
    ) -> Self {
        Self::Mlp(MlpProjector::new_mlp(in_dim, out_dim, hidden_sizes, has_norm))
    }

    /// Create a SubjectLayers projector.
    pub fn new_subject_layers(
        in_channels: usize,
        out_channels: usize,
        config: &SubjectLayersConfig,
    ) -> Self {
        Self::SubjectLayers(SubjectLayers::new(in_channels, out_channels, config))
    }

    /// Forward: x [..., in_dim] → [..., out_dim]
    ///
    /// For SubjectLayers variant, uses average-subjects mode.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        match self {
            Self::Mlp(mlp) => mlp.forward(x),
            Self::SubjectLayers(sl) => {
                // x: [B, T, D] → transpose to [B, D, T] for SubjectLayers
                let nd = x.ndim();
                assert!(nd >= 2);
                if nd == 3 {
                    let perm = x.permute(&[0, 2, 1]); // [B, D, T]
                    let out = sl.forward(&perm, None);  // [B, out, T]
                    out.permute(&[0, 2, 1]) // [B, T, out]
                } else {
                    sl.forward(x, None)
                }
            }
        }
    }

    /// Forward with explicit subject IDs (needed for SubjectLayers variant).
    pub fn forward_with_subjects(&self, x: &Tensor, subject_ids: Option<&[usize]>) -> Tensor {
        match self {
            Self::Mlp(mlp) => mlp.forward(x),
            Self::SubjectLayers(sl) => {
                let nd = x.ndim();
                if nd == 3 {
                    let perm = x.permute(&[0, 2, 1]);
                    let out = sl.forward(&perm, subject_ids);
                    out.permute(&[0, 2, 1])
                } else {
                    sl.forward(x, subject_ids)
                }
            }
        }
    }

    /// Get a mutable reference to the inner MlpProjector (for weight loading).
    pub fn as_mlp_mut(&mut self) -> Option<&mut MlpProjector> {
        match self {
            Self::Mlp(mlp) => Some(mlp),
            _ => None,
        }
    }

    /// Get a mutable reference to the inner SubjectLayers (for weight loading).
    pub fn as_subject_layers_mut(&mut self) -> Option<&mut SubjectLayers> {
        match self {
            Self::SubjectLayers(sl) => Some(sl),
            _ => None,
        }
    }
}

impl MlpProjector {
    /// Create a single linear layer.
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

    /// Create a multi-layer MLP.
    pub fn new_mlp(
        in_dim: usize,
        out_dim: usize,
        hidden_sizes: &[usize],
        has_norm: bool,
    ) -> Self {
        let mut layers = Vec::new();
        let mut prev_dim = in_dim;

        // Intermediate layers: Linear + LayerNorm + GELU
        for &h in hidden_sizes {
            layers.push(ProjectorLayer {
                weight: Tensor::zeros(&[prev_dim, h]),
                bias: Tensor::zeros(&[h]),
                ln_weight: if has_norm { Some(Tensor::ones(&[h])) } else { None },
                ln_bias: if has_norm { Some(Tensor::zeros(&[h])) } else { None },
                has_activation: true,
            });
            prev_dim = h;
        }

        // Final layer: Linear only (no norm, no activation)
        layers.push(ProjectorLayer {
            weight: Tensor::zeros(&[prev_dim, out_dim]),
            bias: Tensor::zeros(&[out_dim]),
            ln_weight: None,
            ln_bias: None,
            has_activation: false,
        });

        Self { layers }
    }

    /// Forward: x [..., in_dim] → [..., out_dim]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let nd = x.ndim();
        let d = *x.shape.last().unwrap();
        let batch: usize = x.shape[..nd - 1].iter().product();
        let batch_shape = x.shape[..nd - 1].to_vec();

        let mut current = x.reshape(&[batch, d]);

        for layer in &self.layers {
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
        }

        let out_dim = *current.shape.last().unwrap();
        let mut out_shape = batch_shape;
        out_shape.push(out_dim);
        current.reshape(&out_shape)
    }
}
