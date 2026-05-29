//! Unified brain-encoder interface for pure-Rust, RLX, and (via CLI) Burn backends.

use std::collections::BTreeMap;

use anyhow::{Context, Result};

use crate::config::ModalityDims;
use crate::model::tribe::TribeV2;
use crate::tensor::Tensor;

#[cfg(feature = "rlx-encoder")]
use crate::model_rlx::TribeRlx;

/// Which compiled encoder implementation to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderKind {
    Rust,
    #[cfg(feature = "rlx-encoder")]
    Rlx,
}

impl EncoderKind {
    pub fn from_backend(backend: &str) -> Self {
        if backend.starts_with("rlx")
            || matches!(
                backend,
                "metal" | "mps" | "mlx" | "cuda" | "rocm" | "hip" | "wgpu" | "gpu" | "vulkan"
            )
        {
            #[cfg(feature = "rlx-encoder")]
            {
                return EncoderKind::Rlx;
            }
            #[cfg(not(feature = "rlx-encoder"))]
            {
                return EncoderKind::Rust;
            }
        }
        EncoderKind::Rust
    }

    pub fn label(self) -> &'static str {
        match self {
            EncoderKind::Rust => "rust",
            #[cfg(feature = "rlx-encoder")]
            EncoderKind::Rlx => "rlx",
        }
    }
}

/// Loaded pretrained encoder (Rust or RLX).
pub enum LoadedEncoder {
    Rust(TribeV2),
    #[cfg(feature = "rlx-encoder")]
    Rlx(TribeRlx),
}

impl LoadedEncoder {
    pub fn kind(&self) -> EncoderKind {
        match self {
            LoadedEncoder::Rust(_) => EncoderKind::Rust,
            #[cfg(feature = "rlx-encoder")]
            LoadedEncoder::Rlx(_) => EncoderKind::Rlx,
        }
    }

    pub fn feature_dims(&self) -> &[ModalityDims] {
        match self {
            LoadedEncoder::Rust(m) => &m.feature_dims,
            #[cfg(feature = "rlx-encoder")]
            LoadedEncoder::Rlx(m) => &m.feature_dims,
        }
    }

    pub fn n_outputs(&self) -> usize {
        match self {
            LoadedEncoder::Rust(m) => m.n_outputs,
            #[cfg(feature = "rlx-encoder")]
            LoadedEncoder::Rlx(m) => m.n_outputs,
        }
    }

    pub fn n_output_timesteps(&self) -> usize {
        match self {
            LoadedEncoder::Rust(m) => m.n_output_timesteps,
            #[cfg(feature = "rlx-encoder")]
            LoadedEncoder::Rlx(m) => m.n_output_timesteps,
        }
    }
}

/// Forward pass: `[B, n_outputs, T']` with optional adaptive pooling over time.
pub trait BrainEncoder {
    fn forward(
        &mut self,
        features: &BTreeMap<String, Tensor>,
        subject_ids: Option<&[usize]>,
        pool_outputs: bool,
    ) -> Tensor;
}

impl BrainEncoder for TribeV2 {
    fn forward(
        &mut self,
        features: &BTreeMap<String, Tensor>,
        subject_ids: Option<&[usize]>,
        pool_outputs: bool,
    ) -> Tensor {
        TribeV2::forward(self, features, subject_ids, pool_outputs)
    }
}

#[cfg(feature = "rlx-encoder")]
impl BrainEncoder for TribeRlx {
    fn forward(
        &mut self,
        features: &BTreeMap<String, Tensor>,
        subject_ids: Option<&[usize]>,
        pool_outputs: bool,
    ) -> Tensor {
        TribeRlx::forward(self, features, subject_ids, pool_outputs)
    }
}

impl BrainEncoder for LoadedEncoder {
    fn forward(
        &mut self,
        features: &BTreeMap<String, Tensor>,
        subject_ids: Option<&[usize]>,
        pool_outputs: bool,
    ) -> Tensor {
        match self {
            LoadedEncoder::Rust(m) => m.forward(features, subject_ids, pool_outputs),
            #[cfg(feature = "rlx-encoder")]
            LoadedEncoder::Rlx(m) => m.forward(features, subject_ids, pool_outputs),
        }
    }
}

/// Load a pretrained encoder from config + safetensors.
pub fn load_encoder(
    kind: EncoderKind,
    config_path: &str,
    weights_path: &str,
    build_args_path: Option<&str>,
    rlx_device_label: Option<&str>,
) -> Result<LoadedEncoder> {
    match kind {
        EncoderKind::Rust => Ok(LoadedEncoder::Rust(TribeV2::from_pretrained(
            config_path,
            weights_path,
            build_args_path,
        )?)),
        #[cfg(feature = "rlx-encoder")]
        EncoderKind::Rlx => {
            let device_label = rlx_device_label
                .unwrap_or_else(|| crate::rlx_device::default_rlx_device_label());
            let device = crate::rlx_device::parse_rlx_device_available(device_label)
                .with_context(|| format!("RLX device '{device_label}'"))?;
            let mut model = TribeRlx::from_pretrained(config_path, weights_path, build_args_path)?;
            model = model.with_device(device);
            Ok(LoadedEncoder::Rlx(model))
        }
    }
}

/// Parse CLI `--backend` into encoder kind + optional RLX device label.
pub fn parse_backend(backend: &str) -> (EncoderKind, Option<String>) {
    #[cfg(feature = "rlx-encoder")]
    {
        if backend.starts_with("rlx-") {
            let dev = backend.strip_prefix("rlx-").unwrap_or("cpu").to_string();
            return (EncoderKind::Rlx, Some(dev));
        }
        if backend == "rlx" {
            return (EncoderKind::Rlx, None);
        }
        if matches!(
            backend,
            "metal" | "mps" | "mlx" | "cuda" | "rocm" | "hip" | "wgpu" | "gpu" | "vulkan"
        ) {
            return (EncoderKind::Rlx, Some(backend.to_string()));
        }
    }
    let _ = backend;
    (EncoderKind::Rust, None)
}

pub fn use_burn_backend(backend: &str) -> bool {
    backend.starts_with("burn")
}
