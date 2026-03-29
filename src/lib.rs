//! # tribev2-rs — TRIBE v2 multimodal fMRI brain encoding model inference in Rust
//!
//! Pure-Rust inference for TRIBE v2 (d'Ascoli et al., 2026), a deep multimodal
//! brain encoding model that predicts fMRI brain responses to naturalistic
//! stimuli (video, audio, text).
//!
//! The model combines feature extractors — **LLaMA 3.2** (text),
//! **V-JEPA2** (video), and **Wav2Vec-BERT** (audio) — into a unified
//! x-transformers Encoder that maps multimodal representations onto the
//! fsaverage5 cortical surface (~20 484 vertices).
//!
//! This crate provides:
//! - Full reimplementation of the `FmriEncoderModel` architecture
//!   (projectors, combiner, x-transformers encoder with ScaleNorm + RoPE,
//!    low-rank head, per-subject prediction layers, adaptive average pooling)
//! - Weight loading from the official PyTorch Lightning checkpoint
//! - Text feature extraction via `llama-cpp-4` (LLaMA 3.2-3B GGUF)
//! - HuggingFace Hub download support
//!
//! ## Quick start
//!
//! ```rust,ignore
//! use tribev2_rs::{TribeV2, TribeV2Config};
//!
//! let model = TribeV2::from_pretrained("facebook/tribev2", None)?;
//! let text_features: Vec<Vec<f32>> = extract_llama_features(...);
//! let preds = model.predict_from_text_features(&text_features, n_timesteps)?;
//! ```

pub mod config;
pub mod weights;
pub mod tensor;
pub mod model;
pub mod model_burn;

// Flat re-exports
pub use config::{TribeV2Config, EncoderConfig, SubjectLayersConfig, ModalityDims, ModelBuildArgs};
pub use model::tribe::TribeV2;
pub use weights::{WeightMap, load_checkpoint};
pub use tensor::Tensor;
