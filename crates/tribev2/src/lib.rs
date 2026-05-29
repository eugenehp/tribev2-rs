//! # tribev2-rs — TRIBE v2 multimodal fMRI brain encoding model inference in Rust
//!
//! Inference for TRIBE v2 (d'Ascoli et al., 2026), a deep multimodal brain
//! encoding model that predicts fMRI brain responses to naturalistic stimuli
//! (video, audio, text).
//!
//! The **pure-Rust** encoder (`TribeV2`) is always available. Optional Cargo
//! features enable **Burn** (`--features burn,wgpu-metal`, …) and **RLX**
//! (`--features rlx-encoder,rlx-metal`, …). Default features are empty — enable
//! backends explicitly. See the workspace README and `bench/README.md`.
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
//! - Text feature extraction via RLX / `rlx-llama32` (LLaMA 3.2-3B GGUF)
//!   with intermediate layer activation extraction
//! - Segment-based batching for long-form inference
//! - Multi-modal inference (text + audio + video)
//! - Brain surface visualization (SVG rendering)
//! - HuggingFace Hub download support
//!
//! ## Quick start
//!
//! ```rust,ignore
//! use tribev2::model::tribe::TribeV2;
//! use tribev2::segments::{SegmentConfig, predict_segmented};
//! use tribev2::plotting::{self, PlotConfig, View, ColorMap};
//!
//! let model = TribeV2::from_pretrained("config.yaml", "model.safetensors", Some("build_args.json"))?;
//! let mut model = TribeV2::from_pretrained("config.yaml", "model.safetensors", None)?;
//! let result = predict_segmented(&mut model, &features, &SegmentConfig::default());
//! let brain = tribev2::fsaverage::load_fsaverage("fsaverage5", "half", "sulcal", None)?;
//! let svg = plotting::render_brain_svg(&result.predictions[0], &brain, &PlotConfig::default());
//! ```

pub mod config;
pub mod brain_encoder;
pub mod data_paths;
pub mod weights;
#[cfg(feature = "hf-download")]
pub mod download;
pub mod tensor;
pub mod model;
#[cfg(feature = "burn")]
pub mod model_burn;
#[cfg(feature = "rlx-encoder")]
pub mod model_rlx;
#[cfg(feature = "rlx-encoder")]
pub mod rlx_device;
pub mod features;
pub mod segments;
pub mod plotting;
pub mod fsaverage;
pub mod events;
pub mod nifti;
pub mod roi;
pub mod metrics;
pub mod subcortical;
pub mod video_output;
pub mod resample;
pub mod pipeline;
pub mod vol_to_surf;

// Flat re-exports
pub use config::{TribeV2Config, EncoderConfig, SubjectLayersConfig, ModalityDims, ModelBuildArgs};
pub use model::tribe::TribeV2;
#[cfg(feature = "rlx-encoder")]
pub use model_rlx::TribeRlx;
pub use brain_encoder::{
    load_encoder, parse_backend, BrainEncoder, EncoderKind, LoadedEncoder,
};
pub use data_paths::{data_dir, parity_refs_available, parity_refs_dir, weights_available};
pub use weights::{WeightMap, load_checkpoint};
pub use tensor::Tensor;
pub use features::{ExtractedFeatures, LlamaFeatureConfig, extract_llama_features, extract_llama_features_timed, zero_features, resample_features};
pub use segments::{Segment, SegmentConfig, SegmentedPrediction, predict_segmented, predict_segments_batched};
pub use plotting::{BrainMesh, PlotConfig, View, ColorMap, render_brain_svg, render_hemisphere_svg, render_multi_view, render_timesteps};
pub use fsaverage::{load_fsaverage, find_fsaverage_dir, fsaverage_size};
pub use events::{Event, EventList, build_events_from_media, text_to_events, transcribe_audio};
pub use nifti::{NiftiConfig, write_nifti, write_nifti_4d, surface_to_volume, load_pial_coords_mni};
pub use roi::{get_hcp_labels, get_hcp_vertex_labels, summarize_by_roi, get_topk_rois, get_roi_indices};
pub use metrics::{pearson_r, pearson_per_vertex, mean_pearson, median_pearson, mse, topk_accuracy, load_ground_truth};
pub use subcortical::{SubcorticalConfig, get_subcortical_labels, get_subcortical_roi_indices, summarize_subcortical};
pub use video_output::{VideoConfig, render_mp4, render_gif};
pub use resample::{resample_surface, compute_resampling_map, ResamplingMap};
pub use pipeline::{PipelineInput, PipelineConfig, PipelineOutput, predict_from_media, text_to_speech};
pub use vol_to_surf::{NiftiVolume, VolToSurfConfig, SamplingKind, Interpolation, vol_to_surf, vol_to_surf_4d};
