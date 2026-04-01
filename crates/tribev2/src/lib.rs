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
//! let result = predict_segmented(&model, &features, &SegmentConfig::default());
//! let brain = tribev2::fsaverage::load_fsaverage("fsaverage5", "half", "sulcal", None)?;
//! let svg = plotting::render_brain_svg(&result.predictions[0], &brain, &PlotConfig::default());
//! ```

pub mod config;
pub mod weights;
#[cfg(feature = "hf-download")]
pub mod download;
pub mod tensor;
pub mod model;
pub mod model_burn;
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

// Flat re-exports
pub use config::{TribeV2Config, EncoderConfig, SubjectLayersConfig, ModalityDims, ModelBuildArgs};
pub use model::tribe::TribeV2;
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
