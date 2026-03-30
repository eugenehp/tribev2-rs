//! # tribev2-audio — Wav2Vec-BERT 2.0 audio feature extraction
//!
//! Extracts intermediate-layer hidden states from Wav2Vec-BERT 2.0
//! (`facebook/w2v-bert-2.0`) for use as audio features in TRIBE v2.
//!
//! Architecture (24 conformer layers, hidden_dim=1024):
//! - Feature encoder: 7-layer CNN (raw 16 kHz waveform → frame-level features)
//! - Feature projection: Linear(512, 1024)
//! - Adapter: Conv1d strided subsampler
//! - 24 Conformer encoder layers:
//!   - Self-attention + convolution module + feed-forward
//! - Output: per-layer hidden states [T', 1024]
//!
//! Layer selection: fractional positions (e.g. `[0.5, 0.75, 1.0]`) mapped
//! to layer indices, then temporally resampled to the target frequency (2 Hz).

pub mod audio_io;
pub mod config;
pub mod feature_encoder;
pub mod conformer;
pub mod model;
pub mod weights;

pub use config::Wav2VecBertConfig;
pub use model::{Wav2VecBert, Wav2VecBertWithConfig, ExtractedAudioFeatures};
