//! Full Wav2Vec-BERT 2.0 model for audio feature extraction.
//!
//! Pipeline:
//! 1. Load raw audio (16 kHz mono)
//! 2. Feature encoder CNN: waveform → frame features [B, 512, T_frames]
//! 3. Feature projection: [B, T_frames, 512] → [B, T_frames, 1024]
//! 4. Adapter: subsample sequence [B, 1024, T_frames] → [B, 1024, T_adapter]
//! 5. Conformer encoder: 24 layers, extract hidden states at selected layers
//! 6. Temporally resample to output frequency (2 Hz)
//! 7. Return [n_layer_groups, 1024, n_timesteps]

use burn::prelude::*;
use crate::config::Wav2VecBertConfig;
use crate::feature_encoder::{FeatureEncoder, FeatureProjection, Adapter};
use crate::conformer::ConformerLayer;

/// Extracted audio features ready for TRIBE v2.
#[derive(Debug, Clone)]
pub struct ExtractedAudioFeatures {
    /// Feature data: [n_layers, feature_dim, n_timesteps]
    pub data: Vec<f32>,
    /// Shape: [n_layers, feature_dim, n_timesteps]
    pub shape: Vec<usize>,
    pub n_layers: usize,
    pub feature_dim: usize,
    pub n_timesteps: usize,
}

/// Full Wav2Vec-BERT 2.0 model.
#[derive(Module, Debug)]
pub struct Wav2VecBert<B: Backend> {
    pub feature_encoder: FeatureEncoder<B>,
    pub feature_projection: FeatureProjection<B>,
    pub adapter: Adapter<B>,
    pub encoder_layers: Vec<ConformerLayer<B>>,
    // config is stored outside the Module derive
}

/// Wrapper that pairs the burn Module with its config.
pub struct Wav2VecBertWithConfig<B: Backend> {
    pub model: Wav2VecBert<B>,
    pub config: Wav2VecBertConfig,
}

impl<B: Backend> Wav2VecBertWithConfig<B> {
    /// Build a new model from config (weights initialized to zero).
    pub fn new(config: &Wav2VecBertConfig, device: &B::Device) -> Self {
        let max_len = 4096;

        let feature_encoder = FeatureEncoder::new(
            &config.conv_dim,
            &config.conv_kernel,
            &config.conv_stride,
            device,
        );

        let feature_projection = FeatureProjection::new(
            config.output_hidden_size,
            config.hidden_size,
            device,
        );

        let adapter = Adapter::new(
            config.hidden_size,
            &config.adapter_kernel_size,
            &config.adapter_stride,
            device,
        );

        let mut encoder_layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            encoder_layers.push(ConformerLayer::new(
                config.hidden_size,
                config.num_attention_heads,
                config.intermediate_size,
                config.conformer_conv_kernel_size,
                max_len,
                device,
            ));
        }

        Self {
            model: Wav2VecBert {
                feature_encoder,
                feature_projection,
                adapter,
                encoder_layers,
            },
            config: config.clone(),
        }
    }

    /// Run the full forward pass and extract hidden states from selected layers.
    ///
    /// `waveform`: raw audio samples at 16 kHz, shape [B, samples]
    ///
    /// Returns a vec of (layer_index, hidden_states [B, T, D]).
    pub fn extract_hidden_states(
        &self,
        waveform: Tensor<B, 2>,
    ) -> Vec<(usize, Tensor<B, 3>)> {
        let layer_indices = self.config.layer_indices();

        // 1. Feature encoder: [B, 1, samples] → [B, 512, T_frames]
        let x = waveform.unsqueeze_dim::<3>(1);
        let x = self.model.feature_encoder.forward(x);

        // 2. Feature projection: [B, 512, T_frames] → [B, T_frames, 1024]
        let x = x.swap_dims(1, 2);
        let x = self.model.feature_projection.forward(x);

        // 3. Adapter: [B, 1024, T_frames] → [B, 1024, T_adapter]
        let x = x.swap_dims(1, 2);
        let x = self.model.adapter.forward(x);
        let x = x.swap_dims(1, 2); // [B, T_adapter, 1024]

        // 4. Conformer layers
        let mut hidden_states = Vec::new();
        let mut x = x;
        for (i, layer) in self.model.encoder_layers.iter().enumerate() {
            x = layer.forward(x);
            if layer_indices.contains(&i) {
                hidden_states.push((i, x.clone()));
            }
        }

        hidden_states
    }

    /// Extract features from raw audio waveform and resample to target frequency.
    ///
    /// `waveform`: [samples] mono f32 at 16 kHz
    /// `duration_secs`: total duration in seconds
    ///
    /// Returns `ExtractedAudioFeatures` with shape [n_layers, hidden_dim, n_timesteps].
    pub fn extract_features(
        &self,
        waveform: &[f32],
        duration_secs: f64,
        device: &B::Device,
    ) -> ExtractedAudioFeatures {
        let n_timesteps = (duration_secs * self.config.frequency).ceil() as usize;
        let layer_indices = self.config.layer_indices();
        let n_layers = layer_indices.len();
        let hidden_dim = self.config.hidden_size;

        let n_samples = waveform.len();
        let input = Tensor::<B, 2>::from_data(
            TensorData::new(waveform.to_vec(), [1, n_samples]),
            device,
        );

        let hidden_states = self.extract_hidden_states(input);

        let mut data = vec![0.0f32; n_layers * hidden_dim * n_timesteps];

        for (li, (_layer_idx, hs)) in hidden_states.iter().enumerate() {
            let [_b, t_model, d] = hs.dims();
            let hs_data: Vec<f32> = hs.to_data().to_vec().unwrap();

            for ti in 0..n_timesteps {
                let src_idx = ((ti as f64 / n_timesteps as f64) * t_model as f64).floor() as usize;
                let src_idx = src_idx.min(t_model - 1);
                for di in 0..d.min(hidden_dim) {
                    data[li * hidden_dim * n_timesteps + di * n_timesteps + ti] =
                        hs_data[src_idx * d + di];
                }
            }
        }

        ExtractedAudioFeatures {
            data,
            shape: vec![n_layers, hidden_dim, n_timesteps],
            n_layers,
            feature_dim: hidden_dim,
            n_timesteps,
        }
    }
}
