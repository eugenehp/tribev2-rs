//! TribeV2 — full FmriEncoderModel reimplementation.
//!
//! Python (model.py `FmriEncoderModel`):
//!
//! Architecture:
//!   1. Per-modality projectors (Linear or MLP): project each modality to hidden//n_modalities
//!   2. Concatenate/sum/stack modality features → [B, T, hidden]
//!   3. Optional combiner (Linear/MLP/Identity)
//!   4. Add time positional embedding
//!   5. x-transformers Encoder
//!   6. Optional low_rank_head: Linear(hidden, low_rank_dim, bias=False)
//!   7. SubjectLayers predictor: per-subject linear → [B, n_outputs, T]
//!   8. AdaptiveAvgPool1d(n_output_timesteps) → [B, n_outputs, T']
//!
//! For the pretrained model:
//!   - 3 modalities (text, audio, video), extractor_aggregation="cat"
//!   - Each projector is a single Linear(layer_dim * n_layers, hidden//3 = 384)
//!   - combiner = None → Identity
//!   - time_pos_embed: [1, 1024, 1152]
//!   - encoder: x_transformers Encoder(dim=1152, depth=8, heads=8, ...)
//!   - low_rank_head: Linear(1152, 2048, bias=False)
//!   - predictor: SubjectLayers(2048, 20484, n_subjects=26)
//!   - pooler: AdaptiveAvgPool1d(100)

use std::collections::BTreeMap;
use crate::tensor::Tensor;
use crate::config::{BrainModelConfig, ModalityDims, ModelBuildArgs, TribeV2Config};
use crate::weights::{WeightMap, load_weights};
use super::projector::Projector;
use super::encoder::XTransformerEncoder;
use super::subject_layers::SubjectLayers;
use super::temporal_smoothing::TemporalSmoothing;

/// Named projector, preserving insertion order.
#[derive(Debug, Clone)]
pub struct NamedProjector {
    pub name: String,
    pub projector: Projector,
}

/// The full TRIBE v2 brain encoding model.
#[derive(Debug, Clone)]
pub struct TribeV2 {
    /// Per-modality projectors, in insertion order matching Python feature_dims.
    pub projectors: Vec<NamedProjector>,

    /// Optional combiner (Linear/MLP). None → identity.
    pub combiner: Option<Projector>,

    /// Optional time positional embedding: [1, max_seq_len, hidden].
    pub time_pos_embed: Option<Tensor>,

    /// x-transformers Encoder (None if linear_baseline).
    pub encoder: Option<XTransformerEncoder>,

    /// Optional subject embedding: [n_subjects, hidden].
    /// Python: `nn.Embedding(n_subjects, hidden)`.
    pub subject_embed: Option<Tensor>,

    /// Optional low-rank head: Linear(hidden, low_rank_dim, bias=False).
    /// Stored as weight [hidden, low_rank_dim].
    pub low_rank_head: Option<Tensor>,

    /// Per-subject prediction head.
    pub predictor: SubjectLayers,

    /// Optional temporal smoothing (depthwise Conv1d).
    pub temporal_smoothing: Option<TemporalSmoothing>,

    /// Feature dimensions per modality.
    pub feature_dims: Vec<ModalityDims>,

    /// Number of output vertices (fsaverage5 = 20484).
    pub n_outputs: usize,

    /// Number of output timesteps (duration_trs).
    pub n_output_timesteps: usize,

    /// Model config.
    pub config: BrainModelConfig,
}

impl TribeV2 {
    /// Build the model from config and feature dimensions.
    /// Weights are initialized to zero; call `load_weights` to populate.
    pub fn new(
        feature_dims: Vec<ModalityDims>,
        n_outputs: usize,
        n_output_timesteps: usize,
        config: &BrainModelConfig,
    ) -> Self {
        let hidden = config.hidden;
        let n_modalities = feature_dims.len();

        // Build projectors (ordered same as feature_dims — matches Python dict order)
        // Only build for modalities with non-None dims (Python: `if tup is None: continue`)
        let mut projectors = Vec::new();
        for md in &feature_dims {
            if let Some((num_layers, feature_dim)) = md.dims {
                let input_dim = if config.layer_aggregation == "cat" {
                    feature_dim * num_layers
                } else {
                    feature_dim
                };
                let output_dim = if config.extractor_aggregation == "cat" {
                    hidden / n_modalities
                } else {
                    hidden
                };
                projectors.push(NamedProjector {
                    name: md.name.clone(),
                    projector: Projector::new_linear(input_dim, output_dim),
                });
            }
            // None dims → no projector, will get zero-padded in aggregate_features
        }

        // Combiner
        let combiner_input_dim = if config.extractor_aggregation == "cat" {
            (hidden / n_modalities) * n_modalities
        } else {
            hidden
        };
        let combiner = if config.combiner.is_some() {
            Some(Projector::new_linear(combiner_input_dim, hidden))
        } else {
            None
        };

        // Time positional embedding
        let time_pos_embed = if config.time_pos_embedding && !config.linear_baseline {
            Some(Tensor::zeros(&[1, config.max_seq_len, hidden]))
        } else {
            None
        };

        // Encoder
        let encoder = if !config.linear_baseline {
            config.encoder.as_ref().map(|enc_config| {
                XTransformerEncoder::new(hidden, enc_config)
            })
        } else {
            None
        };

        // Subject embedding
        let subject_embed = if config.subject_embedding && !config.linear_baseline {
            let n_subjects = config.subject_layers.as_ref().map_or(200, |sl| sl.n_subjects);
            Some(Tensor::zeros(&[n_subjects, hidden]))
        } else {
            None
        };

        // Low-rank head
        let low_rank_head = config.low_rank_head.map(|lr| {
            Tensor::zeros(&[hidden, lr])
        });

        // Predictor
        let bottleneck = config.low_rank_head.unwrap_or(hidden);
        let sl_config = config.subject_layers.clone().unwrap_or_default();
        let predictor = SubjectLayers::new(bottleneck, n_outputs, &sl_config);

        // Temporal smoothing
        let temporal_smoothing = config.temporal_smoothing.as_ref().map(|ts| {
            if let Some(sigma) = ts.sigma {
                TemporalSmoothing::new_gaussian(hidden, ts.kernel_size, sigma)
            } else {
                TemporalSmoothing::new_learnable(hidden, ts.kernel_size)
            }
        });

        Self {
            projectors,
            combiner,
            time_pos_embed,
            subject_embed,
            encoder,
            low_rank_head,
            predictor,
            temporal_smoothing,
            feature_dims,
            n_outputs,
            n_output_timesteps,
            config: config.clone(),
        }
    }

    /// Load a pretrained model from config.yaml + model.safetensors (+ optional build_args.json).
    ///
    /// Mirrors `TribeModel.from_pretrained()` in Python:
    /// 1. Parse config.yaml
    /// 2. Set `average_subjects=True` on subject_layers
    /// 3. Build model from build_args (feature_dims, n_outputs, n_output_timesteps)
    /// 4. Load weights from safetensors
    ///
    /// If `build_args_path` is None, uses the pretrained defaults:
    ///   feature_dims = {text: (3,3072), audio: (3,1024), video: (3,1408)}
    ///   n_outputs = 20484, n_output_timesteps = 100
    pub fn from_pretrained(
        config_path: &str,
        weights_path: &str,
        build_args_path: Option<&str>,
    ) -> anyhow::Result<Self> {
        // 1. Parse config
        let yaml = std::fs::read_to_string(config_path)?;
        let mut config: TribeV2Config = serde_yaml::from_str(&yaml)?;

        // 2. Set average_subjects = true (mirrors Python from_pretrained)
        if let Some(ref mut sl) = config.brain_model_config.subject_layers {
            sl.average_subjects = true;
        }

        // 3. Determine feature_dims, n_outputs, n_output_timesteps
        let (feature_dims, n_outputs, n_output_timesteps) = if let Some(ba_path) = build_args_path {
            let ba = ModelBuildArgs::from_json(ba_path)?;
            (ba.to_modality_dims(), ba.n_outputs, ba.n_output_timesteps)
        } else {
            (ModalityDims::pretrained(), 20484, config.data.duration_trs)
        };

        // 4. Build model
        let mut model = Self::new(
            feature_dims,
            n_outputs,
            n_output_timesteps,
            &config.brain_model_config,
        );

        // 5. Load weights
        let mut wm = WeightMap::from_safetensors(weights_path)?;
        load_weights(&mut wm, &mut model)?;

        Ok(model)
    }

    /// Aggregate features from multiple modalities.
    ///
    /// `features`: map from modality name → tensor [B, L, D, T] or [B, D, T].
    /// Returns: [B, T, H]
    pub fn aggregate_features(
        &self,
        features: &BTreeMap<String, Tensor>,
    ) -> Tensor {
        let n_modalities = self.feature_dims.len();
        let hidden = self.config.hidden;

        // Get B, T from first available modality
        let first = features.values().next().expect("no features provided");
        let b = first.shape[0];
        let t = *first.shape.last().unwrap();

        let mut tensors = Vec::new();

        for md in &self.feature_dims {
            let projector = self.projectors.iter().find(|np| np.name == md.name);
            let has_projector = projector.is_some();
            if has_projector && features.contains_key(&md.name) {
                let projector = &projector.unwrap().projector;
                let data = features.get(&md.name).unwrap();
                let mut data = data.clone();

                // Handle 3D [B, D, T] → 4D [B, 1, D, T]
                if data.ndim() == 3 {
                    data = data.reshape(&[b, 1, data.shape[1], t]);
                }
                // data: [B, L, D, T]

                // Layer aggregation
                let data = if self.config.layer_aggregation == "mean" {
                    // Mean over dim 1 (layers)
                    let l = data.shape[1];
                    let d = data.shape[2];
                    let mut mean_data = vec![0.0f32; b * d * t];
                    for bi in 0..b {
                        for di in 0..d {
                            for ti in 0..t {
                                let mut sum = 0.0f32;
                                for li in 0..l {
                                    sum += data.data[bi * l * d * t + li * d * t + di * t + ti];
                                }
                                mean_data[bi * d * t + di * t + ti] = sum / l as f32;
                            }
                        }
                    }
                    Tensor::from_vec(mean_data, vec![b, d, t])
                } else {
                    // "cat": rearrange 'b l d t -> b (l d) t'
                    let l = data.shape[1];
                    let d = data.shape[2];
                    data.reshape(&[b, l * d, t])
                };

                // Transpose to [B, T, D] for projector
                let data = data.permute(&[0, 2, 1]); // [B, T, D]

                // Apply projector: [B, T, D] → [B, T, H/n_modalities]
                let data = projector.forward(&data);

                tensors.push(data);
            } else {
                // Missing modality → zeros
                let out_dim = if self.config.extractor_aggregation == "cat" {
                    hidden / n_modalities
                } else {
                    hidden
                };
                tensors.push(Tensor::zeros(&[b, t, out_dim]));
            }
        }

        // Aggregate across modalities
        if self.config.extractor_aggregation == "cat" {
            let refs: Vec<&Tensor> = tensors.iter().collect();
            Tensor::cat_last(&refs)
        } else if self.config.extractor_aggregation == "sum" {
            let refs: Vec<&Tensor> = tensors.iter().collect();
            Tensor::sum_tensors(&refs)
        } else {
            // "stack" → cat along dim 1
            let refs: Vec<&Tensor> = tensors.iter().collect();
            Tensor::cat_dim1(&refs)
        }
    }

    /// Transformer forward: combiner + pos_embed + subject_embed + encoder.
    /// Input: [B, T, H] → output: [B, T, H]
    fn transformer_forward(&self, x: &Tensor, subject_ids: Option<&[usize]>) -> Tensor {
        // Combiner
        let mut x = if let Some(ref combiner) = self.combiner {
            combiner.forward(x)
        } else {
            x.clone()
        };

        // Time positional embedding
        if let Some(ref tpe) = self.time_pos_embed {
            let t = x.shape[1];
            let tpe_slice = tpe.slice_dim1(0, t); // [1, T, H]
            // Broadcast add over batch dim
            let tpe_expanded = tpe_slice.reshape(&[1, t, self.config.hidden]);
            // x: [B, T, H], tpe_expanded: [1, T, H]
            let b = x.shape[0];
            let h = self.config.hidden;
            for bi in 0..b {
                for ti in 0..t {
                    for hi in 0..h {
                        let idx = bi * t * h + ti * h + hi;
                        let tpe_idx = ti * h + hi;
                        x.data[idx] += tpe_expanded.data[tpe_idx];
                    }
                }
            }
        }

        // Subject embedding: x = x + subject_embed(subject_id)
        if let Some(ref se) = self.subject_embed {
            if let Some(sids) = subject_ids {
                let (b, t, h) = (x.shape[0], x.shape[1], x.shape[2]);
                for bi in 0..b {
                    let sid = if bi < sids.len() { sids[bi] } else { 0 };
                    let emb_offset = sid * h;
                    for ti in 0..t {
                        for hi in 0..h {
                            x.data[bi * t * h + ti * h + hi] += se.data[emb_offset + hi];
                        }
                    }
                }
            }
        }

        // Encoder
        if let Some(ref encoder) = self.encoder {
            x = encoder.forward(&x);
        }

        x
    }

    /// Full forward pass.
    ///
    /// `features`: map from modality name → tensor [B, L, D, T] or [B, D, T].
    /// `subject_ids`: per-batch subject indices `[B]`. `None` for average_subjects mode.
    /// `pool_outputs`: whether to apply adaptive avg pool.
    ///
    /// Returns: [B, n_outputs, T'] where T' = n_output_timesteps if pooled.
    pub fn forward(
        &self,
        features: &BTreeMap<String, Tensor>,
        subject_ids: Option<&[usize]>,
        pool_outputs: bool,
    ) -> Tensor {
        // 1. Aggregate features → [B, T, H]
        let mut x = self.aggregate_features(features);

        // 2. Temporal smoothing (if present): transpose to [B, H, T], convolve, transpose back
        if let Some(ref ts) = self.temporal_smoothing {
            x = x.permute(&[0, 2, 1]); // [B, H, T]
            x = ts.forward(&x);
            x = x.permute(&[0, 2, 1]); // [B, T, H]
        }

        // 3. Transformer forward (combiner + pos_embed + subject_embed + encoder)
        if !self.config.linear_baseline {
            x = self.transformer_forward(&x, subject_ids);
        }

        // 4. Transpose to [B, H, T]
        x = x.permute(&[0, 2, 1]);

        // 5. Low-rank head
        if let Some(ref lr_weight) = self.low_rank_head {
            // x: [B, H, T] → transpose → [B, T, H] → matmul → [B, T, LR] → transpose → [B, LR, T]
            let (b, h, t) = (x.shape[0], x.shape[1], x.shape[2]);
            x = x.permute(&[0, 2, 1]); // [B, T, H]
            x = x.reshape(&[b * t, h]).matmul(lr_weight).reshape(&[b, t, lr_weight.shape[1]]);
            x = x.permute(&[0, 2, 1]); // [B, LR, T]
        }

        // 6. Predictor: [B, C, T] → [B, n_outputs, T]
        x = self.predictor.forward(&x, subject_ids);

        // 7. Adaptive average pool
        if pool_outputs {
            // x: [B, n_outputs, T] → pool over T
            let (b, d, _t) = (x.shape[0], x.shape[1], x.shape[2]);
            let mut pooled_data = Vec::with_capacity(b * d * self.n_output_timesteps);
            let t_in = x.shape[2];
            for bi in 0..b {
                for di in 0..d {
                    let base = bi * d * t_in + di * t_in;
                    for i in 0..self.n_output_timesteps {
                        let start = (i * t_in) / self.n_output_timesteps;
                        let end = ((i + 1) * t_in) / self.n_output_timesteps;
                        let len = (end - start) as f32;
                        let sum: f32 = x.data[base + start..base + end].iter().sum();
                        pooled_data.push(sum / len);
                    }
                }
            }
            x = Tensor::from_vec(pooled_data, vec![b, d, self.n_output_timesteps]);
        }

        x
    }

    /// Convenience: predict from pre-computed text features.
    ///
    /// `text_features`: [T, L*D] — one row per timestep, features already
    /// aggregated across layers (concatenated).
    ///
    /// Returns: [n_output_timesteps, n_outputs] for a single "batch" item.
    pub fn predict_from_text_features(
        &self,
        text_features: &[Vec<f32>],
        n_timesteps: usize,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        let t = text_features.len().min(n_timesteps);
        if t == 0 {
            anyhow::bail!("no timesteps provided");
        }
        let d = text_features[0].len();

        // Build features map: text only, other modalities zeroed
        let mut features = BTreeMap::new();

        // text: [1, L*D, T] (already layer-concatenated)
        let mut text_data = vec![0.0f32; d * t];
        for ti in 0..t {
            for di in 0..d {
                text_data[di * t + ti] = text_features[ti][di];
            }
        }
        features.insert("text".to_string(), Tensor::from_vec(text_data, vec![1, d, t]));

        // Run forward (average_subjects mode — no subject_ids needed)
        let out = self.forward(&features, None, true);

        // out: [1, n_outputs, n_output_timesteps]
        let n_out = out.shape[1];
        let t_out = out.shape[2];
        let mut result = Vec::with_capacity(t_out);
        for ti in 0..t_out {
            let mut row = Vec::with_capacity(n_out);
            for di in 0..n_out {
                row.push(out.data[di * t_out + ti]);
            }
            result.push(row);
        }
        Ok(result)
    }
}
