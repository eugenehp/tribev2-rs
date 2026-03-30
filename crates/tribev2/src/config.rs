//! Configuration types mirroring the Python TRIBE v2 config.yaml.
//!
//! Field names and semantics match the Python `FmriEncoder` / `TransformerEncoder`
//! / `SubjectLayers` config classes exactly.

use serde::Deserialize;
use std::collections::BTreeMap;

// ── Top-level experiment config (subset relevant to inference) ─────────────

/// Top-level TRIBE v2 configuration, parsed from `config.yaml`.
#[derive(Debug, Clone, Deserialize)]
pub struct TribeV2Config {
    pub brain_model_config: BrainModelConfig,
    pub data: DataConfig,
    #[serde(default)]
    pub average_subjects: bool,
    #[serde(default)]
    pub seed: Option<u64>,
}

// ── Data config ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct DataConfig {
    #[serde(default = "default_features_to_use")]
    pub features_to_use: Vec<String>,
    #[serde(default)]
    pub features_to_mask: Vec<String>,
    #[serde(default = "default_duration_trs")]
    pub duration_trs: usize,
    #[serde(default)]
    pub overlap_trs_val: usize,
    #[serde(default)]
    pub stride_drop_incomplete: bool,
    #[serde(default)]
    pub frequency: Option<f64>,
    pub text_feature: Option<TextFeatureConfig>,
    pub audio_feature: Option<AudioFeatureConfig>,
    pub video_feature: Option<VideoFeatureConfig>,
    pub subject_id: Option<SubjectIdConfig>,
}

fn default_features_to_use() -> Vec<String> {
    vec!["text".into(), "audio".into(), "video".into()]
}

fn default_duration_trs() -> usize { 100 }

#[derive(Debug, Clone, Deserialize)]
pub struct TextFeatureConfig {
    pub model_name: Option<String>,
    #[serde(default)]
    pub layers: Vec<f64>,
    #[serde(default)]
    pub layer_aggregation: Option<String>,
    #[serde(default = "default_frequency")]
    pub frequency: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AudioFeatureConfig {
    pub model_name: Option<String>,
    #[serde(default)]
    pub layers: Vec<f64>,
    #[serde(default)]
    pub layer_aggregation: Option<String>,
    #[serde(default = "default_frequency")]
    pub frequency: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VideoFeatureConfig {
    pub image: Option<VideoImageConfig>,
    #[serde(default)]
    pub layers: Vec<f64>,
    #[serde(default)]
    pub layer_aggregation: Option<String>,
    #[serde(default = "default_frequency")]
    pub frequency: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VideoImageConfig {
    pub model_name: Option<String>,
    #[serde(default)]
    pub layers: Vec<f64>,
    #[serde(default)]
    pub layer_aggregation: Option<String>,
}

fn default_frequency() -> f64 { 2.0 }

#[derive(Debug, Clone, Deserialize)]
pub struct SubjectIdConfig {
    #[serde(default)]
    pub predefined_mapping: Option<BTreeMap<String, usize>>,
}

// ── Brain model config (FmriEncoder) ──────────────────────────────────────

/// Python: `FmriEncoder` in model.py — the top-level brain model config.
#[derive(Debug, Clone, Deserialize)]
pub struct BrainModelConfig {
    /// Projector config (Mlp). When hidden_sizes is None/empty, it's a single Linear.
    #[serde(default)]
    pub projector: MlpConfig,

    /// Combiner config. None → nn.Identity.
    #[serde(default)]
    pub combiner: Option<MlpConfig>,

    /// x_transformers Encoder config. None → no transformer (linear baseline).
    #[serde(default)]
    pub encoder: Option<EncoderConfig>,

    /// Whether to add learned time positional embedding.
    #[serde(default = "default_true")]
    pub time_pos_embedding: bool,

    /// Whether to add learned per-subject embedding.
    #[serde(default)]
    pub subject_embedding: bool,

    /// Per-subject prediction layer config.
    #[serde(default)]
    pub subject_layers: Option<SubjectLayersConfig>,

    /// Hidden dimension of the transformer / combiner output.
    #[serde(default = "default_hidden")]
    pub hidden: usize,

    /// Max sequence length for time positional embedding.
    #[serde(default = "default_max_seq_len")]
    pub max_seq_len: usize,

    /// Dropout (applied to encoder attention / ff / layer drop).
    #[serde(default)]
    pub dropout: f64,

    /// How to combine modality features: "cat", "sum", or "stack".
    #[serde(default = "default_cat")]
    pub extractor_aggregation: String,

    /// How to aggregate layers within a modality: "cat" or "mean".
    #[serde(default = "default_cat")]
    pub layer_aggregation: String,

    /// If true, skip the transformer (just projectors → predictor).
    #[serde(default)]
    pub linear_baseline: bool,

    /// Probability of zeroing out an entire modality during training.
    #[serde(default)]
    pub modality_dropout: f64,

    /// Probability of zeroing out a timestep during training.
    #[serde(default)]
    pub temporal_dropout: f64,

    /// If set, insert Linear(hidden, low_rank_head, bias=False) before predictor.
    #[serde(default)]
    pub low_rank_head: Option<usize>,

    /// Temporal smoothing (depthwise Conv1d with optional Gaussian kernel).
    #[serde(default)]
    pub temporal_smoothing: Option<TemporalSmoothingConfig>,
}

fn default_true() -> bool { true }
fn default_hidden() -> usize { 1152 }
fn default_max_seq_len() -> usize { 1024 }
fn default_cat() -> String { "cat".into() }

impl Default for BrainModelConfig {
    fn default() -> Self {
        Self {
            projector: Default::default(),
            combiner: None,
            encoder: Some(EncoderConfig::default()),
            time_pos_embedding: true,
            subject_embedding: false,
            subject_layers: Some(SubjectLayersConfig::default()),
            hidden: 1152,
            max_seq_len: 1024,
            dropout: 0.0,
            extractor_aggregation: "cat".into(),
            layer_aggregation: "cat".into(),
            linear_baseline: false,
            modality_dropout: 0.0,
            temporal_dropout: 0.0,
            low_rank_head: None,
            temporal_smoothing: None,
        }
    }
}

// ── MLP config ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize, Default)]
pub struct MlpConfig {
    pub input_size: Option<usize>,
    pub hidden_sizes: Option<Vec<usize>>,
    pub norm_layer: Option<String>,
    pub activation_layer: Option<String>,
    #[serde(default = "default_true")]
    pub bias: bool,
    #[serde(default)]
    pub dropout: f64,
    /// Discriminator key used by exca; ignored in Rust.
    #[serde(default)]
    pub name: Option<String>,
}

impl MlpConfig {
    /// Determine what this MLP builds to.
    /// - No hidden_sizes + no output → Identity
    /// - No hidden_sizes + output → single Linear
    /// - hidden_sizes present + output → torchvision MLP
    pub fn is_identity(&self, output_size: Option<usize>) -> bool {
        self.hidden_sizes.as_ref().map_or(true, |h| h.is_empty()) && output_size.is_none()
    }

    pub fn is_single_linear(&self, output_size: Option<usize>) -> bool {
        self.hidden_sizes.as_ref().map_or(true, |h| h.is_empty()) && output_size.is_some()
    }
}

// ── x_transformers Encoder config ─────────────────────────────────────────

/// Mirrors `TransformerEncoder` in neuraltrain, which builds an
/// `x_transformers.Encoder`.
#[derive(Debug, Clone, Deserialize)]
pub struct EncoderConfig {
    #[serde(default = "default_heads")]
    pub heads: usize,

    #[serde(default = "default_depth")]
    pub depth: usize,

    #[serde(default)]
    pub cross_attend: bool,

    #[serde(default)]
    pub causal: bool,

    #[serde(default)]
    pub attn_flash: bool,

    #[serde(default)]
    pub attn_dropout: f64,

    #[serde(default = "default_ff_mult")]
    pub ff_mult: usize,

    #[serde(default)]
    pub ff_dropout: f64,

    #[serde(default = "default_true")]
    pub use_scalenorm: bool,

    #[serde(default)]
    pub use_rmsnorm: bool,

    #[serde(default)]
    pub rel_pos_bias: bool,

    #[serde(default)]
    pub alibi_pos_bias: bool,

    #[serde(default = "default_true")]
    pub rotary_pos_emb: bool,

    #[serde(default)]
    pub rotary_xpos: bool,

    #[serde(default)]
    pub residual_attn: bool,

    #[serde(default = "default_true")]
    pub scale_residual: bool,

    #[serde(default)]
    pub layer_dropout: f64,

    /// Discriminator key from exca; ignored.
    #[serde(default)]
    pub name: Option<String>,
}

fn default_heads() -> usize { 8 }
fn default_depth() -> usize { 8 }
fn default_ff_mult() -> usize { 4 }

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            heads: 8,
            depth: 8,
            cross_attend: false,
            causal: false,
            attn_flash: false,
            attn_dropout: 0.0,
            ff_mult: 4,
            ff_dropout: 0.0,
            use_scalenorm: true,
            use_rmsnorm: false,
            rel_pos_bias: false,
            alibi_pos_bias: false,
            rotary_pos_emb: true,
            rotary_xpos: false,
            residual_attn: false,
            scale_residual: true,
            layer_dropout: 0.0,
            name: None,
        }
    }
}

impl EncoderConfig {
    /// dim_head = dim / heads  (attn_dim_head in x_transformers)
    pub fn dim_head(&self, dim: usize) -> usize {
        dim / self.heads
    }

    /// rotary_emb_dim = max(dim_head // 2, 32)
    pub fn rotary_emb_dim(&self, dim: usize) -> usize {
        (self.dim_head(dim) / 2).max(32)
    }

    /// FF inner dimension = dim * ff_mult
    pub fn ff_inner_dim(&self, dim: usize) -> usize {
        dim * self.ff_mult
    }
}

// ── Subject layers config ─────────────────────────────────────────────────

/// Mirrors `SubjectLayers` in neuraltrain/models/common.py.
#[derive(Debug, Clone, Deserialize)]
pub struct SubjectLayersConfig {
    #[serde(default = "default_n_subjects")]
    pub n_subjects: usize,

    #[serde(default = "default_true")]
    pub bias: bool,

    #[serde(default)]
    pub init_id: bool,

    #[serde(default = "default_gather")]
    pub mode: String,

    #[serde(default)]
    pub subject_dropout: Option<f64>,

    #[serde(default)]
    pub average_subjects: bool,

    /// Discriminator key from exca; ignored.
    #[serde(default)]
    pub name: Option<String>,
}

fn default_n_subjects() -> usize { 25 }
fn default_gather() -> String { "gather".into() }

impl Default for SubjectLayersConfig {
    fn default() -> Self {
        Self {
            n_subjects: 25,
            bias: true,
            init_id: false,
            mode: "gather".into(),
            subject_dropout: Some(0.1),
            average_subjects: false,
            name: None,
        }
    }
}

impl SubjectLayersConfig {
    /// Total number of weight rows (extra row for dropout subject).
    pub fn num_weight_subjects(&self) -> usize {
        if self.subject_dropout.is_some() {
            self.n_subjects + 1
        } else {
            self.n_subjects
        }
    }
}

// ── Temporal smoothing config ─────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct TemporalSmoothingConfig {
    #[serde(default = "default_kernel_size")]
    pub kernel_size: usize,
    #[serde(default)]
    pub sigma: Option<f64>,
    /// Discriminator key from exca; ignored.
    #[serde(default)]
    pub name: Option<String>,
}

fn default_kernel_size() -> usize { 9 }

// ── Feature dimension spec ────────────────────────────────────────────────

/// Per-modality feature dimensions: (num_layers, feature_dim) or None.
/// Python: `feature_dims: dict[str, tuple[int, int] | None]`
///
/// For the pretrained model:
/// - text:  (3, 3072)  — LLaMA-3.2-3B, 3 layer groups, hidden=3072
/// - audio: (3, 1024)  — Wav2Vec-BERT 2.0, 3 layer groups, hidden=1024
/// - video: (3, 1408)  — V-JEPA2 ViT-G, 3 layer groups, hidden=1408
#[derive(Debug, Clone)]
pub struct ModalityDims {
    pub name: String,
    /// None means this modality has no feature dimensions (no projector built).
    pub dims: Option<(usize, usize)>,
}

impl ModalityDims {
    pub fn new(name: &str, num_layers: usize, feature_dim: usize) -> Self {
        Self { name: name.to_string(), dims: Some((num_layers, feature_dim)) }
    }

    pub fn none(name: &str) -> Self {
        Self { name: name.to_string(), dims: None }
    }

    pub fn num_layers(&self) -> usize {
        self.dims.map_or(0, |(l, _)| l)
    }

    pub fn feature_dim(&self) -> usize {
        self.dims.map_or(0, |(_, d)| d)
    }

    /// Pretrained TRIBE v2 modality dims.
    pub fn pretrained() -> Vec<Self> {
        vec![
            Self::new("text", 3, 3072),
            Self::new("audio", 3, 1024),
            Self::new("video", 3, 1408),
        ]
    }
}

/// Build args saved alongside the checkpoint (from `model_build_args` in .ckpt).
/// JSON format produced by `scripts/convert_checkpoint.py`.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelBuildArgs {
    /// feature_dims: {"text": [3, 3072], "audio": [3, 1024], ...} or null
    pub feature_dims: BTreeMap<String, Option<Vec<usize>>>,
    pub n_outputs: usize,
    pub n_output_timesteps: usize,
}

impl ModelBuildArgs {
    /// Load from a JSON file.
    pub fn from_json(path: &str) -> anyhow::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&json)?)
    }

    /// Convert to ordered Vec<ModalityDims>, preserving key order from the JSON.
    pub fn to_modality_dims(&self) -> Vec<ModalityDims> {
        self.feature_dims.iter().map(|(name, dims)| {
            match dims {
                Some(v) if v.len() == 2 => ModalityDims::new(name, v[0], v[1]),
                _ => ModalityDims::none(name),
            }
        }).collect()
    }
}
