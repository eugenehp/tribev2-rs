# tribev2-rs

**TRIBE v2 — Multimodal fMRI Brain Encoding Model — Inference in Rust**

Pure-Rust inference for [TRIBE v2](https://github.com/facebookresearch/tribev2) (d'Ascoli et al., 2026), a deep multimodal brain encoding model that predicts fMRI brain responses to naturalistic stimuli (video, audio, text).

## Architecture

The model combines feature extractors — **LLaMA 3.2** (text), **V-JEPA2** (video), and **Wav2Vec-BERT** (audio) — into a unified [x-transformers](https://github.com/lucidrains/x-transformers) Encoder that maps multimodal representations onto the fsaverage5 cortical surface (~20,484 vertices).

This crate provides a complete Rust reimplementation of the `FmriEncoderModel`:

| Component | Python | Rust |
|-----------|--------|------|
| Per-modality projectors | `nn.Linear` / `torchvision MLP` | `model::projector::Projector` |
| Feature aggregation | concat / sum / stack | `TribeV2::aggregate_features` |
| Combiner | `nn.Linear` / `nn.Identity` | `model::projector::Projector` (optional) |
| Time positional embedding | `nn.Parameter` | `Tensor` |
| Transformer encoder | `x_transformers.Encoder` | `model::encoder::XTransformerEncoder` |
| ScaleNorm | `x_transformers.ScaleNorm` | `model::scalenorm::ScaleNorm` |
| Rotary position embedding | `x_transformers.RotaryEmbedding` | `model::rotary::RotaryEmbedding` |
| Multi-head attention | `x_transformers.Attention` | `model::attention::Attention` |
| FeedForward (GELU) | `x_transformers.FeedForward` | `model::feedforward::FeedForward` |
| Scaled residual | `x_transformers.Residual` | `model::residual::Residual` |
| Low-rank head | `nn.Linear(bias=False)` | `Tensor` matmul |
| Subject layers | `SubjectLayersModel` | `model::subject_layers::SubjectLayers` |
| Temporal smoothing | depthwise `nn.Conv1d` | `model::temporal_smoothing::TemporalSmoothing` |
| Adaptive avg pool | `nn.AdaptiveAvgPool1d` | `Tensor::adaptive_avg_pool1d` |

## Features

- **Text feature extraction**: Uses [llama-cpp-rs](https://github.com/eugenehp/llama-cpp-rs) for LLaMA 3.2-3B inference
- **Weight loading**: Loads from safetensors (converted from PyTorch Lightning checkpoint)
- **HuggingFace Hub**: Download pretrained weights from `facebook/tribev2`
- **GPU acceleration**: Metal (macOS), CUDA, Vulkan via llama-cpp

## Quick Start

### 1. Download weights

```bash
cargo run --bin tribev2-download --features hf-download -- --repo facebook/tribev2
```

### 2. Convert checkpoint to safetensors

```bash
python3 -c "
import torch, safetensors.torch
ckpt = torch.load('weights/best.ckpt', map_location='cpu', weights_only=True)
sd = {k.removeprefix('model.'): v for k, v in ckpt['state_dict'].items()}
safetensors.torch.save_file(sd, 'weights/model.safetensors')
"
```

### 3. Run inference

```bash
cargo run --release --bin tribev2-infer -- \
  --config weights/config.yaml \
  --weights weights/model.safetensors \
  --llama-model path/to/llama-3.2-3b.gguf \
  --prompt "The quick brown fox jumps over the lazy dog" \
  --output predictions.bin
```

### 4. Library usage

```rust
use tribev2_rs::{TribeV2, TribeV2Config, ModalityDims};
use tribev2_rs::weights::{WeightMap, load_weights};

// Load config
let config: TribeV2Config = serde_yaml::from_str(&std::fs::read_to_string("config.yaml")?)?;

// Build model
let feature_dims = ModalityDims::pretrained();
let mut model = TribeV2::new(feature_dims, 20484, 100, &config.brain_model_config);

// Load weights
let mut wm = WeightMap::from_safetensors("model.safetensors")?;
load_weights(&mut wm, &mut model)?;

// Run inference with text features
let predictions = model.predict_from_text_features(&text_features, 100)?;
// predictions: Vec<Vec<f32>> — [n_timesteps × n_vertices]
```

## Pretrained Model Details

From `config.yaml`:

| Parameter | Value |
|-----------|-------|
| Hidden dim | 1152 |
| Encoder depth | 8 |
| Attention heads | 8 |
| FF multiplier | 4× |
| Norm | ScaleNorm |
| Position encoding | Rotary (dim=72) |
| Modalities | text, audio, video |
| Text extractor | LLaMA-3.2-3B (layers 50%, 75%, 100%) |
| Audio extractor | Wav2Vec-BERT 2.0 (layers 50%, 75%, 100%) |
| Video extractor | V-JEPA2 ViT-G (layers 50%, 75%, 100%) |
| Extractor aggregation | Concatenation |
| Layer aggregation | Concatenation |
| Low-rank head | 2048 |
| Subject layers | 25 subjects + dropout |
| Output | fsaverage5 (20,484 vertices) |
| Output timesteps | 100 TRs |

## Feature Flags

| Flag | Description |
|------|-------------|
| `metal` | macOS Metal GPU for LLaMA (default) |
| `cuda` | NVIDIA CUDA GPU for LLaMA |
| `vulkan` | Vulkan GPU for LLaMA |
| `hf-download` | Enable HuggingFace Hub download binary |

## Citation

```bibtex
@article{dAscoli2026TribeV2,
  title={A foundation model of vision, audition, and language for in-silico neuroscience},
  author={d'Ascoli, St{\'e}phane and Rapin, J{\'e}r{\'e}my and Benchetrit, Yohann and
          Brookes, Teon and Begany, Katelyn and Raugel, Jos{\'e}phine and
          Banville, Hubert and King, Jean-R{\'e}mi},
  year={2026}
}
```

## License

Apache-2.0
