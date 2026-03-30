# tribev2-rs

**TRIBE v2 — Multimodal fMRI Brain Encoding Model — Inference in Rust**

Pure-Rust inference engine for [TRIBE v2](https://github.com/facebookresearch/tribev2) (d'Ascoli et al., 2026), a deep multimodal brain encoding model that predicts fMRI brain responses to naturalistic stimuli (video, audio, text).

> **Same model, new runtime.** `tribev2-rs` loads the **exact same pretrained weights** as [`facebook/tribev2`](https://huggingface.co/facebook/tribev2) — no fine-tuning, no quantisation, no architectural changes. Every layer has been independently verified for numerical parity with the Python reference implementation.

![Brain surface visualization](figures/brain_coolwarm.png)
*Predicted cortical activity on the fsaverage5 surface (20,484 vertices), rendered from the pretrained TRIBE v2 model with multi-modal input.*

## Workspace Structure

```
tribev2-rs/
├── crates/
│   ├── tribev2/              Core brain encoding model, CLI, features, plotting
│   ├── tribev2-audio/        Wav2Vec-BERT 2.0 audio feature extraction (burn)
│   └── tribev2-video/        V-JEPA2 ViT-G video feature extraction (burn)
└── scripts/
    └── extract_llama_features.py   True per-layer LLaMA extraction (HuggingFace)
```

### Crate Overview

| Crate | Description |
|-------|-------------|
| **`tribev2`** | FmriEncoderModel (pure-Rust + burn backends), weight loading, segment-based inference, events pipeline, brain surface plotting, CLI |
| **`tribev2-audio`** | Wav2Vec-BERT 2.0 conformer encoder in burn — raw waveform → per-layer hidden states at 2 Hz |
| **`tribev2-video`** | V-JEPA2 ViT-Giant in burn — video frames → 3D patch embedding → ViT layers → per-layer features at 2 Hz |

## Features

- **100% inference parity** with the Python implementation — every operation verified
- **Two backends** — pure-Rust (CPU) and burn (CPU/GPU via NdArray, wgpu Metal, Vulkan)
- **Both backends load pretrained weights** from safetensors
- **Multi-modal inference** — text, audio, and video features simultaneously
- **Text feature extraction** — LLaMA 3.2-3B via llama-cpp (Rust) or HuggingFace (Python script for true per-layer extraction)
- **Audio feature extraction** — Wav2Vec-BERT 2.0 in burn (16 kHz waveform → conformer hidden states)
- **Video feature extraction** — V-JEPA2 ViT-G in burn (frames → 3D patch embedding → ViT hidden states)
- **Segment-based batching** — long-form inference with configurable overlap
- **Brain surface visualization** — SVG rendering on fsaverage5 cortical mesh (6 views, 6 colormaps, colorbars, RGB overlays, MP4 time series)
- **Events pipeline** — whisperX transcription, ffmpeg audio extraction, sentence/context annotation
- **HuggingFace Hub** download support

## Architecture

The model combines feature extractors — **LLaMA 3.2** (text), **V-JEPA2** (video), and **Wav2Vec-BERT** (audio) — into a unified [x-transformers](https://github.com/lucidrains/x-transformers) Encoder that maps multimodal representations onto the fsaverage5 cortical surface (~20,484 vertices).

| Component | Python | Rust (pure) | Rust (burn) |
|-----------|--------|-------------|-------------|
| Projector (Linear/MLP/SubjectLayers) | `Mlp` / `SubjectLayersModel` | `model::projector::Projector` | `model_burn::projector::Projector<B>` |
| Combiner | `Mlp` / `nn.Identity` | `Projector` (optional) | `MlpProjector<B>` (optional) |
| Temporal smoothing | depthwise `Conv1d` | `TemporalSmoothing` | depthwise conv kernel |
| Time positional embedding | `nn.Parameter` | `Tensor` | `Param<Tensor<B,3>>` |
| Subject embedding | `nn.Embedding` | `Tensor` | `Param<Tensor<B,2>>` |
| x-transformers Encoder | `x_transformers.Encoder` | `XTransformerEncoder` | `XTransformerEncoder<B>` |
| ScaleNorm + RoPE + Attention + FF | x_transformers | hand-written | burn ops (+ optional fused CubeCL) |
| Low-rank head | `nn.Linear(bias=False)` | `Tensor` matmul | `Linear<B>` |
| Subject layers | `SubjectLayersModel` | `SubjectLayers` | `SubjectLayers<B>` |
| AdaptiveAvgPool1d | `nn.AdaptiveAvgPool1d` | floor/ceil matching PyTorch | floor/ceil matching PyTorch |
| **Weight loading** | PyTorch `load_state_dict` | `weights::load_weights()` | `model_burn::weights::load_burn_weights()` |

## Quick Start

### 1. Download weights

```bash
cargo run --bin tribev2-download --features hf-download -- \
  --repo eugenehp/tribev2 --output ./data
```

### 2. Run inference

```bash
# Text-only with LLaMA
cargo run --release --bin tribev2-infer -- \
  --config data/config.yaml \
  --weights data/model.safetensors \
  --llama-model path/to/llama-3.2-3b.gguf \
  --prompt "The quick brown fox jumps over the lazy dog"

# Multi-modal with pre-extracted features + brain plots
cargo run --release --bin tribev2-infer -- \
  --config data/config.yaml \
  --weights data/model.safetensors \
  --text-features text.bin \
  --audio-features audio.bin \
  --video-features video.bin \
  --n-timesteps 200 --segment \
  --plot-dir plots/ --view left --cmap coolwarm --colorbar
```

### 3. True per-layer LLaMA features (exact Python parity)

The llama-cpp backend extracts final-layer embeddings only. For true per-layer
hidden states matching the Python pipeline:

```bash
# Extract with HuggingFace (requires: pip install transformers torch)
python scripts/extract_llama_features.py \
  --model meta-llama/Llama-3.2-3B \
  --input transcript.json \
  --output text_features.bin \
  --layers 0.5 0.75 1.0

# Use in Rust (auto-reads .json sidecar for shape metadata)
cargo run --release --bin tribev2-infer -- \
  --config data/config.yaml \
  --weights data/model.safetensors \
  --text-features text_features.bin
```

### 4. Library usage

```rust
use std::collections::BTreeMap;
use tribev2::model::tribe::TribeV2;
use tribev2::tensor::Tensor;

// Load pretrained model
let model = TribeV2::from_pretrained(
    "config.yaml", "model.safetensors", Some("build_args.json"),
)?;

// Build features: [1, n_layers*dim, timesteps]
let mut features = BTreeMap::new();
features.insert("text".into(),  Tensor::zeros(&[1, 9216, 100]));
features.insert("audio".into(), Tensor::zeros(&[1, 3072, 100]));
features.insert("video".into(), Tensor::zeros(&[1, 4224, 100]));

// Forward pass → [1, 20484, 100]
let output = model.forward(&features, None, true);
```

### 5. Burn backend (GPU inference)

```rust
use tribev2::config::{ModalityDims, TribeV2Config};
use tribev2::model_burn::tribe::TribeV2Burn;
use tribev2::model_burn::weights::{BurnWeightStore, load_burn_weights};

type B = burn::backend::NdArray;  // or burn::backend::Wgpu
let device = Default::default();

let config: TribeV2Config = serde_yaml::from_str(&std::fs::read_to_string("config.yaml")?)?;
let dims = ModalityDims::pretrained();

let mut model = TribeV2Burn::<B>::new(&dims, 20484, 100, &config.brain_model_config, &device);

// Load pretrained weights into burn model
let mut ws = BurnWeightStore::from_safetensors("model.safetensors")?;
load_burn_weights(&mut ws, &mut model, &device)?;

// Forward pass
let text  = burn::tensor::Tensor::<B, 3>::zeros([1, 9216, 100], &device);
let audio = burn::tensor::Tensor::<B, 3>::zeros([1, 3072, 100], &device);
let video = burn::tensor::Tensor::<B, 3>::zeros([1, 4224, 100], &device);

let output = model.forward(vec![("text", text), ("audio", audio), ("video", video)]);
// output: [1, 20484, 100]
```

## Audio Feature Extraction (tribev2-audio)

```rust
use tribev2_audio::{Wav2VecBertConfig, Wav2VecBertWithConfig};
use tribev2_audio::audio_io::load_audio;
use tribev2_audio::weights::{WeightStore, load_wav2vec_bert_weights};

type B = burn::backend::NdArray;
let device = Default::default();
let config = Wav2VecBertConfig::default();  // facebook/w2v-bert-2.0

let mut model = Wav2VecBertWithConfig::<B>::new(&config, &device);

// Load HuggingFace weights
let mut ws = WeightStore::from_safetensors("w2v-bert-2.0/model.safetensors")?;
load_wav2vec_bert_weights(&mut ws, &mut model, &device)?;

// Extract features
let waveform = load_audio("audio.wav", 16000)?;
let features = model.extract_features(&waveform, 60.0, &device);
// features: [3, 1024, 120] at 2 Hz
```

## Video Feature Extraction (tribev2-video)

```rust
use tribev2_video::{VJepa2Config, VJepa2WithConfig};
use tribev2_video::video_io;
use tribev2_video::weights::{WeightStore, load_vjepa2_weights};

type B = burn::backend::NdArray;
let device = Default::default();
let config = VJepa2Config::default();  // facebook/vjepa2-vitg-fpc64-256

let mut model = VJepa2WithConfig::<B>::new(&config, &device);

let mut ws = WeightStore::from_safetensors("vjepa2/model.safetensors")?;
load_vjepa2_weights(&mut ws, &mut model, &device)?;

// Extract frames and run model
// (see tribev2-video docs for full frame preprocessing pipeline)
```

## Pretrained Model Details

| Parameter | Value |
|-----------|-------|
| Hidden dim | 1152 |
| Encoder depth | 8 layers (8 attn + 8 FF) |
| Attention heads | 8 |
| FF multiplier | 4× |
| Norm | ScaleNorm |
| Position encoding | Rotary (dim=72) |
| Text extractor | LLaMA-3.2-3B (3 layer groups × 3072) |
| Audio extractor | Wav2Vec-BERT 2.0 (3 layer groups × 1024) |
| Video extractor | V-JEPA2 ViT-G (3 layer groups × 1408) |
| Low-rank head | 2048 |
| Output | fsaverage5 (20,484 vertices), 100 TRs |

## Feature Flags

| Flag | Description |
|------|-------------|
| `ndarray` | Burn NdArray CPU backend (default) |
| `blas-accelerate` | + Apple Accelerate BLAS |
| `wgpu` | Burn wgpu backend (auto-detects Metal/Vulkan/DX12) |
| `wgpu-metal` | + native Metal MSL shaders |
| `wgpu-metal-f16` | + Metal f16 dtype (WMMA) |
| `wgpu-kernels-metal` | + fused CubeCL kernels (fastest macOS) |
| `wgpu-vulkan` | + Vulkan SPIR-V shaders |
| `llama-metal` | Metal GPU for LLaMA (default) |
| `llama-cuda` | CUDA for LLaMA |
| `llama-vulkan` | Vulkan for LLaMA |
| `hf-download` | HuggingFace Hub download support |

## Benchmarks

Full forward pass: 1152-d, 8-layer transformer, 20,484 outputs, 100 timesteps, 3 modalities.

| Backend | Mean (ms) | Speedup |
|---------|----------:|--------:|
| Rust CPU (naive) | 14,516 | 1× |
| Burn NdArray | 316 | 46× |
| Burn NdArray + Accelerate | 143 | 102× |
| Rust CPU + Accelerate | 73 | 199× |
| **Burn wgpu Metal + fused kernels** | **16.8** | **864×** |

```bash
cargo run --release --example bench_burn
cargo run --release --example bench_burn --no-default-features --features wgpu-kernels-metal,llama-metal
```

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

| Component | License |
|-----------|---------|
| Rust source code | [Apache-2.0](LICENSE) |
| Pretrained model weights | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) |
