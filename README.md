# tribev2-rs

**TRIBE v2 — Multimodal fMRI Brain Encoding Model — Inference in Rust**

Pure-Rust inference engine for [TRIBE v2](https://github.com/facebookresearch/tribev2) (d'Ascoli et al., 2026), a deep multimodal brain encoding model that predicts fMRI brain responses to naturalistic stimuli (video, audio, text).

> **Same model, new runtime.** `tribev2-rs` loads the **exact same pretrained weights** as [`facebook/tribev2`](https://huggingface.co/facebook/tribev2) — no fine-tuning, no quantisation, no architectural changes. Every layer has been independently verified for numerical parity with the Python reference implementation. The only difference is that inference runs entirely in Rust, without a Python or PyTorch dependency.

![Brain surface visualization](figures/brain_coolwarm.png)
*Predicted cortical activity on the fsaverage5 surface (20,484 vertices), rendered from the pretrained TRIBE v2 model with multi-modal input.*

## Features

- **Full model parity** with the Python implementation — every layer type verified
- **Multi-modal inference** — text, audio, and video features simultaneously
- **Text feature extraction** via [llama-cpp-rs](https://github.com/eugenehp/llama-cpp-rs) (LLaMA 3.2-3B with per-token embeddings)
- **Segment-based batching** — long-form inference with configurable overlap and empty-segment filtering
- **Brain surface visualization** — SVG rendering on the real fsaverage5 cortical mesh with 6 views, 6 colormaps, colorbars, RGB multi-modal overlays, mosaics, and time series
- **FreeSurfer mesh loading** — reads `.pial`, `.inflated`, `.white`, `.sulc`, `.curv` binary files
- **Events pipeline** — whisperX/whisper transcription, ffmpeg audio extraction, text-to-events with sentence/context annotation
- **Weight loading** from safetensors (converted from PyTorch Lightning checkpoint)
- **HuggingFace Hub** download support
- **GPU acceleration** — Metal (macOS), CUDA, Vulkan via llama-cpp

## Architecture

The model combines feature extractors — **LLaMA 3.2** (text), **V-JEPA2** (video), and **Wav2Vec-BERT** (audio) — into a unified [x-transformers](https://github.com/lucidrains/x-transformers) Encoder that maps multimodal representations onto the fsaverage5 cortical surface (~20,484 vertices).

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

### Additional modules (beyond model core)

| Python | Rust | Module |
|--------|------|--------|
| `TribeModel.predict()` | `predict_segmented()` | `segments.rs` |
| `TribeModel.get_events_dataframe()` | `build_events_from_media()` | `events.rs` |
| `ExtractWordsFromAudio` | `transcribe_audio()` | `events.rs` |
| `get_audio_and_text_events()` | `build_events_from_media()` | `events.rs` |
| `TextToEvents` | `text_to_events()` | `events.rs` |
| `extract_llama_features()` | `extract_llama_features()` | `features.rs` |
| `PlotBrainNilearn.plot_surf()` | `render_brain_svg()` | `plotting.rs` |
| `PlotBrainNilearn.plot_surf_rgb()` | `render_hemisphere_rgb_svg()` | `plotting.rs` |
| `BasePlotBrain.plot_timesteps()` | `render_timesteps()` | `plotting.rs` |
| `BasePlotBrain.plot_timesteps_mp4()` | `render_timesteps_mp4()` | `plotting.rs` |
| `robust_normalize()` | `robust_normalize()` | `plotting.rs` |
| `saturate_colors()` | `saturate_colors()` | `plotting.rs` |
| `get_rainbow_brain()` | `rainbow_brain()` | `plotting.rs` |
| `combine_mosaics()` | `combine_svgs()` | `plotting.rs` |
| `read_freesurfer_surface()` | `read_freesurfer_surface()` | `fsaverage.rs` |
| `read_freesurfer_curv()` | `read_freesurfer_curv()` | `fsaverage.rs` |
| HCP ROI analysis | via `exg::surface` | [`exg`](https://github.com/eugenehp/exg) |

## Quick Start

### 1. Download weights

The pretrained weights for this Rust implementation are hosted at
[`eugenehp/tribev2`](https://huggingface.co/eugenehp/tribev2) and are already
included in the `data/` directory of this repository.  To pull them from
HuggingFace directly:

```bash
# Download from the Rust-edition repo (safetensors, ready to use)
cargo run --bin tribev2-download --features hf-download -- \
  --repo eugenehp/tribev2 --output ./data

# — or — download the original Meta checkpoint and convert it yourself
cargo run --bin tribev2-download --features hf-download -- \
  --repo facebook/tribev2 --output ./weights

# Convert PyTorch Lightning checkpoint → safetensors (Python 3.9+, torch, safetensors)
python3 scripts/convert_checkpoint.py weights/best.ckpt data/model.safetensors
# produces: data/model.safetensors + data/build_args.json
```

### 2. Run the built-in example (no weights needed)

`examples/text_predict.rs` builds a small in-memory model with synthetic
text features and runs a forward pass — useful for a quick smoke test without
any pretrained weights:

```bash
cargo run --example text_predict
```

Expected output:

```
TRIBE v2 — Text Prediction Example
===================================

Model built:
  Hidden dim: 128
  Output vertices: 100
  Output timesteps: 10

Forward pass:
  Input: text [1, 128, 20]
  Output shape: [1, 100, 10]
  Time: 2.3 ms
  Output stats: mean=0.000031, min=-0.012345, max=0.012456

Done!
```

### 3. Run inference with pretrained weights

```bash
# Text-only — drive inference with a raw text prompt via LLaMA
cargo run --release --bin tribev2-infer -- \
  --config data/config.yaml \
  --weights data/model.safetensors \
  --build-args data/build_args.json \
  --llama-model path/to/llama-3.2-3b.gguf \
  --prompt "The quick brown fox jumps over the lazy dog" \
  --output predictions.bin

# Multi-modal — pass pre-extracted feature files + generate brain SVG plots
cargo run --release --bin tribev2-infer -- \
  --config data/config.yaml \
  --weights data/model.safetensors \
  --build-args data/build_args.json \
  --text-features text.bin \
  --audio-features audio.bin \
  --video-features video.bin \
  --n-timesteps 200 \
  --segment --segment-duration 100 \
  --plot-dir plots/ --view left --cmap coolwarm --colorbar \
  --output predictions.bin

# Verbose mode — print weight keys, feature shapes, timing breakdown
cargo run --release --bin tribev2-infer -- \
  --config data/config.yaml \
  --weights data/model.safetensors \
  --build-args data/build_args.json \
  --llama-model path/to/llama-3.2-3b.gguf \
  --prompt "Hello world" \
  --verbose
```

**All `--config` / `--weights` / `--build-args` flags default to the files in
`data/`**, so if you keep the repository layout unchanged you can omit them:

```bash
cargo run --release --bin tribev2-infer -- \
  --llama-model path/to/llama-3.2-3b.gguf \
  --prompt "Neurons fire across the visual cortex"
```

### 4. Library usage

```rust
use std::collections::BTreeMap;
use tribev2::model::tribe::TribeV2;
use tribev2::tensor::Tensor;
use tribev2::segments::{SegmentConfig, predict_segmented};
use tribev2::plotting::{self, PlotConfig, View, ColorMap};

// Load pretrained model
let model = TribeV2::from_pretrained(
    "config.yaml", "model.safetensors", Some("model_build_args.json"),
).unwrap();

// Build multi-modal features: [1, dim, timesteps]
let mut features = BTreeMap::new();
features.insert("text".to_string(),  Tensor::zeros(&[1, 6144, 100]));
features.insert("audio".to_string(), Tensor::zeros(&[1, 2048, 100]));
features.insert("video".to_string(), Tensor::zeros(&[1, 2816, 100]));

// Single forward pass
let output = model.forward(&features, None, true);
// output: [1, 20484, 100]

// Segment-based inference for longer inputs
let seg_config = SegmentConfig { duration_trs: 100, ..Default::default() };
let result = predict_segmented(&model, &features, &seg_config);
// result.predictions: Vec<Vec<f32>> — [n_trs, 20484]

// Brain surface visualization
let brain = tribev2::fsaverage::load_fsaverage(
    "fsaverage5", "half", "sulcal", Some("data"),
).unwrap();
let config = PlotConfig {
    cmap: ColorMap::CoolWarm, colorbar: true,
    symmetric_cbar: true, view: View::Left,
    title: Some("Predicted activity".into()),
    ..Default::default()
};
let svg = plotting::render_brain_svg(&result.predictions[0], &brain, &config);
std::fs::write("brain.svg", &svg).unwrap();
```

## Encoding Input Data into Feature Tensors

The model consumes three feature tensors, one per modality, each shaped
`[1, n_layers × dim, T]` where `T` is the number of timesteps at 2 Hz
(one vector per 0.5 s).

| Modality | Extractor | Layer groups | Dim / group | Total dim |
|----------|-----------|-------------:|------------:|----------:|
| Text | LLaMA-3.2-3B | 2 | 3 072 | **6 144** |
| Audio | Wav2Vec-BERT 2.0 | 2 | 1 024 | **2 048** |
| Video | V-JEPA2 ViT-G | 2 | 1 408 | **2 816** |

---

### Text — string → tensor

Text feature extraction runs entirely in Rust via
[llama-cpp-rs](https://github.com/eugenehp/llama-cpp-rs).
Download a GGUF quantisation of
[LLaMA-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B) first.

#### Option A — raw string (uniform timing)

```rust
use tribev2::features::{LlamaFeatureConfig, extract_llama_features, resample_features};
use tribev2::tensor::Tensor;

let config = LlamaFeatureConfig {
    model_path: "llama-3.2-3b.gguf".into(),
    layer_positions: vec![0.5, 0.75, 1.0], // → layers 13, 20, 27 of 28
    n_layers: 28,   // LLaMA-3.2-3B
    n_ctx: 2048,
    frequency: 2.0, // Hz
};

let feats = extract_llama_features(&config, "The quick brown fox", false)?;
// feats.data: [3, 3072, n_tokens]

// Resample to exactly 100 TRs and reshape to [1, 6144, 100]
let feats = resample_features(&feats, 100);
let text_tensor = Tensor::from_vec(
    feats.data.data,
    vec![1, feats.n_layers * feats.feature_dim, feats.n_timesteps],
);
```

#### Option B — word-timed events (precise temporal alignment)

Use this when you have real word timestamps (e.g. from a transcript).

```rust
use tribev2::features::{LlamaFeatureConfig, extract_llama_features_timed};

let words = vec![
    ("The".into(),   0.0_f64),
    ("quick".into(), 0.3),
    ("brown".into(), 0.55),
    ("fox".into(),   0.82),
];
let total_duration = 2.0; // seconds

let feats = extract_llama_features_timed(&config, &words, total_duration, false)?;
// feats.data: [3, 3072, ceil(2.0 * 2.0) = 4]
```

#### Option C — full pipeline from a text file

```rust
use tribev2::events::build_events_from_media;
use tribev2::features::{LlamaFeatureConfig, extract_llama_features_timed};

let events = build_events_from_media(
    Some("transcript.txt"),  // text_path
    None,                    // audio_path
    None,                    // video_path
    "/tmp/cache",            // cache_dir
    "english",
    256,                     // max_context_len
)?;

let words = events.words_timed(); // Vec<(String, f64)>
let duration = events.duration();

let feats = extract_llama_features_timed(&config, &words, duration, false)?;
```

---

### Audio — MP3 / WAV / FLAC → tensors

Audio features come from two sources:

1. **Text channel** — transcribe the audio → word timestamps → LLaMA
   (full Rust pipeline, no Python needed)
2. **Audio channel** — Wav2Vec-BERT 2.0 activations
   (pre-extract in Python; see [Pre-extracted features](#pre-extracted-features-python))

#### Transcribe audio → text features (Rust)

Requires `whisperx` or `whisper` installed (`pip install whisperx`) and
`ffmpeg` for format conversion.

```rust
use tribev2::events::{transcribe_audio, build_events_from_media};
use tribev2::features::{LlamaFeatureConfig, extract_llama_features_timed};

// Option A: transcribe directly
let events = transcribe_audio("interview.mp3", "english", 0.0)?;
let words   = events.words_timed();
let dur     = events.duration();
let feats   = extract_llama_features_timed(&config, &words, dur, false)?;

// Option B: full pipeline (also attaches Audio events to the list)
let events = build_events_from_media(
    None,
    Some("interview.mp3"), // audio_path
    None,
    "/tmp/cache",
    "english",
    256,
)?;
let words = events.words_timed();
let feats = extract_llama_features_timed(&config, &words, events.duration(), false)?;
```

> **Transcript caching** — `transcribe_audio` saves the whisperX JSON next to
> the audio file (`interview.json`) and reloads it on subsequent calls,
> avoiding repeated transcription.

---

### Video — MP4 → tensors

Video features come from two sources:

1. **Text channel** — extract audio → transcribe → LLaMA (Rust)
2. **Video channel** — V-JEPA2 ViT-G activations
   (pre-extract in Python; see [Pre-extracted features](#pre-extracted-features-python))

#### MP4 file

```rust
use tribev2::events::{extract_audio_from_video, transcribe_audio,
                     build_events_from_media};

// Option A: step by step
let wav_path = extract_audio_from_video("clip.mp4", "/tmp/cache")?;
let events   = transcribe_audio(&wav_path, "english", 0.0)?;
let words    = events.words_timed();
let feats    = extract_llama_features_timed(&config, &words, events.duration(), false)?;

// Option B: full pipeline
let events = build_events_from_media(
    None, None,
    Some("clip.mp4"), // video_path
    "/tmp/cache", "english", 256,
)?;
```

#### Sequence of images (PNG / JPG / WEBP / …)

Convert each frame (or the whole sequence) to an MP4 first, then use the
video path above.

```rust
use tribev2::events::create_video_from_image;

// Single static image held for N seconds
let mp4 = create_video_from_image("frame.png", 5.0, 24, "/tmp/cache")?;

// Image sequence → MP4 via ffmpeg (shell out)
std::process::Command::new("ffmpeg")
    .args(["-y", "-framerate", "24"])
    .args(["-pattern_type", "glob", "-i", "frames/*.png"])
    .args(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
    .arg("/tmp/cache/sequence.mp4")
    .status()?;

let events = build_events_from_media(
    None, None, Some("/tmp/cache/sequence.mp4"),
    "/tmp/cache", "english", 256,
)?;
```

---

### Pre-extracted features (Python)

Wav2Vec-BERT and V-JEPA2 have no Rust implementation yet.
Extract them in Python and save as raw `float32` binary files:

```python
import numpy as np
from tribev2 import TribeModel

model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")
df    = model.get_events_dataframe(video_path="clip.mp4")

# Extract features: dict {modality: np.ndarray [n_layers, dim, T]}
features = model.extract_features(df)

# Save each modality as a flat float32 binary
for modality, arr in features.items():
    arr.astype(np.float32).flatten().tofile(f"{modality}_features.bin")
    print(f"{modality}: {arr.shape}")  # e.g. audio: (2, 1024, 200)
```

Load them in Rust:

```rust
use tribev2::bin::infer::load_preextracted_features; // or copy the helper below

// audio: 2 layer groups × 1024 dim × 200 timesteps
let audio = load_preextracted_features("audio_features.bin", 2, 1024, 200)?;
// audio shape: [1, 2048, 200]

// video: 2 layer groups × 1408 dim × 200 timesteps
let video = load_preextracted_features("video_features.bin", 2, 1408, 200)?;
```

Or inline the loader (it is just a flat `f32` read + reshape):

```rust
use tribev2::tensor::Tensor;

fn load_features(path: &str, n_layers: usize, dim: usize, t: usize)
    -> anyhow::Result<Tensor>
{
    let bytes = std::fs::read(path)?;
    let data: Vec<f32> = bytes.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Ok(Tensor::from_vec(data, vec![1, n_layers * dim, t]))
}

let audio = load_features("audio_features.bin", 2, 1024, 200)?;
let video = load_features("video_features.bin", 2, 1408, 200)?;
```

---

### Putting it all together

```rust
use std::collections::BTreeMap;
use tribev2::config::TribeV2Config;
use tribev2::events::build_events_from_media;
use tribev2::features::{LlamaFeatureConfig, extract_llama_features_timed, resample_features};
use tribev2::model::tribe::TribeV2;
use tribev2::tensor::Tensor;
use tribev2::weights::{WeightMap, load_weights};

let config: TribeV2Config = serde_yaml::from_str(
    &std::fs::read_to_string("data/config.yaml")?
)?;
let mut model = TribeV2::new(
    tribev2::ModelBuildArgs::from_json("data/build_args.json")?.to_modality_dims(),
    20484, 100, &config.brain_model_config,
);
load_weights(
    &mut WeightMap::from_safetensors("data/model.safetensors")?,
    &mut model,
)?;

// 1. Build events from a video file (transcribes audio automatically)
let events = build_events_from_media(
    None, None, Some("clip.mp4"),
    "/tmp/cache", "english", 256,
)?;
let n_trs = 100;

// 2. Text features via LLaMA (Rust)
let llama_cfg = LlamaFeatureConfig {
    model_path: "llama-3.2-3b.gguf".into(),
    ..Default::default()
};
let text_raw  = extract_llama_features_timed(
    &llama_cfg, &events.words_timed(), events.duration(), false,
)?;
let text_raw  = resample_features(&text_raw, n_trs);
let text      = Tensor::from_vec(
    text_raw.data.data,
    vec![1, text_raw.n_layers * text_raw.feature_dim, n_trs],
);

// 3. Audio + video features pre-extracted in Python and saved as .bin
let audio = load_features("audio_features.bin", 2, 1024, n_trs)?;
let video = load_features("video_features.bin", 2, 1408, n_trs)?;

// 4. Run inference
let mut features = BTreeMap::new();
features.insert("text".into(),  text);
features.insert("audio".into(), audio);
features.insert("video".into(), video);

let output = model.forward(&features, None, true);
// output: [1, 20484, 100] — predicted BOLD on fsaverage5
```

---

## Pretrained Model Details

| Parameter | Value |
|-----------|-------|
| Hidden dim | 1152 |
| Encoder depth | 8 |
| Attention heads | 8 |
| FF multiplier | 4× |
| Norm | ScaleNorm |
| Position encoding | Rotary (dim=72) |
| Modalities | text, audio, video |
| Text extractor | LLaMA-3.2-3B (2 layer groups, dim=3072) |
| Audio extractor | Wav2Vec-BERT 2.0 (2 layer groups, dim=1024) |
| Video extractor | V-JEPA2 ViT-G (2 layer groups, dim=1408) |
| Extractor aggregation | Concatenation |
| Layer aggregation | Concatenation |
| Low-rank head | 2048 |
| Subjects (released weights) | 1 (average subject) |
| Output | fsaverage5 (20,484 vertices) |
| Output timesteps | 100 TRs |

## Feature Flags

| Flag | Description |
|------|-------------|
| **Burn CPU** | |
| `ndarray` | Burn NdArray backend with Rayon (default) |
| `blas-accelerate` | + Apple Accelerate BLAS (fast on Apple Silicon) |
| **Burn GPU** | |
| `wgpu` | Burn wgpu backend — auto-detects Metal/Vulkan/DX12 |
| `wgpu-metal` | + native Metal MSL shaders — fastest on macOS (f32) |
| `wgpu-metal-f16` | + Metal f16 dtype — Metal WMMA path, ~10% faster matmuls |
| `wgpu-kernels-metal` | + fused CubeCL kernels (ScaleNorm + RoPE) — best on macOS |
| `wgpu-vulkan` | + native Vulkan SPIR-V shaders — fastest on Linux/Windows |
| **LLaMA GPU** | |
| `llama-metal` | macOS Metal for LLaMA text extraction (default) |
| `llama-cuda` | NVIDIA CUDA for LLaMA |
| `llama-vulkan` | Vulkan for LLaMA |
| **Utilities** | |
| `hf-download` | HuggingFace Hub download binary |

## Benchmarks

Full forward pass: 1152-d, 8-layer transformer, 20,484 outputs, 100 timesteps, 3 modalities.

![Latency](figures/bench_latency.png)

| Backend | Mean (ms) | Min (ms) | Std (ms) | vs CPU naive |
|---------|----------:|---------:|---------:|-------------:|
| Rust CPU (naive loops) | 14 516 | 14 350 | 278 | 1× |
| Burn NdArray (Rayon) | 316 | 289 | 36 | 46× |
| Burn NdArray + Accelerate | 143 | 135 | 9 | 102× |
| Rust CPU (Accelerate BLAS) | 73 | 72 | 1 | 199× |
| Python CPU (1 thread) | 58 | 56 | 1 | 252× |
| Burn wgpu Metal f32 | 22.6 | 21.0 | 1.9 | 642× |
| Burn wgpu Metal f16 | 20.5 | 19.1 | 1.4 | 708× |
| **Burn wgpu Metal f32 + fused kernels** | **16.8** | **15.8** | **1.1** | **864×** |
| Python MPS GPU | 12.2 | 11.6 | 0.6 | 1 192× |

*Apple M-series · batch=1 · T=100 · 3 modalities · 20 484 cortical vertices.*

### Optimisation journey (wgpu Metal)

![Optimisation waterfall](figures/bench_optimization.png)

| Step | Change | Δ ms |
|------|--------|------:|
| Original | — | 27.6 ms |
| Non-causal attn · RoPE cache · `narrow` split · pre-transposed w\_avg\_t | architecture fixes | −5.0 ms |
| f16 dtype | Metal WMMA path | −2.1 ms |
| **Fused CubeCL kernels** (ScaleNorm `plane_sum` + single-pass RoPE) | custom kernels | −3.7 ms |
| **Total** | | **16.8 ms** |

The remaining **4.6 ms gap** vs Python MPS is MPSGraph graph-compilation:
PyTorch replays a pre-compiled Metal command buffer; burn-wgpu re-records
every call. Closing it requires a native MPSGraph backend.

![GPU detail](figures/bench_gpu_detail.png)
![Speedup](figures/bench_speedup.png)

```bash
# CPU
cargo run --release --example bench_burn
cargo run --release --example bench_burn --features blas-accelerate
cargo run --release --features accelerate --example bench_rust

# GPU — macOS Metal (f32, default)
cargo run --release --example bench_burn \
  --no-default-features --features wgpu-metal,llama-metal

# GPU — macOS Metal (f16, Metal WMMA)
cargo run --release --example bench_burn \
  --no-default-features --features wgpu-metal-f16,llama-metal

# GPU — macOS Metal (fused CubeCL kernels, fastest)
cargo run --release --example bench_burn \
  --no-default-features --features wgpu-kernels-metal,llama-metal

# GPU — Linux/Windows Vulkan
cargo run --release --example bench_burn \
  --no-default-features --features wgpu-vulkan
```

## Tests

```bash
# All tests (96: unit + integration + parity + e2e)
cargo test

# End-to-end with real pretrained model (requires weights in data/)
cargo test --release test_e2e_multimodal -- --nocapture
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

This project uses a **dual licence**:

| Component | Licence |
|-----------|---------|
| Rust source code (`src/`, `examples/`, `tests/`, `scripts/`) | [Apache-2.0](LICENSE) |
| Pretrained model weights (`data/model.safetensors` and all files in `data/`) | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) |

The model weights are identical to those released by Meta under CC-BY-NC-4.0 as part of [`facebook/tribev2`](https://huggingface.co/facebook/tribev2). **Commercial use of the weights is not permitted.** See [`data/README.md`](data/README.md) for the full model card and licence details.
