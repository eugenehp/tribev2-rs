# Cargo feature flags

All optional engines and device backends are **off by default** (`tribev2`: `default = []`). The pure-Rust encoder (`TribeV2`) is always built.

## tribev2

| Feature | Enables |
|---------|---------|
| *(default)* | `TribeV2`, CLI, plotting, segments, weights |
| `pure-rust` | Bench/CI marker (core model does not require it) |
| `burn` | `TribeV2Burn`, `model_burn` |
| `ndarray` / `burn-cpu` | Burn CPU matmul |
| `blas-accelerate` | Accelerate BLAS for Burn NdArray |
| `wgpu`, `wgpu-metal`, `wgpu-metal-f16`, `wgpu-kernels-metal`, `wgpu-vulkan` | Burn GPU |
| `rlx-encoder` | `TribeRlx`, `model_rlx` |
| `rlx-cpu`, `rlx-metal`, `rlx-cuda`, `rlx-mlx`, `rlx-gpu`, … | RLX device runtimes |
| `rlx-cuda-enc`, `rlx-gpu-enc` | RLX encoder only (no LLaMA stack; rig/CI) |
| `rlx-text` / `rlx` | LLaMA features via `rlx-llama32` |
| `audio-rlx` / `video-rlx` | Optional extractor crates |
| `all-engines-cpu` | `pure-rust` + `burn` + `rlx-encoder` + `rlx-cpu` |
| `apple-silicon` | Preset: Metal/MLX + Burn wgpu-metal |
| `nvidia-gpu` | Preset: CUDA + Burn wgpu |
| `hf-download` | `tribev2-download` binary |

## tribev2-audio / tribev2-video

`default = []`. Enable one device, e.g. `rlx-metal` or `rlx-cuda`.

## Examples

```bash
# Default — Rust encoder only
cargo build -p tribev2

# Burn GPU (macOS)
cargo build -p tribev2 --no-default-features --features burn,wgpu-metal

# RLX CUDA encoder (Linux)
cargo build -p tribev2 --no-default-features --features rlx-encoder,rlx-cuda-enc
```

See [bench/README.md](../bench/README.md) for benchmark feature sets per backend.
