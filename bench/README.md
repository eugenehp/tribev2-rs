# Encoder benchmarks

Unified benchmark: `cargo run --release -p tribev2 --example bench_encoder`  
(`crates/tribev2/bench/bench_encoder.rs`)

## Methodology

| Item | Value |
|------|--------|
| Model | TRIBE v2 encoder (1152-d hidden, 8 layers, low-rank head 2048) |
| Output | `[1, 20484, 100]` vertices × output TRs |
| Modalities | From `data/build_args.json` (checkpoint layout: 2×3072 text, 2×1024 audio, 2×1408 video) |
| Metric | Mean forward pass time (ms), after warmup |

| Engine | Weights in bench |
|--------|------------------|
| **pure-Rust** (`TribeV2`) | Loads `data/model.safetensors` |
| **RLX** (`TribeRlx`) | Loads `data/model.safetensors` |
| **Burn** (`TribeV2Burn`) | Random init (same architecture; throughput only) |

Sweep all local backends: `./bench/run_all_backends.sh`  
CUDA / Windows / WSL: `./bench/run_cuda_rig.sh` or `./rig.sh --both bench-cuda`

## Latest sweep — Apple Silicon (May 2026)

**Host:** Apple M4 Pro · **T=10** · warmup=1 · runs=2 · `TRIBEV2_DATA_DIR=./data`

### All backends

| Backend | Engine | Device | Mean (ms) | Min | Max | vs Rust |
|---------|--------|--------|----------:|----:|----:|--------:|
| pure-Rust | rust | cpu | 1333.5 | 1326.9 | 1340.1 | 1.0× |
| Burn | burn | ndarray | 163.9 | 147.7 | 180.2 | 8.1× |
| Burn | burn | ndarray-accelerate | 66.7 | 66.1 | 67.4 | 20.0× |
| Burn | burn | wgpu-metal (f32) | 13.6 | 12.6 | 14.5 | 98.1× |
| Burn | burn | wgpu-metal-f16 | **10.6** | 9.8 | 11.3 | **125.8×** |
| RLX | rlx | cpu | 33.8 | 32.2 | 35.4 | 39.5× |
| RLX | rlx | metal | 15.1 | 14.6 | 15.5 | 88.3× |
| RLX | rlx | mlx | 45.7 | 35.1 | 56.4 | 29.2× |
| RLX | rlx | wgpu | 75.8 | 75.7 | 76.0 | 17.6× |

Raw JSON: `bench/results_*.json` (committed).

### Burn vs RLX (GPU on macOS)

| Stack | Mean (ms) | Notes |
|-------|----------:|-------|
| **Burn wgpu Metal f16** | **10.6** | Fastest in sweep |
| Burn wgpu Metal f32 | 13.6 | |
| RLX Metal | 15.1 | Loaded weights; best RLX backend on Apple Silicon |
| RLX MLX | 45.7 | |
| RLX wgpu | 75.8 | Prefer Metal over wgpu on macOS |

### CPU

| Stack | Mean (ms) |
|-------|----------:|
| RLX CPU | 33.8 |
| Burn NdArray + Accelerate | 66.7 |
| Burn NdArray | 163.9 |
| pure-Rust CPU | 1333.5 |

## Historical — T=100 forward (same machine family)

From `figures/bench_table.md` / fused-kernel run (`bench/results_rust_burn_wgpu_metal_kernels.json` in crate tree).

| Backend | Mean (ms) | vs naive Rust |
|---------|----------:|--------------:|
| Rust CPU (naive) | 14,516.5 | 1× |
| Burn NdArray | 316.2 | 46× |
| Burn NdArray + Accelerate | 142.7 | 102× |
| Burn wgpu Metal f32 | 22.6 | 642× |
| Burn wgpu Metal f16 | 20.5 | 708× |
| **Burn wgpu Metal + fused CubeCL** | **16.8** | **864×** |

## Pipeline benchmark — T=20 (README reference run)

Apple M4 Pro; forward + NIfTI + ROI (see main README). Forward-only highlights:

| Backend | Forward (ms) | Speedup vs Rust |
|---------|----------:|----------------:|
| pure-Rust CPU | 3,028 | 1× |
| Burn NdArray CPU | 355 | 8.5× |
| Burn wgpu Metal GPU (warm on device) | **36** | **84×** |

## Remote CUDA (Windows + WSL)

Not yet recorded in-repo. After `./rig.sh --both bench-cuda`:

```bash
./rig.sh fetch-bench
./rig.sh report-cuda
```

Results under `bench/rig/<tag>_windows/` and `bench/rig/<tag>_wsl/`.

## Reproduce

```bash
export TRIBEV2_DATA_DIR="${TRIBEV2_DATA_DIR:-$(pwd)/data}"

# Single backend
cargo run --release -p tribev2 --example bench_encoder \
  --no-default-features --features pure-rust,rlx-encoder,rlx-metal \
  -- --engines rlx-metal --timesteps 50 --warmup 2 --runs 5

# Full local sweep
TIMESTEPS=50 WARMUP=2 RUNS=5 ./bench/run_all_backends.sh
```
