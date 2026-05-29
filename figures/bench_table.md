# TRIBE v2 encoder benchmarks

Canonical tables live in **[bench/README.md](../bench/README.md)** with raw JSON under **`bench/results_*.json`** and **`bench/summary.json`**.

## T=100 historical (forward only)

**Setup:** batch=1, T=100, 3 modalities, 20,484 vertices (3-layer feature layout in older runs).

| Backend | Mean (ms) | vs Rust naive |
|---------|----------:|--------------:|
| Rust CPU (naive loops) | 14,516.5 | 1× |
| Burn NdArray (Rayon) | 316.2 | 46× |
| Burn NdArray + Accelerate | 142.7 | 102× |
| Rust CPU + Accelerate BLAS | 73.1 | 199× |
| Burn wgpu Metal f32 | 22.6 | 642× |
| Burn wgpu Metal f16 | 20.5 | 708× |
| **Burn wgpu Metal + fused CubeCL** | **16.8** | **864×** |

## T=10 sweep (May 2026, checkpoint `build_args.json`)

| Backend | Mean (ms) |
|---------|----------:|
| pure-Rust CPU | 1333.5 |
| RLX CPU | 33.8 |
| Burn NdArray + Accelerate | 66.7 |
| RLX Metal | 15.1 |
| Burn wgpu Metal f32 | 13.6 |
| **Burn wgpu Metal f16** | **10.6** |
