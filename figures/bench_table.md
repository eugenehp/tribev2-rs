| Backend | Mean (ms) | Min (ms) | Std (ms) | Speedup |
|---------|-----------|----------|----------|---------|
| Rust CPU (naive loops) | 14516.5 | 14350.2 | 278.4 | 1× |
| Burn NdArray (Rayon) | 316.2 | 289.2 | 36.3 | 46× |
| Burn NdArray + Accelerate | 142.7 | 134.5 | 8.6 | 102× |
| Rust CPU (Accelerate BLAS) | 73.1 | 71.9 | 0.9 | 199× |
| Python CPU (1 thread) | 57.0 | 55.5 | 0.9 | 255× |
| Burn wgpu (Metal GPU) | 20.3 | 19.6 | 0.4 | 715× |
| Python MPS (GPU) | 11.6 | 11.3 | 0.1 | 1254× |
