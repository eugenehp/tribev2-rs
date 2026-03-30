# TRIBE v2 Forward Pass Benchmark

**Setup:** batch=1, T=100, 3 modalities (text/audio/video), 20 484 cortical vertices

| Backend | Mean (ms) | Min (ms) | Std (ms) | vs CPU naive |
|---------|----------:|---------:|---------:|-------------:|
| Rust CPU (naive loops) | 14516.5 | 14350.2 | 278.4 | 1× |
| Burn NdArray (Rayon) | 316.2 | 289.2 | 36.3 | 46× |
| Burn NdArray + Accelerate | 142.7 | 134.5 | 8.6 | 102× |
| Rust CPU + Accelerate BLAS | 73.1 | 71.9 | 0.9 | 199× |
| Python CPU (1 thread) | 57.6 | 56.0 | 0.9 | 252× |
| Burn wgpu Metal f32 | 22.6 | 21.0 | 1.9 | 642× |
| Burn wgpu Metal f16 | 20.5 | 19.1 | 1.4 | 708× |
| Burn wgpu Metal f32 + fused kernels | 16.8 | 15.8 | 1.1 | 864× |
| Python MPS GPU | 12.2 | 11.6 | 0.6 | 1192× |
