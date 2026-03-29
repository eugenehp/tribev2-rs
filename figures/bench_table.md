| Backend | Mean (ms) | Min (ms) | Std (ms) | vs Py CPU 1t |
|---------|-----------|----------|----------|--------------|
| Python CPU 1-thread | 56.6 | 55.9 | 0.6 | 1.0× |
| Python MPS (GPU) | 11.9 | 11.6 | 0.1 | 4.8× |
| Rust manual Accelerate | 75.2 | 74.8 | 0.4 | 0.8× |
| Rust Burn NdArray | 1091.5 | 1086.4 | 4.5 | 0.1× |
| Rust Burn NdArray+Accel | 938.5 | 928.2 | 5.5 | 0.1× |
| Rust manual CPU (naive) | 13655.3 | 13501.2 | 104.5 | 0.0× |
| Python CPU 1t (2048) | 50.1 | 44.0 | 6.3 | 1.1× |
| Python MPS (2048) | 10.0 | 9.8 | 0.2 | 5.6× |
| Rust Burn wgpu Metal (2048) | 24.7 | 18.9 | 7.5 | 2.3× |
