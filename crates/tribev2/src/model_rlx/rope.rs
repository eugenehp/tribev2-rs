//! CPU RoPE cos/sin tables for x-transformers-style rotary embeddings.

/// Build flat cos and sin tables for sequence length `seq_len` and rotary dim `rot_dim`.
///
/// Returns `(cos, sin)` each of length `seq_len * (rot_dim / 2)`.
pub fn rope_cos_sin(seq_len: usize, rot_dim: usize) -> (Vec<f32>, Vec<f32>) {
    let half = rot_dim / 2;
    let base: f64 = 10000.0;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| {
            let exp = (2 * i) as f64 / rot_dim as f64;
            (1.0 / base.powf(exp)) as f32
        })
        .collect();

    let mut cos = vec![0.0f32; seq_len * half];
    let mut sin = vec![0.0f32; seq_len * half];
    for pos in 0..seq_len {
        for j in 0..half {
            let angle = pos as f32 * inv_freq[j];
            cos[pos * half + j] = angle.cos();
            sin[pos * half + j] = angle.sin();
        }
    }
    (cos, sin)
}
