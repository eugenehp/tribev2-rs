use burn::prelude::*;

/// Compute rotary frequencies for positions [0..seq_len).
/// Returns [seq_len, rot_dim] raw angle values (cat-duplicated).
pub fn build_rotary_freqs<B: Backend>(
    dim: usize,
    seq_len: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let half = dim / 2;
    // Precompute all freq values directly: freqs[pos, j] = pos * inv_freq[j % half]
    let mut data = vec![0.0f32; seq_len * dim];
    for pos in 0..seq_len {
        for j in 0..half {
            let inv_freq = 1.0 / 10000.0f64.powf(2.0 * j as f64 / dim as f64) as f32;
            let val = pos as f32 * inv_freq;
            data[pos * dim + j] = val;
            data[pos * dim + half + j] = val; // duplicated
        }
    }
    Tensor::<B, 2>::from_data(TensorData::new(data, [seq_len, dim]), device)
}

/// Apply rotary position embedding.
/// x: [B, H, N, D], freqs: [N, rot_dim].
/// Rotates first rot_dim dims, passes remaining unchanged.
pub fn apply_rotary<B: Backend>(x: Tensor<B, 4>, freqs: &Tensor<B, 2>) -> Tensor<B, 4> {
    let [b, h, n, d] = x.dims();
    let rot_dim = freqs.dims()[1];
    let half = rot_dim / 2;

    // Split x into rotated and pass-through parts
    let x_rot = x.clone().slice([0..b, 0..h, 0..n, 0..rot_dim]);
    let x1 = x_rot.clone().slice([0..b, 0..h, 0..n, 0..half]);
    let x2 = x_rot.slice([0..b, 0..h, 0..n, half..rot_dim]);

    // freqs [N, rot_dim] → slice first half [N, half] → cos/sin → broadcast [1, 1, N, half]
    let f_half = freqs.clone().slice([0..n, 0..half]);
    let cos_f = f_half.clone().cos().unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0);
    let sin_f = f_half.sin().unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0);

    // rotate_half pattern: [-x2, x1]
    let r1 = x1.clone() * cos_f.clone() - x2.clone() * sin_f.clone();
    let r2 = x2 * cos_f + x1 * sin_f;
    let rotated = Tensor::cat(vec![r1, r2], 3);

    if rot_dim < d {
        let x_pass = x.slice([0..b, 0..h, 0..n, rot_dim..d]);
        Tensor::cat(vec![rotated, x_pass], 3)
    } else {
        rotated
    }
}
