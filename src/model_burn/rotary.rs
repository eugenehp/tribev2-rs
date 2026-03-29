use burn::prelude::*;

/// Compute rotary frequencies for positions [0..seq_len).
/// Returns [seq_len, rot_dim] raw angle values (cat-duplicated).
pub fn build_rotary_freqs<B: Backend>(
    dim: usize,
    seq_len: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let half = dim / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1.0 / 10000.0f64.powf(2.0 * i as f64 / dim as f64) as f32)
        .collect();
    let inv = Tensor::<B, 1>::from_data(TensorData::new(inv_freq, [half]), device);

    let pos: Vec<f32> = (0..seq_len).map(|p| p as f32).collect();
    let pos_t = Tensor::<B, 1>::from_data(TensorData::new(pos, [seq_len]), device);

    // freqs = pos[:, None] * inv[None, :] => [seq_len, half]
    let freqs = pos_t.unsqueeze_dim::<2>(1) * inv.unsqueeze_dim::<2>(0);
    // cat duplicated: [seq_len, dim]
    Tensor::cat(vec![freqs.clone(), freqs], 1)
}

/// Apply rotary position embedding.
/// x: [B, H, N, D], freqs: [N, rot_dim].
/// Rotates first rot_dim dims, passes remaining unchanged.
pub fn apply_rotary<B: Backend>(x: Tensor<B, 4>, freqs: &Tensor<B, 2>) -> Tensor<B, 4> {
    let [b, h, n, d] = x.dims();
    let rot_dim = freqs.dims()[1];
    let half = rot_dim / 2;

    let x_rot = x.clone().slice([0..b, 0..h, 0..n, 0..rot_dim]);
    let x_pass = if rot_dim < d {
        Some(x.clone().slice([0..b, 0..h, 0..n, rot_dim..d]))
    } else {
        None
    };

    // Split rotated part into first/second half
    let x1 = x_rot.clone().slice([0..b, 0..h, 0..n, 0..half]);
    let x2 = x_rot.slice([0..b, 0..h, 0..n, half..rot_dim]);

    // cos/sin from freqs [N, rot_dim]
    let cos_f = freqs.clone().slice([0..n, 0..half]).cos()
        .unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0); // [1, 1, N, half]
    let sin_f = freqs.clone().slice([0..n, 0..half]).sin()
        .unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0);

    // rotate_half: [-x2, x1] pattern
    let r1 = x1.clone() * cos_f.clone() - x2.clone() * sin_f.clone();
    let r2 = x2 * cos_f + x1 * sin_f;
    let rotated = Tensor::cat(vec![r1, r2], 3);

    match x_pass {
        Some(p) => Tensor::cat(vec![rotated, p], 3),
        None => rotated,
    }
}
