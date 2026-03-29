//! Lightweight tensor abstraction for TRIBE v2 inference.
//!
//! This is a minimal f32 tensor with shape metadata, supporting the operations
//! needed for the FmriEncoder forward pass: matmul, add, GELU, ScaleNorm,
//! softmax, RoPE, adaptive average pooling, and depthwise Conv1d.
//!
//! All operations are pure-CPU, single-threaded for correctness-first design.
//! Layout is always row-major / C-contiguous.

use std::fmt;

// ── Accelerate BLAS bindings ──────────────────────────────────────────────

#[cfg(feature = "accelerate")]
extern "C" {
    fn cblas_sgemm(
        order: i32, transa: i32, transb: i32,
        m: i32, n: i32, k: i32,
        alpha: f32,
        a: *const f32, lda: i32,
        b: *const f32, ldb: i32,
        beta: f32,
        c: *mut f32, ldc: i32,
    );
}

#[cfg(feature = "accelerate")]
fn sgemm_accelerate(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    unsafe {
        cblas_sgemm(101, 111, 111, m as i32, n as i32, k as i32,
                    1.0, a.as_ptr(), k as i32, b.as_ptr(), n as i32,
                    0.0, c.as_mut_ptr(), n as i32);
    }
}

#[cfg(feature = "accelerate")]
fn sgemm_accelerate_transA(m: usize, n: usize, k: usize, a: &[f32], lda: usize, b: &[f32], ldb: usize, c: &mut [f32], ldc: usize) {
    unsafe {
        cblas_sgemm(101, 112, 111, m as i32, n as i32, k as i32,
                    1.0, a.as_ptr(), lda as i32, b.as_ptr(), ldb as i32,
                    0.0, c.as_mut_ptr(), ldc as i32);
    }
}

/// Approximate erf using Abramowitz and Stegun formula (max error ~1.5e-7).
fn erf_f32(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t * (0.254829592
        + t * (-0.284496736
        + t * (1.421413741
        + t * (-1.453152027
        + t * 1.061405429))));
    sign * (1.0 - poly * (-x * x).exp())
}

/// A dense f32 tensor in row-major (C-contiguous) order.
#[derive(Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor({:?}, len={})", self.shape, self.data.len())
    }
}

impl Tensor {
    // ── Constructors ──────────────────────────────────────────────────

    pub fn zeros(shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        Self { data: vec![0.0; n], shape: shape.to_vec() }
    }

    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        assert_eq!(data.len(), n, "data len {} != shape product {}", data.len(), n);
        Self { data, shape }
    }

    pub fn numel(&self) -> usize { self.data.len() }

    pub fn ndim(&self) -> usize { self.shape.len() }

    /// Compute strides for row-major layout.
    pub fn strides(&self) -> Vec<usize> {
        let mut s = vec![1usize; self.shape.len()];
        for i in (0..self.shape.len().saturating_sub(1)).rev() {
            s[i] = s[i + 1] * self.shape[i + 1];
        }
        s
    }

    // ── Reshaping ─────────────────────────────────────────────────────

    pub fn reshape(&self, new_shape: &[usize]) -> Self {
        let n: usize = new_shape.iter().product();
        assert_eq!(n, self.numel(), "reshape: {} -> {:?}", self.numel(), new_shape);
        Self { data: self.data.clone(), shape: new_shape.to_vec() }
    }

    pub fn view(&self, new_shape: &[usize]) -> Self {
        self.reshape(new_shape)
    }

    /// Transpose last two dims: (..., M, N) → (..., N, M)
    pub fn transpose_last2(&self) -> Self {
        assert!(self.ndim() >= 2);
        let nd = self.ndim();
        let m = self.shape[nd - 2];
        let n = self.shape[nd - 1];
        let batch: usize = self.shape[..nd - 2].iter().product();
        let mut out = vec![0.0f32; self.numel()];
        for b in 0..batch {
            let base = b * m * n;
            for i in 0..m {
                for j in 0..n {
                    out[base + j * m + i] = self.data[base + i * n + j];
                }
            }
        }
        let mut new_shape = self.shape.clone();
        new_shape[nd - 2] = n;
        new_shape[nd - 1] = m;
        Self { data: out, shape: new_shape }
    }

    /// Permute dimensions: general case.
    pub fn permute(&self, axes: &[usize]) -> Self {
        let nd = self.ndim();
        assert_eq!(axes.len(), nd);
        let old_strides = self.strides();
        let mut new_shape = vec![0usize; nd];
        for (i, &ax) in axes.iter().enumerate() {
            new_shape[i] = self.shape[ax];
        }
        let n = self.numel();
        let mut out = vec![0.0f32; n];
        let new_strides = {
            let mut s = vec![1usize; nd];
            for i in (0..nd.saturating_sub(1)).rev() {
                s[i] = s[i + 1] * new_shape[i + 1];
            }
            s
        };
        for idx in 0..n {
            // Compute multi-index in new layout
            let mut rem = idx;
            let mut old_flat = 0usize;
            for i in 0..nd {
                let coord = rem / new_strides[i];
                rem %= new_strides[i];
                old_flat += coord * old_strides[axes[i]];
            }
            out[idx] = self.data[old_flat];
        }
        Self { data: out, shape: new_shape }
    }

    // ── Slicing ───────────────────────────────────────────────────────

    /// Slice along dim 0: [start..end, ...]
    pub fn slice_dim0(&self, start: usize, end: usize) -> Self {
        assert!(end <= self.shape[0] && start <= end);
        let inner: usize = self.shape[1..].iter().product();
        let data = self.data[start * inner..end * inner].to_vec();
        let mut shape = self.shape.clone();
        shape[0] = end - start;
        Self { data, shape }
    }

    /// Slice along the last dimension: [..., start..end]
    pub fn slice_last_dim(&self, start: usize, end: usize) -> Self {
        let nd = self.ndim();
        let n_last = self.shape[nd - 1];
        assert!(end <= n_last && start <= end);
        let new_last = end - start;
        let batch: usize = self.shape[..nd - 1].iter().product();
        let mut data = Vec::with_capacity(batch * new_last);
        for b in 0..batch {
            let base = b * n_last;
            data.extend_from_slice(&self.data[base + start..base + end]);
        }
        let mut shape = self.shape.clone();
        shape[nd - 1] = new_last;
        Self { data, shape }
    }

    /// Slice along dim 1: [:, start..end, ...]
    pub fn slice_dim1(&self, start: usize, end: usize) -> Self {
        assert!(self.ndim() >= 2);
        let d0 = self.shape[0];
        let d1 = self.shape[1];
        let inner: usize = self.shape[2..].iter().product();
        assert!(end <= d1 && start <= end);
        let new_d1 = end - start;
        let mut data = Vec::with_capacity(d0 * new_d1 * inner);
        for i in 0..d0 {
            let base = i * d1 * inner;
            data.extend_from_slice(&self.data[base + start * inner..base + end * inner]);
        }
        let mut shape = self.shape.clone();
        shape[1] = new_d1;
        Self { data, shape }
    }

    // ── Element-wise ops ──────────────────────────────────────────────

    pub fn add(&self, other: &Tensor) -> Self {
        assert_eq!(self.shape, other.shape, "add shape mismatch: {:?} vs {:?}", self.shape, other.shape);
        let data: Vec<f32> = self.data.iter().zip(&other.data).map(|(a, b)| a + b).collect();
        Self { data, shape: self.shape.clone() }
    }

    /// Broadcast add: self [..., D] + other [D] → [..., D]
    pub fn add_bias(&self, bias: &Tensor) -> Self {
        assert_eq!(bias.ndim(), 1);
        let d = *self.shape.last().unwrap();
        assert_eq!(bias.shape[0], d);
        let mut data = self.data.clone();
        for chunk in data.chunks_exact_mut(d) {
            for (v, b) in chunk.iter_mut().zip(&bias.data) {
                *v += b;
            }
        }
        Self { data, shape: self.shape.clone() }
    }

    /// Broadcast add: self [B, D, T] + other [1, D, 1] → [B, D, T]
    pub fn add_broadcast_d(&self, other: &Tensor) -> Self {
        assert_eq!(self.ndim(), 3);
        // other can be [1, D, 1] or [D]
        let (b, d, t) = (self.shape[0], self.shape[1], self.shape[2]);
        let bias_vals: &[f32] = if other.ndim() == 3 {
            assert_eq!(other.shape, vec![1, d, 1]);
            &other.data
        } else if other.ndim() == 1 {
            assert_eq!(other.shape[0], d);
            &other.data
        } else {
            panic!("unexpected bias shape: {:?}", other.shape);
        };
        let mut data = self.data.clone();
        for bi in 0..b {
            for di in 0..d {
                for ti in 0..t {
                    data[bi * d * t + di * t + ti] += bias_vals[di];
                }
            }
        }
        Self { data, shape: self.shape.clone() }
    }

    pub fn mul_scalar(&self, s: f32) -> Self {
        let data: Vec<f32> = self.data.iter().map(|&v| v * s).collect();
        Self { data, shape: self.shape.clone() }
    }

    pub fn mul_elementwise(&self, other: &Tensor) -> Self {
        assert_eq!(self.shape, other.shape);
        let data: Vec<f32> = self.data.iter().zip(&other.data).map(|(a, b)| a * b).collect();
        Self { data, shape: self.shape.clone() }
    }

    // ── Activations ───────────────────────────────────────────────────

    /// GELU activation (exact formula: x * 0.5 * (1 + erf(x / sqrt(2))))
    pub fn gelu(&self) -> Self {
        let data: Vec<f32> = self.data.iter().map(|&x| {
            x * 0.5 * (1.0 + erf_f32(x / std::f32::consts::SQRT_2))
        }).collect();
        Self { data, shape: self.shape.clone() }
    }

    // ── Normalization ─────────────────────────────────────────────────

    /// ScaleNorm: normalize(x, dim=-1) * scale * g
    /// where scale = dim ** 0.5, g is a learned scalar.
    /// Python: F.normalize(x, dim=-1) * self.scale * self.g
    pub fn scale_norm(&self, g: f32, dim: usize) -> Self {
        let scale = (dim as f32).sqrt();
        let d = *self.shape.last().unwrap();
        let batch: usize = self.shape[..self.ndim() - 1].iter().product();
        let mut data = vec![0.0f32; self.numel()];
        for b in 0..batch {
            let base = b * d;
            let slice = &self.data[base..base + d];
            let norm: f32 = slice.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            for i in 0..d {
                data[base + i] = (slice[i] / norm) * scale * g;
            }
        }
        Self { data, shape: self.shape.clone() }
    }

    /// Layer norm over last dimension (with weight and bias).
    pub fn layer_norm(&self, weight: &Tensor, bias: &Tensor, eps: f64) -> Self {
        let d = *self.shape.last().unwrap();
        assert_eq!(weight.shape[0], d);
        assert_eq!(bias.shape[0], d);
        let batch: usize = self.shape[..self.ndim() - 1].iter().product();
        let mut data = vec![0.0f32; self.numel()];
        for b in 0..batch {
            let base = b * d;
            let slice = &self.data[base..base + d];
            let mean: f32 = slice.iter().sum::<f32>() / d as f32;
            let var: f32 = slice.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / d as f32;
            let inv_std = 1.0 / (var + eps as f32).sqrt();
            for i in 0..d {
                data[base + i] = (slice[i] - mean) * inv_std * weight.data[i] + bias.data[i];
            }
        }
        Self { data, shape: self.shape.clone() }
    }

    // ── Matrix multiply ───────────────────────────────────────────────

    /// Batched matmul: (..., M, K) × (..., K, N) → (..., M, N)
    pub fn matmul(&self, other: &Tensor) -> Self {
        assert!(self.ndim() >= 2 && other.ndim() >= 2);
        let a_nd = self.ndim();
        let b_nd = other.ndim();
        let m = self.shape[a_nd - 2];
        let k = self.shape[a_nd - 1];
        let k2 = other.shape[b_nd - 2];
        let n = other.shape[b_nd - 1];
        assert_eq!(k, k2, "matmul K mismatch: {} vs {}", k, k2);

        let batch_a: usize = self.shape[..a_nd - 2].iter().product();
        let batch_b: usize = other.shape[..b_nd - 2].iter().product();
        let batch = batch_a.max(batch_b);

        let mut out = vec![0.0f32; batch * m * n];

        for b in 0..batch {
            let ba = if batch_a == 1 { 0 } else { b };
            let bb = if batch_b == 1 { 0 } else { b };
            let a_off = ba * m * k;
            let b_off = bb * k * n;
            let o_off = b * m * n;

            #[cfg(feature = "accelerate")]
            {
                sgemm_accelerate(
                    m, n, k,
                    &self.data[a_off..],
                    &other.data[b_off..],
                    &mut out[o_off..],
                );
            }

            #[cfg(not(feature = "accelerate"))]
            {
                for i in 0..m {
                    for j in 0..n {
                        let mut sum = 0.0f32;
                        for p in 0..k {
                            sum += self.data[a_off + i * k + p] * other.data[b_off + p * n + j];
                        }
                        out[o_off + i * n + j] = sum;
                    }
                }
            }
        }
        // Output shape: broadcast batch dims + [M, N]
        let mut out_shape: Vec<usize> = if a_nd > b_nd {
            self.shape[..a_nd - 2].to_vec()
        } else {
            other.shape[..b_nd - 2].to_vec()
        };
        if out_shape.is_empty() || out_shape.iter().product::<usize>() != batch {
            if batch > 1 {
                out_shape = vec![batch];
            } else {
                out_shape = vec![];
            }
        }
        out_shape.push(m);
        out_shape.push(n);
        Self { data: out, shape: out_shape }
    }

    /// Einstein summation: "bct,cd->bdt" — batched linear projection.
    /// self: [B, C, T], weight: [C, D] → [B, D, T]
    ///
    /// Equivalent to: for each b: out[b] = weight^T @ x[b]
    /// i.e. [D, C] @ [C, T] = [D, T]
    pub fn einsum_bct_cd_bdt(&self, weight: &Tensor) -> Self {
        assert_eq!(self.ndim(), 3);
        assert_eq!(weight.ndim(), 2);
        let (b, c, t) = (self.shape[0], self.shape[1], self.shape[2]);
        let (c2, d) = (weight.shape[0], weight.shape[1]);
        assert_eq!(c, c2);
        let mut out = vec![0.0f32; b * d * t];

        #[cfg(feature = "accelerate")]
        {
            for bi in 0..b {
                let x_off = bi * c * t;
                let o_off = bi * d * t;
                sgemm_accelerate_transA(
                    d, t, c,
                    &weight.data, d,
                    &self.data[x_off..], t,
                    &mut out[o_off..], t,
                );
            }
        }

        #[cfg(not(feature = "accelerate"))]
        {
            for bi in 0..b {
                for di in 0..d {
                    for ti in 0..t {
                        let mut sum = 0.0f32;
                        for ci in 0..c {
                            sum += self.data[bi * c * t + ci * t + ti] * weight.data[ci * d + di];
                        }
                        out[bi * d * t + di * t + ti] = sum;
                    }
                }
            }
        }

        Self { data: out, shape: vec![b, d, t] }
    }

    // ── Softmax ───────────────────────────────────────────────────────

    /// Softmax over last dimension.
    pub fn softmax_last(&self) -> Self {
        let d = *self.shape.last().unwrap();
        let batch: usize = self.shape[..self.ndim() - 1].iter().product();
        let mut data = self.data.clone();
        for b in 0..batch {
            let base = b * d;
            let slice = &mut data[base..base + d];
            let max_val = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for v in slice.iter_mut() {
                *v = (*v - max_val).exp();
                sum += *v;
            }
            for v in slice.iter_mut() {
                *v /= sum;
            }
        }
        Self { data, shape: self.shape.clone() }
    }

    // ── Concatenation ─────────────────────────────────────────────────

    /// Concatenate along the last dimension.
    pub fn cat_last(tensors: &[&Tensor]) -> Self {
        assert!(!tensors.is_empty());
        let nd = tensors[0].ndim();
        let batch: usize = tensors[0].shape[..nd - 1].iter().product();
        for t in tensors {
            assert_eq!(t.shape[..nd - 1], tensors[0].shape[..nd - 1]);
        }
        let total_last: usize = tensors.iter().map(|t| *t.shape.last().unwrap()).sum();
        let mut data = Vec::with_capacity(batch * total_last);
        for b in 0..batch {
            for t in tensors {
                let d = *t.shape.last().unwrap();
                let base = b * d;
                data.extend_from_slice(&t.data[base..base + d]);
            }
        }
        let mut shape = tensors[0].shape.clone();
        *shape.last_mut().unwrap() = total_last;
        Self { data, shape }
    }

    /// Concatenate along dim 1: [B, D1, ...] + [B, D2, ...] → [B, D1+D2, ...]
    pub fn cat_dim1(tensors: &[&Tensor]) -> Self {
        assert!(!tensors.is_empty());
        assert!(tensors[0].ndim() >= 2);
        let b = tensors[0].shape[0];
        for t in tensors {
            assert_eq!(t.shape[0], b);
        }
        let inner: usize = tensors[0].shape[2..].iter().product();
        for t in tensors {
            assert_eq!(t.shape[2..].iter().product::<usize>(), inner);
        }
        let total_d1: usize = tensors.iter().map(|t| t.shape[1]).sum();
        let mut data = Vec::with_capacity(b * total_d1 * inner);
        for bi in 0..b {
            for t in tensors {
                let d1 = t.shape[1];
                let base = bi * d1 * inner;
                data.extend_from_slice(&t.data[base..base + d1 * inner]);
            }
        }
        let mut shape = tensors[0].shape.clone();
        shape[1] = total_d1;
        Self { data, shape }
    }

    // ── Pooling ───────────────────────────────────────────────────────

    /// Adaptive average pool over the last dimension.
    /// Input: [..., T_in], output: [..., T_out].
    /// Matches PyTorch `nn.AdaptiveAvgPool1d(T_out)`.
    pub fn adaptive_avg_pool1d(&self, t_out: usize) -> Self {
        let t_in = *self.shape.last().unwrap();
        let batch: usize = self.shape[..self.ndim() - 1].iter().product();
        let mut data = Vec::with_capacity(batch * t_out);
        for b in 0..batch {
            let base = b * t_in;
            for i in 0..t_out {
                let start = (i * t_in) / t_out;
                let end = ((i + 1) * t_in) / t_out;
                let len = (end - start) as f32;
                let sum: f32 = self.data[base + start..base + end].iter().sum();
                data.push(sum / len);
            }
        }
        let mut shape = self.shape.clone();
        *shape.last_mut().unwrap() = t_out;
        Self { data, shape }
    }

    // ── Rotary Position Embedding ─────────────────────────────────────

    /// Apply rotary position embedding (partial rotation).
    /// x: [B, H, N, D], freqs: [N, rot_dim*2] (cos/sin concatenated)
    /// Only the first `rot_dim` dims of each head are rotated.
    pub fn apply_rotary_pos_emb(&self, freqs: &Tensor) -> Self {
        assert_eq!(self.ndim(), 4); // [B, H, N, D]
        let (b, h, n, d) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        let rot_dim = freqs.shape[1]; // freqs: [N, rot_dim] (already cos/sin concatenated)
        assert!(rot_dim % 2 == 0);
        let half_rot = rot_dim / 2;
        assert!(rot_dim <= d, "rot_dim {} > d {}", rot_dim, d);

        // freqs: [N, rot_dim] where first half = cos portion, second half = sin portion
        // Actually in x_transformers: freqs = cat((freqs, freqs), dim=-1) so it's [N, rot_dim]
        // and apply_rotary_pos_emb does: t[..., :rot_dim] * cos + rotate_half(t[..., :rot_dim]) * sin
        let mut data = self.data.clone();
        for bi in 0..b {
            for hi in 0..h {
                for ni in 0..n {
                    let base = bi * h * n * d + hi * n * d + ni * d;
                    // Get freq values for this position
                    let freq_base = ni * rot_dim;
                    // cos and sin are both [half_rot] each, concatenated
                    for i in 0..half_rot {
                        let cos_val = freqs.data[freq_base + i].cos();
                        let sin_val = freqs.data[freq_base + i].sin();
                        let x0 = self.data[base + i];
                        let x1 = self.data[base + i + half_rot];
                        // rotate_half: (-x1, x0)
                        data[base + i] = x0 * cos_val - x1 * sin_val;
                        data[base + i + half_rot] = x1 * cos_val + x0 * sin_val;
                    }
                    // Remaining dims (> rot_dim) stay unchanged
                }
            }
        }
        Self { data, shape: self.shape.clone() }
    }

    // ── Depthwise Conv1d ──────────────────────────────────────────────

    /// Depthwise Conv1d with groups=dim.
    /// Input: [B, D, T], kernel: [D, 1, K], padding: K//2 → output: [B, D, T]
    pub fn depthwise_conv1d(&self, kernel: &Tensor) -> Self {
        assert_eq!(self.ndim(), 3); // [B, D, T]
        assert_eq!(kernel.ndim(), 3); // [D, 1, K]
        let (b, d, t) = (self.shape[0], self.shape[1], self.shape[2]);
        let k = kernel.shape[2];
        let pad = k / 2;
        let mut data = vec![0.0f32; b * d * t];
        for bi in 0..b {
            for di in 0..d {
                for ti in 0..t {
                    let mut sum = 0.0f32;
                    for ki in 0..k {
                        let src_t = ti as isize + ki as isize - pad as isize;
                        if src_t >= 0 && (src_t as usize) < t {
                            sum += self.data[bi * d * t + di * t + src_t as usize]
                                * kernel.data[di * k + ki];
                        }
                    }
                    data[bi * d * t + di * t + ti] = sum;
                }
            }
        }
        Self { data, shape: self.shape.clone() }
    }

    // ── Reduction ─────────────────────────────────────────────────────

    /// Sum along dim 1 of a 3D tensor: [B, N, D] → [B, 1, D] squeezed to [B, D].
    /// Actually for aggregation we need element-wise sum of same-shape tensors.
    pub fn sum_tensors(tensors: &[&Tensor]) -> Self {
        assert!(!tensors.is_empty());
        let mut out = tensors[0].clone();
        for t in &tensors[1..] {
            assert_eq!(out.shape, t.shape);
            for (a, b) in out.data.iter_mut().zip(&t.data) {
                *a += b;
            }
        }
        out
    }

    /// Multiply by a broadcast vector over dim 1: self[B, D, T] * scale[D] → [B, D, T]
    pub fn mul_broadcast_dim1(&self, scale: &Tensor) -> Self {
        assert_eq!(self.ndim(), 3);
        assert_eq!(scale.ndim(), 1);
        let (b, d, t) = (self.shape[0], self.shape[1], self.shape[2]);
        assert_eq!(scale.shape[0], d);
        let mut data = self.data.clone();
        for bi in 0..b {
            for di in 0..d {
                let s = scale.data[di];
                for ti in 0..t {
                    data[bi * d * t + di * t + ti] *= s;
                }
            }
        }
        Self { data, shape: self.shape.clone() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_2d() {
        // [2, 3] x [3, 2] = [2, 2]
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        let c = a.matmul(&b);
        assert_eq!(c.shape, vec![2, 2]);
        assert_eq!(c.data, vec![22.0, 28.0, 49.0, 64.0]);
    }

    #[test]
    fn test_softmax() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let s = t.softmax_last();
        let sum: f32 = s.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_scale_norm() {
        let t = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let out = t.scale_norm(1.0, 2);
        // normalize: [3/5, 4/5], scale=sqrt(2), g=1.0
        let expected_0 = (3.0 / 5.0) * (2.0f32).sqrt();
        assert!((out.data[0] - expected_0).abs() < 1e-5);
    }

    #[test]
    fn test_adaptive_avg_pool1d() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 6]);
        let p = t.adaptive_avg_pool1d(3);
        assert_eq!(p.shape, vec![1, 3]);
        // bins: [0,2), [2,4), [4,6) → means: 1.5, 3.5, 5.5
        assert_eq!(p.data, vec![1.5, 3.5, 5.5]);
    }

    #[test]
    fn test_transpose_last2() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let tr = t.transpose_last2();
        assert_eq!(tr.shape, vec![3, 2]);
        assert_eq!(tr.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_gelu() {
        let t = Tensor::from_vec(vec![0.0, 1.0, -1.0], vec![3]);
        let g = t.gelu();
        assert!((g.data[0] - 0.0).abs() < 1e-5);
        assert!((g.data[1] - 0.8413).abs() < 1e-3);
    }
}
