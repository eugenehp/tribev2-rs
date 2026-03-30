/// Fused CubeCL GPU kernels for the TRIBE v2 encoder.
///
/// **ScaleNorm** — single-pass L2-normalise + scale using `plane_sum`
///   (Metal SIMD-group all-reduce, plane_dim = 64).
///   Replaces: pow → sum_dim → clamp → pow → mul×2 (≥3 dispatches × 17 calls).
///
/// **RoPE rotation** — one kernel dispatch per Q or K application.
///   Replaces: slice×3 + mul×4 + sub + add + cat (≥5 dispatches + 4 allocs × 16 applies).

use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;
use cubecl::server::Handle;
use burn::backend::wgpu::CubeTensor;
use burn_backend::{DType, Shape};
use burn_cubecl::kernel::into_contiguous;

// ── tensor helpers ────────────────────────────────────────────────────────

fn elem_size(dtype: DType) -> usize {
    match dtype {
        DType::F64 | DType::I64 | DType::U64 => 8,
        DType::F32 | DType::Flex32 | DType::I32 | DType::U32 => 4,
        DType::F16 | DType::BF16  | DType::I16  | DType::U16 => 2,
        DType::I8  | DType::U8   | DType::Bool => 1,
        DType::QFloat(_) => 4,
    }
}

fn contiguous_strides(dims: &[usize]) -> Vec<usize> {
    let n = dims.len();
    let mut s = vec![1usize; n];
    for i in (0..n.saturating_sub(1)).rev() {
        s[i] = s[i + 1] * dims[i + 1];
    }
    s
}

fn empty_cube(src: &CubeTensor<WgpuRuntime>, shape: Shape) -> CubeTensor<WgpuRuntime> {
    let n_bytes  = shape.dims.iter().product::<usize>() * elem_size(src.dtype);
    let handle: Handle = src.client.empty(n_bytes);
    CubeTensor {
        client:  src.client.clone(),
        device:  src.device.clone(),
        handle,
        strides: contiguous_strides(&shape.dims),
        shape,
        dtype:   src.dtype,
        qparams: None,
    }
}

// ── ScaleNorm ─────────────────────────────────────────────────────────────
//
// Grid:   CubeDim { x: 64, y: 1, z: 1 },  CubeCount::Static(rows, 1, 1)
// CUBE_POS_X = which row (u32),  UNIT_POS_X = lane within SIMD-group (u32)
//
// Scalar kernel args must be Rust primitive types (u32/f32/usize), NOT `F: Float`
// (F is a CubeCL IR type — it doesn't exist at the Rust host level).

#[cube(launch)]
fn scalenorm_kernel<F: Float>(
    x:     &Tensor<F>,
    g:     &Tensor<F>,           // [1]
    out:   &mut Tensor<F>,
    d:     u32,                  // last dim size (e.g. 1152)
    scale: f32,                  // sqrt(D), cast to F inside
) {
    let row  = CUBE_POS_X;       // u32
    let lane = UNIT_POS_X;       // u32  (0..PLANE_DIM)

    let n_per_lane = d / PLANE_DIM;
    let base = row * d + lane;

    let mut sq = F::new(0.0);
    for i in 0u32..n_per_lane {
        let v = x[(base + i * PLANE_DIM) as usize];
        sq += v * v;
    }

    let row_sq = plane_sum(sq);
    let factor = g[0usize]
        * F::cast_from(scale)
        * F::powf(F::max(row_sq, F::new(1e-12)), F::new(-0.5));

    for i in 0u32..n_per_lane {
        let idx  = (base + i * PLANE_DIM) as usize;
        out[idx] = x[idx] * factor;
    }
}

pub fn launch_scalenorm(
    x:     CubeTensor<WgpuRuntime>,
    g:     CubeTensor<WgpuRuntime>,
    scale: f32,
) -> CubeTensor<WgpuRuntime> {
    let x   = into_contiguous(x);
    let g   = into_contiguous(g);
    let out = empty_cube(&x, x.shape.clone());

    let d    = *x.shape.dims.last().unwrap() as u32;
    let rows = x.shape.dims[..x.shape.num_dims() - 1].iter().product::<usize>() as u32;

    // One plane (64 threads on Metal) per row
    let cube_dim   = CubeDim { x: 64, y: 1, z: 1 };
    let cube_count = CubeCount::Static(rows, 1, 1);

    match x.dtype {
        DType::F32 | DType::Flex32 => unsafe {
            scalenorm_kernel::launch::<f32, WgpuRuntime>(
                &x.client, cube_count, cube_dim,
                TensorArg::from_raw_parts::<f32>(&x.handle,   &x.strides,   &x.shape.dims, 1),
                TensorArg::from_raw_parts::<f32>(&g.handle,   &g.strides,   &g.shape.dims, 1),
                TensorArg::from_raw_parts::<f32>(&out.handle, &out.strides, &out.shape.dims, 1),
                ScalarArg::new(d),
                ScalarArg::new(scale),
            ).expect("scalenorm f32 launch");
        },
        DType::F16 => unsafe {
            scalenorm_kernel::launch::<half::f16, WgpuRuntime>(
                &x.client, cube_count, cube_dim,
                TensorArg::from_raw_parts::<half::f16>(&x.handle,   &x.strides,   &x.shape.dims, 1),
                TensorArg::from_raw_parts::<half::f16>(&g.handle,   &g.strides,   &g.shape.dims, 1),
                TensorArg::from_raw_parts::<half::f16>(&out.handle, &out.strides, &out.shape.dims, 1),
                ScalarArg::new(d),
                ScalarArg::new(scale),
            ).expect("scalenorm f16 launch");
        },
        dt => panic!("scalenorm: unsupported dtype {dt:?}"),
    }
    out
}

// ── RoPE rotation ─────────────────────────────────────────────────────────
//
// Grid: CubeDim { x: 256, y: 1, z: 1 },  CubeCount::Static(ceil(total/256), 1, 1)
// ABSOLUTE_POS (usize) = global thread index across the whole B×H×N×D tensor.
//
// For usize scalar args (ABSOLUTE_POS is usize, so arithmetic with total/d/half
// uses usize for consistency).

#[cube(launch)]
fn rope_kernel<F: Float>(
    x:     &Tensor<F>,
    cos:   &Tensor<F>,           // [N, half]
    sin:   &Tensor<F>,           // [N, half]
    out:   &mut Tensor<F>,
    d:     usize,                // head dim (144)
    half:  usize,                // D/2      (72)
    n_seq: usize,                // seq len  (100)
    total: usize,                // B×H×N×D
) {
    let pos = ABSOLUTE_POS;
    if pos >= total { terminate!(); }

    let d_idx = pos % d;
    let n_idx = (pos / d) % n_seq;

    let result: F = if d_idx < half {
        let cs = n_idx * half + d_idx;
        x[pos] * cos[cs] - x[pos + half] * sin[cs]
    } else {
        let cs = n_idx * half + d_idx - half;
        x[pos - half] * sin[cs] + x[pos] * cos[cs]
    };
    out[pos] = result;
}

pub fn launch_rope(
    x:   CubeTensor<WgpuRuntime>,
    cos: CubeTensor<WgpuRuntime>,
    sin: CubeTensor<WgpuRuntime>,
) -> CubeTensor<WgpuRuntime> {
    let x   = into_contiguous(x);
    let cos = into_contiguous(cos);
    let sin = into_contiguous(sin);
    let out = empty_cube(&x, x.shape.clone());

    let dims  = &x.shape.dims;
    let rank  = dims.len();
    let d     = dims[rank - 1];
    let half  = d / 2;
    let n_seq = dims[rank - 2];
    let total = dims.iter().product::<usize>();

    let cube_dim_x: u32 = 256;
    let cube_count_x    = ((total as u32) + cube_dim_x - 1) / cube_dim_x;
    let cube_dim        = CubeDim { x: cube_dim_x, y: 1, z: 1 };
    let cube_count      = CubeCount::Static(cube_count_x, 1, 1);

    match x.dtype {
        DType::F32 | DType::Flex32 => unsafe {
            rope_kernel::launch::<f32, WgpuRuntime>(
                &x.client, cube_count, cube_dim,
                TensorArg::from_raw_parts::<f32>(&x.handle,   &x.strides,   &x.shape.dims, 1),
                TensorArg::from_raw_parts::<f32>(&cos.handle, &cos.strides, &cos.shape.dims, 1),
                TensorArg::from_raw_parts::<f32>(&sin.handle, &sin.strides, &sin.shape.dims, 1),
                TensorArg::from_raw_parts::<f32>(&out.handle, &out.strides, &out.shape.dims, 1),
                ScalarArg::new(d),
                ScalarArg::new(half),
                ScalarArg::new(n_seq),
                ScalarArg::new(total),
            ).expect("rope f32 launch");
        },
        DType::F16 => unsafe {
            rope_kernel::launch::<half::f16, WgpuRuntime>(
                &x.client, cube_count, cube_dim,
                TensorArg::from_raw_parts::<half::f16>(&x.handle,   &x.strides,   &x.shape.dims, 1),
                TensorArg::from_raw_parts::<half::f16>(&cos.handle, &cos.strides, &cos.shape.dims, 1),
                TensorArg::from_raw_parts::<half::f16>(&sin.handle, &sin.strides, &sin.shape.dims, 1),
                TensorArg::from_raw_parts::<half::f16>(&out.handle, &out.strides, &out.shape.dims, 1),
                ScalarArg::new(d),
                ScalarArg::new(half),
                ScalarArg::new(n_seq),
                ScalarArg::new(total),
            ).expect("rope f16 launch");
        },
        dt => panic!("rope: unsupported dtype {dt:?}"),
    }
    out
}
