/// `FusedOps` — backend-dispatch trait for custom CubeCL kernels.
///
/// Model layers call `B::rope_rotate(...)` and `B::scalenorm(...)`.
/// - `CubeBackend<WgpuRuntime, f32/f16>` → custom CubeCL kernels.
/// - All other backends (Wgpu fusion, NdArray, …) → standard burn tensor ops.

use burn::prelude::*;
use burn::tensor::TensorPrimitive;

// ── Trait ─────────────────────────────────────────────────────────────────

pub trait FusedOps: Backend {
    /// Fused RoPE rotation: [B,H,N,D] → [B,H,N,D].
    fn rope_rotate(
        x:   Tensor<Self, 4>,
        cos: Tensor<Self, 2>,
        sin: Tensor<Self, 2>,
    ) -> Tensor<Self, 4>;

    /// Fused ScaleNorm: L2-normalise each last-dim vector, then multiply by g·scale.
    fn scalenorm(x: Tensor<Self, 3>, g: Tensor<Self, 1>, scale: f32) -> Tensor<Self, 3>;
}

// ── Standard burn-ops fallback (used by all non-CubeBackend backends) ─────

fn rope_rotate_standard<B: Backend>(
    x: Tensor<B, 4>, cos: Tensor<B, 2>, sin: Tensor<B, 2>,
) -> Tensor<B, 4> {
    let [b, h, n, d] = x.dims();
    let half    = cos.dims()[1];
    let rot_dim = half * 2;

    let x_rot = x.clone().slice([0..b, 0..h, 0..n, 0..rot_dim]);
    let x1    = x_rot.clone().slice([0..b, 0..h, 0..n, 0..half]);
    let x2    = x_rot        .slice([0..b, 0..h, 0..n, half..rot_dim]);
    let c     = cos.clone().unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0);
    let s     = sin.clone().unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0);
    let r1    = x1.clone() * c.clone() - x2.clone() * s.clone();
    let r2    = x2 * c + x1 * s;
    let rotated = Tensor::cat(vec![r1, r2], 3);
    if rot_dim < d {
        Tensor::cat(vec![rotated, x.slice([0..b, 0..h, 0..n, rot_dim..d])], 3)
    } else {
        rotated
    }
}

fn scalenorm_standard<B: Backend>(
    x: Tensor<B, 3>, g: Tensor<B, 1>, scale: f32,
) -> Tensor<B, 3> {
    let norm_sq  = x.clone().powf_scalar(2.0).sum_dim(2).clamp_min(1e-24f32);
    let inv_norm = norm_sq.powf_scalar(-0.5);
    x * inv_norm * g.reshape([1, 1, 1]).mul_scalar(scale)
}

// ── CubeBackend (no-fusion) → custom CubeCL kernels ──────────────────────

#[cfg(feature = "wgpu-kernels-metal")]
mod cube_impls {
    use super::*;
    use burn::backend::wgpu::{CubeBackend, CubeTensor};
    use burn_cubecl::{FloatElement, IntElement, BoolElement};
    use cubecl::wgpu::WgpuRuntime;
    use crate::model_burn::kernels;

    // Helper: extract CubeTensor from a Tensor<CubeBackend<...>, D>
    fn cube<B, const D: usize>(t: Tensor<B, D>) -> CubeTensor<WgpuRuntime>
    where
        B: Backend,
        B::FloatTensorPrimitive: Into<CubeTensor<WgpuRuntime>>,
    {
        match t.into_primitive() {
            TensorPrimitive::Float(p) => p.into(),
            _ => panic!("expected float tensor"),
        }
    }

    // Helper: wrap a CubeTensor back into Tensor<CubeBackend<...>, D>
    fn wrap<B, const D: usize>(c: CubeTensor<WgpuRuntime>) -> Tensor<B, D>
    where
        B: Backend,
        CubeTensor<WgpuRuntime>: Into<B::FloatTensorPrimitive>,
    {
        Tensor::from_primitive(TensorPrimitive::Float(c.into()))
    }

    impl<F, I, BT> FusedOps for CubeBackend<WgpuRuntime, F, I, BT>
    where
        F: FloatElement,
        I: IntElement,
        BT: BoolElement,
        // Bridge: CubeTensor ↔ CubeBackend's primitive type
        <Self as Backend>::FloatTensorPrimitive: Into<CubeTensor<WgpuRuntime>>,
        CubeTensor<WgpuRuntime>: Into<<Self as Backend>::FloatTensorPrimitive>,
    {
        fn rope_rotate(
            x: Tensor<Self, 4>, cos: Tensor<Self, 2>, sin: Tensor<Self, 2>,
        ) -> Tensor<Self, 4> {
            wrap(kernels::launch_rope(cube(x), cube(cos), cube(sin)))
        }

        fn scalenorm(x: Tensor<Self, 3>, g: Tensor<Self, 1>, scale: f32) -> Tensor<Self, 3> {
            wrap(kernels::launch_scalenorm(cube(x), cube(g), scale))
        }
    }
}

// ── Fusion-wrapped Wgpu (default) and NdArray → burn-ops fallback ─────────

macro_rules! impl_standard {
    ($ty:ty) => {
        impl FusedOps for $ty {
            fn rope_rotate(x: Tensor<Self,4>, cos: Tensor<Self,2>, sin: Tensor<Self,2>)
                -> Tensor<Self,4> { rope_rotate_standard(x, cos, sin) }
            fn scalenorm(x: Tensor<Self,3>, g: Tensor<Self,1>, scale: f32)
                -> Tensor<Self,3> { scalenorm_standard(x, g, scale) }
        }
    };
}

#[cfg(feature = "wgpu")]
impl_standard!(burn::backend::Wgpu<f32, i32>);
#[cfg(feature = "wgpu")]
impl_standard!(burn::backend::Wgpu<half::f16, i32>);
#[cfg(feature = "ndarray")]
impl_standard!(burn::backend::NdArray<f32>);
