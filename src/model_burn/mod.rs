pub mod scalenorm;
pub mod rotary;
pub mod attention;
pub mod feedforward;
pub mod residual;
pub mod encoder;
pub mod projector;
pub mod subject_layers;
pub mod tribe;

#[cfg(feature = "wgpu-kernels-metal")]
pub mod kernels;
#[cfg(feature = "wgpu-kernels-metal")]
pub mod fused;

#[cfg(feature = "wgpu-kernels-metal")]
pub use fused::FusedOps;
