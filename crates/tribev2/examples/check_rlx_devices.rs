//! List RLX devices compiled into this binary and whether each is available.
//!
//! ```bash
//! cargo run --release -p tribev2 --no-default-features \
//!   --features "pure-rust,rlx-encoder,rlx-cpu,rlx-gpu,rlx-cuda" \
//!   --example check_rlx_devices
//! ```

fn main() {
    println!("RLX backends in this build:");
    for d in rlx::runtime::available_devices() {
        let label = match d {
            rlx::Device::Cpu => "cpu",
            rlx::Device::Metal => "metal",
            rlx::Device::Mlx => "mlx",
            rlx::Device::Cuda => "cuda",
            rlx::Device::Rocm => "rocm",
            rlx::Device::Gpu => "wgpu",
            rlx::Device::Vulkan => "vulkan",
            rlx::Device::Ane => "ane",
            rlx::Device::Tpu => "tpu",
            _ => "?",
        };
        let ok = rlx::runtime::is_available(d);
        println!("  {label:8} ({d:?})  available={ok}");
    }
    println!("\nLabels for CLI: cpu, wgpu, cuda, rocm, vulkan, metal, mlx");
    println!("TribeRlx runs the full graph natively on the selected RLX device.");
}
