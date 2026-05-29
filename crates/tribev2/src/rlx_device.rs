//! RLX device selection for the TRIBE encoder and text-feature extractors.

use anyhow::{bail, Result};

/// Parse a device label (`cpu`, `metal`, `mps`, `mlx`, `cuda`, `rocm`, `wgpu`, …).
#[cfg(feature = "rlx-encoder")]
pub fn parse_rlx_device(s: &str) -> Result<rlx::Device> {
    Ok(match s.trim().to_ascii_lowercase().as_str() {
        "cpu" => rlx::Device::Cpu,
        "metal" | "mps" => rlx::Device::Metal,
        "mlx" => rlx::Device::Mlx,
        "ane" => rlx::Device::Ane,
        "cuda" => rlx::Device::Cuda,
        "rocm" | "hip" => rlx::Device::Rocm,
        "gpu" | "wgpu" => rlx::Device::Gpu,
        "vulkan" => rlx::Device::Vulkan,
        "tpu" => rlx::Device::Tpu,
        other => bail!(
            "unknown RLX device '{other}' (try: cpu, metal, mps, mlx, cuda, rocm, wgpu, vulkan)"
        ),
    })
}

#[cfg(feature = "rlx-encoder")]
pub fn default_rlx_device_label() -> &'static str {
    // CPU is the numerically verified default; pick GPU explicitly via CLI
    // (`--backend rlx-metal`, `rlx-mlx`, etc.).
    "cpu"
}

/// Whether this device was compiled in and registered by RLX.
#[cfg(feature = "rlx-encoder")]
pub fn rlx_device_available(device: rlx::Device) -> bool {
    rlx::runtime::is_available(device)
}

/// Parse and verify the device is available in this build.
#[cfg(feature = "rlx-encoder")]
pub fn parse_rlx_device_available(s: &str) -> Result<rlx::Device> {
    let device = parse_rlx_device(s)?;
    if !rlx_device_available(device) {
        bail!(
            "RLX device '{}' ({}) is not enabled in this build. Rebuild with the matching \
             `rlx-*` Cargo feature (e.g. rlx-metal, rlx-cuda). Enabled: {:?}",
            s,
            device,
            rlx::runtime::available_devices()
        );
    }
    Ok(device)
}

/// List device labels that are compiled into this binary.
#[cfg(feature = "rlx-encoder")]
pub fn available_rlx_device_labels() -> Vec<&'static str> {
    let mut out = Vec::new();
    for d in rlx::runtime::available_devices() {
        let label = match d {
            rlx::Device::Cpu => "cpu",
            rlx::Device::Metal => "metal",
            rlx::Device::Mlx => "mlx",
            rlx::Device::Ane => "ane",
            rlx::Device::Cuda => "cuda",
            rlx::Device::Rocm => "rocm",
            rlx::Device::Gpu => "wgpu",
            rlx::Device::Vulkan => "vulkan",
            rlx::Device::Tpu => "tpu",
            _ => continue,
        };
        if !out.contains(&label) {
            out.push(label);
        }
    }
    if out.is_empty() {
        out.push("cpu");
    }
    out
}

#[cfg(feature = "rlx-encoder")]
pub fn check_rlx_device(s: &str) -> Result<()> {
    parse_rlx_device_available(s).map(|_| ())
}

#[cfg(not(feature = "rlx-encoder"))]
pub fn parse_rlx_device(_s: &str) -> Result<()> {
    bail!("RLX support disabled; rebuild with `--features rlx-encoder`")
}
