# Check RLX wgpu + CUDA backends on native Windows (PowerShell).
# RLX deps resolve from crates.io (see workspace Cargo.toml).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

Write-Host "=== tribev2 RLX GPU backend check (Windows) ==="
Write-Host "root: $Root"
Write-Host "deps: crates.io (rlx / rlx-models-*)"

$env:RUST_BACKTRACE = "1"
$FeatBase = "pure-rust,rlx-encoder,rlx-cpu"
$FeatGpu = "$FeatBase,rlx-gpu-enc"
$FeatCuda = "$FeatBase,rlx-cuda-enc"

Write-Host "`n--- build: rlx-gpu-enc (wgpu) ---"
cargo build --release -p tribev2 --no-default-features --features $FeatGpu

Write-Host "`n--- build: rlx-cuda-enc ---"
try {
    cargo build --release -p tribev2 --no-default-features --features $FeatCuda
} catch {
    Write-Warning "rlx-cuda-enc build failed (CUDA toolkit / drivers?)"
}

Write-Host "`n--- device probe ---"
cargo run --release -p tribev2 --no-default-features --features "$FeatGpu,rlx-cuda-enc" `
    --example check_rlx_devices

$hasData = (Test-Path "data\parity_refs\input_text.bin") -and (Test-Path "data\model.safetensors")
if ($hasData) {
    if (-not $env:TRIBEV2_DATA_DIR) { $env:TRIBEV2_DATA_DIR = Join-Path $Root "data" }

    Write-Host "`n--- parity: CPU (baseline) ---"
    cargo test --release -p tribev2 --no-default-features --features $FeatBase `
        --test rlx_parity -- --nocapture test_rlx_vs_pure_rust

    Write-Host "`n--- parity: wgpu hybrid ---"
    cargo test --release -p tribev2 --no-default-features --features $FeatGpu `
        --test rlx_parity -- --nocapture test_rlx_vs_pure_rust_on_wgpu_device

    Write-Host "`n--- parity: cuda hybrid ---"
    cargo test --release -p tribev2 --no-default-features --features $FeatCuda `
        --test rlx_parity -- --nocapture test_rlx_vs_pure_rust_on_cuda_device
} else {
    Write-Host "`nSKIP parity tests (no data/ weights)."
}

Write-Host "`nDone."
