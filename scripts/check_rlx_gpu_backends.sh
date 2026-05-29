#!/usr/bin/env bash
# Check RLX wgpu + CUDA backends (hybrid parity) on Linux / WSL / Git Bash.
# RLX deps resolve from crates.io (see workspace Cargo.toml).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== tribev2 RLX GPU backend check (Linux/WSL / Git Bash) ==="
echo "root: $ROOT"
uname -a || true
echo "deps: crates.io (rlx / rlx-models-*)"

export RUST_BACKTRACE="${RUST_BACKTRACE:-1}"
FEAT_BASE="pure-rust,rlx-encoder,rlx-cpu"
# Encoder-only features avoid pulling rlx-llama32 / qwen35 (text stack).
FEAT_GPU="${FEAT_BASE},rlx-gpu-enc"
FEAT_CUDA="${FEAT_BASE},rlx-cuda-enc"

echo ""
echo "--- build: rlx-gpu-enc (wgpu, encoder only) ---"
cargo build --release -p tribev2 --no-default-features --features "${FEAT_GPU}"

echo ""
echo "--- build: rlx-cuda-enc ---"
cargo build --release -p tribev2 --no-default-features --features "${FEAT_CUDA}" || {
  echo "WARN: rlx-cuda-enc build failed (no CUDA toolkit?)"
}

echo ""
echo "--- device probe ---"
cargo run --release -p tribev2 --no-default-features --features "${FEAT_GPU},rlx-cuda-enc" \
  --example check_rlx_devices 2>/dev/null || \
cargo run --release -p tribev2 --no-default-features --features "${FEAT_GPU}" \
  --example check_rlx_devices

if [[ -f data/parity_refs/input_text.bin && -f data/model.safetensors ]]; then
  export TRIBEV2_DATA_DIR="${TRIBEV2_DATA_DIR:-$ROOT/data}"
  echo ""
  echo "--- parity: CPU (baseline) ---"
  cargo test --release -p tribev2 --no-default-features --features "${FEAT_BASE}" \
    --test rlx_parity -- --nocapture test_rlx_vs_pure_rust

  echo ""
  echo "--- parity: wgpu hybrid ---"
  cargo test --release -p tribev2 --no-default-features --features "${FEAT_GPU}" \
    --test rlx_parity -- --nocapture test_rlx_vs_pure_rust_on_wgpu_device || true

  echo ""
  echo "--- parity: cuda hybrid ---"
  cargo test --release -p tribev2 --no-default-features --features "${FEAT_CUDA}" \
    --test rlx_parity -- --nocapture test_rlx_vs_pure_rust_on_cuda_device || true
else
  echo ""
  echo "SKIP parity tests (no data/ weights). Set TRIBEV2_DATA_DIR or add data/."
fi

echo ""
echo "Done."
