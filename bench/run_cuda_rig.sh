#!/usr/bin/env bash
# CUDA-focused encoder benchmark for remote rig (Windows MSVC / WSL Ubuntu).
# Writes tagged copies under bench/rig/ for cross-runtime comparison.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export TRIBEV2_DATA_DIR="${TRIBEV2_DATA_DIR:-$ROOT/data}"
TAG="${RIG_BENCH_TAG:-local}"
TIMESTEPS="${TIMESTEPS:-30}"
WARMUP="${WARMUP:-2}"
RUNS="${RUNS:-5}"

COMMON=(--timesteps "$TIMESTEPS" --warmup "$WARMUP" --runs "$RUNS")
OUT_DIR="$ROOT/bench/rig/$TAG"
mkdir -p "$OUT_DIR"

if [[ ! -f "$TRIBEV2_DATA_DIR/model.safetensors" ]]; then
  echo "ERROR: missing $TRIBEV2_DATA_DIR/model.safetensors (run: ./rig.sh sync-data)" >&2
  exit 1
fi

if ! command -v nvidia-smi &>/dev/null; then
  echo "WARN: nvidia-smi not found — CUDA leg will be skipped" >&2
  HAVE_GPU=0
else
  HAVE_GPU=1
  echo "GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
fi

echo "╔══════════════════════════════════════════════════════╗"
echo "║  TRIBE v2 — CUDA rig benchmark (tag=$TAG)            ║"
echo "╚══════════════════════════════════════════════════════╝"
echo "  TRIBEV2_DATA_DIR=$TRIBEV2_DATA_DIR"
echo "  T=$TIMESTEPS warmup=$WARMUP runs=$RUNS"
echo "  output: $OUT_DIR"
echo ""

bench_one() {
  local title=$1 features=$2 engines=$3
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "▶ $title"
  echo "   features=$features engines=$engines"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  cargo run --release -p tribev2 --example bench_encoder \
    --no-default-features --features "$features" -- \
    --engines "$engines" "${COMMON[@]}" || return 1
  # Tag JSON emitted by bench_encoder (bench/results_*.json)
  shopt -s nullglob
  for f in bench/results_*.json; do
    cp -f "$f" "$OUT_DIR/$(basename "$f")"
  done
  shopt -u nullglob
  echo ""
}

# CPU baselines (same weights path as CUDA)
bench_one "pure-Rust CPU" "pure-rust" "rust"
bench_one "RLX CPU" "pure-rust,rlx-encoder,rlx-cpu" "rlx-cpu"

if [[ "$HAVE_GPU" -eq 1 ]]; then
  # Encoder-only CUDA avoids pulling full text stack on the rig.
  bench_one "RLX CUDA" "pure-rust,rlx-encoder,rlx-cuda-enc" "rlx-cuda" || {
    echo "WARN: RLX CUDA bench failed (toolkit / driver?)" >&2
  }
  # Optional: wgpu on NVIDIA (often slower than CUDA; useful for parity)
  bench_one "RLX wgpu" "pure-rust,rlx-encoder,rlx-gpu-enc" "rlx-wgpu" || true
fi

echo "Done → $OUT_DIR"
ls -la "$OUT_DIR" 2>/dev/null || true
