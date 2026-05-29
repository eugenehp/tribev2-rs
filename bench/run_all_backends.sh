#!/usr/bin/env bash
# Sweep TRIBE v2 encoder benchmarks across optional engines/backends.
# Each block is a separate `cargo` build (features are compile-time).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export TRIBEV2_DATA_DIR="${TRIBEV2_DATA_DIR:-$ROOT/data}"
TIMESTEPS="${TIMESTEPS:-50}"
WARMUP="${WARMUP:-2}"
RUNS="${RUNS:-5}"

COMMON=(--timesteps "$TIMESTEPS" --warmup "$WARMUP" --runs "$RUNS")

bench_build() {
  local title=$1
  local features=$2
  shift 2
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "▶ $title"
  echo "   --no-default-features --features $features"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  cargo run --release -p tribev2 --example bench_encoder \
    --no-default-features --features "$features" -- \
    --engines all "${COMMON[@]}" "$@"
}

mkdir -p bench

echo "╔══════════════════════════════════════════════════════╗"
echo "║  TRIBE v2 — encoder benchmark (all backends)         ║"
echo "╚══════════════════════════════════════════════════════╝"
echo "  TRIBEV2_DATA_DIR=$TRIBEV2_DATA_DIR"
echo "  T=$TIMESTEPS warmup=$WARMUP runs=$RUNS"
echo ""

# ── Pure Rust (always) ─────────────────────────────────────
bench_build "pure-Rust CPU" "pure-rust"

# ── Burn CPU backends ────────────────────────────────────────
bench_build "Burn NdArray CPU" "pure-rust,burn,ndarray"

if [[ "$(uname -s)" == Darwin ]]; then
  bench_build "Burn NdArray + Accelerate" "pure-rust,burn,ndarray,blas-accelerate"
  bench_build "Burn wgpu Metal (f32)" "pure-rust,burn,wgpu-metal"
  bench_build "Burn wgpu Metal (f16)" "pure-rust,burn,wgpu-metal,wgpu-metal-f16" || true
fi

# ── RLX CPU ──────────────────────────────────────────────────
bench_build "RLX CPU" "pure-rust,rlx-encoder,rlx-cpu"

# ── RLX GPU (platform-specific) ─────────────────────────────
if [[ "$(uname -s)" == Darwin ]]; then
  bench_build "RLX Metal" "pure-rust,rlx-encoder,rlx-metal" || true
  bench_build "RLX MLX" "pure-rust,rlx-encoder,rlx-mlx" || true
  bench_build "RLX wgpu" "pure-rust,rlx-encoder,rlx-gpu" || true
fi

if command -v nvidia-smi &>/dev/null; then
  bench_build "RLX CUDA" "pure-rust,rlx-encoder,rlx-cuda" || true
fi

echo ""
echo "Done. JSON summaries in bench/results_*.json"
if [[ -f bench/gen_charts.py ]]; then
  echo "Run: python3 bench/gen_charts.py  (optional charts)"
fi
