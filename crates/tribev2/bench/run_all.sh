#!/bin/bash
set -e
cd "$(dirname "$0")/.."

echo "╔══════════════════════════════════════════════════════╗"
echo "║  TRIBE v2 Benchmark Suite — Python vs Rust          ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── Python benchmarks ──────────────────────────────────────
echo "▶ Running Python benchmarks..."
python3 bench/bench_python.py
echo ""

# ── Rust CPU (naive loops) ─────────────────────────────────
echo "▶ Building Rust (CPU naive)..."
cargo build --release --example bench_rust --no-default-features 2>&1 | tail -1
echo "▶ Running Rust CPU (naive)..."
./target/release/examples/bench_rust
echo ""

# ── Rust CPU (Accelerate BLAS) ─────────────────────────────
echo "▶ Building Rust (CPU + Accelerate BLAS)..."
cargo build --release --example bench_rust --no-default-features --features accelerate 2>&1 | tail -1
echo "▶ Running Rust CPU (Accelerate)..."
./target/release/examples/bench_rust
echo ""

# ── Generate charts ────────────────────────────────────────
echo "▶ Generating charts..."
python3 bench/gen_charts.py
echo ""
echo "Done! Results in figures/"
