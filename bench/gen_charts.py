#!/usr/bin/env python3
"""Generate benchmark charts from JSON results.

Charts produced
---------------
bench_latency.png       — all backends, log-scale bar chart
bench_speedup.png       — speedup vs naive Rust CPU
bench_gpu_detail.png    — GPU & accelerated backends, linear scale
bench_optimization.png  — step-by-step wgpu improvement story
bench_table.md          — markdown summary table
"""

import json, os, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
BENCH_DIR   = os.path.dirname(__file__)


def load_results():
    results = {}
    for path in sorted(glob.glob(os.path.join(BENCH_DIR, "results_*.json"))):
        with open(path) as f:
            results.update(json.load(f))
    return results


# ── Colour palette ──────────────────────────────────────────────────────────
C = {
    "rust_cpu":                     "#BBBBBB",
    "rust_cpu_accelerate":          "#8172B3",
    "rust_burn_ndarray":            "#DD8452",
    "rust_burn_ndarray_accelerate": "#64B5CD",
    "python_cpu_1thread":           "#4C72B0",
    "python_cpu_multithread":       "#6EA6D0",
    "python_mps":                   "#C44E52",
    "rust_burn_wgpu_metal":         "#55A868",
    "rust_burn_wgpu_metal_f16":     "#2CA02C",
    "rust_burn_wgpu_metal_kernels": "#1B7837",
}

# Master display order (key, short label, colour key)
ALL = [
    ("rust_cpu",                     "Rust CPU\n(naive)",          "rust_cpu"),
    ("rust_burn_ndarray",            "Burn\nNdArray",              "rust_burn_ndarray"),
    ("rust_burn_ndarray_accelerate", "Burn NdArray\n+Accelerate",  "rust_burn_ndarray_accelerate"),
    ("rust_cpu_accelerate",          "Rust CPU\nAccelerate",       "rust_cpu_accelerate"),
    ("python_cpu_1thread",           "Python\nCPU",                "python_cpu_1thread"),
    ("rust_burn_wgpu_metal",         "Burn wgpu\nf32",             "rust_burn_wgpu_metal"),
    ("rust_burn_wgpu_metal_f16",     "Burn wgpu\nf16",             "rust_burn_wgpu_metal_f16"),
    ("rust_burn_wgpu_metal_kernels", "Burn wgpu\nf32+kernels",     "rust_burn_wgpu_metal_kernels"),
    ("python_mps",                   "Python\nMPS GPU",            "python_mps"),
]


def _bar(ax, indices, means, stds, colors, labels,
         value_fmt="{:.0f}ms", value_offset_frac=0.04):
    bars = ax.bar(indices, means, yerr=stds if any(s > 0 for s in stds) else None,
                  capsize=4, color=colors, edgecolor="black", linewidth=0.6,
                  error_kw=dict(lw=1, capthick=1))
    top = max(means)
    for bar, m, s in zip(bars, means, stds):
        lbl = value_fmt.format(m)
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + s + top * value_offset_frac,
                lbl, ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(indices)
    ax.set_xticklabels(labels, fontsize=9)
    return bars


# ── 1. Full latency (log scale) ─────────────────────────────────────────────
def make_latency_chart(r):
    keys = [(k, l, C[ck]) for k, l, ck in ALL if k in r]
    if not keys: return
    idx    = list(range(len(keys)))
    means  = [r[k]["mean_ms"] for k, _, _ in keys]
    stds   = [r[k].get("std_ms", 0) for k, _, _ in keys]
    colors = [c for _, _, c in keys]
    labels = [l for _, l, _ in keys]

    fig, ax = plt.subplots(figsize=(max(9, len(keys) * 1.7), 5.5))
    _bar(ax, idx, means, stds, colors, labels,
         value_fmt="{:.0f}ms", value_offset_frac=0.0)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x):,}" if x < 1000 else f"{x/1000:.1f}k"))
    ax.set_ylabel("Latency  (ms, log scale)", fontsize=11)
    ax.set_title(
        "TRIBE v2  Forward Pass Latency — All Backends\n"
        "batch=1 · T=100 · 3 modalities · 20 484 cortical vertices",
        fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "bench_latency.png")


# ── 2. Speedup vs naive CPU ─────────────────────────────────────────────────
def make_speedup_chart(r):
    base_key = "rust_cpu"
    if base_key not in r: return
    base   = r[base_key]["mean_ms"]
    keys   = [(k, l, C[ck]) for k, l, ck in ALL if k in r]
    idx    = list(range(len(keys)))
    sp     = [base / r[k]["mean_ms"] for k, _, _ in keys]
    colors = [c for _, _, c in keys]
    labels = [l for _, l, _ in keys]

    fig, ax = plt.subplots(figsize=(max(9, len(keys) * 1.7), 5.5))
    bars = ax.bar(idx, sp, color=colors, edgecolor="black", linewidth=0.6)
    for bar, s in zip(bars, sp):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(sp) * 0.02,
                f"{s:.0f}×", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Speedup vs Rust CPU naive", fontsize=11)
    ax.set_title(
        "TRIBE v2  Speedup vs Rust CPU (naive loops)\n"
        "20 484 vertices · 100 timesteps",
        fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "bench_speedup.png")


# ── 3. GPU detail (linear scale) ────────────────────────────────────────────
def make_gpu_detail_chart(r):
    order = [
        ("python_cpu_1thread",           "Python\nCPU",          "python_cpu_1thread"),
        ("rust_cpu_accelerate",          "Rust CPU\nAccelerate",  "rust_cpu_accelerate"),
        ("rust_burn_wgpu_metal",         "Burn wgpu\nf32",        "rust_burn_wgpu_metal"),
        ("rust_burn_wgpu_metal_f16",     "Burn wgpu\nf16",        "rust_burn_wgpu_metal_f16"),
        ("rust_burn_wgpu_metal_kernels", "Burn wgpu\nf32+kernels","rust_burn_wgpu_metal_kernels"),
        ("python_mps",                   "Python\nMPS GPU",       "python_mps"),
    ]
    keys = [(k, l, C[ck]) for k, l, ck in order if k in r]
    if len(keys) < 2: return
    idx    = list(range(len(keys)))
    means  = [r[k]["mean_ms"] for k, _, _ in keys]
    stds   = [r[k].get("std_ms", 0) for k, _, _ in keys]
    colors = [c for _, _, c in keys]
    labels = [l for _, l, _ in keys]

    fig, ax = plt.subplots(figsize=(max(7, len(keys) * 1.9), 5.5))
    _bar(ax, idx, means, stds, colors, labels, value_fmt="{:.1f}ms")

    # reference line: Python MPS
    if "python_mps" in r:
        mps = r["python_mps"]["mean_ms"]
        ax.axhline(mps, color=C["python_mps"], linestyle="--", linewidth=1.3, alpha=0.7,
                   label=f"Python MPS  {mps:.1f} ms")
        ax.legend(fontsize=9, loc="upper right")

    ax.set_ylabel("Latency  (ms)", fontsize=11)
    ax.set_title(
        "TRIBE v2  GPU & Accelerated Backends\n"
        "20 484 vertices · 100 timesteps · batch=1",
        fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "bench_gpu_detail.png")


# ── 4. Optimisation story (wgpu step-by-step) ───────────────────────────────
def make_optimization_chart(r):
    """Waterfall showing each optimisation applied to the wgpu Metal backend."""
    steps = [
        # (result_key,             label,                             color)
        ("rust_burn_wgpu_metal_baseline", "Original\n(pre-opt)",      "#BBBBBB"),
        # synthesised from history if not present:
        ("rust_burn_wgpu_metal",          "Separate QKV\n+manual attn\n+RoPE cache\n+w_avg_t",
                                                                       C["rust_burn_wgpu_metal"]),
        ("rust_burn_wgpu_metal_f16",      "  + f16\n  dtype",         C["rust_burn_wgpu_metal_f16"]),
        ("rust_burn_wgpu_metal_kernels",  "+ Fused CubeCL\n  kernels", C["rust_burn_wgpu_metal_kernels"]),
        ("python_mps",                    "Python\nMPS",               C["python_mps"]),
    ]

    # Use hard-coded original if not in results
    baseline_ms = 27.6
    known = {}
    for k, l, c in steps:
        if k in r:
            known[k] = r[k]["mean_ms"]
    known.setdefault("rust_burn_wgpu_metal_baseline", baseline_ms)

    present = [(k, l, c) for k, l, c in steps if k in known]
    if len(present) < 3: return

    idx    = list(range(len(present)))
    means  = [known[k] for k, _, _ in present]
    labels = [l for _, l, _ in present]
    colors = [c for _, _, c in present]
    stds   = [r[k].get("std_ms", 0) if k in r else 0 for k, _, _ in present]

    fig, ax = plt.subplots(figsize=(max(8, len(present) * 2.0), 5.5))
    _bar(ax, idx, means, stds, colors, labels, value_fmt="{:.1f}ms", value_offset_frac=0.03)

    # Annotation: delta arrows between bars
    for i in range(1, len(means)):
        delta = means[i] - means[i - 1]
        if abs(delta) > 0.3:
            x0 = idx[i - 1] + 0.4
            x1 = idx[i]     - 0.4
            y  = max(means[i - 1], means[i]) * 1.18
            ax.annotate("", xy=(x1, y), xytext=(x0, y),
                        arrowprops=dict(arrowstyle="->", color="grey", lw=1.2))
            ax.text((x0 + x1) / 2, y * 1.03,
                    f"{delta:+.1f}ms",
                    ha="center", va="bottom", fontsize=8, color="grey")

    ax.set_ylabel("Latency  (ms)", fontsize=11)
    ax.set_title(
        "Burn wgpu Metal — Optimisation Journey\n"
        "Each bar = cumulative improvements applied",
        fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "bench_optimization.png")


# ── 5. Markdown table ────────────────────────────────────────────────────────
def make_table(r):
    entries = [
        ("rust_cpu",                     "Rust CPU (naive loops)"),
        ("rust_burn_ndarray",            "Burn NdArray (Rayon)"),
        ("rust_burn_ndarray_accelerate", "Burn NdArray + Accelerate"),
        ("rust_cpu_accelerate",          "Rust CPU + Accelerate BLAS"),
        ("python_cpu_1thread",           "Python CPU (1 thread)"),
        ("rust_burn_wgpu_metal",         "Burn wgpu Metal f32"),
        ("rust_burn_wgpu_metal_f16",     "Burn wgpu Metal f16"),
        ("rust_burn_wgpu_metal_kernels", "Burn wgpu Metal f32 + fused kernels"),
        ("python_mps",                   "Python MPS GPU"),
    ]
    base = r.get("rust_cpu", {}).get("mean_ms", 1.0)
    lines = [
        "# TRIBE v2 Forward Pass Benchmark",
        "",
        "**Setup:** batch=1, T=100, 3 modalities (text/audio/video), 20 484 cortical vertices",
        "",
        "| Backend | Mean (ms) | Min (ms) | Std (ms) | vs CPU naive |",
        "|---------|----------:|---------:|---------:|-------------:|",
    ]
    for k, label in entries:
        if k not in r: continue
        d  = r[k]
        sp = f"{base / d['mean_ms']:.0f}×"
        lines.append(
            f"| {label} | {d['mean_ms']:.1f} | {d['min_ms']:.1f} |"
            f" {d.get('std_ms', 0):.1f} | {sp} |"
        )
    table = "\n".join(lines)
    path  = os.path.join(FIGURES_DIR, "bench_table.md")
    with open(path, "w") as f:
        f.write(table + "\n")
    print(f"Saved  {path}")
    print(table)


# ── helpers ───────────────────────────────────────────────────────────────────
def _save(fig, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved  {path}")
    plt.close(fig)


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    r = load_results()
    if not r:
        print("No results/*.json found — run benchmarks first.")
        return
    print(f"Loaded {len(r)} result entries\n")

    make_latency_chart(r)
    make_speedup_chart(r)
    make_gpu_detail_chart(r)
    make_optimization_chart(r)
    make_table(r)
    print("\nAll done — figures in figures/")


if __name__ == "__main__":
    main()
