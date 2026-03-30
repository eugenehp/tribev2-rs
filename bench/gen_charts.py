#!/usr/bin/env python3
"""Generate benchmark charts from JSON results."""

import json, os, glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
BENCH_DIR = os.path.dirname(__file__)

def load_results():
    results = {}
    for path in sorted(glob.glob(os.path.join(BENCH_DIR, "results_*.json"))):
        with open(path) as f:
            results.update(json.load(f))
    return results

# All backends for the full model (20484 outputs)
ALL_ORDER = [
    ("rust_cpu",                    "Rust CPU\n(naive)",          "#CCCCCC"),
    ("rust_burn_ndarray",           "Burn\nNdArray",              "#DD8452"),
    ("rust_burn_ndarray_accelerate","Burn NdArray\n+Accelerate",  "#64B5CD"),
    ("rust_cpu_accelerate",         "Rust CPU\nAccelerate",       "#8172B3"),
    ("python_cpu_1thread",          "Python\nCPU",                "#4C72B0"),
    ("rust_burn_wgpu_metal",        "Burn wgpu\nMetal GPU",       "#55A868"),
    ("python_mps",                  "Python\nMPS GPU",            "#C44E52"),
]


def make_latency_chart(results):
    keys = [(k, l, c) for k, l, c in ALL_ORDER if k in results]
    if not keys:
        return
    labels = [l for _, l, _ in keys]
    means  = [results[k]["mean_ms"] for k, _, _ in keys]
    stds   = [results[k].get("std_ms", 0) for k, _, _ in keys]
    colors = [c for _, _, c in keys]

    fig, ax = plt.subplots(figsize=(max(8, len(keys) * 1.6), 5))
    bars = ax.bar(range(len(keys)), means, yerr=stds, capsize=4,
                  color=colors, edgecolor="black", linewidth=0.5)

    for bar, m, s in zip(bars, means, stds):
        y = bar.get_height()
        label = f"{m/1000:.1f}s" if m >= 1000 else f"{m:.0f}ms"
        ax.text(bar.get_x() + bar.get_width() / 2, y + s + max(means) * 0.05,
                label, ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("TRIBE v2 Forward Pass Latency\nbatch=1, T=100, 3 modalities, 20484 vertices",
                 fontsize=13, fontweight="bold")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x:.0f}" if x < 1000 else f"{x/1000:.1f}k"))
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "bench_latency.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


def make_speedup_chart(results):
    # Speedup vs slowest (naive CPU)
    baseline_key = "rust_cpu"
    if baseline_key not in results:
        return
    baseline = results[baseline_key]["mean_ms"]
    keys = [(k, l, c) for k, l, c in ALL_ORDER if k in results]
    labels   = [l for _, l, _ in keys]
    speedups = [baseline / results[k]["mean_ms"] for k, _, _ in keys]
    colors   = [c for _, _, c in keys]

    fig, ax = plt.subplots(figsize=(max(8, len(keys) * 1.6), 5))
    bars = ax.bar(range(len(keys)), speedups, color=colors,
                  edgecolor="black", linewidth=0.5)
    for bar, s in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(speedups) * 0.02,
                f"{s:.0f}×", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Speedup vs naive CPU", fontsize=12)
    ax.set_title("TRIBE v2 Speedup vs Rust CPU (naive loops)\n20484 vertices, 100 timesteps",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "bench_speedup.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


def make_gpu_chart(results):
    """GPU-focused comparison: Python MPS vs Rust wgpu Metal."""
    gpu_order = [
        ("python_cpu_1thread",      "Python\nCPU",          "#4C72B0"),
        ("rust_cpu_accelerate",     "Rust CPU\nAccelerate",  "#8172B3"),
        ("rust_burn_wgpu_metal",    "Burn wgpu\nMetal GPU",  "#55A868"),
        ("python_mps",              "Python\nMPS GPU",       "#C44E52"),
    ]
    keys = [(k, l, c) for k, l, c in gpu_order if k in results]
    if len(keys) < 2:
        return
    labels = [l for _, l, _ in keys]
    means  = [results[k]["mean_ms"] for k, _, _ in keys]
    stds   = [results[k].get("std_ms", 0) for k, _, _ in keys]
    colors = [c for _, _, c in keys]

    fig, ax = plt.subplots(figsize=(max(6, len(keys) * 1.8), 5))
    bars = ax.bar(range(len(keys)), means, yerr=stds, capsize=4,
                  color=colors, edgecolor="black", linewidth=0.5)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + max(means) * 0.03,
                f"{m:.1f}ms", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("TRIBE v2 — GPU vs CPU Comparison\n20484 vertices, 100 timesteps, 3 modalities",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "bench_gpu.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


def make_gpu_speedup_chart(results):
    gpu_order = [
        ("python_cpu_1thread",      "Python CPU",          "#4C72B0"),
        ("rust_cpu_accelerate",     "Rust Accelerate",     "#8172B3"),
        ("rust_burn_wgpu_metal",    "Burn wgpu Metal",     "#55A868"),
        ("python_mps",              "Python MPS",          "#C44E52"),
    ]
    baseline_key = "python_cpu_1thread"
    if baseline_key not in results:
        return
    baseline = results[baseline_key]["mean_ms"]
    keys = [(k, l, c) for k, l, c in gpu_order if k in results]
    labels   = [l for _, l, _ in keys]
    speedups = [baseline / results[k]["mean_ms"] for k, _, _ in keys]
    colors   = [c for _, _, c in keys]

    fig, ax = plt.subplots(figsize=(max(6, len(keys) * 1.8), 5))
    bars = ax.bar(range(len(keys)), speedups, color=colors,
                  edgecolor="black", linewidth=0.5)
    for bar, s in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(speedups) * 0.02,
                f"{s:.1f}×", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Speedup vs Python CPU", fontsize=12)
    ax.set_title("TRIBE v2 — Speedup vs Python CPU\n20484 vertices, full model",
                 fontsize=13, fontweight="bold")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "bench_gpu_speedup.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


def make_table(results):
    all_entries = [
        ("rust_cpu",                    "Rust CPU (naive loops)"),
        ("rust_burn_ndarray",           "Burn NdArray (Rayon)"),
        ("rust_burn_ndarray_accelerate","Burn NdArray + Accelerate"),
        ("rust_cpu_accelerate",         "Rust CPU (Accelerate BLAS)"),
        ("python_cpu_1thread",          "Python CPU (1 thread)"),
        ("rust_burn_wgpu_metal",        "Burn wgpu (Metal GPU)"),
        ("python_mps",                  "Python MPS (GPU)"),
    ]

    baseline = results.get("rust_cpu", {}).get("mean_ms", 1)

    lines = [
        "| Backend | Mean (ms) | Min (ms) | Std (ms) | Speedup |",
        "|---------|-----------|----------|----------|---------|",
    ]
    for k, label in all_entries:
        if k not in results:
            continue
        r = results[k]
        sp = f"{baseline / r['mean_ms']:.0f}×"
        lines.append(f"| {label} | {r['mean_ms']:.1f} | {r['min_ms']:.1f} | {r.get('std_ms',0):.1f} | {sp} |")

    table = "\n".join(lines)
    path = os.path.join(FIGURES_DIR, "bench_table.md")
    with open(path, "w") as f:
        f.write(table + "\n")
    print(f"\nSaved {path}")
    print(table)


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    results = load_results()
    if not results:
        print("No results found")
        return
    print(f"Loaded {len(results)} benchmark results\n")

    make_latency_chart(results)
    make_speedup_chart(results)
    make_gpu_chart(results)
    make_gpu_speedup_chart(results)
    make_table(results)


if __name__ == "__main__":
    main()
