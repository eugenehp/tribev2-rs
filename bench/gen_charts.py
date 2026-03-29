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

# Full model (n_outputs=20484) comparison
FULL_ORDER = [
    ("python_cpu_1thread",          "Python\nCPU 1-thread",     "#4C72B0"),
    ("python_mps",                  "Python\nMPS (GPU)",        "#C44E52"),
    ("rust_cpu_accelerate",         "Rust manual\nAccelerate",  "#8172B3"),
    ("rust_burn_ndarray",           "Rust Burn\nNdArray",       "#DD8452"),
    ("rust_burn_ndarray_accelerate","Rust Burn\nNdArray+Accel", "#64B5CD"),
]

# Fair GPU comparison (n_outputs=2048)
GPU_ORDER = [
    ("python_cpu_1thread_2048", "Python\nCPU 1t",         "#4C72B0"),
    ("python_mps_2048",         "Python\nMPS (GPU)",       "#C44E52"),
    ("rust_burn_wgpu_metal",    "Rust Burn\nwgpu Metal",   "#55A868"),
]


def make_chart(results, order, title, filename, baseline_key=None):
    keys = [(k, l, c) for k, l, c in order if k in results]
    if not keys:
        return
    labels = [l for _, l, _ in keys]
    means = [results[k]["mean_ms"] for k, _, _ in keys]
    stds = [results[k].get("std_ms", 0) for k, _, _ in keys]
    colors = [c for _, _, c in keys]

    fig, ax = plt.subplots(figsize=(max(6, len(keys)*2), 5))
    bars = ax.bar(range(len(keys)), means, yerr=stds, capsize=5,
                  color=colors, edgecolor="black", linewidth=0.5)

    for bar, m, s in zip(bars, means, stds):
        y = bar.get_height()
        label = f"{m/1000:.1f}s" if m >= 1000 else f"{m:.0f}ms"
        ax.text(bar.get_x() + bar.get_width()/2, y + s + max(means)*0.03,
                label, ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    if max(means) / max(min(means), 1) > 10:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{x:.0f}" if x < 1000 else f"{x/1000:.1f}k"))

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


def make_speedup_chart(results, order, baseline_key, title, filename):
    if baseline_key not in results:
        return
    baseline = results[baseline_key]["mean_ms"]
    keys = [(k, l, c) for k, l, c in order if k in results]
    labels = [l for _, l, _ in keys]
    speedups = [baseline / results[k]["mean_ms"] for k, _, _ in keys]
    colors = [c for _, _, c in keys]

    fig, ax = plt.subplots(figsize=(max(6, len(keys)*2), 5))
    bars = ax.bar(range(len(keys)), speedups, color=colors,
                  edgecolor="black", linewidth=0.5)
    for bar, s in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(speedups)*0.02,
                f"{s:.1f}×", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Speedup", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


def make_table(results):
    all_order = FULL_ORDER + [
        ("rust_cpu",              "Rust manual CPU (naive)", "#DD8452"),
        ("python_cpu_1thread_2048", "Python CPU 1t (2048)",   "#4C72B0"),
        ("python_mps_2048",       "Python MPS (2048)",       "#C44E52"),
        ("rust_burn_wgpu_metal",  "Rust Burn wgpu Metal (2048)", "#55A868"),
    ]
    baseline = results.get("python_cpu_1thread", {}).get("mean_ms")

    lines = ["| Backend | Mean (ms) | Min (ms) | Std (ms) | vs Py CPU 1t |",
             "|---------|-----------|----------|----------|--------------|"]
    for k, l, _ in all_order:
        if k not in results:
            continue
        r = results[k]
        l = l.replace("\n", " ")
        sp = f"{baseline / r['mean_ms']:.1f}×" if baseline else "—"
        lines.append(f"| {l} | {r['mean_ms']:.1f} | {r['min_ms']:.1f} | {r.get('std_ms',0):.1f} | {sp} |")

    table = "\n".join(lines)
    path = os.path.join(FIGURES_DIR, "bench_table.md")
    with open(path, "w") as f:
        f.write(table + "\n")
    print(f"Saved {path}")
    print(table)


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    results = load_results()
    if not results:
        print("No results found")
        return
    print(f"Loaded {len(results)} benchmark results\n")

    make_chart(results, FULL_ORDER,
               "TRIBE v2 Forward Pass — Full Model (20484 outputs)\nbatch=1, T=100, 3 modalities",
               "bench_latency.png")

    make_chart(results, GPU_ORDER,
               "TRIBE v2 Forward Pass — GPU Comparison (2048 outputs)\nbatch=1, T=100, 3 modalities",
               "bench_gpu.png")

    make_speedup_chart(results, FULL_ORDER, "python_cpu_1thread",
                       "Speedup vs Python CPU 1-thread (full model)",
                       "bench_speedup.png")

    make_speedup_chart(results, GPU_ORDER, "python_cpu_1thread_2048",
                       "Speedup vs Python CPU 1-thread (2048 outputs)",
                       "bench_gpu_speedup.png")

    make_table(results)

if __name__ == "__main__":
    main()
