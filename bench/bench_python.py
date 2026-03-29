#!/usr/bin/env python3
"""
Benchmark the TRIBE v2 FmriEncoderModel forward pass in Python.

Reconstructs the exact pretrained architecture using only PyTorch + x_transformers,
no neuralset/neuraltrain required. Matches config.yaml from facebook/tribev2.

Tests:
  - CPU (single-thread and multi-thread)
  - MPS (Apple GPU) if available

Outputs JSON results to bench/results_python.json
"""

import json
import os
import sys
import time

import torch
import torch.nn as nn
from x_transformers import Encoder

# ── Architecture (from config.yaml) ───────────────────────────────────────

HIDDEN = 1152
DEPTH = 8
HEADS = 8
FF_MULT = 4
MAX_SEQ_LEN = 1024
LOW_RANK = 2048
N_OUTPUTS = 20484        # fsaverage5
N_OUTPUT_TIMESTEPS = 100
N_SUBJECTS = 25
SUBJECT_DROPOUT = 0.1

# Modality dims: text (3, 3072), audio (3, 1024), video (3, 1408)
# With layer_aggregation="cat", extractor_aggregation="cat"
MODALITIES = {
    "text":  (3, 3072),   # input_dim = 3*3072 = 9216, output_dim = 1152//3 = 384
    "audio": (3, 1024),   # input_dim = 3*1024 = 3072, output_dim = 384
    "video": (3, 1408),   # input_dim = 3*1408 = 4224, output_dim = 384
}


class SubjectLayersModel(nn.Module):
    """Exact replica of neuraltrain SubjectLayersModel."""
    def __init__(self, in_ch, out_ch, n_subjects=25, subject_dropout=0.1):
        super().__init__()
        self.n_subjects = n_subjects
        self.subject_dropout = subject_dropout
        n = n_subjects + (1 if subject_dropout else 0)
        self.weights = nn.Parameter(torch.randn(n, in_ch, out_ch) / in_ch**0.5)
        self.bias = nn.Parameter(torch.randn(n, out_ch) / in_ch**0.5)

    def forward(self, x, subjects=None):
        # average_subjects mode
        w = self.weights[self.n_subjects]
        out = torch.einsum("bct,cd->bdt", x, w)
        out += self.bias[self.n_subjects].view(1, -1, 1)
        return out


class FmriEncoderModel(nn.Module):
    """Exact replica of the pretrained TRIBE v2 FmriEncoderModel."""
    def __init__(self):
        super().__init__()

        n_mod = len(MODALITIES)
        # Projectors
        self.projectors = nn.ModuleDict()
        for name, (n_layers, feat_dim) in MODALITIES.items():
            in_dim = n_layers * feat_dim
            out_dim = HIDDEN // n_mod
            self.projectors[name] = nn.Linear(in_dim, out_dim)

        # No combiner (Identity)

        # Time positional embedding
        self.time_pos_embed = nn.Parameter(torch.randn(1, MAX_SEQ_LEN, HIDDEN))

        # Encoder: x_transformers
        self.encoder = Encoder(
            dim=HIDDEN,
            depth=DEPTH,
            heads=HEADS,
            ff_mult=FF_MULT,
            use_scalenorm=True,
            rotary_pos_emb=True,
            scale_residual=True,
            attn_dim_head=HIDDEN // HEADS,
        )

        # Low-rank head
        self.low_rank_head = nn.Linear(HIDDEN, LOW_RANK, bias=False)

        # Predictor
        self.predictor = SubjectLayersModel(LOW_RANK, N_OUTPUTS, N_SUBJECTS, SUBJECT_DROPOUT)

        # Pooler
        self.pooler = nn.AdaptiveAvgPool1d(N_OUTPUT_TIMESTEPS)

    def forward(self, features):
        """
        features: dict of modality_name -> [B, L*D, T] tensors
        """
        B = None
        T = None
        for v in features.values():
            B = v.shape[0]
            T = v.shape[-1]
            break

        n_mod = len(MODALITIES)
        tensors = []
        for name in MODALITIES:
            if name in features:
                data = features[name]  # [B, L*D, T]
                data = data.transpose(1, 2)  # [B, T, L*D]
                data = self.projectors[name](data)  # [B, T, H//n_mod]
                tensors.append(data)
            else:
                tensors.append(torch.zeros(B, T, HIDDEN // n_mod, device=data.device))

        x = torch.cat(tensors, dim=-1)  # [B, T, H]

        # Time pos embed
        x = x + self.time_pos_embed[:, :x.size(1)]

        # Encoder
        x = self.encoder(x)  # [B, T, H]

        # Transpose to [B, H, T]
        x = x.transpose(1, 2)

        # Low-rank head
        x = self.low_rank_head(x.transpose(1, 2)).transpose(1, 2)  # [B, LR, T]

        # Predictor
        x = self.predictor(x)  # [B, N_OUTPUTS, T]

        # Pool
        x = self.pooler(x)  # [B, N_OUTPUTS, N_OUTPUT_TIMESTEPS]

        return x


def make_input(device, batch_size=1, T=100):
    """Create synthetic input features matching pretrained dims."""
    features = {}
    for name, (n_layers, feat_dim) in MODALITIES.items():
        features[name] = torch.randn(batch_size, n_layers * feat_dim, T, device=device)
    return features


def benchmark(model, features, n_warmup=5, n_runs=20, device_name="cpu"):
    """Run forward pass benchmark."""
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(n_warmup):
            _ = model(features)
            if "mps" in device_name:
                torch.mps.synchronize()

        # Timed runs
        times = []
        for _ in range(n_runs):
            if "mps" in device_name:
                torch.mps.synchronize()
            t0 = time.perf_counter()
            out = model(features)
            if "mps" in device_name:
                torch.mps.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

    return {
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "std_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
        "n_runs": n_runs,
        "output_shape": list(out.shape),
    }


def main():
    results = {}
    n_warmup = 5
    n_runs = 20

    # ── CPU single-thread ──────────────────────────────────────────────
    print("=== Python CPU (1 thread) ===")
    torch.set_num_threads(1)
    model = FmriEncoderModel().float()
    features = make_input("cpu")
    r = benchmark(model, features, n_warmup, n_runs, "cpu")
    print(f"  Mean: {r['mean_ms']:.1f} ms, Min: {r['min_ms']:.1f} ms, Std: {r['std_ms']:.1f} ms")
    results["python_cpu_1thread"] = r

    # ── CPU multi-thread ───────────────────────────────────────────────
    print("=== Python CPU (all threads) ===")
    torch.set_num_threads(torch.get_num_threads())
    # Re-read actual core count
    n_cores = os.cpu_count() or 10
    torch.set_num_threads(n_cores)
    print(f"  Using {torch.get_num_threads()} threads")
    model = FmriEncoderModel().float()
    features = make_input("cpu")
    r = benchmark(model, features, n_warmup, n_runs, "cpu")
    print(f"  Mean: {r['mean_ms']:.1f} ms, Min: {r['min_ms']:.1f} ms, Std: {r['std_ms']:.1f} ms")
    results["python_cpu_multithread"] = r

    # ── MPS (Apple GPU) ────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        print("=== Python MPS (Apple GPU) ===")
        device = torch.device("mps")
        model = FmriEncoderModel().float().to(device)
        features = make_input(device)
        r = benchmark(model, features, n_warmup, n_runs, "mps")
        print(f"  Mean: {r['mean_ms']:.1f} ms, Min: {r['min_ms']:.1f} ms, Std: {r['std_ms']:.1f} ms")
        results["python_mps"] = r

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "results_python.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
