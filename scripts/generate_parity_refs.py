#!/usr/bin/env python3
"""Generate reference outputs from the TRIBE v2 model for parity testing.

Loads the model from safetensors, runs forward passes with deterministic
synthetic features, and saves reference outputs as .bin files.

Usage:
    python3 scripts/generate_parity_refs.py

Outputs saved to data/parity_refs/
"""

import json
import os
import struct
import time
import sys

import numpy as np
import torch
import yaml
from safetensors.torch import load_file

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR = os.path.join(DATA_DIR, "parity_refs")
os.makedirs(OUT_DIR, exist_ok=True)


def save_f32_bin(path, arr):
    """Save a numpy array as raw little-endian f32 with a shape header."""
    arr = arr.astype(np.float32).flatten()
    with open(path, "wb") as f:
        # Header: ndims (u32), then each dim (u32)
        shape = arr.shape if arr.ndim > 0 else (len(arr),)
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<I", len(arr)))
        f.write(arr.tobytes())
    print(f"  Saved {path} ({len(arr)} floats, {os.path.getsize(path)} bytes)")


def save_tensor_bin(path, t):
    """Save a torch tensor with shape header."""
    arr = t.detach().cpu().float().numpy()
    flat = arr.flatten()
    with open(path, "wb") as f:
        f.write(struct.pack("<I", arr.ndim))
        for d in arr.shape:
            f.write(struct.pack("<I", d))
        f.write(flat.tobytes())
    print(f"  Saved {path}: shape={list(arr.shape)}, {os.path.getsize(path)} bytes")


# ── Load config ──────────────────────────────────────────────────────────

print("Loading config...")
with open(os.path.join(DATA_DIR, "config.yaml")) as f:
    config = yaml.unsafe_load(f)

with open(os.path.join(DATA_DIR, "build_args.json")) as f:
    build_args = json.load(f)

feature_dims = build_args["feature_dims"]
n_outputs = build_args["n_outputs"]
n_output_timesteps = build_args["n_output_timesteps"]
hidden = config["brain_model_config"]["hidden"]

print(f"  hidden={hidden}, n_outputs={n_outputs}, n_output_timesteps={n_output_timesteps}")
print(f"  feature_dims={feature_dims}")

# ── Load weights ─────────────────────────────────────────────────────────

print("Loading weights...")
weights_path = os.path.join(DATA_DIR, "model.safetensors")
state_dict = load_file(weights_path)

# Strip "model." prefix if present
state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}

print(f"  {len(state_dict)} tensors loaded")

# Print some key shapes
for key in ["time_pos_embed", "low_rank_head.weight", "predictor.weights", "predictor.bias"]:
    if key in state_dict:
        print(f"  {key}: {list(state_dict[key].shape)}")


# ── Build a minimal forward pass ─────────────────────────────────────────

# We'll build the forward pass step by step to match the Rust implementation
# and save intermediate results for comparison.

torch.manual_seed(42)
np.random.seed(42)

B = 1
T = 20  # small number of timesteps for fast testing

print(f"\nGenerating deterministic features (B={B}, T={T})...")

# Generate deterministic features using linspace patterns
features = {}
for mod_name, (n_layers, feat_dim) in feature_dims.items():
    total_dim = n_layers * feat_dim
    data = torch.zeros(B, total_dim, T)
    for d in range(total_dim):
        for t in range(T):
            # Simple deterministic pattern: sin + cos mix
            data[0, d, t] = 0.1 * np.sin(0.3 * d + 0.7 * t) + 0.05 * np.cos(0.5 * d - 0.2 * t)
    features[mod_name] = data
    save_tensor_bin(os.path.join(OUT_DIR, f"input_{mod_name}.bin"), data)

print(f"  text: {list(features['text'].shape)}")
print(f"  audio: {list(features['audio'].shape)}")
print(f"  video: {list(features['video'].shape)}")


# ── Step 1: Projectors ───────────────────────────────────────────────────

print("\nStep 1: Projectors...")
n_modalities = len(feature_dims)
proj_outputs = {}

for mod_name, (n_layers, feat_dim) in feature_dims.items():
    total_dim = n_layers * feat_dim
    out_dim = hidden // n_modalities

    w_key = f"projectors.{mod_name}.weight"
    b_key = f"projectors.{mod_name}.bias"

    if w_key not in state_dict:
        print(f"  WARNING: {w_key} not found, skipping")
        continue

    W = state_dict[w_key]  # [out_dim, total_dim]
    bias = state_dict.get(b_key, torch.zeros(out_dim))

    x = features[mod_name]  # [B, total_dim, T]
    x = x.transpose(1, 2)   # [B, T, total_dim]
    x = torch.nn.functional.linear(x, W, bias)  # [B, T, out_dim]
    proj_outputs[mod_name] = x

    print(f"  {mod_name}: [{total_dim}] -> [{out_dim}], W={list(W.shape)}")
    save_tensor_bin(os.path.join(OUT_DIR, f"proj_{mod_name}.bin"), x)


# ── Step 2: Concatenate (extractor_aggregation="cat") ────────────────────

print("\nStep 2: Concatenate modalities...")
# Order: text, audio, video (alphabetical from BTreeMap in Rust)
cat_order = sorted(proj_outputs.keys())
x = torch.cat([proj_outputs[k] for k in cat_order], dim=-1)  # [B, T, hidden]
print(f"  Concatenated: {list(x.shape)}")
save_tensor_bin(os.path.join(OUT_DIR, "after_cat.bin"), x)


# ── Step 3: Time positional embedding ────────────────────────────────────

print("\nStep 3: Time positional embedding...")
tpe = state_dict.get("time_pos_embed")
if tpe is not None:
    x = x + tpe[:, :T, :]
    print(f"  Added time_pos_embed: {list(tpe.shape)} -> sliced to T={T}")
    save_tensor_bin(os.path.join(OUT_DIR, "after_pos_embed.bin"), x)


# ── Step 4: Encoder (full x-transformers forward) ────────────────────────

print("\nStep 4: Encoder forward...")
t0 = time.time()

encoder_config = config["brain_model_config"]["encoder"]
depth = encoder_config["depth"]
heads = encoder_config["heads"]
dim_head = hidden // heads
ff_mult = encoder_config.get("ff_mult", 4)
inner_dim = hidden * ff_mult

# Rotary embedding
rot_dim = max(dim_head // 2, 32)
base = 10000.0
inv_freq = 1.0 / (base ** (torch.arange(0, rot_dim, 2).float() / rot_dim))
positions = torch.arange(T).float()
freqs = torch.einsum("i,j->ij", positions, inv_freq)
freqs = torch.cat([freqs, freqs], dim=-1)  # [T, rot_dim]


def rotate_half(x):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(t, freqs):
    rot_dim = freqs.shape[-1]
    t_rot, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    cos_f = freqs.cos().unsqueeze(0).unsqueeze(0)  # [1, 1, T, rot_dim]
    sin_f = freqs.sin().unsqueeze(0).unsqueeze(0)
    t_rot = t_rot * cos_f + rotate_half(t_rot) * sin_f
    return torch.cat([t_rot, t_pass], dim=-1)


def scale_norm(x, g, dim):
    scale = dim ** 0.5
    return torch.nn.functional.normalize(x, dim=-1) * scale * g


# Process encoder layers
for layer_idx in range(depth * 2):
    prefix = f"encoder.layers.{layer_idx}"

    # Pre-norm (ScaleNorm)
    g = state_dict[f"{prefix}.0.0.g"].item()
    residual = x.clone()
    x = scale_norm(x, g, hidden)

    if layer_idx % 2 == 0:
        # Attention layer
        Wq = state_dict[f"{prefix}.1.to_q.weight"]  # [dim, dim]
        Wk = state_dict[f"{prefix}.1.to_k.weight"]
        Wv = state_dict[f"{prefix}.1.to_v.weight"]
        Wo = state_dict[f"{prefix}.1.to_out.weight"]

        q = torch.nn.functional.linear(x, Wq)  # [B, T, dim]
        k = torch.nn.functional.linear(x, Wk)
        v = torch.nn.functional.linear(x, Wv)

        # Reshape to [B, heads, T, dim_head]
        q = q.view(B, T, heads, dim_head).permute(0, 2, 1, 3)
        k = k.view(B, T, heads, dim_head).permute(0, 2, 1, 3)
        v = v.view(B, T, heads, dim_head).permute(0, 2, 1, 3)

        # Apply rotary
        q = apply_rotary(q, freqs)
        k = apply_rotary(k, freqs)

        # Scaled dot-product attention
        scale = dim_head ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(scores, dim=-1, dtype=torch.float32)
        out = torch.matmul(attn, v)

        # Merge heads
        out = out.permute(0, 2, 1, 3).reshape(B, T, hidden)
        x = torch.nn.functional.linear(out, Wo)

    else:
        # FF layer
        W1 = state_dict[f"{prefix}.1.ff.0.0.weight"]
        b1 = state_dict[f"{prefix}.1.ff.0.0.bias"]
        W2 = state_dict[f"{prefix}.1.ff.2.weight"]
        b2 = state_dict[f"{prefix}.1.ff.2.bias"]

        x = torch.nn.functional.linear(x, W1, b1)
        x = torch.nn.functional.gelu(x)
        x = torch.nn.functional.linear(x, W2, b2)

    # Residual with scale
    rs_key = f"{prefix}.2.residual_scale"
    if rs_key in state_dict:
        rs = state_dict[rs_key]
        residual = residual * rs
    x = x + residual

# Final norm
final_g = state_dict["encoder.final_norm.g"].item()
x = scale_norm(x, final_g, hidden)

enc_ms = (time.time() - t0) * 1000
print(f"  Encoder forward: {enc_ms:.0f}ms")
save_tensor_bin(os.path.join(OUT_DIR, "after_encoder.bin"), x)


# ── Step 5: Low-rank head ────────────────────────────────────────────────

print("\nStep 5: Low-rank head...")
lr_weight = state_dict.get("low_rank_head.weight")
if lr_weight is not None:
    # PyTorch Linear(hidden, lr, bias=False): weight is [lr, hidden]
    x = x.transpose(1, 2)  # [B, hidden, T]
    x = torch.nn.functional.linear(x.transpose(1, 2), lr_weight)  # [B, T, lr]
    x = x.transpose(1, 2)  # [B, lr, T]
    print(f"  low_rank_head: {list(lr_weight.shape)} -> {list(x.shape)}")
    save_tensor_bin(os.path.join(OUT_DIR, "after_lowrank.bin"), x)
else:
    x = x.transpose(1, 2)  # [B, hidden, T]


# ── Step 6: Predictor (SubjectLayers, average_subjects mode) ─────────────

print("\nStep 6: Predictor...")
pred_w = state_dict["predictor.weights"]  # [n_subjects, in_ch, out_ch]
pred_b = state_dict.get("predictor.bias")  # [n_subjects, out_ch]

# average_subjects: use last row (index n_subjects, which is 0 when n_subjects=0+dropout=1 row)
# The saved checkpoint already has averaged weights -> shape [1, in_ch, out_ch]
w = pred_w[0]  # [in_ch, out_ch]
print(f"  weights: {list(pred_w.shape)}, using row 0: {list(w.shape)}")

# x: [B, C, T], w: [C, D] -> [B, D, T]
out = torch.einsum("bct,cd->bdt", x, w)
if pred_b is not None:
    b = pred_b[0]  # [out_ch]
    out = out + b.view(1, -1, 1)
    print(f"  bias: {list(pred_b.shape)}")

print(f"  predictor output: {list(out.shape)}")
save_tensor_bin(os.path.join(OUT_DIR, "after_predictor.bin"), out)


# ── Step 7: Adaptive average pooling ─────────────────────────────────────

print("\nStep 7: Adaptive average pool...")
pooled = torch.nn.functional.adaptive_avg_pool1d(out, n_output_timesteps)
print(f"  pooled: {list(pooled.shape)}")
save_tensor_bin(os.path.join(OUT_DIR, "final_output.bin"), pooled)


# ── Statistics ───────────────────────────────────────────────────────────

print("\n=== Output Statistics ===")
p = pooled.detach().numpy()
print(f"  Shape: {p.shape}")
print(f"  Mean: {p.mean():.8f}")
print(f"  Std:  {p.std():.8f}")
print(f"  Min:  {p.min():.8f}")
print(f"  Max:  {p.max():.8f}")
print(f"  First 10 values: {p.flatten()[:10].tolist()}")

# Per-vertex stats (mean across time)
vertex_means = p[0].mean(axis=-1)
print(f"\n  Per-vertex mean (first 10): {vertex_means[:10].tolist()}")
print(f"  Per-vertex mean (last 10):  {vertex_means[-10:].tolist()}")

# Save summary
summary = {
    "shape": list(p.shape),
    "mean": float(p.mean()),
    "std": float(p.std()),
    "min": float(p.min()),
    "max": float(p.max()),
    "first_10": p.flatten()[:10].tolist(),
    "last_10": p.flatten()[-10:].tolist(),
    "vertex_means_first_10": vertex_means[:10].tolist(),
    "vertex_means_last_10": vertex_means[-10:].tolist(),
    "B": B,
    "T": T,
    "hidden": hidden,
    "n_outputs": n_outputs,
    "n_output_timesteps": n_output_timesteps,
}
with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nAll reference files saved to {OUT_DIR}")
print("Done!")
