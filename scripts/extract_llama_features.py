#!/usr/bin/env python3
"""Extract per-layer LLaMA hidden states for TRIBE v2.

Produces binary f32 feature files with true intermediate layer activations,
matching exactly what the Python TRIBE v2 pipeline produces.

Usage:
    # Extract from a text file
    python scripts/extract_llama_features.py \
        --model meta-llama/Llama-3.2-3B \
        --input words.txt \
        --output text_features.bin \
        --layers 0.5 0.75 1.0

    # Extract from whisper transcript (word-level timing)
    python scripts/extract_llama_features.py \
        --model meta-llama/Llama-3.2-3B \
        --input transcript.json \
        --output text_features.bin \
        --layers 0.5 0.75 1.0 \
        --frequency 2.0 \
        --duration 60.0

Output format:
    Binary f32, shape [n_layers, hidden_dim, n_timesteps], C-contiguous.
    A sidecar .json is also written with shape metadata.

    Load in Rust with `--text-features text_features.bin --n-layers 3 --feature-dim 3072`
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def compute_layer_indices(layer_positions, n_total_layers):
    """Map fractional positions to layer indices."""
    return [min(int(f * (n_total_layers - 1)), n_total_layers - 1) for f in layer_positions]


def extract_features_from_text(
    model, tokenizer, text, layer_indices, device="cpu"
):
    """Extract per-token hidden states from selected layers.

    Returns dict of layer_idx -> [n_tokens, hidden_dim] numpy arrays.
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # outputs.hidden_states: tuple of (n_layers+1) tensors, each [1, seq_len, hidden_dim]
    # Index 0 = embedding layer, index i = output of layer i
    hidden_states = {}
    for li in layer_indices:
        # +1 because index 0 is the embedding layer output
        hs = outputs.hidden_states[li + 1][0].cpu().numpy()  # [seq_len, hidden_dim]
        hidden_states[li] = hs

    return hidden_states


def extract_timed_features(
    model, tokenizer, words_with_times, total_duration,
    layer_indices, frequency=2.0, device="cpu", context_len=1024
):
    """Extract features with temporal alignment.

    words_with_times: list of (word, start_time_sec)
    Returns: [n_layer_groups, hidden_dim, n_timesteps] numpy array
    """
    n_timesteps = int(np.ceil(total_duration * frequency))
    hidden_dim = model.config.hidden_size
    n_layers = len(layer_indices)

    # Build contextualized text and tokenize
    full_text = " ".join(w for w, _ in words_with_times)
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True,
                       max_length=context_len).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    n_tokens = inputs.input_ids.shape[1]
    n_words = len(words_with_times)

    # Map words to token ranges (approximate)
    tokens_per_word = max(1, (n_tokens - 1)) / max(1, n_words)

    # Build per-word embeddings for each layer
    word_embeddings = {}  # layer_idx -> [n_words, hidden_dim]
    for li in layer_indices:
        hs = outputs.hidden_states[li + 1][0].cpu().numpy()  # [seq_len, hidden_dim]
        word_embs = []
        for wi in range(n_words):
            tok_start = 1 + int(wi * tokens_per_word)
            tok_end = min(1 + int((wi + 1) * tokens_per_word), n_tokens)
            tok_end = max(tok_end, tok_start + 1)
            word_embs.append(hs[tok_start:tok_end].mean(axis=0))
        word_embeddings[li] = np.array(word_embs)  # [n_words, hidden_dim]

    # Temporal alignment: for each output timestep, find the last word before it
    dt = 1.0 / frequency
    data = np.zeros((n_layers, hidden_dim, n_timesteps), dtype=np.float32)

    for ti in range(n_timesteps):
        t = ti * dt
        # Find last word starting at or before time t
        wi = -1
        for j, (_, start_time) in enumerate(words_with_times):
            if start_time <= t:
                wi = j
        if wi < 0 and n_words > 0:
            wi = 0

        if wi >= 0:
            for li_idx, li in enumerate(layer_indices):
                data[li_idx, :, ti] = word_embeddings[li][wi]

    return data


def main():
    parser = argparse.ArgumentParser(description="Extract LLaMA per-layer features for TRIBE v2")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B",
                        help="HuggingFace model name or path")
    parser.add_argument("--input", required=True,
                        help="Input text file (.txt) or whisper transcript (.json)")
    parser.add_argument("--output", required=True,
                        help="Output binary f32 file")
    parser.add_argument("--layers", nargs="+", type=float, default=[0.5, 0.75, 1.0],
                        help="Layer positions (fractional, 0.0-1.0)")
    parser.add_argument("--frequency", type=float, default=2.0,
                        help="Output feature frequency in Hz")
    parser.add_argument("--duration", type=float, default=None,
                        help="Total duration in seconds (for timed features)")
    parser.add_argument("--device", default="auto",
                        help="Device: cpu, cuda, mps, auto")
    parser.add_argument("--context-len", type=int, default=2048,
                        help="Max context length for tokenization")
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Model dtype")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    # Dtype
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    print(f"Loading model: {args.model} ({args.dtype}) on {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=device,
        output_hidden_states=True
    )
    model.eval()

    n_total_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    layer_indices = compute_layer_indices(args.layers, n_total_layers)
    n_layer_groups = len(layer_indices)

    print(f"Model: {n_total_layers} layers, hidden_dim={hidden_dim}")
    print(f"Extracting layers: {layer_indices} (from positions {args.layers})")

    input_path = Path(args.input)

    if input_path.suffix == ".json":
        # Whisper transcript with timing
        with open(input_path) as f:
            transcript = json.load(f)

        words_with_times = []
        for segment in transcript.get("segments", []):
            for word in segment.get("words", []):
                if "start" in word:
                    words_with_times.append((word["word"].strip(), word["start"]))

        duration = args.duration
        if duration is None:
            # Estimate from last word
            if words_with_times:
                duration = words_with_times[-1][1] + 2.0
            else:
                duration = 10.0

        print(f"Transcript: {len(words_with_times)} words, duration={duration:.1f}s")

        data = extract_timed_features(
            model, tokenizer, words_with_times, duration,
            layer_indices, args.frequency, device, args.context_len
        )
    else:
        # Plain text file
        text = input_path.read_text(encoding="utf-8").strip()
        if not text:
            print("Error: empty input file", file=sys.stderr)
            sys.exit(1)

        print(f"Text: {len(text.split())} words")

        hidden_states = extract_features_from_text(
            model, tokenizer, text, layer_indices, device
        )

        # Build output: [n_layers, hidden_dim, n_tokens]
        n_tokens = next(iter(hidden_states.values())).shape[0]

        if args.duration is not None:
            n_timesteps = int(np.ceil(args.duration * args.frequency))
        else:
            n_timesteps = n_tokens

        data = np.zeros((n_layer_groups, hidden_dim, n_timesteps), dtype=np.float32)
        for li_idx, li in enumerate(layer_indices):
            hs = hidden_states[li]  # [n_tokens, hidden_dim]
            for ti in range(n_timesteps):
                src = min(int(ti * n_tokens / n_timesteps), n_tokens - 1)
                data[li_idx, :, ti] = hs[src]

    # Save binary f32
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    flat = data.astype(np.float32).tobytes()
    with open(output_path, "wb") as f:
        f.write(flat)

    # Save metadata
    meta = {
        "shape": list(data.shape),
        "n_layers": n_layer_groups,
        "hidden_dim": hidden_dim,
        "n_timesteps": data.shape[2],
        "layer_positions": args.layers,
        "layer_indices": layer_indices,
        "model": args.model,
        "frequency": args.frequency,
        "dtype": "float32",
    }
    meta_path = output_path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved: {output_path} ({data.shape}, {len(flat)} bytes)")
    print(f"Metadata: {meta_path}")
    print(f"\nUse in Rust:")
    print(f"  --text-features {output_path} --n-layers {n_layer_groups} --feature-dim {hidden_dim}")


if __name__ == "__main__":
    main()
