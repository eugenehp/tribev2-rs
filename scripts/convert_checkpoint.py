#!/usr/bin/env python3
"""Convert a TRIBE v2 PyTorch Lightning checkpoint to safetensors.

Requires Python 3.9+ (uses str.removeprefix), torch, and safetensors:
    pip install torch safetensors

Usage:
    python3 scripts/convert_checkpoint.py best.ckpt model.safetensors

Outputs:
    model.safetensors          — weights for the Rust crate
    model_build_args.json      — feature_dims, n_outputs, n_output_timesteps

This extracts the state_dict from the .ckpt file, strips the 'model.' prefix,
and saves in safetensors format for loading by the Rust crate.

It also prints the model_build_args (feature_dims, n_outputs, n_output_timesteps)
which are needed to construct the model in Rust.
"""

import sys
import json
import torch

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.ckpt> <output.safetensors>")
        sys.exit(1)

    ckpt_path = sys.argv[1]
    out_path = sys.argv[2]

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # Extract state dict
    state_dict = ckpt["state_dict"]
    print(f"  state_dict keys: {len(state_dict)}")

    # Strip 'model.' prefix
    sd = {}
    for k, v in state_dict.items():
        new_key = k.removeprefix("model.")
        sd[new_key] = v
        print(f"  {k} -> {new_key}  {list(v.shape)}  {v.dtype}")

    # Extract model build args
    if "model_build_args" in ckpt:
        build_args = ckpt["model_build_args"]
        print(f"\nmodel_build_args:")
        print(f"  feature_dims: {build_args['feature_dims']}")
        print(f"  n_outputs: {build_args['n_outputs']}")
        print(f"  n_output_timesteps: {build_args['n_output_timesteps']}")

        # Save as sidecar JSON
        args_path = out_path.replace(".safetensors", "_build_args.json")
        with open(args_path, "w") as f:
            # Convert feature_dims tuples to lists for JSON
            fd = {}
            for k, v in build_args["feature_dims"].items():
                fd[k] = list(v) if v is not None else None
            json.dump({
                "feature_dims": fd,
                "n_outputs": build_args["n_outputs"],
                "n_output_timesteps": build_args["n_output_timesteps"],
            }, f, indent=2)
        print(f"  Saved build args to {args_path}")

    # Save as safetensors
    import safetensors.torch
    safetensors.torch.save_file(sd, out_path)
    print(f"\nSaved safetensors: {out_path}")

    # Verify
    from safetensors import safe_open
    with safe_open(out_path, framework="pt") as f:
        keys = f.keys()
        print(f"  Verified: {len(keys)} tensors")

if __name__ == "__main__":
    main()
