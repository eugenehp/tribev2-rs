# Data directory (not in git except this file + small config)

**Weights are never committed.** Download them locally:

```bash
export TRIBEV2_DATA_DIR="$(pwd)/data"

# HuggingFace (recommended)
cargo run --bin tribev2-download --features hf-download -- \
  --repo eugenehp/tribev2 --output ./data

# Or set TRIBEV2_DATA_DIR to an existing checkout that already has:
#   model.safetensors   (~700 MB)
#   config.yaml
#   build_args.json
```

## Expected layout

| File | In git? | Notes |
|------|---------|--------|
| `config.yaml` | yes | Model hyperparameters |
| `build_args.json` | yes | Feature dims / output shape |
| `model.safetensors` | **no** | Pretrained weights — download or convert from `.ckpt` |
| `parity_refs/` | **no** | Regenerate: `python3 scripts/generate_parity_refs.py` |
| `fsaverage5/` | **no** | FreeSurfer surface (set `FREESURFER_SUBJECTS_DIR` or download) |

Convert PyTorch checkpoint to safetensors:

```bash
python3 scripts/convert_checkpoint.py best.ckpt data/model.safetensors
```
