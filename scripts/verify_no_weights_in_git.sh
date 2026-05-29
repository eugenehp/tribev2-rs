#!/usr/bin/env bash
# Fail if weight files are tracked by git (run in CI or before push).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

bad=$(git ls-files | rg -i '\.(safetensors|ckpt|gguf|pt|pth)(\.|$)|(^|/)data/parity_refs/|model\.safetensors' || true)
if [[ -n "$bad" ]]; then
  echo "ERROR: weight or large data files are tracked by git:" >&2
  echo "$bad" >&2
  echo "Run: git rm --cached <path>  and ensure .gitignore covers them." >&2
  exit 1
fi
echo "OK: no weight files in git index"
