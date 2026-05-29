#!/usr/bin/env python3
"""Print Windows vs WSL CUDA benchmark comparison from bench/rig/<tag>/*.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load_means(tag_dir: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for path in sorted(tag_dir.glob("results_*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        for _key, rec in data.items():
            if not isinstance(rec, dict):
                continue
            engine = rec.get("engine", "")
            device = rec.get("device", "")
            mean = rec.get("mean_ms")
            if mean is None:
                continue
            label = f"{engine}/{device}" if device else str(_key)
            out[label] = float(mean)
    return out


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    rig_dir = root / "bench" / "rig"
    if len(sys.argv) > 1:
        tags = [sys.argv[1]]
    elif rig_dir.is_dir():
        # e.g. 20260529_windows + 20260529_wsl → tag 20260529
        seen: set[str] = set()
        for p in rig_dir.iterdir():
            if not p.is_dir():
                continue
            name = p.name
            for suffix in ("_windows", "_wsl"):
                if name.endswith(suffix):
                    seen.add(name[: -len(suffix)])
                    break
            else:
                seen.add(name)
        tags = sorted(seen)
    else:
        tags = []

    if not tags:
        print("No bench/rig/<tag>/ directories. Run: ./rig.sh --both bench-cuda && ./rig.sh fetch-bench", file=sys.stderr)
        return 1

    rows: list[tuple[str, str, float | None, float | None, str]] = []
    keys = ("rust/cpu", "rlx/cpu", "rlx/cuda", "rlx/wgpu")

    for tag in tags:
        win = load_means(rig_dir / f"{tag}_windows") if (rig_dir / f"{tag}_windows").is_dir() else {}
        wsl = load_means(rig_dir / f"{tag}_wsl") if (rig_dir / f"{tag}_wsl").is_dir() else {}
        # Also accept flat tag dir (single-runtime runs)
        flat = load_means(rig_dir / tag) if (rig_dir / tag).is_dir() else {}
        if flat and not win and not wsl:
            win = flat

        for k in keys:
            wm = win.get(k)
            zm = wsl.get(k)
            if wm is None and zm is None:
                continue
            speedup = ""
            if wm and zm and zm > 0:
                speedup = f"{wm / zm:.2f}x" if wm > zm else f"{zm / wm:.2f}x (wsl faster)"
            rows.append((tag, k, wm, zm, speedup))

    if not rows:
        print("No timing records found under bench/rig/", file=sys.stderr)
        return 1

    print(f"{'tag':<12} {'backend':<14} {'windows_ms':>12} {'wsl_ms':>12} {'notes':<20}")
    print("-" * 74)
    for tag, k, wm, zm, note in rows:
        ws = f"{wm:.1f}" if wm is not None else "—"
        zs = f"{zm:.1f}" if zm is not None else "—"
        print(f"{tag:<12} {k:<14} {ws:>12} {zs:>12} {note:<20}")

    # CUDA-specific summary
    print()
    print("CUDA summary (lower ms = faster forward):")
    for tag in tags:
        win_cuda = load_means(rig_dir / f"{tag}_windows").get("rlx/cuda")
        wsl_cuda = load_means(rig_dir / f"{tag}_wsl").get("rlx/cuda")
        if win_cuda is None and wsl_cuda is None:
            win_cuda = load_means(rig_dir / tag).get("rlx/cuda")
        if win_cuda is None and wsl_cuda is None:
            continue
        ws = f"{win_cuda:.1f} ms" if win_cuda is not None else "n/a"
        zs = f"{wsl_cuda:.1f} ms" if wsl_cuda is not None else "n/a"
        print(f"  [{tag}] Windows RLX CUDA: {ws}  |  WSL RLX CUDA: {zs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
