#!/usr/bin/env python3
"""Render NIfTI volume slices as PNG for README preview.

Usage:
    python3 scripts/render_nifti_slices.py examples/outputs/predictions.nii.gz examples/outputs/nifti_slices.png
"""

import sys
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def main():
    nii_path = sys.argv[1] if len(sys.argv) > 1 else "examples/outputs/predictions.nii.gz"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "examples/outputs/nifti_slices.png"

    print(f"Loading {nii_path}...")
    img = nib.load(nii_path)
    data = img.get_fdata()
    print(f"  Shape: {data.shape}")
    print(f"  Voxel size: {img.header.get_zooms()}")
    print(f"  Affine:\n{img.affine}")

    # For 4D, take first timestep
    if data.ndim == 4:
        # Average first 5 timesteps for cleaner image
        n_avg = min(5, data.shape[3])
        vol = data[:, :, :, :n_avg].mean(axis=3)
        print(f"  Averaged first {n_avg} timesteps")
    else:
        vol = data

    # Find slices with most signal
    nx, ny, nz = vol.shape
    
    # Pick slices through the center of mass of nonzero voxels
    nonzero = np.abs(vol) > 1e-6
    if nonzero.any():
        coords = np.argwhere(nonzero)
        center = coords.mean(axis=0).astype(int)
    else:
        center = np.array([nx // 2, ny // 2, nz // 2])
    
    sx, sy, sz = center
    print(f"  Center of activity: ({sx}, {sy}, {sz})")

    # Get data range for consistent colormap
    vmax = np.percentile(np.abs(vol[nonzero]) if nonzero.any() else [1], 98)
    vmin = -vmax

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), facecolor="black")
    
    slices = [
        ("Axial (z={})".format(sz), vol[:, :, sz].T, "axial"),
        ("Coronal (y={})".format(sy), vol[:, sy, :].T, "coronal"),
        ("Sagittal (x={})".format(sx), vol[sx, :, :].T, "sagittal"),
    ]

    # Top row: three orthogonal views at center
    cmap = "coolwarm"
    for i, (title, slc, name) in enumerate(slices):
        ax = axes[0, i]
        im = ax.imshow(slc, cmap=cmap, vmin=vmin, vmax=vmax,
                       origin="lower", aspect="equal", interpolation="nearest")
        ax.set_title(title, color="white", fontsize=12, fontweight="bold")
        ax.axis("off")

    # Top row 4th: colorbar
    ax_cb = axes[0, 3]
    ax_cb.axis("off")
    cb = fig.colorbar(im, ax=ax_cb, fraction=0.8, pad=0.05, aspect=20)
    cb.set_label("Predicted BOLD", color="white", fontsize=11)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    # Bottom row: axial slices at different z levels
    z_levels = np.linspace(nz * 0.25, nz * 0.75, 4).astype(int)
    for i, z in enumerate(z_levels):
        ax = axes[1, i]
        slc = vol[:, :, z].T
        ax.imshow(slc, cmap=cmap, vmin=vmin, vmax=vmax,
                  origin="lower", aspect="equal", interpolation="nearest")
        ax.set_title(f"z = {z}", color="white", fontsize=11)
        ax.axis("off")

    fig.suptitle(
        f"NIfTI Volume Output — {nx}×{ny}×{nz} voxels, MNI152 space",
        color="white", fontsize=14, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="black", edgecolor="none")
    print(f"  Saved {out_path}")
    plt.close()

    # Also generate a cleaner 3-slice panel
    out_path2 = out_path.replace(".png", "_3view.png")
    fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4), facecolor="black")
    for i, (title, slc, name) in enumerate(slices):
        ax = axes2[i]
        ax.imshow(slc, cmap=cmap, vmin=vmin, vmax=vmax,
                  origin="lower", aspect="equal", interpolation="nearest")
        ax.set_title(title, color="white", fontsize=13, fontweight="bold")
        ax.axis("off")
    plt.tight_layout()
    fig2.savefig(out_path2, dpi=150, bbox_inches="tight",
                 facecolor="black", edgecolor="none")
    print(f"  Saved {out_path2}")
    plt.close()

if __name__ == "__main__":
    main()
