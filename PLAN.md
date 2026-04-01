# PLAN: Close Output/Prediction Gaps Between Python tribev2 and Rust tribev2-rs

## Overview

The Python `tribev2` codebase provides several output types and analysis capabilities that
the Rust port is missing. This plan adds them in priority order, focusing on features that
are implementable purely within the Rust crate (no external Python/nilearn dependency).

---

## Phase 1: HCP ROI Analysis (output enrichment)
**Files:** `crates/tribev2/src/roi.rs` (new), update `lib.rs`, `bin/infer.rs`

The Python codebase provides `get_hcp_labels`, `summarize_by_roi`, `get_topk_rois`,
`get_hcp_vertex_labels` — all mapping the 20,484 fsaverage5 vertices into the 180 HCP-MMP1
brain regions. This is the single most important analysis output for interpreting predictions.

- [x] **1a.** Create `roi.rs` with:
  - Embedded HCP-MMP1 parcellation data (vertex → ROI label for fsaverage5, both hemispheres)
  - `get_hcp_labels() -> HashMap<String, Vec<usize>>` — ROI name → vertex indices
  - `get_hcp_vertex_labels() -> Vec<String>` — per-vertex ROI label
  - `summarize_by_roi(data) -> HashMap<String, f32>` — average prediction per ROI
  - `get_topk_rois(data, k) -> Vec<(String, f32)>` — top-k activated ROIs
  - `get_roi_indices(roi_pattern) -> Vec<usize>` — wildcard ROI selection (e.g., `"V1*"`)
- [x] **1b.** Add `--roi-summary` flag to CLI: print top-k activated brain regions
- [x] **1c.** Add `--roi-output <path.json>` flag: save per-ROI averages as JSON

## Phase 2: Segment Metadata Output
**Files:** update `segments.rs`, `bin/infer.rs`

Python `predict()` returns `(preds, segments)` — each prediction is paired with its temporal
segment including event info. The Rust version discards segment info in the CLI.

- [x] **2a.** Add `--segments-output <path.json>` flag to CLI
- [x] **2b.** Serialize kept segments as JSON array:
  ```json
  [{"start": 0.0, "duration": 0.5, "has_events": true, "timestep_index": 0}, ...]
  ```

## Phase 3: Evaluation Metrics
**Files:** `crates/tribev2/src/metrics.rs` (new), update `lib.rs`, `bin/infer.rs`

Python computes Pearson correlation, retrieval top-k accuracy, and per-subject grouped metrics.
Critical for validating predictions against ground-truth fMRI.

- [x] **3a.** Create `metrics.rs` with:
  - `pearson_correlation(pred, true) -> f32` — per-vertex Pearson r, averaged
  - `pearson_per_vertex(pred, true) -> Vec<f32>` — per-vertex Pearson r map
  - `topk_accuracy(pred, true, k) -> f32` — retrieval accuracy
  - `mse(pred, true) -> f32` — mean squared error
- [x] **3b.** Add `--ground-truth <path.bin>` flag to CLI: load ground-truth fMRI data
- [x] **3c.** When ground truth is provided, print metrics + optionally save correlation map

## Phase 4: Subcortical Prediction Support
**Files:** `crates/tribev2/src/subcortical.rs` (new), update `lib.rs`, `bin/infer.rs`

Python has a full subcortical pipeline (`run_subcortical.py`) using the Harvard-Oxford atlas
mask. This projects predictions to subcortical structures (hippocampus, amygdala, thalamus, etc.).

- [x] **4a.** Create `subcortical.rs` with:
  - Embedded Harvard-Oxford subcortical ROI definitions (name → voxel index ranges)
  - `SubcorticalConfig` — atlas resolution, structure selection
  - `get_subcortical_labels() -> Vec<String>` — list of subcortical structures
  - `get_subcortical_roi_indices(roi) -> Vec<usize>` — voxel indices for a structure
  - `summarize_subcortical(predictions, voxel_scores) -> HashMap<String, f32>`
- [x] **4b.** Add `--subcortical` flag to CLI: report subcortical structure activations
- [x] **4c.** Document that full subcortical model requires a separately trained checkpoint

## Phase 5: MP4 Video Output
**Files:** `crates/tribev2/src/video_output.rs` (new), update `lib.rs`, `bin/infer.rs`

Python generates animated brain activity videos via `plot_timesteps_mp4()` using ffmpeg.
Since we already generate per-timestep SVGs, we can pipe them to ffmpeg.

- [x] **5a.** Create `video_output.rs` with:
  - `render_mp4(predictions, brain, config, output_path)` — calls ffmpeg on temp PNGs/SVGs
  - Support for frame rate, interpolation, title overlay
- [x] **5b.** Add `--mp4 <path.mp4>` flag to CLI
- [x] **5c.** Require `resvg` or `rsvg-convert` for SVG→PNG, then ffmpeg for MP4

## Phase 6: Per-Modality Contribution Maps
**Files:** update `model/tribe.rs`, add to `bin/infer.rs`

Python's `plot_surf_rgb()` renders R/G/B channels for text/audio/video contributions.
This requires running inference with each modality zeroed out to measure contribution.

- [x] **6a.** Add `forward_modality_ablation()` to `TribeV2`:
  - Run full forward, then re-run with each modality zeroed → difference = contribution
- [x] **6b.** Add `--modality-maps <dir>` flag to CLI: save per-modality contribution SVGs
- [x] **6c.** Output per-modality contribution as separate vertex arrays

## Phase 7: Mesh Resampling
**Files:** `crates/tribev2/src/resample.rs` (new), update `lib.rs`

Python supports cross-resolution resampling between fsaverage meshes using kd-tree
interpolation (`get_stat_map()` in `base.py`).

- [x] **7a.** Create `resample.rs` with:
  - `resample_surface(data, from_mesh, to_mesh) -> Vec<f32>` — kd-tree nearest-neighbor
  - Support fsaverage3 through fsaverage6
- [x] **7b.** Add `--output-mesh <fsaverageN>` flag to CLI for resampled output

---

## Execution Order

Phases 1-3 are pure computation, no external deps. Phase 4-5 need embedded atlas data
and ffmpeg. Phase 6-7 are model/mesh extensions.

Total estimated: ~2000 lines of new Rust code across 5 new files + CLI updates.

---

## Execution Status

**All 7 phases completed.** ✅

| Phase | Files | Tests | Status |
|-------|-------|-------|--------|
| 1. HCP ROI Analysis | `roi.rs` (276 lines) | 7 tests | ✅ |
| 2. Segment Metadata | `bin/infer.rs` update | — | ✅ |
| 3. Evaluation Metrics | `metrics.rs` (229 lines) | 7 tests | ✅ |
| 4. Subcortical Support | `subcortical.rs` (223 lines) | 4 tests | ✅ |
| 5. MP4 Video Output | `video_output.rs` (218 lines) | 2 tests | ✅ |
| 6. Modality Contribution | `model/tribe.rs` update | — | ✅ |
| 7. Mesh Resampling | `resample.rs` (258 lines) | 3 tests | ✅ |

**Total: 5 new modules, 66 passing tests, 0 warnings.**

### New CLI flags

```
--roi-summary <k>           Print top-k activated brain regions (HCP-MMP1)
--roi-output <path.json>    Save per-ROI averages as JSON
--hcp-annot-dir <dir>       Path to HCP annotation files (for exact labels)
--segments-output <path>    Save segment metadata as JSON
--ground-truth <path.bin>   Evaluate against ground-truth fMRI data
--topk <k>                  Top-k for retrieval accuracy (default: 1)
--correlation-map <path>    Save per-vertex Pearson r map (binary f32)
--subcortical               Show subcortical structure activations
--mp4 <path.mp4>            Generate animated brain activity video
--video-fps <n>             Video frame rate (default: 2)
--modality-maps <dir>       Per-modality contribution maps via ablation
--output-mesh <fsaverageN>  Resample to different fsaverage resolution
```
