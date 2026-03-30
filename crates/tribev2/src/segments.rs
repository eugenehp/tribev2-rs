//! Segment-based batching for TRIBE v2 inference.
//!
//! Mirrors the Python `TribeModel.predict()` pipeline:
//! 1. Split input features into segments of `duration_trs` timesteps
//! 2. Apply optional overlap between segments
//! 3. Run inference per segment
//! 4. Reassemble predictions, removing empty segments
//!
//! Python (demo_utils.py):
//! ```python
//! for batch in loader:
//!     batch_segments = []
//!     for segment in batch.segments:
//!         for t in np.arange(0, segment.duration - 1e-2, data.TR):
//!             batch_segments.append(segment.copy(offset=t, duration=data.TR))
//!     if self.remove_empty_segments:
//!         keep = np.array([len(s.ns_events) > 0 for s in batch_segments])
//!     y_pred = model(batch).detach().cpu().numpy()
//!     y_pred = rearrange(y_pred, 'b d t -> (b t) d')[keep]
//! ```

use std::collections::BTreeMap;
use crate::tensor::Tensor;
use crate::model::tribe::TribeV2;

/// A segment of features with temporal metadata.
#[derive(Debug, Clone)]
pub struct Segment {
    /// Start time in seconds.
    pub start: f64,
    /// Duration in seconds.
    pub duration: f64,
    /// Whether this segment has any events (non-zero features).
    pub has_events: bool,
}

/// Configuration for segment-based inference.
#[derive(Debug, Clone)]
pub struct SegmentConfig {
    /// Number of output timesteps per segment (duration_trs).
    pub duration_trs: usize,
    /// Number of overlapping timesteps for training (overlap_trs).
    pub overlap_trs: usize,
    /// TR (repetition time) in seconds. Default: 0.5 (at 2Hz).
    pub tr: f64,
    /// Whether to remove segments with all-zero features.
    pub remove_empty_segments: bool,
    /// Feature frequency in Hz (for converting between time and timesteps).
    pub feature_frequency: f64,
    /// Whether to drop incomplete final segments.
    pub stride_drop_incomplete: bool,
}

impl Default for SegmentConfig {
    fn default() -> Self {
        Self {
            duration_trs: 100,
            overlap_trs: 0,
            tr: 0.5,
            remove_empty_segments: true,
            feature_frequency: 2.0,
            stride_drop_incomplete: false,
        }
    }
}

/// Result of segment-based inference.
#[derive(Debug)]
pub struct SegmentedPrediction {
    /// Predictions: [n_kept_segments, n_outputs].
    /// Each row is one TR's worth of prediction (after pool → unravel).
    pub predictions: Vec<Vec<f32>>,
    /// The segments that were kept (non-empty).
    pub segments: Vec<Segment>,
    /// Total number of segments before filtering.
    pub total_segments: usize,
    /// Number of segments kept.
    pub kept_segments: usize,
}

/// Compute segment boundaries for a given total number of timesteps.
///
/// Returns a list of (start_timestep, end_timestep) pairs.
pub fn compute_segment_boundaries(
    total_timesteps: usize,
    config: &SegmentConfig,
) -> Vec<(usize, usize)> {
    let stride = config.duration_trs.saturating_sub(config.overlap_trs).max(1);
    let mut segments = Vec::new();
    let mut start = 0;

    while start < total_timesteps {
        let end = (start + config.duration_trs).min(total_timesteps);
        if config.stride_drop_incomplete && (end - start) < config.duration_trs {
            break;
        }
        segments.push((start, end));
        start += stride;
        if end >= total_timesteps {
            break;
        }
    }

    segments
}

/// Check if a feature tensor slice has any non-zero values.
fn has_nonzero_features(features: &BTreeMap<String, Tensor>, start: usize, end: usize) -> bool {
    for tensor in features.values() {
        let t_dim = *tensor.shape.last().unwrap();
        let s = start.min(t_dim);
        let e = end.min(t_dim);
        if s >= e {
            continue;
        }
        // Check if any value in the time range [s, e) is non-zero
        let batch_size: usize = tensor.shape[..tensor.shape.len() - 1].iter().product();
        for bi in 0..batch_size {
            for ti in s..e {
                let idx = bi * t_dim + ti;
                if idx < tensor.data.len() && tensor.data[idx] != 0.0 {
                    return true;
                }
            }
        }
    }
    false
}

/// Slice features along the time dimension.
///
/// Input features: map of name → [B, ..., T]
/// Returns features sliced to [B, ..., end-start] (or zero-padded if end > T).
fn slice_features(
    features: &BTreeMap<String, Tensor>,
    start: usize,
    end: usize,
) -> BTreeMap<String, Tensor> {
    let segment_len = end - start;
    let mut sliced = BTreeMap::new();

    for (name, tensor) in features {
        let t_dim = *tensor.shape.last().unwrap();
        let ndim = tensor.ndim();

        // Compute batch dimensions (everything except last dim)
        let batch_shape = &tensor.shape[..ndim - 1];
        let batch_size: usize = batch_shape.iter().product();

        let mut new_data = vec![0.0f32; batch_size * segment_len];

        let copy_start = start.min(t_dim);
        let copy_end = end.min(t_dim);
        let copy_len = copy_end.saturating_sub(copy_start);

        if copy_len > 0 {
            for bi in 0..batch_size {
                for ti in 0..copy_len {
                    new_data[bi * segment_len + ti] =
                        tensor.data[bi * t_dim + copy_start + ti];
                }
            }
        }

        let mut new_shape = batch_shape.to_vec();
        new_shape.push(segment_len);
        sliced.insert(name.clone(), Tensor::from_vec(new_data, new_shape));
    }

    sliced
}

/// Run segment-based inference on the model.
///
/// `features`: map from modality name → tensor [1, L*D, T] (batch=1, concatenated layers,
///   T = total timesteps). Features should already be at the model's expected frequency.
/// `model`: the TRIBE v2 model.
/// `config`: segment configuration.
///
/// Returns per-TR predictions as [n_kept_trs, n_outputs].
pub fn predict_segmented(
    model: &TribeV2,
    features: &BTreeMap<String, Tensor>,
    config: &SegmentConfig,
) -> SegmentedPrediction {
    // Determine total timesteps from first feature tensor
    let total_timesteps = features
        .values()
        .next()
        .map(|t| *t.shape.last().unwrap())
        .unwrap_or(0);

    let boundaries = compute_segment_boundaries(total_timesteps, config);
    let mut all_predictions: Vec<Vec<f32>> = Vec::new();
    let mut kept_segments: Vec<Segment> = Vec::new();
    let mut total_trs = 0usize;
    let mut kept_trs = 0usize;

    for (start, end) in &boundaries {
        let segment_len = end - start;

        // Check if segment is empty
        let _has_events = has_nonzero_features(features, *start, *end);

        // Slice features for this segment
        let seg_features = slice_features(features, *start, *end);

        // Run model forward pass (pool_outputs = true)
        let output = model.forward(&seg_features, None, true);
        // output: [1, n_outputs, n_output_timesteps]

        let n_outputs = output.shape[1];
        let n_out_ts = output.shape[2];

        // Unravel to per-TR predictions: [n_out_ts, n_outputs]
        for ti in 0..n_out_ts {
            total_trs += 1;

            // Check if this TR should be kept
            let keep = if config.remove_empty_segments {
                // Check if the corresponding input timestep range has events
                let input_start = *start + (ti * segment_len) / n_out_ts;
                let input_end = *start + ((ti + 1) * segment_len) / n_out_ts;
                has_nonzero_features(features, input_start, input_end.max(input_start + 1))
            } else {
                true
            };

            if keep || !config.remove_empty_segments {
                let mut row = Vec::with_capacity(n_outputs);
                for di in 0..n_outputs {
                    row.push(output.data[di * n_out_ts + ti]);
                }
                all_predictions.push(row);
                kept_segments.push(Segment {
                    start: (*start + ti) as f64 * config.tr,
                    duration: config.tr,
                    has_events: keep,
                });
                kept_trs += 1;
            }
        }
    }

    SegmentedPrediction {
        predictions: all_predictions,
        segments: kept_segments,
        total_segments: total_trs,
        kept_segments: kept_trs,
    }
}

/// Convenience: run segment-based inference without per-TR unraveling.
///
/// Returns predictions as [n_segments, n_outputs, n_output_timesteps],
/// one entry per segment (not per TR).
pub fn predict_segments_batched(
    model: &TribeV2,
    features: &BTreeMap<String, Tensor>,
    config: &SegmentConfig,
) -> Vec<(Tensor, Segment)> {
    let total_timesteps = features
        .values()
        .next()
        .map(|t| *t.shape.last().unwrap())
        .unwrap_or(0);

    let boundaries = compute_segment_boundaries(total_timesteps, config);
    let mut results = Vec::new();

    for (start, end) in &boundaries {
        let segment_len = end - start;
        let has_events = has_nonzero_features(features, *start, *end);

        if config.remove_empty_segments && !has_events {
            continue;
        }

        let segment = Segment {
            start: *start as f64 * config.tr,
            duration: segment_len as f64 * config.tr,
            has_events,
        };

        let seg_features = slice_features(features, *start, *end);
        let output = model.forward(&seg_features, None, true);
        results.push((output, segment));
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_segment_boundaries_no_overlap() {
        let config = SegmentConfig {
            duration_trs: 10,
            overlap_trs: 0,
            ..Default::default()
        };
        let segs = compute_segment_boundaries(25, &config);
        assert_eq!(segs, vec![(0, 10), (10, 20), (20, 25)]);
    }

    #[test]
    fn test_compute_segment_boundaries_with_overlap() {
        let config = SegmentConfig {
            duration_trs: 10,
            overlap_trs: 5,
            ..Default::default()
        };
        let segs = compute_segment_boundaries(20, &config);
        assert_eq!(segs, vec![(0, 10), (5, 15), (10, 20)]);
    }

    #[test]
    fn test_compute_segment_boundaries_drop_incomplete() {
        let config = SegmentConfig {
            duration_trs: 10,
            overlap_trs: 0,
            stride_drop_incomplete: true,
            ..Default::default()
        };
        let segs = compute_segment_boundaries(25, &config);
        assert_eq!(segs, vec![(0, 10), (10, 20)]);
    }

    #[test]
    fn test_compute_segment_boundaries_exact() {
        let config = SegmentConfig {
            duration_trs: 10,
            overlap_trs: 0,
            ..Default::default()
        };
        let segs = compute_segment_boundaries(20, &config);
        assert_eq!(segs, vec![(0, 10), (10, 20)]);
    }

    #[test]
    fn test_slice_features() {
        let mut features = BTreeMap::new();
        let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
        features.insert("text".to_string(), Tensor::from_vec(data, vec![2, 10]));

        let sliced = slice_features(&features, 3, 7);
        let t = sliced.get("text").unwrap();
        assert_eq!(t.shape, vec![2, 4]);
        // Row 0: [3, 4, 5, 6]
        assert_eq!(t.data[0], 3.0);
        assert_eq!(t.data[1], 4.0);
        assert_eq!(t.data[2], 5.0);
        assert_eq!(t.data[3], 6.0);
        // Row 1: [13, 14, 15, 16]
        assert_eq!(t.data[4], 13.0);
        assert_eq!(t.data[5], 14.0);
    }

    #[test]
    fn test_slice_features_with_padding() {
        let mut features = BTreeMap::new();
        let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
        features.insert("a".to_string(), Tensor::from_vec(data, vec![2, 3]));

        let sliced = slice_features(&features, 2, 5);
        let t = sliced.get("a").unwrap();
        assert_eq!(t.shape, vec![2, 3]);
        // Row 0: [2, 0, 0] (only index 2 exists, rest is zero-padded)
        assert_eq!(t.data[0], 2.0);
        assert_eq!(t.data[1], 0.0);
        assert_eq!(t.data[2], 0.0);
    }
}
