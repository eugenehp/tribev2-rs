//! Additional unit tests covering config, weights, model construction,
//! plotting, events, features, segments, and fsaverage modules.

use std::collections::BTreeMap;
use tribev2_rs::config::*;
use tribev2_rs::model::tribe::TribeV2;
use tribev2_rs::tensor::Tensor;
use tribev2_rs::plotting;
use tribev2_rs::segments;
use tribev2_rs::events;
use tribev2_rs::features;

// ── Config ────────────────────────────────────────────────────────────────

#[test]
fn test_config_defaults() {
    let cfg = BrainModelConfig {
        hidden: 256,
        max_seq_len: 512,
        extractor_aggregation: "cat".into(),
        layer_aggregation: "cat".into(),
        linear_baseline: false,
        time_pos_embedding: true,
        subject_embedding: false,
        dropout: 0.0,
        modality_dropout: 0.0,
        temporal_dropout: 0.0,
        low_rank_head: None,
        combiner: None,
        temporal_smoothing: None,
        projector: Default::default(),
        encoder: Some(EncoderConfig::default()),
        subject_layers: Some(SubjectLayersConfig::default()),
    };
    assert_eq!(cfg.hidden, 256);
    assert_eq!(cfg.encoder.as_ref().unwrap().depth, 8);
    assert_eq!(cfg.encoder.as_ref().unwrap().heads, 8);
}

#[test]
fn test_encoder_config_dim_head() {
    let enc = EncoderConfig::default();
    assert_eq!(enc.dim_head(1152), 144);
    assert_eq!(enc.rotary_emb_dim(1152), 72);
    assert_eq!(enc.ff_inner_dim(1152), 4608);
}

#[test]
fn test_subject_layers_config_weight_count() {
    let sl = SubjectLayersConfig {
        n_subjects: 10,
        subject_dropout: Some(0.1),
        ..Default::default()
    };
    assert_eq!(sl.num_weight_subjects(), 11);

    let sl2 = SubjectLayersConfig {
        n_subjects: 10,
        subject_dropout: None,
        ..Default::default()
    };
    assert_eq!(sl2.num_weight_subjects(), 10);
}

#[test]
fn test_modality_dims_pretrained() {
    let dims = ModalityDims::pretrained();
    assert_eq!(dims.len(), 3);
    assert_eq!(dims[0].name, "text");
    assert_eq!(dims[0].dims, Some((3, 3072)));
    assert_eq!(dims[1].name, "audio");
    assert_eq!(dims[2].name, "video");
}

#[test]
fn test_modality_dims_none() {
    let md = ModalityDims::none("missing");
    assert!(md.dims.is_none());
    assert_eq!(md.num_layers(), 0);
    assert_eq!(md.feature_dim(), 0);
}

#[test]
fn test_config_yaml_parse() {
    // Minimal config that should parse
    let yaml = r#"
brain_model_config:
  hidden: 64
  encoder:
    depth: 2
    heads: 4
data:
  features_to_use: [text]
  duration_trs: 50
"#;
    let config: Result<TribeV2Config, _> = serde_yaml::from_str(yaml);
    assert!(config.is_ok());
    let cfg = config.unwrap();
    assert_eq!(cfg.brain_model_config.hidden, 64);
    assert_eq!(cfg.data.duration_trs, 50);
}

// ── Model construction ────────────────────────────────────────────────────

#[test]
fn test_model_build_single_modality() {
    let dims = vec![ModalityDims::new("text", 2, 128)];
    let config = BrainModelConfig {
        hidden: 128,
        max_seq_len: 64,
        extractor_aggregation: "cat".into(),
        layer_aggregation: "cat".into(),
        linear_baseline: true,
        time_pos_embedding: false,
        subject_embedding: false,
        dropout: 0.0, modality_dropout: 0.0, temporal_dropout: 0.0,
        low_rank_head: None, combiner: None, temporal_smoothing: None,
        projector: Default::default(),
        encoder: None,
        subject_layers: Some(SubjectLayersConfig {
            n_subjects: 1, bias: false, subject_dropout: None,
            average_subjects: false, ..Default::default()
        }),
    };
    let model = TribeV2::new(dims, 100, 10, &config);
    assert_eq!(model.projectors.len(), 1);
    assert_eq!(model.projectors[0].name, "text");
    assert!(model.encoder.is_none());
    assert!(model.combiner.is_none());
}

#[test]
fn test_model_build_three_modalities() {
    let dims = vec![
        ModalityDims::new("text", 1, 64),
        ModalityDims::new("audio", 1, 32),
        ModalityDims::new("video", 1, 48),
    ];
    let config = BrainModelConfig {
        hidden: 48, // must be divisible by 3 for cat
        max_seq_len: 32,
        extractor_aggregation: "cat".into(),
        layer_aggregation: "cat".into(),
        linear_baseline: true,
        time_pos_embedding: false,
        subject_embedding: false,
        dropout: 0.0, modality_dropout: 0.0, temporal_dropout: 0.0,
        low_rank_head: None, combiner: None, temporal_smoothing: None,
        projector: Default::default(),
        encoder: None,
        subject_layers: Some(SubjectLayersConfig {
            n_subjects: 1, bias: true, subject_dropout: None,
            average_subjects: false, ..Default::default()
        }),
    };
    let model = TribeV2::new(dims, 50, 5, &config);
    assert_eq!(model.projectors.len(), 3);

    let mut features = BTreeMap::new();
    features.insert("text".to_string(), Tensor::zeros(&[1, 64, 8]));
    features.insert("audio".to_string(), Tensor::zeros(&[1, 32, 8]));
    features.insert("video".to_string(), Tensor::zeros(&[1, 48, 8]));

    let output = model.forward(&features, Some(&[0]), true);
    assert_eq!(output.shape, vec![1, 50, 5]);
}

#[test]
fn test_model_with_low_rank_head() {
    let dims = vec![ModalityDims::new("text", 1, 32)];
    let config = BrainModelConfig {
        hidden: 32,
        max_seq_len: 16,
        extractor_aggregation: "cat".into(),
        layer_aggregation: "cat".into(),
        linear_baseline: true,
        time_pos_embedding: false,
        subject_embedding: false,
        dropout: 0.0, modality_dropout: 0.0, temporal_dropout: 0.0,
        low_rank_head: Some(16),
        combiner: None, temporal_smoothing: None,
        projector: Default::default(),
        encoder: None,
        subject_layers: Some(SubjectLayersConfig {
            n_subjects: 1, bias: false, subject_dropout: None,
            average_subjects: false, ..Default::default()
        }),
    };
    let model = TribeV2::new(dims, 20, 4, &config);
    assert!(model.low_rank_head.is_some());

    let mut features = BTreeMap::new();
    features.insert("text".to_string(), Tensor::zeros(&[1, 32, 8]));
    let output = model.forward(&features, Some(&[0]), true);
    assert_eq!(output.shape, vec![1, 20, 4]);
}

#[test]
fn test_model_with_encoder() {
    let dims = vec![ModalityDims::new("text", 1, 64)];
    let config = BrainModelConfig {
        hidden: 64,
        max_seq_len: 32,
        extractor_aggregation: "cat".into(),
        layer_aggregation: "cat".into(),
        linear_baseline: false,
        time_pos_embedding: true,
        subject_embedding: false,
        dropout: 0.0, modality_dropout: 0.0, temporal_dropout: 0.0,
        low_rank_head: None, combiner: None, temporal_smoothing: None,
        projector: Default::default(),
        encoder: Some(EncoderConfig { depth: 1, heads: 2, ff_mult: 2, ..Default::default() }),
        subject_layers: Some(SubjectLayersConfig {
            n_subjects: 1, bias: false, subject_dropout: None,
            average_subjects: false, ..Default::default()
        }),
    };
    let model = TribeV2::new(dims, 10, 3, &config);
    assert!(model.encoder.is_some());
    assert!(model.time_pos_embed.is_some());

    let mut features = BTreeMap::new();
    features.insert("text".to_string(), Tensor::zeros(&[1, 64, 6]));
    let output = model.forward(&features, Some(&[0]), true);
    assert_eq!(output.shape, vec![1, 10, 3]);
}

// ── Tensor ops ────────────────────────────────────────────────────────────

#[test]
fn test_tensor_permute() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 2, 3]);
    let p = t.permute(&[0, 2, 1]);
    assert_eq!(p.shape, vec![1, 3, 2]);
}

#[test]
fn test_tensor_reshape() {
    let t = Tensor::from_vec(vec![1.0; 24], vec![2, 3, 4]);
    let r = t.reshape(&[6, 4]);
    assert_eq!(r.shape, vec![6, 4]);
    assert_eq!(r.data.len(), 24);
}

#[test]
fn test_tensor_add() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    let b = Tensor::from_vec(vec![10.0, 20.0, 30.0], vec![3]);
    let c = a.add(&b);
    assert_eq!(c.data, vec![11.0, 22.0, 33.0]);
}

#[test]
fn test_tensor_einsum_bct_cd_bdt() {
    // Identity weight
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
    let w = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
    let out = x.einsum_bct_cd_bdt(&w);
    assert_eq!(out.data, vec![1.0, 2.0, 3.0, 4.0]);
}

// ── Segments ──────────────────────────────────────────────────────────────

#[test]
fn test_segment_config_default() {
    let cfg = segments::SegmentConfig::default();
    assert_eq!(cfg.duration_trs, 100);
    assert_eq!(cfg.overlap_trs, 0);
    assert_eq!(cfg.tr, 0.5);
    assert!(cfg.remove_empty_segments);
}

#[test]
fn test_predict_segmented_small() {
    let dims = vec![ModalityDims::new("a", 1, 8)];
    let config = BrainModelConfig {
        hidden: 8, max_seq_len: 32,
        extractor_aggregation: "cat".into(), layer_aggregation: "cat".into(),
        linear_baseline: true, time_pos_embedding: false, subject_embedding: false,
        dropout: 0.0, modality_dropout: 0.0, temporal_dropout: 0.0,
        low_rank_head: None, combiner: None, temporal_smoothing: None,
        projector: Default::default(), encoder: None,
        subject_layers: Some(SubjectLayersConfig {
            n_subjects: 1, bias: false, subject_dropout: None,
            average_subjects: false, ..Default::default()
        }),
    };
    let model = TribeV2::new(dims, 4, 5, &config);

    let mut features = BTreeMap::new();
    features.insert("a".to_string(), Tensor::zeros(&[1, 8, 12]));

    let seg_cfg = segments::SegmentConfig {
        duration_trs: 5, overlap_trs: 0,
        remove_empty_segments: false,
        ..Default::default()
    };
    let result = segments::predict_segmented(&model, &features, &seg_cfg);
    assert!(result.predictions.len() > 0);
    assert_eq!(result.predictions[0].len(), 4);
}

// ── Events ────────────────────────────────────────────────────────────────

#[test]
fn test_events_has_audio_video() {
    let mut el = events::EventList::new();
    assert!(!events::has_audio(&el));
    assert!(!events::has_video(&el));
    el.push(events::Event::audio("/tmp/test.wav", 0.0, 5.0));
    assert!(events::has_audio(&el));
    assert!(!events::has_video(&el));
    el.push(events::Event::video("/tmp/test.mp4", 0.0, 5.0));
    assert!(events::has_video(&el));
}

#[test]
fn test_events_get_paths() {
    let mut el = events::EventList::new();
    el.push(events::Event::audio("/tmp/a.wav", 0.0, 1.0));
    el.push(events::Event::video("/tmp/v.mp4", 0.0, 1.0));
    assert_eq!(events::get_audio_path(&el), Some("/tmp/a.wav".into()));
    assert_eq!(events::get_video_path(&el), Some("/tmp/v.mp4".into()));
}

#[test]
fn test_events_get_words_in_range() {
    let el = events::text_to_events("one two three four five", 5.0);
    // 5 words over 5s: starts at 0.0, 1.0, 2.0, 3.0, 4.0
    let words = events::get_words_in_range(&el, 1.5, 3.5, false);
    assert_eq!(words, vec!["three", "four"]);  // words starting at 2.0, 3.0
}

#[test]
fn test_events_get_text_in_range() {
    let el = events::text_to_events("hello beautiful world", 3.0);
    let text = events::get_text_in_range(&el, 0.0, 2.5);
    assert!(text.contains("hello"));
    assert!(text.contains("beautiful"));
}

#[test]
fn test_events_remove_duplicates() {
    let mut el = events::EventList::new();
    el.push(events::Event::word("hi", 0.0, 0.5));
    el.push(events::Event::word("hi", 0.0, 0.5));
    el.push(events::Event::word("bye", 1.0, 0.5));
    assert_eq!(el.events.len(), 3);
    events::remove_duplicate_events(&mut el, &["text", "start"]);
    assert_eq!(el.events.len(), 2);
}

// ── Features ──────────────────────────────────────────────────────────────

#[test]
fn test_features_resample_downsample() {
    let mut f = features::zero_features(1, 2, 8);
    f.data.data = (0..16).map(|i| i as f32).collect();
    let r = features::resample_features(&f, 4);
    assert_eq!(r.n_timesteps, 4);
    assert_eq!(r.data.shape, vec![1, 2, 4]);
}

#[test]
fn test_features_layer_indices_edge() {
    let indices = features::compute_layer_indices(&[0.0, 0.5, 1.0], 10);
    assert_eq!(indices, vec![0, 4, 9]);
}

#[test]
fn test_features_layer_indices_single() {
    let indices = features::compute_layer_indices(&[1.0], 1);
    assert_eq!(indices, vec![0]);
}

// ── Plotting ──────────────────────────────────────────────────────────────

#[test]
fn test_plot_view_names() {
    assert_eq!(plotting::View::Left.name(), "left");
    assert_eq!(plotting::View::MedLeft.name(), "medial_left");
    assert_eq!(plotting::View::Dorsal.name(), "dorsal");
}

#[test]
fn test_plot_view_from_str() {
    assert_eq!(plotting::View::from_str("left"), Some(plotting::View::Left));
    assert_eq!(plotting::View::from_str("dorsal"), Some(plotting::View::Dorsal));
    assert_eq!(plotting::View::from_str("invalid"), None);
}

#[test]
fn test_plot_colormap_range() {
    for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
        let (r, g, b) = plotting::ColorMap::Hot.map(t);
        assert!(r <= 255);
        assert!(g <= 255);
        assert!(b <= 255);
    }
}

#[test]
fn test_plot_pval_stars() {
    assert_eq!(plotting::pval_stars(0.0001), "***");
    assert_eq!(plotting::pval_stars(0.001), "**");
    assert_eq!(plotting::pval_stars(0.01), "*");
    assert_eq!(plotting::pval_stars(0.1), "");
}

#[test]
fn test_plot_saturate_colors() {
    let colors = vec![(0.5, 0.3, 0.1)];
    let saturated = plotting::saturate_colors(&colors, 2.0);
    // Higher saturation should push channels away from gray
    let (r, g, b) = saturated[0];
    assert!(r > 0.5); // red pushed up
    assert!(b < 0.1); // blue pushed down
    let _ = g;
}

#[test]
fn test_plot_rgb_overlay() {
    let r = vec![1.0, 0.0, 0.5];
    let g = vec![0.0, 1.0, 0.5];
    let b = vec![0.0, 0.0, 0.5];
    let result = plotting::rgb_overlay(&[&r, &g, &b], 99.0, 0.0, None);
    assert_eq!(result.len(), 3);
    assert!(result[0].0 > 200); // red channel high for vertex 0
    assert!(result[1].1 > 200); // green channel high for vertex 1
}

#[test]
fn test_plot_render_colorbar_svg() {
    let svg = plotting::render_colorbar_svg(
        plotting::ColorMap::Hot, 0.0, 1.0, 80, 300, Some("R"), "vertical",
    );
    assert!(svg.contains("<svg"));
    assert!(svg.contains("0.00"));
    assert!(svg.contains("1.00"));
}

#[test]
fn test_plot_combine_svgs() {
    let svg1 = r#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50"><rect fill="red"/></svg>"#.to_string();
    let svg2 = svg1.clone();
    let combined = plotting::combine_svgs(&[svg1, svg2], 100, 50, 2, 5);
    assert!(combined.contains("translate("));
    assert!(combined.contains("width=\"205\"")); // 100 + 5 + 100
}

#[test]
fn test_plot_rainbow_brain() {
    let coords: Vec<f32> = (0..30).map(|i| i as f32 * 0.1).collect();
    let colors = plotting::rainbow_brain(&coords, 10);
    assert_eq!(colors.len(), 10);
    for (r, g, b) in &colors {
        assert!(*r <= 255 && *g <= 255 && *b <= 255);
    }
}

#[test]
fn test_plot_tight_crop() {
    // 4x4 image, white background, one red pixel at (1,1)
    let mut img = vec![255u8; 4 * 4 * 3];
    img[1 * 4 * 3 + 1 * 3] = 255; // R
    img[1 * 4 * 3 + 1 * 3 + 1] = 0; // G
    img[1 * 4 * 3 + 1 * 3 + 2] = 0; // B
    let (cropped, w, h) = plotting::tight_crop(&img, 4, 4, 3, (255, 255, 255), 5);
    assert!(w <= 4);
    assert!(h <= 4);
    assert!(cropped.len() > 0);
}

// ── Fsaverage ─────────────────────────────────────────────────────────────

#[test]
fn test_fsaverage_sizes() {
    assert_eq!(tribev2_rs::fsaverage::fsaverage_size("fsaverage5"), Some(10242));
    assert_eq!(tribev2_rs::fsaverage::fsaverage_size("fsaverage3"), Some(642));
    assert_eq!(tribev2_rs::fsaverage::fsaverage_size("nope"), None);
}

// ── Weights ───────────────────────────────────────────────────────────────

#[test]
fn test_weight_map_try_take() {
    // Empty map
    let mut wm = tribev2_rs::WeightMap {
        tensors: std::collections::HashMap::new(),
    };
    assert!(wm.try_take("nonexistent").is_none());
    wm.tensors.insert("test".into(), (vec![1.0, 2.0], vec![2]));
    let t = wm.try_take("test").unwrap();
    assert_eq!(t.data, vec![1.0, 2.0]);
    assert!(wm.try_take("test").is_none()); // consumed
}
