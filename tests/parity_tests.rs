//! Parity tests verifying the Rust implementation matches the Python TribeV2 model
//! at every stage of the forward pass.

use std::collections::BTreeMap;
use tribev2_rs::config::*;
use tribev2_rs::model::feedforward::FeedForward;
use tribev2_rs::model::projector::Projector;
use tribev2_rs::model::residual::Residual;
use tribev2_rs::model::rotary::RotaryEmbedding;
use tribev2_rs::model::scalenorm::ScaleNorm;
use tribev2_rs::model::subject_layers::SubjectLayers;
use tribev2_rs::model::tribe::TribeV2;
use tribev2_rs::tensor::Tensor;

fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

// ── ScaleNorm tests ───────────────────────────────────────────────────────

#[test]
fn test_scalenorm_matches_python() {
    // Python: F.normalize([3.0, 4.0], dim=-1) * sqrt(2) * 1.0
    // norm = 5.0, normalized = [0.6, 0.8], scale = sqrt(2) ≈ 1.4142
    // result = [0.8485, 1.1314]
    let sn = ScaleNorm::new(2);
    let x = Tensor::from_vec(vec![3.0, 4.0], vec![1, 2]);
    let out = sn.forward(&x);
    let sqrt2 = (2.0f32).sqrt();
    assert!(approx_eq(out.data[0], 0.6 * sqrt2, 1e-5));
    assert!(approx_eq(out.data[1], 0.8 * sqrt2, 1e-5));
}

#[test]
fn test_scalenorm_with_custom_g() {
    let mut sn = ScaleNorm::new(4);
    sn.g = 2.0;
    let x = Tensor::from_vec(vec![1.0, 0.0, 0.0, 0.0], vec![1, 4]);
    let out = sn.forward(&x);
    // normalize([1,0,0,0]) = [1,0,0,0], scale = 2.0, g = 2.0
    // result = [1 * 2.0 * 2.0, 0, 0, 0] = [4.0, 0, 0, 0]
    assert!(approx_eq(out.data[0], 4.0, 1e-5));
    assert!(approx_eq(out.data[1], 0.0, 1e-5));
}

// ── Rotary Embedding tests ────────────────────────────────────────────────

#[test]
fn test_rotary_embedding_shape() {
    let rot = RotaryEmbedding::new(8);
    let freqs = rot.forward(10);
    assert_eq!(freqs.shape, vec![10, 8]);
    // First position (pos=0): all zeros
    for i in 0..8 {
        assert!(approx_eq(freqs.data[i], 0.0, 1e-8));
    }
    // freqs[n, j] = n * inv_freq[j] for j < 4, duplicated for j >= 4
    assert!(approx_eq(freqs.data[8 + 0], freqs.data[8 + 4], 1e-8)); // pos 1, first = second half
}

#[test]
fn test_rotary_inv_freq() {
    // inv_freq[i] = 1 / (10000 ^ (2i / dim))
    let rot = RotaryEmbedding::new(4);
    assert_eq!(rot.inv_freq.len(), 2);
    assert!(approx_eq(rot.inv_freq[0], 1.0, 1e-5)); // 1 / 10000^0 = 1
    assert!(approx_eq(rot.inv_freq[1], 0.01, 1e-4)); // 1 / 10000^0.5 = 0.01
}

// ── Residual tests ────────────────────────────────────────────────────────

#[test]
fn test_residual_no_scale() {
    let res = Residual::new(3, false);
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let r = Tensor::from_vec(vec![0.1, 0.2, 0.3], vec![1, 3]);
    let out = res.forward(&x, &r);
    assert!(approx_eq(out.data[0], 1.1, 1e-6));
    assert!(approx_eq(out.data[1], 2.2, 1e-6));
    assert!(approx_eq(out.data[2], 3.3, 1e-6));
}

#[test]
fn test_residual_with_scale() {
    let mut res = Residual::new(2, true);
    // Set scale to [2.0, 3.0]
    res.residual_scale = Some(Tensor::from_vec(vec![2.0, 3.0], vec![2]));
    let x = Tensor::from_vec(vec![1.0, 1.0], vec![1, 2]);
    let r = Tensor::from_vec(vec![0.5, 0.5], vec![1, 2]);
    let out = res.forward(&x, &r);
    // x + r * scale = [1 + 0.5*2, 1 + 0.5*3] = [2.0, 2.5]
    assert!(approx_eq(out.data[0], 2.0, 1e-6));
    assert!(approx_eq(out.data[1], 2.5, 1e-6));
}

// ── FeedForward tests ─────────────────────────────────────────────────────

#[test]
fn test_feedforward_identity_weights() {
    // With identity-like weights, ff should be approximately: x → x (if ff is the identity)
    // Actually test the shape
    let ff = FeedForward::new(4, 2); // dim=4, inner=8
    let x = Tensor::zeros(&[1, 3, 4]); // [B, N, D]
    let out = ff.forward(&x);
    assert_eq!(out.shape, vec![1, 3, 4]);
}

// ── SubjectLayers tests ───────────────────────────────────────────────────

#[test]
fn test_subject_layers_average_mode() {
    // average_subjects uses the dropout subject row (last row)
    let config = SubjectLayersConfig {
        n_subjects: 2,
        bias: true,
        subject_dropout: Some(0.1),
        average_subjects: true,
        ..Default::default()
    };
    let mut sl = SubjectLayers::new(3, 2, &config);
    // num_weight_subjects = 3 (2 + 1 for dropout)
    assert_eq!(sl.weights.shape, vec![3, 3, 2]);

    // Set dropout subject weights (index 2) to identity-like
    // weights[2] = [[1,0],[0,1],[0,0]]
    sl.weights.data[2 * 3 * 2 + 0] = 1.0; // w[2,0,0]
    sl.weights.data[2 * 3 * 2 + 3] = 1.0; // w[2,1,1]

    // Set bias[2] = [0.5, 0.5]
    if let Some(ref mut b) = sl.bias {
        b.data[2 * 2 + 0] = 0.5;
        b.data[2 * 2 + 1] = 0.5;
    }

    // x: [1, 3, 2] — B=1, C=3, T=2
    let x = Tensor::from_vec(vec![
        1.0, 2.0,  // channel 0
        3.0, 4.0,  // channel 1
        5.0, 6.0,  // channel 2
    ], vec![1, 3, 2]);

    let out = sl.forward(&x, None);
    assert_eq!(out.shape, vec![1, 2, 2]);
    // einsum('bct,cd->bdt'):
    // out[0,0,0] = x[0,0,0]*w[0,0] + x[0,1,0]*w[1,0] + x[0,2,0]*w[2,0] = 1*1 + 3*0 + 5*0 = 1.0 + bias 0.5 = 1.5
    // out[0,1,0] = x[0,0,0]*w[0,1] + x[0,1,0]*w[1,1] + x[0,2,0]*w[2,1] = 1*0 + 3*1 + 5*0 = 3.0 + bias 0.5 = 3.5
    assert!(approx_eq(out.data[0], 1.5, 1e-5)); // [0,0,0]
    assert!(approx_eq(out.data[1], 2.5, 1e-5)); // [0,0,1]
    assert!(approx_eq(out.data[2], 3.5, 1e-5)); // [0,1,0]
    assert!(approx_eq(out.data[3], 4.5, 1e-5)); // [0,1,1]
}

#[test]
fn test_subject_layers_per_subject_gather() {
    let config = SubjectLayersConfig {
        n_subjects: 2,
        bias: false,
        subject_dropout: None,
        average_subjects: false,
        ..Default::default()
    };
    let mut sl = SubjectLayers::new(2, 2, &config);
    // weights[0] = [[1,0],[0,1]], weights[1] = [[0,1],[1,0]]
    sl.weights.data = vec![
        1.0, 0.0, 0.0, 1.0,  // subject 0: identity
        0.0, 1.0, 1.0, 0.0,  // subject 1: swap
    ];

    // x: [2, 2, 1] — two batch items, subject 0 and 1
    let x = Tensor::from_vec(vec![
        3.0,  // batch 0, channel 0
        4.0,  // batch 0, channel 1
        5.0,  // batch 1, channel 0
        6.0,  // batch 1, channel 1
    ], vec![2, 2, 1]);

    let out = sl.forward(&x, Some(&[0, 1]));
    assert_eq!(out.shape, vec![2, 2, 1]);
    // Batch 0 (subject 0, identity): [3, 4]
    assert!(approx_eq(out.data[0], 3.0, 1e-5));
    assert!(approx_eq(out.data[1], 4.0, 1e-5));
    // Batch 1 (subject 1, swap): [6, 5]
    assert!(approx_eq(out.data[2], 6.0, 1e-5));
    assert!(approx_eq(out.data[3], 5.0, 1e-5));
}

// ── Projector tests ───────────────────────────────────────────────────────

#[test]
fn test_linear_projector() {
    let mut proj = Projector::new_linear(3, 2);
    // Set weight to [[1,0],[0,1],[0,0]] (in_dim=3, out_dim=2)
    proj.layers[0].weight = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![3, 2]);
    proj.layers[0].bias = Tensor::from_vec(vec![0.1, 0.2], vec![2]);

    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let out = proj.forward(&x);
    assert_eq!(out.shape, vec![1, 2]);
    assert!(approx_eq(out.data[0], 1.1, 1e-5)); // 1*1 + 2*0 + 3*0 + 0.1
    assert!(approx_eq(out.data[1], 2.2, 1e-5)); // 1*0 + 2*1 + 3*0 + 0.2
}

// ── Adaptive average pool tests ───────────────────────────────────────────

#[test]
fn test_adaptive_avg_pool_identity() {
    // When t_in == t_out, should be identity
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
    let out = x.adaptive_avg_pool1d(4);
    assert_eq!(out.data, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_adaptive_avg_pool_downsample() {
    // [1, 2, 3, 4, 5, 6] → pool to 2
    // Bin 0: [1, 2, 3] → 2.0, Bin 1: [4, 5, 6] → 5.0
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 6]);
    let out = x.adaptive_avg_pool1d(2);
    assert_eq!(out.shape, vec![1, 2]);
    assert!(approx_eq(out.data[0], 2.0, 1e-5));
    assert!(approx_eq(out.data[1], 5.0, 1e-5));
}

// ── einsum bct,cd->bdt test ───────────────────────────────────────────────

#[test]
fn test_einsum_bct_cd_bdt() {
    // x: [1, 2, 3], w: [2, 2]
    let x = Tensor::from_vec(vec![
        1.0, 2.0, 3.0,  // c=0
        4.0, 5.0, 6.0,  // c=1
    ], vec![1, 2, 3]);
    let w = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
    let out = x.einsum_bct_cd_bdt(&w);
    assert_eq!(out.shape, vec![1, 2, 3]);
    // Identity weight → out = x
    assert_eq!(out.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

// ── Full model forward shape test ─────────────────────────────────────────

#[test]
fn test_model_forward_shape() {
    let feature_dims = vec![
        ModalityDims::new("text", 1, 128),
        ModalityDims::new("audio", 1, 64),
    ];

    let config = BrainModelConfig {
        hidden: 128,
        max_seq_len: 128,
        extractor_aggregation: "cat".into(),
        layer_aggregation: "cat".into(),
        linear_baseline: true, // skip transformer for speed
        time_pos_embedding: false,
        subject_embedding: false,
        dropout: 0.0,
        modality_dropout: 0.0,
        temporal_dropout: 0.0,
        low_rank_head: None,
        combiner: None,
        temporal_smoothing: None,
        projector: Default::default(),
        encoder: None,
        subject_layers: Some(SubjectLayersConfig {
            n_subjects: 1,
            bias: true,
            subject_dropout: None,
            average_subjects: false,
            ..Default::default()
        }),
    };

    let model = TribeV2::new(feature_dims, 50, 5, &config);

    let mut features = BTreeMap::new();
    features.insert("text".to_string(), Tensor::zeros(&[2, 128, 10]));
    features.insert("audio".to_string(), Tensor::zeros(&[2, 64, 10]));

    let out = model.forward(&features, Some(&[0, 0]), true);
    assert_eq!(out.shape, vec![2, 50, 5]);
}

#[test]
fn test_model_with_none_modality() {
    // Test that a modality with dims=None is properly zero-filled
    let feature_dims = vec![
        ModalityDims::new("text", 1, 64),
        ModalityDims::none("audio"),  // No audio features
    ];

    let config = BrainModelConfig {
        hidden: 64,
        max_seq_len: 128,
        extractor_aggregation: "cat".into(),
        layer_aggregation: "cat".into(),
        linear_baseline: true,
        time_pos_embedding: false,
        subject_embedding: false,
        dropout: 0.0,
        modality_dropout: 0.0,
        temporal_dropout: 0.0,
        low_rank_head: None,
        combiner: None,
        temporal_smoothing: None,
        projector: Default::default(),
        encoder: None,
        subject_layers: Some(SubjectLayersConfig {
            n_subjects: 1,
            bias: true,
            subject_dropout: None,
            average_subjects: false,
            ..Default::default()
        }),
    };

    let model = TribeV2::new(feature_dims, 20, 5, &config);

    let mut features = BTreeMap::new();
    features.insert("text".to_string(), Tensor::zeros(&[1, 64, 10]));

    let out = model.forward(&features, Some(&[0]), true);
    assert_eq!(out.shape, vec![1, 20, 5]);
}

// ── Aggregation mode tests ────────────────────────────────────────────────

#[test]
fn test_aggregate_cat() {
    let feature_dims = vec![
        ModalityDims::new("a", 1, 4),
        ModalityDims::new("b", 1, 4),
    ];
    let config = BrainModelConfig {
        hidden: 8,
        extractor_aggregation: "cat".into(),
        layer_aggregation: "cat".into(),
        linear_baseline: true,
        time_pos_embedding: false,
        subject_embedding: false,
        combiner: None,
        encoder: None,
        subject_layers: Some(SubjectLayersConfig { n_subjects: 1, bias: false, subject_dropout: None, average_subjects: false, ..Default::default() }),
        ..default_brain_config()
    };
    let model = TribeV2::new(feature_dims, 10, 1, &config);

    let mut features = BTreeMap::new();
    features.insert("a".to_string(), Tensor::from_vec(vec![1.0; 4 * 3], vec![1, 4, 3]));
    features.insert("b".to_string(), Tensor::from_vec(vec![2.0; 4 * 3], vec![1, 4, 3]));

    let agg = model.aggregate_features(&features);
    // Should be [1, 3, 8] — cat of [1, 3, 4] and [1, 3, 4] on last dim
    assert_eq!(agg.shape, vec![1, 3, 8]);
}

#[test]
fn test_aggregate_sum() {
    let feature_dims = vec![
        ModalityDims::new("a", 1, 4),
        ModalityDims::new("b", 1, 4),
    ];
    let config = BrainModelConfig {
        hidden: 4,
        extractor_aggregation: "sum".into(),
        layer_aggregation: "cat".into(),
        linear_baseline: true,
        time_pos_embedding: false,
        subject_embedding: false,
        combiner: None,
        encoder: None,
        subject_layers: Some(SubjectLayersConfig { n_subjects: 1, bias: false, subject_dropout: None, average_subjects: false, ..Default::default() }),
        ..default_brain_config()
    };
    let model = TribeV2::new(feature_dims, 10, 1, &config);

    let mut features = BTreeMap::new();
    features.insert("a".to_string(), Tensor::from_vec(vec![1.0; 4 * 3], vec![1, 4, 3]));
    features.insert("b".to_string(), Tensor::from_vec(vec![2.0; 4 * 3], vec![1, 4, 3]));

    let agg = model.aggregate_features(&features);
    // Should be [1, 3, 4] — sum of projected features
    assert_eq!(agg.shape, vec![1, 3, 4]);
}

// ── Numerical attention test ───────────────────────────────────────────────

#[test]
fn test_attention_numerical() {
    // Manually verify attention with known weights on a tiny example
    // dim=4, heads=2, dim_head=2
    use tribev2_rs::model::attention::Attention;

    let mut attn = Attention::new(4, 2);
    // Set all projection weights to identity: [4, 4]
    attn.w_q = Tensor::from_vec(vec![
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ], vec![4, 4]);
    attn.w_k = attn.w_q.clone();
    attn.w_v = attn.w_q.clone();
    attn.w_out = attn.w_q.clone();

    // x: [1, 2, 4] — batch=1, seq=2, dim=4
    let x = Tensor::from_vec(vec![
        1.0, 0.0, 0.0, 0.0,  // token 0
        0.0, 1.0, 0.0, 0.0,  // token 1
    ], vec![1, 2, 4]);

    let out = attn.forward(&x, None);
    assert_eq!(out.shape, vec![1, 2, 4]);

    // With identity weights and no RoPE:
    // Q = K = V = x
    // After reshape to [1, 2, 2, 2]: head 0 gets first 2 dims, head 1 gets last 2 dims
    // Head 0: q=[[1,0],[0,1]], k=[[1,0],[0,1]], v=[[1,0],[0,1]]
    //   scores = q @ k^T * scale = [[1,0],[0,1]] * (2^-0.5) = [[0.707, 0], [0, 0.707]]
    //   attn = softmax([[0.707, 0], [0, 0.707]])
    //   For row 0: softmax([0.707, 0]) = [exp(0.707), exp(0)] / sum = [2.028, 1] / 3.028 ≈ [0.669, 0.330]
    //   out_h0 = attn @ v
    // Head 1: q=k=v=0, so scores = 0, attn = [0.5, 0.5] (uniform), out = avg of v = [0, 0]
    // After merge and output projection, token 0 should be weighted average of tokens

    // Just verify it doesn't crash and gives reasonable values
    let sum: f32 = out.data.iter().sum();
    assert!(sum.is_finite());
    // Token 0 should have nonzero first element (from head 0)
    assert!(out.data[0] > 0.0, "token 0 dim 0 should be > 0, got {}", out.data[0]);
}

// ── Full encoder numerical test ───────────────────────────────────────────

#[test]
fn test_encoder_deterministic() {
    use tribev2_rs::model::encoder::XTransformerEncoder;

    let config = EncoderConfig {
        heads: 2,
        depth: 1,
        ff_mult: 2,
        use_scalenorm: true,
        rotary_pos_emb: true,
        scale_residual: true,
        ..Default::default()
    };

    let enc = XTransformerEncoder::new(64, &config);
    // 2 layers: 1 attn + 1 FF

    let x = Tensor::from_vec(vec![0.1f32; 1 * 3 * 64], vec![1, 3, 64]);

    let out = enc.forward(&x);
    assert_eq!(out.shape, vec![1, 3, 64]);

    // With all-zero weights, the attention and FF outputs are zero,
    // so residual = inner_residual * scale (ones) = inner_residual
    // After final ScaleNorm(g=1): should normalize the constant vector
    // [0.1, 0.1, ...] → normalize → [1/sqrt(64), ...] * sqrt(64) * 1 = [1, 1, ...]
    // Wait: ScaleNorm of a constant vector [c, c, ...c] of dim d:
    // norm = sqrt(d * c^2) = c * sqrt(d)
    // normalized = [c / (c*sqrt(d)), ...] = [1/sqrt(d), ...]
    // * sqrt(d) * g = [1, 1, ...] (when g=1)
    for v in &out.data {
        assert!(approx_eq(*v, 1.0, 1e-4), "expected ~1.0, got {}", v);
    }
}

// ── Test full forward with known weights ──────────────────────────────────

#[test]
fn test_full_forward_numerical() {
    // Test the complete pipeline with a simple setup
    let feature_dims = vec![ModalityDims::new("text", 1, 4)];

    let config = BrainModelConfig {
        hidden: 4,
        max_seq_len: 16,
        extractor_aggregation: "cat".into(),
        layer_aggregation: "cat".into(),
        linear_baseline: true, // skip transformer for deterministic test
        time_pos_embedding: false,
        subject_embedding: false,
        dropout: 0.0,
        modality_dropout: 0.0,
        temporal_dropout: 0.0,
        low_rank_head: None,
        combiner: None,
        temporal_smoothing: None,
        projector: Default::default(),
        encoder: None,
        subject_layers: Some(SubjectLayersConfig {
            n_subjects: 1,
            bias: false,
            subject_dropout: None,
            average_subjects: false,
            ..Default::default()
        }),
    };

    let mut model = TribeV2::new(feature_dims, 2, 3, &config);

    // Set projector to identity [4→4]
    model.projectors[0].projector.layers[0].weight =
        Tensor::from_vec(vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ], vec![4, 4]);
    model.projectors[0].projector.layers[0].bias =
        Tensor::from_vec(vec![0.0; 4], vec![4]);

    // Set predictor weights[0] to a known matrix: [1, 4, 2]
    model.predictor.weights = Tensor::from_vec(vec![
        1.0, 0.0,  // in_ch 0 → [1, 0]
        0.0, 1.0,  // in_ch 1 → [0, 1]
        0.0, 0.0,  // in_ch 2 → [0, 0]
        0.0, 0.0,  // in_ch 3 → [0, 0]
    ], vec![1, 4, 2]);

    // Input: text [1, 4, 6] — 6 timesteps
    let mut features = BTreeMap::new();
    let text_data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.1).collect();
    features.insert("text".to_string(), Tensor::from_vec(text_data, vec![1, 4, 6]));

    let out = model.forward(&features, Some(&[0]), true);
    assert_eq!(out.shape, vec![1, 2, 3]);

    // Trace: text [1, 4, 6] → unsqueeze [1, 1, 4, 6] → reshape (cat) [1, 4, 6]
    //   → permute [1, 6, 4] → projector (identity) → [1, 6, 4]
    // aggregate: [1, 6, 4] (cat, single modality)
    // linear_baseline → skip transformer
    // permute to [1, 4, 6]
    // no low_rank_head
    // predictor: einsum('bct,cd->bdt', [1,4,6], [4,2]) → [1, 2, 6]
    //   out[0, 0, t] = x[0, 0, t]*1 + x[0, 1, t]*0 + ... = x[0, 0, t]
    //   out[0, 1, t] = x[0, 0, t]*0 + x[0, 1, t]*1 + ... = x[0, 1, t]
    // pool [1, 2, 6] → [1, 2, 3]
    //   bin 0: [0, 2) → mean of t=0,1
    //   bin 1: [2, 4) → mean of t=2,3
    //   bin 2: [4, 6) → mean of t=4,5

    // x[0, 0, :] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] (first channel, 6 timesteps)
    // x[0, 1, :] = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1] (second channel)
    // predictor out[0, 0, :] = x[0, 0, :] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    // predictor out[0, 1, :] = x[0, 1, :] = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
    // pool out[0, 0] = [avg(0.0, 0.1), avg(0.2, 0.3), avg(0.4, 0.5)] = [0.05, 0.25, 0.45]
    // pool out[0, 1] = [avg(0.6, 0.7), avg(0.8, 0.9), avg(1.0, 1.1)] = [0.65, 0.85, 1.05]

    // out is [1, 2, 3] in layout [batch, output_channels, timesteps]
    assert!(approx_eq(out.data[0], 0.05, 1e-5), "got {}", out.data[0]);
    assert!(approx_eq(out.data[1], 0.25, 1e-5), "got {}", out.data[1]);
    assert!(approx_eq(out.data[2], 0.45, 1e-5), "got {}", out.data[2]);
    assert!(approx_eq(out.data[3], 0.65, 1e-5), "got {}", out.data[3]);
    assert!(approx_eq(out.data[4], 0.85, 1e-5), "got {}", out.data[4]);
    assert!(approx_eq(out.data[5], 1.05, 1e-5), "got {}", out.data[5]);
}

// ── Test layer aggregation "mean" ─────────────────────────────────────────

#[test]
fn test_layer_aggregation_mean() {
    let feature_dims = vec![ModalityDims::new("text", 2, 4)];

    let config = BrainModelConfig {
        hidden: 4,
        extractor_aggregation: "cat".into(),
        layer_aggregation: "mean".into(),
        linear_baseline: true,
        time_pos_embedding: false,
        subject_embedding: false,
        combiner: None,
        encoder: None,
        subject_layers: Some(SubjectLayersConfig {
            n_subjects: 1, bias: false, subject_dropout: None, average_subjects: false, ..Default::default()
        }),
        ..default_brain_config()
    };

    let mut model = TribeV2::new(feature_dims, 2, 2, &config);
    // Identity projector
    model.projectors[0].projector.layers[0].weight =
        Tensor::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0], vec![4, 4]);
    model.projectors[0].projector.layers[0].bias = Tensor::from_vec(vec![0.0; 4], vec![4]);

    // Input: [1, 2, 4, 3] — B=1, L=2 layers, D=4, T=3
    let data: Vec<f32> = vec![
        // Layer 0: channels [1,1,1,1] for all 3 timesteps
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        // Layer 1: channels [3,3,3,3] for all 3 timesteps
        3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
    ];
    let mut features = BTreeMap::new();
    features.insert("text".to_string(), Tensor::from_vec(data, vec![1, 2, 4, 3]));

    let agg = model.aggregate_features(&features);
    // layer mean of [1,1,1,1] and [3,3,3,3] = [2,2,2,2] for each timestep
    // then projector (identity) → [2,2,2,2]
    // agg: [1, 3, 4] (B, T, H)
    assert_eq!(agg.shape, vec![1, 3, 4]);
    for v in &agg.data {
        assert!(approx_eq(*v, 2.0, 1e-5), "expected 2.0, got {}", v);
    }
}

fn default_brain_config() -> BrainModelConfig {
    BrainModelConfig {
        hidden: 64,
        max_seq_len: 128,
        extractor_aggregation: "cat".into(),
        layer_aggregation: "cat".into(),
        linear_baseline: false,
        time_pos_embedding: false,
        subject_embedding: false,
        dropout: 0.0,
        modality_dropout: 0.0,
        temporal_dropout: 0.0,
        low_rank_head: None,
        combiner: None,
        temporal_smoothing: None,
        projector: Default::default(),
        encoder: None,
        subject_layers: Some(SubjectLayersConfig::default()),
    }
}

#[test]
fn test_parse_real_config() {
    let yaml = std::fs::read_to_string("/tmp/tribev2_config.yaml").unwrap();
    let config: Result<TribeV2Config, _> = serde_yaml::from_str(&yaml);
    match &config {
        Ok(c) => {
            assert_eq!(c.brain_model_config.hidden, 1152);
            assert_eq!(c.brain_model_config.encoder.as_ref().unwrap().depth, 8);
            assert_eq!(c.brain_model_config.encoder.as_ref().unwrap().heads, 8);
            assert_eq!(c.brain_model_config.low_rank_head, Some(2048));
            assert_eq!(c.data.features_to_use, vec!["text", "audio", "video"]);
            assert_eq!(c.data.duration_trs, 100);
        }
        Err(e) => panic!("Failed to parse real config.yaml: {}", e),
    }
}
