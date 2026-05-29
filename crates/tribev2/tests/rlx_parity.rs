//! RLX encoder parity: `TribeRlx` vs pure-Rust `TribeV2` and Python references.

mod parity_metrics;

use std::collections::BTreeMap;
use std::path::Path;

use parity_metrics::{compare_slices, log_top_diffs, pearson, ParityReport};
use tribev2::model::tribe::TribeV2;
use tribev2::model_rlx::TribeRlx;
use tribev2::tensor::Tensor;

fn data_dir() -> std::path::PathBuf {
    tribev2::data_dir()
}

fn refs_dir() -> std::path::PathBuf {
    tribev2::parity_refs_dir()
}

fn data_path(rel: &str) -> String {
    data_dir().join(rel).to_string_lossy().into_owned()
}

fn refs_exist() -> bool {
    tribev2::weights_available() && tribev2::parity_refs_available()
}

fn load_ref(name: &str) -> Tensor {
    let path = refs_dir().join(name);
    let bytes = std::fs::read(&path).unwrap();
    let ndims = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let mut shape = Vec::with_capacity(ndims);
    let mut offset = 4;
    for _ in 0..ndims {
        let d = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        shape.push(d);
        offset += 4;
    }
    let n_floats: usize = shape.iter().product();
    let data: Vec<f32> = (0..n_floats)
        .map(|i| f32::from_le_bytes(bytes[offset + i * 4..offset + i * 4 + 4].try_into().unwrap()))
        .collect();
    Tensor::from_vec(data, shape)
}

fn assert_parity(x: &[f32], y: &[f32], label: &str) -> ParityReport {
    let r = compare_slices(x, y);
    r.assert_full_parity(label);
    r
}

#[test]
fn test_rlx_cat_vs_pure_rust() {
    if !refs_exist() {
        eprintln!("SKIP: parity refs / weights not found");
        return;
    }
    use tribev2::model_rlx::graph::{build_tribe_cat_graph, TribeSpec};
    use tribev2::model_rlx::weights::{apply_params, build_tribe_params, load_safetensors};

    let input_text = load_ref("input_text.bin");
    let input_audio = load_ref("input_audio.bin");
    let input_video = load_ref("input_video.bin");

    let rust_model = TribeV2::from_pretrained(
        &data_path("config.yaml"),
        &data_path("model.safetensors"),
        Some(&data_path("build_args.json")),
    )
    .unwrap();

    let mut features = BTreeMap::new();
    features.insert("text".to_string(), input_text.clone());
    features.insert("audio".to_string(), input_audio.clone());
    features.insert("video".to_string(), input_video.clone());
    let rust_cat = rust_model.aggregate_features(&features);

    let mod_order: Vec<String> = rust_model
        .feature_dims
        .iter()
        .map(|m| m.name.clone())
        .collect();
    eprintln!("modality concat order: {mod_order:?}");

    let mut raw = load_safetensors(&data_path("model.safetensors")).unwrap();
    let spec = TribeSpec::new(1, 20, 6144, 2048, 2816, 20484);
    let params = build_tribe_params(&mut raw, &spec).unwrap();
    let session = rlx::Session::new(rlx::Device::Cpu);
    let mut compiled = session.compile(build_tribe_cat_graph(&spec, &mod_order));
    apply_params(&mut compiled, &params);
    let outs = compiled.run(&[
        ("text", input_text.data.as_slice()),
        ("audio", input_audio.data.as_slice()),
        ("video", input_video.data.as_slice()),
    ]);
    let rlx_cat = Tensor::from_vec(outs[0].clone(), rust_cat.shape.clone());

    for np in &rust_model.projectors {
        eprintln!("rust projector: {}", np.name);
    }
    if let Some(pw) = params.get("projectors.text.weight") {
        eprintln!("rlx text weight shape {:?}", pw.shape);
    } else {
        eprintln!("missing projectors.text.weight in ParamMap");
    }

    let ref_cat = load_ref_with_header("after_cat.bin");
    let r_rust_ref = pearson(&rust_cat.data, &ref_cat.data);
    let r_rlx_ref = pearson(&rlx_cat.data, &ref_cat.data);
    eprintln!(
        "after_cat Pearson: rust/ref={r_rust_ref:.10} rlx/ref={r_rlx_ref:.10}"
    );

    assert_parity(&rust_cat.data, &rlx_cat.data, "cat vs rust");
}

#[test]
fn test_rlx_encoder_vs_pure_rust() {
    if !refs_exist() {
        eprintln!("SKIP: parity refs / weights not found");
        return;
    }
    use tribev2::model_rlx::graph::{build_tribe_encoder_graph, TribeSpec};
    use tribev2::model_rlx::weights::{apply_params, build_tribe_params, load_safetensors};

    let input_text = load_ref("input_text.bin");
    let input_audio = load_ref("input_audio.bin");
    let input_video = load_ref("input_video.bin");
    let ref_enc = load_ref_with_header("after_encoder.bin");

    let rust_model = TribeV2::from_pretrained(
        &data_path("config.yaml"),
        &data_path("model.safetensors"),
        Some(&data_path("build_args.json")),
    )
    .unwrap();
    let mod_order: Vec<String> = rust_model
        .feature_dims
        .iter()
        .map(|m| m.name.clone())
        .collect();

    let mut features = BTreeMap::new();
    features.insert("text".to_string(), input_text.clone());
    features.insert("audio".to_string(), input_audio.clone());
    features.insert("video".to_string(), input_video.clone());
    let rust_enc = rust_model.forward_encoder(&features);

    let mut raw = load_safetensors(&data_path("model.safetensors")).unwrap();
    let spec = TribeSpec::new(1, 20, 6144, 2048, 2816, 20484);
    let params = build_tribe_params(&mut raw, &spec).unwrap();
    let (cos, sin) = tribev2::model_rlx::rope::rope_cos_sin(20, spec.rot_dim);
    let session = rlx::Session::new(rlx::Device::Cpu);
    let mut compiled = session.compile(build_tribe_encoder_graph(&spec, &mod_order));
    apply_params(&mut compiled, &params);
    let outs = compiled.run(&[
        ("text", input_text.data.as_slice()),
        ("audio", input_audio.data.as_slice()),
        ("video", input_video.data.as_slice()),
        ("rope_cos", &cos),
        ("rope_sin", &sin),
    ]);
    let rlx_enc = Tensor::from_vec(outs[0].clone(), rust_enc.shape.clone());

    assert_parity(&rust_enc.data, &rlx_enc.data, "encoder vs rust");
    let r_ref = compare_slices(&rlx_enc.data, &ref_enc.data);
    r_ref.log("encoder vs python ref");
    assert!(r_ref.pearson > 0.999999, "encoder vs python ref");
}

#[cfg(feature = "rlx-metal")]
#[test]
fn test_rlx_cat_on_metal() {
    if !refs_exist() {
        return;
    }
    use tribev2::model_rlx::graph::{build_tribe_cat_graph, TribeSpec};
    use tribev2::model_rlx::weights::{apply_params, build_tribe_params, load_safetensors};

    let input_text = load_ref("input_text.bin");
    let input_audio = load_ref("input_audio.bin");
    let input_video = load_ref("input_video.bin");

    let rust_model = TribeV2::from_pretrained(
        &data_path("config.yaml"),
        &data_path("model.safetensors"),
        Some(&data_path("build_args.json")),
    )
    .unwrap();
    let mod_order: Vec<String> = rust_model.feature_dims.iter().map(|m| m.name.clone()).collect();
    let mut features = BTreeMap::new();
    features.insert("text".to_string(), input_text.clone());
    features.insert("audio".to_string(), input_audio.clone());
    features.insert("video".to_string(), input_video.clone());
    let rust_cat = rust_model.aggregate_features(&features);

    let mut raw = load_safetensors(&data_path("model.safetensors")).unwrap();
    let spec = TribeSpec::new(1, 20, 6144, 2048, 2816, 20484);
    let params = build_tribe_params(&mut raw, &spec).unwrap();
    let session = rlx::Session::new(rlx::Device::Metal);
    let mut compiled = session.compile(build_tribe_cat_graph(&spec, &mod_order));
    apply_params(&mut compiled, &params);
    let outs = compiled.run(&[
        ("text", input_text.data.as_slice()),
        ("audio", input_audio.data.as_slice()),
        ("video", input_video.data.as_slice()),
    ]);
    let rlx_cat = Tensor::from_vec(outs[0].clone(), rust_cat.shape.clone());
    assert_parity(&rust_cat.data, &rlx_cat.data, "cat on metal");
}

/// Bisect GPU encoder divergence layer-by-layer (depth=1..N).
#[cfg(any(feature = "rlx-metal", feature = "rlx-gpu", feature = "rlx-gpu-enc"))]
fn encoder_depth_parity(device: rlx::Device, label: &str, depth: usize) {
    if !refs_exist() || !rlx::runtime::is_available(device) {
        return;
    }
    use tribev2::model_rlx::graph::{build_tribe_encoder_graph, TribeSpec};
    use tribev2::model_rlx::weights::{apply_params, build_tribe_params, load_safetensors};

    let input_text = load_ref("input_text.bin");
    let input_audio = load_ref("input_audio.bin");
    let input_video = load_ref("input_video.bin");
    let rust_model = TribeV2::from_pretrained(
        &data_path("config.yaml"),
        &data_path("model.safetensors"),
        Some(&data_path("build_args.json")),
    )
    .unwrap();
    let mod_order: Vec<String> = rust_model.feature_dims.iter().map(|m| m.name.clone()).collect();
    let mut features = BTreeMap::new();
    features.insert("text".to_string(), input_text.clone());
    features.insert("audio".to_string(), input_audio.clone());
    features.insert("video".to_string(), input_video.clone());
    let rust_enc = rust_model.forward_encoder(&features);

    let mut raw = load_safetensors(&data_path("model.safetensors")).unwrap();
    let spec = TribeSpec::new(1, 20, 6144, 2048, 2816, 20484).with_depth(depth);
    let params = build_tribe_params(&mut raw, &spec).unwrap();
    let (cos, sin) = tribev2::model_rlx::rope::rope_cos_sin(20, spec.rot_dim);

    let mut cpu = rlx::Session::new(rlx::Device::Cpu).compile(build_tribe_encoder_graph(
        &spec,
        &mod_order,
    ));
    apply_params(&mut cpu, &params);
    let cpu_out = cpu
        .run(&[
            ("text", input_text.data.as_slice()),
            ("audio", input_audio.data.as_slice()),
            ("video", input_video.data.as_slice()),
            ("rope_cos", &cos),
            ("rope_sin", &sin),
        ])
        .remove(0);

    let mut gpu = rlx::Session::new(device).compile(build_tribe_encoder_graph(&spec, &mod_order));
    apply_params(&mut gpu, &params);
    let gpu_out = gpu
        .run(&[
            ("text", input_text.data.as_slice()),
            ("audio", input_audio.data.as_slice()),
            ("video", input_video.data.as_slice()),
            ("rope_cos", &cos),
            ("rope_sin", &sin),
        ])
        .remove(0);

    compare_slices(&cpu_out, &gpu_out).log(&format!("{label} depth={depth} gpu vs cpu rlx"));
    if depth == spec.depth {
        let rlx_enc = Tensor::from_vec(gpu_out, rust_enc.shape.clone());
        compare_slices(&rust_enc.data, &rlx_enc.data)
            .log(&format!("{label} depth={depth} gpu vs rust"));
    }
}

#[cfg(feature = "rlx-metal")]
#[test]
fn test_rlx_encoder_depth_bisect_metal() {
    for depth in 1..=8 {
        encoder_depth_parity(rlx::Device::Metal, "metal", depth);
    }
}

#[cfg(feature = "rlx-metal")]
#[test]
fn test_sdpa_bhsd_metal_vs_cpu() {
    if !rlx::runtime::is_available(rlx::Device::Metal) {
        return;
    }
    use rlx::ir::shape;
    use rlx::prelude::*;

    let (b, h, s, d) = (1usize, 8usize, 20usize, 144usize);
    let n = b * h * s * d;
    let q: Vec<f32> = (0..n).map(|i| ((i % 97) as f32 * 0.01).sin()).collect();
    let k: Vec<f32> = (0..n).map(|i| ((i % 59) as f32 * 0.02).cos()).collect();
    let v: Vec<f32> = (0..n).map(|i| ((i % 43) as f32 * 0.03).sin()).collect();

    let mut g = Graph::new("sdpa");
    let qi = g.input("q", Shape::new(&[b, h, s, d], DType::F32));
    let ki = g.input("k", Shape::new(&[b, h, s, d], DType::F32));
    let vi = g.input("v", Shape::new(&[b, h, s, d], DType::F32));
    let attn_shape = shape::attention_shape(g.shape(qi));
    let out = g.attention_kind(qi, ki, vi, h, d, MaskKind::None, attn_shape);
    g.set_outputs(vec![out]);

    let mut cpu = Session::new(Device::Cpu).compile(g.clone());
    let cpu_out = cpu.run(&[("q", &q), ("k", &k), ("v", &v)]).remove(0);
    let mut metal = Session::new(Device::Metal).compile(g);
    let metal_out = metal.run(&[("q", &q), ("k", &k), ("v", &v)]).remove(0);
    assert_parity(&cpu_out, &metal_out, "sdpa bhsd metal vs cpu");
}

#[cfg(any(feature = "rlx-gpu", feature = "rlx-gpu-enc"))]
#[test]
fn test_sdpa_bhsd_wgpu_vs_cpu() {
    if !rlx::runtime::is_available(rlx::Device::Gpu) {
        return;
    }
    use rlx::ir::shape;
    use rlx::prelude::*;

    for (b, h, s, d) in [(1usize, 1usize, 2usize, 2usize), (1, 8, 20, 144)] {
        let n = b * h * s * d;
        let q: Vec<f32> = (0..n).map(|i| ((i % 97) as f32 * 0.01).sin()).collect();
        let k: Vec<f32> = (0..n).map(|i| ((i % 59) as f32 * 0.02).cos()).collect();
        let v: Vec<f32> = (0..n).map(|i| ((i % 43) as f32 * 0.03).sin()).collect();

        let mut g = Graph::new("sdpa");
        let qi = g.input("q", Shape::new(&[b, h, s, d], DType::F32));
        let ki = g.input("k", Shape::new(&[b, h, s, d], DType::F32));
        let vi = g.input("v", Shape::new(&[b, h, s, d], DType::F32));
        let attn_shape = shape::attention_shape(g.shape(qi));
        let out = g.attention_kind(qi, ki, vi, h, d, MaskKind::None, attn_shape);
        g.set_outputs(vec![out]);

        let mut cpu = Session::new(Device::Cpu).compile(g.clone());
        let cpu_out = cpu.run(&[("q", &q), ("k", &k), ("v", &v)]).remove(0);
        let mut gpu = Session::new(Device::Gpu).compile(g);
        let gpu_out = gpu.run(&[("q", &q), ("k", &k), ("v", &v)]).remove(0);
        assert_parity(
            &cpu_out,
            &gpu_out,
            &format!("sdpa bhsd [{b},{h},{s},{d}] wgpu vs cpu"),
        );
    }
}

#[cfg(any(feature = "rlx-gpu", feature = "rlx-gpu-enc"))]
#[test]
fn test_scale_norm_wgpu_vs_cpu() {
    if !rlx::runtime::is_available(rlx::Device::Gpu) {
        return;
    }
    use rlx::prelude::*;
    use tribev2::model_rlx::graph::build_tribe_final_norm_graph;
    use tribev2::model_rlx::graph::TribeSpec;
    use tribev2::model_rlx::weights::{apply_params, build_tribe_params, load_safetensors};

    let spec = TribeSpec::new(1, 20, 6144, 2048, 2816, 20484);
    let n = 1 * 20 * 1152;
    let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001).sin()).collect();
    let mut raw = load_safetensors(&data_path("model.safetensors")).unwrap();
    let params = build_tribe_params(&mut raw, &spec).unwrap();
    let graph = build_tribe_final_norm_graph(&spec);
    let mut cpu = Session::new(Device::Cpu).compile(graph.clone());
    apply_params(&mut cpu, &params);
    let cpu_out = cpu.run(&[("x", &x)]).remove(0);
    let mut gpu = Session::new(Device::Gpu).compile(graph);
    apply_params(&mut gpu, &params);
    let gpu_out = gpu.run(&[("x", &x)]).remove(0);
    assert_parity(&cpu_out, &gpu_out, "scale_norm wgpu vs cpu");
}

#[cfg(feature = "rlx-metal")]
#[test]
fn test_scale_norm_metal_vs_cpu() {
    if !rlx::runtime::is_available(rlx::Device::Metal) {
        return;
    }
    use rlx::prelude::*;
    use tribev2::model_rlx::graph::build_tribe_final_norm_graph;
    use tribev2::model_rlx::graph::TribeSpec;
    use tribev2::model_rlx::weights::{apply_params, build_tribe_params, load_safetensors};

    let spec = TribeSpec::new(1, 20, 6144, 2048, 2816, 20484);
    let n = 1 * 20 * 1152;
    let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001).sin()).collect();
    let mut raw = load_safetensors(&data_path("model.safetensors")).unwrap();
    let params = build_tribe_params(&mut raw, &spec).unwrap();
    let graph = build_tribe_final_norm_graph(&spec);
    let mut cpu = Session::new(Device::Cpu).compile(graph.clone());
    apply_params(&mut cpu, &params);
    let cpu_out = cpu.run(&[("x", &x)]).remove(0);
    let mut metal = Session::new(Device::Metal).compile(graph);
    apply_params(&mut metal, &params);
    let metal_out = metal.run(&[("x", &x)]).remove(0);
    assert_parity(&cpu_out, &metal_out, "scale_norm metal vs cpu");
}

#[cfg(feature = "rlx-metal")]
#[test]
fn test_rlx_first_attn_on_metal() {
    if !refs_exist() || !rlx::runtime::is_available(rlx::Device::Metal) {
        return;
    }
    use tribev2::model_rlx::graph::{build_tribe_encoder_attn_only_graph, TribeSpec};
    use tribev2::model_rlx::weights::{apply_params, build_tribe_params, load_safetensors};

    let input_text = load_ref("input_text.bin");
    let input_audio = load_ref("input_audio.bin");
    let input_video = load_ref("input_video.bin");
    let mut raw = load_safetensors(&data_path("model.safetensors")).unwrap();
    let spec = TribeSpec::new(1, 20, 6144, 2048, 2816, 20484);
    let params = build_tribe_params(&mut raw, &spec).unwrap();
    let (cos, sin) = tribev2::model_rlx::rope::rope_cos_sin(20, spec.rot_dim);
    let mod_order = vec!["audio".into(), "text".into(), "video".into()];
    let graph = build_tribe_encoder_attn_only_graph(&spec, &mod_order, 1);
    let inputs = [
        ("text", input_text.data.as_slice()),
        ("audio", input_audio.data.as_slice()),
        ("video", input_video.data.as_slice()),
        ("rope_cos", &cos),
        ("rope_sin", &sin),
    ];
    let mut cpu = rlx::Session::new(rlx::Device::Cpu).compile(graph.clone());
    apply_params(&mut cpu, &params);
    let cpu_out = cpu.run(&inputs).remove(0);
    let mut metal = rlx::Session::new(rlx::Device::Metal).compile(graph);
    apply_params(&mut metal, &params);
    let metal_out = metal.run(&inputs).remove(0);
    assert_parity(&cpu_out, &metal_out, "first attn block metal vs cpu rlx");
}

#[cfg(feature = "rlx-metal")]
#[test]
fn test_rlx_first_block_on_metal() {
    if !refs_exist() || !rlx::runtime::is_available(rlx::Device::Metal) {
        return;
    }
    use tribev2::model_rlx::graph::{build_tribe_encoder_first_block_graph, TribeSpec};
    use tribev2::model_rlx::weights::{apply_params, build_tribe_params, load_safetensors};

    let input_text = load_ref("input_text.bin");
    let input_audio = load_ref("input_audio.bin");
    let input_video = load_ref("input_video.bin");
    let mut raw = load_safetensors(&data_path("model.safetensors")).unwrap();
    let spec = TribeSpec::new(1, 20, 6144, 2048, 2816, 20484);
    let params = build_tribe_params(&mut raw, &spec).unwrap();
    let (cos, sin) = tribev2::model_rlx::rope::rope_cos_sin(20, spec.rot_dim);
    let mod_order = vec!["audio".into(), "text".into(), "video".into()];
    let graph = build_tribe_encoder_first_block_graph(&spec, &mod_order);
    let inputs = [
        ("text", input_text.data.as_slice()),
        ("audio", input_audio.data.as_slice()),
        ("video", input_video.data.as_slice()),
        ("rope_cos", &cos),
        ("rope_sin", &sin),
    ];
    let mut cpu = rlx::Session::new(rlx::Device::Cpu).compile(graph.clone());
    apply_params(&mut cpu, &params);
    let cpu_out = cpu.run(&inputs).remove(0);
    let mut metal = rlx::Session::new(rlx::Device::Metal).compile(graph);
    apply_params(&mut metal, &params);
    let metal_out = metal.run(&inputs).remove(0);
    compare_slices(&cpu_out, &metal_out).log("first block (attn+ff) metal vs cpu");
}

#[cfg(feature = "rlx-metal")]
#[test]
fn test_rlx_final_norm_on_metal() {
    if !refs_exist() || !rlx::runtime::is_available(rlx::Device::Metal) {
        return;
    }
    use tribev2::model_rlx::graph::{build_tribe_encoder_first_block_graph, build_tribe_final_norm_graph, TribeSpec};
    use tribev2::model_rlx::weights::{apply_params, build_tribe_params, load_safetensors};

    let input_text = load_ref("input_text.bin");
    let input_audio = load_ref("input_audio.bin");
    let input_video = load_ref("input_video.bin");
    let mut raw = load_safetensors(&data_path("model.safetensors")).unwrap();
    let spec = TribeSpec::new(1, 20, 6144, 2048, 2816, 20484);
    let params = build_tribe_params(&mut raw, &spec).unwrap();
    let (cos, sin) = tribev2::model_rlx::rope::rope_cos_sin(20, spec.rot_dim);
    let mod_order = vec!["audio".into(), "text".into(), "video".into()];
    let mut block = rlx::Session::new(rlx::Device::Cpu).compile(build_tribe_encoder_first_block_graph(
        &spec,
        &mod_order,
    ));
    apply_params(&mut block, &params);
    let block_out = block
        .run(&[
            ("text", input_text.data.as_slice()),
            ("audio", input_audio.data.as_slice()),
            ("video", input_video.data.as_slice()),
            ("rope_cos", &cos),
            ("rope_sin", &sin),
        ])
        .remove(0);

    let norm_graph = build_tribe_final_norm_graph(&spec);
    let mut cpu = rlx::Session::new(rlx::Device::Cpu).compile(norm_graph.clone());
    apply_params(&mut cpu, &params);
    let cpu_out = cpu.run(&[("x", block_out.as_slice())]).remove(0);
    let mut metal = rlx::Session::new(rlx::Device::Metal).compile(norm_graph);
    apply_params(&mut metal, &params);
    let metal_out = metal.run(&[("x", block_out.as_slice())]).remove(0);
    assert_parity(&cpu_out, &metal_out, "final_norm metal vs cpu rlx");
}

#[cfg(any(feature = "rlx-gpu", feature = "rlx-gpu-enc"))]
#[test]
fn test_rlx_final_norm_on_wgpu() {
    if !refs_exist() || !rlx::runtime::is_available(rlx::Device::Gpu) {
        return;
    }
    use tribev2::model_rlx::graph::{build_tribe_encoder_first_block_graph, build_tribe_final_norm_graph, TribeSpec};
    use tribev2::model_rlx::weights::{apply_params, build_tribe_params, load_safetensors};

    let input_text = load_ref("input_text.bin");
    let input_audio = load_ref("input_audio.bin");
    let input_video = load_ref("input_video.bin");
    let mut raw = load_safetensors(&data_path("model.safetensors")).unwrap();
    let spec = TribeSpec::new(1, 20, 6144, 2048, 2816, 20484);
    let params = build_tribe_params(&mut raw, &spec).unwrap();
    let (cos, sin) = tribev2::model_rlx::rope::rope_cos_sin(20, spec.rot_dim);
    let mod_order = vec!["audio".into(), "text".into(), "video".into()];
    let mut block = rlx::Session::new(rlx::Device::Cpu).compile(build_tribe_encoder_first_block_graph(
        &spec,
        &mod_order,
    ));
    apply_params(&mut block, &params);
    let block_out = block
        .run(&[
            ("text", input_text.data.as_slice()),
            ("audio", input_audio.data.as_slice()),
            ("video", input_video.data.as_slice()),
            ("rope_cos", &cos),
            ("rope_sin", &sin),
        ])
        .remove(0);

    let norm_graph = build_tribe_final_norm_graph(&spec);
    let mut cpu = rlx::Session::new(rlx::Device::Cpu).compile(norm_graph.clone());
    apply_params(&mut cpu, &params);
    let cpu_out = cpu.run(&[("x", block_out.as_slice())]).remove(0);
    let mut gpu = rlx::Session::new(rlx::Device::Gpu).compile(norm_graph);
    apply_params(&mut gpu, &params);
    let gpu_out = gpu.run(&[("x", block_out.as_slice())]).remove(0);
    assert_parity(&cpu_out, &gpu_out, "final_norm wgpu vs cpu rlx");
}

#[cfg(feature = "rlx-metal")]
#[test]
fn test_rlx_encoder_on_metal() {
    if !refs_exist() {
        return;
    }
    use tribev2::model_rlx::graph::{build_tribe_encoder_graph, TribeSpec};
    use tribev2::model_rlx::weights::{apply_params, build_tribe_params, load_safetensors};

    let input_text = load_ref("input_text.bin");
    let input_audio = load_ref("input_audio.bin");
    let input_video = load_ref("input_video.bin");

    let rust_model = TribeV2::from_pretrained(
        &data_path("config.yaml"),
        &data_path("model.safetensors"),
        Some(&data_path("build_args.json")),
    )
    .unwrap();
    let mod_order: Vec<String> = rust_model
        .feature_dims
        .iter()
        .map(|m| m.name.clone())
        .collect();

    let mut features = BTreeMap::new();
    features.insert("text".to_string(), input_text.clone());
    features.insert("audio".to_string(), input_audio.clone());
    features.insert("video".to_string(), input_video.clone());
    let rust_enc = rust_model.forward_encoder(&features);

    let mut raw = load_safetensors(&data_path("model.safetensors")).unwrap();
    let spec = TribeSpec::new(1, 20, 6144, 2048, 2816, 20484);
    let params = build_tribe_params(&mut raw, &spec).unwrap();
    let (cos, sin) = tribev2::model_rlx::rope::rope_cos_sin(20, spec.rot_dim);
    let session = rlx::Session::new(rlx::Device::Metal);
    let mut compiled = session.compile(build_tribe_encoder_graph(&spec, &mod_order));
    apply_params(&mut compiled, &params);
    let outs = compiled.run(&[
        ("text", input_text.data.as_slice()),
        ("audio", input_audio.data.as_slice()),
        ("video", input_video.data.as_slice()),
        ("rope_cos", &cos),
        ("rope_sin", &sin),
    ]);
    let rlx_enc = Tensor::from_vec(outs[0].clone(), rust_enc.shape.clone());

    let mut cpu_compiled = rlx::Session::new(rlx::Device::Cpu).compile(
        tribev2::model_rlx::graph::build_tribe_encoder_graph(&spec, &mod_order),
    );
    apply_params(&mut cpu_compiled, &params);
    let cpu_outs = cpu_compiled.run(&[
        ("text", input_text.data.as_slice()),
        ("audio", input_audio.data.as_slice()),
        ("video", input_video.data.as_slice()),
        ("rope_cos", &cos),
        ("rope_sin", &sin),
    ]);
    let cpu_enc = Tensor::from_vec(cpu_outs[0].clone(), rust_enc.shape.clone());
    compare_slices(&cpu_enc.data, &rlx_enc.data).log("encoder metal vs cpu rlx");
    log_top_diffs(&cpu_enc.data, &rlx_enc.data, "encoder metal vs cpu rlx", 8);
    let n_bad = cpu_enc
        .data
        .iter()
        .zip(rlx_enc.data.iter())
        .filter(|(&a, &b)| (a - b).abs() > 1e-3)
        .count();
    eprintln!(
        "encoder metal: {n_bad}/{} elements with |diff|>1e-3",
        cpu_enc.data.len()
    );

    assert_parity(&rust_enc.data, &rlx_enc.data, "encoder on metal");
}

#[cfg(any(feature = "rlx-gpu", feature = "rlx-gpu-enc"))]
#[test]
fn test_rlx_tail_on_wgpu_native() {
    if !refs_exist() || !rlx::runtime::is_available(rlx::Device::Gpu) {
        return;
    }
    use tribev2::model_rlx::graph::{build_tribe_encoder_graph, build_tribe_tail_graph, TribeSpec};
    use tribev2::model_rlx::weights::{apply_params, build_tribe_params, load_safetensors};

    let input_text = load_ref("input_text.bin");
    let input_audio = load_ref("input_audio.bin");
    let input_video = load_ref("input_video.bin");
    let rust_model = TribeV2::from_pretrained(
        &data_path("config.yaml"),
        &data_path("model.safetensors"),
        Some(&data_path("build_args.json")),
    )
    .unwrap();
    let mod_order: Vec<String> = rust_model.feature_dims.iter().map(|m| m.name.clone()).collect();
    let mut features = BTreeMap::new();
    features.insert("text".to_string(), input_text.clone());
    features.insert("audio".to_string(), input_audio.clone());
    features.insert("video".to_string(), input_video.clone());
    let rust_enc = rust_model.forward_encoder(&features);

    let mut raw = load_safetensors(&data_path("model.safetensors")).unwrap();
    let spec = TribeSpec::new(1, 20, 6144, 2048, 2816, 20484);
    let params = build_tribe_params(&mut raw, &spec).unwrap();
    let (cos, sin) = tribev2::model_rlx::rope::rope_cos_sin(20, spec.rot_dim);

    let mut cpu = rlx::Session::new(rlx::Device::Cpu).compile(build_tribe_tail_graph(&spec));
    apply_params(&mut cpu, &params);
    let cpu_out = cpu.run(&[("enc", rust_enc.data.as_slice())]).remove(0);

    let mut gpu = rlx::Session::new(rlx::Device::Gpu).compile(build_tribe_tail_graph(&spec));
    apply_params(&mut gpu, &params);
    let gpu_out = gpu.run(&[("enc", rust_enc.data.as_slice())]).remove(0);

    let rust_out = rust_model.forward(&features, None, false);
    compare_slices(&rust_out.data, &cpu_out).log("tail cpu rlx vs rust");
    assert_parity(&cpu_out, &gpu_out, "tail wgpu vs cpu rlx");
}

#[cfg(any(feature = "rlx-gpu", feature = "rlx-gpu-enc"))]
#[test]
fn test_rlx_cat_on_wgpu() {
    if !refs_exist() || !rlx::runtime::is_available(rlx::Device::Gpu) {
        return;
    }
    use tribev2::model_rlx::graph::{build_tribe_cat_graph, TribeSpec};
    use tribev2::model_rlx::weights::{apply_params, build_tribe_params, load_safetensors};

    let input_text = load_ref("input_text.bin");
    let input_audio = load_ref("input_audio.bin");
    let input_video = load_ref("input_video.bin");
    let rust_model = TribeV2::from_pretrained(
        &data_path("config.yaml"),
        &data_path("model.safetensors"),
        Some(&data_path("build_args.json")),
    )
    .unwrap();
    let mod_order: Vec<String> = rust_model.feature_dims.iter().map(|m| m.name.clone()).collect();
    let mut features = BTreeMap::new();
    features.insert("text".to_string(), input_text.clone());
    features.insert("audio".to_string(), input_audio.clone());
    features.insert("video".to_string(), input_video.clone());
    let rust_cat = rust_model.aggregate_features(&features);

    let mut raw = load_safetensors(&data_path("model.safetensors")).unwrap();
    let spec = TribeSpec::new(1, 20, 6144, 2048, 2816, 20484);
    let params = build_tribe_params(&mut raw, &spec).unwrap();
    let session = rlx::Session::new(rlx::Device::Gpu);
    let mut compiled = session.compile(build_tribe_cat_graph(&spec, &mod_order));
    apply_params(&mut compiled, &params);
    let outs = compiled.run(&[
        ("text", input_text.data.as_slice()),
        ("audio", input_audio.data.as_slice()),
        ("video", input_video.data.as_slice()),
    ]);
    let rlx_cat = Tensor::from_vec(outs[0].clone(), rust_cat.shape.clone());
    assert_parity(&rust_cat.data, &rlx_cat.data, "cat on wgpu");
}

#[cfg(any(feature = "rlx-gpu", feature = "rlx-gpu-enc"))]
#[test]
fn test_rlx_encoder_on_wgpu() {
    if !refs_exist() || !rlx::runtime::is_available(rlx::Device::Gpu) {
        return;
    }
    use tribev2::model_rlx::graph::{build_tribe_encoder_graph, TribeSpec};
    use tribev2::model_rlx::weights::{apply_params, build_tribe_params, load_safetensors};

    let input_text = load_ref("input_text.bin");
    let input_audio = load_ref("input_audio.bin");
    let input_video = load_ref("input_video.bin");
    let rust_model = TribeV2::from_pretrained(
        &data_path("config.yaml"),
        &data_path("model.safetensors"),
        Some(&data_path("build_args.json")),
    )
    .unwrap();
    let mod_order: Vec<String> = rust_model.feature_dims.iter().map(|m| m.name.clone()).collect();
    let mut features = BTreeMap::new();
    features.insert("text".to_string(), input_text.clone());
    features.insert("audio".to_string(), input_audio.clone());
    features.insert("video".to_string(), input_video.clone());
    let rust_enc = rust_model.forward_encoder(&features);

    let mut raw = load_safetensors(&data_path("model.safetensors")).unwrap();
    let spec = TribeSpec::new(1, 20, 6144, 2048, 2816, 20484);
    let params = build_tribe_params(&mut raw, &spec).unwrap();
    let (cos, sin) = tribev2::model_rlx::rope::rope_cos_sin(20, spec.rot_dim);
    let session = rlx::Session::new(rlx::Device::Gpu);
    let mut compiled = session.compile(build_tribe_encoder_graph(&spec, &mod_order));
    apply_params(&mut compiled, &params);
    let outs = compiled.run(&[
        ("text", input_text.data.as_slice()),
        ("audio", input_audio.data.as_slice()),
        ("video", input_video.data.as_slice()),
        ("rope_cos", &cos),
        ("rope_sin", &sin),
    ]);
    let rlx_enc = Tensor::from_vec(outs[0].clone(), rust_enc.shape.clone());
    assert_parity(&rust_enc.data, &rlx_enc.data, "encoder on wgpu");
}

#[cfg(feature = "rlx-metal")]
#[test]
fn test_rlx_lowrank_on_metal() {
    if !refs_exist() {
        return;
    }
    use tribev2::model_rlx::graph::{build_tribe_lowrank_graph, TribeSpec};
    use tribev2::model_rlx::weights::{apply_params, build_tribe_params, load_safetensors};

    let input_text = load_ref("input_text.bin");
    let input_audio = load_ref("input_audio.bin");
    let input_video = load_ref("input_video.bin");

    let rust_model = TribeV2::from_pretrained(
        &data_path("config.yaml"),
        &data_path("model.safetensors"),
        Some(&data_path("build_args.json")),
    )
    .unwrap();
    let mod_order: Vec<String> = rust_model.feature_dims.iter().map(|m| m.name.clone()).collect();

    let mut features = BTreeMap::new();
    features.insert("text".to_string(), input_text.clone());
    features.insert("audio".to_string(), input_audio.clone());
    features.insert("video".to_string(), input_video.clone());
    let enc = rust_model.forward_encoder(&features);
    let lr_w = rust_model.low_rank_head.as_ref().unwrap();
    let (b, t, h) = (1usize, 20usize, 1152usize);
    let mut x = enc.permute(&[0, 2, 1]);
    x = x.permute(&[0, 2, 1]);
    let rust_lr = x
        .reshape(&[b * t, h])
        .matmul(lr_w)
        .reshape(&[b, t, lr_w.shape[1]])
        .permute(&[0, 2, 1]);

    let mut raw = load_safetensors(&data_path("model.safetensors")).unwrap();
    let spec = TribeSpec::new(1, 20, 6144, 2048, 2816, 20484);
    let params = build_tribe_params(&mut raw, &spec).unwrap();
    let (cos, sin) = tribev2::model_rlx::rope::rope_cos_sin(20, spec.rot_dim);
    let session = rlx::Session::new(rlx::Device::Metal);
    let mut compiled = session.compile(build_tribe_lowrank_graph(&spec, &mod_order));
    apply_params(&mut compiled, &params);
    let outs = compiled.run(&[
        ("text", input_text.data.as_slice()),
        ("audio", input_audio.data.as_slice()),
        ("video", input_video.data.as_slice()),
        ("rope_cos", &cos),
        ("rope_sin", &sin),
    ]);
    let rlx_lr = Tensor::from_vec(outs[0].clone(), rust_lr.shape.clone());
    assert_parity(&rust_lr.data, &rlx_lr.data, "lowrank on metal");
}

fn device_forward_parity(device: rlx::Device, label: &str) {
    // Native full-graph forward (no CPU hybrid).
    if !refs_exist() {
        eprintln!("SKIP: parity refs / weights not found");
        return;
    }
    if !rlx::runtime::is_available(device) {
        eprintln!("SKIP: RLX device {label} not available in this build");
        return;
    }

    let input_text = load_ref("input_text.bin");
    let input_audio = load_ref("input_audio.bin");
    let input_video = load_ref("input_video.bin");

    let rust_model = TribeV2::from_pretrained(
        &data_path("config.yaml"),
        &data_path("model.safetensors"),
        Some(&data_path("build_args.json")),
    )
    .unwrap();

    let mut rlx_model = TribeRlx::from_pretrained(
        &data_path("config.yaml"),
        &data_path("model.safetensors"),
        Some(&data_path("build_args.json")),
    )
    .unwrap()
    .with_device(device);

    let mut features = BTreeMap::new();
    features.insert("text".to_string(), input_text);
    features.insert("audio".to_string(), input_audio);
    features.insert("video".to_string(), input_video);

    let rust_out = rust_model.forward(&features, None, true);
    let rlx_out = rlx_model.forward(&features, None, true);
    assert_parity(&rust_out.data, &rlx_out.data, &format!("TribeRlx forward ({label})"));
}

#[cfg(feature = "rlx-metal")]
#[test]
fn test_rlx_vs_pure_rust_on_metal_device() {
    device_forward_parity(rlx::Device::Metal, "metal");
}

#[cfg(any(feature = "rlx-gpu", feature = "rlx-gpu-enc"))]
#[test]
fn test_rlx_vs_pure_rust_on_wgpu_device() {
    device_forward_parity(rlx::Device::Gpu, "wgpu");
}

#[cfg(any(feature = "rlx-gpu", feature = "rlx-gpu-enc"))]
#[test]
fn test_rlx_full_graph_on_wgpu_native() {
    if !refs_exist() {
        return;
    }
    if !rlx::runtime::is_available(rlx::Device::Gpu) {
        eprintln!("SKIP: wgpu not available");
        return;
    }
    use tribev2::model_rlx::graph::{build_tribe_graph, TribeSpec};
    use tribev2::model_rlx::weights::{apply_params, build_tribe_params, load_safetensors};

    let input_text = load_ref("input_text.bin");
    let input_audio = load_ref("input_audio.bin");
    let input_video = load_ref("input_video.bin");

    let rust_model = TribeV2::from_pretrained(
        &data_path("config.yaml"),
        &data_path("model.safetensors"),
        Some(&data_path("build_args.json")),
    )
    .unwrap();
    let mod_order: Vec<String> = rust_model.feature_dims.iter().map(|m| m.name.clone()).collect();
    let mut features = BTreeMap::new();
    features.insert("text".to_string(), input_text.clone());
    features.insert("audio".to_string(), input_audio.clone());
    features.insert("video".to_string(), input_video.clone());
    let rust_out = rust_model.forward(&features, None, false);

    let mut raw = load_safetensors(&data_path("model.safetensors")).unwrap();
    let spec = TribeSpec::new(1, 20, 6144, 2048, 2816, 20484);
    let params = build_tribe_params(&mut raw, &spec).unwrap();
    let (cos, sin) = tribev2::model_rlx::rope::rope_cos_sin(20, spec.rot_dim);
    let session = rlx::Session::new(rlx::Device::Gpu);
    let mut compiled = session.compile(build_tribe_graph(&spec, &mod_order));
    apply_params(&mut compiled, &params);
    let outs = compiled.run(&[
        ("text", input_text.data.as_slice()),
        ("audio", input_audio.data.as_slice()),
        ("video", input_video.data.as_slice()),
        ("rope_cos", &cos),
        ("rope_sin", &sin),
    ]);
    let rlx_out = Tensor::from_vec(outs[0].clone(), rust_out.shape.clone());

    assert_parity(&rust_out.data, &rlx_out.data, "native wgpu full graph");
}

#[cfg(any(feature = "rlx-cuda", feature = "rlx-cuda-enc"))]
#[test]
fn test_rlx_vs_pure_rust_on_cuda_device() {
    device_forward_parity(rlx::Device::Cuda, "cuda");
}

#[test]
fn test_rlx_lowrank_vs_pure_rust() {
    if !refs_exist() {
        return;
    }
    use tribev2::model_rlx::graph::{build_tribe_lowrank_graph, TribeSpec};
    use tribev2::model_rlx::weights::{apply_params, build_tribe_params, load_safetensors};

    let input_text = load_ref("input_text.bin");
    let input_audio = load_ref("input_audio.bin");
    let input_video = load_ref("input_video.bin");
    let ref_lr = load_ref_with_header("after_lowrank.bin");

    let rust_model = TribeV2::from_pretrained(
        &data_path("config.yaml"),
        &data_path("model.safetensors"),
        Some(&data_path("build_args.json")),
    )
    .unwrap();

    let mut features = BTreeMap::new();
    features.insert("text".to_string(), input_text.clone());
    features.insert("audio".to_string(), input_audio.clone());
    features.insert("video".to_string(), input_video.clone());

    let enc = rust_model.forward_encoder(&features);
    let lr_w = rust_model.low_rank_head.as_ref().unwrap();
    let (b, t, h) = (1usize, 20usize, 1152usize);
    let mut x = enc.permute(&[0, 2, 1]);
    x = x.permute(&[0, 2, 1]);
    let rust_lr = x
        .reshape(&[b * t, h])
        .matmul(lr_w)
        .reshape(&[b, t, lr_w.shape[1]])
        .permute(&[0, 2, 1]);

    let mod_order: Vec<String> = rust_model.feature_dims.iter().map(|m| m.name.clone()).collect();
    let mut raw = load_safetensors(&data_path("model.safetensors")).unwrap();
    let spec = TribeSpec::new(1, 20, 6144, 2048, 2816, 20484);
    let params = build_tribe_params(&mut raw, &spec).unwrap();
    let (cos, sin) = tribev2::model_rlx::rope::rope_cos_sin(20, spec.rot_dim);
    let session = rlx::Session::new(rlx::Device::Cpu);
    let mut compiled = session.compile(build_tribe_lowrank_graph(&spec, &mod_order));
    apply_params(&mut compiled, &params);
    let outs = compiled.run(&[
        ("text", input_text.data.as_slice()),
        ("audio", input_audio.data.as_slice()),
        ("video", input_video.data.as_slice()),
        ("rope_cos", &cos),
        ("rope_sin", &sin),
    ]);
    let rlx_lr = Tensor::from_vec(outs[0].clone(), rust_lr.shape.clone());

    let mut enc_compiled = session.compile(tribev2::model_rlx::graph::build_tribe_encoder_graph(
        &spec,
        &mod_order,
    ));
    apply_params(&mut enc_compiled, &params);
    let enc_out = enc_compiled.run(&[
        ("text", input_text.data.as_slice()),
        ("audio", input_audio.data.as_slice()),
        ("video", input_video.data.as_slice()),
        ("rope_cos", &cos),
        ("rope_sin", &sin),
    ]);
    let rlx_enc = Tensor::from_vec(enc_out[0].clone(), enc.shape.clone());
    let mut x2 = rlx_enc.clone();
    let (b, t, h) = (1usize, 20usize, 1152usize);
    x2 = x2.permute(&[0, 2, 1]);
    x2 = x2.permute(&[0, 2, 1]);
    let rust_lr_on_rlx = x2
        .reshape(&[b * t, h])
        .matmul(lr_w)
        .reshape(&[b, t, lr_w.shape[1]])
        .permute(&[0, 2, 1]);

    assert_parity(&rust_lr.data, &rlx_lr.data, "lowrank vs rust");
    let r_ref = compare_slices(&rust_lr.data, &ref_lr.data);
    r_ref.log("lowrank vs python ref");
    assert!(r_ref.pearson > 0.999999, "lowrank vs python ref");
}

#[test]
fn test_rlx_predictor_vs_pure_rust() {
    if !refs_exist() {
        return;
    }
    use tribev2::model_rlx::graph::{build_tribe_lowrank_graph, build_tribe_predictor_graph, TribeSpec};
    use tribev2::model_rlx::weights::{apply_params, build_tribe_params, load_safetensors};

    let input_text = load_ref("input_text.bin");
    let input_audio = load_ref("input_audio.bin");
    let input_video = load_ref("input_video.bin");
    let ref_pred = load_ref_with_header("after_predictor.bin");

    let rust_model = TribeV2::from_pretrained(
        &data_path("config.yaml"),
        &data_path("model.safetensors"),
        Some(&data_path("build_args.json")),
    )
    .unwrap();
    let mut features = BTreeMap::new();
    features.insert("text".to_string(), input_text.clone());
    features.insert("audio".to_string(), input_audio.clone());
    features.insert("video".to_string(), input_video.clone());
    let rust_pred = rust_model.forward(&features, None, false);

    let mod_order: Vec<String> = rust_model.feature_dims.iter().map(|m| m.name.clone()).collect();
    let mut raw = load_safetensors(&data_path("model.safetensors")).unwrap();
    let spec = TribeSpec::new(1, 20, 6144, 2048, 2816, 20484);
    let params = build_tribe_params(&mut raw, &spec).unwrap();
    let (cos, sin) = tribev2::model_rlx::rope::rope_cos_sin(20, spec.rot_dim);
    let session = rlx::Session::new(rlx::Device::Cpu);
    let mut lr_compiled = session.compile(build_tribe_lowrank_graph(&spec, &mod_order));
    apply_params(&mut lr_compiled, &params);
    let lr_out = lr_compiled.run(&[
        ("text", input_text.data.as_slice()),
        ("audio", input_audio.data.as_slice()),
        ("video", input_video.data.as_slice()),
        ("rope_cos", &cos),
        ("rope_sin", &sin),
    ]);
    let mut pred_compiled = session.compile(build_tribe_predictor_graph(&spec));
    apply_params(&mut pred_compiled, &params);
    let pred_out = pred_compiled.run(&[("x", lr_out[0].as_slice())]);
    let rlx_pred = Tensor::from_vec(pred_out[0].clone(), rust_pred.shape.clone());

    assert_parity(&rust_pred.data, &rlx_pred.data, "predictor vs rust");
}

#[test]
fn test_rlx_vs_pure_rust() {
    if !refs_exist() {
        eprintln!("SKIP: parity refs / weights not found under {}", data_dir().display());
        return;
    }

    let input_text = load_ref("input_text.bin");
    let input_audio = load_ref("input_audio.bin");
    let input_video = load_ref("input_video.bin");

    let rust_model = TribeV2::from_pretrained(
        &data_path("config.yaml"),
        &data_path("model.safetensors"),
        Some(&data_path("build_args.json")),
    )
    .unwrap();

    let mut rlx_model = TribeRlx::from_pretrained(
        &data_path("config.yaml"),
        &data_path("model.safetensors"),
        Some(&data_path("build_args.json")),
    )
    .unwrap();

    let mut features = BTreeMap::new();
    features.insert("text".to_string(), input_text.clone());
    features.insert("audio".to_string(), input_audio.clone());
    features.insert("video".to_string(), input_video.clone());

    let rust_out = rust_model.forward(&features, None, true);
    let rust_pred = rust_model.forward(&features, None, false);
    let rlx_out = rlx_model.forward(&features, None, true);

    assert_eq!(rust_out.shape, rlx_out.shape, "output shape mismatch");
    let rlx_pred = rlx_model.forward(&features, None, false);
    assert_parity(&rust_pred.data, &rlx_pred.data, "predictor unpooled vs rust");
    assert_parity(&rust_out.data, &rlx_out.data, "full forward vs rust");
}

#[test]
fn test_rlx_vs_python_final_output() {
    let final_ref = refs_dir().join("final_output.bin");
    if !final_ref.exists() {
        eprintln!("SKIP: {} not found", final_ref.display());
        return;
    }
    if !refs_exist() {
        eprintln!("SKIP: weights not found");
        return;
    }

    let input_text = load_ref("input_text.bin");
    let input_audio = load_ref("input_audio.bin");
    let input_video = load_ref("input_video.bin");
    let python_out = load_ref_with_header("final_output.bin");

    let mut rlx_model = TribeRlx::from_pretrained(
        &data_path("config.yaml"),
        &data_path("model.safetensors"),
        Some(&data_path("build_args.json")),
    )
    .unwrap();

    let mut features = BTreeMap::new();
    features.insert("text".to_string(), input_text);
    features.insert("audio".to_string(), input_audio);
    features.insert("video".to_string(), input_video);

    let rlx_out = rlx_model.forward(&features, None, true);
    assert_eq!(rlx_out.shape, python_out.shape);

    assert_parity(&rlx_out.data, &python_out.data, "final output vs python");
}

fn load_ref_with_header(name: &str) -> Tensor {
    let path = refs_dir().join(name);
    let path = path.to_string_lossy().into_owned();
    let bytes = std::fs::read(&path).unwrap();
    let ndims = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let mut shape = Vec::with_capacity(ndims);
    let mut offset = 4;
    for _ in 0..ndims {
        let d = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        shape.push(d);
        offset += 4;
    }
    let n_floats: usize = shape.iter().product();
    let data: Vec<f32> = (0..n_floats)
        .map(|i| f32::from_le_bytes(bytes[offset + i * 4..offset + i * 4 + 4].try_into().unwrap()))
        .collect();
    Tensor::from_vec(data, shape)
}
