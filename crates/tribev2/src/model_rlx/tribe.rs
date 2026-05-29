//! RLX-backed TRIBE v2 encoder.

use std::collections::{BTreeMap, HashMap};

use crate::config::{BrainModelConfig, ModalityDims, ModelBuildArgs, TribeV2Config};
use crate::model_rlx::graph::{build_tribe_graph, TribeSpec};
use crate::model_rlx::rope::rope_cos_sin;
use crate::model_rlx::weights::{apply_params, build_tribe_params, load_safetensors, ParamMap};
use crate::tensor::Tensor;

/// TRIBE v2 brain model on RLX.
pub struct TribeRlx {
    pub n_outputs: usize,
    pub n_output_timesteps: usize,
    pub config: BrainModelConfig,
    pub feature_dims: Vec<ModalityDims>,
    text_in: usize,
    audio_in: usize,
    video_in: usize,
    modality_order: Vec<String>,
    device: rlx::Device,
    params: ParamMap,
    session: rlx::Session,
    cache: HashMap<(usize, usize), rlx::CompiledGraph>,
}

impl TribeRlx {
    pub fn from_pretrained(
        config_path: &str,
        weights_path: &str,
        build_args_path: Option<&str>,
    ) -> anyhow::Result<Self> {
        let yaml = std::fs::read_to_string(config_path)?;
        let mut config: TribeV2Config = serde_yaml::from_str(&yaml)?;
        if let Some(ref mut sl) = config.brain_model_config.subject_layers {
            sl.average_subjects = true;
            sl.n_subjects = 0;
        }

        let (feature_dims, n_outputs, n_output_timesteps) = if let Some(ba_path) =
            build_args_path
        {
            let ba = ModelBuildArgs::from_json(ba_path)?;
            (
                ba.to_modality_dims(),
                ba.n_outputs,
                ba.n_output_timesteps,
            )
        } else {
            (ModalityDims::pretrained(), 20484, config.data.duration_trs)
        };

        let mod_in = |name: &str| -> usize {
            feature_dims
                .iter()
                .find(|m| m.name == name)
                .and_then(|m| m.dims.map(|(l, d)| l * d))
                .unwrap_or(0)
        };
        let text_in = mod_in("text");
        let audio_in = mod_in("audio");
        let video_in = mod_in("video");
        let modality_order: Vec<String> = feature_dims.iter().map(|m| m.name.clone()).collect();

        let device = rlx::Device::Cpu;
        let mut raw = load_safetensors(weights_path)?;
        let spec = TribeSpec::new(1, 20, text_in, audio_in, video_in, n_outputs);
        let params = build_tribe_params(&mut raw, &spec)?;
        let session = rlx::Session::new(device);

        Ok(Self {
            n_outputs,
            n_output_timesteps,
            config: config.brain_model_config,
            feature_dims,
            text_in,
            audio_in,
            video_in,
            modality_order,
            device,
            params,
            session,
            cache: HashMap::new(),
        })
    }

    pub fn with_device(mut self, device: rlx::Device) -> Self {
        self.device = device;
        self.session = rlx::Session::new(device);
        self.cache.clear();
        self
    }

    fn compiled_for(&mut self, b: usize, t: usize) -> &mut rlx::CompiledGraph {
        let key = (b, t);
        if !self.cache.contains_key(&key) {
            let spec = TribeSpec::new(
                b,
                t,
                self.text_in,
                self.audio_in,
                self.video_in,
                self.n_outputs,
            );
            let graph = build_tribe_graph(&spec, &self.modality_order);
            let mut compiled = self.session.compile(graph);
            apply_params(&mut compiled, &self.params);
            self.cache.insert(key, compiled);
        }
        self.cache.get_mut(&key).unwrap()
    }

    /// Forward pass matching [`crate::model::tribe::TribeV2::forward`].
    pub fn forward(
        &mut self,
        features: &BTreeMap<String, Tensor>,
        _subject_ids: Option<&[usize]>,
        pool_outputs: bool,
    ) -> Tensor {
        let first = features.values().next().expect("no features");
        let b = first.shape[0];
        let t = *first.shape.last().unwrap();
        let spec = TribeSpec::new(
            b,
            t,
            self.text_in,
            self.audio_in,
            self.video_in,
            self.n_outputs,
        );

        let text = features
            .get("text")
            .map(|x| x.data.as_slice())
            .unwrap_or(&[] as &[f32]);
        let audio = features
            .get("audio")
            .map(|x| x.data.as_slice())
            .unwrap_or(&[] as &[f32]);
        let video = features
            .get("video")
            .map(|x| x.data.as_slice())
            .unwrap_or(&[] as &[f32]);

        let (cos, sin) = rope_cos_sin(t, spec.rot_dim);
        let compiled = self.compiled_for(b, t);
        let outs = compiled.run(&[
            ("text", text),
            ("audio", audio),
            ("video", video),
            ("rope_cos", &cos),
            ("rope_sin", &sin),
        ]);
        let out = outs.into_iter().next().expect("graph output");
        let mut x = Tensor::from_vec(out, vec![b, spec.n_outputs, t]);

        if pool_outputs {
            x = adaptive_avg_pool1d(&x, self.n_output_timesteps);
        }
        x
    }
}

/// Match PyTorch `AdaptiveAvgPool1d` pooling over the last axis.
fn adaptive_avg_pool1d(x: &Tensor, out_t: usize) -> Tensor {
    let (b, d, t_in) = (x.shape[0], x.shape[1], x.shape[2]);
    let mut data = vec![0.0f32; b * d * out_t];
    for bi in 0..b {
        for di in 0..d {
            let base = bi * d * t_in + di * t_in;
            for i in 0..out_t {
                let start = (i * t_in) / out_t;
                let end = ((i + 1) * t_in + out_t - 1) / out_t;
                let len = (end - start).max(1) as f32;
                let sum: f32 = (start..end)
                    .map(|ti| x.data[base + ti])
                    .sum();
                data[bi * d * out_t + di * out_t + i] = sum / len;
            }
        }
    }
    Tensor::from_vec(data, vec![b, d, out_t])
}
