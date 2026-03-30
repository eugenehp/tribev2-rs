use std::time::Instant;
use burn::prelude::*;
use burn::backend::NdArray as B;
use tribev2_rs::config::*;
use tribev2_rs::model_burn::*;

fn main() {
    let dev = burn::backend::ndarray::NdArrayDevice::Cpu;
    let dim = 1152;
    let t_len = 100;

    let sn = scalenorm::ScaleNorm::<B>::new(dim, &dev);
    let x = Tensor::<B,3>::ones([1, t_len, dim], &dev).mul_scalar(0.1);
    let t0 = Instant::now();
    for _ in 0..10 { let _ = sn.forward(x.clone()); }
    println!("ScaleNorm x10:    {:>8.1} ms", t0.elapsed().as_secs_f64()*100.0);

    let attn = attention::Attention::<B>::new(dim, 8, &dev);
    let t0 = Instant::now();
    for _ in 0..2 { let _ = attn.forward(x.clone(), None); }
    println!("Attention x2:     {:>8.1} ms", t0.elapsed().as_secs_f64()*500.0);

    let ff = feedforward::FeedForward::<B>::new(dim, 4, &dev);
    let t0 = Instant::now();
    for _ in 0..2 { let _ = ff.forward(x.clone()); }
    println!("FeedForward x2:   {:>8.1} ms", t0.elapsed().as_secs_f64()*500.0);

    let proj = projector::Projector::<B>::new(9216, 384, &dev);
    let px = Tensor::<B,3>::ones([1, t_len, 9216], &dev).mul_scalar(0.01);
    let t0 = Instant::now();
    for _ in 0..10 { let _ = proj.forward(px.clone()); }
    println!("Projector x10:    {:>8.1} ms", t0.elapsed().as_secs_f64()*100.0);

    let slc = SubjectLayersConfig { n_subjects: 25, bias: true, subject_dropout: Some(0.1), average_subjects: true, ..Default::default() };
    let sl = subject_layers::SubjectLayers::<B>::new(2048, 20484, &slc, &dev);
    let sx = Tensor::<B,3>::ones([1, 2048, t_len], &dev).mul_scalar(0.01);
    let t0 = Instant::now();
    for _ in 0..2 { let _ = sl.forward_average(sx.clone()); }
    println!("SubjectLayers x2: {:>8.1} ms", t0.elapsed().as_secs_f64()*500.0);

    let big = Tensor::<B,3>::ones([1, 20484, t_len], &dev);
    let t0 = Instant::now();
    let _ = tribe::adaptive_avg_pool1d_pub::<B>(big, 100);
    println!("AvgPool1d x1:     {:>8.1} ms", t0.elapsed().as_secs_f64()*1000.0);
}
