use burn::prelude::*;
use burn::nn::{Linear, LinearConfig};

/// Single linear projector (pretrained TRIBE v2 uses no hidden layers).
#[derive(Module, Debug)]
pub struct Projector<B: Backend> {
    pub linear: Linear<B>,
}

impl<B: Backend> Projector<B> {
    pub fn new(in_dim: usize, out_dim: usize, device: &B::Device) -> Self {
        Self {
            linear: LinearConfig::new(in_dim, out_dim).with_bias(true).init(device),
        }
    }

    /// x: [..., in_dim] → [..., out_dim]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.linear.forward(x)
    }
}
