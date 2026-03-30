use burn::prelude::*;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::gelu;

/// FeedForward from x_transformers: Linear→GELU→Linear (with bias).
#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    pub fc1: Linear<B>,
    pub fc2: Linear<B>,
}

impl<B: Backend> FeedForward<B> {
    pub fn new(dim: usize, mult: usize, device: &B::Device) -> Self {
        let inner = dim * mult;
        Self {
            fc1: LinearConfig::new(dim, inner).with_bias(true).init(device),
            fc2: LinearConfig::new(inner, dim).with_bias(true).init(device),
        }
    }

    /// x: [B, N, D] → [B, N, D]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.fc2.forward(gelu(self.fc1.forward(x)))
    }
}
