//! TRIBE v2 brain encoder on RLX (`Session` / `CompiledGraph`).
//!
//! Mirrors [`crate::model::tribe::TribeV2`] with the same weight keys and
//! forward math; dynamic shapes (sequence length) are handled by caching
//! one compiled graph per `(batch, timesteps)` pair.

pub mod graph;
pub mod rope;
pub mod tribe;
pub mod weights;

pub use tribe::TribeRlx;
