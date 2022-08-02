// Benchmarks can be run with `cargo +nightly bench --features bench`
#![cfg_attr(feature = "bench", feature(test))]

// TODO: switch to links for documentation comments (from `Spike` to [Spike])?

pub mod nn;
mod sync;

// Re-exports
pub use nn::{NN, Spike};
pub use nn::builder::NNBuilder;
pub use nn::model::{Model, Layer};
pub use nn::model::lif;
