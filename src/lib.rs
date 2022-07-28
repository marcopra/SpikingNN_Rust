// Benchmarks can be run with `cargo +nightly bench --features bench`
#![cfg_attr(feature = "bench", feature(test))]

mod nn;

#[cfg(feature = "per-neuron-parallelism")]
mod sync_per_neuron;
#[cfg(feature = "per-neuron-parallelism")]
use sync_per_neuron as sync;
#[cfg(not(feature = "per-neuron-parallelism"))]
mod sync;

// Re-exports
pub use nn::NN;
pub use nn::builder::NNBuilder;
pub use nn::model::Model;
pub use nn::model::lif::*;
