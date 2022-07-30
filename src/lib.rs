// Benchmarks can be run with `cargo +nightly bench --features bench`
#![cfg_attr(feature = "bench", feature(test))]

mod nn;
mod sync;

// Re-exports
pub use nn::NN;
pub use nn::builder::NNBuilder;
pub use nn::model::Model;
pub use nn::model::lif::*;
