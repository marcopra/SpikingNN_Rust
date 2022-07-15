//! Provides a `NNBuilder` type to create neural networks.
//! Building a neural network through a `NNBuilder` can not fail thanks to
//! the robust integration with Rust's type system, which allows us to
//! test validity at compile time.

use std::marker::PhantomData;

use crate::NN;

use super::Neuron;

pub trait Dim { }

struct Zero { }
impl Dim for Zero { }

struct NotZero<const N: usize> { }
impl<const N: usize> Dim for NotZero<N> { }

/// Helper type that implements the builder pattern for `NN`.
/// 
/// # Examples
/// 
/// Creating a simple 2-layers NN:
/// 
/// ```
/// let nn = NNBuilder::new()
///     // Insert entry layer
///     .layer(
///         &[Neuron{}, Neuron{}],
///         &[0.1, 3.0],
///         &[
///             [0.0, -0.3],
///             [-1.5, 0.0]
///         ]
///     )
///     // Insert secondary (and exit) layer
///     .layer(
///         &[Neuron{}, Neuron{}, Neuron{}],
///         &[
///             [1.1, 2.2],
///             [3.3, 4.4],
///             [5.5, 6.6]
///         ],
///         &[
///             [0.0, -0.1, -0.2],
///             [-0.3, 0.0, -0.4],
///             [-0.5, -0.6, 0.0]
///         ]
///     )
///     .build();
/// ```
#[derive(Clone, Default)]
pub struct NNBuilder<D: Dim> {
    /// Needed because of D, which would otherwise be unused
    _phantom: PhantomData<D>,
    // TODO
}

// Note that we allow creation only of NNBuilders with LEN_LAST_LAYER equal to zero
impl NNBuilder<Zero> {
    /// Create a new instance of `NNBuilder`.
    /// Every instance of this type can be used to build one `NN`.
    pub fn new() -> Self {
        Self { _phantom: PhantomData }
    }

    /// Add the entry layer to the neural network.
    /// Note: diagonal intra-weights (i.e. from and to the same neuron) are ignored.
    pub fn layer<const N: usize>(
        &mut self,
        _layer: &[Neuron; N],
        _input_weights: &[f64; N],
        _intra_weights: &[[f64; N]; N]
    ) -> NNBuilder<NotZero<N>>
    {
        todo!("Add entry layer to empty NNBuilder")
    }
}

impl<const LEN_LAST_LAYER: usize> NNBuilder<NotZero<LEN_LAST_LAYER>> {
    /// Add a layer to the neural network.
    /// Note: diagonal intra-weights (i.e. from and to the same neuron) are ignored.
    pub fn layer<const N: usize>(
        &mut self,
        _layer: &[Neuron; N],
        _input_weights: &[[f64; LEN_LAST_LAYER]; N],
        _intra_weights: &[[f64; N]; N]
    ) -> NNBuilder<NotZero<N>>
    {
        todo!("Add layer to NNBuilder")
    }

    /// Build the `NN`
    pub fn build(&self) -> NN {
        todo!("Build NN from builder")
    }
}
