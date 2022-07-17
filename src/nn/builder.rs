//! Provides a `NNBuilder` type to create neural networks.
//! 
//! Building a neural network through a `NNBuilder` can not fail thanks to
//! the robust integration with Rust's type system, which allows us to
//! test validity at compile time.

use std::marker::PhantomData;

use crate::{NN, Synapse};
// use crate::nn::neuron::Neuron è la stessa cosa della riga sotto?
use super::neuron::Neuron;
use super::neuron::NeuronConfig;

pub trait Dim { }

struct Zero;
impl Dim for Zero { }

struct NotZero<const N: usize>;
impl<const N: usize> Dim for NotZero<N> { }

struct Dynamic;
impl Dim for Dynamic { }

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
///     // Insert 2nd (and exit) layer
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

impl NNBuilder<Dynamic> {
    /// Create a new dynamically sized instance of `NNBuilder`.
    /// Every instance of this type can be used to build one `NN`.
    /// 
    /// In a dynamic `NNBuilder` size checks are performed at runtime, allowing for creation of `NN`s whose
    /// size is not known at compile time, at the small cost of the checks necessary to ensure its validity.
    pub fn new() -> Self {
        Self { _phantom: PhantomData }
    }

    /// Add a layer to the neural network.
    /// 
    /// Note: input and intra weights are flattened row-major matrices (one row for each neuron in the layer).
    pub fn layer(&mut self, _layer: &[Neuron], _input_weights: &[Synapse], _intra_weights: &[Synapse]) -> Self {
        todo!("Perform size checks and add layer to dynamic NNBuilder")
    }
}

impl NNBuilder<Zero> {
    /// Create a new statically sized instance of `NNBuilder`.
    /// Every instance of this type can be used to build one `NN`.
    /// 
    /// In a static `NNBuilder` size checks are performed at compile time for maximum efficiency.
    /// This has the obvious drawback of requiring that the sizes of the resulting `NN` be known at compile time as well.
    pub fn new() -> Self {
        Self { _phantom: PhantomData }
    }

    /// Add the entry layer to the neural network.
    /// 
    /// Note: diagonal intra-weights (i.e. from and to the same neuron) are ignored.
    pub fn layer<const N: usize>(
        &mut self,
        _layer: &[Neuron; N],
        _input_weights: &[Synapse; N],
        _intra_weights: &[[Synapse; N]; N]
    ) -> NNBuilder<NotZero<N>>
    {
        todo!("Add entry layer to empty static NNBuilder")
    }
}

impl<const LEN_LAST_LAYER: usize> NNBuilder<NotZero<LEN_LAST_LAYER>> {
    /// Add a layer to the neural network.
    /// 
    /// Note: diagonal intra-weights (i.e. from and to the same neuron) are ignored.
    pub fn layer<const N: usize>(
        &mut self,
        _layer: &[Neuron; N],
        _input_weights: &[[Synapse; LEN_LAST_LAYER]; N],
        _intra_weights: &[[Synapse; N]; N]
    ) -> NNBuilder<NotZero<N>>
    {
        todo!("Add layer to static NNBuilder")
    }
}

impl<D: Dim> NNBuilder<D> {
    /// Build the `NN`
    pub fn build(self) -> NN {
        todo!("Build NN from NNBuilder")
    }
}
