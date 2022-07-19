//! Provides a `NNBuilder` type to create neural networks.
//! 
//! Building a neural network through a `NNBuilder` can not fail thanks to
//! the robust integration with Rust's type system, which allows us to
//! test validity at compile time.
//! 
//! Additionally, a dynamically checked variant is supplied for building neural networks whose
//! dimensions are not known at compile time.

// TODO: intra-weights square matrix? Should we request a Nx(N-1) matrix to dismiss the useless self weight?

use std::{marker::PhantomData, borrow::Borrow, fmt::Debug};
use thiserror::Error;
use crate::{NN, Synapse, Neuron, matrix::Matrix};
pub trait Dim: Copy { }

#[derive(Clone, Copy)]
pub struct Zero;
impl Dim for Zero { }

#[derive(Clone, Copy)]
pub struct NotZero<const N: usize>;
impl<const N: usize> Dim for NotZero<N> { }

#[derive(Clone, Copy)]
pub struct Dynamic;
impl Dim for Dynamic { }

/// An error type for the dynamic variant of NNBuilder.
/// All the error variants contain the builder that generated them, for reuse.
#[derive(Error, Debug)]
pub enum DynamicBuilderError {
    #[error("Empty builder can not be built")]
    EmptyNN(NNBuilder<Dynamic>),

    #[error("Invalid input sizes provided for layer")]
    InvalidSizes(NNBuilder<Dynamic>)
}

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
#[derive(Clone)]
pub struct NNBuilder<D: Dim> {
    /// Inner, growing `NN`
    nn: NN,
    /// Needed because of D, which would otherwise be unused
    _phantom: PhantomData<D>,
}

impl NNBuilder<Dynamic> {
    /// Create a new dynamically sized instance of `NNBuilder`.
    /// Every instance of this type can be used to build one `NN`.
    /// 
    /// In a dynamic `NNBuilder` size checks are performed at runtime, allowing for creation of `NN`s whose
    /// size is not known at compile time, at the small cost of the checks necessary to ensure its validity.
    pub fn new_dynamic() -> Self {
        Self { nn: Self::new_nn(), _phantom: PhantomData }
    }

    /// Add a layer to the neural network.
    /// 
    /// Note: input and intra weights are flattened row-major matrices (one row for each neuron in the layer).
    pub fn layer(
        mut self,
        layer: impl Borrow<[Neuron]>,
        input_weights: impl Borrow<[Synapse]>,
        intra_weights: impl Borrow<[Synapse]>
    ) -> Result<Self, DynamicBuilderError>
    {
        let len_last_layer = self.nn.layers.last().map(|(l, _)| l.len()).unwrap_or(0);
        let n = layer.borrow().len();
        
        // Check layer len not zero
        if n == 0 {
            return Err(DynamicBuilderError::InvalidSizes(self));
        }

        // Check size compatibilities
        if intra_weights.borrow().len() != n*n {
            return Err(DynamicBuilderError::InvalidSizes(self));
        }

        if input_weights.borrow().len() != (
            if len_last_layer == 0 { n } else { len_last_layer * n }
        ) {
            return Err(DynamicBuilderError::InvalidSizes(self));
        }

        // Finally, insert layer into nn
        self.nn.layers.push((layer.borrow().to_vec(), Matrix::from_raw_data(n, n, intra_weights.borrow().to_vec())));

        if len_last_layer == 0 {
            self.nn.input_weights = input_weights.borrow().to_vec();
        } else {
            self.nn.synapses.push(Matrix::from_raw_data(len_last_layer, n, input_weights.borrow().to_vec()));
        }
        
        Ok(self)
    }

    /// Build the `NN`
    pub fn build(self) -> Result<NN, DynamicBuilderError> {
        if self.nn.layers.is_empty() {
            Err(DynamicBuilderError::EmptyNN(self))
        } else {
            Ok(self.inner_build())
        }
    }
}

impl NNBuilder<Zero> {
    /// Create a new statically sized instance of `NNBuilder`.
    /// Every instance of this type can be used to build one `NN`.
    /// 
    /// In a static `NNBuilder` size checks are performed at compile time for maximum efficiency.
    /// This has the obvious drawback of requiring that the sizes of the resulting `NN` be known at compile time as well.
    pub fn new() -> Self {
        Self { nn: Self::new_nn(), _phantom: PhantomData }
    }

    /// Add the entry layer to the neural network.
    /// 
    /// Note: diagonal intra-weights (i.e. from and to the same neuron) are ignored.
    pub fn layer<const N: usize>(
        mut self,
        layer: impl Borrow<[Neuron; N]>,
        input_weights: impl Borrow<[Synapse; N]>,
        intra_weights: impl Borrow<[[Synapse; N]; N]>
    ) -> NNBuilder<NotZero<N>>
    {
        // Insert input weights
        self.nn.input_weights = input_weights.borrow().to_vec();

        // Insert layer
        self.nn.layers.push((layer.borrow().to_vec(), intra_weights.borrow().into()));
        
        self.morph()
    }
}

impl<const LEN_LAST_LAYER: usize> NNBuilder<NotZero<LEN_LAST_LAYER>> {
    /// Add a layer to the neural network.
    /// 
    /// Note: diagonal intra-weights (i.e. from and to the same neuron) are ignored.
    pub fn layer<const N: usize>(
        mut self,
        layer: impl Borrow<[Neuron; N]>,
        input_weights: impl Borrow<[[Synapse; LEN_LAST_LAYER]; N]>,
        intra_weights: impl Borrow<[[Synapse; N]; N]>
    ) -> NNBuilder<NotZero<N>>
    {
        // Insert layer
        self.nn.layers.push((layer.borrow().to_vec(), intra_weights.borrow().into()));

        // Insert input synapse mesh
        self.nn.synapses.push(input_weights.borrow().into());

        self.morph()
    }

    /// Build the `NN`
    pub fn build(self) -> NN {
        self.inner_build()
    }
}

impl<D: Dim> NNBuilder<D> {
    /// Create a new, empty `NN`
    fn new_nn() -> NN {
        NN {
            input_weights: vec![],
            layers: vec![],
            synapses: vec![]
        }
    }

    /// Morph into another diensionality variant
    fn morph<E: Dim>(self) -> NNBuilder<E> {
        NNBuilder { nn: self.nn, _phantom: PhantomData }
    }

    /// Build the `NN`.
    /// Note: we don't expose a global 'build' in order to:
    ///  - not allow building NNBuilder<Zero> variants
    ///  - allow checking dimensions at runtime for NNBuilder<Dynamic> variants
    fn inner_build(self) -> NN {
        self.nn
    }
}

impl Default for NNBuilder<Zero> {
    fn default() -> Self {
        Self::new()
    }
}

impl Debug for NNBuilder<Dynamic> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NNBuilder").finish()
    }
}
