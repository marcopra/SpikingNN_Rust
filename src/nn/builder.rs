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
use ndarray::{Array2, Array1};
use thiserror::Error;
use crate::{NN, Model};

use super::model::Layer;
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
pub enum DynamicBuilderError<M: Model> {
    #[error("Empty builder can not be built")]
    EmptyNN(NNBuilder<M, Dynamic>),

    #[error("Invalid input sizes provided for layer")]
    InvalidSizes(NNBuilder<M, Dynamic>)
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
pub struct NNBuilder<M: Model, D: Dim> {
    /// Inner, growing `NN`
    nn: NN<M>,
    /// Needed because of D, which would otherwise be unused
    _phantom: PhantomData<D>,
}

impl<M: Model> NNBuilder<M, Dynamic> {
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
        neurons: impl Borrow<[M::Neuron]>,
        input_weights: impl Borrow<[f64]>,
        intra_weights: impl Borrow<[f64]>
    ) -> Result<Self, DynamicBuilderError<M>>
    {
        let len_last_layer = self.nn.layers.last().map(|l| l.neurons.len()).unwrap_or(0);
        let n = neurons.borrow().len();
        
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

        let input_weights = if len_last_layer == 0 {
            Array2::from_diag(&Array1::from_vec(input_weights.borrow().to_vec()))
        } else {
            Array2::from_shape_vec((len_last_layer, n), input_weights.borrow().to_vec()).unwrap()
        };

        // Finally, insert layer into nn
        let new_layer = Layer {
            neurons: neurons.borrow().to_vec(),
            input_weights,
            intra_weights: Array2::from_shape_vec((n, n), intra_weights.borrow().to_vec()).unwrap()
        };
        self.nn.layers.push(new_layer);

        Ok(self)
    }

    /// Build the `NN`
    pub fn build(self) -> Result<NN<M>, DynamicBuilderError<M>> {
        if self.nn.layers.is_empty() {
            Err(DynamicBuilderError::EmptyNN(self))
        } else {
            Ok(self.inner_build())
        }
    }
}

impl<M: Model> NNBuilder<M, Zero> {
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
        neurons: impl Borrow<[M::Neuron; N]>,
        input_weights: impl Borrow<[f64; N]>,
        intra_weights: impl Borrow<[[f64; N]; N]>
    ) -> NNBuilder<M, NotZero<N>>
    {
        let new_layer = Layer {
            neurons: neurons.borrow().to_vec(),
            input_weights: Array2::from_diag(&Array1::from_vec(input_weights.borrow().to_vec())),
            intra_weights: Array2::from_shape_vec((N, N), intra_weights.borrow().iter().flatten().cloned().collect()).unwrap()
        };
        self.nn.layers.push(new_layer);
        
        self.morph()
    }
}

impl<M: Model, const LEN_LAST_LAYER: usize> NNBuilder<M, NotZero<LEN_LAST_LAYER>> {
    /// Add a layer to the neural network.
    /// 
    /// Note: diagonal intra-weights (i.e. from and to the same neuron) are ignored.
    pub fn layer<const N: usize>(
        mut self,
        neurons: impl Borrow<[M::Neuron; N]>,
        input_weights: impl Borrow<[[f64; N]; LEN_LAST_LAYER]>,
        intra_weights: impl Borrow<[[f64; N]; N]>
    ) -> NNBuilder<M, NotZero<N>>
    {
        let new_layer = Layer {
            neurons: neurons.borrow().to_vec(),
            input_weights: Array2::from_shape_vec((LEN_LAST_LAYER, N), input_weights.borrow().iter().flatten().cloned().collect()).unwrap(),
            intra_weights: Array2::from_shape_vec((N, N), intra_weights.borrow().iter().flatten().cloned().collect()).unwrap()
        };
        self.nn.layers.push(new_layer);
        
        self.morph()
    }

    /// Build the `NN`
    pub fn build(self) -> NN<M> {
        self.inner_build()
    }
}

impl<M: Model, D: Dim> NNBuilder<M, D> {
    /// Create a new, empty `NN`
    fn new_nn() -> NN<M> {
        NN {
            layers: vec![]
        }
    }

    /// Morph into another diensionality variant
    fn morph<E: Dim>(self) -> NNBuilder<M, E> {
        NNBuilder { nn: self.nn, _phantom: PhantomData }
    }

    /// Build the `NN`.
    /// Note: we don't expose a global 'build' in order to:
    ///  - not allow building NNBuilder<Zero> variants
    ///  - allow checking dimensions at runtime for NNBuilder<Dynamic> variants
    fn inner_build(self) -> NN<M> {
        self.nn
    }
}

impl<M: Model> Default for NNBuilder<M, Zero> {
    fn default() -> Self {
        Self::new()
    }
}

impl<M: Model> Debug for NNBuilder<M, Dynamic> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NNBuilder").finish()
    }
}

//TODO 
#[cfg(test)]
mod tests {
    use crate::lif::{LifNeuron, LifNeuronConfig};

    #[test]
    fn test_building_new_nn() {
        let nc = LifNeuronConfig::new(
            0.2,
            0.1, 
            0.45, 
            0.23);

        let _neurons = LifNeuron::new_vec([nc].to_vec(), 3);

        /*let my_nn = NNBuilder::new()
        .layer(
            &[neurons],
            &[0.1, 3.0],
            &[
                [0.0, -0.3],
                [-1.5, 0.0]
            ]
        )
        // Insert 2nd (and exit) layer
        .layer(
            &[Neuron{}, Neuron{}, Neuron{}],
            &[
                [1.1, 2.2],
                [3.3, 4.4],
                [5.5, 6.6]
            ],
            &[
                [0.0, -0.1, -0.2],
                [-0.3, 0.0, -0.4],
                [-0.5, -0.6, 0.0]
            ]);*/
    }

    #[test]
    fn test_access_layers(){

    }

    #[test]
    fn test_access_single_neuron(){
        
    }
} 