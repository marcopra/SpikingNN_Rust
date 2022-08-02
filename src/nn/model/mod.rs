pub mod lif;

use std::{fmt::Debug, ops::{Index, IndexMut}};

use ndarray::Array2;

/// An applicable model for spiking neural networks
pub trait Model: 'static + Debug + Clone {
    /// A struct for a single Neuron of the SNN.
    /// Each Neuron has its own parameters such as _current membrane tension_, _threshold tension_ etc...
    type Neuron: 'static + Sized + Clone + Sync + RefInto<Self::SolverVars>;

    /// Contains the dynamic variables for each Neuron used by the solver
    type SolverVars: Default + Send + Sync;
    
    /// Helper type to build neurons
    type Config: RefInto<Self::Neuron>;

    /// How much a spike weighs. This is used to convert from the bool return type of `handle_spike` to
    /// f64, used for synapses.
    const SPIKE_WEIGHT: f64;

    /// Handle the incoming spike
    fn handle_spike(neuron: &Self::Neuron, vars: &mut Self::SolverVars, weighted_input_val: f64, ts: u128) -> f64;

    fn set_new_params(neuron: &mut Self::Neuron, nc: &Self::Config);

    #[cfg(feature = "simd")]
    type Neuronx4: Send;
    #[cfg(feature = "simd")]
    type SolverVarsx4: Send;
    #[cfg(feature = "simd")]
    fn neuron_x4_from_neurons(neurons: &[Self::Neuron]) -> Self::Neuronx4;
    #[cfg(feature = "simd")]
    fn vars_x4_from_vars(vars: &[Self::SolverVars]) -> Self::SolverVarsx4;
    #[cfg(feature = "simd")]
    fn handle_spike_x4(neurons: &Self::Neuronx4, vars: &mut Self::SolverVarsx4, weighted_input_val: packed_simd::f64x4, ts: u128) -> packed_simd::f64x4;
}

pub trait RefInto<T> { }
impl<T, U> RefInto<T> for U where for<'a> &'a U: Into<T> { }

#[derive(Clone)]
pub struct Layer<M: Model> {
    /// List of all neurons in this layer
    pub(crate) neurons: Vec<M::Neuron>,
    /// Matrix of the input weights. For the first layer, this must be a square diagonal matrix.
    pub(crate) input_weights: Array2<f64>,
    /// Square matrix of the intra-layer weights
    pub(crate) intra_weights: Array2<f64>
}

impl<M: Model> Layer<M> {
    /// Return the number of neurons in this `Layer`
    pub fn num_neurons(&self) -> usize {
        self.neurons.len()
    }

    /// Get the specified neuron, or `None` if the index is out of bounds.
    /// 
    /// An unchecked variant of this functionality is provided via the `Index` implementation.
    pub fn get_neuron(&self, neuron: usize) -> Option<&M::Neuron> {
        self.neurons.get(neuron)
    }

    /// Get a mutable reference to the specified neuron, or `None` if the index is out of bounds.
    /// 
    /// An unchecked variant of this functionality is provided via the `IndexMut` implementation.
    pub fn get_neuron_mut(&mut self, neuron: usize) -> Option<&mut M::Neuron> {
        self.neurons.get_mut(neuron)
    }

    /// Get the intra-layer weight from and to the specified neurons, or `None` if any index is out of bounds.
    /// 
    /// An unchecked variant of this functionality is provided via the `Index` implementation.
    pub fn get_intra_weight(&self, from: usize, to: usize) -> Option<f64> {
        self.intra_weights.get((from, to)).copied()
    }

    /// Get a mutable reference to the intra-layer weight from and to the specified neurons, or `None` if any index is out of bounds.
    /// 
    /// An unchecked variant of this functionality is provided via the `IndexMut` implementation.
    pub fn get_intra_weight_mut(&mut self, from: usize, to: usize) -> Option<&mut f64> {
        self.intra_weights.get_mut((from, to))
    }
}

impl<M: Model> Index<usize> for Layer<M> {
    type Output = M::Neuron;

    fn index(&self, index: usize) -> &Self::Output {
        &self.neurons[index]
    }
}

impl<M: Model> IndexMut<usize> for Layer<M> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.neurons[index]
    }
}

impl<M: Model> Index<(usize, usize)> for Layer<M> {
    type Output = f64;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.intra_weights[index]
    }
}

impl<M: Model> IndexMut<(usize, usize)> for Layer<M> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.intra_weights[index]
    }
}
