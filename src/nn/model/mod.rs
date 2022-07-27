pub mod lif;

use std::fmt::Debug;

use ndarray::Array2;

/// An applicable model for spiking neural networks
pub trait Model: Debug {
    /// A struct for a single Neuron of the SNN.
    /// Each Neuron has its own parameters such as _current membrane tension_, _threshold tension_ etc...
    type Neuron: 'static + Sized + Clone + Sync + RefInto<Self::SolverVars>;

    /// Contains the dynamic variables for each Neuron used by the solver
    type SolverVars: Default;
    
    /// Helper type to build neurons
    type Config: RefInto<Self::Neuron>;

    /// How much a spike weighs. This is used to convert from the bool return type of `handle_spike` to
    /// f64, used for synapses.
    const SPIKE_WEIGHT: f64;

    /// Handle the incoming spike
    fn handle_spike(neuron: &Self::Neuron, vars: &mut Self::SolverVars, weighted_input_val: f64, ts: u128) -> f64;

    fn set_new_params(neuron: &mut Self::Neuron, nc: &Self::Config);
}

pub trait RefInto<T> { }
impl<T, U> RefInto<T> for U where for<'a> &'a U: Into<T> { }

#[derive(Clone)]
pub(crate) struct Layer<M: Model> {
    /// List of all neurons in this layer
    pub neurons: Vec<M::Neuron>,
    /// Matrix of the input weights. For the first layer, this must be a square diagonal matrix.
    pub input_weights: Array2<f64>,
    /// Square matrix of the intra-layer weights
    pub intra_weights: Array2<f64>
}
