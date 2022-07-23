pub mod lif;

use ndarray::Array2;

/// An applicable model for spiking neural networks
pub trait Model {
    /// A struct for a single Neuron of the SNN.
    /// Each Neuron has its own parameters such as _current membrane tension_, _threshold tension_ etc...
    type Neuron: Sized + Clone;
    
    /// Helper type to build neurons
    type Config: RefInto<Self::Neuron>;

    /// How much a spike weighs. This is used to convert from the bool return type of `handle_spike` to
    /// f64, used for synapses.
    const SPIKE_WEIGHT: f64;

    /// Handle the incoming spike
    fn handle_spike(neuron: &mut Self::Neuron, weighted_input_val: f64) -> bool;

    fn set_new_params(neuron: &mut Self::Neuron, nc: &Self::Config);
}

pub trait RefInto<T> { }
impl<T, U> RefInto<T> for U where for<'a> &'a U: Into<T> { }

pub(super) type Layer<M> = (Vec<<M as Model>::Neuron>, Array2<f64>);
