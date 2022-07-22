pub mod lif;

use ndarray::Array2;
use crate::NN;

use super::{Spike};

/// An applicable model for spiking neural networks
pub trait Model {
    /// A struct for a single Neuron of the SNN.
    /// Each Neuron has its own parameters such as _current membrane tension_, _threshold tension_ etc...
    type Neuron: Neuron;

    /// A link between two `Neuron`s.
    /// 
    /// This is the 'weight' of the connection; it can be positive, negative or null.
    type Synapse: Clone;
}

pub trait Neuron: Sized + Clone {
    /// Helper type to build neurons
    type Config: RefInto<Self>;

    /// Handle the incoming spike
    fn handle_spike<M>(&mut self, weighted_input_val: f64, nn: &NN<M>) -> Option<Spike>
        where M: Model<Neuron = Self>;

    fn set_new_params(&mut self, nc: &Self::Config);
}

pub trait RefInto<T> { }
impl<T, U> RefInto<T> for U where for<'a> &'a U: Into<T> { }

pub(super) type Layer<M> = (Vec<<M as Model>::Neuron>, Array2<<M as Model>::Synapse>);
