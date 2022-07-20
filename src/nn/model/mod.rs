pub mod lif;

use crate::{NN, matrix::Matrix};
use super::Spike;

/// An applicable model for spiking neural networks
pub trait Model {
    /// A struct for a single Neuron of the SNN.
    /// Each Neuron has its own parametres such as _current membrane tension_, _threshold tension_ etc...
    type Neuron: Neuron;

    /// A link between two `Neuron`s.
    /// 
    /// This is the 'weight' of the connection; it can be positive, negative or null.
    type Synapse: Clone;
}

pub trait Neuron: Sized + Clone {
    /// Helper type to build neurons
    type Config: Into<Self>;

    /// Handle the incoming spike
    fn handle_spike<M>(spike: Spike, nn: &NN<M>) -> Option<Spike>
        where M: Model<Neuron = Self>;
}

pub(super) type Layer<M> = (Vec<<M as Model>::Neuron>, Matrix<<M as Model>::Synapse>);
