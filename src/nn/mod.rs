use crate::Neuron;

use crate::matrix::Matrix;

pub(crate) mod builder;
pub(crate) mod neuron;

/// A link between two `Neuron`s.
/// 
/// This is the 'weight' of the connection; it can be positive, negative or null.
pub type Synapse = f64;

/// Represents the 'spike' that stimulates a neuron in a spiking neural network.
pub struct Spike {
    // TODO
}

/// The Neural Network itself.
/// 
/// This organizes `Neuron`s into consecutive layers, each constituted of some amount of `Neuron`s.
/// `Neuron`s of the same or consecutive layers are connected by a weighted `Synapse`.
/// 
/// A neural network is stimulated by `Spike`s applied to the `Neuron`s of the entry layer.
#[derive(Clone)]
pub struct NN {
    /// Input weight for each of the `Neuron`s in the entry layer
    input_weights: Vec<Synapse>,
    /// All the layers of the neural network. Every layer contains the list of its `Neuron`s and
    /// a square `Matrix` for the intra-layer weights.
    layers: Vec<(Vec<Neuron>, Matrix<Synapse>)>,
    /// Vec of `Synapse` meshes between each consecutive pair of layers
    synapses: Vec<Matrix<Synapse>>
}
