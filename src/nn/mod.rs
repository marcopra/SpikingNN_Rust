use crate::Model;

use crate::matrix::Matrix;

use self::model::Layer;

pub mod model;
pub(crate) mod builder;
pub(crate) mod neuron;

/// Represents the 'spike' that stimulates a neuron in a spiking neural network.
#[derive(Clone, Copy)]
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
pub struct NN<M: Model> {
    /// Input weight for each of the `Neuron`s in the entry layer
    input_weights: Vec<M::Synapse>,
    /// All the layers of the neural network. Every layer contains the list of its `Neuron`s and
    /// a square `Matrix` for the intra-layer weights.
    layers: Vec<Layer<M>>,
    /// Vec of `Synapse` meshes between each consecutive pair of layers
    synapses: Vec<Matrix<M::Synapse>>
}
