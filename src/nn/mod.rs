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
/// A neural network is stimulated by `Impulse`s applied to the `Neuron`s of the entry layer.
pub struct NN {
    // TODO
}
