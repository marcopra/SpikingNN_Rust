use crate::Model;

#[derive(Clone, Debug)]
pub struct Neuron {
    // TODO...
}

#[derive(Clone, Debug)]
pub struct NeuronConfig {
    // TODO
}

impl From<NeuronConfig> for Neuron {
    fn from(_: NeuronConfig) -> Self {
        todo!()
    }
}

impl super::Neuron for Neuron {
    type Config = NeuronConfig;

    fn handle_spike<M>(spike: crate::nn::Spike, nn: &crate::NN<M>) -> Option<crate::nn::Spike>
    where M: crate::Model<Neuron = Self>
    {
        todo!()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LeakyIntegrateFire;

impl Model for LeakyIntegrateFire {
    type Neuron = Neuron;
    type Synapse = f64;
}
