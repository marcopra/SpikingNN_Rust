use std::sync::mpsc::{Receiver, Sender};

use ndarray::Array2;

use crate::{nn::model::Layer, Model};

pub(crate) struct LayerManager<'a, M: Model> {
    layer: &'a Layer<M>,
    vars: Vec<M::SolverVars>,
    receiver: Receiver<(u128, Array2<f64>)>,
    sender: Sender<(u128, Array2<f64>)>
}

impl<'a, M: Model> LayerManager<'a, M> where for<'b> &'b M::Neuron: Into<M::SolverVars> {
    pub fn new(layer: &'a Layer<M>, receiver: Receiver<(u128, Array2<f64>)>, sender: Sender<(u128, Array2<f64>)>) -> Self {
        let vars = layer.neurons.iter().map(|neuron| neuron.into()).collect();
    
        Self {
            layer,
            vars,
            receiver,
            sender
        }
    }

    pub fn run(mut self) {
        for (ts, spike) in self.receiver {
            let mut weighted_inputs = spike.dot(&self.layer.input_weights);

            loop {
                let mut spiked = false;
                let mut output = Array2::zeros((1, self.layer.neurons.len()));
                
                for (neuron_id, (neuron, vars)) in self.layer.neurons.iter().zip(self.vars.iter_mut()).enumerate() {
                    let input = weighted_inputs[(0, neuron_id)];

                    if input != 0.0 {
                        let o = M::handle_spike(neuron, vars, input, ts);
                        spiked |= o > 0.5; // TODO: do we really want this?
                        output[(0, neuron_id)] = o;
                    }
                }

                if spiked {
                    weighted_inputs = output.dot(&self.layer.intra_weights);
                    self.sender.send((ts, output)).unwrap();
                } else {
                    break;
                }
            }
        }
    }
}
