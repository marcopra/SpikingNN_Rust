#[cfg(feature = "async")]
use tokio::sync::mpsc::{Receiver, Sender};
// use tokio::sync::mpsc::{UnboundedReceiver as Receiver, UnboundedSender as Sender};
#[cfg(not(feature = "async"))]
use std::sync::mpsc::{Receiver, Sender};

use ndarray::Array2;

use crate::{nn::model::Layer, Model};

pub(crate) struct LayerManager<'a, M: Model> {
    layer: &'a Layer<M>,
    vars: Vec<M::SolverVars>,
    receiver: Receiver<(u128, Array2<f64>)>,
    sender: Sender<(u128, Array2<f64>)>,
}

impl<'a, M: Model> LayerManager<'a, M> where for<'b> &'b M::Neuron: Into<M::SolverVars> {
    pub fn new(
        layer: &'a Layer<M>,
        receiver: Receiver<(u128, Array2<f64>)>,
        sender: Sender<(u128, Array2<f64>)>, 
    ) -> Self
    {
        let vars = layer.neurons.iter().map(|neuron| neuron.into()).collect();
    
        Self {
            layer,
            vars,
            receiver,
            sender,
        }
    }

    #[cfg(not(feature = "async"))]
    pub fn run(mut self) {
        for (ts, spike) in self.receiver {
            let mut weighted_inputs = spike.dot(&self.layer.input_weights);

            loop {
                let mut spiked = false;
                
                let output = Array2::from_shape_fn((1, self.layer.neurons.len()), |(_, neuron_id)| {
                    let o = M::handle_spike(
                        &self.layer.neurons[neuron_id],
                        &mut self.vars[neuron_id],
                        weighted_inputs[(0, neuron_id)],
                        ts
                    );
                    spiked |= o > 0.5;
                    o
                });

                if spiked {
                    weighted_inputs = output.dot(&self.layer.intra_weights);
                    self.sender.send((ts, output)).unwrap();
                } else {
                    break;
                }
            }
        }
    }

    #[cfg(feature = "async")]
    pub async fn run(mut self) {
        let mut weighted_inputs;
        
        while let Some((ts, spike)) = self.receiver.recv().await {
            weighted_inputs = spike.dot(&self.layer.input_weights);

            loop {
                let mut spiked = false;

                let output = Array2::from_shape_fn((1, self.layer.neurons.len()), |(_, neuron_id)| {
                    let o = M::handle_spike(
                        &self.layer.neurons[neuron_id],
                        &mut self.vars[neuron_id],
                        weighted_inputs[(0, neuron_id)],
                        ts
                    );
                    spiked |= o > 0.5;
                    o
                });
                
                if spiked {
                    weighted_inputs = output.dot(&self.layer.intra_weights);
                    self.sender.send((ts, output)).await.unwrap();
                } else {
                    break;
                }
            }
        }
    }
}
