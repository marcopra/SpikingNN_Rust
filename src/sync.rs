#[cfg(feature = "async")]
use tokio::sync::mpsc::{Receiver, Sender};
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

    #[cfg(all(not(feature = "async"), not(feature = "simd")))]
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

    #[cfg(all(not(feature = "async"), feature = "simd"))]
    pub fn run(mut self) {
        use packed_simd::f64x4;

        let (neurons, neuron_remainder) = {
            let chunks = self.layer.neurons.chunks_exact(4);
            let remainder = chunks.remainder();
            
            (
                chunks.into_iter().map(|chunk| M::neuron_x4_from_neurons(chunk)).collect::<Vec<_>>(),
                remainder
            )
        };

        let num_vec = neurons.len();

        let (mut vars, vars_remainder) = {
            let chunks = self.vars.chunks_exact_mut(4);
            
            (
                chunks.into_iter().map(|chunk| M::vars_x4_from_vars(chunk)).collect::<Vec<_>>(),
                &mut self.vars[4*num_vec..]
            )
        };

        let mut weighted_inputs;

        for (ts, spike) in self.receiver {
            weighted_inputs = spike.dot(&self.layer.input_weights);
            let mut weighted_inputs_slice = weighted_inputs.as_slice().unwrap();

            loop {
                let mut spiked = false;
                let mut output = Array2::zeros((1, self.layer.neurons.len()));
                let output_slice = output.as_slice_mut().unwrap();

                for (i, (neurons, vars)) in neurons.iter().zip(vars.iter_mut()).enumerate() {
                    let o = M::handle_spike_x4(
                        neurons,
                        vars,
                        unsafe { f64x4::from_slice_unaligned_unchecked(&weighted_inputs_slice[4*i..(4*i + 4)]) },
                        ts as _
                    );

                    spiked |= o.gt(f64x4::splat(0.5)).any();

                    unsafe {
                        o.write_to_slice_unaligned_unchecked(&mut output_slice[4*i..(4*i + 4)]);
                    };
                }

                for (i, (neuron, vars)) in neuron_remainder.iter().zip(vars_remainder.iter_mut()).enumerate() {
                    let o = M::handle_spike(
                        neuron,
                        vars,
                        weighted_inputs[(0, num_vec*4 + i)],
                        ts
                    );
                    spiked |= o > 0.5;
                    output[(0, num_vec*4 + i)] = o;
                }

                if spiked {
                    weighted_inputs = output.dot(&self.layer.intra_weights);
                    weighted_inputs_slice = weighted_inputs.as_slice().unwrap();
                    self.sender.send((ts, output)).unwrap();
                } else {
                    break;
                }
            }
        }
    }

    #[cfg(all(feature = "async", not(feature = "simd")))]
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

    #[cfg(all(feature = "async", feature = "simd"))]
    pub async fn run(mut self) {
        use packed_simd::f64x4;

        let (neurons, neuron_remainder) = {
            let chunks = self.layer.neurons.chunks_exact(4);
            let remainder = chunks.remainder();
            
            (
                chunks.into_iter().map(|chunk| M::neuron_x4_from_neurons(chunk)).collect::<Vec<_>>(),
                remainder
            )
        };

        let num_vec = neurons.len();

        let (mut vars, vars_remainder) = {
            let chunks = self.vars.chunks_exact_mut(4);
            
            (
                chunks.into_iter().map(|chunk| M::vars_x4_from_vars(chunk)).collect::<Vec<_>>(),
                &mut self.vars[4*num_vec..]
            )
        };

        let mut weighted_inputs;

        while let Some((ts, spike)) = self.receiver.recv().await {
            weighted_inputs = spike.dot(&self.layer.input_weights);
            let mut weighted_inputs_slice = weighted_inputs.as_slice().unwrap();

            loop {
                let mut spiked = false;
                let mut output = Array2::zeros((1, self.layer.neurons.len()));
                let output_slice = output.as_slice_mut().unwrap();

                for (i, (neurons, vars)) in neurons.iter().zip(vars.iter_mut()).enumerate() {
                    let o = M::handle_spike_x4(
                        neurons,
                        vars,
                        unsafe { f64x4::from_slice_unaligned_unchecked(&weighted_inputs_slice[4*i..(4*i + 4)]) },
                        ts as _
                    );

                    spiked |= o.gt(f64x4::splat(0.5)).any();

                    unsafe {
                        o.write_to_slice_unaligned_unchecked(&mut output_slice[4*i..(4*i + 4)]);
                    };
                }

                for (i, (neuron, vars)) in neuron_remainder.iter().zip(vars_remainder.iter_mut()).enumerate() {
                    let o = M::handle_spike(
                        neuron,
                        vars,
                        weighted_inputs[(0, num_vec*4 + i)],
                        ts
                    );
                    spiked |= o > 0.5;
                    output[(0, num_vec*4 + i)] = o;
                }

                if spiked {
                    weighted_inputs = output.dot(&self.layer.intra_weights);
                    weighted_inputs_slice = weighted_inputs.as_slice().unwrap();
                    self.sender.send((ts, output)).await.unwrap();
                } else {
                    break;
                }
            }
        }
    }
}
