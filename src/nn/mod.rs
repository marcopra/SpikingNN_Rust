use crate::Model;

use self::model::Layer;
use std::fmt;

pub mod model;
pub(crate) mod builder;
pub(crate) mod solver_v1;

#[cfg(test)]
mod tests;

/// Represents the 'spike' that stimulates a neuron in a spiking neural network.
///  
/// The parameter _'ts'_ stands for 'Time of the Spike' and represents the time when the spike occurs
/// while the parameter _'neuron_id'_ stands to

// TODO Provare Efficienza una tupla al posto di una struct
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Spike {
    pub ts: u128,
    pub neuron_id: usize
}

impl Spike {
    //Di interfaccia
    pub fn new(ts: u128, neuron_id: usize) -> Spike{
        Spike {
            ts,
            neuron_id
        }
    }

    //Di interfaccia
    /// Create an array of spikes for a single neuron, given its ID.
    /// 
    /// You can also give an unordered ts array as shown in the following example.
    /// # Example of Usage
    /// 
    /// ```
    ///  let spikes_neuron_2 = [11, 9, 23, 43, 42].to_vec();
    ///  let spike_vec_for_neuron_2 = Spike::spike_vec_for(neuron_id: 2, ts_vec: spikes_neuron_2 );
    /// 
    /// ```
    pub fn spike_vec_for(neuron_id: usize, ts_vec: Vec<u128>) -> Vec<Spike> {

        let mut spike_vec : Vec<Spike> = Vec::with_capacity(ts_vec.len());
        
        //Creating the Spikes array for a single Neuron
        for ts in ts_vec.into_iter() {
            spike_vec.push(Spike::new(ts, neuron_id));
        }

        //Order the ts vector
        spike_vec.sort();

        spike_vec
    }


    /// Create an ordered array starting from all the spikes sent to the NN.
    /// It takes a Matrix where each row i-th represents an array of spike for neuron i-th
    /// then a single Vec is created. Eventually the array is sorted
    /// 
    /// # Example
    /// ```
    ///  use crate::nn::Spike;
    /// 
    ///  let spikes_neuron_1 = [11, 9, 23, 43, 42].to_vec();
    ///  let spike_vec_for_neuron_1 = Spike::spike_vec_for(2, spikes_neuron_1 );
    ///  
    ///  let spikes_neuron_2 = [1, 29, 3, 11, 22].to_vec();
    ///  let spike_vec_for_neuron_2 = Spike::spike_vec_for(2, spikes_neuron_2 );
    ///  
    ///  let mut spikes: Vec<Vec<Spike>> = Vec::new();
    ///  spikes.push(spike_vec_for_neuron_1);
    ///  spikes.push(spike_vec_for_neuron_2);
    ///  
    ///  let sorted_spike_array_for_nn: Vec<Spike> = Spike::create_terminal_vec(spikes)
    /// 
    /// ```
    pub fn create_terminal_vec(spikes: Vec<Vec<Spike>>) -> Vec<Spike> {
        let mut res: Vec<Spike> = Vec::new();

        for line in spikes {
            for spike in line {
                res.push(spike);
            }
        }
        res.sort(); //ascending
        //TODO cancellare? res.sort_by(|a, b| a.ts.partial_cmp(&b.ts));
    
        res
    }
}

impl fmt::Display for Spike {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.ts, self.neuron_id)
    }
}

/// The Neural Network itself.
/// 
/// This organizes `Neuron`s into consecutive layers, each constituted of some amount of `Neuron`s.
/// `Neuron`s of the same or consecutive layers are connected by a weighted `Synapse`.
/// 
/// A neural network is stimulated by `Spike`s applied to the `Neuron`s of the entry layer.
#[derive(Clone)]
pub struct NN<M: Model> {
    /// All the sorted layers of the neural network
    layers: Vec<Layer<M>>
}

// I need to explicitly request RefInto<SolverVars> for Neuron because of a limitation in the Rust compiler with respect
// to implied bounds. See: https://users.rust-lang.org/t/hrtb-in-trait-definition-for-associated-types/78687
impl<M: Model> NN<M> where for<'a> &'a M::Neuron: Into<M::SolverVars> {
    /// Solve the neural network stimulated by the provided spikes.
    /// 
    /// This function returns a list of every spike's timestamp generated by every neuron.
    #[cfg(feature = "per-neuron-parallelism")]
    pub fn solve(&self, spikes: Vec<Spike>) -> Vec<Vec<u128>> {
        use crate::sync::LayerManager;
        use std::{mem::{transmute, replace}, thread, sync::{Arc, mpsc::channel}};
        use ndarray::Array2;
        
        // These will be respectively the first layer's sender and the last layer's receiver
        let (sender, mut receiver) = channel();

        // Inject spikes into first layer
        {
            let mut spike_iterator = spikes.into_iter().peekable();
            while let Some(Spike {ts, neuron_id}) = spike_iterator.next() {
                let mut to_send = Array2::zeros((1, self.layers[0].neurons.len()));
                to_send[(0, neuron_id)] = 1.0; // Should we validate neuron_ids?

                while let Some(Spike {neuron_id, ..}) = spike_iterator.next_if(|s| s.ts == ts) {
                    to_send[(0, neuron_id)] = 1.0;
                }

                sender.send((ts, to_send)).unwrap();
            }
        }

        // Drop the first sender.
        // This will cause a chain reaction that will ultimately lead to the last receiver being closed.
        drop(sender);
        
        for Layer {neurons, input_weights, intra_weights} in &self.layers {
            let (layer_sender, layer_receiver) = channel();
            
            // Create the LayerManager for this layer
            let (mngr, tokens) = LayerManager::new(
                neurons.len(),
                replace(&mut receiver, layer_receiver),
                layer_sender,
                input_weights,
                intra_weights
            );

            // We're gonna share mngr with multiple threads. Since I know the threads will live less than
            // the lifetime 'a of the LayerManager<'a> (which is the lifetime of self), I can use some unsafe to allow this.
            let mngr = Arc::new(unsafe { transmute::<_, LayerManager<'_>>(mngr) });

            // Create a new thread for each neuron of the layer
            for (neuron, token) in neurons.iter().zip(tokens.into_iter()) {
                // Same as for mngr, use an anonymous lifetime to pass to the thread
                let neuron = unsafe { transmute::<_, &M::Neuron>(neuron) };
                let mngr = Arc::clone(&mngr);

                thread::spawn(move || {
                    let mut solver_vars: M::SolverVars = neuron.into();
                    
                    while let Some((ts, weighted_input_val)) = mngr.next(&token) {
                        let output = M::handle_spike(neuron, &mut solver_vars, weighted_input_val, ts);
                        let spiked = output > 0.5; // TODO: do we really want this?
                        mngr.commit(&token, spiked, output);
                    }
                });
            }
        }

        // Read spikes from last layer and convert to proper format for output
        let mut res = vec![vec![]; self.layers.last().unwrap().neurons.len()];
        for (ts, spike) in receiver {
            for (neuron_id, _) in spike.into_iter().enumerate().filter(|(_, v)| *v > 0.5) { // TODO: do we really want this?
                res[neuron_id].push(ts);
            }
        }

        res
    }

    #[cfg(all(not(feature = "per-neuron-parallelism"), not(feature = "async")))]
    pub fn solve(&self, spikes: Vec<Spike>) -> Vec<Vec<u128>> {
        use crate::sync::LayerManager;
        use std::{mem::{transmute, replace}, thread, sync::mpsc::channel};
        use ndarray::Array2;
        
        // These will be respectively the first layer's sender and the last layer's receiver
        let (sender, mut receiver) = channel();

        // Inject spikes into first layer
        {
            let mut spike_iterator = spikes.into_iter().peekable();
            while let Some(Spike {ts, neuron_id}) = spike_iterator.next() {
                let mut to_send = Array2::zeros((1, self.layers[0].neurons.len()));
                to_send[(0, neuron_id)] = 1.0; // Should we validate neuron_ids?

                while let Some(Spike {neuron_id, ..}) = spike_iterator.next_if(|s| s.ts == ts) {
                    to_send[(0, neuron_id)] = 1.0;
                }

                sender.send((ts, to_send)).unwrap();
            }
        }

        // Drop the first sender.
        // This will cause a chain reaction that will ultimately lead to the last receiver being closed.
        drop(sender);

        for layer in &self.layers {
            let layer = unsafe { transmute::<_, &_>(layer) };
            let (layer_sender, mut layer_receiver) = channel();
            layer_receiver = replace(&mut receiver, layer_receiver);
            
            thread::spawn(move || {
                let mngr = LayerManager::<M>::new(
                    layer,
                    layer_receiver,
                    layer_sender
                );

                mngr.run();
            });
        }

        // Read spikes from last layer and convert to proper format for output
        let mut res = vec![vec![]; self.layers.last().unwrap().neurons.len()];
        for (ts, spike) in receiver {
            for (neuron_id, _) in spike.into_iter().enumerate().filter(|(_, v)| *v > 0.5) { // TODO: do we really want this?
                res[neuron_id].push(ts);
            }
        }

        res
    }

    #[cfg(all(not(feature = "per-neuron-parrallelism"), feature = "async"))]
    pub async fn solve(&self, spikes: Vec<Spike>) -> Vec<Vec<u128>> {
        use crate::sync::LayerManager;
        use std::mem::{transmute, replace};
        use ndarray::Array2;
        use tokio::{task, sync::mpsc::channel};
        
        // These will be respectively the first layer's sender and the last layer's receiver
        let (sender, mut receiver) = channel(10);

        let s = unsafe {transmute::<_, &Self>(self)};
        
        // Inject spikes into first layer
        task::spawn(async move {
            let mut spike_iterator = spikes.into_iter().peekable();
            
            while let Some(Spike {ts, neuron_id}) = spike_iterator.next() {
                let mut to_send = Array2::zeros((1, s.layers[0].neurons.len()));
                to_send[(0, neuron_id)] = 1.0; // Should we validate neuron_ids?

                while let Some(Spike {neuron_id, ..}) = spike_iterator.next_if(|s| s.ts == ts) {
                    to_send[(0, neuron_id)] = 1.0;
                }

                sender.send((ts, to_send)).await.unwrap();
            }

            // Drop the first sender.
            // This will cause a chain reaction that will ultimately lead to the last receiver being closed.
            drop(sender);
        });

        for layer in &self.layers {
            let layer = unsafe { transmute::<_, &Layer<M>>(layer) };
            let (layer_sender, mut layer_receiver) = channel(10);
            layer_receiver = replace(&mut receiver, layer_receiver);

            task::spawn(async move {
                let mngr = LayerManager::<M>::new(
                    layer,
                    layer_receiver,
                    layer_sender,
                );

                mngr.run().await
            });
        }

        // Read spikes from last layer and convert to proper format for output
        let mut res = vec![vec![]; self.layers.last().unwrap().neurons.len()];
        while let Some((ts, spike)) = receiver.recv().await {
            for (neuron_id, _) in spike.iter().enumerate().filter(|(_, v)| **v > 0.5) { // TODO: do we really want this?
                res[neuron_id].push(ts);
            }
        }

        res
    }
}
