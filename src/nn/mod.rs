use crate::Model;

use self::model::Layer;
use std::{fmt, ops::{Index, IndexMut}, borrow::Borrow};
use ndarray::Array2;
use thiserror::Error;

pub mod model;
pub(crate) mod builder;

#[cfg(test)]
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
    pub fn new(ts: u128, neuron_id: usize) -> Spike {
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

#[derive(Error, Debug)]
pub enum NNConcatError {
    #[error("Provided intra-nn weights have invalid dimensions")]
    InvalidWeightsLen
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

impl<M: Model> NN<M> {
    /// Return the number of layers in this neural network.
    /// 
    /// This is always guaranteed to be greater than zero
    /// by the `NNBuilder` necessary to construct an instance of `NN`.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get the specified layer, or `None` if the index is out of bounds.
    /// 
    /// An unchecked variant of this functionality is provided via the `Index` implementation.
    pub fn get_layer(&self, layer: usize) -> Option<&Layer<M>> {
        self.layers.get(layer)
    }

    /// Get a mutable reference to the specified layer, or `None` if the index is out of bounds.
    /// 
    /// An unchecked variant of this functionality is provided via the `IndexMut` implementation.
    pub fn get_layer_mut(&mut self, layer: usize) -> Option<&mut Layer<M>> {
        self.layers.get_mut(layer)
    }

    /// Get the neuron at the specified position of the specified layer, or `None` if any index is out of bounds.
    /// 
    /// An unchecked variant of this functionality is provided via the `Index` implementation.
    pub fn get_neuron(&self, layer: usize, neuron: usize) -> Option<&M::Neuron> {
        self.layers.get(layer)?.neurons.get(neuron)
    }

    /// Get a mutable reference to the neuron at the specified position of the specified layer, or `None` if any index is out of bounds.
    /// 
    /// An unchecked variant of this functionality is provided via the `IndexMut` implementation.
    pub fn get_neuron_mut(&mut self, layer: usize, neuron: usize) -> Option<&mut M::Neuron> {
        self.layers.get_mut(layer)?.neurons.get_mut(neuron)
    }

    /// Get the input weight to the specified entry-layer neuron
    pub fn get_input_weight(&self, to: usize) -> Option<f64> {
        self.layers[0].input_weights.get((to, to)).copied()
    }

    /// Get a mutable reference to the specified entry-layer neuron
    pub fn get_input_weight_mut(&mut self, to: usize) -> Option<&mut f64> {
        self.layers[0].input_weights.get_mut((to, to))
    }

    /// Get the intra or input weight determined by the `from` and `to` neurons.
    /// 
    /// The given neurons must be of the same or consecutive layers, otherwise this function will
    /// return `None`.
    /// 
    /// An unchecked variant of this functionality is provided via the `Index` implementation.
    pub fn get_weight(&self, from: (usize, usize), to: (usize, usize)) -> Option<f64> {
        if from.0 == to.0 {
            // Intra-layer weight
            self.get_layer(from.0)?.intra_weights.get((from.1, to.1)).copied()
        } else if from.0 + 1 == to.0 {
            // Inter-layer weight
            self.get_layer(to.0)?.input_weights.get((from.1, to.1)).copied()
        } else {
            None
        }
    }

    /// Get a mutable reference to the intra or input weight determined by the `from` and `to` neurons.
    /// 
    /// The given neurons must be of the same or consecutive layers, otherwise this function will
    /// return `None`.
    /// 
    /// An unchecked variant of this functionality is provided via the `IndexMut` implementation.
    pub fn get_weight_mut(&mut self, from: (usize, usize), to: (usize, usize)) -> Option<&mut f64> {
        if from.0 == to.0 {
            // Intra-layer weight
            self.get_layer_mut(from.0)?.intra_weights.get_mut((from.1, to.1))
        } else if from.0 + 1 == to.0 {
            // Inter-layer weight
            self.get_layer_mut(to.0)?.input_weights.get_mut((from.1, to.1))
        } else {
            None
        }
    }

    /// Extend this `NN` in place by appending the other provided network to it.
    /// 
    /// The two neural networks are merged via the provided new input weights, which will replace `other`'s.
    /// 
    /// In case of errors, `self` will be preserved.
    pub fn extend(&mut self, other: &Self, intra_nn_weights: impl Borrow<[f64]>) -> Result<(), NNConcatError> {
        let new_input_weights = Array2::from_shape_vec(
            (self.layers.last().unwrap().num_neurons(), other.layers[0].num_neurons()),
            intra_nn_weights.borrow().to_vec()
        ).map_err(|_| NNConcatError::InvalidWeightsLen)?;

        let old_len = self.num_layers();
        self.layers.extend_from_slice(&other.layers[..]);
        self.layers[old_len].input_weights = new_input_weights;

        Ok(())
    }

    /// Concatenate this `NN` with another one, to obtain a new `NN`.
    /// 
    /// The two neural networks are merged via the provided new input weights, which will replace `other`'s.
    pub fn concat(&self, other: &Self, intra_nn_weights: impl Borrow<[f64]>) -> Result<Self, NNConcatError> {
        let mut new_nn = self.clone();
        new_nn.extend(other, intra_nn_weights).map(|_| new_nn)
    }
}

// I need to explicitly request RefInto<SolverVars> for Neuron because of a limitation in the Rust compiler with respect
// to implied bounds. See: https://users.rust-lang.org/t/hrtb-in-trait-definition-for-associated-types/78687
impl<M: Model> NN<M> where for<'a> &'a M::Neuron: Into<M::SolverVars> {
    /// Solve the neural network stimulated by the provided spikes.
    /// 
    /// This function returns a list of every spike's timestamp generated by every neuron.
    #[cfg(not(feature = "async"))]
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

    /// Solve the neural network stimulated by the provided spikes.
    /// 
    /// This function returns a list of every spike's timestamp generated by every neuron.
    #[cfg(feature = "async")]
    pub async fn solve(&self, spikes: Vec<Spike>) -> Vec<Vec<u128>> {
        use crate::sync::LayerManager;
        use std::mem::{transmute, replace};
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

impl<M: Model> Index<usize> for NN<M> {
    type Output = Layer<M>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.layers[index]
    }
}

impl<M: Model> IndexMut<usize> for NN<M> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.layers[index]
    }
}

impl<M: Model> Index<(usize, usize)> for NN<M> {
    type Output = M::Neuron;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.layers[index.0].neurons[index.1]
    }
}

impl<M: Model> IndexMut<(usize, usize)> for NN<M> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.layers[index.0].neurons[index.1]
    }
}

impl<M: Model> Index<((usize, usize), (usize, usize))> for NN<M> {
    type Output = f64;

    fn index(&self, index: ((usize, usize), (usize, usize))) -> &Self::Output {
        if index.0.0 == index.1.0 {
            // Get intra-layer weight
            &self.layers[index.0.0].intra_weights[(index.0.1, index.1.1)]
        } else if index.0.0 + 1 == index.1.0 {
            // Get inter-layer weight
            &self.layers[index.1.0].input_weights[(index.0.1, index.1.1)]
        } else {
            panic!("Synapse index was invalid")
        }
    }
}

impl<M: Model> IndexMut<((usize, usize), (usize, usize))> for NN<M> {
    fn index_mut(&mut self, index: ((usize, usize), (usize, usize))) -> &mut Self::Output {
        if index.0.0 == index.1.0 {
            // Get intra-layer weight
            &mut self.layers[index.0.0].intra_weights[(index.0.1, index.1.1)]
        } else if index.0.0 + 1 == index.1.0 {
            // Get inter-layer weight
            &mut self.layers[index.1.0].input_weights[(index.0.1, index.1.1)]
        } else {
            panic!("Synapse index was invalid")
        }
    }
}

impl<M: Model> IntoIterator for NN<M> {
    type Item = Layer<M>;
    type IntoIter = <Vec<Layer<M>> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.layers.into_iter()
    }
}

impl<'a, M: Model> IntoIterator for &'a NN<M> {
    type Item = &'a Layer<M>;
    type IntoIter = <&'a Vec<Layer<M>> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.layers.iter()
    }
}

impl<'a, M: Model> IntoIterator for &'a mut NN<M> {
    type Item = &'a mut Layer<M>;
    type IntoIter = <&'a mut Vec<Layer<M>> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.layers.iter_mut()
    }
}
