//! Neural network-related types

use crate::Model;

use self::layer::Layer;
use std::{fmt, ops::{Index, IndexMut}, borrow::Borrow};
use ndarray::Array2;
use thiserror::Error;

pub mod layer;
pub mod model;
pub mod builder;

#[cfg(all(test, not(feature = "expose-test-solver")))]
pub(crate) mod solver_v1;
#[cfg(any(
    all(not(test), feature = "expose-test-solver"),
    all(test,      feature = "expose-test-solver")
))]
pub mod solver_v1;
#[cfg(test)]
mod tests;

/// Represents the 'spike' that stimulates a neuron in a spiking neural network.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Spike {
    /// Stands for "time of the spike", and represents a timestamp of when the spike occurs
    pub ts: u128,
    /// Index of the neuron this spike applies to inside its layer
    pub neuron_id: usize
}

impl Spike {
    /// Create a new spike at time `ts` for neuron `neuron_id`
    /// 
    /// # Examples
    /// 
    /// Create a [Spike] at instant 18 for neuron 3:
    /// 
    /// ```
    /// # use pds_spiking_nn::Spike;
    /// let spike = Spike::new(18, 3);
    /// assert_eq!(spike, Spike { ts: 18, neuron_id: 3 });
    /// ```
    pub fn new(ts: u128, neuron_id: usize) -> Spike {
        Spike {
            ts,
            neuron_id
        }
    }

    /// Create an array of spikes for a single neuron, given its ID.
    /// The `ts_vec` does not need to be ordered.
    /// 
    /// # Examples
    /// 
    /// Construct a [Vec] of [Spike]s for neuron 2 from an unordered set of timestamps:
    /// 
    /// ```
    /// # use pds_spiking_nn::Spike;
    /// let spikes_ts = [11, 9, 23, 43, 42].to_vec();
    /// let spike_vec = Spike::spike_vec_for(2, spikes_ts);
    /// 
    /// assert_eq!(
    ///     spike_vec.into_iter().map(|Spike {ts, ..}| ts).collect::<Vec<_>>(),
    ///     vec![9, 11, 23, 42, 43]
    /// );
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
    /// 
    /// It takes a Matrix where i-th row represents an array of spikes for the i-th entry neuron,
    /// then a single Vec is created. Eventually the array is sorted.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use pds_spiking_nn::Spike;
    /// let spikes_neuron_1 = [11, 9, 23, 43, 42].to_vec();
    /// let spike_vec_for_neuron_1 = Spike::spike_vec_for(1, spikes_neuron_1);
    /// 
    /// let spikes_neuron_2 = [1, 29, 3, 11, 22].to_vec();
    /// let spike_vec_for_neuron_2 = Spike::spike_vec_for(2, spikes_neuron_2);
    /// 
    /// let mut spikes = Vec::new();
    /// spikes.push(spike_vec_for_neuron_1);
    /// spikes.push(spike_vec_for_neuron_2);
    /// 
    /// let sorted_spike_array_for_nn = Spike::create_terminal_vec(spikes);
    /// 
    /// let mut iter = sorted_spike_array_for_nn.into_iter();
    /// 
    /// assert_eq!(iter.next(), Some(Spike {ts: 1, neuron_id: 2}));
    /// assert_eq!(iter.next(), Some(Spike {ts: 3, neuron_id: 2}));
    /// assert_eq!(iter.next(), Some(Spike {ts: 9, neuron_id: 1}));
    /// ```
    pub fn create_terminal_vec(spikes: Vec<Vec<Spike>>) -> Vec<Spike> {
        let mut res: Vec<Spike> = Vec::new();

        for line in spikes {
            for spike in line {
                res.push(spike);
            }
        }
        res.sort(); //ascending
    
        res
    }
}

impl fmt::Display for Spike {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Spike(ts: {}, neuron_id: {})", self.ts, self.neuron_id)
    }
}

/// Error for [NN]'s [concat](NN::concat) and [extend](NN::extend).
/// 
/// Only one variant is needed because only one kind of error can happen.
#[derive(Error, Debug)]
pub enum NNConcatError {
    #[error("Provided intra-nn weights have invalid dimensions")]
    InvalidWeightsLen
}

/// The Neural Network itself.
/// 
/// This organizes [Neuron](Model::Neuron)s into consecutive layers, each constituted of some amount of [Neuron](Model::Neuron)s.
/// [Neuron](Model::Neuron)s of the same or consecutive layers are connected by a weighted synapse [f64].
/// 
/// A neural network is stimulated by [Spike]s applied to the [Neuron](Model::Neuron)s of its entry layer.
/// 
/// Create a new [NN] through the builder at [NNBuilder](crate::NNBuilder).
#[derive(Clone)]
pub struct NN<M: Model> {
    /// All the sorted layers of the neural network
    layers: Vec<Layer<M>>
}

impl<M: Model> NN<M> {
    /// Return the number of layers in this neural network.
    /// 
    /// This is always guaranteed to be greater than zero
    /// by the [NNBuilder](crate::NNBuilder) necessary to construct an instance of [NN].
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use pds_spiking_nn::{NNBuilder, lif::*};
    /// // Create a sample nn
    /// let nn = NNBuilder::<LeakyIntegrateFire, _>::new()
    ///     .layer(
    ///         [
    ///             LifNeuron::new(&LifNeuronConfig::new(1.0, 0.5, 3.0, 1.2)),
    ///             From::from(&LifNeuronConfig::new(1.0, 0.4, 3.1, 1.1))
    ///         ],
    ///         [1.5, 1.8],
    ///         [[0.0, -0.3], [-0.2, 0.0]]
    ///     )
    ///     .build();
    /// 
    /// assert_eq!(nn.num_layers(), 1);
    /// ```
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get the specified layer, or [None] if the index is out of bounds.
    /// 
    /// An unchecked variant of this functionality is provided via the [Index] implementation.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use pds_spiking_nn::{NNBuilder, lif::*};
    /// // Create a sample nn
    /// let nn = NNBuilder::<LeakyIntegrateFire, _>::new()
    ///     .layer(
    ///         [
    ///             LifNeuron::new(&LifNeuronConfig::new(1.0, 0.5, 3.0, 1.2)),
    ///             From::from(&LifNeuronConfig::new(1.0, 0.4, 3.1, 1.1))
    ///         ],
    ///         [1.5, 1.8],
    ///         [[0.0, -0.3], [-0.2, 0.0]]
    ///     )
    ///     .build();
    /// 
    /// // Get a reference to the first layer
    /// let first_layer = nn.get_layer(0);
    /// 
    /// assert_eq!(first_layer.unwrap().num_neurons(), 2);
    /// ```
    pub fn get_layer(&self, layer: usize) -> Option<&Layer<M>> {
        self.layers.get(layer)
    }

    /// Get a mutable reference to the specified layer, or [None] if the index is out of bounds.
    /// 
    /// An unchecked variant of this functionality is provided via the [IndexMut] implementation.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use pds_spiking_nn::{NNBuilder, lif::*};
    /// // Create a sample nn
    /// let mut nn = NNBuilder::<LeakyIntegrateFire, _>::new()
    ///     .layer(
    ///         [
    ///             LifNeuron::new(&LifNeuronConfig::new(1.0, 0.5, 3.0, 1.2)),
    ///             From::from(&LifNeuronConfig::new(1.0, 0.4, 3.1, 1.1))
    ///         ],
    ///         [1.5, 1.8],
    ///         [[0.0, -0.3], [-0.2, 0.0]]
    ///     )
    ///     .build();
    /// 
    /// // Modify an intra-layer weight
    /// *nn.get_layer_mut(0).unwrap().get_intra_weight_mut(0, 1).unwrap() = -1.7;
    /// 
    /// assert_eq!(nn[0][(0, 1)], -1.7);
    /// ```
    pub fn get_layer_mut(&mut self, layer: usize) -> Option<&mut Layer<M>> {
        self.layers.get_mut(layer)
    }

    /// Get the neuron at the specified position of the specified layer, or [None] if any index is out of bounds.
    /// 
    /// An unchecked variant of this functionality is provided via the [Index] implementation.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use pds_spiking_nn::{NNBuilder, lif::*};
    /// // Create a sample nn
    /// let nn = NNBuilder::<LeakyIntegrateFire, _>::new()
    ///     .layer(
    ///         [
    ///             LifNeuron::new(&LifNeuronConfig::new(1.0, 0.5, 3.0, 1.2)),
    ///             From::from(&LifNeuronConfig::new(1.0, 0.4, 3.1, 1.1))
    ///         ],
    ///         [1.5, 1.8],
    ///         [[0.0, -0.3], [-0.2, 0.0]]
    ///     )
    ///     .build();
    /// 
    /// // Get a reference to the second neuron of the only layer of the nn
    /// let neuron = nn.get_neuron(0, 1);
    /// 
    /// println!("{:?}", neuron); // Some(LifNeuron { v_rest: 1.0, v_reset: 0.4, v_threshold: 3.1, tau: 1.1 })
    /// ```
    pub fn get_neuron(&self, layer: usize, neuron: usize) -> Option<&M::Neuron> {
        self.layers.get(layer)?.neurons.get(neuron)
    }

    /// Get a mutable reference to the neuron at the specified position of the specified layer, or [None] if any index is out of bounds.
    /// 
    /// An unchecked variant of this functionality is provided via the [IndexMut] implementation.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use pds_spiking_nn::{NNBuilder, lif::*};
    /// // Create a sample nn
    /// let mut nn = NNBuilder::<LeakyIntegrateFire, _>::new()
    ///     .layer(
    ///         [
    ///             LifNeuron::new(&LifNeuronConfig::new(1.0, 0.5, 3.0, 1.2)),
    ///             From::from(&LifNeuronConfig::new(1.0, 0.4, 3.1, 1.1))
    ///         ],
    ///         [1.5, 1.8],
    ///         [[0.0, -0.3], [-0.2, 0.0]]
    ///     )
    ///     .build();
    /// 
    /// // Get a mutable reference to the second neuron of the only layer of the nn
    /// let mut neuron = nn.get_neuron_mut(0, 1).unwrap();
    /// neuron.v_rest += 0.2;
    /// 
    /// assert_eq!(nn[0][1].v_rest, 1.2);
    /// ```
    pub fn get_neuron_mut(&mut self, layer: usize, neuron: usize) -> Option<&mut M::Neuron> {
        self.layers.get_mut(layer)?.neurons.get_mut(neuron)
    }

    /// Get the input weight to the specified entry-layer neuron
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use pds_spiking_nn::{NNBuilder, lif::*};
    /// // Create a sample nn
    /// let nn = NNBuilder::<LeakyIntegrateFire, _>::new()
    ///     .layer(
    ///         [
    ///             LifNeuron::new(&LifNeuronConfig::new(1.0, 0.5, 3.0, 1.2)),
    ///             From::from(&LifNeuronConfig::new(1.0, 0.4, 3.1, 1.1))
    ///         ],
    ///         [1.5, 1.8],
    ///         [[0.0, -0.3], [-0.2, 0.0]]
    ///     )
    ///     .build();
    /// 
    /// assert_eq!(nn.get_input_weight(0), Some(1.5));
    /// assert_eq!(nn.get_input_weight(2), None);
    /// ```
    pub fn get_input_weight(&self, to: usize) -> Option<f64> {
        self.layers[0].input_weights.get((to, to)).copied()
    }

    /// Get a mutable reference to the specified entry-layer neuron
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use pds_spiking_nn::{NNBuilder, lif::*};
    /// // Create a sample nn
    /// let mut nn = NNBuilder::<LeakyIntegrateFire, _>::new()
    ///     .layer(
    ///         [
    ///             LifNeuron::new(&LifNeuronConfig::new(1.0, 0.5, 3.0, 1.2)),
    ///             From::from(&LifNeuronConfig::new(1.0, 0.4, 3.1, 1.1))
    ///         ],
    ///         [1.5, 1.8],
    ///         [[0.0, -0.3], [-0.2, 0.0]]
    ///     )
    ///     .build();
    /// 
    /// // Modify the second input weight
    /// *nn.get_input_weight_mut(1).unwrap() = 2.0;
    /// 
    /// assert_eq!(nn.get_input_weight(1), Some(2.0));
    /// ```
    pub fn get_input_weight_mut(&mut self, to: usize) -> Option<&mut f64> {
        self.layers[0].input_weights.get_mut((to, to))
    }

    /// Get the intra or input weight determined by the `from` and `to` neurons.
    /// 
    /// The given neurons must be of the same or consecutive layers, otherwise this function will
    /// return [None].
    /// 
    /// An unchecked variant of this functionality is provided via the [Index] implementation.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use pds_spiking_nn::{NNBuilder, lif::*};
    /// // Create a sample nn
    /// let nn = NNBuilder::<LeakyIntegrateFire, _>::new()
    ///     .layer(
    ///         [
    ///             LifNeuron::new(&LifNeuronConfig::new(1.0, 0.5, 3.0, 1.2)),
    ///             From::from(&LifNeuronConfig::new(1.0, 0.4, 3.1, 1.1))
    ///         ],
    ///         [1.5, 1.8],
    ///         [[0.0, -0.3], [-0.2, 0.0]]
    ///     )
    ///     .build();
    /// 
    /// assert_eq!(nn.get_weight((0, 1), (0, 0)), Some(-0.2));
    /// assert_eq!(nn.get_weight((0, 0), (1, 2)), None);
    /// assert_eq!(nn.get_weight((0, 1), (2, 0)), None);
    /// ```
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
    /// return [None].
    /// 
    /// An unchecked variant of this functionality is provided via the [IndexMut] implementation.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use pds_spiking_nn::{NNBuilder, lif::*};
    /// // Create a sample nn
    /// let mut nn = NNBuilder::<LeakyIntegrateFire, _>::new()
    ///     .layer(
    ///         [
    ///             LifNeuron::new(&LifNeuronConfig::new(1.0, 0.5, 3.0, 1.2)),
    ///             From::from(&LifNeuronConfig::new(1.0, 0.4, 3.1, 1.1))
    ///         ],
    ///         [1.5, 1.8],
    ///         [[0.0, -0.3], [-0.2, 0.0]]
    ///     )
    ///     .build();
    /// 
    /// // Modify a weight
    /// *nn.get_weight_mut((0, 1), (0, 0)).unwrap() = -0.5;
    /// 
    /// assert_eq!(nn[0][(1, 0)], -0.5);
    /// ```
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

    /// Extend this`[NN] in place by appending the other provided network to it.
    /// 
    /// The two neural networks are merged via the provided new input weights, which will replace `other`'s.
    /// 
    /// In case of errors, `self` will be preserved.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use pds_spiking_nn::{NNBuilder, lif::*};
    /// // Create a sample nn
    /// let mut nn1 = NNBuilder::<LeakyIntegrateFire, _>::new()
    ///     .layer(
    ///         [
    ///             LifNeuron::new(&LifNeuronConfig::new(1.0, 0.5, 3.0, 1.2)),
    ///             From::from(&LifNeuronConfig::new(1.0, 0.4, 3.1, 1.1))
    ///         ],
    ///         [1.5, 1.8],
    ///         [[0.0, -0.3], [-0.2, 0.0]]
    ///     )
    ///     .build();
    /// 
    /// let nn2 = NNBuilder::<LeakyIntegrateFire, _>::new()
    ///     .layer(
    ///         [LifNeuron::new(&LifNeuronConfig::new(0.9, 0.5, 2.8, 1.4))],
    ///         [1.3],
    ///         [[0.0]]
    ///     )
    ///     .build();
    /// 
    /// // Extend nn1 by concatenating nn2 to it in place
    /// assert!(nn1.extend(&nn2, [1.3, 1.4]).is_ok());
    /// assert_eq!(nn1.num_layers(), 2);
    /// assert_eq!(nn1[((0, 1), (1, 0))], 1.4);
    /// ```
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

    /// Concatenate this [NN] with another one, to obtain a new [NN].
    /// 
    /// The two neural networks are merged via the provided new input weights, which will replace `other`'s.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use pds_spiking_nn::{NNBuilder, lif::*};
    /// // Create a sample nn
    /// let nn = NNBuilder::<LeakyIntegrateFire, _>::new()
    ///     .layer(
    ///         [
    ///             LifNeuron::new(&LifNeuronConfig::new(1.0, 0.5, 3.0, 1.2)),
    ///             From::from(&LifNeuronConfig::new(1.0, 0.4, 3.1, 1.1))
    ///         ],
    ///         [1.5, 1.8],
    ///         [[0.0, -0.3], [-0.2, 0.0]]
    ///     )
    ///     .build();
    /// 
    /// // Create a new NN by concatenating nn to itself
    /// let new_nn = nn.concat(&nn, [1.2, 1.4, 1.5, 1.2]).unwrap();
    /// 
    /// assert_eq!(new_nn.num_layers(), 2);
    /// assert_eq!(new_nn[((0, 0), (1, 1))], 1.4);
    /// ```
    pub fn concat(&self, other: &Self, intra_nn_weights: impl Borrow<[f64]>) -> Result<Self, NNConcatError> {
        let mut new_nn = self.clone();
        new_nn.extend(other, intra_nn_weights).map(|_| new_nn)
    }

    /// Returns an iterator over references of every layer
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use pds_spiking_nn::{NNBuilder, lif::*};
    /// let nn = NNBuilder::<LeakyIntegrateFire, _>::new()
    ///     .layer(
    ///         [
    ///             LifNeuron::new(&LifNeuronConfig::new(1.0, 0.5, 3.0, 1.2)),
    ///             From::from(&LifNeuronConfig::new(1.0, 0.4, 3.1, 1.1))
    ///         ],
    ///         [1.5, 1.8],
    ///         [[0.0, -0.3], [-0.2, 0.0]]
    ///     )
    ///     .build();
    /// 
    /// let mut iterator = nn.iter();
    /// assert!(iterator.next().is_some());
    /// assert!(iterator.next().is_none());
    /// ```
    pub fn iter(&self) -> <&Vec<Layer<M>> as IntoIterator>::IntoIter {
        self.into_iter()
    }

    /// Returns an iterator over mutable references of every layer
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use pds_spiking_nn::{NNBuilder, lif::*};
    /// let mut nn = NNBuilder::<LeakyIntegrateFire, _>::new()
    ///     .layer(
    ///         [
    ///             LifNeuron::new(&LifNeuronConfig::new(1.0, 0.5, 3.0, 1.2)),
    ///             From::from(&LifNeuronConfig::new(1.0, 0.4, 3.1, 1.1))
    ///         ],
    ///         [1.5, 1.8],
    ///         [[0.0, -0.3], [-0.2, 0.0]]
    ///     )
    ///     .build();
    /// 
    /// let mut iterator = nn.iter_mut();
    /// 
    /// iterator.next().unwrap()[0].v_rest += 1.0;
    /// assert!(iterator.next().is_none());
    /// ```
    pub fn iter_mut(&mut self) -> <&mut Vec<Layer<M>> as IntoIterator>::IntoIter {
        self.into_iter()
    }
}

impl<M: Model> NN<M> where for<'a> &'a M::Neuron: Into<M::SolverVars> {
    /// Solve the neural network stimulated by the provided spikes.
    /// 
    /// This function returns a list of every spike's timestamp generated by every neuron.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use pds_spiking_nn::{NNBuilder, Spike, lif::*};
    /// let nn = NNBuilder::<LeakyIntegrateFire, _>::new()
    ///     .layer(
    ///         [
    ///             LifNeuron::new(&LifNeuronConfig::new(1.0, 0.5, 3.0, 1.2)),
    ///             From::from(&LifNeuronConfig::new(1.0, 0.4, 3.1, 1.1))
    ///         ],
    ///         [1.5, 1.8],
    ///         [[0.0, -0.3], [-0.2, 0.0]]
    ///     )
    ///     .build();
    /// 
    /// let spikes = Spike::create_terminal_vec(vec![
    ///     Spike::spike_vec_for(0, vec![1, 3, 4]),
    ///     Spike::spike_vec_for(1, vec![2, 3, 6])
    /// ]);
    /// 
    /// assert_eq!(nn.solve(spikes), vec![vec![4], vec![3]]);
    /// ```
    #[cfg(not(feature = "async"))]
    pub fn solve(&self, spikes: Vec<Spike>) -> Vec<Vec<u128>> {
        use crate::sync::LayerManager;
        use std::{mem::{transmute, replace}, thread, sync::mpsc::channel};
        
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
            for (neuron_id, _) in spike.into_iter().enumerate().filter(|(_, v)| *v > 0.5) {
                res[neuron_id].push(ts);
            }
        }

        res
    }

    /// Solve the neural network stimulated by the provided spikes.
    /// 
    /// This function returns a list of every spike's timestamp generated by every neuron.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use pds_spiking_nn::{NNBuilder, Spike, lif::*};
    /// # use tokio::runtime::Runtime;
    /// # let runtime = Runtime::new().unwrap();
    /// let nn = NNBuilder::<LeakyIntegrateFire, _>::new()
    ///     .layer(
    ///         [
    ///             LifNeuron::new(&LifNeuronConfig::new(1.0, 0.5, 3.0, 1.2)),
    ///             From::from(&LifNeuronConfig::new(1.0, 0.4, 3.1, 1.1))
    ///         ],
    ///         [1.5, 1.8],
    ///         [[0.0, -0.3], [-0.2, 0.0]]
    ///     )
    ///     .build();
    /// 
    /// let spikes = Spike::create_terminal_vec(vec![
    ///     Spike::spike_vec_for(0, vec![1, 3, 4]),
    ///     Spike::spike_vec_for(1, vec![2, 3, 6])
    /// ]);
    /// 
    /// # runtime.block_on(async {
    /// assert_eq!(nn.solve(spikes).await, vec![vec![4], vec![3]]);
    /// # });
    /// ```
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
            for (neuron_id, _) in spike.iter().enumerate().filter(|(_, v)| **v > 0.5) {
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
