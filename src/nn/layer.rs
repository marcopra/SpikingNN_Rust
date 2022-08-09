//! `Layer` type for each layer of the neural network

use std::ops::{Index, IndexMut};
use ndarray::Array2;
use crate::Model;

/// A single layer in the neural network
/// 
/// This contains all the neurons of the layer, as well as the intra-layer weights and input weights from
/// the previous layer.
#[derive(Clone)]
pub struct Layer<M: Model> {
    /// List of all neurons in this layer
    pub(crate) neurons: Vec<M::Neuron>,
    /// Matrix of the input weights. For the first layer, this must be a square diagonal matrix.
    pub(crate) input_weights: Array2<f64>,
    /// Square matrix of the intra-layer weights
    pub(crate) intra_weights: Array2<f64>
}

impl<M: Model> Layer<M> {
    /// Return the number of neurons in this [Layer]
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use pds_spiking_nn::{NNBuilder, lif::*};
    /// let nn = NNBuilder::<LeakyIntegrateFire, _>::new()
    ///     .layer(
    ///         [
    ///             LifNeuron::new(&LifNeuronConfig::new(1.2, 0.4, 2.7, 0.9)),
    ///             LifNeuron::new(&LifNeuronConfig::new(0.8, 0.2, 2.5, 1.2)),
    ///             LifNeuron::new(&LifNeuronConfig::new(1.1, 0.6, 3.1, 1.2))
    ///         ],
    ///         [1.0, 1.1, 0.7],
    ///         [[0.0, -0.2, -0.3], [-0.2, 0.0, -0.1], [-0.4, -0.3, 0.0]]
    ///     )
    ///     .build();
    /// 
    /// assert_eq!(nn[0].num_neurons(), 3);
    /// ```
    pub fn num_neurons(&self) -> usize {
        self.neurons.len()
    }

    /// Get the specified neuron, or [None] if the index is out of bounds.
    /// 
    /// An unchecked variant of this functionality is provided via the [Index] implementation.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use pds_spiking_nn::{NNBuilder, lif::*};
    /// let nn = NNBuilder::<LeakyIntegrateFire, _>::new()
    ///     .layer(
    ///         [
    ///             LifNeuron::new(&LifNeuronConfig::new(1.2, 0.4, 2.7, 0.9)),
    ///             LifNeuron::new(&LifNeuronConfig::new(0.8, 0.2, 2.5, 1.2)),
    ///             LifNeuron::new(&LifNeuronConfig::new(1.1, 0.6, 3.1, 1.2))
    ///         ],
    ///         [1.0, 1.1, 0.7],
    ///         [[0.0, -0.2, -0.3], [-0.2, 0.0, -0.1], [-0.4, -0.3, 0.0]]
    ///     )
    ///     .build();
    /// 
    /// // Get a reference to the second neuron in the layer
    /// let neuron = nn[0].get_neuron(1);
    /// ```
    pub fn get_neuron(&self, neuron: usize) -> Option<&M::Neuron> {
        self.neurons.get(neuron)
    }

    /// Get a mutable reference to the specified neuron, or [None] if the index is out of bounds.
    /// 
    /// An unchecked variant of this functionality is provided via the [IndexMut] implementation.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use pds_spiking_nn::{NNBuilder, lif::*};
    /// let mut nn = NNBuilder::<LeakyIntegrateFire, _>::new()
    ///     .layer(
    ///         [
    ///             LifNeuron::new(&LifNeuronConfig::new(1.2, 0.4, 2.7, 0.9)),
    ///             LifNeuron::new(&LifNeuronConfig::new(0.8, 0.2, 2.5, 1.2)),
    ///             LifNeuron::new(&LifNeuronConfig::new(1.1, 0.6, 3.1, 1.2))
    ///         ],
    ///         [1.0, 1.1, 0.7],
    ///         [[0.0, -0.2, -0.3], [-0.2, 0.0, -0.1], [-0.4, -0.3, 0.0]]
    ///     )
    ///     .build();
    /// 
    /// // Modify the third neuron
    /// *nn[0].get_neuron_mut(2).unwrap() = LifNeuron::new(&LifNeuronConfig::new(1.0, 0.4, 3.0, 1.2));
    /// ```
    pub fn get_neuron_mut(&mut self, neuron: usize) -> Option<&mut M::Neuron> {
        self.neurons.get_mut(neuron)
    }

    /// Get the intra-layer weight from and to the specified neurons, or [None] if any index is out of bounds.
    /// 
    /// An unchecked variant of this functionality is provided via the [Index] implementation.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use pds_spiking_nn::{NNBuilder, lif::*};
    /// let nn = NNBuilder::<LeakyIntegrateFire, _>::new()
    ///     .layer(
    ///         [
    ///             LifNeuron::new(&LifNeuronConfig::new(1.2, 0.4, 2.7, 0.9)),
    ///             LifNeuron::new(&LifNeuronConfig::new(0.8, 0.2, 2.5, 1.2)),
    ///             LifNeuron::new(&LifNeuronConfig::new(1.1, 0.6, 3.1, 1.2))
    ///         ],
    ///         [1.0, 1.1, 0.7],
    ///         [[0.0, -0.2, -0.3], [-0.2, 0.0, -0.1], [-0.4, -0.3, 0.0]]
    ///     )
    ///     .build();
    /// 
    /// assert_eq!(nn[0].get_intra_weight(0, 2), Some(-0.3));
    /// assert_eq!(nn[0].get_intra_weight(2, 3), None);
    /// ```
    pub fn get_intra_weight(&self, from: usize, to: usize) -> Option<f64> {
        self.intra_weights.get((from, to)).copied()
    }

    /// Get a mutable reference to the intra-layer weight from and to the specified neurons, or [None] if any index is out of bounds.
    /// 
    /// An unchecked variant of this functionality is provided via the [IndexMut] implementation.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use pds_spiking_nn::{NNBuilder, lif::*};
    /// let mut nn = NNBuilder::<LeakyIntegrateFire, _>::new()
    ///     .layer(
    ///         [
    ///             LifNeuron::new(&LifNeuronConfig::new(1.2, 0.4, 2.7, 0.9)),
    ///             LifNeuron::new(&LifNeuronConfig::new(0.8, 0.2, 2.5, 1.2)),
    ///             LifNeuron::new(&LifNeuronConfig::new(1.1, 0.6, 3.1, 1.2))
    ///         ],
    ///         [1.0, 1.1, 0.7],
    ///         [[0.0, -0.2, -0.3], [-0.2, 0.0, -0.1], [-0.4, -0.3, 0.0]]
    ///     )
    ///     .build();
    /// 
    /// assert_eq!(nn[0][(1, 2)], -0.1);
    /// 
    /// // Modify an intra-weight
    /// *nn[0].get_intra_weight_mut(1, 2).unwrap() = -0.3;
    /// 
    /// assert_eq!(nn[0][(1, 2)], -0.3);
    /// ```
    pub fn get_intra_weight_mut(&mut self, from: usize, to: usize) -> Option<&mut f64> {
        self.intra_weights.get_mut((from, to))
    }

    /// Returns an ordered iterator over all the neurons in this layer.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use pds_spiking_nn::{NNBuilder, lif::*};
    /// let nn = NNBuilder::<LeakyIntegrateFire, _>::new()
    ///     .layer(
    ///         [
    ///             LifNeuron::new(&LifNeuronConfig::new(1.2, 0.4, 2.7, 0.9)),
    ///             LifNeuron::new(&LifNeuronConfig::new(0.8, 0.2, 2.5, 1.2)),
    ///             LifNeuron::new(&LifNeuronConfig::new(1.1, 0.6, 3.1, 1.2))
    ///         ],
    ///         [1.0, 1.1, 0.7],
    ///         [[0.0, -0.2, -0.3], [-0.2, 0.0, -0.1], [-0.4, -0.3, 0.0]]
    ///     )
    ///     .build();
    /// 
    /// let mut neurons = nn[0].iter_neurons();
    /// 
    /// assert!(neurons.next().is_some());
    /// assert!(neurons.next().is_some());
    /// assert!(neurons.next().is_some());
    /// 
    /// assert!(neurons.next().is_none());
    /// ```
    pub fn iter_neurons(&self) -> <&Vec<M::Neuron> as IntoIterator>::IntoIter {
        self.neurons.iter()
    }

    /// Returns an ordered mutable iterator over all the neurons in this layer.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use pds_spiking_nn::{NNBuilder, lif::*};
    /// let mut nn = NNBuilder::<LeakyIntegrateFire, _>::new()
    ///     .layer(
    ///         [
    ///             LifNeuron::new(&LifNeuronConfig::new(1.2, 0.4, 2.7, 0.9)),
    ///             LifNeuron::new(&LifNeuronConfig::new(0.8, 0.2, 2.5, 1.2)),
    ///             LifNeuron::new(&LifNeuronConfig::new(1.1, 0.6, 3.1, 1.2))
    ///         ],
    ///         [1.0, 1.1, 0.7],
    ///         [[0.0, -0.2, -0.3], [-0.2, 0.0, -0.1], [-0.4, -0.3, 0.0]]
    ///     )
    ///     .build();
    /// 
    /// let mut neurons = nn[0].iter_mut_neurons();
    /// 
    /// // Modify the first neuron
    /// *neurons.next().unwrap() = LifNeuron::new(&LifNeuronConfig::new(1.0, 0.5, 2.7, 1.0));
    /// ```
    pub fn iter_mut_neurons(&mut self) -> <&mut Vec<M::Neuron> as IntoIterator>::IntoIter {
        self.neurons.iter_mut()
    }

    /// Consumes _self_ and returns a sorted iterator over all the neurons in this layer.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use pds_spiking_nn::{NNBuilder, lif::*};
    /// let nn = NNBuilder::<LeakyIntegrateFire, _>::new()
    ///     .layer(
    ///         [
    ///             LifNeuron::new(&LifNeuronConfig::new(1.2, 0.4, 2.7, 0.9)),
    ///             LifNeuron::new(&LifNeuronConfig::new(0.8, 0.2, 2.5, 1.2)),
    ///             LifNeuron::new(&LifNeuronConfig::new(1.1, 0.6, 3.1, 1.2))
    ///         ],
    ///         [1.0, 1.1, 0.7],
    ///         [[0.0, -0.2, -0.3], [-0.2, 0.0, -0.1], [-0.4, -0.3, 0.0]]
    ///     )
    ///     .build();
    /// 
    /// // Owned iterator over all neurons of every layer of nn
    /// let neurons = nn.into_iter().map(|layer| layer.into_iter_neurons()).flatten();
    /// ```
    pub fn into_iter_neurons(self) -> <Vec<M::Neuron> as IntoIterator>::IntoIter {
        self.neurons.into_iter()
    }
}

impl<M: Model> Index<usize> for Layer<M> {
    type Output = M::Neuron;

    fn index(&self, index: usize) -> &Self::Output {
        &self.neurons[index]
    }
}

impl<M: Model> IndexMut<usize> for Layer<M> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.neurons[index]
    }
}

impl<M: Model> Index<(usize, usize)> for Layer<M> {
    type Output = f64;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.intra_weights[index]
    }
}

impl<M: Model> IndexMut<(usize, usize)> for Layer<M> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.intra_weights[index]
    }
}
