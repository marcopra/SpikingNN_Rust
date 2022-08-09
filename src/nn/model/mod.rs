//! Main `Model` trait for expanding this library to work with other models. Leaky integrate and fire is built in.

pub mod lif;

use std::fmt::Debug;

/// An applicable model for spiking neural networks
pub trait Model: 'static + Debug + Clone {
    /// A struct for a single Neuron of the SNN.
    /// Each Neuron has its own parameters such as _current membrane tension_, _threshold tension_ etc...
    type Neuron: 'static + Sized + Clone + Sync + RefInto<Self::SolverVars>;

    /// Contains the dynamic variables for each Neuron used by the solver
    type SolverVars: Default + Send + Sync;
    
    /// Helper type to build neurons
    type Config: RefInto<Self::Neuron>;

    /// Receive the incoming spike and update the vars for the given neuron.
    /// 
    /// _weighted_input_vals_ is the sum of every input weight to the neuron that is spiking.
    /// 
    /// This function must return either 1.0 in case the neuron generated a spike, or 0.0 otherwise.
    fn handle_spike(neuron: &Self::Neuron, vars: &mut Self::SolverVars, weighted_input_val: f64, ts: u128) -> f64;

    /// Structure that's responsible for 4 consecutive neurons of the same layer
    #[cfg(feature = "simd")]
    type Neuronx4: Send;

    /// Structure that's responsible for 4 `SolverVars`
    #[cfg(feature = "simd")]
    type SolverVarsx4: Send;
    
    /// Generate a [Neuronx4] from 4 [Neuron]s.
    /// 
    /// # Panics
    /// 
    /// Panics if _neurons_'s length is less than 4.
    #[cfg(feature = "simd")]
    fn neuron_x4_from_neurons(neurons: &[Self::Neuron]) -> Self::Neuronx4;

    /// Generate a [SolverVarsx4] from 4 [SolverVars].
    /// 
    /// # Panics
    /// 
    /// Panics if _vars_'s length is less than 4.
    #[cfg(feature = "simd")]
    fn vars_x4_from_vars(vars: &[Self::SolverVars]) -> Self::SolverVarsx4;

    /// Same as `handle_spike` but for a simd structure of 4 neurons simultaneously
    #[cfg(feature = "simd")]
    fn handle_spike_x4(neurons: &Self::Neuronx4, vars: &mut Self::SolverVarsx4, weighted_input_val: packed_simd::f64x4, ts: u128) -> packed_simd::f64x4;
}

/// A type is [RefInto<T>] if its reference can be converted to `T`.
/// 
/// Unfortunately, the Rust compiler currently has trouble keeping track of bounds of this kind,
/// so every time this bound is needed, it's necessary to explicitly "request" it.
/// 
/// See: <https://users.rust-lang.org/t/hrtb-in-trait-definition-for-associated-types/78687>
pub trait RefInto<T> { }
impl<T, U> RefInto<T> for U where for<'a> &'a U: Into<T> { }
