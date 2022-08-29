//! Implementation of the Leaky Integrate and Fire (LIF) model for spiking neural networks

use crate::Model;

/// A struct for a single Neuron of the SNN.
/// Each Neuron has its own parameters such as _current membrane tension_, _threshold tension_ etc...
/// 
/// # Examples
/// 
/// Create a single neuron through a [LifNeuronConfig]:
/// 
/// ```
/// # use pds_spiking_nn::lif::*;
/// // Create the "config". This can be used to build a new neuron
/// let nc = LifNeuronConfig::new(1.2, 0.6, 3.0, 1.0);
/// 
/// // Get the neuron from the config
/// let neuron = LifNeuron::new(&nc);
/// ```
#[derive(Clone, Debug)]
pub struct LifNeuron {
    /// Rest potential
    pub v_rest: f64,
    /// Reset potential
    pub v_reset: f64,
    /// Threshold potential
    pub v_threshold: f64,
    /// Membrane's time constant. This is the product of its capacity and resistance
    pub tau: f64,
}

/// A struct with variables only used in simulation (solve)
#[derive(Clone, Debug, Default)]
pub struct LifSolverVars {
    v_mem: f64,
    ts_old: u128,  
}

impl From<&LifNeuron> for LifSolverVars {
    fn from(neuron: &LifNeuron) -> Self {
        Self {
            v_mem: neuron.v_rest,
            ts_old: 0
        }
    }
}

impl LifSolverVars {

    ///Get variables only used in simulation (solve) -> (v_mem, ts_old)
    pub fn get_vars(&mut self) -> (f64, u128){

        (self.v_mem, self.ts_old)
    }
}

/// A struct used to create a specific configuration, simply reusable for other neurons
/// 
/// # Examples
/// 
/// ```
/// # use pds_spiking_nn::lif::*;
/// let config_one = LifNeuronConfig::new(0.9, 0.3, 2.5, 1.3);
/// let config_two = LifNeuronConfig::new(1.3, 0.6, 2.6, 0.9);
/// 
/// let neuron_one = LifNeuron::new(&config_one);
/// let neuron_two = LifNeuron::new(&config_one);
/// // ...
/// let neuron_four = LifNeuron::new(&config_two);
/// ```
#[derive(Clone, Debug)]
pub struct LifNeuronConfig {
    v_rest: f64,
    v_reset: f64,
    v_threshold: f64,
    tau: f64
}

impl From<&LifNeuronConfig> for LifNeuron {
    fn from(lif_nc: &LifNeuronConfig) -> Self {
        Self::new(lif_nc)
    }
}

/// Simd aggregate of four [LifNeuron]s
#[cfg(feature = "simd")]
pub struct LifNeuronx4 {
    v_rest: packed_simd::f64x4,
    v_reset: packed_simd::f64x4,
    v_threshold: packed_simd::f64x4,
    tau: packed_simd::f64x4
}

/// Simd aggregate of four [LifSolverVars]
#[cfg(feature = "simd")]
pub struct LifSolverVarsx4 {
    v_mem: packed_simd::f64x4,
    ts_old: packed_simd::f64x4
}

/// Model provided by this library as example.
/// 
/// You can this empty type to construct lif NNs, see the documentation at [NNBuilder](crate::NNBuilder) for details.
#[derive(Clone, Copy, Debug)]
pub struct LeakyIntegrateFire;

impl Model for LeakyIntegrateFire {
    type Neuron = LifNeuron;
    type SolverVars = LifSolverVars;
    type Config = LifNeuronConfig;

    /// Update the value of current membrane tension, reading any new spike.
    /// When the neuron receives one or more impulses, it computes the new tension of the membrane,
    /// and saves the updated value in the provided `SolverVars` variable.
    /// 
    /// See [LifNeuron] struct for further info.
    /// 
    /// # Examples
    /// We create a general Neuron, called _neuron_one_.
    /// 
    /// This neuron receives a spike at time of spike _ts_ from a number of its input synapses.
    /// The overall weighted input value of this spike (i.e. the sum, across every lit up input synapse,
    /// of the weight of that synapse) is provided via the _weighted_input_val_ parameter.
    /// 
    /// The output of this function is 1.0 iff the neuron has generated a new spike at time _ts_, or 0.0 otherwise.
    /// 
    /// ```
    /// # use pds_spiking_nn::{Model, lif::*};
    /// let config_one = LifNeuronConfig::new(1.1, 0.4, 2.4, 1.1);
    /// let neuron_one = LifNeuron::new(&config_one);
    /// # let ts = 1;
    /// # let weighted_input_val = 1.0;
    /// # let mut vars = From::from(&neuron_one);
    /// 
    /// let output = LeakyIntegrateFire::handle_spike(&neuron_one, &mut vars, weighted_input_val, ts);
    /// assert!(output == 0.0 || output == 1.0);
    /// ```
    #[inline]
    fn handle_spike(neuron: &LifNeuron, vars: &mut LifSolverVars, weighted_input_val: f64, ts: u128) -> f64 {
        // This early exit serves as a small optimization
        if weighted_input_val == 0.0 { return 0.0 }
        
        let delta_t: f64 = (ts - vars.ts_old) as f64;
        vars.ts_old = ts;

        // compute the new v_mem value
        vars.v_mem = neuron.v_rest + (vars.v_mem - neuron.v_rest) * (-delta_t / neuron.tau).exp() + weighted_input_val;

        if vars.v_mem > neuron.v_threshold {
            vars.v_mem = neuron.v_reset;
            1. 
        } else {
            0.
        }
    }

    #[cfg(feature = "simd")]
    type Neuronx4 = LifNeuronx4;
    #[cfg(feature = "simd")]
    type SolverVarsx4 = LifSolverVarsx4;

    #[cfg(feature = "simd")]
    #[inline]
    fn neuron_x4_from_neurons(neurons: &[LifNeuron]) -> LifNeuronx4 {
        LifNeuronx4 {
            v_rest: From::from([neurons[0].v_rest, neurons[1].v_rest, neurons[2].v_rest, neurons[3].v_rest]),
            v_reset: From::from([neurons[0].v_reset, neurons[1].v_reset, neurons[2].v_reset, neurons[3].v_reset]),
            v_threshold: From::from([neurons[0].v_threshold, neurons[1].v_threshold, neurons[2].v_threshold, neurons[3].v_threshold]),
            tau: From::from([neurons[0].tau, neurons[1].tau, neurons[2].tau, neurons[3].tau])
        }
    }
    #[cfg(feature = "simd")]
    #[inline]
    fn vars_x4_from_vars(vars: &[LifSolverVars]) -> LifSolverVarsx4 {
        LifSolverVarsx4 {
            v_mem: From::from([vars[0].v_mem, vars[1].v_mem, vars[2].v_mem, vars[3].v_mem]),
            ts_old: From::from([vars[0].ts_old as _, vars[1].ts_old as _, vars[2].ts_old as _, vars[3].ts_old as _])
        }
    }
    #[cfg(feature = "simd")]
    #[inline]
    fn handle_spike_x4(neurons: &Self::Neuronx4, vars: &mut Self::SolverVarsx4, weighted_input_vals: packed_simd::f64x4, ts: u128) -> packed_simd::f64x4 {
        use packed_simd::f64x4;

        // Mixing u128x4 (512 bit vectors) with f64x4 (256 bit vectors) pretty much nullifies any performance improvement
        // that we may get from explicit simd (on systems without AVX-512 or similar anyway)
        let ts = f64x4::splat(ts as _);
        let dt: f64x4 = ts - vars.ts_old;
        vars.ts_old = ts;
        
        // The exp() right here is the only reason why I went with packed_simd instead of the portable_simd in std
        vars.v_mem = neurons.v_rest + (vars.v_mem - neurons.v_rest) * (-dt / neurons.tau).exp() + weighted_input_vals;

        let fired = vars.v_mem.gt(neurons.v_threshold);
        vars.v_mem = fired.select(neurons.v_reset, vars.v_mem);

        fired.select(f64x4::splat(1.0), f64x4::splat(0.0))
    }
}

// IMPLEMENTATION FOR LIF NEURONS & LIF NEURON CONFIG

impl LifNeuron {
    /// Create a new [LifNeuron] from a reference to a [LifNeuronConfig].
    /// 
    /// The same conversion can be obtained via the impl of `From<&LifNeuronConfig> for LifNeuron`.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use pds_spiking_nn::lif::*;
    /// let config = LifNeuronConfig::new(1.0, 0.5, 2.0, 1.0);
    /// let neuron = LifNeuron::new(&config);
    /// ```
    pub fn new(nc: &LifNeuronConfig) -> LifNeuron {
        LifNeuron {
            // parameters
            v_rest:  nc.v_rest,
            v_reset:  nc.v_reset ,
            v_threshold:  nc.v_threshold ,
            tau:  nc.tau,
        }
    }

    /// Create a new array of [LifNeuron] structs, starting from a given array of [LifNeuronConfig].
    /// 
    /// If _ncs_ contains a single element, it will be used for 
    /// all the _dim_ neurons required.
    /// Otherwise it will create a Neuron for each specified NeuronConfig
    /// 
    /// # Panics
    /// 
    /// Panics if the NeuronConfig array has a lenght (greater than one) 
    /// which differs from _'dim'_.
    /// 
    /// # Examples
    /// 
    /// Create many identical neurons from a single config:
    /// 
    /// ```
    /// # use pds_spiking_nn::lif::*;
    /// let config = vec![LifNeuronConfig::new(1.0, 0.5, 2.0, 1.0)];
    /// let neurons = LifNeuron::new_vec(config, 10);
    /// 
    /// assert_eq!(neurons.len(), 10);
    /// ```
    /// Create many different neurons from an array of different configs:
    /// 
    /// ```
    /// # use pds_spiking_nn::lif::*;
    /// let configs = vec![
    ///     LifNeuronConfig::new(1.0, 0.5, 2.0, 1.0),
    ///     LifNeuronConfig::new(1.1, 0.4, 2.1, 0.9),
    ///     LifNeuronConfig::new(1.2, 0.3, 2.2, 0.8)
    /// ];
    /// let neurons = LifNeuron::new_vec(configs, 3);
    /// 
    /// assert_eq!(neurons.len(), 3);
    /// ```
    /// Panics if dimensions don't match up:
    /// 
    /// ```should_panic
    /// # use pds_spiking_nn::lif::*;
    /// let configs = vec![
    ///     LifNeuronConfig::new(1.0, 0.5, 2.0, 1.0),
    ///     LifNeuronConfig::new(1.1, 0.4, 2.1, 0.9),
    ///     LifNeuronConfig::new(1.2, 0.3, 2.2, 0.8)
    /// ];
    /// let neurons = LifNeuron::new_vec(configs, 10); // Panic! expected 3, received 10
    /// ```
    pub fn new_vec(ncs: Vec<LifNeuronConfig>, dim: usize) -> Vec<LifNeuron>{
        let mut res: Vec<LifNeuron> = Vec::with_capacity(dim);

        // you can specify a single NeuronConfig block 
        // and it will be used for all neuron you asked
        if ncs.len() == 1{  
            let nc = &ncs[0];
            for _i in 0..dim {
                res.push(LifNeuron::new(nc));
            }
        }

        //or you can specify an array of NeuronConfig, one for each neuron
        else {
            if ncs.len() != dim{
                panic!("--> X  Error: Number of configuration and number of Neurons differ!")
            }

            res = ncs.iter().map(|cfg| cfg.into()).collect();
        }

        res
    }

}

impl LifNeuronConfig {
    /// Create a new [LifNeuronConfig], which can be used to build one or more identical neurons.
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use pds_spiking_nn::lif::*;
    /// let config = LifNeuronConfig::new(1.0, 0.5, 2.0, 1.0);
    /// let neuron: LifNeuron = From::from(&config);
    /// ```
    pub fn new(
        v_rest: f64,
        v_reset: f64,
        v_threshold: f64,
        tau: f64
    ) -> LifNeuronConfig
    {
        LifNeuronConfig{
            v_rest,
            v_reset,
            v_threshold,
            tau
        }
    }
}
