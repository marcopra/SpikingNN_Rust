use crate::Model;

/// LifNeuron
/// ------
/// 
/// A struct for a single Neuron of the SNN.
/// Each Neuron has its own parameters such as _current membrane tension_, _threshold tension_ etc...
/// 
/// Usage Example
/// --------------
/// 
/// ```
/// let nc = LifNeuronConfig::new(parm1, parm2, ...)
/// let neuron = LifNeuron::new(&nc)
/// ```
/// 
/// 
///
#[derive(Clone, Debug)]
pub struct LifNeuron { // TODO: public fields?
    v_rest: f64,
    v_reset: f64,
    v_threshold: f64,
    tau: f64,
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

/// LifNeuronConfig
/// ------------
/// 
/// A struct used to create a specific configuration, simply reusable for other neurons
/// 
/// Example
/// --------
/// 
/// ```
/// let config_one = LifNeuronConfig::new(parm1, parm2, ...);
/// let config_two = LifNeuronConfig::new(parm1, parm2, ...);
/// 
/// let neuron_one = LifNeuron::new(config_one);
/// let neuron_two = LifNeuron::new(config_one);
///     ...
/// let neuron_four = LifNeuron::new(config_two);
/// ```
///
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

#[cfg(feature = "simd")]
pub struct LifNeuronx4 {
    v_rest: packed_simd::f64x4,
    v_reset: packed_simd::f64x4,
    v_threshold: packed_simd::f64x4,
    tau: packed_simd::f64x4
}

#[cfg(feature = "simd")]
pub struct LifSolverVarsx4 {
    v_mem: packed_simd::f64x4,
    ts_old: packed_simd::f64x4
}

#[derive(Clone, Copy, Debug)]
pub struct LeakyIntegrateFire;

impl Model for LeakyIntegrateFire {
    type Neuron = LifNeuron;
    type SolverVars = LifSolverVars;
    type Config = LifNeuronConfig;

    /// Update the value of current membrane tension, reading any new spike.
    /// When the neuron receives one or more impulses, it compute the new tension of the membrane,
    /// thanks to a specific configurable model.
    /// 
    /// See **LifNeuron** struct for further info
    /// 
    ///# Example
    /// _The following example is also a recomend usage template for layer made up of these neurons._
    /// 
    /// We create a general Neuron, called **neuron one**.
    /// 
    /// This neuron has (as Input) an associated cell of the spike vector
    /// created at time t+dt by the previous layer which is supposed to be 
    /// **weighted_spike_val[[1]]**.
    /// 
    /// Instead **time_of_spike** represents the actual instant when the spike occurs/takes place.
    ///
    /// Finally **out_spike_train[[1]]** is a cell of an array which contains each spike generated 
    /// from neurons of this same layer.
    /// ```
    ///     let weighted_spike_val: Vec<f64> = [val1, val2, ...].to_vec();
    ///     let mut out_spike_train: Vec<f64> = Vec::new();
    /// 
    ///     let config_one = LifNeuronConfig::new(parm1, parm2, ...);
    ///     let neuron_one = LifNeuron::new(config_one);
    /// 
    ///     neuron_one.update_v_mem(time_of_spike, weighted_spike_val[1], &mut out_spike_train[1])
    /// ```
    /// After this code, the neuron may possibly have fired the spike.
    
    //TODO Cambiare da Option<Spike> a 1 o 0 per uso interno per andare a creare il vettore di output da moltiplicare con la matrice
    #[inline]
    fn handle_spike(neuron: &LifNeuron, vars: &mut LifSolverVars, weighted_input_val: f64, ts: u128) -> f64 {
        // This early exit serves as a small optimization
        if weighted_input_val == 0.0 { return 0.0 }
        
        let delta_t: f64 = (ts - vars.ts_old) as f64;
        vars.ts_old = ts;

        //calcola il nuovo val
        vars.v_mem = neuron.v_rest + (vars.v_mem - neuron.v_rest) * (-delta_t / neuron.tau).exp() + weighted_input_val;

        if vars.v_mem > neuron.v_threshold {                       //TODO 
            vars.v_mem = neuron.v_reset;

            //TODO change return...
            //Fire Spike
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

        let ts = f64x4::splat(ts as _);
        let dt: f64x4 = ts - vars.ts_old;
        vars.ts_old = ts;
        
        vars.v_mem = neurons.v_rest + (vars.v_mem - neurons.v_rest) * (-dt / neurons.tau).exp() + weighted_input_vals;

        let fired = vars.v_mem.gt(neurons.v_threshold);
        vars.v_mem = fired.select(neurons.v_reset, vars.v_mem);

        fired.select(f64x4::splat(1.0), f64x4::splat(0.0))
    }
}

// IMPLEMENTATION FOR LIF NEURONS & LIF NEURON CONFIG

impl LifNeuron {
    pub fn new(nc: &LifNeuronConfig ) -> LifNeuron {
        LifNeuron {
            // parameters
            v_rest:  nc.v_rest,
            v_reset:  nc.v_reset ,
            v_threshold:  nc.v_threshold ,
            tau:  nc.tau,
        }
    }

    /// Create a new array of Neuron structs, starting from a given array of NeuronConfig.
    /// 
    /// If the array of NeuronConfig contains a single element, it will be used for 
    /// all the _'dim'_ neurons required.
    /// Otherwise it will create a Neuron for each specified NeuronConfig
    /// 
    /// # Panics
    /// Panics if the NeuronConfig array has a lenght (greater than one) 
    /// which differs from _'dim'_.
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
    pub fn new(
        v_rest: f64,
        v_reset: f64,
        v_threshold: f64,
        tau: f64,) -> LifNeuronConfig{

        LifNeuronConfig{
            v_rest,
            v_reset,
            v_threshold,
            tau
        }
    }
}
