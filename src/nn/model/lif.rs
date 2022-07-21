use crate::{Model, nn::Spike};

#[derive(Clone, Debug)]
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
pub struct LifNeuron{
    v_mem_current: f64,
    v_mem_old: f64,
    v_rest: f64,
    v_reset: f64,
    v_threshold: f64,
    tau: f64,
    ts_old: u128,  
    ts_curr: u128
}



#[derive(Clone, Debug)]
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
pub struct LifNeuronConfig {
    v_mem_current: f64,
    v_rest: f64,
    v_reset: f64,
    v_threshold: f64,
    tau: f64
}

impl From<LifNeuronConfig> for LifNeuron {
    fn from( lif_nc: LifNeuronConfig) -> Self {
        
        LifNeuron {
            //parameters
            v_mem_current: lif_nc.v_mem_current ,
            v_mem_old: 0.0,
            v_rest:  lif_nc.v_rest,
            v_reset:  lif_nc.v_reset ,
            v_threshold:  lif_nc.v_threshold ,
            tau:  lif_nc.tau,
            ts_old: 0,
            ts_curr: 0
        }
    }
}


impl super::Neuron for LifNeuron {
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
    fn handle_spike<M>(&mut self, weighted_input_val: f64, nn: &crate::NN<M>) -> Option<crate::nn::Spike>
    where M: crate::Model<Neuron = Self>
    {
         let delta_t: f64 = (self.ts_old - self.ts_curr) as f64;

        //calcola il nuovo val
        self.v_mem_current = self.v_rest + (self.v_mem_old - self.v_rest) 
                        *(delta_t / self.tau).exp() + weighted_input_val;

        if self.v_mem_current > self.v_threshold{                       //TODO 
            self.v_mem_current = self.v_reset;

            //TODO change return...
            return Some( Spike::new(0, 1)); 
        }
        None
    }

    fn set_new_params(&mut self, nc: &Self::Config) {
        self.v_mem_current = nc.v_mem_current;
        self.v_rest = nc.v_rest;
        self.v_reset = nc.v_reset;
        self.v_threshold =  nc.v_threshold;
        self.tau = nc.tau;
    }
  
}

#[derive(Clone, Copy, Debug)]
pub struct LeakyIntegrateFire;

impl Model for LeakyIntegrateFire {
    type Neuron = LifNeuron;
    type Synapse = f64;
}


// IMPLEMENTATION FOR LIF NEURONS & LIF NEURON CONFIG

impl LifNeuron {
    pub fn new(nc: &LifNeuronConfig ) -> LifNeuron {

        LifNeuron {
            //parameters
            v_mem_current:  nc.v_mem_current ,
            v_mem_old: 0.0,
            v_rest:  nc.v_rest,
            v_reset:  nc.v_reset ,
            v_threshold:  nc.v_threshold ,
            tau:  nc.tau,

            //other indipendent from the LifNeuron
            ts_old: 0,
            ts_curr: 0
        }
    }

        /// Create a new array of Neuron structs, starting from a given array of NeuronConfig.
    /// 
    /// If the array of NeuronConfig contains a single element, it will be used for 
    /// all the _'dim'_ neurons required.
    /// Otherwise it will create a Neuron for each specified NeuronConfig
    /// 
    /// # Panic
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

            for i in 0..dim {
                res.push(LifNeuron::new(&ncs[i]));
            }
        }

        return res
        
    }

}


impl LifNeuronConfig {
    pub fn new(
        v_mem_current: f64,
        v_rest: f64,
        v_reset: f64,
        v_threshold: f64,
        tau: f64,) -> LifNeuronConfig{

        LifNeuronConfig{
            v_mem_current,
            v_rest,
            v_reset,
            v_threshold,
            tau
        }
    }
    
}

#[cfg(test)]
mod tests {
    use crate::Neuron;

    use super::{LifNeuron, LifNeuronConfig};
    
    #[test]
    fn test_config_neurons() {
        //Config Definitions
        let nc = LifNeuronConfig::new(
            0.3, 
            0.2,
            0.1, 
            0.45, 
            0.23);
    
        let nc2 = LifNeuronConfig::new(
            0.1, 
            2.3, 
            12.2, 
            0.8, 
            0.7);

        let ne = LifNeuron::from(nc.clone());
        let mut neuron = LifNeuron::new(&nc);

        neuron.set_new_params(&nc2);
    }


    // fn test_bozza(){
    //     let first_spike_input_n1: Vec<u16> = [Spike::new(12,1), Spike::new(34,3),Spike::new(123,2)].to_vec();
    //     let first_spike_input_n2: Vec<u8> = [0, 0, 0, 1, 0, 0, 0, 1, 0].to_vec();

    // }

}
