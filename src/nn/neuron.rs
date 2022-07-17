use core::panic;
use libm::exp;

/*
TODO: Implementare la possibilità di modificare 
i parametri dei vari neuroni singolarmente
*/

/// Neuron
/// ------
/// 
/// A struct for a single Neuron of the SNN.
/// Each Neuron has its own parametres such as _current membrane tension_, _threshold tension_ etc...
/// 
/// Usage Example
/// --------------
/// 
/// ```
/// let neuron = Neuron::new(parm1, parm2, ...)
/// ```
/// 
/// 
/// 

pub struct Neuron{
    v_mem_current: f64,
    v_mem_old: f64,
    v_rest: f64,
    v_reset: f64,
    v_threshold: f64,
    tau: f64,
    ts_old: u128,  
    ts_curr: u128,
}

/// NeuronConfig
/// ------------
/// 
/// A struct used to create a specific configuration, simply reusable for other neurons
/// 
/// Example
/// --------
/// 
/// ```
/// let config_one = NeuronConfig::new(parm1, parm2, ...);
/// let config_two = NeuronConfig::new(parm1, parm2, ...);
/// 
/// let neuron_one = Neuron::new(config_one);
/// let neuron_two = Neuron::new(config_one);
///     ...
/// let neuron_four = Neuron::new(config_two);
/// ```
/// 
pub struct NeuronConfig{
    v_mem_current: f64,
    v_rest: f64,
    v_reset: f64,
    v_threshold: f64,
    tau: f64,
}

impl Neuron {

    pub fn new(nc: &NeuronConfig ) -> Neuron {

        Neuron {
            v_mem_current:  nc.v_mem_current ,
            v_mem_old: 0.0,
            v_rest:  nc.v_rest,
            v_reset:  nc.v_reset ,
            v_threshold:  nc.v_threshold ,
            tau:  nc.tau,
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
    pub fn new_vec(ncs: Vec<NeuronConfig>, dim: usize) -> Vec<Neuron>{
        
        let mut res: Vec<Neuron> = Vec::with_capacity(dim);

        // you can specify a single NeuronConfig block 
        // and it will be used for all neuron you asked
        if ncs.len() == 1{  
            let nc = &ncs[0];
            for i in 0..dim {
                res.push(Neuron::new(nc));
            }
        }

        //or you can specify an array of NeuronConfig, one for each neuron
        else {
            if ncs.len() != dim{
                panic!("--> X  Error: Number of configuration and number of Neurons differ!")
            }

            for i in 0..dim {
                res.push(Neuron::new(&ncs[i]));
            }
        }

        return res
        
    }

    /// Update the value of current membrane tension, reading any new spike.
    /// When the neuron receives one or more impulses, it compute the new tension of the membrane,
    /// thanks to a specific configurable model.
    /// 
    /// See **Neuron** struct for further info
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
    /// Finally **out_spike_train[1]** is a cell of an array which contains each spike generated 
    /// from neurons of this same layer.
    /// ```
    /// let weighted_spike_val: Vec<f64> = [val1, val2, val3, ...].to_vec();
    /// let mut out_spike_train: Vec<f64> = Vec::new();
    /// 
    /// let config_one = NeuronConfig::new(parm1, parm2, ...);
    /// let neuron_one = Neuron::new(config_one);
    /// 
    /// neuron_one.update_v_mem(time_of_spike, weighted_spike_val[1], &mut out_spike_train[1])
    /// ```
    /// After this code, the neuron may possibly have fired the spike.
    ///
    /// 
    pub fn update_v_mem(&mut self, t_curr_spike: u128, weighted_spikes_val: f64, out_spike: &mut f64){

        //TODO Implementare passaggio generico di una funzione, potenzialmente diversa dalla LIF
        // per renderla configurabile a piacere...

        let delta_t: f64 = (self.ts_old - self.ts_curr) as f64;

        self.v_mem_current = self.v_rest + (self.v_mem_old - self.v_rest) 
                        *exp( delta_t / self.tau ) + weighted_spikes_val;

        if self.v_mem_current > self.v_threshold{
            Neuron::fire_impulse(out_spike);
        }
    }

    //Readability...
    pub fn fire_impulse(out_spike: &mut f64){
        *out_spike = 1.0;
    }
}    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    


    
    
    
    
    
    
    
    
    
    
    

    
    

    

    
    
    

    
    
    

    
    
    
    
