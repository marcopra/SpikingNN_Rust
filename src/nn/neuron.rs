use core::panic;

/*
TODO: Implementare la possibilità di modificare 
i parametri dei vari neuroni singolarmente
*/


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
    parametres: Vec<f64>
}

//Da implementare nel Tratto MODEL
impl NeuronConfig {
    pub fn new(
        v_mem_current: f64,
        v_rest: f64,
        v_reset: f64,
        v_threshold: f64,
        tau: f64,) -> NeuronConfig{

        //load params into the vec
        let mut params: Vec<f64> = Vec::new();
        params.push(v_mem_current);
        params.push(v_rest);
        params.push(v_reset);
        params.push(v_threshold);
        params.push(tau);

        NeuronConfig{
            v_mem_current: v_mem_current,
            v_rest: v_rest,
            v_reset: v_reset,
            v_threshold: v_threshold,
            tau: tau,
            parametres: params.clone()
        }
    }
    
}


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
/// let nc = NeuronConfig::new(parm1, parm2, ...)
/// let neuron = Neuron::new(&nc)
/// ```
/// 
/// 
///
#[derive(Clone)]
pub struct Neuron{
    v_mem_current: f64,
    v_mem_old: f64,
    v_rest: f64,
    v_reset: f64,
    v_threshold: f64,
    tau: f64,
    ts_old: u128,  
    ts_curr: u128,
    pub parametres: Vec<f64>
}

impl Neuron {

    pub fn new(nc: &NeuronConfig ) -> Neuron {

        //TODO parametri individuali ridondanti, già contenuti nel vettore parametre,
        // usato per settare il neurone
        Neuron {
            //parametres
            v_mem_current:  nc.v_mem_current ,
            v_mem_old: 0.0,
            v_rest:  nc.v_rest,
            v_reset:  nc.v_reset ,
            v_threshold:  nc.v_threshold ,
            tau:  nc.tau,
            parametres: nc.parametres.clone(),

            //other indipendent from the neuron
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
    /// Finally **out_spike_train[[1]]** is a cell of an array which contains each spike generated 
    /// from neurons of this same layer.
    /// ```
    ///     let weighted_spike_val: Vec<f64> = [val1, val2, ...].to_vec();
    ///     let mut out_spike_train: Vec<f64> = Vec::new();
    /// 
    ///     let config_one = NeuronConfig::new(parm1, parm2, ...);
    ///     let neuron_one = Neuron::new(config_one);
    /// 
    ///     neuron_one.update_v_mem(time_of_spike, weighted_spike_val[1], &mut out_spike_train[1])
    /// ```
    /// After this code, the neuron may possibly have fired the spike.
    ///
    /// 
    pub fn update_v_mem(&mut self, t_curr_spike: u128, weighted_spikes_val: f64, out_spike: &mut f64){

        //TODO Implementare passaggio generico di una funzione, potenzialmente diversa dalla LIF
        // per renderla configurabile a piacere...

        let delta_t: f64 = (self.ts_old - self.ts_curr) as f64;

        //calcola il nuovo val
        self.v_mem_current = self.v_rest + (self.v_mem_old - self.v_rest) 
                        *(delta_t / self.tau).exp() + weighted_spikes_val;

        //e lo carica nell'array
        self.parametres[0] = self.v_mem_current;

        if self.v_mem_current > self.v_threshold{
            Neuron::fire_impulse(out_spike);
            self.v_mem_current = self.v_reset;
        }
    }


    //Readability...
    pub fn fire_impulse(out_spike: &mut f64){
        *out_spike = 1.0; 
        //TODO provare con i boolean che sono 1 bit, 
        // anche se si devono moltiplicare poi per dei pesi f64...
    }

    // CONFIGURATION CHANGE

    /// Use an array of parametres (correctly order) to set the array 
    /// of parametres of the neuron.
    /// This let you create a new NeuronConfig, and use it to modify neurons already created
    /// maybe for training or something else...
    pub fn set_new_param(&mut self, nc: &NeuronConfig){
        self.parametres = nc.parametres.clone();
        self.update_parametres();
    }

    //Da definire nell'implementazione del tratto

    /// A function that maps each parametre in the array, into an explicit parametre, 
    /// defined in the Neuron model. The number of parametres and the name associated 
    /// to them is defined in the Trait Implementation of Model and Neuron.
    /// 
    /// This function reads cells of Neuron.parametres, and overwrite the old value 
    /// of a specified parametre, with the new value read.
    /// 
    /// YOU MUST PAY ATTENTION TO THE ORDER OF ASSIGNMENTS
    pub fn update_parametres(&mut self){


        /*v_mem_current:  nc.v_mem_current ,
            v_rest:  nc.v_rest,
            v_reset:  nc.v_reset ,
            v_threshold:  nc.v_threshold ,
            tau:  nc.tau, */

        /*DUBBIO: 
        La V_mem_Current è un parametro del neurone o un 
        parametro di simulazione come ts_old ts_curr etc?
        Credo debba essere un parametro di simulazione perchè il neurone parte dalla v_rest*/

        //self.v_mem_current = self.parametres[0];
        self.v_rest = self.parametres[1];
        self.v_reset = self.parametres[2];
        self.v_threshold = self.parametres[3];
        self.tau = self.parametres[4];
    }

    pub fn print_params(&self){
        println!("--------------------");
        for param in self.parametres.iter(){
            println!("{}", param);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Neuron, NeuronConfig};
    
    #[test]
    fn test_config_neurons() {
        //Config Definitions
        let nc = NeuronConfig::new(
            0.3, 
            0.2,
            0.1, 
            0.45, 
            0.23);
    
        let nc2 = NeuronConfig::new(
            0.1, 
            2.3, 
            12.2, 
            0.8, 
            0.7);

        let mut neuron = Neuron::new(&nc);

        neuron.print_params();
        neuron.set_new_param(&nc2);
        neuron.print_params();
    }
}
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    


    
    
    
    
    
    
    
    
    
    
    

    
    

    

    
    
    

    
    
    

    
    
    
    
