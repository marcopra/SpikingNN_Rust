use crate::Model;

use crate::matrix::Matrix;

use self::model::Layer;
use std::fmt;

pub mod model;
pub(crate) mod builder;
pub(crate) mod neuron;

/// Represents the 'spike' that stimulates a neuron in a spiking neural network.
///  
/// The parameter _'ts'_ stands for 'Time of the Spike' and represents the time when the spike occurs
/// while the parameter _'neuron_id'_ stands to

// TODO Provare Efficienza una tupla al posto di una struct
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Spike {
    ts: u128,
    neuron_id: usize
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
    ///  let my_vec = [11, 9, 23, 43, 42].to_vec();
    ///  let spike_vec_for_neuron_2 = Spike::spike_vec_for(neuron_id: 2, ts_vec: my_vec );
    /// 
    /// 
    /// ```
    /// 
    pub fn spike_vec_for(neuron_id: usize, ts_vec: Vec<u128>) -> Vec<Spike> {

        let mut spike_vec : Vec<Spike> = Vec::with_capacity(ts_vec.len());
        
        
        
        //Creating the Spikes array for a single Neuron
        for ts in ts_vec.into_iter() {
            spike_vec.push(Spike::new(ts, neuron_id));
        }

        //Order the ts vector
        spike_vec.sort();

        return spike_vec;
    }

    /// Create an ordered array starting from all the spikes sent to the NN.
    /// It takes a Matrix where each row i-th represents an array of spike for neuron i-th
    /// then a single Vec is created. Eventually the array is sorted
    /// 
    /// # Example
    /// 
    pub fn create_terminal_vec(spikes: Vec<Vec<Spike>>) -> Vec<Spike> {
        let mut res: Vec<Spike> = Vec::new();

        for line in spikes {
            for spike in line {
                res.push(spike);
            }
        }
        res.sort(); //ascending
        //res.sort_by(|a, b| a.ts.partial_cmp(&b.ts));
    
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
    /// Input weight for each of the `Neuron`s in the entry layer
    input_weights: Vec<M::Synapse>,
    /// All the layers of the neural network. Every layer contains the list of its `Neuron`s and
    /// a square `Matrix` for the intra-layer weights.
    layers: Vec<Layer<M>>,
    /// Vec of `Synapse` meshes between each consecutive pair of layers
    synapses: Vec<Matrix<M::Synapse>>
}




#[cfg(test)]
mod tests {
    use crate::nn::Spike;
    

    
    #[test]
    fn test_spike_vec_for(){
        
    }

    #[test]
    fn test_sort_spike(){

        let ts1 = [0, 1, 4, 5].to_vec();
        let ts2 = [0, 3, 6, 7].to_vec();
        let mut multiple_spike_vec : Vec<Vec<Spike>> = Vec::new();
        
        let spike1 = Spike::spike_vec_for(1, ts1);
        let spike2 = Spike::spike_vec_for(2, ts2);

        multiple_spike_vec.push(spike1);
        multiple_spike_vec.push(spike2);

        let input_vec = Spike::create_terminal_vec(multiple_spike_vec);

        for el in input_vec.iter(){
            println!("{:?}", el);
        }

        
        
    }

}

