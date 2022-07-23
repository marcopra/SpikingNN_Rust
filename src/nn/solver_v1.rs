
use crate::nn::{Spike, NN};
use ndarray::{Array2, arr1};

/// A struct used to manage the input spikes given to a specified NN, 
/// in order to generate output spikes
pub struct Solver<M: crate::nn::Model>{
    input_spikes: Vec<Spike>,
    network: NN<M>,
    output_spikes: Vec<Spike>
}

impl<M: crate::nn::Model> Solver<M> {
    

    pub fn new(input_spikes: Vec<Spike>, network: NN<M>) -> Solver<M> {
        Solver { 
            input_spikes, 
            network, 
            output_spikes: Vec::new() }
    }

    /// Each spike of the input_spike vec is sent to the corresponding neuron 
    /// of the input layer, one by one.
    pub fn solve(&mut self){

        //[{1, 1}, {2, 3}, {2, 2}, {3,4}]
        let mut t_current = self.input_spikes[0].ts;
        let mut vec_nid = Vec::new();

        for spike in self.input_spikes.iter() {
            
            //se
            if spike.ts != t_current {

                //elabora le spike all'istante precedente
                Self::apply_spike_to_input_layer_neuron(vec_nid, t_current, &mut self.network);
                vec_nid = Vec::new();

                // Aggiorna per la spike al tempo corrente
                vec_nid.push(spike.neuron_id);
                t_current = spike.ts;
            }
            else{
                vec_nid.push(spike.neuron_id);
            }

            //TODO gestire simultaneità
        }

        // Gestione dell'ultima spike..
        Self::apply_spike_to_input_layer_neuron(vec_nid, t_current, &mut self.network)
    }

    fn apply_spike_to_input_layer_neuron(vec_neuron_id: Vec<usize>, ts: u128, network: &mut NN<M>) {

        //[2 ]
        let n_neurons_layer0 = network.layers[0].0.len();
        let mut input_vec : Vec<f32>= Vec::with_capacity(n_neurons_layer0);
        let mut index = 0;

        //costruisce il vettore di spike per il primo layer di input al tempot t_current
        for i in 0..input_vec.len() {
            
            if vec_neuron_id.contains(&i){
                input_vec[i] = 1.0;
            }
            else{
                input_vec[i] = 0.0;
            }
        }

        //[0 1 0 0 1] vettore di input da moltiplicare con la matrice (diagonale in questo caso) dei pesi
        let mut weighted_input_val: Vec<M::Synapse> = Vec::new();

        /*for (&n, &w) in input_vec.iter().zip(network.input_weights.iter()) {
            
            //weighted_input_val.push(n*w);  
        }*/

        
        //Matrice dei pesi

        
        //[0 1 0 0 1] x diag(w1, w2, w3, k4, k5) = weighted_input_val
        //vado a prendere il neurone neuron_id-esimo
        for &neuron_id in vec_neuron_id.iter(){
            let neuron_vec = &network.layers[0].0;
            //let a = &neuron_vec[neuron_id].handle_spike(weighted_input_val, network);
        }
        //faccio handle_spike(spike) e ritiriamo il suo output (una sorta di spike ma per gestione interna)

        //creo quindi un vettore di output del primo layer

        //moltiplichiamo il vettore di output per la matrice dei pesi (riga-> (vettore di spike)' x matrice -> matrice_pesi)'
        //e otteniamo il vettore di input per il layer successivo
        
    }
}