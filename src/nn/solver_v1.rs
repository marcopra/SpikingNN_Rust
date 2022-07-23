
use crate::nn::{Spike, NN};
use ndarray::{Array2, OwnedRepr, ArrayBase, Dim};

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

        for spike in self.input_spikes.iter() {
            let spike_vec_from_input_layer = Self::apply_spike_to_input_layer_neuron(spike.neuron_id, spike.ts, &mut self.network);
            Self::infer_spike_vec(spike_vec_from_input_layer);
        }
    }

    fn infer_spike_vec(spike_vec: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>){
        todo!();
        return ;
    }

    //TODO fare doc per questa funzione che è un po complessa ad occhio
    //TODO trovare nome + esplicito e conciso
    fn apply_spike_to_input_layer_neuron(neuron_id: usize, ts: u128, network: &mut NN<M>)-> Array2<f64> {

        //[2 ]
        let n_neurons_layer0 = network.layers[0].0.len();

        let weighted_input_val: f64 = network.input_weights[neuron_id];  

         //prendo il neurone n_id-esimo dal layer
        let neuron = &mut network.layers[0].0[neuron_id];
        
        //faccio handle_spike(spike) e ritiriamo il suo output (una sorta di spike ma per gestione interna)
        let spike_result = M::handle_spike(neuron, weighted_input_val);
        
        //vettore con un solo elemento a 1 in posizione neuro_id-esima
        let mut vec_spike: Vec<f64> = Vec::new();

        //TODO Fare una funzione che racchiuda sti for brutti
        for i in 0..n_neurons_layer0 {
    
            if i == neuron_id{
                vec_spike[i] = 1.;
            }
            else{
                vec_spike[i] = 0.;
            }
        }
        
        let arr_spike = Array2::from_shape_vec((1, n_neurons_layer0), vec_spike).unwrap();

        let intra_layer_weights = &network.layers[0].1;
        
        //Vettore di valori da dare agli altri neuroni dello stesso layer
        let intra_layer_weighted_val = arr_spike.dot(intra_layer_weights);

        //Per ogni altro neurone del layer (Tutti tranne quello che riceve la 
        //spike in ingresso) calcoliamo la nuova tensione
        for n_id in 0..n_neurons_layer0 {
            if n_id != neuron_id{
                let neuron = &mut network.layers[0].0[n_id];
                M::handle_spike(neuron, intra_layer_weighted_val[[n_id,0]]);           
            }
        }
        
        return arr_spike;
    }

    /*
    pub fn SIMULT_solve(&mut self){

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


    fn SIMULT_apply_spike_to_input_layer_neuron(vec_neuron_id: Vec<usize>, ts: u128, network: &mut NN<M>) {

        //[2 ]
        let n_neurons_layer0 = network.layers[0].0.len();
        let mut input_vec : Vec<f64>= Vec::with_capacity(n_neurons_layer0);
        let mut index = 0;

        //costruisce il vettore di spike per il primo layer di input al tempo t_current
        for i in 0..input_vec.len() {
            
            if vec_neuron_id.contains(&i){
                input_vec[i] = 1.;
            }
            else{
                input_vec[i] = 0.;
            }
        }

        let mut weighted_input_val: Vec<f64> = Vec::new();

        for (&n, &w) in input_vec.iter().zip(network.input_weights.iter()) {
            weighted_input_val.push(n*w);  
        }
        
        let intra_layer_weights = network.layers[0].1;
        for ((&n, &w), ind) in input_vec.iter().zip(intra_layer_weights.iter()).enumerate() {
            weighted_input_val[ind] += n*w;  
        }
        
        
        //Per ogni neurone nel vettore vec_id (che hanno le spike simultanee)
        for &neuron_id in vec_neuron_id.iter(){
            //prendo il neurone n_id-esimo dal layer
            let neuron = &mut network.layers[0].0[neuron_id];
            
            //faccio handle_spike(spike) e ritiriamo il suo output (una sorta di spike ma per gestione interna)
            //TODO gestione intralayer
            let results = M::handle_spike(neuron, weighted_input_val[neuron_id]);

        }


        //TODO gestione intralayer
       

        //creo quindi un vettore di output del primo layer

        //moltiplichiamo il vettore di output per la matrice dei pesi (riga-> (vettore di spike)' x matrice -> matrice_pesi)'
        //e otteniamo il vettore di input per il layer successivo
        
    }
    */
}