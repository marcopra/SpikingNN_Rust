
use std::process::Output;

use crate::{nn::{Spike, NN}, Model};
use ndarray::{Array2, OwnedRepr, ArrayBase, Dim};

/// A struct used to manage the input spikes given to a specified NN, 
/// in order to generate output spikes
pub struct Solver<M: Model>{
    input_spikes: Vec<Spike>,
    network: NN<M>,
    sim_network: SimulatedNN<M>,
    output_spikes: Vec<Spike>
}

struct SimulatedNeuron<M: Model> { 
    vars: M::SolverVars
}

impl<M: Model> SimulatedNeuron<M> {
    pub fn new() -> SimulatedNeuron<M>{
        SimulatedNeuron { vars: Default::default() }
    }
}
struct SimulatedNN<M: Model> {
    layers: Vec<Vec<SimulatedNeuron<M>>>
}

impl<M: Model> SimulatedNN<M> {

    fn new() -> Self{
        Self { 
            layers: Vec::new(),
        }
    }

    fn add_layer(&mut self, layer: Vec<SimulatedNeuron<M>>){
        self.layers.push(layer);
    }
}

impl<M: Model> Solver<M> {
    
    pub fn new(input_spikes: Vec<Spike>, network: NN<M>) -> Solver<M> {
        Solver { 
            input_spikes, 
            sim_network: Solver::init_neuron_vars(&network),
            network, 
            output_spikes: Vec::new() }
    }

    /// Each spike of the input_spike vec is sent to the corresponding neuron 
    /// of the input layer, one by one.
    pub fn solve(&mut self) -> Vec<Vec<u128>>{

        //Neuron variables inizialization 
        let mut sim_network = Self::init_neuron_vars(&(self.network));
        let mut nn_output: Vec<Vec<u128>> = Vec::new();
        

        for spike in self.input_spikes.iter() {

            //prende la dim dell'input
            let dim_input = self.network.layers[0].neurons.len();

            //Crea il primo array da moltiplicare con la prima matrice (diagonale) dei pesi (matrice di input)
            let spike_array = single_spike_to_vec(spike.neuron_id, dim_input);

            //Propagation of spikes inside the network
            let res = Solver::infer_spike_vec(&self.network, &mut sim_network, spike_array, spike.ts);
            nn_output.push(res);
        }
        nn_output
    }

    /// _*--> (Internal Use Only)*_
    /// 
    /// Create a temporary NN, parallel to the real one passed as a parameter
    /// 
    /// This new NN will contain only variables like v_mem, ts_old etc
    fn init_neuron_vars(network: & NN<M>) -> SimulatedNN<M> {
        
        let mut sim_nn = SimulatedNN::new();
        let mut sim_neuron: SimulatedNeuron<M>;
        let mut sim_layer: Vec<SimulatedNeuron<M>>;

        for layer in network.layers.iter() {
            sim_layer = Vec::with_capacity(layer.neurons.len());

            for neuron in layer.neurons.iter() {
                sim_neuron = SimulatedNeuron::new();
                sim_layer.push(sim_neuron);
            }
            sim_nn.add_layer(sim_layer);
        }
        sim_nn
    }

    /// Propagate Spikes inside the network and then create a Vec of spike
    fn infer_spike_vec(
                network: & NN<M>, 
                sim_network: &mut SimulatedNN<M>, 
                spike_vec: ArrayBase<OwnedRepr<f64>, 
                Dim<[usize; 2]>>, ts: u128) -> Vec<u128> {

        let mut out_spikes: Vec<u128>;

        //crea il vettore che tiene le spike generate dai vari neuroni 
        let mut output_vec: Vec<f64> = Vec::new();

        //init del vettore che conterrà i parametri del layer i-esimo
        let mut neuron_params: &Vec<M::Neuron> ;
        //del vec che conterrà le variabili del simulated_layer i-esimo
        let mut neuron_vars: &mut Vec<SimulatedNeuron<M>> ;
        

        let mut current_spike_vec = spike_vec;

        //per ogni layer della rete prende il layer e il rispettivo layer simulato (Con le vars)
        // crea i vettori di support per l'input e per l'output del layer i-esimo 

        // Elabora per ogni neurone del layer il rispettivo output (se spike o meno)
        for (layer, sim_layer) in network.layers.iter().zip(&mut sim_network.layers){
            
            //prende i params del layer i-esimo
            neuron_params = &layer.neurons;
            //prende le vars del layer i-esimo
            neuron_vars = sim_layer;

            // qui current_spike_vec è qualcosa del tipo [0 1 0 0 0]' oppure  [ 0 1 0 0 1] ed
            // è generato dal layer precedente.
            //creo il vettore dei valori di input per i neuroni ricevuti dal layer precedente, tramite prodotto vec x mat
            
            println!("EEE OOOOO O OO O O  {}\n\n STTTOOOOP", current_spike_vec);
            println!("EEE OOOOO O OO O O  {}\n\n STTTOOOOP", layer.input_weights);
            let weighted_input_val = current_spike_vec.dot(&layer.input_weights);

            // per ogni neurone, attivo la funzione handle_spike coi suoi parametri e le sue variabili, 
            // prese dai vettori inizializzati precedentemente
            // raccolgo l'output nel vettore
            // Gestisce gli input dal layer precedente
            for (i, neuron) in layer.neurons.iter().enumerate(){
                
                let res = M::handle_spike(neuron, 
                    &mut neuron_vars[i].vars, 
                    weighted_input_val[[0,i]], 
                    ts);
                output_vec.push(res);
            }

            //aggiorna la spike di input corrente con il vettore di spike appena creato
            println!("EEE OOOOO O OO O O  {:?}\n\n STTTOOOOP", current_spike_vec);
            current_spike_vec =  Array2::from_shape_vec([1, output_vec.len()], output_vec.clone()).unwrap();

            /*una volta che il layer ha elaborato l'input, bisogna simulare 
            le spike che arrivano ai neuroni dello stesso strato usando il nuovo current_spike_vec aggiornato*/

            //Svuota il vettore che tiene le spike generate dai vari neuroni
            // e lo prepara per la prossima iterazione sui layer
            output_vec.clear();

            //creo il vettore dei valori di input per i neuroni ricevuti dal neurone dello stesso layer che ha fatto la spike, tramite prodotto vec x mat
            let intra_layer_input_val = current_spike_vec.dot(&layer.intra_weights);

            // per ogni neurone, attivo la funzione handle_spike coi suoi parametri e le sue variabili, 
            // prese dai vettori inizializzati precedentemente
            // raccolgo l'output nel vettore
            // Gestisce gli input dai neuroni dello stesso layer
            for (i, neuron) in layer.neurons.iter().enumerate(){
                
                M::handle_spike(neuron, 
                    &mut neuron_vars[i].vars, 
                    intra_layer_input_val[[0,i]], 
                    ts);
            }
        }

        out_spikes = to_u128_vec(&output_vec, ts);
        out_spikes


    }

    //TODO CERCARE DI UNIRE QUESTA FUNZIONE ALLA INFER_SPIKE

    fn apply_spike_to_input_layer_neuron(
                                neuron_id: usize, 
                                ts: u128, 
                                network: &NN<M>, 
                                sim_network: &mut SimulatedNN<M>)-> Array2<f64> {

        //get dimension of the input layer
        let n_neurons_layer0 = network.layers[0].neurons.len();

        //input val for neuron_id-th neuron is 1 times the corresponding input_weight
        let weighted_input_val: f64 = network.layers[0].input_weights[(0, neuron_id)];  

        //Obtain the neuron_id-th neuron (parameters and variables) from the input layer 
        let neuron_params = &network.layers[0].neurons[neuron_id];
        let neuron_vars = &mut sim_network.layers[0][neuron_id].vars;

        //faccio handle_spike(spike) e ritiriamo il suo output (una sorta di spike ma per gestione interna)
        let spike_result = M::handle_spike(neuron_params, neuron_vars, weighted_input_val, ts);
        
        //vettore con un solo elemento a 1 in posizione neuro_id-esima
        let mut vec_spike: Vec<f64> = Vec::new();
        
        let arr_spike = single_spike_to_vec(neuron_id, n_neurons_layer0);

        let intra_layer_weights = &network.layers[0].intra_weights;
        
        //Vettore di valori da dare agli altri neuroni dello stesso layer
        let intra_layer_weighted_val = arr_spike.dot(intra_layer_weights);

        //Per ogni altro neurone del layer (Tutti tranne quello che riceve la 
        //spike in ingresso) calcoliamo la nuova tensione
        for n_id in 0..n_neurons_layer0 {
            if n_id != neuron_id{
                let neuron = &network.layers[0].neurons[n_id];
                let sim_neuron = &mut sim_network.layers[0][n_id].vars;
                M::handle_spike(
                        neuron, 
                        sim_neuron,  
                        intra_layer_weighted_val[[n_id,0]], 
                        ts);           
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

    /// Create a zero array, but with a single '1' in the neuron_id-th position
    /// 
    /// # Example 
    /// 
    ///  TODO @marcopra (prova)
    fn single_spike_to_vec(neuron_id: usize, dim: usize) -> ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> {

        let mut res: Vec<f64> = Vec::new();

        for i in 0..dim {
            if i == neuron_id {res.push(1.);}
            else {res.push(0.)}
        }
        Array2::from_shape_vec([1, dim], res).unwrap()
    }

    /// Create a vec of u128 (val_to_set) starting from a f64 array and a val to use if the f64 is greater than 0 
    /// 
    /// If in the i-th position the val of he input vec is greater than 0, the new vec will have 'val_to_set in that position, otherwise it will have a 0
    fn to_u128_vec(vec: &Vec<f64>, val_to_set: u128) -> Vec<u128>{

        let mut res: Vec<u128> =  Vec::new();

        for &val in vec.iter(){

            if val > 0.0 { res.push(val_to_set)}
            else {res.push(0_u128)};
        }   
        res 
    }

#[cfg(test)]
mod tests {
    
    #[test]
    fn test_init_simulated_nn() {
        
    }

    fn test_correct_management_of_example_spike(){

    }
}