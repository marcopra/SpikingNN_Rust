


use pds_spiking_nn::{Neuron, NeuronConfig};

fn main() {

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