use pds_spiking_nn::{NNBuilder, NN, Model, lif::{LeakyIntegrateFire, LifNeuron, LifNeuronConfig}, Spike, test_solver};

#[cfg(not(feature = "async"))]
fn solve_nn<M: Model>(nn: &NN<M>, spikes: Vec<Spike>) -> Vec<Vec<u128>>
where for<'a> &'a M::Neuron: Into<M::SolverVars> // This redundant trait bound is unfortunately needed due to a
                                                 // current limitation of the Rust compiler,
                                                 // see: <https://users.rust-lang.org/t/hrtb-in-trait-definition-for-associated-types/78687>
{
    nn.solve(spikes)
}

#[cfg(feature = "async")]
fn solve_nn<M: Model>(nn: &NN<M>, spikes: Vec<Spike>) -> Vec<Vec<u128>>
where for<'a> &'a M::Neuron: Into<M::SolverVars>
{
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    runtime.block_on(nn.solve(spikes))
}

fn main() {

    /* ******** Single-thread solver DEMO ******** */

    // Create a neural network
    
    // Create a new static builder.
    // This can be used to construct "static" nns,
    // i.e. nn whose size is known at compile time.
    let builder = NNBuilder::<LeakyIntegrateFire, _>::new();

    // 1. Passthrough Neural Network using a single thread solver 
    println!("DEMO 1: Passthrough Neural Network using a single thread solver ");
    

    // We can defina a reusable Neuron configuration
    // to semplify the building of our NN. 
    let config = LifNeuronConfig::new(2.0, 0.5, 2.1, 1.0);


    // The builder can be used by adding layers one by one,
    // via its 'layer' method, which consumes the builder and
    // returns a new instance.
    // The easiest way to use the builder is by chaining 'layer' calls,
    // like this
    let nn_passthrough = NNBuilder::<LeakyIntegrateFire, _>::new()
        // Every layer is defined by its neurons (with the given order!),
        // the input-weights (synapses from the previous layer, or network
        // inputs), and the intra-weights (mesh of synapses that connect
        // different neurons of the same layer)
        .layer(
            [
                // We use here the same neuron configuration for 
                // all the neurons in this layer.
                From::from(&config),
                From::from(&config),
                From::from(&config)
            ],
            [
                // We define here the Inter-weights matrix. It refers
                // to the links between the between this layer and the previous.
                // Note: if we are defining the 1st layer (Layer 0), the Inter-weights matrix
                //       refers to the input links of the NN.
                1.0, 1.0, 1.0
            ],
            [
                // Intra-weights are always square matrices.
                // The diagonal is however usually null, but it is not enforced
                // in this library, just be warned that "bad" networks can be
                // created, and attempting to solve them might result in
                // an infinite number of output spikes being generated.
                // In this case, we use a null-matrix to define the
                // passthrough NN
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0]
            ]
        )
        .build();
    
    // The function Spike::create_terminal_vec will produce the Vec that can be
    // then passed to NN::solve. It accepts a Vec<Vec<Spike>>, one inner Vec
    // for every neuron in the entry layer.
    let spikes = Spike::create_terminal_vec(
        vec![
            Spike::spike_vec_for(0, vec![1, 2, 3, 5, 6, 7]),
            Spike::spike_vec_for(1, vec![2, 6, 7, 9]),
            Spike::spike_vec_for(2, vec![2, 5, 6, 10, 11])
        ]
    );

    // For the sake of this demo, let's ensure the correctness of the result
    // produced by the internal single threaded solver. The single threaded solver
    // will be used later as comparator for the multi-threaded solver, available for the user.
    // Note: this is here just to showcase the test solver, but is not meant 
    //       to be a typical usage example.
    //       The test solver is - as the name implies - test-only.
    let output1_single_thread_solver = test_solver::Solver::new(spikes, nn_passthrough).solve();

    // This is expected result from the passthrough NN.
    let expected_output = vec![
        vec![1, 2, 3, 5, 6, 7],
        vec![2, 6, 7, 9],
        vec![2, 5, 6, 10, 11]
    ] ;

    // Check if the two outputs are the same 
    if output1_single_thread_solver != expected_output {
        panic!("Huh-oh, this should not happen. \n--------->DEMO 1: FAILED \n\n");
    }

    println!("---------> DEMO 1: COMPLETED \n\n");
    println!("-----------------------------------------------------------");


    // 2. multi-layer Neural Network example using a single thread solver 
    // Now, we show a demo of a general Neural Network. 
    // The expected result has been computed by hand
    println!("DEMO 2: General Neural Network using a single thread solver ");


    // Create a neural network

    // Here, we define a general multi-layer NN by using our building pattern.
    let nn = NNBuilder::<LeakyIntegrateFire, _>::new()
        // Layer 0 (1st Layer)
        .layer(
            [
                // Neurons require different parameters depending on the model
                // being used
                LifNeuron::new(&LifNeuronConfig::new(2.0, 0.5, 3.0, 1.6)),
                LifNeuron::new(&LifNeuronConfig::new(1.5, 0.4, 3.1, 1.2)),
                LifNeuron::new(&LifNeuronConfig::new(1.6, 0.3, 2.8, 1.1))
            ],
            [1.1, 1.2, 1.1],
            [
                [0.0, -0.1, -0.2],
                [-0.15, 0.0, -0.1],
                [-0.2, -0.15, 0.0]
            ]
        )
        // Layer 1 (2nd Layer)
        .layer(
            [
                LifNeuron::new(&LifNeuronConfig::new(1.8, 0.5, 2.8, 1.5)),
                LifNeuron::new(&LifNeuronConfig::new(1.7, 0.8, 2.6, 1.6))
            ],
            [
                [0.9, 0.85],
                [0.8, 0.9],
                [0.85, 0.7]
            ],
            [
                [0.0, -0.2],
                [-0.15, 0.0]
            ]
        )
        .build();

    let spikes = Spike::create_terminal_vec(
        vec![
            Spike::spike_vec_for(0, vec![2, 5, 6, 10]),
            Spike::spike_vec_for(1, vec![3, 7, 8, 10]),
            Spike::spike_vec_for(2, vec![4, 9, 12])
        ]
    );


    let output2_single_thread_solver = test_solver::Solver::new(spikes, nn).solve();

    // This is expected result from the passthrough NN.
    let expected_output = vec![
        vec![8],
        vec![6]
    ] ;
    
    // Check if the two outputs are the same 
    if output2_single_thread_solver != expected_output {
        panic!("Huh-oh, this should not happen. \n---------> DEMO 2: FAILED \n\n");
    }

    println!("---------> DEMO 2: COMPLETED \n\n");
    println!("-----------------------------------------------------------");

    /* ******** multi-thread solver DEMO ******** */

    // 3. Neural Network example using a multi thread solver 
    println!("DEMO 3: Neural Network using the multi thread solver ");
    let nn = builder
        .layer(
            
            [
                LifNeuron::new(&LifNeuronConfig::new(1.0, 0.5, 2.5, 0.9)),
                LifNeuron::new(&LifNeuronConfig::new(1.2, 0.6, 2.4, 1.2)),
            ],
            [
                1.3, 1.1
            ],
            [
                [0.0, -0.3],
                [-0.2, 0.0]
            ]
        )
        .layer(
            [
                LifNeuron::new(&LifNeuronConfig::new(1.0, 0.3, 2.5, 1.2)),
                LifNeuron::new(&LifNeuronConfig::new(1.1, 0.4, 2.6, 1.2)),
                LifNeuron::new(&LifNeuronConfig::new(1.2, 0.4, 3.0, 1.0))
            ],
            [
                [1.2, 1.3, 1.2],
                [1.4, 1.3, 1.5]
            ],
            [
                [0.0, -0.2, -0.3],
                [-0.3, 0.0, -0.3],
                [-0.2, -0.1, 0.0]
            ]
        )
        .build();

    // Define the input spikes

    // The function Spike::create_terminal_vec will produce the Vec that can be
    // then passed to NN::solve. It accepts a Vec<Vec<Spike>>, one inner Vec
    // for every neuron in the entry layer.
    // In our little example the first layer has just 2 neurons.
    let spikes = Spike::create_terminal_vec(
        vec![
            // Use the function Spike::spike_vec_for to create the inner Vecs
            Spike::spike_vec_for(0 /* 1st neuron */, vec![1, 3, 4, 7, 8]),
            Spike::spike_vec_for(1 /* 2nd neuron */, vec![1, 4, 5, 7, 9])
        ]
    );

    // Solve the network
    
    // This is how to solve the network normally.
    // Note: to show how to use the async feature, an intermediate function is needed here;
    //       usually this wouldn't be the case.
    let output = solve_nn(&nn, spikes.clone());

    // For the sake of this demo, let's also compare the output spikes
    // with the result produced by the internal single threaded solver.
    // Note: this is here just to showcase the test solver, but is not meant 
    //       to be a typical usage example.
    //       The test solver is - as the name implies - test-only.
    let output_test = test_solver::Solver::new(spikes, nn).solve();

    // Check if the two solvers produced the same result
    if output != output_test {
        panic!("Huh-oh, this should not happen. \n---------> DEMO 3: FAILED \n\n");
    }

    println!("---------> DEMO 3: COMPLETED \n\n");

    println!("************ END of DEMO ************");
}
