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
    // Create a neural network
    
    // Create a new static builder.
    // This can be used to construct "static" nns,
    // i.e. nn whose size is known at compile time.
    let builder = NNBuilder::<LeakyIntegrateFire, _>::new();

    // The builder can be used by adding layers one by one,
    // via its 'layer' method, which consumes the builder and
    // returns a new instance.
    // The easiest way to use the builder is by chaining 'layer' calls,
    // like this
    let nn = builder
        .layer(
            // Every layer is defined by its neurons (with the given order!),
            // the input-weights (synapses from the previous layer, or network
            // inputs), and the intra-weights (mesh of synapses that connect
            // different neurons of the same layer)
            [
                // Neurons require different parameters depending on the model
                // being used
                LifNeuron::new(&LifNeuronConfig::new(1.0, 0.5, 2.5, 0.9)),
                LifNeuron::new(&LifNeuronConfig::new(1.2, 0.6, 2.4, 1.2)),
            ],
            [
                1.3, 1.1
            ],
            [
                // Intra-weights are always square matrices.
                // The diagonal is however usually null, but it is not enforced
                // in this library, just be warned that "bad" networks can be
                // created, and attempting to solve them might result in
                // an infinite number of output spikes being generated
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
        panic!("Huh-oh, this should not happen");
    }
}
