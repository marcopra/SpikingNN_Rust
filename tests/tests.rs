use pds_spiking_nn::{NNBuilder, Spike, lif::*};

#[test]
fn test_build_empty_nn() {
    let nn = NNBuilder::<LeakyIntegrateFire, _>::new_dynamic().build();
    assert!(nn.is_err());
}

#[cfg(not(feature = "async"))]
#[test]
fn test_passthrough_nn() {
    let config = LifNeuronConfig::new(2.0, 0.5, 2.1, 1.0);
    
    let nn = NNBuilder::<LeakyIntegrateFire, _>::new()
        .layer(
            [
                From::from(&config),
                From::from(&config),
                From::from(&config)
            ],
            [
                1.0, 1.0, 1.0
            ],
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0]
            ]
        )
        .build();
    
    let spikes = Spike::create_terminal_vec(
        vec![
            Spike::spike_vec_for(0, vec![1, 2, 3, 5, 6, 7]),
            Spike::spike_vec_for(1, vec![2, 6, 7, 9]),
            Spike::spike_vec_for(2, vec![2, 5, 6, 10, 11])
        ]
    );

    assert_eq!(
        nn.solve(spikes),
        vec![
            vec![1, 2, 3, 5, 6, 7],
            vec![2, 6, 7, 9],
            vec![2, 5, 6, 10, 11]
        ]
    );
}

#[cfg(feature = "async")]
#[tokio::test]
async fn test_passthrough_nn() {
    let config = LifNeuronConfig::new(2.0, 0.5, 2.1, 1.0);
    
    let nn = NNBuilder::<LeakyIntegrateFire, _>::new()
        .layer(
            [
                From::from(&config),
                From::from(&config),
                From::from(&config)
            ],
            [
                1.0, 1.0, 1.0
            ],
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0]
            ]
        )
        .build();
    
    let spikes = Spike::create_terminal_vec(
        vec![
            Spike::spike_vec_for(0, vec![1, 2, 3, 5, 6, 7]),
            Spike::spike_vec_for(1, vec![2, 6, 7, 9]),
            Spike::spike_vec_for(2, vec![2, 5, 6, 10, 11])
        ]
    );

    assert_eq!(
        nn.solve(spikes).await,
        vec![
            vec![1, 2, 3, 5, 6, 7],
            vec![2, 6, 7, 9],
            vec![2, 5, 6, 10, 11]
        ]
    );
}

#[cfg(not(feature = "async"))]
#[test]
fn test_hand_solved() {
    let nn = NNBuilder::<LeakyIntegrateFire, _>::new()
        .layer(
            [
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

    assert_eq!(
        nn.solve(spikes),
        vec![
            vec![8],
            vec![6]
        ]
    );
}

#[cfg(feature = "async")]
#[tokio::test]
async fn test_hand_solved() {
    let nn = NNBuilder::<LeakyIntegrateFire, _>::new()
        .layer(
            [
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

    assert_eq!(
        nn.solve(spikes).await,
        vec![
            vec![8],
            vec![6]
        ]
    );
}

#[test]
fn test_spike_vec_for() {
    assert_eq!(
        Spike::spike_vec_for(4, vec![4, 7, 3, 10, 11, 2]),
        {
            let mut v = vec![4, 7, 3, 10, 11, 2]
                .into_iter()
                .map(|ts| Spike {neuron_id: 4, ts})
                .collect::<Vec<_>>();
            
            v.sort();
            v
        }
    );
}

#[test]
fn test_spike_vec_for_empty() {
    assert_eq!(
        Spike::spike_vec_for(1, vec![]),
        vec![]
    );
}

#[test]
fn test_spike_vec_for_repeating() {
    assert_eq!(
        Spike::spike_vec_for(7, vec![1, 1, 1, 5, 1]),
        vec![1, 1, 1, 1, 5].into_iter().map(|ts| Spike {neuron_id: 7, ts}).collect::<Vec<_>>()
    );
}

#[test]
fn test_create_terminal_vec(){
    let spikes_neuron_1 = [11, 9, 23, 43, 42].to_vec();
    let spike_vec_for_neuron_1 = Spike::spike_vec_for(1, spikes_neuron_1 );
    
    let spikes_neuron_2 = [1, 29, 3, 11, 22].to_vec();
    let spike_vec_for_neuron_2 = Spike::spike_vec_for(2, spikes_neuron_2 );
    
    let spikes: Vec<Vec<Spike>> = [spike_vec_for_neuron_1, spike_vec_for_neuron_2].to_vec();
    
    let sorted_spike_array_for_nn: Vec<Spike> = Spike::create_terminal_vec(spikes);
    
    assert_eq!(
        sorted_spike_array_for_nn,
        {
            let mut v = [11, 9, 23, 43, 42].into_iter()
                .map(|ts| Spike {neuron_id: 1, ts})
                .chain([1, 29, 3, 11, 22].into_iter().map(|ts| Spike {neuron_id: 2, ts}))
                .collect::<Vec<_>>();
            
            v.sort();
            v
        }
    );
}

#[test]
fn test_nn_get_params() {
    let nn = NNBuilder::<LeakyIntegrateFire, _>::new()
        .layer(
            [
                LifNeuron::new(&LifNeuronConfig::new(1.0, 0.5, 2.8, 0.9)),
                LifNeuron::new(&LifNeuronConfig::new(1.2, 0.6, 2.9, 1.2)),
            ],
            [
                1.2, 1.1
            ],
            [
                [0.0, -0.3],
                [-0.2, 0.0]
            ]
        )
        .layer(
            [
                LifNeuron::new(&LifNeuronConfig::new(0.8, 0.3, 2.5, 1.2)),
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
    
    assert_eq!(nn.get_input_weight(0), Some(1.2));
    assert_eq!(nn.get_input_weight(1), Some(1.1));
    assert_eq!(nn.get_input_weight(2), None);

    assert_eq!(nn[0][(0, 0)], 0.0);
    assert_eq!(nn[0][(0, 1)], -0.3);
    assert_eq!(nn[0][(1, 0)], -0.2);
    assert_eq!(nn[0][(1, 1)], 0.0);

    assert_eq!(nn[((0, 0), (1, 0))], 1.2);
    assert_eq!(nn.get_weight((0, 1), (1, 1)), Some(1.3));
    assert_eq!(nn.get_weight((1, 0), (0, 0)), None);
}

#[test]
fn test_nn_update_params() {
    let mut nn = NNBuilder::<LeakyIntegrateFire, _>::new()
        .layer(
            [
                LifNeuron::new(&LifNeuronConfig::new(1.0, 0.5, 2.8, 0.9)),
                LifNeuron::new(&LifNeuronConfig::new(1.2, 0.6, 2.9, 1.2)),
            ],
            [
                1.2, 1.1
            ],
            [
                [0.0, -0.3],
                [-0.2, 0.0]
            ]
        )
        .layer(
            [
                LifNeuron::new(&LifNeuronConfig::new(0.8, 0.3, 2.5, 1.2)),
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
    
    nn[0][0].v_rest = 2.0;
    assert_eq!(nn.get_neuron(0, 0).unwrap().v_rest, 2.0);

    *nn.get_input_weight_mut(1).unwrap() = 1.5;
    assert_eq!(nn.get_input_weight(1), Some(1.5));

    nn[0][(0, 0)] = 1.0;
    assert_eq!(nn.get_layer(0).unwrap().get_intra_weight(0, 0), Some(1.0));

    nn[((0, 1), (1, 1))] = 3.0;
    assert_eq!(nn.get_weight((0, 1), (1, 1)), Some(3.0));

    for neuron in nn.iter_mut().flat_map(|l| l.iter_mut_neurons()) {
        neuron.v_reset += 1.0;
    }

    assert_eq!(nn[0][1].v_reset, 1.6);
    assert_eq!(nn[1][2].v_reset, 1.4);
}

#[cfg(feature = "expose-test-solver")]
#[test]
fn test_solver_v1() {
    use pds_spiking_nn::test_solver;
    
    let nn = NNBuilder::<LeakyIntegrateFire, _>::new()
        .layer(
            [
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

    let mut test_solver = test_solver::Solver::new(spikes, nn);

    assert_eq!(
        test_solver.solve(),
        vec![
            vec![8],
            vec![6]
        ]
    );
}
