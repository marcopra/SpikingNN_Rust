use std::{ops::Range, num::NonZeroUsize};
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;
use crate::{nn::{Spike, solver_v1::Solver}, NNBuilder, LeakyIntegrateFire, LifNeuronConfig, NN, LifNeuron};

fn random_lif_neuron<Rng: RngCore>(rng: &mut Rng) -> LifNeuron {
    LifNeuron::from(&LifNeuronConfig::new(
        rng.gen_range(0.8..2.5),
        rng.gen_range(0.2..1.5),
        rng.gen_range(1.5..3.5),
        rng.gen_range(0.1..5.0)
    ))
}

fn create_random_lif_nn(seed: u64, num_layers: NonZeroUsize, layer_size_range: Range<NonZeroUsize>, num_spikes: usize) -> (NN<LeakyIntegrateFire>, Vec<Spike>) {
    let mut rng = Pcg64Mcg::seed_from_u64(seed);
    let mut builder = NNBuilder::new_dynamic();
    let num_layers = num_layers.get();
    let layer_size_range = layer_size_range.start.get()..layer_size_range.end.get();
    let mut last_layer_size = 1;

    for _ in 0..num_layers {
        let layer_size = rng.gen_range(layer_size_range.clone());
        
        builder = builder.layer(
            (0..layer_size).into_iter().map(|_| random_lif_neuron(&mut rng)).collect::<Vec<_>>(),
            (0..last_layer_size*layer_size).into_iter().map(|_| rng.gen_range(0.5..2.5)).collect::<Vec<_>>(),
            (0..layer_size*layer_size).into_iter().map(|_| rng.gen_range(-1.0..-0.05)).collect::<Vec<_>>()
        ).unwrap();

        last_layer_size = layer_size;
    }

    let nn = builder.build().unwrap();

    let mut spikes = vec![vec![]; nn.layers[0].neurons.len()];

    rand::seq::index::sample(&mut rng, num_spikes * 5, num_spikes)
        .into_iter()
        .map(|u| u as u128)
        .for_each(|ts| {
            spikes[rng.gen_range(0..nn.layers[0].neurons.len())].push(ts); // TODO: add support for simultaneous spikes to different entry neurons
        });

    let spikes = spikes.into_iter().enumerate().map(|(i, s)| Spike::spike_vec_for(i, s)).collect();
    
    (nn, Spike::create_terminal_vec(spikes))
}

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

#[test]
fn test_create_terminal_vec(){
    let spikes_neuron_1 = [11, 9, 23, 43, 42].to_vec();
    let spike_vec_for_neuron_1 = Spike::spike_vec_for(1, spikes_neuron_1 );
    
    let spikes_neuron_2 = [1, 29, 3, 11, 22].to_vec();
    let spike_vec_for_neuron_2 = Spike::spike_vec_for(2, spikes_neuron_2 );
    
    let spikes: Vec<Vec<Spike>> = [spike_vec_for_neuron_1, 
                                        spike_vec_for_neuron_2].to_vec();
    
    let sorted_spike_array_for_nn: Vec<Spike> = Spike::create_terminal_vec(spikes);
    println!("{:?}", sorted_spike_array_for_nn);
}

#[test]
fn test_solve_nn() {
    // Create a stupidly simple NN
    let cfg = LifNeuronConfig::new(1.0, 0.5, 2.0, 1.0);
    let nn = NNBuilder::<LeakyIntegrateFire, _>::new()
        .layer(
            [From::from(&cfg), From::from(&cfg)],
            [1.2, 2.3],
            [[0.0, -0.8], [-0.6, 0.0]]
        )
        .layer(
            [From::from(&cfg), From::from(&cfg), From::from(&cfg)],
            [
                [1.5, 1.2, 1.6],
                [1.2, 1.4, 1.4]
            ],
            [
                [0.0, -0.4, -0.3],
                [-0.5, 0.0, -0.5],
                [-0.8, -0.4, 0.0]
            ]
        )
        .build();

    // Create some input spikes
    let spikes = Spike::create_terminal_vec(vec![
        Spike::spike_vec_for(0, vec![0, 1, 4, 6, 8, 10, 14]),
        Spike::spike_vec_for(1, vec![2, 3, 5, 7, 11, 20]) // No simultaneous spikes
    ]);

    let output = nn.solve(spikes.clone());
    println!("\n\nOUTPUT MULTI THREAD: {:?}", output); // [[0, 2, 5, 7, 10, 14, 20], [0, 2, 5, 7, 10, 14, 20], [0, 1, 2, 5, 6, 7, 10, 14, 20]]

    println!("Then ------------------------------------");
    let mut single_solver = Solver::new(spikes, nn);
    let second_output = single_solver.solve();

    println!("\n\nOUTPUT SINGLE THREAD: {:?}", second_output);
}

#[test]
fn test_solve_nn2() {
    let nn = NNBuilder::<LeakyIntegrateFire, _>::new()
        .layer(
            [
                From::from(&LifNeuronConfig::new(2., 0.5, 3., 1.6)),
                From::from(&LifNeuronConfig::new(1.5, 0.4, 3.1, 1.2)),
                From::from(&LifNeuronConfig::new(1.6, 0.3, 2.8, 1.1))
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
                From::from(&LifNeuronConfig::new(1.8, 0.5, 2.8, 1.5)),
                From::from(&LifNeuronConfig::new(1.7, 0.8, 2.6, 1.6))
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
    
    let spikes = Spike::create_terminal_vec(vec![
        Spike::spike_vec_for(0, vec![2, 5, 6, 10]),
        Spike::spike_vec_for(1, vec![3, 7, 8, 10]),
        Spike::spike_vec_for(2, vec![4, 9, 12])
    ]);

    let output = nn.solve(spikes.clone());
    println!("OUTPUT MULTI THREAD: {:?}", output);

    let mut single_solver = Solver::new(spikes, nn);
    let output2 = single_solver.solve();
    println!("OUTPUT SINGLE THREAD: {:?}", output2);
}

#[test]
fn test_random_nn() {
    let (nn, spikes) = create_random_lif_nn(
        53278,
        10.try_into().unwrap(),
        3.try_into().unwrap()..10.try_into().unwrap(),
        10
    );
    println!("{:?}", nn.solve(spikes));
}

#[cfg(feature = "bench")]
mod benches {
    extern crate test;
    use test::{Bencher, black_box};

    use super::{create_random_lif_nn, super::solver_v1::Solver};

    #[bench]
    fn bench_tiny_single(b: &mut Bencher) {
        let (nn, spikes) = create_random_lif_nn(
            8436798,
            3.try_into().unwrap(),
            1.try_into().unwrap()..3.try_into().unwrap(),
            5
        );
        let mut solver = Solver::new(spikes, nn);
    
        b.iter(|| black_box(solver.solve()));
    }

    #[bench]
    fn bench_tiny_multi(b: &mut Bencher) {
        let (nn, spikes) = create_random_lif_nn(
            8436798,
            3.try_into().unwrap(),
            1.try_into().unwrap()..3.try_into().unwrap(),
            5
        );

        b.iter(|| black_box(nn.solve(spikes.clone())));
    }

    #[bench]
    fn bench_small_single(b: &mut Bencher) {
        let (nn, spikes) = create_random_lif_nn(
            498247,
            15.try_into().unwrap(),
            4.try_into().unwrap()..12.try_into().unwrap(),
            25
        );
        let mut solver = Solver::new(spikes, nn);
    
        b.iter(|| black_box(solver.solve()));
    }

    #[bench]
    fn bench_small_multi(b: &mut Bencher) {
        let (nn, spikes) = create_random_lif_nn(
            498247,
            15.try_into().unwrap(),
            4.try_into().unwrap()..12.try_into().unwrap(),
            25
        );

        b.iter(|| black_box(nn.solve(spikes.clone())));
    }

    #[bench]
    fn bench_medium_single(b: &mut Bencher) {
        let (nn, spikes) = create_random_lif_nn(
            543513,
            50.try_into().unwrap(),
            10.try_into().unwrap()..20.try_into().unwrap(),
            75
        );
        let mut solver = Solver::new(spikes, nn);
    
        b.iter(|| black_box(solver.solve()));
    }

    #[bench]
    fn bench_medium_multi(b: &mut Bencher) {
        let (nn, spikes) = create_random_lif_nn(
            543513,
            50.try_into().unwrap(),
            10.try_into().unwrap()..20.try_into().unwrap(),
            75
        );

        b.iter(|| black_box(nn.solve(spikes.clone())));
    }

    #[bench]
    fn bench_big_single(b: &mut Bencher) {
        let (nn, spikes) = create_random_lif_nn(
            136415635468,
            200.try_into().unwrap(),
            20.try_into().unwrap()..35.try_into().unwrap(),
            350
        );
        let mut solver = Solver::new(spikes, nn);
    
        b.iter(|| black_box(solver.solve()));
    }

    #[bench]
    fn bench_big_multi(b: &mut Bencher) {
        let (nn, spikes) = create_random_lif_nn(
            136415635468,
            200.try_into().unwrap(),
            20.try_into().unwrap()..35.try_into().unwrap(),
            350
        );

        b.iter(|| black_box(nn.solve(spikes.clone())));
    }

    #[bench]
    fn bench_huge_single(b: &mut Bencher) {
        let (nn, spikes) = create_random_lif_nn(
            3546846,
            1500.try_into().unwrap(),
            50.try_into().unwrap()..80.try_into().unwrap(),
            500
        );
        let mut solver = Solver::new(spikes, nn);
    
        b.iter(|| black_box(solver.solve()));
    }

    #[bench]
    fn bench_huge_multi(b: &mut Bencher) {
        let (nn, spikes) = create_random_lif_nn(
            3546846,
            1500.try_into().unwrap(),
            50.try_into().unwrap()..80.try_into().unwrap(),
            500
        );

        b.iter(|| black_box(nn.solve(spikes.clone())));
    }
}
