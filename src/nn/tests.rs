use std::{ops::Range, num::NonZeroUsize};
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;
use crate::{nn::{Spike, solver_v1::Solver}, NNBuilder, LeakyIntegrateFire, LifNeuronConfig, NN, LifNeuron};

fn random_lif_neuron<Rng: RngCore>(rng: &mut Rng) -> LifNeuron {
    let v_rest = rng.gen_range(0.8..2.5);
    
    LifNeuron::from(&LifNeuronConfig::new(
        v_rest,
        rng.gen_range(v_rest*0.15..v_rest*0.5),
        rng.gen_range(v_rest*1.5..v_rest*3.),
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
            (0..layer_size*layer_size).into_iter().enumerate().map(|(i, _)| if i % (layer_size + 1) == 0 { 0.0 } else { rng.gen_range(-1.0..-0.05) }).collect::<Vec<_>>()
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

#[cfg(not(feature = "async"))]
#[test]
fn test_tiny_sync() {
    let (nn, spikes) = create_random_lif_nn(
        8436798,
        3.try_into().unwrap(),
        1.try_into().unwrap()..3.try_into().unwrap(),
        5
    );

    let mut solver = Solver::new(spikes.clone(), nn.clone());
    assert_eq!(solver.solve(), nn.solve(spikes));
}

#[cfg(feature = "async")]
#[tokio::test]
async fn test_tiny_async() {
    let (nn, spikes) = create_random_lif_nn(
        8436798,
        3.try_into().unwrap(),
        1.try_into().unwrap()..3.try_into().unwrap(),
        5
    );

    let mut solver = Solver::new(spikes.clone(), nn.clone());
    assert_eq!(solver.solve(), nn.solve(spikes).await);
}

#[cfg(not(feature = "async"))]
#[test]
fn test_small_sync() {
    let (nn, spikes) = create_random_lif_nn(
        498247,
        15.try_into().unwrap(),
        4.try_into().unwrap()..12.try_into().unwrap(),
        25
    );

    let mut solver = Solver::new(spikes.clone(), nn.clone());
    assert_eq!(solver.solve(), nn.solve(spikes));
}

#[cfg(feature = "async")]
#[tokio::test]
async fn test_small_async() {
    let (nn, spikes) = create_random_lif_nn(
        498247,
        15.try_into().unwrap(),
        4.try_into().unwrap()..12.try_into().unwrap(),
        25
    );

    let mut solver = Solver::new(spikes.clone(), nn.clone());
    assert_eq!(solver.solve(), nn.solve(spikes).await);
}

#[cfg(not(feature = "async"))]
#[test]
fn test_medium_sync() {
    let (nn, spikes) = create_random_lif_nn(
        543513,
        50.try_into().unwrap(),
        10.try_into().unwrap()..20.try_into().unwrap(),
        75
    );

    let mut solver = Solver::new(spikes.clone(), nn.clone());
    assert_eq!(solver.solve(), nn.solve(spikes));
}

#[cfg(feature = "async")]
#[tokio::test]
async fn test_medium_async() {
    let (nn, spikes) = create_random_lif_nn(
        543513,
        50.try_into().unwrap(),
        10.try_into().unwrap()..20.try_into().unwrap(),
        75
    );

    let mut solver = Solver::new(spikes.clone(), nn.clone());
    assert_eq!(solver.solve(), nn.solve(spikes).await);
}

#[cfg(not(feature = "async"))]
#[test]
fn test_big_sync() {
    let (nn, spikes) = create_random_lif_nn(
        136415635468,
        200.try_into().unwrap(),
        20.try_into().unwrap()..35.try_into().unwrap(),
        350
    );

    let mut solver = Solver::new(spikes.clone(), nn.clone());
    assert_eq!(solver.solve(), nn.solve(spikes));
}

#[cfg(feature = "async")]
#[tokio::test]
async fn test_big_async() {
    let (nn, spikes) = create_random_lif_nn(
        136415635468,
        200.try_into().unwrap(),
        20.try_into().unwrap()..35.try_into().unwrap(),
        350
    );

    let mut solver = Solver::new(spikes.clone(), nn.clone());
    assert_eq!(solver.solve(), nn.solve(spikes).await);
}

#[cfg(not(feature = "async"))]
#[test]
fn test_huge_sync() {
    let (nn, spikes) = create_random_lif_nn(
        3546846,
        1500.try_into().unwrap(),
        50.try_into().unwrap()..80.try_into().unwrap(),
        500
    );

    let mut solver = Solver::new(spikes.clone(), nn.clone());
    assert_eq!(solver.solve(), nn.solve(spikes));
}

#[cfg(feature = "async")]
#[tokio::test]
async fn test_huge_async() {
    let (nn, spikes) = create_random_lif_nn(
        3546846,
        1500.try_into().unwrap(),
        50.try_into().unwrap()..80.try_into().unwrap(),
        500
    );

    let mut solver = Solver::new(spikes.clone(), nn.clone());
    assert_eq!(solver.solve(), nn.solve(spikes).await);
}

#[cfg(feature = "bench")]
mod benches {
    extern crate test;
    use test::{Bencher, black_box};
    use tokio::runtime::Builder;

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

    #[cfg(not(feature = "async"))]
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

    #[cfg(feature = "async")]
    #[bench]
    fn bench_tiny_async(b: &mut Bencher) {
        let runtime = Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        
        let (nn, spikes) = create_random_lif_nn(
            8436798,
            3.try_into().unwrap(),
            1.try_into().unwrap()..3.try_into().unwrap(),
            5
        );

        b.iter(|| runtime.block_on(black_box(nn.solve(spikes.clone()))));
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

    #[cfg(not(feature = "async"))]
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

    #[cfg(feature = "async")]
    #[bench]
    fn bench_small_async(b: &mut Bencher) {
        let runtime = Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        
        let (nn, spikes) = create_random_lif_nn(
            498247,
            15.try_into().unwrap(),
            4.try_into().unwrap()..12.try_into().unwrap(),
            25
        );

        b.iter(|| runtime.block_on(black_box(nn.solve(spikes.clone()))));
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

    #[cfg(not(feature = "async"))]
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

    #[cfg(feature = "async")]
    #[bench]
    fn bench_medium_async(b: &mut Bencher) {
        let runtime = Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        
        let (nn, spikes) = create_random_lif_nn(
            543513,
            50.try_into().unwrap(),
            10.try_into().unwrap()..20.try_into().unwrap(),
            75
        );

        b.iter(|| runtime.block_on(black_box(nn.solve(spikes.clone()))));
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

    #[cfg(not(feature = "async"))]
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

    #[cfg(feature = "async")]
    #[bench]
    fn bench_big_async(b: &mut Bencher) {
        let runtime = Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        
        let (nn, spikes) = create_random_lif_nn(
            136415635468,
            200.try_into().unwrap(),
            20.try_into().unwrap()..35.try_into().unwrap(),
            350
        );

        b.iter(|| runtime.block_on(black_box(nn.solve(spikes.clone()))));
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

    #[cfg(not(feature = "async"))]
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

    #[cfg(feature = "async")]
    #[bench]
    fn bench_huge_async(b: &mut Bencher) {
        let runtime = Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        
        let (nn, spikes) = create_random_lif_nn(
            3546846,
            1500.try_into().unwrap(),
            50.try_into().unwrap()..80.try_into().unwrap(),
            500
        );

        b.iter(|| runtime.block_on(black_box(nn.solve(spikes.clone()))));
    }
}
