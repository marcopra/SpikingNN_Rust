# PdS-SpikingNN

This Rust library can create and resolve spiking neural networks defined for any possible applicable model, thanks to the powerful extensibility achieved through Rust's type system: simply implement the `Model` trait for your personally defined custom model and be good to go!

By default, the **_Leaky Integrate and Fire_** model is provided in the `lif` submodule.

## Getting started

Create a new neural network via the `NNBuilder`; this type can either perform static, compile-time checks on the size of the different layers of the network or, in case the nn's size can not be known at compile time, a _dynamic_ variant of the builder (which does size checks at runtime and can therefore return errors) can be used.

### Create a neural network statically

```rust
use pds_spiking_nn::{NNBuilder, lif::*};

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
```

The same neural network can be created dynamically with `NNBuilder::new_dynamic`.

### Define the input spikes that will stimulate the network

The `NN::solve` method requires spikes to be passed as a single `Vec` of `Spike` instances. Spikes can be either created manually, or through the provided helper functions. The latter method is shown in the next example.

```rust
use pds_spiking_nn::Spike;

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
```

### Solve the network

Finally, call the `NN::solve` method with the spikes `Vec` to "solve" the network and get as output the timestamps of every generated spike in the output layer, for every neuron.

```rust
let output = nn.solve(spikes);
```

## Features

This crate provides the following cargo features, which can be enabled at will:

 - **async** - `NN::solve` becomes an async function, which can be run with your favorite runtime. Internally, the implementation uses [tokio](https://crates.io/crates/tokio), and will spawn tokio `task`s in place of threads. The rationale for this is that, on larger networks, the parallelization strategy of firing a kernel thread for every layer will quickly result in hundreds if not thousands of threads, thus producing massive overhead due to the context switch between all of them. By employing _green threads_ (in the form of tasks), the user can effectively spread their allocation on a more reasonable number of kernel threads, hence dramatically improving the performance.  _If you enable this feature, remember to `.await` the `Future` returned by `NN::solve`!_
  - **simd** - enable explicit SIMD support for the solver through [packed_simd](https://github.com/rust-lang/packed_simd) (**_this requires the latest nightly compiler_**). If this feature flag is enabled, the `Model` trait will require the "x4" version of the `Neuron` and `SolverVars` types, together with their respective `handle_spike` function. The default implementation of the _lif_ model will exploit 256 bit wide vectorization extensions, like `AVX` on x86 platforms. _To obtain the most out of this feature, remember to enable the necessary extensions for rustc through, for example, the "-C target-features" compiler flag._

Neither of these features are enabled by default, but their usage is strongly recommended when possible due to the performance improvement they can provide. See the [Performance](#performance) section for details.

## Performance

Performance has been one of the most important metrics when developing this crate. Below are the results of running the internal benchmarks on a **Ryzen 5 5600X** system on **Ubuntu 22.04 LTS** with the **rustc 1.64.0-nightly (1b57946a4 2022-08-03)** toolchain (latest at the time of writing) and using the **"-C target-cpu=native"** flag. These numbers are pretty meaningless on their own, but can be used to compare the impact of the different features over the vanilla implementation.

| nn size* | no features | `async`** | `simd` | `async`**,`simd` |
| ------- | ----------- | ----- | ---- | ---------- |
| **tiny** | 24.357 us | 11.120 us | 24.746 us | 10.205 us |
| **small** | 128.526 us | 381.797 us | 130.221 us | 338.130 us |
| **medium** | 667.860 us | 1.680322 ms | 657.395 us | 1.596312 ms |
| **big** | 10.120520 ms | 8.361438 ms | 9.322529 ms | 7.537765 ms |
| **huge** | 308.548927 ms | 239.486940 ms | 285.931535 ms | 218.387319 ms |

*_You can check the definition of these neural networks (number of layers, number of neurons for each layer, number of spikes used to stimulate them) in the `nn::tests::benches` module._

**_The solver was run with the default multi-threaded `tokio` runtime, which spawns a number of kernel threads equal to the number of available logical threads, which is 12 for this CPU._

As you can see, for very small networks, the barebones multi-threaded implementation usually edges out the alternatives, but for moderately big to "huge" neural networks, **`async` provides a consistent ~20% improvement, and `simd` another ~9-10% on top of it**.
