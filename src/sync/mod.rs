//! Helper types for parallelization

pub(crate) mod programmable_barrier;

use std::{sync::{mpsc::{Receiver, Sender}, atomic::{AtomicUsize, Ordering, AtomicBool}}, cell::UnsafeCell};
use ndarray::Array2;
use self::programmable_barrier::ProgrammableBarrier;

/// Utility type that, by being opaque to the user, allows the implementation to
/// assert the authenticity of a neuron, thus making the method `LayerManager::next` safe.
pub struct NeuronToken {
    /// Id of the neuron in the layer (0..neuron_count)
    neuron_id: usize,
    /// Id of the `LayerManager` instance. Problems may occur when this overflows (it can happen because of
    /// the static `AtomicUsize` used to create it).
    layer_manager_id: usize
}

pub(crate) struct LayerManager<'a> {
    /// Receive spike arrays from previous layer
    input: Receiver<(u128, Array2<f64>)>,
    /// Send spike arrays to next layer
    output: Sender<(u128, Array2<f64>)>,
    /// Input synapse mesh, borrowed from the `NN`
    synapses_input: &'a Array2<f64>,
    /// Intra synapse mesh, borrowed from the `NN`
    synapses_intra: &'a Array2<f64>,
    /// Used to synchronize all the neurons of this layer to each other
    barrier: ProgrammableBarrier<Array2<f64>>,
    /// Neurons write into this array during computation. Because every neuron will access a mutually exclusive element of
    /// the array, the UnsafeCell is fine
    cur: UnsafeCell<(u128, Array2<f64>)>,
    /// Id of this LayerManager. Having an id embedded into the corresponding NeuronToken allows us
    /// to always be sure of a neuron's ability to use this manager.
    id: usize,
    /// Did any neuron spike during this time slot?
    /// This is an atomic because multiple threads can access the same bool simultaneously
    spiked: AtomicBool,
}

/// `LayerManager` is Sync because the receivers are accessed in mutual exclusion via the `ProgrammableBarrier`
unsafe impl<'a> Sync for LayerManager<'a> { }

impl<'a> LayerManager<'a> {
    /// Create a new `LayerManager` instance.
    /// The returned `Vec<NeuronToken>` should be given to the individual threads in order,
    /// meaning that i-th `NeuronToken` must correspond with i-th neuron in the `NN`'s layer.
    pub fn new(
        num_neurons: usize,
        input: Receiver<(u128, Array2<f64>)>,
        output: Sender<(u128, Array2<f64>)>,
        synapses_input: &'a Array2<f64>,
        synapses_intra: &'a Array2<f64>
    ) -> (Self, Vec<NeuronToken>)
    {
        static ATOMIC_STATIC: AtomicUsize = AtomicUsize::new(0);
        let layer_manager_id = ATOMIC_STATIC.fetch_add(1, Ordering::Relaxed);
        
        (
            Self {
                input,
                output,
                synapses_input,
                synapses_intra,
                barrier: ProgrammableBarrier::new(num_neurons),
                cur: UnsafeCell::new((Default::default(), Array2::zeros((1, num_neurons)))),
                id: layer_manager_id,
                spiked: false.into()
            },
            (0..num_neurons).into_iter().map(|neuron_id| NeuronToken { neuron_id, layer_manager_id }).collect()
        )
    }
    
    /// Wait for the next spike.
    /// 
    /// If the return value is a `None`, it means that no spikes will ever be received
    /// on this interface, and the neuron must finish, otherwise the contained value
    /// is the tuple with timestamp of the spike and weighted input for the specific neuron.
    pub fn next(&self, token: &NeuronToken) -> Option<(u128, f64)> {
        // Token valid?
        assert_eq!(token.layer_manager_id, self.id, "Used invalid token for LayerManager");
        
        self.barrier.wait(|| {
            // Check if any neuron spiked during this time slot
            if self.spiked.fetch_and(false, Ordering::SeqCst) {
                // Need to clone to be able to send to the next layer, ughh
                let spike_array = unsafe {
                    &*self.cur.get()
                };

                // Send to next layer
                self.output.send(spike_array.clone()).unwrap();

                // Handle intra spikes (note that the timestamp remains the same)
                Some(spike_array.1.dot(self.synapses_intra))
            } else {
                // No neuron spiked -> no need to send anything to the next layer
                // Get next spike from input receiver
                self.input.recv().ok().map(|(ts, arr)| {
                    // Set new timestamp
                    unsafe {
                        (*self.cur.get()).0 = ts;
                    }

                    arr.dot(self.synapses_input)
                })
            }
        }).map(|a| (unsafe { (*self.cur.get()).0 }, a[(0, token.neuron_id)]))
    }

    /// Commit the result of the previous spike by the 
    pub fn commit(&self, token: &NeuronToken, spiked: bool, val: f64) {
        // Token valid?
        assert_eq!(token.layer_manager_id, self.id, "Used invalid token for LayerManager");

        // Update spiked flag
        self.spiked.fetch_or(spiked, Ordering::SeqCst); // TODO: Can we relax this ordering? (probably)

        // Set val in shared array
        unsafe {
            (*self.cur.get()).1[(0, token.neuron_id)] = val;
        };
    }
}
