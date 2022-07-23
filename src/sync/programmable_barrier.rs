use std::{sync::{Mutex, Condvar}, cell::UnsafeCell};

/// A variation of the `std::sync::Barrier` that allows exactly one thread to execute
/// some provided code before unlocking every other thread.
/// By definition, that code will be executed in mutual exclusion.
pub struct ProgrammableBarrier<T> {
    lock: Mutex<BarrierState>,
    cvar: Condvar,
    num_threads: usize,

    /// The result of the mutually exclusive function go here,
    /// and every thread then return a reference to this.
    wait_result: UnsafeCell<Option<T>>
}

/// What's inside the `Barrier`'s `Mutex`
struct BarrierState {
    count: usize,
    generation_id: usize
}

// Not sure Send is necessary
unsafe impl<T: Send + Sync> Sync for ProgrammableBarrier<T> { }

impl<T: Sync> ProgrammableBarrier<T> {
    /// Create a new `ProgrammableBarrier` that will wait for `n` threads
    pub fn new(n: usize) -> Self {
        Self {
            lock: Mutex::new(BarrierState { count: 0, generation_id: 0 }),
            cvar: Condvar::new(),
            num_threads: n,
            wait_result: UnsafeCell::new(None)
        }
    }

    /// Put the thread to sleep until the necessary amount of threads have called this, then release all of them.
    /// 
    /// Function `f` is executed in mutual exclusion by just one of the threads that call this,
    /// and a reference to its return value is then returned by `wait` to all calling threads.
    pub fn wait<F: FnOnce() -> Option<T>>(&self, f: F) -> Option<&T> {
        let mut lock = self.lock.lock().unwrap();
        let local_gen = lock.generation_id;
        lock.count += 1;

        if lock.count < self.num_threads {
            lock = self.cvar.wait_while(lock, |bs| local_gen == bs.generation_id).unwrap();
        } else {
            // This code here will execute in mutual exclusion by just one of the participating threads.
            // Before unlocking all the other threads, execute the given f
            unsafe {
                *self.wait_result.get() = f();
            }

            lock.count = 0;
            lock.generation_id = lock.generation_id.wrapping_add(1);
            self.cvar.notify_all();
        }

        unsafe { (&*self.wait_result.get()).as_ref() }
    }
}
