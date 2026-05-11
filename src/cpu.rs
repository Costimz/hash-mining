/*!
 * cpu.rs: Pure-Rust CPU mining backend. Uses keccak-asm for hashing.
 * Slower than CUDA but always available as a fallback or for self-test.
 *
 * Hot loop: pre-builds a 64-byte buffer with the challenge and prefix
 * fixed, then only updates bytes 56..63 (the counter) before each hash.
 */

use crate::types::{SharedCounter, SharedStop};
use alloy::primitives::U256;
use keccak_asm::Digest;
use std::sync::Arc;
use std::thread;

#[derive(Debug, Clone)]
pub struct CpuHit {
    pub counter: u64,
    pub hash_be: [u8; 32],
}

pub struct CpuWorker {
    challenge: [u8; 32],
    prefix: [u8; 24],
    difficulty: U256,
    base_counter: u64,
    end_counter: u64,
    stop: SharedStop,
    progress: SharedCounter,
}

impl CpuWorker {
    pub fn new(
        challenge: [u8; 32],
        prefix: [u8; 24],
        difficulty: U256,
        base_counter: u64,
        end_counter: u64,
        stop: SharedStop,
        progress: SharedCounter,
    ) -> Self {
        Self {
            challenge,
            prefix,
            difficulty,
            base_counter,
            end_counter,
            stop,
            progress,
        }
    }

    pub fn run(self) -> Option<CpuHit> {
        let mut buf = [0u8; 64];
        buf[..32].copy_from_slice(&self.challenge);
        buf[32..56].copy_from_slice(&self.prefix);

        let mut counter = self.base_counter;
        let mut bumped: u64 = 0;
        let mut hasher = keccak_asm::Keccak256::new();
        let diff_bytes = self.difficulty.to_be_bytes::<32>();
        while counter < self.end_counter {
            if (bumped & 0xFFFF) == 0 && self.stop.is_set() {
                self.progress.add(bumped);
                return None;
            }
            buf[56..64].copy_from_slice(&counter.to_be_bytes());
            hasher.update(&buf);
            let h = hasher.finalize_reset();
            let h: [u8; 32] = h.into();
            if be_less_than(&h, &diff_bytes) {
                self.progress.add(bumped + 1);
                return Some(CpuHit { counter, hash_be: h });
            }
            counter = counter.wrapping_add(1);
            bumped = bumped.wrapping_add(1);
        }
        self.progress.add(bumped);
        None
    }
}

#[inline(always)]
fn be_less_than(a: &[u8; 32], b: &[u8; 32]) -> bool {
    for i in 0..32 {
        if a[i] < b[i] { return true; }
        if a[i] > b[i] { return false; }
    }
    false
}

/* Spawn `n` workers, each over a disjoint counter range. Returns the
 * first hit found (other workers stop via the shared stop flag). */
pub fn run_pool(
    n_threads: usize,
    challenge: [u8; 32],
    prefix: [u8; 24],
    difficulty: U256,
    base: u64,
    batch: u64,
    stop: SharedStop,
    progress: SharedCounter,
) -> Option<CpuHit> {
    let per = batch / (n_threads as u64).max(1);
    let mut handles = Vec::with_capacity(n_threads);
    for i in 0..n_threads {
        let w = CpuWorker::new(
            challenge,
            prefix,
            difficulty,
            base + (i as u64) * per,
            base + ((i as u64) + 1) * per,
            Arc::clone(&stop),
            Arc::clone(&progress),
        );
        handles.push(thread::spawn(move || w.run()));
    }
    let mut hit: Option<CpuHit> = None;
    for h in handles {
        if let Ok(Some(r)) = h.join() {
            stop.set();
            if hit.is_none() { hit = Some(r); }
        }
    }
    hit
}

