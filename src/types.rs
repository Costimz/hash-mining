/*!
 * types.rs: shared value types used across modules.
 */

use alloy::primitives::{Address, B256, U256};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct Job {
    pub challenge: [u8; 32],
    pub difficulty: U256,
    pub epoch: u64,
    pub miner: Address,
}

impl Job {
    pub fn challenge_be(&self) -> B256 { B256::from(self.challenge) }
    pub fn difficulty_be(&self) -> [u8; 32] { self.difficulty.to_be_bytes() }
}

#[derive(Clone, Debug)]
pub struct Hit {
    pub nonce: [u8; 32],
    pub hash: [u8; 32],
    pub epoch: u64,
    pub miner: Address,
}

#[derive(Default)]
pub struct StopFlag(AtomicBool);
impl StopFlag {
    pub fn set(&self) { self.0.store(true, Ordering::SeqCst); }
    pub fn clear(&self) { self.0.store(false, Ordering::SeqCst); }
    pub fn is_set(&self) -> bool { self.0.load(Ordering::SeqCst) }
}

#[derive(Default)]
pub struct Counter(AtomicU64);
impl Counter {
    pub fn add(&self, n: u64) -> u64 { self.0.fetch_add(n, Ordering::Relaxed) + n }
    pub fn get(&self) -> u64 { self.0.load(Ordering::Relaxed) }
}

pub type SharedCounter = Arc<Counter>;
pub type SharedStop = Arc<StopFlag>;
