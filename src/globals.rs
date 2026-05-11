/*!
 * globals.rs: shared compile-time constants and process-wide defaults.
 */

use parking_lot::RwLock;
use once_cell::sync::Lazy;

static SUBMIT_RPCS: Lazy<RwLock<Vec<String>>> = Lazy::new(|| {
    RwLock::new(vec![
        "https://rpc.mevblocker.io/fast".to_string(),
        "https://rpc.flashbots.net/fast".to_string(),
    ])
});

pub fn submit_rpcs() -> Vec<String> {
    SUBMIT_RPCS.read().clone()
}

pub fn set_submit_rpcs(rpcs: Vec<String>) {
    *SUBMIT_RPCS.write() = rpcs;
}

pub const DEFAULT_CONTRACT: &str = "0xAC7b5d06fa1e77D08aea40d46cB7C5923A87A0cc";
pub const EPOCH_BLOCKS: u64 = 100;
pub const MAX_MINTS_PER_BLOCK: u64 = 10;
