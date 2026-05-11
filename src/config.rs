/*!
 * config.rs: TOML configuration schema. One `Config` per `mine`
 * invocation. Loaded from disk, validated, then handed to the
 * coordinator and submitter; thereafter the values are treated as
 * immutable for the life of the process.
 */

use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub chain:     ChainConfig,
    pub wallet:    WalletConfig,
    pub mine:      MineConfig,
    pub submit:    SubmitConfig,
    #[serde(default)]
    pub economics: EconomicsConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChainConfig {
    pub read_rpc:    String,
    pub submit_rpcs: Vec<String>,
    #[serde(default = "default_chain_id")]
    pub chain_id:    u64,
    pub contract:    String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WalletConfig {
    /* Name of the env variable that holds the hex-encoded private key.
     * Required. There is no on-disk wallet mode; persist the key in
     * your shell rc, a systemd unit's Environment=, or a secrets
     * manager. */
    pub private_key_env: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MineConfig {
    #[serde(default = "default_backend")]
    pub backend:          String,
    #[serde(default)]
    pub cuda_devices:     Vec<usize>,
    #[serde(default)]
    pub cpu_threads:      usize,
    #[serde(default = "default_batch_size")]
    pub batch_size:       u64,
    #[serde(default = "default_poll_interval_ms")]
    pub poll_interval_ms: u64,
    #[serde(default = "default_true")]
    pub event_subscribe:  bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SubmitConfig {
    #[serde(default = "default_priority_tip_gwei")]
    pub priority_tip_gwei:           f64,
    #[serde(default = "default_max_priority_tip_gwei")]
    pub max_priority_tip_gwei:       f64,
    #[serde(default = "default_base_fee_multiplier")]
    pub base_fee_multiplier:         u64,
    #[serde(default = "default_gas_min")]
    pub gas_min:                     u64,
    #[serde(default = "default_gas_max")]
    pub gas_max:                     u64,
    #[serde(default = "default_gas_estimate_timeout_ms")]
    pub gas_estimate_timeout_ms:     u64,
    #[serde(default = "default_submit_timeout_ms")]
    pub submit_timeout_ms:           u64,
    #[serde(default = "default_true")]
    pub preflight_challenge_recheck: bool,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct EconomicsConfig {
    #[serde(default)]
    pub min_hash_price_usd:      f64,
    #[serde(default)]
    pub gas_budget_per_hour_usd: f64,
    #[serde(default)]
    pub min_balance_eth:         f64,
}

impl Config {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("failed to read {}: {e}", path.display()))?;
        let cfg: Config = toml::from_str(&raw)
            .map_err(|e| anyhow::anyhow!("failed to parse {}: {e}", path.display()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> anyhow::Result<()> {
        if self.chain.submit_rpcs.is_empty() {
            anyhow::bail!("chain.submit_rpcs must not be empty");
        }
        if self.chain.contract.is_empty() {
            anyhow::bail!("chain.contract must not be empty");
        }
        if self.wallet.private_key_env.is_empty() {
            anyhow::bail!("wallet.private_key_env must be set");
        }
        Ok(())
    }
}

/* -------- serde defaults -------- */

fn default_chain_id() -> u64 { 1 }
fn default_backend() -> String { "auto".to_string() }
fn default_batch_size() -> u64 { 5_000_000 }
fn default_poll_interval_ms() -> u64 { 2_000 }
fn default_true() -> bool { true }
fn default_priority_tip_gwei() -> f64 { 6.0 }
fn default_max_priority_tip_gwei() -> f64 { 25.0 }
fn default_base_fee_multiplier() -> u64 { 3 }
fn default_gas_min() -> u64 { 200_000 }
fn default_gas_max() -> u64 { 400_000 }
fn default_gas_estimate_timeout_ms() -> u64 { 4_000 }
fn default_submit_timeout_ms() -> u64 { 12_000 }
