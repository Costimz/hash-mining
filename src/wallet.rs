/*!
 * wallet.rs: load the EIP-1559 signing key from an environment variable
 * named in the wallet config. The env-var name is configured per-miner
 * so a single host can run multiple addresses by exporting different
 * variables.
 */

use crate::config::WalletConfig;
use alloy::signers::local::PrivateKeySigner;
use anyhow::{anyhow, Context, Result};

/* Resolve the configured env var, hex-decode its contents, and return
 * a signer bound to the derived address. */
pub fn load_signer(cfg: &WalletConfig) -> Result<PrivateKeySigner> {
    let env_var = cfg.private_key_env.as_str();
    if env_var.is_empty() {
        return Err(anyhow!("wallet.private_key_env is required"));
    }
    let raw = std::env::var(env_var)
        .with_context(|| format!("env var {env_var} for private key is not set"))?;
    parse_hex_secret(raw.trim())
}

fn parse_hex_secret(raw: &str) -> Result<PrivateKeySigner> {
    let stripped = raw.strip_prefix("0x").unwrap_or(raw);
    let bytes = hex::decode(stripped).context("private key is not valid hex")?;
    if bytes.len() != 32 {
        return Err(anyhow!("private key must be 32 bytes, got {}", bytes.len()));
    }
    let arr: [u8; 32] = bytes.try_into().unwrap();
    Ok(PrivateKeySigner::from_bytes(&arr.into())?)
}
