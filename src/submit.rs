/*!
 * submit.rs: build and submit `mine(nonce)` transactions through a
 * private orderflow endpoint. We sign locally with the supplied
 * `PrivateKeySigner`, then race two HTTP POSTs to the configured
 * submit_rpcs and wait for the first to acknowledge.
 *
 * Receipt watching is delegated to the read provider since private
 * relays don't expose canonical receipts.
 */

use crate::chain::{encode_mine_call, ReadClient};
use crate::config::SubmitConfig;
use alloy::consensus::{SignableTransaction, TxEip1559, TxEnvelope};
use alloy::eips::eip2930::AccessList;
use alloy::network::TxSignerSync;
use alloy::primitives::{Address, Bytes, TxKind, U256};
use alloy::providers::Provider;
use alloy::signers::local::PrivateKeySigner;
use alloy::rlp::Encodable;
use anyhow::{anyhow, Context, Result};
use serde_json::json;
use std::time::Duration;

const GWEI: u128 = 1_000_000_000;

pub struct Submitter {
    pub signer: PrivateKeySigner,
    pub miner: Address,
    pub contract: Address,
    pub chain_id: u64,
    pub cfg: SubmitConfig,
    pub http: reqwest::Client,
}

#[derive(Debug, Clone)]
pub struct SubmissionPlan {
    pub max_fee_per_gas: u128,
    pub max_priority_fee_per_gas: u128,
    pub gas_limit: u64,
    pub nonce: u64,
}

#[derive(Debug)]
pub enum SubmitOutcome {
    Mined { tx_hash: [u8; 32], block_number: u64, reward: U256 },
    Reverted { tx_hash: [u8; 32], reason: String },
    Timeout,
    Dropped,
}

impl Submitter {
    pub fn new(
        signer: PrivateKeySigner,
        contract: Address,
        chain_id: u64,
        cfg: SubmitConfig,
    ) -> Self {
        let miner = signer.address();
        let http = reqwest::Client::builder()
            .timeout(Duration::from_millis(cfg.submit_timeout_ms))
            .pool_max_idle_per_host(8)
            .build()
            .expect("reqwest client build");
        Self { signer, miner, contract, chain_id, cfg, http }
    }

    /* Build the EIP-1559 plan using the live base fee plus a tip. */
    pub async fn plan(&self, read: &ReadClient, tip_gwei: f64) -> Result<SubmissionPlan> {
        let base_fee_wei = tokio::time::timeout(
            Duration::from_millis(self.cfg.gas_estimate_timeout_ms),
            read.provider.get_block_by_number(alloy::eips::BlockNumberOrTag::Pending),
        )
        .await
        .context("base fee fetch timed out")?
        .context("eth_getBlockByNumber failed")?
        .and_then(|b| b.header.base_fee_per_gas)
        .map(|b| b as u128)
        .unwrap_or(30 * GWEI);

        let tip = (tip_gwei.max(0.0) * (GWEI as f64)) as u128;
        let cap = (self.cfg.max_priority_tip_gwei.max(0.0) * (GWEI as f64)) as u128;
        let tip = tip.min(cap);

        let mult = self.cfg.base_fee_multiplier as u128;
        let max_fee_per_gas = base_fee_wei.saturating_mul(mult).saturating_add(tip);

        let nonce_tx = read.provider.get_transaction_count(self.miner).pending().await
            .context("eth_getTransactionCount failed")?;

        Ok(SubmissionPlan {
            max_fee_per_gas,
            max_priority_fee_per_gas: tip,
            gas_limit: self.cfg.gas_max,
            nonce: nonce_tx,
        })
    }

    /* Sign the raw EIP-1559 tx that calls mine(nonce). */
    pub fn sign_mine_tx(&self, plan: &SubmissionPlan, mine_nonce: &[u8; 32]) -> Result<Vec<u8>> {
        let data = encode_mine_call(mine_nonce);
        let mut tx = TxEip1559 {
            chain_id: self.chain_id,
            nonce: plan.nonce,
            gas_limit: plan.gas_limit,
            max_fee_per_gas: plan.max_fee_per_gas,
            max_priority_fee_per_gas: plan.max_priority_fee_per_gas,
            to: TxKind::Call(self.contract),
            value: U256::ZERO,
            access_list: AccessList::default(),
            input: Bytes::from(data),
        };
        let signature = self.signer.sign_transaction_sync(&mut tx)
            .map_err(|e| anyhow!("sign failed: {e}"))?;
        let envelope = TxEnvelope::Eip1559(tx.into_signed(signature));
        let mut buf = Vec::with_capacity(256);
        envelope.encode(&mut buf);
        Ok(buf)
    }

    /* Race two POST submissions: the first to ack wins, the other is
     * ignored. Returns the tx hash that was accepted. */
    pub async fn race_submit(&self, raw_tx: &[u8]) -> Result<[u8; 32]> {
        let hex_tx = format!("0x{}", hex::encode(raw_tx));
        let endpoints: Vec<String> = std::env::var("HASH_MINER_SUBMIT_RPCS")
            .ok()
            .map(|s| s.split(',').map(|x| x.trim().to_string()).collect())
            .unwrap_or_default();
        let endpoints = if endpoints.is_empty() {
            crate::globals::submit_rpcs()
        } else {
            endpoints
        };

        if endpoints.is_empty() {
            return Err(anyhow!("no submit RPCs configured"));
        }

        let mut tasks = Vec::with_capacity(endpoints.len());
        for ep in &endpoints {
            let http = self.http.clone();
            let url = ep.clone();
            let hex_tx = hex_tx.clone();
            tasks.push(tokio::spawn(async move {
                let body = json!({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "eth_sendRawTransaction",
                    "params": [hex_tx],
                });
                http.post(&url)
                    .json(&body)
                    .send().await
                    .and_then(|r| r.error_for_status())?
                    .json::<serde_json::Value>().await
            }));
        }

        let mut last_err: Option<String> = None;
        for t in tasks {
            match t.await {
                Ok(Ok(v)) => {
                    if let Some(r) = v.get("result").and_then(|x| x.as_str()) {
                        let stripped = r.strip_prefix("0x").unwrap_or(r);
                        let bytes = hex::decode(stripped).context("bad tx hash hex")?;
                        if bytes.len() == 32 {
                            let mut h = [0u8; 32];
                            h.copy_from_slice(&bytes);
                            return Ok(h);
                        }
                    }
                    if let Some(e) = v.get("error") {
                        last_err = Some(e.to_string());
                    }
                }
                Ok(Err(e)) => { last_err = Some(e.to_string()); }
                Err(e)     => { last_err = Some(e.to_string()); }
            }
        }
        Err(anyhow!("all submit endpoints failed: {}", last_err.unwrap_or_default()))
    }

    /* Poll for the receipt; report success, revert reason, or timeout. */
    pub async fn await_receipt(&self, read: &ReadClient, tx_hash: [u8; 32]) -> SubmitOutcome {
        let deadline = std::time::Instant::now() + Duration::from_millis(self.cfg.submit_timeout_ms);
        let hash = alloy::primitives::TxHash::from(tx_hash);
        loop {
            if std::time::Instant::now() >= deadline {
                return SubmitOutcome::Timeout;
            }
            match read.provider.get_transaction_receipt(hash).await {
                Ok(Some(rcpt)) => {
                    let block_number = rcpt.block_number.unwrap_or(0);
                    if rcpt.status() {
                        let reward = extract_reward(&rcpt);
                        return SubmitOutcome::Mined { tx_hash, block_number, reward };
                    }
                    let reason = match &rcpt.transaction_index {
                        Some(_) => extract_revert(read, &rcpt).await,
                        None    => "no-tx-index".to_string(),
                    };
                    return SubmitOutcome::Reverted { tx_hash, reason };
                }
                Ok(None) => tokio::time::sleep(Duration::from_millis(500)).await,
                Err(_)   => tokio::time::sleep(Duration::from_millis(500)).await,
            }
        }
    }
}

fn extract_reward(rcpt: &alloy::rpc::types::TransactionReceipt) -> U256 {
    use alloy::sol_types::SolEvent;
    use crate::chain::Hash::Mined;
    for log in rcpt.logs() {
        let prim = log.inner.clone();
        if let Ok(decoded) = Mined::decode_log(&prim) {
            return decoded.reward;
        }
    }
    U256::ZERO
}

async fn extract_revert(_read: &ReadClient, _rcpt: &alloy::rpc::types::TransactionReceipt) -> String {
    "revert".to_string()
}
