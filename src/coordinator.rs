/*!
 * coordinator.rs: top-level mining loop. Owns one `CudaContext` per
 * device, the read client, the submitter, and the metrics. Watches for
 * epoch rotations and re-targets workers when the challenge changes.
 *
 * Each device runs in its own tokio blocking task; coordination is via
 * a tokio mpsc channel for hits and a tokio watch for the current job.
 */

use crate::chain::{verify_hit, ReadClient};
use crate::cuda::{nonce_from_counter, CudaContext};
use crate::metrics::Metrics;
use crate::submit::{SubmitOutcome, Submitter};
use crate::types::{Hit, Job, SharedStop, StopFlag};
use alloy::primitives::U256;
use anyhow::Result;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, watch};
use tracing::{error, info, warn};

pub struct Coordinator {
    pub read: Arc<ReadClient>,
    pub submitter: Submitter,
    pub devices: Vec<usize>,
    pub metrics: Arc<Metrics>,
    pub batch_iters: i32,
    pub min_balance_wei: U256,
}

impl Coordinator {
    pub async fn run(self) -> Result<()> {
        let (job_tx, job_rx) = watch::channel(self.bootstrap_job().await?);
        let (hit_tx, mut hit_rx) = mpsc::channel::<Hit>(16);
        let stop: SharedStop = Arc::new(StopFlag::default());

        for &device_id in &self.devices {
            let job_rx = job_rx.clone();
            let hit_tx = hit_tx.clone();
            let stop   = Arc::clone(&stop);
            let metrics = Arc::clone(&self.metrics);
            let batch_iters = self.batch_iters;
            tokio::task::spawn_blocking(move || {
                if let Err(e) = run_device(device_id, job_rx, hit_tx, stop, metrics, batch_iters) {
                    error!(?e, device_id, "device worker terminated");
                }
            });
        }

        let read = Arc::clone(&self.read);
        let job_tx_for_watcher = job_tx.clone();
        tokio::spawn(async move { epoch_watcher(read, job_tx_for_watcher).await; });

        let metrics_clone = Arc::clone(&self.metrics);
        tokio::spawn(async move { tick_logger(metrics_clone).await; });

        while let Some(hit) = hit_rx.recv().await {
            stop.set();
            let cur = job_tx.borrow().clone();
            if cur.epoch != hit.epoch {
                warn!(
                    hit_epoch = hit.epoch,
                    cur_epoch = cur.epoch,
                    "share stale (epoch rotated)"
                );
                stop.clear();
                continue;
            }
            self.metrics.bump_hits();
            self.handle_hit(hit, &cur).await;
            stop.clear();
        }
        Ok(())
    }

    async fn bootstrap_job(&self) -> Result<Job> {
        let challenge = self.read.challenge().await?;
        let difficulty = self.read.current_difficulty().await?;
        let state = self.read.mining_state().await?;
        info!(
            epoch = state.epoch,
            era = %state.era,
            diff = %crate::metrics::fmt_u256_sci(difficulty),
            reward = %crate::metrics::fmt_token_1e18(state.reward),
            "job"
        );
        Ok(Job {
            challenge,
            difficulty,
            epoch: state.epoch,
            miner: self.read.miner,
        })
    }

    async fn handle_hit(&self, hit: Hit, job: &Job) {
        info!(
            epoch = hit.epoch,
            counter = %format!("0x{}", hex::encode(&hit.nonce[24..32])),
            hash = %format!("0x{}...", hex::encode(&hit.hash[..6])),
            "share found"
        );
        let host = match verify_hit(&job.challenge, &hit.nonce, job.difficulty) {
            Ok(h) => h,
            Err(e) => { warn!(reason = %e, "share rejected (host verify)"); return; }
        };
        if host != hit.hash {
            warn!("share rejected (kernel hash != host hash)");
            return;
        }
        match self.submit_with_recheck(&hit, job).await {
            Ok(()) => self.metrics.bump_mints(),
            Err(e) => {
                self.metrics.bump_submit_failures();
                warn!(reason = %e, "submit failed");
            }
        }
    }

    async fn submit_with_recheck(&self, hit: &Hit, job: &Job) -> Result<()> {
        let now_challenge = self.read.challenge().await?;
        if now_challenge != job.challenge {
            anyhow::bail!("challenge rotated mid-submit");
        }
        let bal = self.read.balance_eth(self.submitter.miner).await?;
        if bal < self.min_balance_wei {
            anyhow::bail!("eth balance below threshold ({bal} wei)");
        }
        let plan = self.submitter.plan(&self.read, self.submitter.cfg.priority_tip_gwei).await?;
        let raw = self.submitter.sign_mine_tx(&plan, &hit.nonce)?;
        let tx_hash = self.submitter.race_submit(&raw).await?;
        info!(
            tx = %format!("0x{}", hex::encode(&tx_hash)),
            tip = %format!("{}gwei", format_gwei(plan.max_priority_fee_per_gas)),
            cap = %format!("{}gwei", format_gwei(plan.max_fee_per_gas)),
            "submit"
        );
        match self.submitter.await_receipt(&self.read, tx_hash).await {
            SubmitOutcome::Mined { block_number, reward, .. } => {
                info!(
                    block = block_number,
                    reward = %crate::metrics::fmt_token_1e18(reward),
                    "accepted"
                );
                Ok(())
            }
            SubmitOutcome::Reverted { reason, .. } => {
                anyhow::bail!("reverted: {reason}")
            }
            SubmitOutcome::Timeout => anyhow::bail!("timeout"),
            SubmitOutcome::Dropped => anyhow::bail!("dropped"),
        }
    }
}

fn run_device(
    device_id: usize,
    job_rx: watch::Receiver<Job>,
    hit_tx: mpsc::Sender<Hit>,
    stop: SharedStop,
    metrics: Arc<Metrics>,
    batch_iters: i32,
) -> Result<()> {
    let mut ctx = CudaContext::new(device_id)
        .map_err(|e| anyhow::anyhow!("cuda init device {device_id}: {e}"))?;
    let info = *ctx.info();
    let name = info.name_str();
    metrics.register_device(device_id, name.clone());
    info!(
        gpu = device_id,
        name = %name,
        cc = format!("{}.{}", info.compute_major, info.compute_minor),
        sm = info.sm_count,
        "gpu online"
    );

    let (rec_b, rec_t, _rec_i) = ctx.recommended_launch()
        .map_err(|e| anyhow::anyhow!("recommended_launch failed: {e}"))?;
    let iters = batch_iters;

    let mut prefix = [0u8; 24];
    {
        use rand::RngCore;
        rand::thread_rng().fill_bytes(&mut prefix);
        prefix[0] = device_id as u8;
    }

    let mut job_rx = job_rx;
    let mut cur_job = job_rx.borrow().clone();
    let diff_be: [u8; 32] = cur_job.difficulty_be();
    ctx.set_job(&cur_job.challenge, &diff_be, &prefix)
        .map_err(|e| anyhow::anyhow!("set_job failed: {e}"))?;

    let mut base_counter: u64 = 0u64;
    loop {
        if job_rx.has_changed().unwrap_or(false) {
            cur_job = job_rx.borrow_and_update().clone();
            let diff_be: [u8; 32] = cur_job.difficulty_be();
            ctx.set_job(&cur_job.challenge, &diff_be, &prefix)
                .map_err(|e| anyhow::anyhow!("set_job (rotate) failed: {e}"))?;
            base_counter = 0;
            stop.clear();
        }
        if stop.is_set() {
            std::thread::sleep(Duration::from_millis(20));
            continue;
        }

        let (result, attempted) = ctx.search(base_counter, rec_b, rec_t, iters)
            .map_err(|e| anyhow::anyhow!("search failed: {e}"))?;
        metrics.bump_hashes(attempted);
        metrics.bump_device_hashes(device_id, attempted);
        base_counter = base_counter.wrapping_add(attempted);

        if result.found != 0 {
            let nonce = nonce_from_counter(&prefix, result.counter);
            let mut hash = [0u8; 32];
            for (i, &v) in result.hash_be.iter().enumerate() {
                hash[i*8..(i+1)*8].copy_from_slice(&v.to_be_bytes());
            }
            let hit = Hit {
                nonce,
                hash,
                epoch: cur_job.epoch,
                miner: cur_job.miner,
            };
            if hit_tx.blocking_send(hit).is_err() {
                error!("coordinator hit channel closed");
                return Ok(());
            }
            stop.set();
        }
    }
}

async fn epoch_watcher(read: Arc<ReadClient>, job_tx: watch::Sender<Job>) {
    let mut tick = tokio::time::interval(Duration::from_millis(2_000));
    let mut cur = job_tx.borrow().clone();
    loop {
        tick.tick().await;
        let new_chal = match read.challenge().await {
            Ok(c) => c,
            Err(e) => { warn!(?e, "challenge poll failed"); continue; }
        };
        let new_diff = match read.current_difficulty().await {
            Ok(d) => d,
            Err(e) => { warn!(?e, "difficulty poll failed"); continue; }
        };
        let state = match read.mining_state().await {
            Ok(s) => s,
            Err(e) => { warn!(?e, "miningState poll failed"); continue; }
        };
        if new_chal != cur.challenge || state.epoch != cur.epoch {
            info!(
                from = cur.epoch,
                to = state.epoch,
                diff = %crate::metrics::fmt_u256_sci(new_diff),
                "epoch rotated"
            );
            cur = Job {
                challenge: new_chal,
                difficulty: new_diff,
                epoch: state.epoch,
                miner: cur.miner,
            };
            let _ = job_tx.send(cur.clone());
        } else if new_diff != cur.difficulty {
            info!(
                diff = %crate::metrics::fmt_u256_sci(new_diff),
                "difficulty adjusted"
            );
            cur.difficulty = new_diff;
            let _ = job_tx.send(cur.clone());
        }
    }
}

async fn tick_logger(metrics: Arc<Metrics>) {
    let mut tick = tokio::time::interval(Duration::from_secs(10));
    tick.tick().await;
    loop {
        tick.tick().await;
        let s = metrics.snapshot();
        let per_gpu = s.devices.iter()
            .map(|d| format!("gpu{}={}", d.device_id, crate::metrics::fmt_rate(d.rate).trim()))
            .collect::<Vec<_>>()
            .join(" ");
        let line = format!(
            "{} | total {} | avg {} | up {} | hits {} | mints {}",
            per_gpu,
            crate::metrics::fmt_rate(s.total_rate).trim(),
            crate::metrics::fmt_rate(s.avg_rate).trim(),
            crate::metrics::fmt_uptime(s.elapsed),
            s.hits,
            s.mints,
        );
        info!("{}", line);
    }
}

fn format_gwei(wei: u128) -> String {
    format!("{:.2}", wei as f64 / 1e9)
}

