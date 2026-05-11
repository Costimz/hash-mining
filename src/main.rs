/*!
 * main.rs: CLI entry point. Subcommands:
 *   - devices    list visible CUDA devices and recommended launch params
 *   - selftest   verify the kernel's keccak matches a host reference
 *   - bench      run the kernel at maximum-difficulty (every hash wins)
 *                or at a chosen difficulty for N seconds and report MH/s
 *   - mine       end-to-end mining: connect, watch, solve, submit
 *   - wallet new mint a new hex private-key file
 */

#![allow(dead_code)]

mod chain;
mod config;
mod coordinator;
mod cpu;
mod cuda;
mod globals;
mod metrics;
mod submit;
mod types;
mod wallet;

use crate::chain::{keccak_host, verify_hit, ReadClient};
use crate::config::Config;
use crate::cuda::{list_devices, nonce_from_counter, CudaContext};
use crate::metrics::Metrics;
use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{info, warn};

#[derive(Parser, Debug)]
#[command(name = "hash-miner", about = "CUDA + Rust miner for the $HASH protocol")]
struct Cli {
    #[arg(long, global = true, default_value = "info")]
    log_level: String,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    Devices,

    Selftest {
        #[arg(long, default_value_t = 0)]
        device: usize,
    },

    Bench {
        #[arg(long, default_value_t = 10)]
        seconds: u64,

        #[arg(long, default_values_t = [0usize])]
        device: Vec<usize>,

        #[arg(long, default_value_t = 64)]
        iters: i32,

        #[arg(long)]
        difficulty_hex: Option<String>,

        /* Optional manual override for kernel launch shape. Useful for
         * sweeping the (blocks, tpb) grid to find the throughput peak on
         * a given GPU without recompiling. */
        #[arg(long)]
        blocks: Option<i32>,

        #[arg(long)]
        tpb: Option<i32>,
    },

    Mine {
        #[arg(long)]
        config: PathBuf,
    },

    Wallet {
        #[command(subcommand)]
        sub: WalletCmd,
    },
}

#[derive(Subcommand, Debug)]
enum WalletCmd {
    /* Generate a fresh address + private key. The key is printed to
     * stdout only - the miner reads its key from an environment
     * variable, so put the printed key in your shell rc, systemd unit,
     * or a secrets manager. There is no on-disk wallet file. */
    New,
}

fn init_tracing(level: &str) {
    use tracing_subscriber::EnvFilter;
    let filter = EnvFilter::try_new(level)
        .unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .compact()
        .init();
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    init_tracing(&cli.log_level);

    match cli.cmd {
        Cmd::Devices         => cmd_devices(),
        Cmd::Selftest { device } => cmd_selftest(device),
        Cmd::Bench { seconds, device, iters, difficulty_hex, blocks, tpb } => {
            cmd_bench(seconds, device, iters, difficulty_hex, blocks, tpb)
        }
        Cmd::Mine { config }     => cmd_mine(config).await,
        Cmd::Wallet { sub }      => cmd_wallet(sub),
    }
}

fn cmd_devices() -> Result<()> {
    let devs = list_devices().map_err(|e| anyhow!(e))?;
    if devs.is_empty() {
        println!("no CUDA devices visible");
        return Ok(());
    }
    println!("{} CUDA device(s):", devs.len());
    for d in &devs {
        println!(
            "  [{}] {} (cc {}.{}, {} SMs, {:.1} GB)",
            d.device_id,
            d.name_str(),
            d.compute_major,
            d.compute_minor,
            d.sm_count,
            (d.total_memory_bytes as f64) / (1024.0 * 1024.0 * 1024.0)
        );
    }
    let ctx = CudaContext::new(devs[0].device_id as usize).map_err(|e| anyhow!(e))?;
    let (b, t, i) = ctx.recommended_launch().map_err(|e| anyhow!("recommended_launch: {e}"))?;
    println!("recommended launch on device 0: blocks={b} threads={t} iters={i}");
    Ok(())
}

fn cmd_selftest(device: usize) -> Result<()> {
    let mut ctx = CudaContext::new(device).map_err(|e| anyhow!(e))?;
    let info = ctx.info();
    info!(
        device,
        name = %info.name_str(),
        "selftest"
    );

    /* Vector 1: all-zero input - well-known keccak. */
    let zero_in = [0u8; 64];
    let zero_out = keccak_host(&zero_in);
    let kernel_out = ctx.self_test(&zero_in).map_err(|e| anyhow!("self_test: {e}"))?;
    if zero_out != kernel_out {
        warn!(
            host = %hex::encode(zero_out),
            kernel = %hex::encode(kernel_out),
            "MISMATCH for zero input"
        );
        return Err(anyhow!("zero-input vector failed"));
    }
    info!(hash = %hex::encode(zero_out), "zero-input ok");

    /* Vector 2: arbitrary input. */
    let mut buf = [0u8; 64];
    for i in 0..64 { buf[i] = i as u8; }
    let host = keccak_host(&buf);
    let kern = ctx.self_test(&buf).map_err(|e| anyhow!("self_test: {e}"))?;
    if host != kern {
        warn!(
            host = %hex::encode(host),
            kernel = %hex::encode(kern),
            "MISMATCH for sequence input"
        );
        return Err(anyhow!("sequence vector failed"));
    }
    info!(hash = %hex::encode(host), "sequence input ok");

    /* Vector 3: mining-shape input via mine_kernel by setting difficulty
       to max (every hash wins). Verify the kernel's hit hash matches the
       host's keccak for the same (challenge, prefix, counter). */
    let challenge = [
        0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0,
        0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88,
        0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x00, 0x11,
        0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99,
    ];
    let prefix = [0xa5u8; 24];
    let max_diff = [0xffu8; 32];
    ctx.set_job(&challenge, &max_diff, &prefix).map_err(|e| anyhow!(e))?;

    let (b, t, _) = ctx.recommended_launch().map_err(|e| anyhow!(e))?;
    let (res, _attempted) = ctx.search(0, b.min(64), t.min(32), 1).map_err(|e| anyhow!(e))?;
    if res.found == 0 {
        return Err(anyhow!("mine_kernel: expected at-max-diff to always hit"));
    }

    let nonce = nonce_from_counter(&prefix, res.counter);
    let mut full = [0u8; 64];
    full[..32].copy_from_slice(&challenge);
    full[32..].copy_from_slice(&nonce);
    let host_hash = keccak_host(&full);

    let mut kern_hash = [0u8; 32];
    for (i, &v) in res.hash_be.iter().enumerate() {
        kern_hash[i*8..(i+1)*8].copy_from_slice(&v.to_be_bytes());
    }
    if host_hash != kern_hash {
        warn!(
            host = %hex::encode(host_hash),
            kernel = %hex::encode(kern_hash),
            counter = res.counter,
            "MISMATCH for mine-kernel hit"
        );
        return Err(anyhow!("mine-kernel vector failed"));
    }
    info!(
        counter = res.counter,
        nonce = %hex::encode(nonce),
        hash = %hex::encode(host_hash),
        "mine-kernel hit ok"
    );

    /* Verify it passes the host's verify_hit function. */
    let _h = verify_hit(&challenge, &nonce, alloy::primitives::U256::MAX)
        .map_err(|e| anyhow!("host verify_hit failed: {e}"))?;
    info!("verify_hit ok");

    println!("selftest PASSED");
    Ok(())
}

fn cmd_bench(
    seconds: u64,
    devices: Vec<usize>,
    iters: i32,
    difficulty_hex: Option<String>,
    blocks_override: Option<i32>,
    tpb_override: Option<i32>,
) -> Result<()> {
    let diff = match difficulty_hex {
        Some(s) => {
            let s = s.strip_prefix("0x").unwrap_or(&s);
            let b = hex::decode(s)?;
            if b.len() != 32 { return Err(anyhow!("difficulty hex must be 32 bytes")); }
            let mut a = [0u8; 32];
            a.copy_from_slice(&b);
            a
        }
        None => {
            /* Very small target so we essentially never hit. Roughly 2^-32. */
            let mut d = [0u8; 32];
            d[3] = 0x01;
            d
        }
    };

    let challenge = [0xa5u8; 32];
    let metrics = Arc::new(Metrics::new());

    let mut handles = Vec::new();
    for d in devices {
        let metrics = Arc::clone(&metrics);
        let challenge = challenge;
        let diff = diff;
        handles.push(std::thread::spawn(move || -> Result<()> {
            let mut ctx = CudaContext::new(d).map_err(|e| anyhow!(e))?;
            let info = *ctx.info();
            info!(
                device = d,
                name = %info.name_str(),
                "bench device"
            );
            let mut prefix = [0u8; 24];
            prefix[0] = d as u8;
            ctx.set_job(&challenge, &diff, &prefix).map_err(|e| anyhow!(e))?;
            let (mut b, mut t, _) = ctx.recommended_launch().map_err(|e| anyhow!(e))?;
            if let Some(bo) = blocks_override { b = bo; }
            if let Some(to) = tpb_override { t = to; }
            info!(device = d, blocks = b, tpb = t, iters = iters, "launch params");
            let mut base: u64 = 0;
            let end = Instant::now() + Duration::from_secs(seconds);
            while Instant::now() < end {
                let (_res, attempted) = ctx.search(base, b, t, iters)
                    .map_err(|e| anyhow!(e))?;
                metrics.bump_hashes(attempted);
                base = base.wrapping_add(attempted);
            }
            Ok(())
        }));
    }
    let t0 = Instant::now();
    for h in handles { let _ = h.join().map_err(|_| anyhow!("worker join"))?; }
    let dt = t0.elapsed().as_secs_f64();
    let total = metrics.hashes();
    println!(
        "bench: {} hashes in {:.2}s = {:.3} GH/s",
        total,
        dt,
        (total as f64) / dt / 1e9,
    );
    Ok(())
}

async fn cmd_mine(config_path: PathBuf) -> Result<()> {
    let cfg = Config::load(&config_path)?;
    globals::set_submit_rpcs(cfg.chain.submit_rpcs.clone());

    let signer = wallet::load_signer(&cfg.wallet)?;
    let miner_addr = signer.address();
    let contract = chain::parse_addr(&cfg.chain.contract)?;

    info!(
        version = env!("CARGO_PKG_VERSION"),
        "hash-miner starting"
    );
    info!(miner = %miner_addr, "wallet");
    info!(addr = %contract, chain_id = cfg.chain.chain_id, "contract");
    info!(rpc = %cfg.chain.read_rpc, "read rpc");
    info!(endpoints = cfg.chain.submit_rpcs.len(), "submit relays");

    let read = Arc::new(ReadClient::connect(&cfg.chain.read_rpc, contract, miner_addr).await?);
    if read.chain_id != cfg.chain.chain_id {
        return Err(anyhow!(
            "chain_id mismatch: rpc reports {}, config wants {}",
            read.chain_id, cfg.chain.chain_id,
        ));
    }
    let gen = read.genesis_complete().await?;
    if !gen {
        return Err(anyhow!("genesisComplete is false; protocol not ready for mining"));
    }

    let submitter = submit::Submitter::new(
        signer,
        contract,
        read.chain_id,
        cfg.submit.clone(),
    );

    let devices = if cfg.mine.cuda_devices.is_empty() {
        let n = cuda::device_count().map_err(|e| anyhow!("device_count: {e}"))?;
        (0..n).collect()
    } else {
        cfg.mine.cuda_devices.clone()
    };
    if devices.is_empty() {
        return Err(anyhow!("no CUDA devices available"));
    }
    info!(count = devices.len(), ids = ?devices, "gpus");

    let metrics = Arc::new(Metrics::new());
    let min_balance_wei = wei_from_eth(cfg.economics.min_balance_eth);

    let coord = coordinator::Coordinator {
        read,
        submitter,
        devices,
        metrics,
        batch_iters: cfg.mine.batch_size.min(i32::MAX as u64) as i32,
        min_balance_wei,
    };
    coord.run().await
}

fn cmd_wallet(sub: WalletCmd) -> Result<()> {
    match sub {
        WalletCmd::New => {
            let signer = alloy::signers::local::PrivateKeySigner::random();
            let secret = signer.to_bytes();
            println!("address:     {}", signer.address());
            println!("private key: 0x{}", hex::encode(secret.as_slice()));
            Ok(())
        }
    }
}

fn wei_from_eth(eth: f64) -> alloy::primitives::U256 {
    if eth <= 0.0 { return alloy::primitives::U256::ZERO; }
    let wei_f = eth * 1e18;
    alloy::primitives::U256::from(wei_f as u128)
}
