/*!
 * metrics.rs: hash-rate accounting. One global Metrics owns:
 *  - lifetime totals (hashes, hits, mints, submit failures, epochs)
 *  - a per-device hashes counter and last-tick snapshot, used by the
 *    status logger to produce per-GPU instantaneous rates.
 *
 * Each device-worker registers itself once via `register_device` and
 * then bumps per-device hashes via `bump_device_hashes(device_id, n)`.
 * The status logger calls `snapshot` once per tick to read out an
 * atomic view of all counters and the per-device hash deltas since
 * the previous snapshot.
 */

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

pub struct Metrics {
    started: Instant,
    last_snapshot: Mutex<Instant>,

    hashes: AtomicU64,
    last_snapshot_hashes: AtomicU64,

    hits: AtomicU64,
    mints: AtomicU64,
    submit_failures: AtomicU64,
    epoch_rotations: AtomicU64,

    devices: Mutex<HashMap<usize, DeviceCounters>>,
}

struct DeviceCounters {
    name: String,
    hashes: AtomicU64,
    last_snapshot_hashes: AtomicU64,
}

/* One immutable view returned by `Metrics::snapshot`. The logger
 * formats this; the Metrics object stays untouched apart from updating
 * the last-snapshot baselines for next tick. */
pub struct Snapshot {
    pub elapsed: f64,            /* seconds since miner start */
    pub dt: f64,                 /* seconds since previous snapshot */
    pub total_hashes: u64,
    pub total_rate: f64,         /* hashes/sec across all devices since last tick */
    pub avg_rate: f64,           /* hashes/sec averaged over uptime */
    pub hits: u64,
    pub mints: u64,
    pub submit_failures: u64,
    pub epoch_rotations: u64,
    pub devices: Vec<DeviceSnapshot>,
}

pub struct DeviceSnapshot {
    pub device_id: usize,
    pub name: String,
    pub rate: f64,               /* hashes/sec since last tick */
    pub total: u64,
}

impl Metrics {
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            started: now,
            last_snapshot: Mutex::new(now),
            hashes: AtomicU64::new(0),
            last_snapshot_hashes: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            mints: AtomicU64::new(0),
            submit_failures: AtomicU64::new(0),
            epoch_rotations: AtomicU64::new(0),
            devices: Mutex::new(HashMap::new()),
        }
    }

    pub fn register_device(&self, device_id: usize, name: String) {
        let mut g = self.devices.lock();
        g.entry(device_id).or_insert_with(|| DeviceCounters {
            name,
            hashes: AtomicU64::new(0),
            last_snapshot_hashes: AtomicU64::new(0),
        });
    }

    pub fn bump_hashes(&self, n: u64) { self.hashes.fetch_add(n, Ordering::Relaxed); }
    pub fn bump_device_hashes(&self, device_id: usize, n: u64) {
        if let Some(d) = self.devices.lock().get(&device_id) {
            d.hashes.fetch_add(n, Ordering::Relaxed);
        }
    }
    pub fn bump_hits(&self) { self.hits.fetch_add(1, Ordering::Relaxed); }
    pub fn bump_mints(&self) { self.mints.fetch_add(1, Ordering::Relaxed); }
    pub fn bump_submit_failures(&self) { self.submit_failures.fetch_add(1, Ordering::Relaxed); }
    pub fn bump_epoch_rotations(&self) { self.epoch_rotations.fetch_add(1, Ordering::Relaxed); }

    pub fn hashes(&self) -> u64 { self.hashes.load(Ordering::Relaxed) }
    pub fn hits(&self) -> u64 { self.hits.load(Ordering::Relaxed) }
    pub fn mints(&self) -> u64 { self.mints.load(Ordering::Relaxed) }

    pub fn snapshot(&self) -> Snapshot {
        let now = Instant::now();
        let mut last = self.last_snapshot.lock();
        let dt = now.duration_since(*last).as_secs_f64().max(1e-6);

        let total = self.hashes();
        let prev_total = self.last_snapshot_hashes.swap(total, Ordering::Relaxed);
        let total_rate = (total.saturating_sub(prev_total) as f64) / dt;

        let elapsed = self.started.elapsed().as_secs_f64().max(1e-6);
        let avg_rate = (total as f64) / elapsed;

        let mut devs = Vec::new();
        for (&id, d) in self.devices.lock().iter() {
            let dt_total = d.hashes.load(Ordering::Relaxed);
            let dt_prev  = d.last_snapshot_hashes.swap(dt_total, Ordering::Relaxed);
            let rate = (dt_total.saturating_sub(dt_prev) as f64) / dt;
            devs.push(DeviceSnapshot { device_id: id, name: d.name.clone(), rate, total: dt_total });
        }
        devs.sort_by_key(|d| d.device_id);

        *last = now;

        Snapshot {
            elapsed,
            dt,
            total_hashes: total,
            total_rate,
            avg_rate,
            hits: self.hits(),
            mints: self.mints(),
            submit_failures: self.submit_failures.load(Ordering::Relaxed),
            epoch_rotations: self.epoch_rotations.load(Ordering::Relaxed),
            devices: devs,
        }
    }
}

/* Format a hash rate as "<num> <unit>" where unit auto-scales. */
pub fn fmt_rate(hps: f64) -> String {
    const K: f64 = 1_000.0;
    const M: f64 = 1_000_000.0;
    const G: f64 = 1_000_000_000.0;
    const T: f64 = 1_000_000_000_000.0;
    if      hps >= T { format!("{:6.2} TH/s", hps / T) }
    else if hps >= G { format!("{:6.2} GH/s", hps / G) }
    else if hps >= M { format!("{:6.2} MH/s", hps / M) }
    else if hps >= K { format!("{:6.2} KH/s", hps / K) }
    else             { format!("{:6.0}  H/s", hps) }
}

pub fn fmt_uptime(secs: f64) -> String {
    let s = secs as u64;
    let d = s / 86_400;
    let h = (s % 86_400) / 3600;
    let m = (s % 3600) / 60;
    let sec = s % 60;
    if d > 0      { format!("{d}d{h:02}h{m:02}m") }
    else if h > 0 { format!("{h}h{m:02}m{sec:02}s") }
    else if m > 0 { format!("{m}m{sec:02}s") }
    else          { format!("{sec}s") }
}

/* Render a U256 as decimal scientific with 3 significant digits, e.g.
 * 411376139330... -> "4.11e62". Useful for difficulty values that span
 * 60+ orders of magnitude. */
pub fn fmt_u256_sci(v: alloy::primitives::U256) -> String {
    if v.is_zero() { return "0".to_string(); }
    let s = v.to_string();
    let n = s.len();
    if n <= 4 { return s; }
    let lead = &s[..1];
    let rest = &s[1..3];
    format!("{lead}.{rest}e{}", n - 1)
}

/* Render a token amount in 18-decimal base units as a human number with
 * up to 4 fractional digits, e.g. 100000000000000000000 -> "100.0000". */
pub fn fmt_token_1e18(v: alloy::primitives::U256) -> String {
    const ONE: u128 = 1_000_000_000_000_000_000;
    let whole = (v / alloy::primitives::U256::from(ONE)).to::<u128>();
    let frac  = (v % alloy::primitives::U256::from(ONE)).to::<u128>();
    /* 4 decimal places */
    let frac4 = frac / 100_000_000_000_000;
    format!("{whole}.{:04}", frac4)
}
