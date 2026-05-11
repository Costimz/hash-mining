/*!
 * cuda.rs: Rust bindings to the extern "C" surface defined in cuda/miner.h.
 * One `CudaContext` owns the per-device CUDA state. The wrapper takes a
 * `&mut self` for every mutating call so we don't need to share contexts
 * across threads; each worker thread owns one.
 */

use std::os::raw::{c_char, c_int, c_uchar};

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct MinerResult {
    pub found: u32,
    pub _pad: u32,
    pub counter: u64,
    pub hash_be: [u64; 4],
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct MinerDeviceInfo {
    pub device_id: c_int,
    pub compute_major: c_int,
    pub compute_minor: c_int,
    pub sm_count: c_int,
    pub max_threads_per_block: c_int,
    pub warp_size: c_int,
    pub name: [c_char; 256],
    pub total_memory_bytes: u64,
}

impl Default for MinerDeviceInfo {
    fn default() -> Self {
        Self {
            device_id: 0,
            compute_major: 0,
            compute_minor: 0,
            sm_count: 0,
            max_threads_per_block: 0,
            warp_size: 0,
            name: [0; 256],
            total_memory_bytes: 0,
        }
    }
}

impl MinerDeviceInfo {
    pub fn name_str(&self) -> String {
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(self.name.as_ptr() as *const u8, self.name.len())
        };
        let nul = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
        String::from_utf8_lossy(&bytes[..nul]).into_owned()
    }
}

#[repr(C)]
struct OpaqueCtx {
    _private: [u8; 0],
}

extern "C" {
    fn miner_device_count() -> c_int;
    fn miner_device_info(device_id: c_int, out: *mut MinerDeviceInfo) -> c_int;
    fn miner_create(device_id: c_int) -> *mut OpaqueCtx;
    fn miner_destroy(ctx: *mut OpaqueCtx);
    fn miner_set_job(
        ctx: *mut OpaqueCtx,
        challenge: *const c_uchar,
        difficulty: *const c_uchar,
        nonce_prefix: *const c_uchar,
    ) -> c_int;
    fn miner_recommended_launch(
        ctx: *mut OpaqueCtx,
        blocks: *mut c_int,
        tpb: *mut c_int,
        iters: *mut c_int,
    ) -> c_int;
    fn miner_search(
        ctx: *mut OpaqueCtx,
        base_counter: u64,
        blocks: c_int,
        threads_per_block: c_int,
        iters_per_thread: c_int,
        out: *mut MinerResult,
        attempted: *mut u64,
    ) -> c_int;
    fn miner_self_test(
        ctx: *mut OpaqueCtx,
        input_64: *const c_uchar,
        out_hash: *mut c_uchar,
    ) -> c_int;
}

pub fn device_count() -> Result<usize, i32> {
    let n = unsafe { miner_device_count() };
    if n < 0 { Err(n) } else { Ok(n as usize) }
}

pub fn device_info(device_id: usize) -> Result<MinerDeviceInfo, i32> {
    let mut info = MinerDeviceInfo::default();
    let r = unsafe { miner_device_info(device_id as c_int, &mut info) };
    if r != 0 { Err(r) } else { Ok(info) }
}

pub struct CudaContext {
    ptr: *mut OpaqueCtx,
    device_id: usize,
    info: MinerDeviceInfo,
}

unsafe impl Send for CudaContext {}

impl CudaContext {
    pub fn new(device_id: usize) -> Result<Self, String> {
        let info = device_info(device_id).map_err(|e| format!("cudaGetDeviceProperties failed: {e}"))?;
        let ptr = unsafe { miner_create(device_id as c_int) };
        if ptr.is_null() {
            return Err(format!("miner_create({device_id}) returned NULL"));
        }
        Ok(Self { ptr, device_id, info })
    }

    pub fn device_id(&self) -> usize { self.device_id }
    pub fn info(&self) -> &MinerDeviceInfo { &self.info }

    pub fn set_job(
        &mut self,
        challenge: &[u8; 32],
        difficulty: &[u8; 32],
        nonce_prefix: &[u8; 24],
    ) -> Result<(), i32> {
        let r = unsafe {
            miner_set_job(self.ptr, challenge.as_ptr(), difficulty.as_ptr(), nonce_prefix.as_ptr())
        };
        if r != 0 { Err(r) } else { Ok(()) }
    }

    pub fn recommended_launch(&self) -> Result<(i32, i32, i32), i32> {
        let mut b = 0;
        let mut t = 0;
        let mut it = 0;
        let r = unsafe { miner_recommended_launch(self.ptr, &mut b, &mut t, &mut it) };
        if r != 0 { Err(r) } else { Ok((b, t, it)) }
    }

    pub fn search(
        &mut self,
        base_counter: u64,
        blocks: i32,
        threads_per_block: i32,
        iters_per_thread: i32,
    ) -> Result<(MinerResult, u64), i32> {
        let mut out = MinerResult::default();
        let mut attempted = 0u64;
        let r = unsafe {
            miner_search(
                self.ptr,
                base_counter,
                blocks,
                threads_per_block,
                iters_per_thread,
                &mut out,
                &mut attempted,
            )
        };
        if r != 0 { Err(r) } else { Ok((out, attempted)) }
    }

    pub fn self_test(&mut self, input_64: &[u8; 64]) -> Result<[u8; 32], i32> {
        let mut out = [0u8; 32];
        let r = unsafe {
            miner_self_test(self.ptr, input_64.as_ptr(), out.as_mut_ptr())
        };
        if r != 0 { Err(r) } else { Ok(out) }
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { miner_destroy(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

pub fn list_devices() -> Result<Vec<MinerDeviceInfo>, String> {
    let n = device_count().map_err(|e| format!("device_count failed: {e}"))?;
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        v.push(device_info(i).map_err(|e| format!("device_info({i}) failed: {e}"))?);
    }
    Ok(v)
}

/* Tracks counters and reconstructs the full 32-byte big-endian nonce
 * from a (prefix, counter) pair. The prefix is fixed per-launch; the
 * counter is what the kernel writes back. */
pub fn nonce_from_counter(prefix: &[u8; 24], counter: u64) -> [u8; 32] {
    let mut out = [0u8; 32];
    out[..24].copy_from_slice(prefix);
    out[24..].copy_from_slice(&counter.to_be_bytes());
    out
}

