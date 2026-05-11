/**
 * @file kernel.cu
 * @brief Keccak-256 PoW search kernel and host C API for $HASH mining.
 *
 * Input layout (64 bytes, identical to abi.encode(bytes32, uint256)):
 *   bytes  0..31 = challenge (BE)
 *   bytes 32..55 = nonce_prefix (24 BE bytes, fixed per launch)
 *   bytes 56..63 = counter (8 BE bytes, varies per try)
 *
 * Keccak rate is 136 bytes, capacity 64; the 64-byte input fits in one
 * absorption block. Padding: 0x01 at byte 64, 0x80 at byte 135.
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>

#include "miner.h"

#define KECCAK_ROUNDS 24

/* -------- per-device constant memory --------
 * c_chal, c_prefix kept for the self-test path and any future debug.
 * The mining kernel itself reads c_b1 (the 25 round-1 b-lane values
 * derived from challenge+prefix+padding) and c_diff. */
__constant__ uint64_t c_chal[4];
__constant__ uint64_t c_prefix[3];
__constant__ uint64_t c_diff[4];
__constant__ uint64_t c_b1[25];

__constant__ uint64_t RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL,
};

__device__ miner_result_t d_result;

/* -------- helpers -------- */

__device__ __forceinline__ uint64_t rotl64(uint64_t v, int n) {
    return (v << n) | (v >> (64 - n));
}

__device__ __forceinline__ uint64_t bswap64(uint64_t x) {
    uint32_t lo = (uint32_t)x;
    uint32_t hi = (uint32_t)(x >> 32);
    uint32_t bl = __byte_perm(lo, 0, 0x0123);
    uint32_t bh = __byte_perm(hi, 0, 0x0123);
    return ((uint64_t)bl << 32) | (uint64_t)bh;
}

__device__ __forceinline__ void keccak_round(uint64_t s[25], uint64_t rc) {
    uint64_t C0 = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20];
    uint64_t C1 = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21];
    uint64_t C2 = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22];
    uint64_t C3 = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23];
    uint64_t C4 = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24];

    uint64_t D0 = C4 ^ rotl64(C1, 1);
    uint64_t D1 = C0 ^ rotl64(C2, 1);
    uint64_t D2 = C1 ^ rotl64(C3, 1);
    uint64_t D3 = C2 ^ rotl64(C4, 1);
    uint64_t D4 = C3 ^ rotl64(C0, 1);

    /* theta + rho + pi: dest lane labels follow standard FIPS-202. */
    uint64_t b00 =          s[ 0] ^ D0;
    uint64_t b10 = rotl64(  s[ 1] ^ D1,  1);
    uint64_t b20 = rotl64(  s[ 2] ^ D2, 62);
    uint64_t b05 = rotl64(  s[ 3] ^ D3, 28);
    uint64_t b15 = rotl64(  s[ 4] ^ D4, 27);
    uint64_t b16 = rotl64(  s[ 5] ^ D0, 36);
    uint64_t b01 = rotl64(  s[ 6] ^ D1, 44);
    uint64_t b11 = rotl64(  s[ 7] ^ D2,  6);
    uint64_t b21 = rotl64(  s[ 8] ^ D3, 55);
    uint64_t b06 = rotl64(  s[ 9] ^ D4, 20);
    uint64_t b07 = rotl64(  s[10] ^ D0,  3);
    uint64_t b17 = rotl64(  s[11] ^ D1, 10);
    uint64_t b02 = rotl64(  s[12] ^ D2, 43);
    uint64_t b12 = rotl64(  s[13] ^ D3, 25);
    uint64_t b22 = rotl64(  s[14] ^ D4, 39);
    uint64_t b23 = rotl64(  s[15] ^ D0, 41);
    uint64_t b08 = rotl64(  s[16] ^ D1, 45);
    uint64_t b18 = rotl64(  s[17] ^ D2, 15);
    uint64_t b03 = rotl64(  s[18] ^ D3, 21);
    uint64_t b13 = rotl64(  s[19] ^ D4,  8);
    uint64_t b14 = rotl64(  s[20] ^ D0, 18);
    uint64_t b24 = rotl64(  s[21] ^ D1,  2);
    uint64_t b09 = rotl64(  s[22] ^ D2, 61);
    uint64_t b19 = rotl64(  s[23] ^ D3, 56);
    uint64_t b04 = rotl64(  s[24] ^ D4, 14);

    /* chi: s'[x,y] = b[x,y] ^ ((~b[x+1,y]) & b[x+2,y]) */
    s[ 0] = b00 ^ ((~b01) & b02);
    s[ 1] = b01 ^ ((~b02) & b03);
    s[ 2] = b02 ^ ((~b03) & b04);
    s[ 3] = b03 ^ ((~b04) & b00);
    s[ 4] = b04 ^ ((~b00) & b01);
    s[ 5] = b05 ^ ((~b06) & b07);
    s[ 6] = b06 ^ ((~b07) & b08);
    s[ 7] = b07 ^ ((~b08) & b09);
    s[ 8] = b08 ^ ((~b09) & b05);
    s[ 9] = b09 ^ ((~b05) & b06);
    s[10] = b10 ^ ((~b11) & b12);
    s[11] = b11 ^ ((~b12) & b13);
    s[12] = b12 ^ ((~b13) & b14);
    s[13] = b13 ^ ((~b14) & b10);
    s[14] = b14 ^ ((~b10) & b11);
    s[15] = b15 ^ ((~b16) & b17);
    s[16] = b16 ^ ((~b17) & b18);
    s[17] = b17 ^ ((~b18) & b19);
    s[18] = b18 ^ ((~b19) & b15);
    s[19] = b19 ^ ((~b15) & b16);
    s[20] = b20 ^ ((~b21) & b22);
    s[21] = b21 ^ ((~b22) & b23);
    s[22] = b22 ^ ((~b23) & b24);
    s[23] = b23 ^ ((~b24) & b20);
    s[24] = b24 ^ ((~b20) & b21);

    s[0] ^= rc;
}

__device__ __forceinline__ void keccak_f1600(uint64_t s[25]) {
    #pragma unroll
    for (int r = 0; r < KECCAK_ROUNDS; r++) {
        keccak_round(s, RC[r]);
    }
}

/* Round 0 of keccak_f1600 specialised for the mining input shape:
 * - lanes s[0..6]  are constants from the (challenge, prefix) job
 * - lane  s[7]     = bswap64(counter)       (the only variable lane)
 * - lane  s[8]     = 0x01                    (Keccak domain pad)
 * - lane  s[9..15] = 0
 * - lane  s[16]    = 0x8000000000000000      (final pad bit)
 * - lane  s[17..24]= 0
 *
 * For these inputs, 14 of the 25 post-theta-rho-pi b-lane values are
 * job-constants and 11 depend linearly on rotations of s[7]. We upload
 * the constants and "pre" values to c_b1[] from the host (see
 * miner_set_job). The kernel here just composes those with rotations of
 * s[7] and runs chi + iota for round 0.
 *
 * The 11 variable lanes and the corresponding rotation of s[7] are
 * baked into the body below. They follow from the standard FIPS-202
 * rho/pi lane map applied to (challenge||prefix||counter||pad). */
__device__ __forceinline__ void keccak_round1_fast(uint64_t s[25], uint64_t s7) {
    uint64_t b00 = c_b1[ 0];
    uint64_t b01 = c_b1[ 1] ^ rotl64(s7, 45);
    uint64_t b02 = c_b1[ 2];
    uint64_t b03 = c_b1[ 3] ^ rotl64(s7, 21);
    uint64_t b04 = c_b1[ 4];
    uint64_t b05 = c_b1[ 5] ^ rotl64(s7, 28);
    uint64_t b06 = c_b1[ 6];
    uint64_t b07 = c_b1[ 7];
    uint64_t b08 = c_b1[ 8] ^ rotl64(s7, 46);
    uint64_t b09 = c_b1[ 9];
    uint64_t b10 = c_b1[10] ^ rotl64(s7,  2);
    uint64_t b11 = c_b1[11] ^ rotl64(s7,  6);
    uint64_t b12 = c_b1[12] ^ rotl64(s7, 25);
    uint64_t b13 = c_b1[13];
    uint64_t b14 = c_b1[14];
    uint64_t b15 = c_b1[15];
    uint64_t b16 = c_b1[16];
    uint64_t b17 = c_b1[17] ^ rotl64(s7, 11);
    uint64_t b18 = c_b1[18];
    uint64_t b19 = c_b1[19] ^ rotl64(s7, 56);
    uint64_t b20 = c_b1[20];
    uint64_t b21 = c_b1[21] ^ rotl64(s7, 55);
    uint64_t b22 = c_b1[22];
    uint64_t b23 = c_b1[23];
    uint64_t b24 = c_b1[24] ^ rotl64(s7,  3);

    /* chi + iota with RC[0] = 0x1 */
    s[ 0] = (b00 ^ ((~b01) & b02)) ^ 0x0000000000000001ULL;
    s[ 1] =  b01 ^ ((~b02) & b03);
    s[ 2] =  b02 ^ ((~b03) & b04);
    s[ 3] =  b03 ^ ((~b04) & b00);
    s[ 4] =  b04 ^ ((~b00) & b01);
    s[ 5] =  b05 ^ ((~b06) & b07);
    s[ 6] =  b06 ^ ((~b07) & b08);
    s[ 7] =  b07 ^ ((~b08) & b09);
    s[ 8] =  b08 ^ ((~b09) & b05);
    s[ 9] =  b09 ^ ((~b05) & b06);
    s[10] =  b10 ^ ((~b11) & b12);
    s[11] =  b11 ^ ((~b12) & b13);
    s[12] =  b12 ^ ((~b13) & b14);
    s[13] =  b13 ^ ((~b14) & b10);
    s[14] =  b14 ^ ((~b10) & b11);
    s[15] =  b15 ^ ((~b16) & b17);
    s[16] =  b16 ^ ((~b17) & b18);
    s[17] =  b17 ^ ((~b18) & b19);
    s[18] =  b18 ^ ((~b19) & b15);
    s[19] =  b19 ^ ((~b15) & b16);
    s[20] =  b20 ^ ((~b21) & b22);
    s[21] =  b21 ^ ((~b22) & b23);
    s[22] =  b22 ^ ((~b23) & b24);
    s[23] =  b23 ^ ((~b24) & b20);
    s[24] =  b24 ^ ((~b20) & b21);
}

/* Final keccak round truncated to the 4 lanes the difficulty compare
 * needs. We still need the full 25-lane theta column sum, but rho/pi
 * for 20 lanes and chi for 21 lanes are dropped, since only s[0..3]
 * post-chi are read. iota applies only to s[0]. */
__device__ __forceinline__ void keccak_round_final(const uint64_t s[25], uint64_t out[4]) {
    uint64_t C0 = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20];
    uint64_t C1 = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21];
    uint64_t C2 = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22];
    uint64_t C3 = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23];
    uint64_t C4 = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24];

    uint64_t D0 = C4 ^ rotl64(C1, 1);
    uint64_t D1 = C0 ^ rotl64(C2, 1);
    uint64_t D2 = C1 ^ rotl64(C3, 1);
    uint64_t D3 = C2 ^ rotl64(C4, 1);
    uint64_t D4 = C3 ^ rotl64(C0, 1);

    uint64_t b0 =          s[ 0] ^ D0;
    uint64_t b1 = rotl64(  s[ 6] ^ D1, 44);
    uint64_t b2 = rotl64(  s[12] ^ D2, 43);
    uint64_t b3 = rotl64(  s[18] ^ D3, 21);
    uint64_t b4 = rotl64(  s[24] ^ D4, 14);

    out[0] = (b0 ^ ((~b1) & b2)) ^ RC[23];
    out[1] =  b1 ^ ((~b2) & b3);
    out[2] =  b2 ^ ((~b3) & b4);
    out[3] =  b3 ^ ((~b4) & b0);
}

/* -------- mining kernel -------- */

__global__ __launch_bounds__(256, 3) void mine_kernel(uint64_t base_counter, int iters_per_thread) {
    if (d_result.found) return;

    const uint64_t tid     = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    const uint64_t my_base = base_counter + tid * (uint64_t)iters_per_thread;

    const uint64_t D0 = c_diff[0];
    const uint64_t D1 = c_diff[1];
    const uint64_t D2 = c_diff[2];
    const uint64_t D3 = c_diff[3];

    for (int k = 0; k < iters_per_thread; k++) {
        const uint64_t counter = my_base + (uint64_t)k;
        const uint64_t s7      = bswap64(counter);

        uint64_t s[25];
        /* round 0 from precomputed mid-state */
        keccak_round1_fast(s, s7);

        /* rounds 1..22 (RC[1] through RC[22]) */
        #pragma unroll
        for (int r = 1; r < KECCAK_ROUNDS - 1; r++) {
            keccak_round(s, RC[r]);
        }

        /* round 23 truncated to the 4 lanes we will compare */
        uint64_t out4[4];
        keccak_round_final(s, out4);

        const uint64_t h0 = bswap64(out4[0]);

        /* Early exit: at production difficulty the leading u64 makes the
         * vast majority of hashes obviously > target. Defer h1..h3 work
         * until that test is inconclusive. */
        if (h0 > D0) continue;

        const uint64_t h1 = bswap64(out4[1]);
        const uint64_t h2 = bswap64(out4[2]);
        const uint64_t h3 = bswap64(out4[3]);

        bool hit;
        if      (h0 < D0) hit = true;
        else if (h1 < D1) hit = true;
        else if (h1 > D1) hit = false;
        else if (h2 < D2) hit = true;
        else if (h2 > D2) hit = false;
        else              hit = (h3 < D3);

        if (hit) {
            if (atomicCAS(&d_result.found, 0u, 1u) == 0u) {
                d_result.counter    = counter;
                d_result.hash_be[0] = h0;
                d_result.hash_be[1] = h1;
                d_result.hash_be[2] = h2;
                d_result.hash_be[3] = h3;
            }
            return;
        }
    }
}

/* -------- self-test kernel: one thread hashes a literal 64-byte buffer -------- */
__global__ void self_test_kernel(const uint8_t* in64, uint8_t* out32) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint64_t s[25];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint64_t v = 0;
        #pragma unroll
        for (int b = 0; b < 8; b++) {
            v |= ((uint64_t)in64[i * 8 + b]) << (b * 8);
        }
        s[i] = v;
    }
    s[ 8] = 0x0000000000000001ULL;
    s[ 9] = 0; s[10] = 0; s[11] = 0; s[12] = 0;
    s[13] = 0; s[14] = 0; s[15] = 0;
    s[16] = 0x8000000000000000ULL;
    s[17] = 0; s[18] = 0; s[19] = 0; s[20] = 0;
    s[21] = 0; s[22] = 0; s[23] = 0; s[24] = 0;

    keccak_f1600(s);

    /* Lanes serialise little-endian byte-by-byte to produce the keccak
       digest in its natural byte order (which is what eth.utils.keccak256
       returns and what bytes32 stores). */
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint64_t v = s[i];
        #pragma unroll
        for (int b = 0; b < 8; b++) {
            out32[i * 8 + b] = (uint8_t)(v >> (b * 8));
        }
    }
}

/* -------- host-side context -------- */

struct miner_ctx {
    int device_id;
    cudaStream_t stream;
    miner_result_t* h_pinned;
    int sm_count;
    int max_threads_per_block;
};

#define CK(expr) do { cudaError_t _e = (expr); if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #expr, __FILE__, __LINE__, cudaGetErrorString(_e)); \
    return -(int)_e; } } while (0)

#define CK_NULL(expr) do { cudaError_t _e = (expr); if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #expr, __FILE__, __LINE__, cudaGetErrorString(_e)); \
    return NULL; } } while (0)

extern "C" int miner_device_count(void) {
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess) return -(int)e;
    return n;
}

extern "C" int miner_device_info(int device_id, miner_device_info_t* out) {
    if (!out) return -1;
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, device_id));
    out->device_id              = device_id;
    out->compute_major          = p.major;
    out->compute_minor          = p.minor;
    out->sm_count               = p.multiProcessorCount;
    out->max_threads_per_block  = p.maxThreadsPerBlock;
    out->warp_size              = p.warpSize;
    out->total_memory_bytes     = (unsigned long long)p.totalGlobalMem;
    strncpy(out->name, p.name, sizeof(out->name) - 1);
    out->name[sizeof(out->name) - 1] = '\0';
    return 0;
}

extern "C" miner_ctx_t* miner_create(int device_id) {
    CK_NULL(cudaSetDevice(device_id));

    miner_ctx_t* c = (miner_ctx_t*)calloc(1, sizeof(miner_ctx_t));
    if (!c) return NULL;
    c->device_id = device_id;

    cudaDeviceProp p;
    CK_NULL(cudaGetDeviceProperties(&p, device_id));
    c->sm_count              = p.multiProcessorCount;
    c->max_threads_per_block = p.maxThreadsPerBlock;

    CK_NULL(cudaStreamCreateWithFlags(&c->stream, cudaStreamNonBlocking));
    CK_NULL(cudaHostAlloc((void**)&c->h_pinned, sizeof(miner_result_t),
                          cudaHostAllocDefault));
    return c;
}

extern "C" void miner_destroy(miner_ctx_t* c) {
    if (!c) return;
    cudaSetDevice(c->device_id);
    if (c->h_pinned) cudaFreeHost(c->h_pinned);
    if (c->stream)   cudaStreamDestroy(c->stream);
    free(c);
}

/* Convert 32 BE bytes (e.g. challenge from getChallenge) into 4 lane u64s.
 * Each lane i is the LE u64 read of input bytes [i*8..(i+1)*8). On x86 LE
 * hosts this is just a memcpy, but we do it byte-by-byte for portability. */
static void be32_to_lanes_le(const uint8_t in[32], uint64_t out[4]) {
    for (int i = 0; i < 4; i++) {
        uint64_t v = 0;
        for (int b = 0; b < 8; b++) {
            v |= ((uint64_t)in[i * 8 + b]) << (b * 8);
        }
        out[i] = v;
    }
}

/* Convert 24 BE bytes (nonce prefix) into 3 lane u64s. */
static void be24_to_lanes_le(const uint8_t in[24], uint64_t out[3]) {
    for (int i = 0; i < 3; i++) {
        uint64_t v = 0;
        for (int b = 0; b < 8; b++) {
            v |= ((uint64_t)in[i * 8 + b]) << (b * 8);
        }
        out[i] = v;
    }
}

/* Convert 32 BE bytes (difficulty) into 4 BE u64 chunks suitable for
 * lexicographic MSB-first compare against bswap64(lane). diff[0] holds
 * the MSB-most 8 bytes. */
static void be32_to_u64_be(const uint8_t in[32], uint64_t out[4]) {
    for (int i = 0; i < 4; i++) {
        uint64_t v = 0;
        for (int b = 0; b < 8; b++) {
            v = (v << 8) | (uint64_t)in[i * 8 + b];
        }
        out[i] = v;
    }
}

/* Host-side u64 left-rotate, mirrors the device rotl64 semantics. */
static inline uint64_t rotl64_h(uint64_t v, int n) {
    return (v << n) | (v >> (64 - n));
}

/* Build the 25 round-0 "b" lane values for the keccak_round1_fast path,
 * given the challenge and nonce prefix as lane vectors.
 *
 * For the 14 lanes whose value does not depend on the counter, we store
 * the full value. For the 11 lanes that depend linearly on rotl(s7,k)
 * for various k, we store the constant "pre" half (the kernel XORs in
 * rotl(s7,k) at runtime). The choice of which lane is which is fixed by
 * the FIPS-202 rho/pi map plus our input layout; see the device-side
 * keccak_round1_fast for the rotation amounts. */
static void build_round1_table(const uint64_t L[7], uint64_t out[25]) {
    const uint64_t L0 = L[0], L1 = L[1], L2 = L[2], L3 = L[3];
    const uint64_t L4 = L[4], L5 = L[5], L6 = L[6];
    const uint64_t PAD_LO  = 0x0000000000000001ULL;
    const uint64_t PAD_HI  = 0x8000000000000000ULL;

    const uint64_t C0 = L0 ^ L5;
    const uint64_t C1 = L1 ^ L6 ^ PAD_HI;
    const uint64_t C3 = L3 ^ PAD_LO;
    const uint64_t C4 = L4;

    const uint64_t D0 = C4 ^ rotl64_h(C1, 1);
    const uint64_t D2 = C1 ^ rotl64_h(C3, 1);
    const uint64_t D4 = C3 ^ rotl64_h(C0, 1);
    const uint64_t D1 = C0 ^ rotl64_h(L2, 1);  /* constant part; s7 part on GPU */
    const uint64_t D3 = L2 ^ rotl64_h(C4, 1);  /* constant part; s7 part on GPU */

    out[ 0] = L0 ^ D0;
    out[ 1] = rotl64_h(L6 ^ D1, 44);
    out[ 2] = rotl64_h(D2, 43);
    out[ 3] = rotl64_h(D3, 21);
    out[ 4] = rotl64_h(D4, 14);
    out[ 5] = rotl64_h(L3 ^ D3, 28);
    out[ 6] = rotl64_h(D4, 20);
    out[ 7] = rotl64_h(D0, 3);
    out[ 8] = rotl64_h(PAD_HI ^ D1, 45);
    out[ 9] = rotl64_h(D2, 61);
    out[10] = rotl64_h(L1 ^ D1, 1);
    out[11] = rotl64_h(D2, 6);
    out[12] = rotl64_h(D3, 25);
    out[13] = rotl64_h(D4, 8);
    out[14] = rotl64_h(D0, 18);
    out[15] = rotl64_h(L4 ^ D4, 27);
    out[16] = rotl64_h(L5 ^ D0, 36);
    out[17] = rotl64_h(D1, 10);
    out[18] = rotl64_h(D2, 15);
    out[19] = rotl64_h(D3, 56);
    out[20] = rotl64_h(L2 ^ D2, 62);
    out[21] = rotl64_h(PAD_LO ^ D3, 55);
    out[22] = rotl64_h(D4, 39);
    out[23] = rotl64_h(D0, 41);
    out[24] = rotl64_h(D1, 2);
}

extern "C" int miner_set_job(miner_ctx_t* c,
                              const uint8_t challenge[32],
                              const uint8_t difficulty[32],
                              const uint8_t nonce_prefix[24])
{
    if (!c || !challenge || !difficulty || !nonce_prefix) return -1;
    CK(cudaSetDevice(c->device_id));

    uint64_t chal_lanes[4];
    uint64_t prefix_lanes[3];
    uint64_t diff_be[4];

    be32_to_lanes_le(challenge, chal_lanes);
    be24_to_lanes_le(nonce_prefix, prefix_lanes);
    be32_to_u64_be(difficulty, diff_be);

    uint64_t lanes[7] = {
        chal_lanes[0], chal_lanes[1], chal_lanes[2], chal_lanes[3],
        prefix_lanes[0], prefix_lanes[1], prefix_lanes[2],
    };
    uint64_t b1_table[25];
    build_round1_table(lanes, b1_table);

    CK(cudaMemcpyToSymbolAsync(c_chal,   chal_lanes,   sizeof(chal_lanes),
                               0, cudaMemcpyHostToDevice, c->stream));
    CK(cudaMemcpyToSymbolAsync(c_prefix, prefix_lanes, sizeof(prefix_lanes),
                               0, cudaMemcpyHostToDevice, c->stream));
    CK(cudaMemcpyToSymbolAsync(c_diff,   diff_be,      sizeof(diff_be),
                               0, cudaMemcpyHostToDevice, c->stream));
    CK(cudaMemcpyToSymbolAsync(c_b1,     b1_table,     sizeof(b1_table),
                               0, cudaMemcpyHostToDevice, c->stream));
    CK(cudaStreamSynchronize(c->stream));
    return 0;
}

extern "C" int miner_recommended_launch(miner_ctx_t* c,
                                         int* out_blocks,
                                         int* out_tpb,
                                         int* out_iters)
{
    if (!c || !out_blocks || !out_tpb || !out_iters) return -1;
    /* Empirically tuned on RTX 2080 Ti (sm_75) with the round-1
     * precomputed kernel: TPB=256 wins consistently over 64 and 128,
     * and oversubscribing the SMs ~64x gives the warp scheduler enough
     * resident blocks to hide the per-iter LD.E for c_b1. Each launch
     * works out to roughly 30-50 ms at 1.5+ GH/s, which keeps challenge
     * rotation latency low for the production loop. */
    *out_tpb    = 256;
    *out_blocks = c->sm_count * 64;
    *out_iters  = 64;
    return 0;
}

extern "C" int miner_search(miner_ctx_t* c,
                             uint64_t base_counter,
                             int blocks,
                             int threads_per_block,
                             int iters_per_thread,
                             miner_result_t* out,
                             uint64_t* attempted)
{
    if (!c || !out) return -1;
    CK(cudaSetDevice(c->device_id));

    /* Reset the result block. */
    miner_result_t zero;
    memset(&zero, 0, sizeof(zero));
    CK(cudaMemcpyToSymbolAsync(d_result, &zero, sizeof(zero), 0,
                               cudaMemcpyHostToDevice, c->stream));

    dim3 grid(blocks, 1, 1);
    dim3 block(threads_per_block, 1, 1);
    mine_kernel<<<grid, block, 0, c->stream>>>(base_counter, iters_per_thread);

    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) {
        fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(le));
        return -(int)le;
    }

    CK(cudaMemcpyFromSymbolAsync(c->h_pinned, d_result,
                                 sizeof(miner_result_t), 0,
                                 cudaMemcpyDeviceToHost, c->stream));
    CK(cudaStreamSynchronize(c->stream));

    *out = *c->h_pinned;
    if (attempted) {
        *attempted = (uint64_t)blocks
                   * (uint64_t)threads_per_block
                   * (uint64_t)iters_per_thread;
    }
    return 0;
}

extern "C" int miner_self_test(miner_ctx_t* c,
                                const uint8_t input_64[64],
                                uint8_t out_hash[32])
{
    if (!c || !input_64 || !out_hash) return -1;
    CK(cudaSetDevice(c->device_id));

    uint8_t* d_in  = NULL;
    uint8_t* d_out = NULL;
    CK(cudaMalloc((void**)&d_in,  64));
    CK(cudaMalloc((void**)&d_out, 32));

    CK(cudaMemcpyAsync(d_in, input_64, 64, cudaMemcpyHostToDevice, c->stream));
    self_test_kernel<<<1, 1, 0, c->stream>>>(d_in, d_out);
    CK(cudaGetLastError());
    CK(cudaMemcpyAsync(out_hash, d_out, 32, cudaMemcpyDeviceToHost, c->stream));
    CK(cudaStreamSynchronize(c->stream));

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
