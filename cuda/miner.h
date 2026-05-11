/**
 * @file miner.h
 * @brief Extern C surface that Rust calls into for CUDA-side keccak mining.
 */

#ifndef HASH_MINER_CUDA_H
#define HASH_MINER_CUDA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Opaque per-device miner context.
 */
typedef struct miner_ctx miner_ctx_t;

/**
 * Result block written by the kernel when a hit is found.
 *  - found:    non-zero once the kernel has written a hit
 *  - counter:  u64 counter that produced the hit
 *  - hash_be:  the 32-byte keccak result in big-endian u64 chunks
 */
typedef struct {
    uint32_t found;
    uint32_t _pad;
    uint64_t counter;
    uint64_t hash_be[4];
} miner_result_t;

/**
 * Properties of a CUDA device, populated by miner_device_info.
 */
typedef struct {
    int device_id;
    int compute_major;
    int compute_minor;
    int sm_count;
    int max_threads_per_block;
    int warp_size;
    char name[256];
    unsigned long long total_memory_bytes;
} miner_device_info_t;

/**
 * Return number of CUDA devices visible, or negative on error.
 */
int miner_device_count(void);

/**
 * Fill info for device_id. Returns 0 on success, negative on error.
 */
int miner_device_info(int device_id, miner_device_info_t* out);

/**
 * Create a context bound to device_id. The context owns:
 *  - a result buffer in device memory
 *  - a host-pinned mirror for fast readback
 *  - a default stream
 * Returns NULL on failure.
 */
miner_ctx_t* miner_create(int device_id);

/**
 * Destroy a context, free all resources.
 */
void miner_destroy(miner_ctx_t* ctx);

/**
 * Upload a new (challenge, difficulty, nonce_prefix) tuple to constant
 * memory. challenge and difficulty are 32-byte big-endian, nonce_prefix
 * is 24 bytes. Returns 0 on success.
 */
int miner_set_job(miner_ctx_t* ctx,
                  const uint8_t challenge[32],
                  const uint8_t difficulty[32],
                  const uint8_t nonce_prefix[24]);

/**
 * Run one search wave starting at base_counter. blocks * threads_per_block
 * threads each try iters_per_thread counters in sequence.
 *
 * On success returns total hashes attempted in *attempted (always
 * blocks*threads*iters even on hit, so caller can use it for hashrate).
 *
 * If a hit was found, *out has out->found != 0 and the counter / hash_be
 * filled in. Otherwise out->found == 0.
 *
 * Returns 0 on success, negative on CUDA error.
 */
int miner_search(miner_ctx_t* ctx,
                 uint64_t base_counter,
                 int blocks,
                 int threads_per_block,
                 int iters_per_thread,
                 miner_result_t* out,
                 uint64_t* attempted);

/**
 * Recommended launch parameters for the bound device. Caller may override.
 */
int miner_recommended_launch(miner_ctx_t* ctx,
                             int* out_blocks,
                             int* out_threads_per_block,
                             int* out_iters_per_thread);

/**
 * Run the keccak permutation on a single 64-byte input. Used by self-test
 * to verify the kernel matches a host-side reference. The 32-byte hash
 * is written to out_hash in big-endian.
 */
int miner_self_test(miner_ctx_t* ctx,
                    const uint8_t input_64[64],
                    uint8_t out_hash[32]);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* HASH_MINER_CUDA_H */
