# hash-miner

CUDA + Rust off-chain miner for the `$HASH` ERC-20 PoW protocol
(`0xAC7b5d06fa1e77D08aea40d46cB7C5923A87A0cc` on Ethereum mainnet).

The kernel ports the protocol's WGSL reference shader to native CUDA
with `uint64_t` lanes, with a host-side mid-state for round 0 and a
truncated final round. The host is Rust on top of `alloy` 2.x, with
`keccak-asm` for CPU keccak verification.

## Stack

- **Rust 1.91+**, `tokio` async runtime.
- **alloy 2.x** for RPC, `sol!` bindings, EIP-1559 tx signing.
- **keccak-asm** (XKCP asm-optimised) for host keccak.
- **CUDA 13.0** kernel, compiled with `nvcc -O3 --use_fast_math
  -maxrregcount=80` and fat-binary'd for `sm_75;sm_86;sm_89;sm_90`.

## Benchmark (reference machine)

`2 x NVIDIA GeForce RTX 2080 Ti, cc 7.5, 68 SMs each`:

```
$ make bench
bench: 9.5e10 hashes in 31.0s = 3.07 GH/s
```

~1.55 GH/s per RTX 2080 Ti. Modern cards scale roughly linearly with
CUDA-core count; expect ~3.3 GH/s on a 3080, ~5.5 GH/s on a 4070,
~11 GH/s on a 4090.

---

## Quick start

```
make setup                                  # build + install /usr/local/bin/hash-miner
make wallet                                 # generate an address + private key
cp config.example.toml config.toml          # then edit RPC + private_key_env
export MINER1_PRIVKEY=0x<your private key>  # the value from `make wallet`
make mine                                   # start mining
```

The miner reads its signing key from the env var named in
`[wallet].private_key_env`. There is no on-disk wallet file.

---

## Setup (in detail)

### 1. Install the toolchain

You need a recent Rust toolchain and a CUDA install (toolkit, not just
the driver). The build script invokes `nvcc` and links `libcudart`.

```
# Rust (Linux/macOS)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
. "$HOME/.cargo/env"
rustup toolchain install stable

# CUDA: install via your distro's package manager or from
# https://developer.nvidia.com/cuda-downloads. Tested against CUDA 13.0.
nvcc --version
nvidia-smi
```

If your CUDA lives outside `/usr/local/cuda`, point at it explicitly:

```
export CUDA_PATH=/opt/cuda-13.0
export CUDA_HOME=/opt/cuda-13.0
```

### 2. Build and install

```
make setup
```

This is equivalent to `make release` then `make install`, which
symlinks the release binary into `$(PREFIX)/bin/hash-miner` (default
prefix is `/usr/local`; override with `PREFIX=$HOME/.local make
install` for a userland install).

Override target GPU SMs for your build (the default covers Turing
through Hopper): `CUDA_ARCH=89` for Ada (4070/4080/4090),
`CUDA_ARCH=86` for Ampere (3080/3090), `CUDA_ARCH=75` for Turing
(2080/2080 Ti). Pass it through the cargo invocation:

```
CUDA_ARCH=89 make release
```

### 3. Verify the GPU

```
make devices
make selftest
```

`selftest` runs three vectors through the CUDA kernel and verifies the
output is bit-identical to `keccak-asm`. If this fails, do not proceed
- the CUDA build is broken and any "hits" would not pass on-chain
verification.

### 4. Generate a wallet

```
make wallet
```

Output:

```
address:     0x95E7DE2eC3fD90Aa015cf55e3AD592e2B3222FAb
private key: 0xb6a38d752ce7c4b8664cf3f1a5559bd9c0e537082489dd4976d51d323c08f29d
```

Save the private key somewhere only you can read (password manager,
secrets vault, encrypted note). Anyone with this key can drain the
address.

Fund the printed address with enough ETH to cover gas on a few mints;
`min_balance_eth` in `[economics]` is the floor below which the miner
refuses to submit.

### 5. Configure

```
cp config.example.toml config.toml
$EDITOR config.toml
```

Required edits:

- `chain.read_rpc` - websocket preferred for sub-second event
  subscribe. Alchemy / Infura / your own node all work.
- `chain.submit_rpcs` - private orderflow only. The defaults
  (`https://rpc.mevblocker.io/fast` and `https://rpc.flashbots.net/fast`)
  are fine.
- `chain.contract` - keep the mainnet $HASH address.
- `wallet.private_key_env` - name of the env var that will hold the
  hex private key (default `MINER1_PRIVKEY`).
- `mine.cuda_devices` - leave empty for "use all visible GPUs", or
  list device indices: `[0, 1]`.
- `submit.priority_tip_gwei` - start at 6, EMA-adjusted vs other miners.

### 6. Run

Export the key the config refers to, then start the miner:

```
export MINER1_PRIVKEY=0xb6a38d752ce7c4b8664cf3f1a5559bd9c0e537082489dd4976d51d323c08f29d
make mine
```

For a persistent setup, put the `export` in your shell rc or a systemd
unit's `Environment=` line.

The startup log looks like:

```
INFO hash-miner starting version="0.1.0"
INFO wallet miner=0x252CAED5983d2251D2834d82FeAB166393CfCd1e
INFO contract addr=0xAC7b5d06fa1e77D08aea40d46cB7C5923A87A0cc chain_id=1
INFO read rpc rpc=wss://...
INFO submit relays endpoints=2
INFO gpus count=2 ids=[0, 1]
INFO job epoch=250719 era=0 diff=4.11e62 reward=100.0000
INFO gpu online gpu=0 name="NVIDIA GeForce RTX 2080 Ti" cc=7.5 sm=68
INFO gpu online gpu=1 name="NVIDIA GeForce RTX 2080 Ti" cc=7.5 sm=68
```

Then every 10 s, the rate line:

```
INFO gpu0=1.55 GH/s gpu1=1.55 GH/s | total 3.07 GH/s | avg 3.05 GH/s | up 1m20s | hits 0 | mints 0
```

On a hit / accepted mint:

```
INFO share found gpu=0 epoch=250719 counter=0x... hash=0xab12cd34...
INFO submit tx=0x... tip=6.00gwei cap=42.30gwei
INFO accepted block=22500201 reward=100.0000
```

---

## Make targets

| Target              | What it does                                                          |
|---------------------|-----------------------------------------------------------------------|
| `make build`        | Debug build (`cargo build`).                                          |
| `make release`      | Optimized build (`cargo build --release`). Default target.            |
| `make install`      | Symlink `$(PREFIX)/bin/hash-miner` -> release binary.                 |
| `make uninstall`    | Remove the symlink.                                                   |
| `make setup`        | `release` + `install` (one-shot first-time setup).                    |
| `make wallet`       | Generate a new address + private key.                                 |
| `make selftest`     | Verify the CUDA kernel against host keccak (3 vectors).               |
| `make bench`        | Run a 30 s hash-rate benchmark across all visible GPUs.               |
| `make devices`      | List visible CUDA devices + recommended launch shape.                 |
| `make mine`         | Start the miner. Requires `$(PRIVKEY_ENV)` exported and config.toml.  |
| `make clean`        | `cargo clean`.                                                        |
| `make help`         | Print the target table.                                               |

Overridable variables:

- `PREFIX` (default `/usr/local`) - install prefix.
- `BIN_DIR` (default `$(PREFIX)/bin`) - symlink directory.
- `CARGO` (default `cargo`) - cargo binary to use.
- `CONFIG` (default `./config.toml`) - which config to load on `make mine`.
- `PRIVKEY_ENV` (default `MINER1_PRIVKEY`) - env var checked by `make mine`.
- `BENCH_SECS` (default `30`) - duration of `make bench`.
- `BENCH_ITERS` (default `64`) - iters-per-thread passed to bench.

Examples:

```
PREFIX=$HOME/.local make install     # userland install
PRIVKEY_ENV=MINER2_PRIVKEY make mine # second wallet
BENCH_SECS=60 make bench             # longer bench
```

---

## Direct CLI

The make targets are thin wrappers. Once installed you can call the
binary directly:

```
hash-miner devices
hash-miner selftest --device 0
hash-miner bench --device 0 --device 1 --seconds 30 --iters 64
hash-miner wallet new
hash-miner mine --config ./config.toml
```

### Bench tunables

The recommended launch (TPB=256, blocks=64*SM_count, iters=64) was
tuned for sm_75. Sweep launch parameters without recompiling:

```
hash-miner bench --device 0 --tpb 256 --blocks 8704 --iters 128 --seconds 30
```

`--iters` controls how many counters each thread tries per kernel
launch. Higher = better steady-state throughput but slower reaction to
challenge rotation. 64-128 is a good range for production.

`--difficulty-hex <32-byte hex>` overrides the dummy difficulty if you
want to benchmark against a production-like target.

---

## Configuration reference

See `config.example.toml`. Required fields:

- `chain.read_rpc` - websocket preferred for sub-second event subscribe.
- `chain.submit_rpcs` - private orderflow only.
- `chain.contract` - mainnet $HASH contract.
- `wallet.private_key_env` - name of the env var holding the hex
  private key. Export the key before running.
- `mine.cuda_devices` - empty = all visible.
- `submit.priority_tip_gwei` - start at 6, EMA-adjusted vs other miners.

---

## Layout

```
cuda/
  miner.h        extern "C" surface (Rust calls this)
  kernel.cu      Keccak-f[1600] kernel + host C API
src/
  main.rs        CLI
  config.rs      TOML schema (Config, ChainConfig, WalletConfig, ...)
  chain.rs       alloy sol! bindings + keccak_host + custom error decode
  wallet.rs      load_signer(): env-var hex private key
  cuda.rs        FFI bindings + CudaContext + list_devices()
  cpu.rs         CPU fallback miner using keccak-asm
  submit.rs      EIP-1559 signing + race-submission to private RPCs
  coordinator.rs top-level state machine
  metrics.rs     per-device hash-rate tracking + log formatters
  globals.rs     shared defaults
  types.rs       shared value types (Job, Hit, StopFlag, ...)
build.rs         nvcc invocation
Makefile         make targets (setup / install / wallet / bench / mine / ...)
```

## Submission flow (mine)

1. Worker hit -> `coordinator.handle_hit`
2. Host re-verifies with keccak-asm (`verify_hit`)
3. Pre-flight `getChallenge(myAddr)` recheck (drop on rotation)
4. Build EIP-1559 tx: `maxFee = baseFee * 3 + tip`, `gasLimit` from
   config range
5. Sign locally with `PrivateKeySigner`
6. Race POST `eth_sendRawTransaction` across all `submit_rpcs`
7. Watch for the receipt; log success or decoded revert reason

Custom error selectors decoded: `InsufficientWork`, `ProofAlreadyUsed`,
`BlockCapReached`, `SupplyExhausted`, `GenesisNotComplete`.

## Operational notes

- The private key in the env var is the wallet. Anyone who can read
  the process environment can drain the funded address. Use shell rc,
  systemd `Environment=`, or a secrets manager rather than a literal
  `export` in shared history.
- Per-device thread owns the CUDA context (no cross-device sharing).
- Each device gets a unique 24-byte nonce prefix so two devices never
  search the same `(prefix, counter)`.
- Epoch rotation triggers a `cudaMemcpyToSymbol` of the new challenge
  and difficulty; in-flight counters are dropped.
- The miner refuses to start if `genesisComplete()` is false or if the
  RPC's `chain_id` disagrees with `config.chain.chain_id`.
