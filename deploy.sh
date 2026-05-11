#!/usr/bin/env bash
# deploy.sh: one-shot setup for hash-miner on a fresh CUDA Linux box.
# Usage: bash deploy.sh <RPC_WSS_URL> <PRIVATE_KEY_HEX>
# Example: bash deploy.sh wss://eth-mainnet.g.alchemy.com/v2/abc123 0xdeadbeef...

set -euo pipefail

RPC_URL="${1:-}"
PRIVKEY="${2:-}"

if [[ -z "$RPC_URL" || -z "$PRIVKEY" ]]; then
    echo "Usage: bash deploy.sh <wss://rpc-url> <0xprivate-key>"
    exit 1
fi

# ── 1. Rust ────────────────────────────────────────────────────────────────────
if ! command -v cargo &>/dev/null; then
    echo "[1/5] Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
else
    echo "[1/5] Rust already installed ($(rustc --version))"
    source "$HOME/.cargo/env" 2>/dev/null || true
fi

# ── 2. CUDA check ──────────────────────────────────────────────────────────────
echo "[2/5] Checking CUDA..."
if ! command -v nvcc &>/dev/null; then
    echo "ERROR: nvcc not found. Install CUDA toolkit first:"
    echo "  https://developer.nvidia.com/cuda-downloads"
    echo "  Or on Ubuntu: apt install nvidia-cuda-toolkit"
    exit 1
fi
nvcc --version | head -1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── 3. Clone / update repo ────────────────────────────────────────────────────
echo "[3/5] Cloning repo..."
REPO_DIR="$HOME/hash-mining"
if [[ -d "$REPO_DIR/.git" ]]; then
    git -C "$REPO_DIR" pull
else
    git clone https://github.com/Costimz/hash-mining.git "$REPO_DIR"
fi
cd "$REPO_DIR"

# ── 4. Build with CUDA ────────────────────────────────────────────────────────
echo "[4/5] Building (this takes a few minutes)..."
# Detect GPU architecture and compile for it only (faster build)
GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
echo "  GPU compute capability: $GPU_ARCH"
CUDA_ARCH="$GPU_ARCH" cargo build --release 2>&1

echo "  Build complete: $(./target/release/hash-miner --help 2>&1 | head -1)"

# ── 5. Write config ───────────────────────────────────────────────────────────
echo "[5/5] Writing config..."
cat > config.toml <<TOML
[chain]
read_rpc    = "$RPC_URL"
submit_rpcs = [
    "https://rpc.mevblocker.io/fast",
    "https://rpc.flashbots.net/fast",
]
chain_id = 1
contract = "0xAC7b5d06fa1e77D08aea40d46cB7C5923A87A0cc"

[wallet]
private_key_env = "MINER_PRIVKEY"

[mine]
backend          = "cuda"
cuda_devices     = []
batch_size       = 64
poll_interval_ms = 2000
event_subscribe  = true

[submit]
priority_tip_gwei         = 6.0
max_priority_tip_gwei     = 25.0
base_fee_multiplier       = 3
gas_min                   = 200000
gas_max                   = 400000
gas_estimate_timeout_ms   = 4000
submit_timeout_ms         = 12000

[economics]
min_hash_price_usd   = 0.0
gas_budget_per_hour_usd = 0.0
min_balance_eth      = 0.05
TOML

# ── 6. Create start script ────────────────────────────────────────────────────
cat > start-miner.sh <<'SCRIPT'
#!/usr/bin/env bash
cd "$(dirname "$0")"
source "$HOME/.cargo/env" 2>/dev/null || true
export MINER_PRIVKEY="${MINER_PRIVKEY:?set MINER_PRIVKEY before running}"
exec ./target/release/hash-miner mine --config config.toml
SCRIPT
chmod +x start-miner.sh

echo ""
echo "════════════════════════════════════════════════════════"
echo " Setup complete!"
echo ""
echo " To start mining:"
echo "   export MINER_PRIVKEY=$PRIVKEY"
echo "   bash $REPO_DIR/start-miner.sh"
echo ""
echo " To run in the background (tmux):"
echo "   tmux new -s miner"
echo "   export MINER_PRIVKEY=$PRIVKEY"
echo "   bash $REPO_DIR/start-miner.sh"
echo "   # Ctrl+B then D to detach"
echo "════════════════════════════════════════════════════════"
