/*!
 * chain.rs: Contract bindings and chain helpers via Alloy. We expose a
 * `ChainClient` that wraps a (read provider, signer wallet) pair and gives
 * us the few view calls plus the encoded `mine(nonce)` calldata we need.
 *
 * Submission deliberately does NOT use a normal alloy contract instance:
 * we want to send raw signed transactions to private orderflow endpoints,
 * which means we sign locally and POST the rlp-encoded bytes via plain
 * eth_sendRawTransaction.
 */

use alloy::primitives::{Address, U256};
use alloy::providers::{DynProvider, Provider, ProviderBuilder};
use alloy::sol;
use alloy::sol_types::{SolCall, SolError};
use anyhow::{anyhow, Context, Result};
use std::str::FromStr;
use std::sync::Arc;

sol! {
    #[sol(rpc)]
    interface Hash {
        function getChallenge(address miner) external view returns (bytes32);
        function currentDifficulty() external view returns (uint256);
        function currentReward() external view returns (uint256);
        function epochBlocksLeft() external view returns (uint256);
        function miningState() external view returns (
            uint256 era,
            uint256 reward,
            uint256 difficulty,
            uint256 minted,
            uint256 remaining,
            uint256 epoch,
            uint256 epochBlocksLeft
        );
        function balanceOf(address account) external view returns (uint256);
        function mine(uint256 nonce) external;
        function genesisComplete() external view returns (bool);
        function mintsInBlock(uint256 blockNumber) external view returns (uint256);
        function usedProofs(bytes32 key) external view returns (bool);
        function totalMints() external view returns (uint256);
        function totalMiningMinted() external view returns (uint256);

        event Mined(address indexed miner, uint256 nonce, uint256 reward, uint256 era);
        event DifficultyAdjusted(uint256 from, uint256 to, uint256 blocksTaken);
        event Halving(uint256 era, uint256 newReward);

        error InsufficientWork();
        error ProofAlreadyUsed();
        error BlockCapReached();
        error SupplyExhausted();
        error GenesisNotComplete();
    }
}

pub struct ReadClient {
    pub provider: Arc<DynProvider>,
    pub contract: Address,
    pub miner: Address,
    pub chain_id: u64,
}

impl ReadClient {
    pub async fn connect(rpc_url: &str, contract: Address, miner: Address) -> Result<Self> {
        let url = rpc_url.to_string();
        let provider: DynProvider = if url.starts_with("ws") {
            let ws = alloy::providers::WsConnect::new(url.clone());
            ProviderBuilder::new().connect_ws(ws).await
                .with_context(|| format!("ws connect to {url} failed"))?
                .erased()
        } else {
            ProviderBuilder::new()
                .connect_http(url.parse().context("invalid http rpc url")?)
                .erased()
        };
        let chain_id = provider.get_chain_id().await
            .context("eth_chainId failed")?;
        Ok(Self {
            provider: Arc::new(provider),
            contract,
            miner,
            chain_id,
        })
    }

    pub async fn challenge(&self) -> Result<[u8; 32]> {
        let c = Hash::new(self.contract, self.provider.as_ref());
        let r = c.getChallenge(self.miner).call().await?;
        Ok(r.0)
    }

    pub async fn current_difficulty(&self) -> Result<U256> {
        let c = Hash::new(self.contract, self.provider.as_ref());
        Ok(c.currentDifficulty().call().await?)
    }

    pub async fn current_reward(&self) -> Result<U256> {
        let c = Hash::new(self.contract, self.provider.as_ref());
        Ok(c.currentReward().call().await?)
    }

    pub async fn epoch_blocks_left(&self) -> Result<u64> {
        let c = Hash::new(self.contract, self.provider.as_ref());
        let r = c.epochBlocksLeft().call().await?;
        Ok(r.try_into().unwrap_or(0))
    }

    pub async fn balance_eth(&self, who: Address) -> Result<U256> {
        Ok(self.provider.get_balance(who).await?)
    }

    pub async fn balance_hash(&self) -> Result<U256> {
        let c = Hash::new(self.contract, self.provider.as_ref());
        Ok(c.balanceOf(self.miner).call().await?)
    }

    pub async fn genesis_complete(&self) -> Result<bool> {
        let c = Hash::new(self.contract, self.provider.as_ref());
        Ok(c.genesisComplete().call().await?)
    }

    pub async fn mining_state(&self) -> Result<MiningState> {
        let c = Hash::new(self.contract, self.provider.as_ref());
        let r = c.miningState().call().await?;
        Ok(MiningState {
            era: r.era,
            reward: r.reward,
            difficulty: r.difficulty,
            minted: r.minted,
            remaining: r.remaining,
            epoch: r.epoch.try_into().unwrap_or(0),
            epoch_blocks_left: r.epochBlocksLeft.try_into().unwrap_or(0),
        })
    }

    pub async fn block_number(&self) -> Result<u64> {
        Ok(self.provider.get_block_number().await?)
    }
}

#[derive(Debug, Clone)]
pub struct MiningState {
    pub era: U256,
    pub reward: U256,
    pub difficulty: U256,
    pub minted: U256,
    pub remaining: U256,
    pub epoch: u64,
    pub epoch_blocks_left: u64,
}

/* Compute the challenge locally from the contract formula:
 *   keccak256(abi.encode(uint256(chainId), address(contract), miner, epoch))
 *
 * Used to pre-warm caches before epoch rotation. */
pub fn compute_challenge(chain_id: u64, contract: Address, miner: Address, epoch: u64) -> [u8; 32] {
    let mut buf = [0u8; 128];
    let cid = U256::from(chain_id);
    buf[..32].copy_from_slice(&cid.to_be_bytes::<32>());
    buf[44..64].copy_from_slice(contract.as_slice());
    buf[76..96].copy_from_slice(miner.as_slice());
    let ep = U256::from(epoch);
    buf[96..128].copy_from_slice(&ep.to_be_bytes::<32>());
    keccak_host(&buf)
}

/* Fastest CPU keccak path (XKCP optimised ASM via keccak-asm). Used to
 * verify hits before paying gas and to compute challenges locally for
 * cache pre-warming around epoch rotations. */
pub fn keccak_host(data: &[u8]) -> [u8; 32] {
    use keccak_asm::{Digest, Keccak256};
    let mut h = Keccak256::new();
    h.update(data);
    let out = h.finalize();
    let mut a = [0u8; 32];
    a.copy_from_slice(&out);
    a
}

/* Verify a hit on the host: keccak256(abi.encode(challenge, nonce)) < difficulty.
 * Returns the hash if it verifies, else error. */
pub fn verify_hit(challenge: &[u8; 32], nonce: &[u8; 32], difficulty: U256) -> Result<[u8; 32]> {
    let mut buf = [0u8; 64];
    buf[..32].copy_from_slice(challenge);
    buf[32..].copy_from_slice(nonce);
    let h = keccak_host(&buf);
    let hash_u = U256::from_be_bytes(h);
    if hash_u >= difficulty {
        return Err(anyhow!(
            "host verify: hash {:#066x} >= difficulty {:#066x}",
            hash_u, difficulty
        ));
    }
    Ok(h)
}

/* Build the encoded calldata for `mine(uint256 nonce)`. */
pub fn encode_mine_call(nonce_be: &[u8; 32]) -> Vec<u8> {
    let nonce = U256::from_be_bytes(*nonce_be);
    Hash::mineCall { nonce }.abi_encode()
}

/* Parse a contract address string. */
pub fn parse_addr(s: &str) -> Result<Address> {
    Address::from_str(s).map_err(|e| anyhow!("bad address {s}: {e}"))
}

/* Decode the known custom error selectors so we can label revert reasons. */
pub fn decode_custom_error(data: &[u8]) -> Option<&'static str> {
    if data.len() < 4 { return None; }
    let sel: [u8; 4] = data[..4].try_into().unwrap();
    if sel == <Hash::InsufficientWork as SolError>::SELECTOR    { return Some("InsufficientWork"); }
    if sel == <Hash::ProofAlreadyUsed as SolError>::SELECTOR    { return Some("ProofAlreadyUsed"); }
    if sel == <Hash::BlockCapReached as SolError>::SELECTOR     { return Some("BlockCapReached"); }
    if sel == <Hash::SupplyExhausted as SolError>::SELECTOR     { return Some("SupplyExhausted"); }
    if sel == <Hash::GenesisNotComplete as SolError>::SELECTOR  { return Some("GenesisNotComplete"); }
    None
}

