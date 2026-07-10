//! EVM query types and field selection for the ingest layer.

#[cfg(feature = "pyo3")]
use anyhow::Context;
use serde::{Deserialize, Serialize};

/// EVM blockchain data query specifying block range, filters, and field selections.
#[derive(Default, Debug, Clone)]
#[cfg_attr(feature = "pyo3", derive(pyo3::FromPyObject))]
pub struct Query {
    pub from_block: u64,
    pub to_block: Option<u64>,
    pub include_all_blocks: bool,
    pub transactions: Vec<TransactionRequest>,
    pub logs: Vec<LogRequest>,
    pub traces: Vec<TraceRequest>,
    pub fields: Fields,
}

/// A 32-byte hash value (block hash, transaction hash).
#[derive(Debug, Clone, Copy)]
pub struct Hash(pub [u8; 32]);

/// A 20-byte EVM address.
#[derive(Debug, Clone, Copy)]
pub struct Address(pub [u8; 20]);

/// A 4-byte function selector (first 4 bytes of keccak256 of the function signature).
#[derive(Debug, Clone, Copy)]
pub struct Sighash(pub [u8; 4]);

/// A 32-byte event log topic value.
#[derive(Debug, Clone, Copy)]
pub struct Topic(pub [u8; 32]);

#[cfg(feature = "pyo3")]
fn extract_hex<const N: usize>(ob: &pyo3::Bound<'_, pyo3::PyAny>) -> pyo3::PyResult<[u8; N]> {
    use pyo3::types::PyAnyMethods;

    let s: &str = ob.extract()?;
    let s = s.strip_prefix("0x").context("strip 0x prefix")?;
    let mut out = [0; N];
    faster_hex::hex_decode(s.as_bytes(), &mut out).context("decode hex")?;

    Ok(out)
}

#[cfg(feature = "pyo3")]
impl<'a, 'py> pyo3::FromPyObject<'a, 'py> for Hash {
    type Error = pyo3::PyErr;
    fn extract(ob: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        let out = extract_hex(&ob)?;
        Ok(Self(out))
    }
}

#[cfg(feature = "pyo3")]
impl<'a, 'py> pyo3::FromPyObject<'a, 'py> for Address {
    type Error = pyo3::PyErr;
    fn extract(ob: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        let out = extract_hex(&ob)?;
        Ok(Self(out))
    }
}

#[cfg(feature = "pyo3")]
impl<'a, 'py> pyo3::FromPyObject<'a, 'py> for Sighash {
    type Error = pyo3::PyErr;
    fn extract(ob: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        let out = extract_hex(&ob)?;
        Ok(Self(out))
    }
}

#[cfg(feature = "pyo3")]
impl<'a, 'py> pyo3::FromPyObject<'a, 'py> for Topic {
    type Error = pyo3::PyErr;
    fn extract(ob: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        let out = extract_hex(&ob)?;
        Ok(Self(out))
    }
}

/// Filters for selecting EVM transactions by sender, recipient, sighash, status, etc.
#[derive(Default, Debug, Clone)]
#[cfg_attr(feature = "pyo3", derive(pyo3::FromPyObject))]
pub struct TransactionRequest {
    pub from_: Vec<Address>,
    pub to: Vec<Address>,
    pub sighash: Vec<Sighash>,
    pub status: Vec<u8>,
    pub type_: Vec<u8>,
    pub contract_deployment_address: Vec<Address>,
    pub hash: Vec<Hash>,
    pub include_logs: bool,
    pub include_traces: bool,
    pub include_blocks: bool,
}

/// Filters for selecting EVM event logs by contract address and topics.
#[derive(Default, Debug, Clone)]
#[cfg_attr(feature = "pyo3", derive(pyo3::FromPyObject))]
#[expect(clippy::struct_excessive_bools, reason = "field selection flags")]
pub struct LogRequest {
    pub address: Vec<Address>,
    pub topic0: Vec<Topic>,
    pub topic1: Vec<Topic>,
    pub topic2: Vec<Topic>,
    pub topic3: Vec<Topic>,
    pub include_transactions: bool,
    pub include_transaction_logs: bool,
    pub include_transaction_traces: bool,
    pub include_blocks: bool,
}

/// Filters for selecting EVM execution traces by address, call type, etc.
#[derive(Default, Debug, Clone)]
#[cfg_attr(feature = "pyo3", derive(pyo3::FromPyObject))]
#[expect(clippy::struct_excessive_bools, reason = "field selection flags")]
pub struct TraceRequest {
    pub from_: Vec<Address>,
    pub to: Vec<Address>,
    pub address: Vec<Address>,
    pub call_type: Vec<String>,
    pub reward_type: Vec<String>,
    pub type_: Vec<String>,
    pub sighash: Vec<Sighash>,
    pub author: Vec<Address>,
    pub include_transactions: bool,
    pub include_transaction_logs: bool,
    pub include_transaction_traces: bool,
    pub include_blocks: bool,
}

/// Controls which columns are included in the response for each table type.
#[derive(Deserialize, Serialize, Default, Debug, Clone, Copy)]
#[serde(default)]
#[cfg_attr(feature = "pyo3", derive(pyo3::FromPyObject))]
pub struct Fields {
    pub block: BlockFields,
    pub transaction: TransactionFields,
    pub log: LogFields,
    pub trace: TraceFields,
}

impl Fields {
    pub fn all() -> Self {
        Self {
            block: BlockFields::all(),
            transaction: TransactionFields::all(),
            log: LogFields::all(),
            trace: TraceFields::all(),
        }
    }
}

/// Field selector for EVM block data columns.
#[derive(Default, Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default)]
#[cfg_attr(feature = "pyo3", derive(pyo3::FromPyObject))]
#[expect(clippy::struct_excessive_bools, reason = "field selection flags")]
pub struct BlockFields {
    pub number: bool,
    pub hash: bool,
    pub parent_hash: bool,
    pub nonce: bool,
    pub sha3_uncles: bool,
    pub logs_bloom: bool,
    pub transactions_root: bool,
    pub state_root: bool,
    pub receipts_root: bool,
    pub miner: bool,
    pub difficulty: bool,
    pub total_difficulty: bool,
    pub extra_data: bool,
    pub size: bool,
    pub gas_limit: bool,
    pub gas_used: bool,
    pub timestamp: bool,
    pub uncles: bool,
    pub base_fee_per_gas: bool,
    pub blob_gas_used: bool,
    pub excess_blob_gas: bool,
    pub parent_beacon_block_root: bool,
    pub withdrawals_root: bool,
    pub withdrawals: bool,
    pub l1_block_number: bool,
    pub send_count: bool,
    pub send_root: bool,
    pub mix_hash: bool,
}

impl BlockFields {
    pub fn all() -> Self {
        BlockFields {
            number: true,
            hash: true,
            parent_hash: true,
            nonce: true,
            sha3_uncles: true,
            logs_bloom: true,
            transactions_root: true,
            state_root: true,
            receipts_root: true,
            miner: true,
            difficulty: true,
            total_difficulty: true,
            extra_data: true,
            size: true,
            gas_limit: true,
            gas_used: true,
            timestamp: true,
            uncles: true,
            base_fee_per_gas: true,
            blob_gas_used: true,
            excess_blob_gas: true,
            parent_beacon_block_root: true,
            withdrawals_root: true,
            withdrawals: true,
            l1_block_number: true,
            send_count: true,
            send_root: true,
            mix_hash: true,
        }
    }
}

/// Field selector for EVM transaction data columns.
#[derive(Default, Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default)]
#[cfg_attr(feature = "pyo3", derive(pyo3::FromPyObject))]
#[expect(clippy::struct_excessive_bools, reason = "field selection flags")]
pub struct TransactionFields {
    pub block_hash: bool,
    pub block_number: bool,
    #[serde(rename = "from")]
    pub from_: bool,
    pub gas: bool,
    pub gas_price: bool,
    pub hash: bool,
    pub input: bool,
    pub nonce: bool,
    pub to: bool,
    pub transaction_index: bool,
    pub value: bool,
    pub v: bool,
    pub r: bool,
    pub s: bool,
    pub max_priority_fee_per_gas: bool,
    pub max_fee_per_gas: bool,
    pub chain_id: bool,
    pub cumulative_gas_used: bool,
    pub effective_gas_price: bool,
    pub gas_used: bool,
    pub contract_address: bool,
    pub logs_bloom: bool,
    #[serde(rename = "type")]
    pub type_: bool,
    pub root: bool,
    pub status: bool,
    pub sighash: bool,
    pub y_parity: bool,
    pub access_list: bool,
    pub l1_fee: bool,
    pub l1_gas_price: bool,
    pub l1_fee_scalar: bool,
    pub gas_used_for_l1: bool,
    pub max_fee_per_blob_gas: bool,
    pub blob_versioned_hashes: bool,
    pub deposit_nonce: bool,
    pub blob_gas_price: bool,
    pub deposit_receipt_version: bool,
    pub blob_gas_used: bool,
    pub l1_base_fee_scalar: bool,
    pub l1_blob_base_fee: bool,
    pub l1_blob_base_fee_scalar: bool,
    pub l1_block_number: bool,
    pub mint: bool,
    pub source_hash: bool,
}

impl TransactionFields {
    pub fn all() -> Self {
        TransactionFields {
            block_hash: true,
            block_number: true,
            from_: true,
            gas: true,
            gas_price: true,
            hash: true,
            input: true,
            nonce: true,
            to: true,
            transaction_index: true,
            value: true,
            v: true,
            r: true,
            s: true,
            max_priority_fee_per_gas: true,
            max_fee_per_gas: true,
            chain_id: true,
            cumulative_gas_used: true,
            effective_gas_price: true,
            gas_used: true,
            contract_address: true,
            logs_bloom: true,
            type_: true,
            root: true,
            status: true,
            sighash: true,
            y_parity: true,
            access_list: true,
            l1_fee: true,
            l1_gas_price: true,
            l1_fee_scalar: true,
            gas_used_for_l1: true,
            max_fee_per_blob_gas: true,
            blob_versioned_hashes: true,
            deposit_nonce: true,
            blob_gas_price: true,
            deposit_receipt_version: true,
            blob_gas_used: true,
            l1_base_fee_scalar: true,
            l1_blob_base_fee: true,
            l1_blob_base_fee_scalar: true,
            l1_block_number: true,
            mint: true,
            source_hash: true,
        }
    }
}

/// Field selector for EVM log data columns.
#[derive(Default, Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default)]
#[cfg_attr(feature = "pyo3", derive(pyo3::FromPyObject))]
#[expect(clippy::struct_excessive_bools, reason = "field selection flags")]
pub struct LogFields {
    pub removed: bool,
    pub log_index: bool,
    pub transaction_index: bool,
    pub transaction_hash: bool,
    pub block_hash: bool,
    pub block_number: bool,
    pub address: bool,
    pub data: bool,
    pub topic0: bool,
    pub topic1: bool,
    pub topic2: bool,
    pub topic3: bool,
}

impl LogFields {
    pub fn all() -> Self {
        LogFields {
            removed: true,
            log_index: true,
            transaction_index: true,
            transaction_hash: true,
            block_hash: true,
            block_number: true,
            address: true,
            data: true,
            topic0: true,
            topic1: true,
            topic2: true,
            topic3: true,
        }
    }
}

/// Field selector for EVM trace data columns.
#[derive(Default, Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default)]
#[cfg_attr(feature = "pyo3", derive(pyo3::FromPyObject))]
#[expect(clippy::struct_excessive_bools, reason = "field selection flags")]
pub struct TraceFields {
    #[serde(rename = "from")]
    pub from_: bool,
    pub to: bool,
    pub call_type: bool,
    pub gas: bool,
    pub input: bool,
    pub init: bool,
    pub value: bool,
    pub author: bool,
    pub reward_type: bool,
    pub block_hash: bool,
    pub block_number: bool,
    pub address: bool,
    pub code: bool,
    pub gas_used: bool,
    pub output: bool,
    pub subtraces: bool,
    pub trace_address: bool,
    pub transaction_hash: bool,
    pub transaction_position: bool,
    #[serde(rename = "type")]
    pub type_: bool,
    pub error: bool,
    pub sighash: bool,
    pub action_address: bool,
    pub balance: bool,
    pub refund_address: bool,
}

impl TraceFields {
    pub fn all() -> Self {
        TraceFields {
            from_: true,
            to: true,
            call_type: true,
            gas: true,
            input: true,
            init: true,
            value: true,
            author: true,
            reward_type: true,
            block_hash: true,
            block_number: true,
            address: true,
            code: true,
            gas_used: true,
            output: true,
            subtraces: true,
            trace_address: true,
            transaction_hash: true,
            transaction_position: true,
            type_: true,
            error: true,
            sighash: true,
            action_address: true,
            balance: true,
            refund_address: true,
        }
    }
}
