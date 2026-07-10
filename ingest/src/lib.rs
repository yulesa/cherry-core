#![expect(
    clippy::should_implement_trait,
    reason = "LogKind::from_str is a constructor pattern, not Trait impl"
)]
#![expect(
    clippy::field_reassign_with_default,
    reason = "ProviderConfig is built by setting fields after construction"
)]

//! # tiders-ingest
//!
//! Streams blockchain data from multiple provider backends as Apache Arrow RecordBatches.
//!
//! Supports both EVM (Ethereum) and SVM (Solana) chains through a unified [`Query`] enum.
//! Data is fetched from one of three provider backends:
//!
//! - **SQD** ([`ProviderKind::Sqd`]) — SQD Network portal for historical data.
//! - **HyperSync** ([`ProviderKind::Hypersync`]) — Envio HyperSync for fast historical data.
//! - **RPC** ([`ProviderKind::Rpc`]) — Direct JSON-RPC node connection.
//!
//! Use [`start_stream`] to create an async stream of `BTreeMap<String, RecordBatch>` where
//! keys are table names (e.g. "blocks", "transactions", "logs").

use std::{collections::BTreeMap, pin::Pin, sync::Arc};

#[cfg(feature = "pyo3")]
use anyhow::anyhow;
use anyhow::{Context, Result};
use arrow::array::UInt64Array;
use arrow::compute::kernels::aggregate::max as array_max;
use arrow::record_batch::RecordBatch;
use futures_lite::{Stream, StreamExt};
use provider::common::{evm_query_to_generic, svm_query_to_generic};
use serde::de::DeserializeOwned;

pub mod evm;
mod provider;
mod rayon_async;
pub mod svm;

/// Top-level query type: either an EVM or SVM blockchain data query.
#[derive(Debug, Clone)]
pub enum Query {
    Evm(evm::Query),
    Svm(svm::Query),
}

#[cfg(feature = "pyo3")]
impl<'a, 'py> pyo3::FromPyObject<'a, 'py> for Query {
    type Error = pyo3::PyErr;
    fn extract(ob: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        use pyo3::types::PyAnyMethods;

        let kind = ob.getattr("kind").context("get kind attribute")?;
        let kind: &str = kind.extract().context("kind as str")?;

        let query = ob.getattr("params").context("get params attribute")?;

        match kind {
            "evm" => Ok(Self::Evm(query.extract().context("parse query")?)),
            "svm" => Ok(Self::Svm(query.extract().context("parse query")?)),
            _ => Err(anyhow!("unknown query kind: {kind}").into()),
        }
    }
}

/// Configuration for a data provider, including connection details and retry settings.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "pyo3", derive(pyo3::FromPyObject))]
pub struct ProviderConfig {
    pub kind: ProviderKind,
    pub url: Option<String>,
    pub bearer_token: Option<String>,
    pub max_num_retries: Option<usize>,
    pub retry_backoff_ms: Option<u64>,
    pub retry_base_ms: Option<u64>,
    pub retry_ceiling_ms: Option<u64>,
    pub req_timeout_millis: Option<u64>,
    pub stop_on_head: bool,
    pub head_poll_interval_millis: Option<u64>,
    pub buffer_size: Option<usize>,
    // RPC-specific fields
    pub compute_units_per_second: Option<u64>,
    pub batch_size: Option<usize>,
    pub reorg_safe_distance: Option<u64>,
    pub trace_method: Option<RpcTraceMethod>,
}

impl ProviderConfig {
    pub fn new(kind: ProviderKind) -> Self {
        Self {
            kind,
            url: None,
            bearer_token: None,
            max_num_retries: None,
            retry_backoff_ms: None,
            retry_base_ms: None,
            retry_ceiling_ms: None,
            req_timeout_millis: None,
            stop_on_head: false,
            head_poll_interval_millis: None,
            buffer_size: None,
            compute_units_per_second: None,
            batch_size: None,
            reorg_safe_distance: None,
            trace_method: None,
        }
    }
}

/// Selects which RPC method to use for fetching EVM execution traces.
#[derive(Debug, Clone, Copy)]
pub enum RpcTraceMethod {
    /// Uses `trace_block` (Parity/OpenEthereum style).
    TraceBlock,
    /// Uses `debug_traceBlockByNumber` (Geth style).
    DebugTraceBlockByNumber,
}

#[cfg(feature = "pyo3")]
impl<'a, 'py> pyo3::FromPyObject<'a, 'py> for RpcTraceMethod {
    type Error = pyo3::PyErr;
    fn extract(ob: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {

        let out: &str = ob.extract().context("read as string")?;

        match out {
            "trace_block" => Ok(Self::TraceBlock),
            "debug_trace_block_by_number" => Ok(Self::DebugTraceBlockByNumber),
            _ => Err(anyhow!("unknown trace method: {out}").into()),
        }
    }
}

/// The type of data provider backend.
#[derive(Debug, Clone, Copy)]
pub enum ProviderKind {
    /// SQD Network portal.
    Sqd,
    /// Envio HyperSync.
    Hypersync,
    /// Direct JSON-RPC node.
    Rpc,
}

#[cfg(feature = "pyo3")]
impl<'a, 'py> pyo3::FromPyObject<'a, 'py> for ProviderKind {
    type Error = pyo3::PyErr;
    fn extract(ob: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {

        let out: &str = ob.extract().context("read as string")?;

        match out {
            "sqd" => Ok(Self::Sqd),
            "hypersync" => Ok(Self::Hypersync),
            "rpc" => Ok(Self::Rpc),
            _ => Err(anyhow!("unknown provider kind: {out}").into()),
        }
    }
}

/// One yielded chunk from [`start_stream`]: the projected tables plus the
/// highest block-id observed in this chunk (`None` if the chunk has no rows
/// in any table carrying a block-id column).
#[derive(Debug)]
pub struct StreamItem {
    pub data: BTreeMap<String, RecordBatch>,
    pub last_block: Option<u64>,
}

/// Public stream type returned by [`start_stream`].
pub type DataStream = Pin<Box<dyn Stream<Item = Result<StreamItem>> + Send + Sync>>;

/// Internal stream type returned by each provider before projection /
/// `last_block` extraction.
pub(crate) type ProviderStream =
    Pin<Box<dyn Stream<Item = Result<BTreeMap<String, RecordBatch>>> + Send + Sync>>;

/// Returns the maximum value across every column in `data` whose name is one
/// of the known block-id column names (`block_number` for EVM non-block
/// tables, `number` for the EVM blocks table, `slot` for SVM tables).
///
/// Runs against pre-projection batches, where these columns are guaranteed to
/// be present (they are required by the local query filter step).
fn extract_last_block(data: &BTreeMap<String, RecordBatch>) -> Option<u64> {
    const BLOCK_ID_COLUMNS: &[&str] = &["block_number", "number", "slot"];

    let mut out: Option<u64> = None;
    for batch in data.values() {
        for col_name in BLOCK_ID_COLUMNS {
            let Some(col) = batch.column_by_name(col_name) else {
                continue;
            };
            let Some(arr) = col.as_any().downcast_ref::<UInt64Array>() else {
                continue;
            };
            if let Some(m) = array_max(arr) {
                out = Some(out.map_or(m, |cur| cur.max(m)));
            }
        }
    }
    out
}

fn make_req_fields<T: DeserializeOwned>(query: &tiders_query::Query) -> Result<T> {
    let mut req_fields_query = query.clone();
    req_fields_query
        .add_request_and_include_fields()
        .context("add req and include fields")?;

    let fields = req_fields_query
        .fields
        .into_iter()
        .map(|(k, v)| -> Result<_> {
            Ok((
                k.strip_suffix('s')
                    .context("field key should end with 's'")?
                    .to_owned(),
                v.into_iter()
                    .map(|v| (v, true))
                    .collect::<BTreeMap<String, bool>>(),
            ))
        })
        .collect::<Result<BTreeMap<String, _>>>()?;

    let json_value = serde_json::to_value(&fields).context("serialize fields to JSON")?;
    serde_json::from_value(json_value).context("deserialize fields from JSON")
}

/// Creates an async stream of Arrow RecordBatches from the configured provider.
pub async fn start_stream(provider_config: ProviderConfig, mut query: Query) -> Result<DataStream> {
    let generic_query = match &mut query {
        Query::Evm(evm_query) => {
            let generic_query = evm_query_to_generic(evm_query).context("validate evm query")?;

            evm_query.fields = make_req_fields(&generic_query).context("make req fields")?;

            generic_query
        }
        Query::Svm(svm_query) => {
            let generic_query = svm_query_to_generic(svm_query);

            svm_query.fields = make_req_fields(&generic_query).context("make req fields")?;

            generic_query
        }
    };
    let generic_query = Arc::new(generic_query);

    let stream = match provider_config.kind {
        ProviderKind::Sqd => {
            provider::sqd::start_stream(provider_config, query).context("start sqd stream")?
        }
        ProviderKind::Hypersync => provider::hypersync::start_stream(provider_config, query)
            .await
            .context("start hypersync stream")?,
        ProviderKind::Rpc => {
            provider::rpc::start_stream(&provider_config, query).context("start rpc stream")?
        }
    };

    let stream = stream.then(move |res| {
        let generic_query = Arc::clone(&generic_query);
        async {
            rayon_async::spawn(move || {
                res.and_then(move |data| {
                    let last_block = extract_last_block(&data);
                    let data = tiders_query::run_query(&data, &generic_query)
                        .context("run local query")?;
                    Ok(StreamItem { data, last_block })
                })
            })
            .await
            .context("rayon task was cancelled")
            .and_then(|r| r)
        }
    });

    Ok(Box::pin(stream))
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::svm::*;
    use parquet::arrow::ArrowWriter;
    use std::fs::File;

    #[tokio::test]
    #[ignore]
    async fn simple_svm_start_stream() {
        let mut provider_config = ProviderConfig::new(ProviderKind::Sqd);
        provider_config.url = Some("https://portal.sqd.dev/datasets/solana-mainnet".to_string());

        let program_id = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA";
        let program_id: [u8; 32] = bs58::decode(program_id)
            .into_vec()
            .unwrap()
            .try_into()
            .unwrap();
        let program_id = Address(program_id);

        let query = crate::Query::Svm(svm::Query {
            from_block: 329443000,
            to_block: Some(329443000),
            include_all_blocks: false,
            fields: Fields {
                instruction: InstructionFields::all(),
                transaction: TransactionFields::default(),
                log: LogFields::default(),
                balance: BalanceFields::default(),
                token_balance: TokenBalanceFields::default(),
                reward: RewardFields::default(),
                block: BlockFields::default(),
            },
            instructions: vec![
                // InstructionRequest::default() ,
                InstructionRequest {
                    program_id: vec![program_id],
                    discriminator: vec![Data(vec![12, 96, 49, 128, 22])],
                    ..Default::default()
                },
            ],
            transactions: vec![],
            logs: vec![],
            balances: vec![],
            token_balances: vec![],
            rewards: vec![],
        });
        let mut stream = start_stream(provider_config, query).await.unwrap();
        let item = stream.next().await.unwrap().unwrap();
        for (k, v) in item.data.into_iter() {
            let mut file = File::create(format!("{}.parquet", k)).unwrap();
            let mut writer = ArrowWriter::try_new(&mut file, v.schema(), None).unwrap();
            writer.write(&v).unwrap();
            writer.close().unwrap();
        }
    }

    /// End-to-end smoke test for the public `start_stream` surface against the
    /// SQD ethereum-mainnet portal: fetches ERC-20 Transfer logs over a small
    /// block range and asserts the `StreamItem` invariants
    /// (`from_block` / `to_block` are mirrored on the query; `last_block` is
    /// monotonic, within the requested range, and reaches `to_block` once the
    /// stream exhausts).
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn simple_start_stream() {
        use crate::evm::{Fields, LogFields, LogRequest, Query as EvmQuery, Topic};

        // keccak256("Transfer(address,address,uint256)")
        let transfer_topic0: [u8; 32] = {
            let mut out = [0u8; 32];
            faster_hex::hex_decode(
                b"ddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef",
                &mut out,
            )
            .unwrap();
            out
        };

        let from_block = 18_000_000u64;
        let to_block = 18_000_010u64;
        // let to_block = 18_001_000u64;

        let mut provider_config = ProviderConfig::new(ProviderKind::Sqd);
        provider_config.url = Some("https://portal.sqd.dev/datasets/ethereum-mainnet".to_string());
        provider_config.stop_on_head = true;

        // let mut provider_config = ProviderConfig::new(ProviderKind::Hypersync);
        // provider_config.url =
        //     Some("https://eth.hypersync.xyz/".to_string());
        // provider_config.stop_on_head = true;
        // provider_config.bearer_token = Some("add_token".to_string());

        let mut fields = Fields::default();
        fields.log = LogFields {
            log_index: true,
            block_number: true,
            address: true,
            topic0: true,
            ..LogFields::default()
        };

        let query = crate::Query::Evm(EvmQuery {
            from_block,
            to_block: Some(to_block),
            include_all_blocks: false,
            logs: vec![LogRequest {
                topic0: vec![Topic(transfer_topic0)],
                ..LogRequest::default()
            }],
            transactions: vec![],
            traces: vec![],
            fields,
        });

        let mut stream = start_stream(provider_config, query).await.unwrap();

        println!("from_block={} to_block={}", from_block, to_block);

        let mut highest: Option<u64> = None;
        let mut total_rows: usize = 0;

        while let Some(res) = stream.next().await {
            let item = res.unwrap();
            println!("item last_block={:?}", item.last_block);

            // Every yielded last_block must sit inside the requested window.
            if let Some(lb) = item.last_block {
                assert!(
                    lb >= from_block,
                    "last_block {} < from_block {}",
                    lb,
                    from_block
                );
                assert!(lb <= to_block, "last_block {} > to_block {}", lb, to_block);
            }

            // Monotonicity across items.
            if let (Some(prev), Some(cur)) = (highest, item.last_block) {
                assert!(
                    cur >= prev,
                    "last_block went backwards: {} -> {}",
                    prev,
                    cur
                );
            }
            if item.last_block.is_some() {
                highest = item.last_block;
            }

            if let Some(logs) = item.data.get("logs") {
                total_rows += logs.num_rows();
            }
        }

        println!(
            "final: from_block={} to_block={} last_block={:?} total_rows={}",
            from_block, to_block, highest, total_rows
        );

        // The stream must have delivered at least one Transfer log in a
        // 10-block window of mainnet, and must have reached the configured
        // upper bound by the time it closed.
        assert!(
            total_rows > 0,
            "expected at least one Transfer log in window"
        );
        assert_eq!(highest, Some(to_block));
    }
}
