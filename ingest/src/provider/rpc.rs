//! Direct JSON-RPC provider for EVM blockchain data.

use crate::{evm, ProviderConfig, ProviderStream, Query, RpcTraceMethod};
use anyhow::{anyhow, Context, Result};
use futures_lite::StreamExt;
use tiders_rpc_client::{Client, ClientConfig};

pub fn start_stream(provider_config: &ProviderConfig, query: Query) -> Result<ProviderStream> {
    let evm_query = match query {
        Query::Evm(q) => q,
        Query::Svm(_) => return Err(anyhow!("RPC provider does not support SVM queries")),
    };

    let rpc_query = map_query(&evm_query);
    let client_config = map_client_config(provider_config)?;

    let client = Client::new(client_config)?;
    let stream = client.stream(rpc_query).context("create rpc stream")?;

    let stream = stream.map(|res| {
        res.map(Client::response_to_btree)
            .context("map rpc response")
    });

    Ok(Box::pin(stream))
}

fn map_client_config(cfg: &ProviderConfig) -> Result<ClientConfig> {
    let url = cfg
        .url
        .clone()
        .with_context(|| "RPC provider requires a url")?;

    let mut client_config = ClientConfig::new(url);

    client_config.bearer_token.clone_from(&cfg.bearer_token);

    if let Some(v) = cfg.max_num_retries {
        client_config.max_num_retries =
            u32::try_from(v).context("max_num_retries exceeds u32 range")?;
    }
    if let Some(v) = cfg.retry_backoff_ms {
        client_config.retry_backoff_ms = v;
    }
    if let Some(v) = cfg.retry_base_ms {
        client_config.retry_base_ms = v;
    }
    if let Some(v) = cfg.retry_ceiling_ms {
        client_config.retry_ceiling_ms = v;
    }
    if let Some(v) = cfg.req_timeout_millis {
        client_config.req_timeout_millis = v;
    }

    client_config.stop_on_head = cfg.stop_on_head;

    if let Some(v) = cfg.head_poll_interval_millis {
        client_config.head_poll_interval_millis = v;
    }
    if let Some(v) = cfg.buffer_size {
        client_config.buffer_size = v;
    }

    if let Some(v) = cfg.compute_units_per_second {
        client_config.compute_units_per_second = Some(v);
    }
    if let Some(v) = cfg.batch_size {
        client_config.batch_size = Some(v);
    }
    if let Some(v) = cfg.reorg_safe_distance {
        client_config.reorg_safe_distance = v;
    }

    if let Some(v) = cfg.trace_method {
        client_config.trace_method = Some(match v {
            RpcTraceMethod::TraceBlock => tiders_rpc_client::TraceMethod::TraceBlock,
            RpcTraceMethod::DebugTraceBlockByNumber => {
                tiders_rpc_client::TraceMethod::DebugTraceBlockByNumber
            }
        });
    }

    Ok(client_config)
}

fn map_query(q: &evm::Query) -> tiders_rpc_client::Query {
    tiders_rpc_client::Query {
        from_block: q.from_block,
        to_block: q.to_block,
        include_all_blocks: q.include_all_blocks,
        logs: q.logs.iter().map(map_log_request).collect(),
        transactions: q.transactions.iter().map(map_tx_request).collect(),
        traces: q.traces.iter().map(map_trace_request).collect(),
        fields: map_fields(&q.fields),
    }
}

fn map_log_request(r: &evm::LogRequest) -> tiders_rpc_client::LogRequest {
    tiders_rpc_client::LogRequest {
        address: r
            .address
            .iter()
            .map(|a| tiders_rpc_client::Address(a.0))
            .collect(),
        topic0: r
            .topic0
            .iter()
            .map(|t| tiders_rpc_client::Topic(t.0))
            .collect(),
        topic1: r
            .topic1
            .iter()
            .map(|t| tiders_rpc_client::Topic(t.0))
            .collect(),
        topic2: r
            .topic2
            .iter()
            .map(|t| tiders_rpc_client::Topic(t.0))
            .collect(),
        topic3: r
            .topic3
            .iter()
            .map(|t| tiders_rpc_client::Topic(t.0))
            .collect(),
        include_transactions: r.include_transactions,
        include_transaction_logs: r.include_transaction_logs,
        include_transaction_traces: r.include_transaction_traces,
        include_blocks: r.include_blocks,
    }
}

fn map_tx_request(r: &evm::TransactionRequest) -> tiders_rpc_client::TransactionRequest {
    tiders_rpc_client::TransactionRequest {
        from_: r
            .from_
            .iter()
            .map(|a| tiders_rpc_client::Address(a.0))
            .collect(),
        to: r
            .to
            .iter()
            .map(|a| tiders_rpc_client::Address(a.0))
            .collect(),
        sighash: r
            .sighash
            .iter()
            .map(|s| tiders_rpc_client::Sighash(s.0))
            .collect(),
        status: r.status.clone(),
        type_: r.type_.clone(),
        contract_deployment_address: r
            .contract_deployment_address
            .iter()
            .map(|a| tiders_rpc_client::Address(a.0))
            .collect(),
        hash: r
            .hash
            .iter()
            .map(|h| tiders_rpc_client::Hash(h.0))
            .collect(),
        include_logs: r.include_logs,
        include_traces: r.include_traces,
        include_blocks: r.include_blocks,
    }
}

fn map_trace_request(r: &evm::TraceRequest) -> tiders_rpc_client::TraceRequest {
    tiders_rpc_client::TraceRequest {
        from_: r
            .from_
            .iter()
            .map(|a| tiders_rpc_client::Address(a.0))
            .collect(),
        to: r
            .to
            .iter()
            .map(|a| tiders_rpc_client::Address(a.0))
            .collect(),
        address: r
            .address
            .iter()
            .map(|a| tiders_rpc_client::Address(a.0))
            .collect(),
        call_type: r.call_type.clone(),
        reward_type: r.reward_type.clone(),
        type_: r.type_.clone(),
        sighash: r
            .sighash
            .iter()
            .map(|s| tiders_rpc_client::Sighash(s.0))
            .collect(),
        author: r
            .author
            .iter()
            .map(|a| tiders_rpc_client::Address(a.0))
            .collect(),
        trace_method: tiders_rpc_client::TraceMethod::default(),
        include_transactions: r.include_transactions,
        include_transaction_logs: r.include_transaction_logs,
        include_transaction_traces: r.include_transaction_traces,
        include_blocks: r.include_blocks,
    }
}

fn map_fields(f: &evm::Fields) -> tiders_rpc_client::Fields {
    tiders_rpc_client::Fields {
        block: tiders_rpc_client::BlockFields {
            number: f.block.number,
            hash: f.block.hash,
            parent_hash: f.block.parent_hash,
            nonce: f.block.nonce,
            sha3_uncles: f.block.sha3_uncles,
            logs_bloom: f.block.logs_bloom,
            transactions_root: f.block.transactions_root,
            state_root: f.block.state_root,
            receipts_root: f.block.receipts_root,
            miner: f.block.miner,
            difficulty: f.block.difficulty,
            total_difficulty: f.block.total_difficulty,
            extra_data: f.block.extra_data,
            size: f.block.size,
            gas_limit: f.block.gas_limit,
            gas_used: f.block.gas_used,
            timestamp: f.block.timestamp,
            uncles: f.block.uncles,
            base_fee_per_gas: f.block.base_fee_per_gas,
            blob_gas_used: f.block.blob_gas_used,
            excess_blob_gas: f.block.excess_blob_gas,
            parent_beacon_block_root: f.block.parent_beacon_block_root,
            withdrawals_root: f.block.withdrawals_root,
            withdrawals: f.block.withdrawals,
            l1_block_number: f.block.l1_block_number,
            send_count: f.block.send_count,
            send_root: f.block.send_root,
            mix_hash: f.block.mix_hash,
        },
        transaction: tiders_rpc_client::TransactionFields {
            block_hash: f.transaction.block_hash,
            block_number: f.transaction.block_number,
            from: f.transaction.from_,
            gas: f.transaction.gas,
            gas_price: f.transaction.gas_price,
            hash: f.transaction.hash,
            input: f.transaction.input,
            nonce: f.transaction.nonce,
            to: f.transaction.to,
            transaction_index: f.transaction.transaction_index,
            value: f.transaction.value,
            v: f.transaction.v,
            r: f.transaction.r,
            s: f.transaction.s,
            max_priority_fee_per_gas: f.transaction.max_priority_fee_per_gas,
            max_fee_per_gas: f.transaction.max_fee_per_gas,
            chain_id: f.transaction.chain_id,
            cumulative_gas_used: f.transaction.cumulative_gas_used,
            effective_gas_price: f.transaction.effective_gas_price,
            gas_used: f.transaction.gas_used,
            contract_address: f.transaction.contract_address,
            logs_bloom: f.transaction.logs_bloom,
            type_: f.transaction.type_,
            root: f.transaction.root,
            status: f.transaction.status,
            sighash: f.transaction.sighash,
            y_parity: f.transaction.y_parity,
            access_list: f.transaction.access_list,
            l1_fee: f.transaction.l1_fee,
            l1_gas_price: f.transaction.l1_gas_price,
            l1_fee_scalar: f.transaction.l1_fee_scalar,
            gas_used_for_l1: f.transaction.gas_used_for_l1,
            max_fee_per_blob_gas: f.transaction.max_fee_per_blob_gas,
            blob_versioned_hashes: f.transaction.blob_versioned_hashes,
            deposit_nonce: f.transaction.deposit_nonce,
            blob_gas_price: f.transaction.blob_gas_price,
            deposit_receipt_version: f.transaction.deposit_receipt_version,
            blob_gas_used: f.transaction.blob_gas_used,
            l1_base_fee_scalar: f.transaction.l1_base_fee_scalar,
            l1_blob_base_fee: f.transaction.l1_blob_base_fee,
            l1_blob_base_fee_scalar: f.transaction.l1_blob_base_fee_scalar,
            l1_block_number: f.transaction.l1_block_number,
            mint: f.transaction.mint,
            source_hash: f.transaction.source_hash,
        },
        log: tiders_rpc_client::LogFields {
            removed: f.log.removed,
            log_index: f.log.log_index,
            transaction_index: f.log.transaction_index,
            transaction_hash: f.log.transaction_hash,
            block_hash: f.log.block_hash,
            block_number: f.log.block_number,
            address: f.log.address,
            data: f.log.data,
            topic0: f.log.topic0,
            topic1: f.log.topic1,
            topic2: f.log.topic2,
            topic3: f.log.topic3,
        },
        trace: tiders_rpc_client::TraceFields {
            from: f.trace.from_,
            to: f.trace.to,
            call_type: f.trace.call_type,
            gas: f.trace.gas,
            input: f.trace.input,
            init: f.trace.init,
            value: f.trace.value,
            author: f.trace.author,
            reward_type: f.trace.reward_type,
            block_hash: f.trace.block_hash,
            block_number: f.trace.block_number,
            address: f.trace.address,
            code: f.trace.code,
            gas_used: f.trace.gas_used,
            output: f.trace.output,
            subtraces: f.trace.subtraces,
            trace_address: f.trace.trace_address,
            transaction_hash: f.trace.transaction_hash,
            transaction_position: f.trace.transaction_position,
            type_: f.trace.type_,
            error: f.trace.error,
            sighash: f.trace.sighash,
            action_address: f.trace.action_address,
            balance: f.trace.balance,
            refund_address: f.trace.refund_address,
        },
    }
}

#[cfg(test)]
mod tests {
    use futures_lite::StreamExt;

    use super::*;
    use crate::{evm, ProviderConfig, ProviderKind, Query};

    fn rpc_config(url: &str) -> ProviderConfig {
        let mut cfg = ProviderConfig::new(ProviderKind::Rpc);
        cfg.url = Some(url.to_owned());
        cfg
    }

    #[tokio::test]
    async fn start_stream_returns_empty_batches() {
        let query = Query::Evm(evm::Query {
            from_block: 0,
            to_block: Some(0),
            include_all_blocks: false,
            ..Default::default()
        });

        let mut stream = start_stream(&rpc_config("http://localhost:8545"), query).unwrap();
        let data = stream.next().await.unwrap().unwrap();

        for batch in data.values() {
            assert_eq!(batch.num_rows(), 0);
        }
    }

    #[test]
    fn config_mapping_and_error_paths() {
        // missing url → error
        let no_url_err = start_stream(
            &ProviderConfig::new(ProviderKind::Rpc),
            Query::Evm(evm::Query::default()),
        )
        .err()
        .unwrap();
        assert!(no_url_err.to_string().contains("url"));

        // SVM query → error
        let svm_err = start_stream(
            &rpc_config("http://localhost:8545"),
            Query::Svm(crate::svm::Query::default()),
        )
        .err()
        .unwrap();
        assert!(svm_err.to_string().contains("SVM"));

        // connection config fields are passed through
        let mut cfg = rpc_config("http://node:8545");
        cfg.max_num_retries = Some(3);
        cfg.retry_backoff_ms = Some(500);
        cfg.retry_base_ms = Some(100);
        cfg.retry_ceiling_ms = Some(5000);
        cfg.req_timeout_millis = Some(10_000);
        let client_cfg = map_client_config(&cfg).unwrap();
        assert_eq!(client_cfg.url, "http://node:8545");
        assert_eq!(client_cfg.max_num_retries, 3);
        assert_eq!(client_cfg.retry_backoff_ms, 500);
        assert_eq!(client_cfg.retry_base_ms, 100);
        assert_eq!(client_cfg.retry_ceiling_ms, 5000);
        assert_eq!(client_cfg.req_timeout_millis, 10_000);

        // stream behaviour fields are passed through
        let mut cfg = rpc_config("http://node:8545");
        cfg.stop_on_head = true;
        cfg.head_poll_interval_millis = Some(2000);
        cfg.buffer_size = Some(32);
        let client_cfg = map_client_config(&cfg).unwrap();
        assert!(client_cfg.stop_on_head);
        assert_eq!(client_cfg.head_poll_interval_millis, 2000);
        assert_eq!(client_cfg.buffer_size, 32);
    }

    #[test]
    fn query_and_fields_mapping() {
        // block range, log address/topic, and all-true fields round-trip
        let addr = evm::Address([0xAB; 20]);
        let topic = evm::Topic([0xCD; 32]);
        let q = evm::Query {
            from_block: 100,
            to_block: Some(200),
            include_all_blocks: true,
            logs: vec![evm::LogRequest {
                address: vec![addr],
                topic0: vec![topic],
                ..Default::default()
            }],
            ..Default::default()
        };
        let rpc_q = map_query(&q);
        assert_eq!(rpc_q.from_block, 100);
        assert_eq!(rpc_q.to_block, Some(200));
        assert!(rpc_q.include_all_blocks);
        assert_eq!(rpc_q.logs[0].address[0].0, [0xAB; 20]);
        assert_eq!(rpc_q.logs[0].topic0[0].0, [0xCD; 32]);

        let rpc_f = map_fields(&evm::Fields::all());
        assert!(rpc_f.block.number && rpc_f.block.hash);
        assert!(rpc_f.transaction.from && rpc_f.transaction.gas_used);
        assert!(rpc_f.log.topic0);
        assert!(rpc_f.trace.call_type);
    }
}
