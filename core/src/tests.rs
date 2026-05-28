use std::sync::Arc;

use futures_lite::StreamExt;
use hypersync_client::{self, ClientConfig, StreamConfig};
use tiders_evm_decode::{decode_events, signature_to_topic0};
use tiders_ingest::evm::{Address, Topic};

async fn erc20(cfg: tiders_ingest::ProviderConfig, query: tiders_ingest::Query) {
    let signature = "Transfer(address indexed from, address indexed to, uint256 amount)";
    let mut stream = tiders_ingest::start_stream(cfg, query).await.unwrap();

    while let Some(v) = stream.next().await {
        let v = v.unwrap();
        let decoded = decode_events(
            signature,
            v.data.get("logs").unwrap(),
            false,
            false,
            false,
            false,
        )
        .unwrap();
        dbg!(decoded);
    }
}

fn erc20_query() -> tiders_ingest::Query {
    let signature = "Transfer(address indexed from, address indexed to, uint256 amount)";
    tiders_ingest::Query::Evm(tiders_ingest::evm::Query {
        from_block: 18123123,
        to_block: Some(18123222),
        fields: tiders_ingest::evm::Fields::all(),
        logs: vec![tiders_ingest::evm::LogRequest {
            address: vec![Address(decode_hex(
                "0xdAC17F958D2ee523a2206206994597C13D831ec7",
            ))],
            topic0: vec![Topic(signature_to_topic0(signature).unwrap())],
            ..Default::default()
        }],
        ..Default::default()
    })
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn erc20_hypersync() {
    let provider = tiders_ingest::ProviderConfig::new(tiders_ingest::ProviderKind::Hypersync);
    let query = erc20_query();
    erc20(provider, query).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn erc20_sqd() {
    let provider = tiders_ingest::ProviderConfig {
        url: Some("https://portal.sqd.dev/datasets/ethereum-mainnet".to_owned()),
        ..tiders_ingest::ProviderConfig::new(tiders_ingest::ProviderKind::Sqd)
    };
    erc20(provider, erc20_query()).await;
}

fn decode_hex<const N: usize>(hex: &str) -> [u8; N] {
    let mut dst = [0; N];
    faster_hex::hex_decode(
        hex.strip_prefix("0x").unwrap().as_bytes(),
        dst.as_mut_slice(),
    )
    .unwrap();
    dst
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn decode_nested_list() {
    let client = hypersync_client::Client::new(ClientConfig {
        url: Some("https://10.hypersync.xyz".parse().unwrap()),
        ..Default::default()
    })
    .unwrap();
    let client = Arc::new(client);

    let signature =
        "ConfiguredQuests(address editor, uint256[] questIdList, (bool, bool, bool)[] details)";

    let query = serde_json::from_value(serde_json::json!({
        "from_block": 0,
        "logs": [{
            "address": ["0xC5893DcAB9AD32Fa47923FEbdE89883C62BfFbd6"],
            "topics": [[hypersync_client::format::LogArgument::try_from(signature_to_topic0(signature).unwrap().as_slice()).unwrap()]]
        }],
        "field_selection": {
            "log": hypersync_client::schema::log()
                .fields
                .iter()
                .map(|f| f.name.clone())
                .collect::<Vec<String>>(),
        }
    })).unwrap();

    let res = client
        .collect_arrow(query, StreamConfig::default())
        .await
        .unwrap();

    let logs = res.data.logs.iter().map(polars_arrow_to_arrow_rs);

    for batch in logs {
        let decoded = decode_events(signature, &batch, false, false, false, false).unwrap();

        dbg!(decoded);
    }
}

fn polars_arrow_to_arrow_rs(
    batch: &hypersync_client::ArrowBatch,
) -> arrow::record_batch::RecordBatch {
    let data_type = polars_arrow::datatypes::ArrowDataType::Struct(batch.schema.fields.clone());
    let arr = polars_arrow::array::StructArray::new(
        data_type.clone(),
        batch.chunk.columns().to_vec(),
        None,
    );

    let arr: arrow::ffi::FFI_ArrowArray =
        unsafe { std::mem::transmute(polars_arrow::ffi::export_array_to_c(Box::new(arr))) };
    let schema: arrow::ffi::FFI_ArrowSchema = unsafe {
        std::mem::transmute(polars_arrow::ffi::export_field_to_c(
            &polars_arrow::datatypes::Field::new("", data_type, false),
        ))
    };

    let mut arr_data = unsafe { arrow::ffi::from_ffi(arr, &schema).unwrap() };

    arr_data.align_buffers();

    let arr = arrow::array::StructArray::from(arr_data);

    arrow::record_batch::RecordBatch::from(arr)
}
