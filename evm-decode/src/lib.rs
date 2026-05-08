//! # tiders-evm-decode
//!
//! Decodes EVM smart contract data from binary format into Apache Arrow RecordBatches.
//!
//! Supports three types of decoding:
//! - **Event logs** ([`decode_events`]) — Decodes indexed topics and non-indexed body data
//!   using Solidity event signatures (e.g. `"Transfer(address indexed,address indexed,uint256)"`).
//! - **Function call inputs** ([`decode_call_inputs`]) — Decodes ABI-encoded calldata.
//! - **Function call outputs** ([`decode_call_outputs`]) — Decodes ABI-encoded return data.
//!
//! Also provides ABI parsing utilities ([`abi_events`], [`abi_functions`]) to extract
//! event/function signatures from JSON ABI files, and schema generation functions
//! to preview the Arrow output schema without decoding data.
//!
//! All decoded output uses Arrow's columnar format with support for arbitrarily nested
//! Solidity types (tuples, arrays, structs) mapped to Arrow Struct and List types.

mod abi;
mod arrow_convert;

use std::sync::Arc;

use alloy_dyn_abi::{DynSolCall, DynSolEvent, DynSolType, DynSolValue, Specifier};
use anyhow::{anyhow, Context, Result};
use arrow::{
    array::{
        Array, BinaryArray, GenericBinaryArray, LargeBinaryArray, OffsetSizeTrait, RecordBatch,
        StructArray,
    },
    compute,
    datatypes::{DataType, Field, Schema},
};

pub use abi::*;
use arrow_convert::{
    build_topic0_mask, decode_body_named, decode_topic, param_to_arrow_dtype, to_arrow,
    to_arrow_dtype,
};

/// Returns topic0 based on a human-readable Solidity event signature
/// (e.g. `"Transfer(address indexed,address indexed,uint256)"`).
pub fn signature_to_topic0(signature: &str) -> Result<[u8; 32]> {
    let event = alloy_json_abi::Event::parse(signature).context("parse event signature")?;
    Ok(event.selector().into())
}

/// Returns topic0 based on a JSON ABI fragment for an event
/// (e.g. the `abi_json` field from [`crate::abi_events`]).
pub fn abi_to_topic0(abi_json: &str) -> Result<[u8; 32]> {
    let event: alloy_json_abi::Event =
        serde_json::from_str(abi_json).context("parse event ABI JSON")?;
    Ok(event.selector().into())
}

/// Decodes given call input data in arrow format to arrow format.
/// Output Arrow schema is auto generated based on the function signature.
/// Handles any level of nesting with Lists/Structs.
///
/// Writes `null` for data rows that fail to decode if `allow_decode_fail` is set to `true`.
/// Errors when a row fails to decode if `allow_decode_fail` is set to `false`.
pub fn decode_call_inputs<I: OffsetSizeTrait>(
    signature: &str,
    data: &GenericBinaryArray<I>,
    allow_decode_fail: bool,
) -> Result<RecordBatch> {
    decode_call_impl::<true, I>(signature, data, allow_decode_fail)
}

/// Decodes given call output data in arrow format to arrow format.
/// Output Arrow schema is auto generated based on the function signature.
/// Handles any level of nesting with Lists/Structs.
///
/// Writes `null` for data rows that fail to decode if `allow_decode_fail` is set to `true`.
/// Errors when a row fails to decode if `allow_decode_fail` is set to `false`.
pub fn decode_call_outputs<I: OffsetSizeTrait>(
    signature: &str,
    data: &GenericBinaryArray<I>,
    allow_decode_fail: bool,
) -> Result<RecordBatch> {
    decode_call_impl::<false, I>(signature, data, allow_decode_fail)
}

fn decode_call_impl<const IS_INPUT: bool, I: OffsetSizeTrait>(
    signature: &str,
    data: &GenericBinaryArray<I>,
    allow_decode_fail: bool,
) -> Result<RecordBatch> {
    let (call, resolved) = resolve_function_signature(signature)?;

    let schema = function_signature_to_arrow_schemas_impl(&call, &resolved)
        .context("convert event signature to arrow schema")?;
    let schema = if IS_INPUT { schema.0 } else { schema.1 };

    let mut arrays: Vec<Arc<dyn Array + 'static>> = Vec::with_capacity(schema.fields().len());

    let mut decoded = Vec::<Option<DynSolValue>>::with_capacity(data.len());

    for blob in data {
        match blob {
            Some(blob) => {
                let decode_res = if IS_INPUT {
                    resolved.abi_decode_input(blob)
                } else {
                    resolved.abi_decode_output(blob)
                };
                match decode_res {
                    Ok(data) => decoded.push(Some(DynSolValue::Tuple(data))),
                    Err(e) if allow_decode_fail => {
                        log::debug!("failed to decode function call data: {e}");
                        decoded.push(None);
                    }
                    Err(e) => {
                        return Err(anyhow!("failed to decode function call data: {e}"));
                    }
                }
            }
            None => decoded.push(None),
        }
    }

    let sol_type = if IS_INPUT {
        DynSolType::Tuple(resolved.types().to_vec())
    } else {
        DynSolType::Tuple(resolved.returns().types().to_vec())
    };

    let array = to_arrow(&sol_type, decoded, allow_decode_fail).context("map params to arrow")?;
    let arr = array
        .as_any()
        .downcast_ref::<StructArray>()
        .context("expected struct array from to_arrow")?;

    for f in arr.columns() {
        arrays.push(f.clone());
    }

    RecordBatch::try_new(Arc::new(schema), arrays).context("construct arrow batch")
}

/// Returns the Arrow schemas for a function's inputs and outputs as `(input_schema, output_schema)`.
pub fn function_signature_to_arrow_schemas(signature: &str) -> Result<(Schema, Schema)> {
    let (func, resolved) = resolve_function_signature(signature)?;
    function_signature_to_arrow_schemas_impl(&func, &resolved)
}

fn function_signature_to_arrow_schemas_impl(
    func: &alloy_json_abi::Function,
    call: &DynSolCall,
) -> Result<(Schema, Schema)> {
    let mut input_fields = Vec::with_capacity(call.types().len());
    let mut output_fields = Vec::with_capacity(call.returns().types().len());

    for (i, (sol_t, param)) in call.types().iter().zip(func.inputs.iter()).enumerate() {
        let dtype = to_arrow_dtype(sol_t).context("map to arrow type")?;
        let name = if param.name() == "" {
            format!("param{i}")
        } else {
            param.name().to_owned()
        };
        input_fields.push(Arc::new(Field::new(name, dtype, true)));
    }

    for (i, (sol_t, param)) in call
        .returns()
        .types()
        .iter()
        .zip(func.outputs.iter())
        .enumerate()
    {
        let dtype = to_arrow_dtype(sol_t).context("map to arrow type")?;
        let name = if param.name() == "" {
            format!("param{i}")
        } else {
            param.name().to_owned()
        };
        output_fields.push(Arc::new(Field::new(name, dtype, true)));
    }

    Ok((Schema::new(input_fields), Schema::new(output_fields)))
}

fn resolve_function_signature(signature: &str) -> Result<(alloy_json_abi::Function, DynSolCall)> {
    let event = alloy_json_abi::Function::parse(signature).context("parse function signature")?;
    let resolved = event.resolve().context("resolve function signature")?;

    Ok((event, resolved))
}

/// Decodes given event data in arrow format to arrow format.
/// Output Arrow schema is auto generated based on the event signature.
/// Handles any level of nesting with Lists/Structs.
///
/// When `filter_by_topic0` is `true`, only rows whose `topic0` column matches the
/// event's selector are decoded. Non-matching rows are silently filtered out.
///
/// When `hstack` is `true`, the original input columns (after any topic0 filtering)
/// are appended alongside the decoded columns in the output.
///
/// Writes `null` for data rows that fail to decode if `allow_decode_fail` is set to `true`.
/// Errors when a row fails to decode if `allow_decode_fail` is set to `false`.
pub fn decode_events(
    signature: &str,
    data: &RecordBatch,
    allow_decode_fail: bool,
    filter_by_topic0: bool,
    hstack: bool,
) -> Result<RecordBatch> {
    let (event, resolved) = resolve_event_signature(signature)?;

    // Optionally filter input to only rows whose topic0 matches the event selector.
    let data = if filter_by_topic0 {
        filter_by_topic0_impl(&event, data)?
    } else {
        data.clone()
    };

    let schema = event_signature_to_arrow_schema_impl(&event)
        .context("convert event signature to arrow schema")?;

    let mut fields: Vec<Arc<Field>> = schema.fields().iter().cloned().collect();
    let mut arrays: Vec<Arc<dyn Array + 'static>> = Vec::with_capacity(fields.len());

    for (sol_type, topic_name) in resolved
        .indexed()
        .iter()
        .zip(&["topic1", "topic2", "topic3"])
    {
        let col = data
            .column_by_name(topic_name)
            .context("get topic column")?;

        if col.data_type() == &DataType::Binary {
            let arr = col
                .as_any()
                .downcast_ref::<BinaryArray>()
                .context("downcast to BinaryArray")?;
            decode_topic(sol_type, arr, allow_decode_fail, &mut arrays).context("decode topic")?;
        } else if col.data_type() == &DataType::LargeBinary {
            let arr = col
                .as_any()
                .downcast_ref::<LargeBinaryArray>()
                .context("downcast to LargeBinaryArray")?;
            decode_topic(sol_type, arr, allow_decode_fail, &mut arrays).context("decode topic")?;
        }
    }

    let body_col = data.column_by_name("data").context("get data column")?;
    let body_sol_type = DynSolType::Tuple(resolved.body().to_vec());
    let body_params: Vec<&alloy_json_abi::EventParam> =
        event.inputs.iter().filter(|i| !i.indexed).collect();

    if body_col.data_type() == &DataType::Binary {
        let arr = body_col
            .as_any()
            .downcast_ref::<BinaryArray>()
            .context("downcast to BinaryArray")?;
        decode_body_named(&body_sol_type, &body_params, arr, allow_decode_fail, &mut arrays)
            .context("decode body")?;
    } else if body_col.data_type() == &DataType::LargeBinary {
        let arr = body_col
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .context("downcast to LargeBinaryArray")?;
        decode_body_named(&body_sol_type, &body_params, arr, allow_decode_fail, &mut arrays)
            .context("decode body")?;
    }

    if hstack {
        for (i, col) in data.columns().iter().enumerate() {
            fields.push(data.schema().field(i).clone().into());
            arrays.push(col.clone());
        }
    }

    let output_schema = Schema::new(fields);
    let batch =
        RecordBatch::try_new(Arc::new(output_schema), arrays).context("construct arrow batch")?;
    tiders_cast::flatten_record_batch(&batch).context("flatten decoded batch")
}

/// Returns the Arrow schema that [`decode_events`] would produce for the given event signature.
pub fn event_signature_to_arrow_schema(signature: &str) -> Result<Schema> {
    let (event, _) = resolve_event_signature(signature)?;
    let schema = event_signature_to_arrow_schema_impl(&event)?;
    Ok(tiders_cast::flatten_schema(&schema))
}

/// Builds the Arrow schema for an event directly from its [`alloy_json_abi::Event`].
///
/// Indexed params come first (matching the topic decode order), followed by
/// body params. Tuple params use [`param_to_arrow_dtype`] so component names
/// are preserved as named `Struct` fields.
fn event_signature_to_arrow_schema_impl(sig: &alloy_json_abi::Event) -> Result<Schema> {
    let mut fields = Vec::<Arc<Field>>::new();

    for (i, input) in sig.inputs.iter().enumerate() {
        if input.indexed {
            let name =
                if input.name.is_empty() { format!("param{i}") } else { input.name.clone() };
            let dtype = param_to_arrow_dtype(&input.ty, &input.components)
                .context("map indexed param to arrow type")?;
            fields.push(Arc::new(Field::new(name, dtype, true)));
        }
    }
    for (i, input) in sig.inputs.iter().enumerate() {
        if !input.indexed {
            let name =
                if input.name.is_empty() { format!("param{i}") } else { input.name.clone() };
            let dtype = param_to_arrow_dtype(&input.ty, &input.components)
                .context("map body param to arrow type")?;
            fields.push(Arc::new(Field::new(name, dtype, true)));
        }
    }

    Ok(Schema::new(fields))
}

/// Parse an event from either a human-readable Solidity signature or a JSON
/// fragment (the format produced by [`crate::abi_events`]).
///
/// Accepts three formats:
/// - JSON ABI fragment: `{"type":"event","name":"Swap","inputs":[...],...}`
/// - Standard HR signature: `Swap(address indexed sender, uint256 amount)`
/// - Full HR signature with named tuple fields (alloy `full_signature()` output):
///   `Swap(address indexed sender, tuple(int256 a, int256 b) data)` — alloy's
///   `Event::parse` rejects named types inside tuple parens, so we parse this
///   ourselves, build a JSON value, and deserialise to preserve component names.
fn parse_event_str(signature: &str) -> Result<alloy_json_abi::Event> {
    if signature.starts_with('{') {
        return serde_json::from_str(signature).context("parse event JSON");
    }
    if signature.contains("tuple(") {
        log::warn!(
            "Event signature contains `tuple(...)` which will produce unnamed fields in the Arrow schema. \
             Consider passing a JSON ABI fragment instead.\n  \
             Signature: {signature}"
        );
    }
    alloy_json_abi::Event::parse(signature).map_err(|e| {
        anyhow!(
            "{e}\n  \
             Hint: if the signature contains named tuple fields (e.g. `tuple(int256 foo, ...)`),\n  \
             either strip the inner names (e.g. `(int256,...)`) or pass a JSON ABI fragment."
        )
    })
}

/// Find the index of the closing `)` that matches the `(` at `open`.
fn resolve_event_signature(signature: &str) -> Result<(alloy_json_abi::Event, DynSolEvent)> {
    let event = parse_event_str(signature)?;
    let resolved = event.resolve().context("resolve event signature")?;
    Ok((event, resolved))
}

/// Filters a RecordBatch to only rows where the `topic0` column matches the
/// event's selector hash. If `topic0` column is not present, returns the data
/// unchanged. Non-matching rows are always silently filtered out.
fn filter_by_topic0_impl(event: &alloy_json_abi::Event, data: &RecordBatch) -> Result<RecordBatch> {
    let Some(topic0_col) = data.column_by_name("topic0") else {
        return Ok(data.clone());
    };

    let selector = event.selector();

    let mask =
        build_topic0_mask(topic0_col, selector.as_slice()).context("build topic0 filter mask")?;

    let non_matching = mask.iter().filter(|v| !v.unwrap_or(false)).count();
    if non_matching > 0 {
        log::debug!(
            "filtering out {non_matching} events whose topic0 does not match '{}'",
            event.full_signature()
        );
    }

    compute::filter_record_batch(data, &mask).context("filter record batch by topic0")
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_primitives::{I256, U256};
    use arrow::datatypes::{Fields, Int32Type};

    #[test]
    fn test_int_overflow_with_allow_decode_fail() {
        // When decoding all pool events without topic filtering, a Swap decoder
        // may successfully ABI-decode body data from a different event type,
        // producing an int24 value that doesn't fit in i32. With allow_decode_fail=true
        // this should produce null instead of erroring.
        let sol_values = vec![Some(DynSolValue::Int(I256::MAX, 24))];
        let result = arrow_convert::to_int_impl::<Int32Type>(24, &sol_values, true);
        assert!(result.is_ok());
        let arr = result.unwrap();
        assert!(arr.is_null(0));

        // Without allow_decode_fail, it should error
        let sol_values = vec![Some(DynSolValue::Int(I256::MAX, 24))];
        let result = arrow_convert::to_int_impl::<Int32Type>(24, &sol_values, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_topic0_filtering_with_allow_decode_fail() {
        use arrow::array::GenericBinaryBuilder;

        let swap_sig = "Swap(address indexed sender, address indexed recipient, int256 amount0, int256 amount1, uint160 sqrtPriceX96, uint128 liquidity, int24 tick)";
        let mint_sig = "Mint(address sender, address indexed owner, int24 indexed tickLower, int24 indexed tickUpper, uint128 amount, uint256 amount0, uint256 amount1)";

        let swap_selector = signature_to_topic0(swap_sig).unwrap();
        let mint_selector = signature_to_topic0(mint_sig).unwrap();

        // Build a batch with 2 rows: one Swap event and one Mint event
        let mut topic0_builder = GenericBinaryBuilder::<i32>::new();
        let mut topic1_builder = GenericBinaryBuilder::<i32>::new();
        let mut topic2_builder = GenericBinaryBuilder::<i32>::new();
        let mut topic3_builder = GenericBinaryBuilder::<i32>::new();
        let mut data_builder = GenericBinaryBuilder::<i32>::new();

        let addr = [0u8; 32];

        // Row 0: Swap event
        topic0_builder.append_value(swap_selector);
        topic1_builder.append_value(addr);
        topic2_builder.append_value(addr);
        topic3_builder.append_null();
        let amount0 = I256::try_from(-1000i64).unwrap();
        let amount1 = I256::try_from(2000i64).unwrap();
        let sqrt_price: U256 = U256::from(1u64) << 96;
        let liquidity = U256::from(1000000u64);
        let tick = I256::try_from(-100i64).unwrap();
        let mut swap_body = Vec::new();
        swap_body.extend_from_slice(&amount0.to_be_bytes::<32>());
        swap_body.extend_from_slice(&amount1.to_be_bytes::<32>());
        swap_body.extend_from_slice(&sqrt_price.to_be_bytes::<32>());
        swap_body.extend_from_slice(&liquidity.to_be_bytes::<32>());
        swap_body.extend_from_slice(&tick.to_be_bytes::<32>());
        data_builder.append_value(&swap_body);

        // Row 1: Mint event (different topic0, different body layout)
        topic0_builder.append_value(mint_selector);
        topic1_builder.append_value(addr);
        topic2_builder.append_value(addr);
        topic3_builder.append_value(addr);
        let mint_body = vec![0u8; 32 * 4]; // sender, amount, amount0, amount1
        data_builder.append_value(&mint_body);

        let schema = Arc::new(Schema::new(vec![
            Field::new("topic0", DataType::Binary, true),
            Field::new("topic1", DataType::Binary, true),
            Field::new("topic2", DataType::Binary, true),
            Field::new("topic3", DataType::Binary, true),
            Field::new("data", DataType::Binary, true),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(topic0_builder.finish()),
                Arc::new(topic1_builder.finish()),
                Arc::new(topic2_builder.finish()),
                Arc::new(topic3_builder.finish()),
                Arc::new(data_builder.finish()),
            ],
        )
        .unwrap();

        // With filter_by_topic0=true, should filter to only the Swap row
        let result = decode_events(swap_sig, &batch, true, true, false).unwrap();
        assert_eq!(result.num_rows(), 1, "should only decode the Swap row");

        // With filter_by_topic0=true and hstack=true, decoded + input columns are returned
        let result = decode_events(swap_sig, &batch, true, true, true).unwrap();
        assert_eq!(result.num_rows(), 1);
        // Should have decoded columns + original input columns
        assert!(
            result.column_by_name("topic0").is_some(),
            "hstack should include original input columns"
        );
    }

    #[test]
    #[ignore]
    fn test_decode_events_named_tuple_via_abi_json() {
        use arrow::array::{BinaryArray, Decimal128Array, GenericBinaryBuilder};

        // JSON ABI fragment with:
        //  - a nested static tuple (premiumDelta.breakdown) → tests two-level struct recursion
        //  - a variable-length tuple array (rewards: tuple[]) → tests List<Struct> → Utf8
        let abi_json = r#"{
            "type": "event",
            "name": "RefreshPremium",
            "inputs": [
                {"name": "assetId", "type": "uint256", "indexed": true, "components": []},
                {"name": "spoke",   "type": "address", "indexed": true, "components": []},
                {"name": "premiumDelta", "type": "tuple", "indexed": false, "components": [
                    {"name": "sharesDelta",    "type": "int256", "components": []},
                    {"name": "offsetRayDelta", "type": "int256", "components": []},
                    {"name": "breakdown", "type": "tuple", "components": [
                        {"name": "base",  "type": "uint128", "components": []},
                        {"name": "bonus", "type": "uint128", "components": []}
                    ]}
                ]},
                {"name": "rewards", "type": "tuple[]", "indexed": false, "components": [
                    {"name": "token",  "type": "address", "components": []},
                    {"name": "amount", "type": "uint256", "components": []}
                ]}
            ],
            "anonymous": false
        }"#;

        let asset_id = U256::from(42u64);
        let spoke_addr = [1u8; 20];
        let shares_delta = I256::try_from(-100i64).unwrap();
        let offset_ray_delta = I256::try_from(200i64).unwrap();
        let base: u128 = 500;
        let bonus: u128 = 999;
        let reward_token_0 = [2u8; 20];
        let reward_amount_0 = U256::from(1000u64);
        let reward_token_1 = [3u8; 20];
        let reward_amount_1 = U256::from(2000u64);

        // topic1: uint256 → 32-byte big-endian
        let topic1 = asset_id.to_be_bytes::<32>();

        // topic2: address → left-padded to 32 bytes
        let mut topic2 = [0u8; 32];
        topic2[12..].copy_from_slice(&spoke_addr);

        // Body: ABI-encoded sequence (premiumDelta, rewards).
        // premiumDelta is static (4 × 32 = 128 bytes); rewards is dynamic (tuple[]).
        // ABI head/tail layout:
        //   [0..128]   premiumDelta inline (4 words)
        //   [128..160] offset for rewards = 160 (start of tail, relative to head start)
        //   [160..192] rewards array length = 2
        //   [192..256] rewards[0]: token (32 bytes) + amount (32 bytes)
        //   [256..320] rewards[1]: token (32 bytes) + amount (32 bytes)
        let premiumdelta_word_count: u64 = 4; // sharesDelta + offsetRayDelta + base + bonus
        let head_size: u64 = premiumdelta_word_count * 32 + 32; // +32 for rewards offset word
        let mut body = Vec::new();
        // premiumDelta fields (static, encoded inline in the head)
        body.extend_from_slice(&shares_delta.to_be_bytes::<32>());
        body.extend_from_slice(&offset_ray_delta.to_be_bytes::<32>());
        body.extend_from_slice(&U256::from(base).to_be_bytes::<32>());
        body.extend_from_slice(&U256::from(bonus).to_be_bytes::<32>());
        // offset pointing to the start of rewards tail data
        body.extend_from_slice(&U256::from(head_size).to_be_bytes::<32>());
        // rewards tail: length + elements
        body.extend_from_slice(&U256::from(2u64).to_be_bytes::<32>());
        let mut reward_token_0_padded = [0u8; 32];
        reward_token_0_padded[12..].copy_from_slice(&reward_token_0);
        body.extend_from_slice(&reward_token_0_padded);
        body.extend_from_slice(&reward_amount_0.to_be_bytes::<32>());
        let mut reward_token_1_padded = [0u8; 32];
        reward_token_1_padded[12..].copy_from_slice(&reward_token_1);
        body.extend_from_slice(&reward_token_1_padded);
        body.extend_from_slice(&reward_amount_1.to_be_bytes::<32>());

        let selector = abi_to_topic0(abi_json).unwrap();

        let mut topic0_b = GenericBinaryBuilder::<i32>::new();
        let mut topic1_b = GenericBinaryBuilder::<i32>::new();
        let mut topic2_b = GenericBinaryBuilder::<i32>::new();
        let mut topic3_b = GenericBinaryBuilder::<i32>::new();
        let mut data_b = GenericBinaryBuilder::<i32>::new();

        topic0_b.append_value(selector);
        topic1_b.append_value(topic1);
        topic2_b.append_value(topic2);
        topic3_b.append_null();
        data_b.append_value(&body);

        let schema = Arc::new(Schema::new(vec![
            Field::new("topic0", DataType::Binary, true),
            Field::new("topic1", DataType::Binary, true),
            Field::new("topic2", DataType::Binary, true),
            Field::new("topic3", DataType::Binary, true),
            Field::new("data",   DataType::Binary, true),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(topic0_b.finish()),
                Arc::new(topic1_b.finish()),
                Arc::new(topic2_b.finish()),
                Arc::new(topic3_b.finish()),
                Arc::new(data_b.finish()),
            ],
        )
        .unwrap();

        let result = decode_events(abi_json, &batch, false, false, false).unwrap();

        assert_eq!(result.num_rows(), 1);

        // Output must be fully flat — no Struct columns survive
        for field in result.schema().fields() {
            assert!(
                !matches!(field.data_type(), DataType::Struct(_)),
                "field '{}' should not be Struct after flattening",
                field.name()
            );
        }

        // Indexed params are top-level; nested tuple fields are dot-joined two levels deep;
        // tuple[] is flattened to a single Utf8 column (variable-length → serialised).
        let out_schema = result.schema();
        assert!(out_schema.field_with_name("assetId").is_ok());
        assert!(out_schema.field_with_name("spoke").is_ok());
        assert!(out_schema.field_with_name("premiumDelta.sharesDelta").is_ok());
        assert!(out_schema.field_with_name("premiumDelta.offsetRayDelta").is_ok());
        assert!(out_schema.field_with_name("premiumDelta.breakdown.base").is_ok());
        assert!(out_schema.field_with_name("premiumDelta.breakdown.bonus").is_ok());
        assert_eq!(
            out_schema.field_with_name("rewards").unwrap().data_type(),
            &DataType::Utf8,
            "variable-length tuple[] should be serialised to Utf8"
        );

        // Print the human-readable signature derived from the JSON ABI fragment
        let event: alloy_json_abi::Event = serde_json::from_str(abi_json).unwrap();
        println!("Human-readable signature: {}", event.full_signature());

        // Print the serialised rewards string (List<Struct> → Utf8)
        use arrow::array::StringArray;
        let rewards_col = result
            .column_by_name("rewards").unwrap()
            .as_any().downcast_ref::<StringArray>().unwrap();
        println!("rewards[0]: {}", rewards_col.value(0));

        // spoke: address decodes to 20 raw bytes
        let spoke_col = result
            .column_by_name("spoke").unwrap()
            .as_any().downcast_ref::<BinaryArray>().unwrap();
        assert_eq!(spoke_col.value(0), spoke_addr);

        // breakdown.base and breakdown.bonus: uint128 → Decimal128(38, 0)
        let base_col = result
            .column_by_name("premiumDelta.breakdown.base").unwrap()
            .as_any().downcast_ref::<Decimal128Array>().unwrap();
        assert_eq!(base_col.value(0), base as i128);

        let bonus_col = result
            .column_by_name("premiumDelta.breakdown.bonus").unwrap()
            .as_any().downcast_ref::<Decimal128Array>().unwrap();
        assert_eq!(bonus_col.value(0), bonus as i128);

        // Save decoded batch to parquet in the crate root
        use parquet::arrow::ArrowWriter;
        use std::fs::File;
        let file = File::create("refresh_premium_decoded.parquet").unwrap();
        let mut writer = ArrowWriter::try_new(file, result.schema(), None).unwrap();
        writer.write(&result).unwrap();
        writer.close().unwrap();
    }

    #[test]
    #[ignore]
    fn nested_event_signature_to_schema() {
        let sig = "ConfiguredQuests(address editor, uint256[][], address indexed my_addr, (bool,bool[],(bool, uint256[]))[] questDetails)";

        let schema = event_signature_to_arrow_schema(sig).unwrap();

        let expected_schema = Schema::new(vec![
            Arc::new(Field::new("my_addr", DataType::Binary, true)),
            Arc::new(Field::new("editor", DataType::Binary, true)),
            Arc::new(Field::new(
                "param1",
                DataType::List(Arc::new(Field::new(
                    "",
                    DataType::List(Arc::new(Field::new("", DataType::Decimal256(76, 0), true))),
                    true,
                ))),
                true,
            )),
            Arc::new(Field::new(
                "questDetails",
                DataType::List(Arc::new(Field::new(
                    "",
                    DataType::Struct(Fields::from(vec![
                        Arc::new(Field::new("param0", DataType::Boolean, true)),
                        Arc::new(Field::new(
                            "param1",
                            DataType::List(Arc::new(Field::new("", DataType::Boolean, true))),
                            true,
                        )),
                        Arc::new(Field::new(
                            "param2",
                            DataType::Struct(Fields::from(vec![
                                Arc::new(Field::new("param0", DataType::Boolean, true)),
                                Arc::new(Field::new(
                                    "param1",
                                    DataType::List(Arc::new(Field::new(
                                        "",
                                        DataType::Decimal256(76, 0),
                                        true,
                                    ))),
                                    true,
                                )),
                            ])),
                            true,
                        )),
                    ])),
                    true,
                ))),
                true,
            )),
        ]);

        assert_eq!(schema, expected_schema);
    }

    #[test]
    #[ignore]
    fn i256_to_arrow_i256() {
        for val in [
            I256::MIN,
            I256::MAX,
            I256::MAX / I256::try_from(2i32).unwrap(),
        ] {
            let out = arrow::datatypes::i256::from_be_bytes(val.to_be_bytes::<32>());

            assert_eq!(val.to_string(), out.to_string());
        }
    }

    #[test]
    #[ignore]
    fn read_parquet_with_real_data() {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        use std::fs::File;
        let builder =
            ParquetRecordBatchReaderBuilder::try_new(File::open("logs.parquet").unwrap()).unwrap();
        let mut reader = builder.build().unwrap();
        let logs = reader.next().unwrap().unwrap();

        let signature =
            "PairCreated(address indexed token0, address indexed token1, address pair,uint256)";

        let decoded = decode_events(signature, &logs, false, false, false).unwrap();

        // Save the filtered instructions to a new parquet file
        let mut file = File::create("decoded_logs.parquet").unwrap();
        let mut writer =
            parquet::arrow::ArrowWriter::try_new(&mut file, decoded.schema(), None).unwrap();
        writer.write(&decoded).unwrap();
        writer.close().unwrap();
    }
}
