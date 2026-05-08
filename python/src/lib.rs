//! # tiders-core Python bindings
//!
//! PyO3-based Python module exposing the tiders-core library via PyArrow.
//!
//! All functions accept and return PyArrow arrays/RecordBatches, using Arrow's
//! C Data Interface for zero-copy data transfer between Rust and Python.
//!
//! The module registers functions for:
//! - **Casting & encoding**: `cast`, `hex_encode`, `base58_encode`, `u256_to_binary`, etc.
//! - **EVM decoding**: `evm_decode_events`, `evm_decode_call_inputs`, `evm_abi_events`, etc.
//! - **SVM decoding**: `svm_decode_instructions`, `svm_decode_logs`, etc.
//! - **Data ingestion**: via the `ingest` submodule.

use std::sync::LazyLock;

use anyhow::{anyhow, Context};
use arrow::array::{
    Array, ArrayData, BinaryArray, Decimal256Array, LargeBinaryArray, LargeStringArray,
    RecordBatch, StringArray,
};
use arrow::datatypes::{DataType, Schema};
use arrow::pyarrow::{FromPyArrow, ToPyArrow};
use baselib::svm_decode::{InstructionSignature, LogSignature};
use pyo3::prelude::*;
use tikv_jemallocator::Jemalloc;

mod ingest;

#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

#[expect(
    clippy::unwrap_used,
    reason = "Runtime creation is essential and only happens once at startup"
)]
static TOKIO_RUNTIME: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
});

#[pymodule]
fn tiders_core(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Info)
        .parse_default_env()
        .try_init()
        .ok();

    m.add_function(wrap_pyfunction!(cast, m)?)?;
    m.add_function(wrap_pyfunction!(cast_schema, m)?)?;
    m.add_function(wrap_pyfunction!(cast_by_type, m)?)?;
    m.add_function(wrap_pyfunction!(cast_schema_by_type, m)?)?;
    m.add_function(wrap_pyfunction!(base58_encode, m)?)?;
    m.add_function(wrap_pyfunction!(base58_encode_column, m)?)?;
    m.add_function(wrap_pyfunction!(hex_encode, m)?)?;
    m.add_function(wrap_pyfunction!(prefix_hex_encode, m)?)?;
    m.add_function(wrap_pyfunction!(hex_encode_column, m)?)?;
    m.add_function(wrap_pyfunction!(prefix_hex_encode_column, m)?)?;
    m.add_function(wrap_pyfunction!(base58_decode_column, m)?)?;
    m.add_function(wrap_pyfunction!(hex_decode_column, m)?)?;
    m.add_function(wrap_pyfunction!(prefix_hex_decode_column, m)?)?;
    m.add_function(wrap_pyfunction!(u256_column_from_binary, m)?)?;
    m.add_function(wrap_pyfunction!(u256_column_to_binary, m)?)?;
    m.add_function(wrap_pyfunction!(u256_to_binary, m)?)?;
    m.add_function(wrap_pyfunction!(svm_decode_instructions, m)?)?;
    m.add_function(wrap_pyfunction!(svm_decode_logs, m)?)?;
    m.add_function(wrap_pyfunction!(instruction_signature_to_arrow_schema, m)?)?;
    m.add_function(wrap_pyfunction!(evm_decode_call_inputs, m)?)?;
    m.add_function(wrap_pyfunction!(evm_decode_call_outputs, m)?)?;
    m.add_function(wrap_pyfunction!(evm_decode_events, m)?)?;
    m.add_function(wrap_pyfunction!(evm_event_signature_to_arrow_schema, m)?)?;
    m.add_function(wrap_pyfunction!(
        evm_function_signature_to_arrow_schemas,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(evm_signature_to_topic0, m)?)?;
    m.add_function(wrap_pyfunction!(evm_abi_to_topic0, m)?)?;
    m.add_class::<EvmAbiEvent>()?;
    m.add_class::<EvmAbiFunction>()?;
    m.add_function(wrap_pyfunction!(evm_abi_events, m)?)?;
    m.add_function(wrap_pyfunction!(evm_abi_functions, m)?)?;
    m.add_function(wrap_pyfunction!(flatten_batch, m)?)?;
    m.add_function(wrap_pyfunction!(base58_encode_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(base58_decode_string, m)?)?;
    ingest::ingest_module(py, m)?;

    Ok(())
}

struct CastDataType(DataType);

impl<'py> pyo3::FromPyObject<'py> for CastDataType {
    fn extract_bound(ob: &pyo3::Bound<'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        let dt = DataType::from_pyarrow_bound(ob)?;
        Ok(Self(dt))
    }
}

#[pyfunction]
fn cast(
    map: Vec<(String, CastDataType)>,
    batch: &Bound<'_, PyAny>,
    allow_cast_fail: bool,
    py: Python<'_>,
) -> PyResult<PyObject> {
    let batch = RecordBatch::from_pyarrow_bound(batch).context("convert batch from pyarrow")?;
    let map = map
        .into_iter()
        .map(|(name, dt)| (name, dt.0))
        .collect::<Vec<_>>();

    let batch = baselib::cast::cast(&map, &batch, allow_cast_fail).context("cast")?;

    Ok(batch.to_pyarrow(py).context("map result back to pyarrow")?)
}

#[pyfunction]
fn cast_schema(
    map: Vec<(String, CastDataType)>,
    schema: &Bound<'_, PyAny>,
    py: Python<'_>,
) -> PyResult<PyObject> {
    let schema = Schema::from_pyarrow_bound(schema).context("convert schema from pyarrow")?;
    let map = map
        .into_iter()
        .map(|(name, dt)| (name, dt.0))
        .collect::<Vec<_>>();

    let schema = baselib::cast::cast_schema(&map, &schema).context("cast")?;

    Ok(schema
        .to_pyarrow(py)
        .context("map result back to pyarrow")?)
}

#[pyfunction]
fn cast_by_type(
    batch: &Bound<'_, PyAny>,
    from_type: &Bound<'_, PyAny>,
    to_type: &Bound<'_, PyAny>,
    allow_cast_fail: bool,
    py: Python<'_>,
) -> PyResult<PyObject> {
    let batch = RecordBatch::from_pyarrow_bound(batch).context("convert batch from pyarrow")?;

    let from_type =
        DataType::from_pyarrow_bound(from_type).context("convert from_type to pyarrow")?;
    let to_type = DataType::from_pyarrow_bound(to_type).context("convert to_type to pyarrow")?;

    let batch = baselib::cast::cast_by_type(&batch, &from_type, &to_type, allow_cast_fail)
        .context("cast")?;

    Ok(batch.to_pyarrow(py).context("map result back to pyarrow")?)
}

#[pyfunction]
fn cast_schema_by_type(
    schema: &Bound<'_, PyAny>,
    from_type: &Bound<'_, PyAny>,
    to_type: &Bound<'_, PyAny>,
    py: Python<'_>,
) -> PyResult<PyObject> {
    let schema = Schema::from_pyarrow_bound(schema).context("convert schema from pyarrow")?;

    let from_type =
        DataType::from_pyarrow_bound(from_type).context("convert from_type to pyarrow")?;
    let to_type = DataType::from_pyarrow_bound(to_type).context("convert to_type to pyarrow")?;

    let schema =
        baselib::cast::cast_schema_by_type(&schema, &from_type, &to_type).context("cast")?;

    Ok(schema
        .to_pyarrow(py)
        .context("map result back to pyarrow")?)
}

#[pyfunction]
fn hex_encode(batch: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<PyObject> {
    let batch = RecordBatch::from_pyarrow_bound(batch).context("convert batch from pyarrow")?;

    let batch = baselib::cast::hex_encode::<false>(&batch).context("encode to hex")?;

    Ok(batch.to_pyarrow(py).context("map result back to pyarrow")?)
}

#[pyfunction]
fn base58_encode(batch: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<PyObject> {
    let batch = RecordBatch::from_pyarrow_bound(batch).context("convert batch from pyarrow")?;

    let batch = baselib::cast::base58_encode(&batch).context("encode to base58")?;

    Ok(batch.to_pyarrow(py).context("map result back to pyarrow")?)
}

#[pyfunction]
fn prefix_hex_encode(batch: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<PyObject> {
    let batch = RecordBatch::from_pyarrow_bound(batch).context("convert batch from pyarrow")?;

    let batch = baselib::cast::hex_encode::<true>(&batch).context("encode to prefix hex")?;

    Ok(batch.to_pyarrow(py).context("map result back to pyarrow")?)
}

#[pyfunction]
fn u256_to_binary(batch: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<PyObject> {
    let batch = RecordBatch::from_pyarrow_bound(batch).context("convert batch from pyarrow")?;

    let batch = baselib::cast::u256_to_binary(&batch).context("map u256 columns to binary")?;

    Ok(batch.to_pyarrow(py).context("map result back to pyarrow")?)
}

#[pyfunction]
fn base58_encode_column(col: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<PyObject> {
    let mut col = ArrayData::from_pyarrow_bound(col).context("convert column from pyarrow")?;

    // Align buffers that may be unaligned from Python (e.g. flight client).
    // See: https://github.com/apache/arrow/issues/43552
    col.align_buffers();

    if col.data_type() != &DataType::Binary {
        return Err(anyhow!("unexpected data type {}. Expected Binary", col.data_type()).into());
    }
    let col = BinaryArray::from(col);

    let col = baselib::cast::base58_encode_column(&col);

    Ok(col
        .into_data()
        .to_pyarrow(py)
        .context("map result back to pyarrow")?)
}

#[pyfunction]
fn hex_encode_column(col: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<PyObject> {
    hex_encode_column_impl::<false>(col, py)
}

#[pyfunction]
fn prefix_hex_encode_column(col: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<PyObject> {
    hex_encode_column_impl::<true>(col, py)
}

fn hex_encode_column_impl<const PREFIXED: bool>(
    col: &Bound<'_, PyAny>,
    py: Python<'_>,
) -> PyResult<PyObject> {
    let mut col = ArrayData::from_pyarrow_bound(col).context("convert column from pyarrow")?;

    // Align buffers that may be unaligned from Python (e.g. flight client).
    // See: https://github.com/apache/arrow/issues/43552
    col.align_buffers();

    if col.data_type() == &DataType::Binary {
        let col = BinaryArray::from(col);
        let col = baselib::cast::hex_encode_column::<PREFIXED, i32>(&col);
        Ok(col
            .into_data()
            .to_pyarrow(py)
            .context("map result back to pyarrow")?)
    } else if col.data_type() == &DataType::LargeBinary {
        let col = LargeBinaryArray::from(col);
        let col = baselib::cast::hex_encode_column::<PREFIXED, i64>(&col);
        Ok(col
            .into_data()
            .to_pyarrow(py)
            .context("map result back to pyarrow")?)
    } else {
        Err(anyhow!(
            "unexpected data type {}. Expected Binary or LargeBinary",
            col.data_type()
        )
        .into())
    }
}

#[pyfunction]
fn base58_decode_column(col: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<PyObject> {
    let mut col = ArrayData::from_pyarrow_bound(col).context("convert column from pyarrow")?;

    // Align buffers that may be unaligned from Python (e.g. flight client).
    // See: https://github.com/apache/arrow/issues/43552
    col.align_buffers();

    if col.data_type() == &DataType::Utf8 {
        let col = StringArray::from(col);
        let col = baselib::cast::base58_decode_column(&col).context("base58 decode")?;
        Ok(col
            .into_data()
            .to_pyarrow(py)
            .context("map result back to pyarrow")?)
    } else if col.data_type() == &DataType::LargeUtf8 {
        let col = LargeStringArray::from(col);
        let col = baselib::cast::base58_decode_column(&col).context("base58 decode")?;
        Ok(col
            .into_data()
            .to_pyarrow(py)
            .context("map result back to pyarrow")?)
    } else {
        Err(anyhow!(
            "unexpected data type {}. Expected String or LargeString",
            col.data_type()
        )
        .into())
    }
}

#[pyfunction]
fn hex_decode_column(col: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<PyObject> {
    hex_decode_column_impl::<false>(col, py)
}

#[pyfunction]
fn prefix_hex_decode_column(col: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<PyObject> {
    hex_decode_column_impl::<true>(col, py)
}

fn hex_decode_column_impl<const PREFIXED: bool>(
    col: &Bound<'_, PyAny>,
    py: Python<'_>,
) -> PyResult<PyObject> {
    let mut col = ArrayData::from_pyarrow_bound(col).context("convert column from pyarrow")?;

    // Align buffers that may be unaligned from Python (e.g. flight client).
    // See: https://github.com/apache/arrow/issues/43552
    col.align_buffers();

    if col.data_type() == &DataType::Utf8 {
        let col = StringArray::from(col);
        let col = baselib::cast::hex_decode_column::<PREFIXED, i32>(&col).context("hex decode")?;
        Ok(col
            .into_data()
            .to_pyarrow(py)
            .context("map result back to pyarrow")?)
    } else if col.data_type() == &DataType::LargeUtf8 {
        let col = LargeStringArray::from(col);
        let col = baselib::cast::hex_decode_column::<PREFIXED, i64>(&col).context("hex decode")?;
        Ok(col
            .into_data()
            .to_pyarrow(py)
            .context("map result back to pyarrow")?)
    } else {
        Err(anyhow!(
            "unexpected data type {}. Expected String or LargeString",
            col.data_type()
        )
        .into())
    }
}

#[pyfunction]
fn u256_column_from_binary(col: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<PyObject> {
    let mut col = ArrayData::from_pyarrow_bound(col).context("convert column from pyarrow")?;

    // Align buffers that may be unaligned from Python (e.g. flight client).
    // See: https://github.com/apache/arrow/issues/43552
    col.align_buffers();

    if col.data_type() != &DataType::Binary {
        return Err(anyhow!("unexpected data type {}. Expected Binary", col.data_type()).into());
    }
    let col = BinaryArray::from(col);

    let col = baselib::cast::u256_column_from_binary(&col).context("u256 from binary")?;

    Ok(col
        .into_data()
        .to_pyarrow(py)
        .context("map result back to pyarrow")?)
}

#[pyfunction]
fn u256_column_to_binary(col: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<PyObject> {
    let mut col = ArrayData::from_pyarrow_bound(col).context("convert column from pyarrow")?;

    // Align buffers that may be unaligned from Python (e.g. flight client).
    // See: https://github.com/apache/arrow/issues/43552
    col.align_buffers();

    if col.data_type() != &DataType::Decimal256(76, 0) {
        return Err(anyhow!(
            "unexpected data type {}. Expected Decimal256",
            col.data_type()
        )
        .into());
    }
    let col = Decimal256Array::from(col);

    let col = baselib::cast::u256_column_to_binary(&col).context("u256 to binary")?;

    Ok(col
        .into_data()
        .to_pyarrow(py)
        .context("map result back to pyarrow")?)
}

#[pyfunction]
#[pyo3(signature = (signature, batch, allow_decode_fail=false, filter_by_discriminator=false, hstack=false))]
fn svm_decode_instructions(
    signature: &Bound<'_, PyAny>,
    batch: &Bound<'_, PyAny>,
    allow_decode_fail: bool,
    filter_by_discriminator: bool,
    hstack: bool,
    py: Python<'_>,
) -> PyResult<PyObject> {
    let batch = RecordBatch::from_pyarrow_bound(batch).context("convert batch from pyarrow")?;

    let instruction_signature = signature.extract::<InstructionSignature>()?;
    let batch = baselib::svm_decode::decode_instructions_batch(
        &instruction_signature,
        &batch,
        allow_decode_fail,
        filter_by_discriminator,
        hstack,
    )
    .context("decode instruction batch")?;

    Ok(batch.to_pyarrow(py).context("map result back to pyarrow")?)
}

#[pyfunction]
#[pyo3(signature = (signature, batch, allow_decode_fail=false, hstack=false))]
fn svm_decode_logs(
    signature: &Bound<'_, PyAny>,
    batch: &Bound<'_, PyAny>,
    allow_decode_fail: bool,
    hstack: bool,
    py: Python<'_>,
) -> PyResult<PyObject> {
    let batch = RecordBatch::from_pyarrow_bound(batch).context("convert batch from pyarrow")?;

    let log_signature = signature.extract::<LogSignature>()?;

    let batch =
        baselib::svm_decode::decode_logs_batch(&log_signature, &batch, allow_decode_fail, hstack)
            .context("decode log batch")?;

    Ok(batch.to_pyarrow(py).context("map result back to pyarrow")?)
}

#[pyfunction]
fn instruction_signature_to_arrow_schema(
    signature: &Bound<'_, PyAny>,
    py: Python<'_>,
) -> PyResult<PyObject> {
    let signature = signature.extract::<InstructionSignature>()?;
    let schema = baselib::svm_decode::instruction_signature_to_arrow_schema(&signature)
        .context("signature to schema")?;

    Ok(schema
        .to_pyarrow(py)
        .context("map result back to pyarrow")?)
}

#[pyfunction]
fn evm_decode_call_inputs(
    signature: &str,
    col: &Bound<'_, PyAny>,
    allow_decode_fail: bool,
    py: Python<'_>,
) -> PyResult<PyObject> {
    let mut col = ArrayData::from_pyarrow_bound(col).context("convert column from pyarrow")?;

    // Align buffers that may be unaligned from Python (e.g. flight client).
    // See: https://github.com/apache/arrow/issues/43552
    col.align_buffers();

    if col.data_type() != &DataType::Binary {
        return Err(anyhow!("unexpected data type {}. Expected Binary", col.data_type()).into());
    }
    let col = BinaryArray::from(col);

    let batch = baselib::evm_decode::decode_call_inputs(signature, &col, allow_decode_fail)
        .context("decode cal inputs")?;

    Ok(batch.to_pyarrow(py).context("map result back to pyarrow")?)
}

#[pyfunction]
fn evm_decode_call_outputs(
    signature: &str,
    col: &Bound<'_, PyAny>,
    allow_decode_fail: bool,
    py: Python<'_>,
) -> PyResult<PyObject> {
    let mut col = ArrayData::from_pyarrow_bound(col).context("convert column from pyarrow")?;

    // Align buffers that may be unaligned from Python (e.g. flight client).
    // See: https://github.com/apache/arrow/issues/43552
    col.align_buffers();

    if col.data_type() != &DataType::Binary {
        return Err(anyhow!("unexpected data type {}. Expected Binary", col.data_type()).into());
    }
    let col = BinaryArray::from(col);

    let batch = baselib::evm_decode::decode_call_outputs(signature, &col, allow_decode_fail)
        .context("decode cal outputs")?;

    Ok(batch.to_pyarrow(py).context("map result back to pyarrow")?)
}

#[pyfunction]
#[pyo3(signature = (signature, batch, allow_decode_fail=false, filter_by_topic0=false, hstack=false))]
fn evm_decode_events(
    signature: &str,
    batch: &Bound<'_, PyAny>,
    allow_decode_fail: bool,
    filter_by_topic0: bool,
    hstack: bool,
    py: Python<'_>,
) -> PyResult<PyObject> {
    let batch = RecordBatch::from_pyarrow_bound(batch).context("convert batch from pyarrow")?;

    let batch = baselib::evm_decode::decode_events(
        signature,
        &batch,
        allow_decode_fail,
        filter_by_topic0,
        hstack,
    )
    .context("decode events")?;

    Ok(batch.to_pyarrow(py).context("map result back to pyarrow")?)
}

#[pyfunction]
fn evm_event_signature_to_arrow_schema(signature: &str, py: Python<'_>) -> PyResult<PyObject> {
    let schema = baselib::evm_decode::event_signature_to_arrow_schema(signature)
        .context("signature to schema")?;

    Ok(schema
        .to_pyarrow(py)
        .context("map result back to pyarrow")?)
}

#[pyfunction]
fn evm_function_signature_to_arrow_schemas(
    signature: &str,
    py: Python<'_>,
) -> PyResult<(PyObject, PyObject)> {
    let (input_schema, output_schema) =
        baselib::evm_decode::function_signature_to_arrow_schemas(signature)
            .context("signature to schemas")?;

    let input_schema = input_schema
        .to_pyarrow(py)
        .context("input schema to pyarrow")?;
    let output_schema = output_schema
        .to_pyarrow(py)
        .context("output schema to pyarrow")?;

    Ok((input_schema, output_schema))
}

#[pyfunction]
fn evm_signature_to_topic0(signature: &str) -> PyResult<String> {
    let topic0 = baselib::evm_decode::signature_to_topic0(signature)?;
    Ok(format!("0x{}", faster_hex::hex_string(topic0.as_slice())))
}

#[pyfunction]
fn evm_abi_to_topic0(abi_json: &str) -> PyResult<String> {
    let topic0 = baselib::evm_decode::abi_to_topic0(abi_json)?;
    Ok(format!("0x{}", faster_hex::hex_string(topic0.as_slice())))
}

#[pyclass(frozen, get_all)]
#[derive(Clone)]
struct EvmAbiEvent {
    name: String,
    name_snake_case: String,
    signature: String,
    selector_signature: String,
    topic0: String,
    abi_json: String,
}

#[pyclass(frozen, get_all)]
#[derive(Clone)]
struct EvmAbiFunction {
    name: String,
    name_snake_case: String,
    signature: String,
    selector_signature: String,
    selector: String,
    abi_json: String,
}

#[pyfunction]
fn evm_abi_events(json_str: &str) -> PyResult<Vec<EvmAbiEvent>> {
    let events = baselib::evm_decode::abi_events(json_str).context("parse abi events")?;
    Ok(events
        .into_iter()
        .map(|e| EvmAbiEvent {
            name: e.name,
            name_snake_case: e.name_snake_case,
            signature: e.signature,
            selector_signature: e.selector_signature,
            topic0: e.topic0,
            abi_json: e.abi_json,
        })
        .collect())
}

#[pyfunction]
fn evm_abi_functions(json_str: &str) -> PyResult<Vec<EvmAbiFunction>> {
    let functions = baselib::evm_decode::abi_functions(json_str).context("parse abi functions")?;
    Ok(functions
        .into_iter()
        .map(|f| EvmAbiFunction {
            name: f.name,
            name_snake_case: f.name_snake_case,
            signature: f.signature,
            selector_signature: f.selector_signature,
            selector: f.selector,
            abi_json: f.abi_json,
        })
        .collect())
}

#[pyfunction]
fn flatten_batch(batch: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<PyObject> {
    let batch = RecordBatch::from_pyarrow_bound(batch).context("convert batch from pyarrow")?;

    let batch = baselib::cast::flatten_record_batch(&batch).context("flatten batch")?;

    Ok(batch.to_pyarrow(py).context("map result back to pyarrow")?)
}

#[pyfunction]
fn base58_encode_bytes(bytes: &[u8]) -> String {
    bs58::encode(bytes)
        .with_alphabet(bs58::Alphabet::BITCOIN)
        .into_string()
}

#[pyfunction]
fn base58_decode_string(s: &str) -> PyResult<Vec<u8>> {
    bs58::decode(s)
        .with_alphabet(bs58::Alphabet::BITCOIN)
        .into_vec()
        .context("decode bs58")
        .map_err(Into::into)
}
