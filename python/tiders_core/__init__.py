"""tiders_core - High-performance blockchain data indexer engine library.

A Rust-powered Python library for ingesting and decoding blockchain data from
EVM (Ethereum) and SVM (Solana) chains. Provides efficient data casting,
encoding/decoding (base58, hex, u256), and ABI/IDL-based decoding using
Apache Arrow columnar format for zero-copy interoperability.
"""

import tiders_core.tiders_core as cc
from . import svm_decode
from typing import Tuple
import pyarrow


def cast(
    map: list[Tuple[str, pyarrow.DataType]],
    data: pyarrow.RecordBatch,
    allow_cast_fail: bool = False,
) -> pyarrow.RecordBatch:
    """Cast columns in a RecordBatch to specified data types.

    Args:
        map: List of (column_name, target_type) pairs defining the desired casts.
        data: The input RecordBatch whose columns will be cast.
        allow_cast_fail: If True, columns that fail to cast are dropped instead of
            raising an error.

    Returns:
        A new RecordBatch with the specified columns cast to their target types.
    """
    return cc.cast(map, data, allow_cast_fail)


def cast_schema(
    map: list[Tuple[str, pyarrow.DataType]], schema: pyarrow.Schema
) -> pyarrow.Schema:
    """Cast field types in a schema to specified data types.

    Args:
        map: List of (field_name, target_type) pairs defining the desired casts.
        schema: The input schema whose fields will be retyped.

    Returns:
        A new Schema with the specified fields cast to their target types.
    """
    return cc.cast_schema(map, schema)


def cast_by_type(
    data: pyarrow.RecordBatch,
    from_type: pyarrow.DataType,
    to_type: pyarrow.DataType,
    allow_cast_fail: bool = False,
) -> pyarrow.RecordBatch:
    """Cast all columns of a given type to a different type.

    Args:
        data: The input RecordBatch.
        from_type: The source data type to match against.
        to_type: The target data type to cast matching columns to.
        allow_cast_fail: If True, columns that fail to cast are dropped instead of
            raising an error.

    Returns:
        A new RecordBatch with all columns matching ``from_type`` cast to ``to_type``.
    """
    return cc.cast_by_type(data, from_type, to_type, allow_cast_fail)


def cast_schema_by_type(
    schema: pyarrow.Schema, from_type: pyarrow.DataType, to_type: pyarrow.DataType
) -> pyarrow.Schema:
    """Cast all fields of a given type in a schema to a different type.

    Args:
        schema: The input schema.
        from_type: The source data type to match against.
        to_type: The target data type for matching fields.

    Returns:
        A new Schema with all fields matching ``from_type`` retyped to ``to_type``.
    """
    return cc.cast_schema_by_type(schema, from_type, to_type)


def base58_encode(data: pyarrow.RecordBatch) -> pyarrow.RecordBatch:
    """Base58-encode all binary columns in a RecordBatch.

    Converts every binary column to its Base58 string representation using the
    Bitcoin alphabet.

    Args:
        data: A RecordBatch containing binary columns to encode.

    Returns:
        A new RecordBatch with binary columns replaced by UTF-8 string columns.
    """
    return cc.base58_encode(data)


def hex_encode(data: pyarrow.RecordBatch) -> pyarrow.RecordBatch:
    """Hex-encode all binary columns in a RecordBatch (no ``0x`` prefix).

    Args:
        data: A RecordBatch containing binary columns to encode.

    Returns:
        A new RecordBatch with binary columns replaced by hex-encoded UTF-8 string columns.
    """
    return cc.hex_encode(data)


def prefix_hex_encode(data: pyarrow.RecordBatch) -> pyarrow.RecordBatch:
    """Hex-encode all binary columns in a RecordBatch with a ``0x`` prefix.

    Args:
        data: A RecordBatch containing binary columns to encode.

    Returns:
        A new RecordBatch with binary columns replaced by ``0x``-prefixed hex string columns.
    """
    return cc.prefix_hex_encode(data)


def base58_encode_column(col: pyarrow.Array) -> pyarrow.Array:
    """Base58-encode a single binary column.

    Args:
        col: A PyArrow Binary array to encode.

    Returns:
        A UTF-8 string array with Base58-encoded values.
    """
    return cc.base58_encode_column(col)


def hex_encode_column(col: pyarrow.Array) -> pyarrow.Array:
    """Hex-encode a single binary column (no ``0x`` prefix).

    Args:
        col: A PyArrow Binary or LargeBinary array to encode.

    Returns:
        A UTF-8 string array with hex-encoded values.
    """
    return cc.hex_encode_column(col)


def prefix_hex_encode_column(col: pyarrow.Array) -> pyarrow.Array:
    """Hex-encode a single binary column with a ``0x`` prefix.

    Args:
        col: A PyArrow Binary or LargeBinary array to encode.

    Returns:
        A UTF-8 string array with ``0x``-prefixed hex-encoded values.
    """
    return cc.prefix_hex_encode_column(col)


def base58_decode_column(col: pyarrow.Array) -> pyarrow.Array:
    """Decode a Base58-encoded string column to binary.

    Args:
        col: A PyArrow Utf8 or LargeUtf8 array containing Base58 strings.

    Returns:
        A Binary array with decoded byte values.
    """
    return cc.base58_decode_column(col)


def hex_decode_column(col: pyarrow.Array) -> pyarrow.Array:
    """Decode a hex-encoded string column to binary (no ``0x`` prefix expected).

    Args:
        col: A PyArrow Utf8 or LargeUtf8 array containing hex strings.

    Returns:
        A Binary array with decoded byte values.
    """
    return cc.hex_decode_column(col)


def prefix_hex_decode_column(col: pyarrow.Array) -> pyarrow.Array:
    """Decode a ``0x``-prefixed hex string column to binary.

    Args:
        col: A PyArrow Utf8 or LargeUtf8 array containing ``0x``-prefixed hex strings.

    Returns:
        A Binary array with decoded byte values.
    """
    return cc.prefix_hex_decode_column(col)


def u256_column_from_binary(col: pyarrow.Array) -> pyarrow.Array:
    """Convert a binary column to Decimal256 representation.

    Args:
        col: A PyArrow Binary array containing 32-byte big-endian unsigned integers.

    Returns:
        A Decimal256 array representing the u256 values.
    """
    return cc.u256_column_from_binary(col)


def u256_column_to_binary(col: pyarrow.Array) -> pyarrow.Array:
    """Convert a Decimal256 column to binary representation.

    Args:
        col: A PyArrow Decimal256 array.

    Returns:
        A Binary array containing 32-byte big-endian unsigned integer representations.
    """
    return cc.u256_column_to_binary(col)


def flatten_batch(data: pyarrow.RecordBatch) -> pyarrow.RecordBatch:
    """Flatten all Struct columns in a RecordBatch into top-level columns.

    Recurses into nested Struct types, dot-joining names at each level.
    ``FixedSizeList(n, Struct(...))`` expands to n columns named ``parent.0``,
    ``parent.1``, … each further expanded if they contain structs.
    ``List(Struct(...))`` (variable-length) is serialised to a UTF-8 string column.
    All other column types are passed through unchanged.

    When two expanded names collide, the second occurrence is renamed ``name_1``,
    the third ``name_2``, and so on (the first occurrence keeps its original name).

    Args:
        data: A RecordBatch potentially containing nested Struct columns.

    Returns:
        A new RecordBatch with all Struct columns expanded to top-level columns.
    """
    return cc.flatten_batch(data)


def u256_to_binary(data: pyarrow.RecordBatch) -> pyarrow.RecordBatch:
    """Convert all Decimal256 columns in a RecordBatch to binary.

    Args:
        data: A RecordBatch containing Decimal256 columns.

    Returns:
        A new RecordBatch with Decimal256 columns replaced by Binary columns.
    """
    return cc.u256_to_binary(data)


def svm_decode_instructions(
    signature: svm_decode.InstructionSignature,
    batch: pyarrow.RecordBatch,
    allow_decode_fail: bool = False,
    filter_by_discriminator: bool = False,
    hstack: bool = False,
) -> pyarrow.RecordBatch:
    """Decode Solana (SVM) instructions from a RecordBatch using an instruction signature.

    Parses the raw instruction data and accounts according to the provided IDL-based
    signature definition, producing decoded parameter columns.

    Args:
        signature: The instruction signature defining the discriminator, parameters,
            and account names for decoding.
        batch: A RecordBatch containing raw instruction data (must include a ``data``
            column with binary-encoded instruction bytes).
        allow_decode_fail: If True, rows that fail to decode are filled with nulls
            instead of raising an error.
        filter_by_discriminator: If True, only rows whose instruction data starts with
            the expected discriminator are decoded. Non-matching rows are filtered out.
        hstack: If True, the original input columns are appended alongside the decoded
            columns in the output.

    Returns:
        A new RecordBatch with decoded instruction parameters and named account columns.
        When ``hstack`` is True, original input columns are also included.
    """
    return cc.svm_decode_instructions(
        signature, batch, allow_decode_fail, filter_by_discriminator, hstack
    )


def svm_decode_logs(
    signature: svm_decode.LogSignature,
    batch: pyarrow.RecordBatch,
    allow_decode_fail: bool = False,
    hstack: bool = False,
) -> pyarrow.RecordBatch:
    """Decode Solana (SVM) program logs from a RecordBatch using a log signature.

    Parses raw log event data according to the provided signature definition,
    producing decoded parameter columns.

    Args:
        signature: The log signature defining the expected parameter types.
        batch: A RecordBatch containing raw log data.
        allow_decode_fail: If True, rows that fail to decode are filled with nulls
            instead of raising an error.
        hstack: If True, the original input columns are appended alongside the decoded
            columns in the output.

    Returns:
        A new RecordBatch with decoded log event parameters.
        When ``hstack`` is True, original input columns are also included.
    """
    return cc.svm_decode_logs(signature, batch, allow_decode_fail, hstack)


def instruction_signature_to_arrow_schema(
    signature: svm_decode.InstructionSignature,
) -> pyarrow.Schema:
    """Convert an SVM instruction signature to an Arrow schema.

    Useful for understanding the output schema before decoding, or for creating
    empty tables with the correct types.

    Args:
        signature: The instruction signature to convert.

    Returns:
        A PyArrow Schema describing the columns that ``svm_decode_instructions``
        would produce for this signature.
    """
    return cc.instruction_signature_to_arrow_schema(signature)


def evm_decode_call_inputs(
    signature: str, data: pyarrow.Array, allow_decode_fail: bool = False
) -> pyarrow.RecordBatch:
    """Decode EVM function call input data using an ABI function signature.

    Args:
        signature: The Solidity function signature (e.g. ``"transfer(address,uint256)"``).
        data: A Binary array containing ABI-encoded call input data (including the
            4-byte selector).
        allow_decode_fail: If True, rows that fail to decode are filled with nulls
            instead of raising an error.

    Returns:
        A RecordBatch with one column per function input parameter.
    """
    return cc.evm_decode_call_inputs(signature, data, allow_decode_fail)


def evm_decode_call_outputs(
    signature: str, data: pyarrow.Array, allow_decode_fail: bool = False
) -> pyarrow.RecordBatch:
    """Decode EVM function call output (return) data using an ABI function signature.

    Args:
        signature: The Solidity function signature (e.g. ``"balanceOf(address)"``).
        data: A Binary array containing ABI-encoded return data.
        allow_decode_fail: If True, rows that fail to decode are filled with nulls
            instead of raising an error.

    Returns:
        A RecordBatch with one column per function output parameter.
    """
    return cc.evm_decode_call_outputs(signature, data, allow_decode_fail)


def evm_decode_events(
    signature: str,
    data: pyarrow.RecordBatch,
    allow_decode_fail: bool = False,
    filter_by_topic0: bool = False,
    hstack: bool = False,
) -> pyarrow.RecordBatch:
    """Decode EVM event log data using an ABI event signature.

    Args:
        signature: The Solidity event signature
            (e.g. ``"Transfer(address indexed,address indexed,uint256)"``).
        data: A RecordBatch containing raw log data with topic and data columns.
        allow_decode_fail: If True, rows that fail to decode are filled with nulls
            instead of raising an error.
        filter_by_topic0: If True, only rows whose ``topic0`` matches the event's
            selector are decoded. Non-matching rows are silently filtered out.
        hstack: If True, the original input columns (after any topic0 filtering)
            are appended alongside the decoded columns in the output.

    Returns:
        A RecordBatch with one column per event parameter (both indexed and non-indexed).
        When ``hstack`` is True, original input columns are also included.
    """
    return cc.evm_decode_events(
        signature, data, allow_decode_fail, filter_by_topic0, hstack
    )


def evm_event_signature_to_arrow_schema(signature: str) -> pyarrow.Schema:
    """Convert an EVM event signature to an Arrow schema.

    Args:
        signature: The Solidity event signature
            (e.g. ``"Transfer(address indexed,address indexed,uint256)"``).

    Returns:
        A PyArrow Schema describing the columns that ``evm_decode_events`` would produce.
    """
    return cc.evm_event_signature_to_arrow_schema(signature)


def evm_transaction_signature_to_arrow_schemas(
    signature: str,
) -> Tuple[pyarrow.Schema, pyarrow.Schema]:
    """Convert an EVM function signature to Arrow schemas for inputs and outputs.

    Args:
        signature: The Solidity function signature (e.g. ``"transfer(address,uint256)"``).

    Returns:
        A tuple of (input_schema, output_schema) where each is a PyArrow Schema
        describing the decoded columns for function inputs and outputs respectively.
    """
    return cc.evm_transaction_signature_to_arrow_schemas(signature)


def evm_signature_to_topic0(signature: str) -> str:
    """Compute the topic0 (keccak256 hash) for an EVM event signature.

    Args:
        signature: The human-readable Solidity event signature
            (e.g. ``"Transfer(address,address,uint256)"``).

    Returns:
        The ``0x``-prefixed hex-encoded keccak256 hash of the event signature.
    """
    return cc.evm_signature_to_topic0(signature)


def evm_abi_to_topic0(abi_json: str) -> str:
    """Compute the topic0 (keccak256 hash) from a JSON ABI fragment for an event.

    Args:
        abi_json: A JSON ABI fragment string for a single event
            (e.g. the ``abi_json`` field from :func:`evm_abi_events`).

    Returns:
        The ``0x``-prefixed hex-encoded keccak256 hash of the event signature.
    """
    return cc.evm_abi_to_topic0(abi_json)


class EvmAbiEvent:
    """Parsed event info extracted from a JSON ABI.

    Attributes:
        name: Event name (e.g. ``"Swap"``).
        name_snake_case: Event name in snake_case (e.g. ``"swap"``).
        signature: Human-readable signature with outer param names and indexed markers
            but unnamed tuple components
            (e.g. ``"Swap(address indexed sender, (int256,int256) data)"``).
            Can be passed directly to :func:`evm_decode_events`.
        selector_signature: Canonical selector signature without names
            (e.g. ``"Swap(address,address,int256,int256,uint160,uint128,int24)"``).
        topic0: topic0 as ``0x``-prefixed hex string.
        abi_json: Full JSON ABI fragment preserving all component names.
            Can be passed to :func:`evm_abi_to_topic0` or :func:`evm_decode_events`.
    """

    name: str
    name_snake_case: str
    signature: str
    selector_signature: str
    topic0: str
    abi_json: str


class EvmAbiFunction:
    """Parsed function info extracted from a JSON ABI.

    Attributes:
        name: Function name (e.g. ``"swap"``).
        name_snake_case: Function name in snake_case (e.g. ``"swap"``).
        signature: Human-readable signature with names
            (e.g. ``"swap(address recipient, bool zeroForOne, int256 amountSpecified, ...)"``).
        selector_signature: Canonical selector signature without names
            (e.g. ``"swap(address,bool,int256,uint160,bytes)"``).
        selector: 4-byte selector as ``0x``-prefixed hex string.
        abi_json: Full JSON ABI fragment preserving all parameter names.
    """

    name: str
    name_snake_case: str
    signature: str
    selector_signature: str
    selector: str
    abi_json: str


def evm_abi_events(json_str: str) -> list[EvmAbiEvent]:
    """Parse a JSON ABI string and extract all events.

    Args:
        json_str: The full JSON ABI as a string (e.g. read from a ``.json`` file).

    Returns:
        A list of :class:`EvmAbiEvent` objects, one per event in the ABI.
        Each contains the event name, human-readable signature (suitable for
        :func:`evm_decode_events`), canonical selector signature, topic0 hash,
        and a full JSON ABI fragment (suitable for :func:`evm_abi_to_topic0`).
    """
    return cc.evm_abi_events(json_str)


def evm_abi_functions(json_str: str) -> list[EvmAbiFunction]:
    """Parse a JSON ABI string and extract all functions.

    Args:
        json_str: The full JSON ABI as a string (e.g. read from a ``.json`` file).

    Returns:
        A list of :class:`EvmAbiFunction` objects, one per function in the ABI.
        Each contains the function name, human-readable signature, canonical
        selector signature, and 4-byte selector hash.
    """
    return cc.evm_abi_functions(json_str)


def base58_encode_bytes(b: bytes) -> str:
    """Base58-encode a single bytes value using the Bitcoin alphabet.

    Args:
        b: The raw bytes to encode.

    Returns:
        The Base58-encoded string.
    """
    return cc.base58_encode_bytes(b)


def base58_decode_string(s: str) -> bytes:
    """Decode a Base58-encoded string to bytes using the Bitcoin alphabet.

    Args:
        s: The Base58-encoded string.

    Returns:
        The decoded raw bytes.
    """
    return cc.base58_decode_string(s)
