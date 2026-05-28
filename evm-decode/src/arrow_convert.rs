use std::sync::Arc;

use alloy_dyn_abi::{DynSolType, DynSolValue, Specifier};
use alloy_json_abi::{EventParam, Param};
use alloy_primitives::{I256, U256};
use anyhow::{anyhow, Context, Result};
use arrow::{
    array::{
        builder, Array, ArrowPrimitiveType, BooleanArray, FixedSizeListArray, GenericBinaryArray,
        ListArray, OffsetSizeTrait, RecordBatch, StructArray,
    },
    buffer::{NullBuffer, OffsetBuffer},
    datatypes::{
        DataType, Field, Fields, Int16Type, Int32Type, Int64Type, Int8Type, Schema, UInt16Type,
        UInt32Type, UInt64Type, UInt8Type,
    },
};

/// Maps a Solidity dynamic type to its corresponding Arrow data type.
///
/// Handles nested types recursively: tuples become Struct, arrays become List.
/// Integer types are mapped to the smallest Arrow type that fits the bit width,
/// with types >64 bits using Decimal128/Decimal256.
///
/// When `large_int_as_binary` is `true`, signed and unsigned integers wider than
/// 64 bits (i.e. `int128`/`int256`/`uint128`/`uint256`) are mapped to
/// `DataType::Binary` (32-byte big-endian, two's-complement for signed) instead
/// of `Decimal128`/`Decimal256`.
pub(crate) fn to_arrow_dtype(sol_type: &DynSolType, large_int_as_binary: bool) -> Result<DataType> {
    match sol_type {
        DynSolType::Bool => Ok(DataType::Boolean),
        DynSolType::Bytes | DynSolType::Address | DynSolType::FixedBytes(_) => Ok(DataType::Binary),
        DynSolType::String => Ok(DataType::Utf8),
        DynSolType::Int(num_bits) => {
            if large_int_as_binary && *num_bits > 64 {
                Ok(DataType::Binary)
            } else {
                Ok(num_bits_to_int_type(*num_bits))
            }
        }
        DynSolType::Uint(num_bits) => {
            if large_int_as_binary && *num_bits > 64 {
                Ok(DataType::Binary)
            } else {
                Ok(num_bits_to_uint_type(*num_bits))
            }
        }
        DynSolType::Array(inner_type) => {
            let inner_type = to_arrow_dtype(inner_type, large_int_as_binary).context("map inner")?;
            Ok(DataType::List(Arc::new(Field::new("", inner_type, true))))
        }
        DynSolType::FixedArray(inner_type, n) => {
            let inner_type = to_arrow_dtype(inner_type, large_int_as_binary).context("map inner")?;
            Ok(DataType::FixedSizeList(
                Arc::new(Field::new("", inner_type, true)),
                i32::try_from(*n).context("fixed array size exceeds i32")?,
            ))
        }
        DynSolType::Function => Err(anyhow!(
            "decoding 'Function' typed value in function signature isn't supported."
        )),
        DynSolType::Tuple(fields) => {
            let mut arrow_fields = Vec::<Arc<Field>>::with_capacity(fields.len());

            for (i, f) in fields.iter().enumerate() {
                let inner_dt = to_arrow_dtype(f, large_int_as_binary).context("map field dt")?;
                arrow_fields.push(Arc::new(Field::new(format!("param{i}"), inner_dt, true)));
            }

            Ok(DataType::Struct(Fields::from(arrow_fields)))
        }
    }
}

/// Like [`to_arrow_dtype`] but uses [`Param`] component names for tuple fields.
///
/// For a plain `tuple` type the components are mapped to a named Arrow `Struct`.
/// For arrays of tuples (`tuple[]`, `tuple[N]`) the component names are not
/// preserved in the inner type — they fall through to [`to_arrow_dtype`] so the
/// schema and the data arrays produced by [`decode_body_named`] remain consistent.
pub(crate) fn param_to_arrow_dtype(
    ty: &str,
    components: &[Param],
    large_int_as_binary: bool,
) -> Result<DataType> {
    if ty == "tuple" && !components.is_empty() {
        let fields = components
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let inner = param_to_arrow_dtype(&c.ty, &c.components, large_int_as_binary)?;
                let name = if c.name.is_empty() {
                    format!("param{i}")
                } else {
                    c.name.clone()
                };
                Ok(Arc::new(Field::new(name, inner, true)))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(DataType::Struct(Fields::from(fields)))
    } else if !components.is_empty() {
        // Array-of-tuple or other complex type: use DynSolType resolution so the
        // schema matches the unnamed fields produced by to_arrow / to_struct.
        let p = Param {
            ty: ty.to_string(),
            name: String::new(),
            components: components.to_vec(),
            internal_type: None,
        };
        let sol_type = p.resolve().map_err(|e| anyhow!("{e}"))?;
        to_arrow_dtype(&sol_type, large_int_as_binary)
    } else {
        let sol_type = DynSolType::parse(ty).map_err(|e| anyhow!("{e}"))?;
        to_arrow_dtype(&sol_type, large_int_as_binary)
    }
}

/// Maps a Solidity unsigned integer bit width to the smallest Arrow data type that fits.
pub(crate) fn num_bits_to_uint_type(num_bits: usize) -> DataType {
    if num_bits <= 8 {
        DataType::UInt8
    } else if num_bits <= 16 {
        DataType::UInt16
    } else if num_bits <= 32 {
        DataType::UInt32
    } else if num_bits <= 64 {
        DataType::UInt64
    } else if num_bits <= 128 {
        DataType::Decimal128(38, 0)
    } else if num_bits <= 256 {
        DataType::Decimal256(76, 0)
    } else {
        unreachable!()
    }
}

/// Maps a Solidity signed integer bit width to the smallest Arrow data type that fits.
pub(crate) fn num_bits_to_int_type(num_bits: usize) -> DataType {
    if num_bits <= 8 {
        DataType::Int8
    } else if num_bits <= 16 {
        DataType::Int16
    } else if num_bits <= 32 {
        DataType::Int32
    } else if num_bits <= 64 {
        DataType::Int64
    } else if num_bits <= 128 {
        DataType::Decimal128(38, 0)
    } else if num_bits <= 256 {
        DataType::Decimal256(76, 0)
    } else {
        unreachable!()
    }
}

/// Converts a column of decoded Solidity values into an Arrow array.
///
/// Dispatches to type-specific builders based on the Solidity type. Handles
/// nested types (tuples, arrays) by recursive decomposition.
pub(crate) fn to_arrow(
    sol_type: &DynSolType,
    sol_values: Vec<Option<DynSolValue>>,
    allow_decode_fail: bool,
    large_int_as_binary: bool,
) -> Result<Arc<dyn Array>> {
    match sol_type {
        DynSolType::Bool => to_bool(&sol_values),
        DynSolType::Bytes | DynSolType::Address | DynSolType::FixedBytes(_) => {
            to_binary(&sol_values)
        }
        DynSolType::String => to_string(&sol_values),
        DynSolType::Int(num_bits) => {
            if large_int_as_binary && *num_bits > 64 {
                to_binary_from_int_word(*num_bits, &sol_values)
            } else {
                to_int(*num_bits, &sol_values, allow_decode_fail)
            }
        }
        DynSolType::Uint(num_bits) => {
            if large_int_as_binary && *num_bits > 64 {
                to_binary_from_int_word(*num_bits, &sol_values)
            } else {
                to_uint(*num_bits, &sol_values, allow_decode_fail)
            }
        }
        DynSolType::Array(inner_type) => {
            to_list(inner_type, sol_values, allow_decode_fail, large_int_as_binary)
        }
        DynSolType::FixedArray(inner_type, n) => to_fixed_list(
            inner_type,
            *n,
            sol_values,
            allow_decode_fail,
            large_int_as_binary,
        ),
        DynSolType::Function => Err(anyhow!(
            "decoding 'Function' typed value in function signature isn't supported."
        )),
        DynSolType::Tuple(fields) => {
            to_struct(fields, sol_values, allow_decode_fail, large_int_as_binary)
        }
    }
}

fn to_int(
    num_bits: usize,
    sol_values: &[Option<DynSolValue>],
    allow_decode_fail: bool,
) -> Result<Arc<dyn Array>> {
    match num_bits_to_int_type(num_bits) {
        DataType::Int8 => to_int_impl::<Int8Type>(num_bits, sol_values, allow_decode_fail),
        DataType::Int16 => to_int_impl::<Int16Type>(num_bits, sol_values, allow_decode_fail),
        DataType::Int32 => to_int_impl::<Int32Type>(num_bits, sol_values, allow_decode_fail),
        DataType::Int64 => to_int_impl::<Int64Type>(num_bits, sol_values, allow_decode_fail),
        DataType::Decimal128(_, _) => to_decimal128(num_bits, sol_values),
        DataType::Decimal256(_, _) => to_decimal256(num_bits, sol_values),
        dt => Err(anyhow!("unexpected int data type: {dt:?}")),
    }
}

fn to_uint(
    num_bits: usize,
    sol_values: &[Option<DynSolValue>],
    allow_decode_fail: bool,
) -> Result<Arc<dyn Array>> {
    match num_bits_to_uint_type(num_bits) {
        DataType::UInt8 => to_int_impl::<UInt8Type>(num_bits, sol_values, allow_decode_fail),
        DataType::UInt16 => to_int_impl::<UInt16Type>(num_bits, sol_values, allow_decode_fail),
        DataType::UInt32 => to_int_impl::<UInt32Type>(num_bits, sol_values, allow_decode_fail),
        DataType::UInt64 => to_int_impl::<UInt64Type>(num_bits, sol_values, allow_decode_fail),
        DataType::Decimal128(_, _) => to_decimal128(num_bits, sol_values),
        DataType::Decimal256(_, _) => to_decimal256(num_bits, sol_values),
        dt => Err(anyhow!("unexpected uint data type: {dt:?}")),
    }
}

/// Writes wide signed/unsigned integers (>64 bits) as 32-byte big-endian Binary,
/// matching the on-wire ABI word layout. Two's-complement is preserved for signed
/// values, so the high 16 bytes of a negative `int128` are `0xFF` (sign extension).
/// Used when `large_int_as_binary` is enabled to keep `int128`/`int256`/
/// `uint128`/`uint256` values losslessly accessible without going through Arrow's
/// signed `Decimal128`/`Decimal256`.
fn to_binary_from_int_word(
    num_bits: usize,
    sol_values: &[Option<DynSolValue>],
) -> Result<Arc<dyn Array>> {
    let mut builder = builder::BinaryBuilder::new();

    for val in sol_values {
        match val {
            Some(DynSolValue::Int(v, nb)) => {
                if num_bits != *nb {
                    return Err(anyhow!("bit width mismatch: expected {num_bits}, got {nb}"));
                }
                builder.append_value(v.to_be_bytes::<32>());
            }
            Some(DynSolValue::Uint(v, nb)) => {
                if num_bits != *nb {
                    return Err(anyhow!("bit width mismatch: expected {num_bits}, got {nb}"));
                }
                builder.append_value(v.to_be_bytes::<32>());
            }
            Some(other) => {
                return Err(anyhow!(
                    "found unexpected value. Expected: int/uint, Found: {other:?}"
                ));
            }
            None => {
                builder.append_null();
            }
        }
    }

    Ok(Arc::new(builder.finish()))
}

/// Converts `int128`/`uint128` values into Arrow `Decimal128(38, 0)`.
///
/// Reinterprets the 32-byte big-endian word as `i128` by taking the low 16 bytes.
/// For `int128` two's-complement, the high 16 bytes carry sign extension (`0x00`
/// for positive, `0xFF` for negative) and the low 16 bytes are exactly the
/// `i128` bit pattern. For `uint128`, the high 16 bytes are always zero.
///
/// `uint128` values in `[2^127, 2^128 - 1]` will appear negative when read as
/// signed `Decimal128`; the bit pattern is preserved, callers must reinterpret
/// if they need unsigned semantics (or use `large_int_as_binary`).
fn to_decimal128(num_bits: usize, sol_values: &[Option<DynSolValue>]) -> Result<Arc<dyn Array>> {
    let mut builder = builder::Decimal128Builder::new();

    for val in sol_values {
        match val {
            Some(DynSolValue::Int(v, nb)) => {
                if num_bits != *nb {
                    return Err(anyhow!("bit width mismatch: expected {num_bits}, got {nb}"));
                }
                let bytes = v.to_be_bytes::<32>();
                let lo: [u8; 16] = bytes[16..].try_into().unwrap();
                builder.append_value(i128::from_be_bytes(lo));
            }
            Some(DynSolValue::Uint(v, nb)) => {
                if num_bits != *nb {
                    return Err(anyhow!("bit width mismatch: expected {num_bits}, got {nb}"));
                }
                let bytes = v.to_be_bytes::<32>();
                let lo: [u8; 16] = bytes[16..].try_into().unwrap();
                builder.append_value(i128::from_be_bytes(lo));
            }
            Some(other) => {
                return Err(anyhow!(
                    "found unexpected value. Expected: int/uint, Found: {other:?}"
                ));
            }
            None => {
                builder.append_null();
            }
        }
    }

    builder = builder.with_data_type(DataType::Decimal128(38, 0));

    Ok(Arc::new(builder.finish()))
}

/// Converts `int256`/`uint256` values into Arrow `Decimal256(76, 0)`.
///
/// Reinterprets the 32-byte big-endian word as `i256` bit-for-bit.
///
/// `uint256` values in `[2^255, 2^256 - 1]` will appear negative when read as
/// signed `Decimal256`; the bit pattern is preserved, callers must reinterpret
/// if they need unsigned semantics (or use `large_int_as_binary`).
fn to_decimal256(num_bits: usize, sol_values: &[Option<DynSolValue>]) -> Result<Arc<dyn Array>> {
    let mut builder = builder::Decimal256Builder::new();

    for val in sol_values {
        match val {
            Some(DynSolValue::Int(v, nb)) => {
                if num_bits != *nb {
                    return Err(anyhow!("bit width mismatch: expected {num_bits}, got {nb}"));
                }
                let v = arrow::datatypes::i256::from_be_bytes(v.to_be_bytes::<32>());
                builder.append_value(v);
            }
            Some(DynSolValue::Uint(v, nb)) => {
                if num_bits != *nb {
                    return Err(anyhow!("bit width mismatch: expected {num_bits}, got {nb}"));
                }
                let v = arrow::datatypes::i256::from_be_bytes(v.to_be_bytes::<32>());
                builder.append_value(v);
            }
            Some(other) => {
                return Err(anyhow!(
                    "found unexpected value. Expected: int/uint, Found: {other:?}"
                ));
            }
            None => {
                builder.append_null();
            }
        }
    }

    builder = builder.with_data_type(DataType::Decimal256(76, 0));

    Ok(Arc::new(builder.finish()))
}

pub(crate) fn to_int_impl<T>(
    num_bits: usize,
    sol_values: &[Option<DynSolValue>],
    allow_decode_fail: bool,
) -> Result<Arc<dyn Array>>
where
    T: ArrowPrimitiveType,
    T::Native: TryFrom<I256> + TryFrom<U256>,
{
    let mut builder = builder::PrimitiveBuilder::<T>::new();

    for val in sol_values {
        match val {
            Some(val) => match val {
                DynSolValue::Int(v, nb) => {
                    if num_bits != *nb {
                        return Err(anyhow!("bit width mismatch: expected {num_bits}, got {nb}"));
                    }
                    match T::Native::try_from(*v) {
                        Ok(native) => builder.append_value(native),
                        Err(_) if allow_decode_fail => {
                            log::debug!("failed to convert int value {v} to native type");
                            builder.append_null();
                        }
                        Err(_) => {
                            return Err(anyhow!("failed to convert int value to native type"));
                        }
                    }
                }
                DynSolValue::Uint(v, nb) => {
                    if num_bits != *nb {
                        return Err(anyhow!("bit width mismatch: expected {num_bits}, got {nb}"));
                    }
                    match T::Native::try_from(*v) {
                        Ok(native) => builder.append_value(native),
                        Err(_) if allow_decode_fail => {
                            log::debug!("failed to convert uint value {v} to native type");
                            builder.append_null();
                        }
                        Err(_) => {
                            return Err(anyhow!("failed to convert uint value to native type"));
                        }
                    }
                }
                _ => {
                    return Err(anyhow!(
                        "found unexpected value. Expected: int/uint, Found: {val:?}"
                    ));
                }
            },
            None => {
                builder.append_null();
            }
        }
    }

    Ok(Arc::new(builder.finish()))
}

fn to_list(
    sol_type: &DynSolType,
    sol_values: Vec<Option<DynSolValue>>,
    allow_decode_fail: bool,
    large_int_as_binary: bool,
) -> Result<Arc<dyn Array>> {
    let mut lengths = Vec::with_capacity(sol_values.len());
    let mut values = Vec::with_capacity(sol_values.len() * 2);
    let mut validity = Vec::with_capacity(sol_values.len() * 2);

    let mut all_valid = true;

    for val in sol_values {
        if let Some(val) = val {
            match val {
                DynSolValue::Array(inner_vals) | DynSolValue::FixedArray(inner_vals) => {
                    lengths.push(inner_vals.len());
                    values.extend(inner_vals.into_iter().map(Some));
                    validity.push(true);
                }
                _ => {
                    return Err(anyhow!(
                        "found unexpected value. Expected list type, Found: {val:?}"
                    ));
                }
            }
        } else {
            lengths.push(0);
            validity.push(false);
            all_valid = false;
        }
    }

    let values = to_arrow(sol_type, values, allow_decode_fail, large_int_as_binary)
        .context("map inner")?;
    let field = Field::new(
        "",
        to_arrow_dtype(sol_type, large_int_as_binary).context("construct data type")?,
        true,
    );
    let list_arr = ListArray::try_new(
        Arc::new(field),
        OffsetBuffer::from_lengths(lengths),
        values,
        if all_valid {
            None
        } else {
            Some(NullBuffer::from(validity))
        },
    )
    .context("construct list array")?;
    Ok(Arc::new(list_arr))
}

fn to_fixed_list(
    sol_type: &DynSolType,
    n: usize,
    sol_values: Vec<Option<DynSolValue>>,
    allow_decode_fail: bool,
    large_int_as_binary: bool,
) -> Result<Arc<dyn Array>> {
    let mut values = Vec::with_capacity(sol_values.len() * n);
    let mut validity = Vec::with_capacity(sol_values.len());
    let mut all_valid = true;

    for val in sol_values {
        match val {
            Some(DynSolValue::FixedArray(inner_vals)) => {
                if inner_vals.len() != n {
                    return Err(anyhow!(
                        "fixed array length mismatch: expected {n}, got {}",
                        inner_vals.len()
                    ));
                }
                values.extend(inner_vals.into_iter().map(Some));
                validity.push(true);
            }
            Some(other) => {
                return Err(anyhow!(
                    "found unexpected value. Expected: FixedArray, Found: {other:?}"
                ));
            }
            None => {
                for _ in 0..n {
                    values.push(None);
                }
                validity.push(false);
                all_valid = false;
            }
        }
    }

    let inner_values = to_arrow(sol_type, values, allow_decode_fail, large_int_as_binary)
        .context("map inner")?;
    let field = Arc::new(Field::new(
        "",
        to_arrow_dtype(sol_type, large_int_as_binary).context("construct data type")?,
        true,
    ));
    let null_buf = if all_valid {
        None
    } else {
        Some(NullBuffer::from(validity))
    };

    let list_arr = FixedSizeListArray::try_new(
        field,
        i32::try_from(n).context("array size exceeds i32")?,
        inner_values,
        null_buf,
    )
    .context("construct fixed size list array")?;
    Ok(Arc::new(list_arr))
}

fn to_struct(
    fields: &[DynSolType],
    sol_values: Vec<Option<DynSolValue>>,
    allow_decode_fail: bool,
    large_int_as_binary: bool,
) -> Result<Arc<dyn Array>> {
    // Handle empty tuple (e.g. events where all params are indexed and body is empty)
    if fields.is_empty() {
        return Ok(Arc::new(StructArray::new_empty_fields(
            sol_values.len(),
            None,
        )));
    }

    let mut values = vec![Vec::with_capacity(sol_values.len()); fields.len()];

    for val in sol_values {
        match val {
            Some(val) => match val {
                DynSolValue::Tuple(inner_vals) => {
                    if values.len() != inner_vals.len() {
                        let expected = values.len();
                        let found = inner_vals.len();
                        return Err(anyhow!(
                            "found unexpected length tuple value. Expected: {expected}, Found: {found}"
                        ));
                    }
                    for (v, inner) in values.iter_mut().zip(inner_vals) {
                        v.push(Some(inner.clone()));
                    }
                }
                _ => {
                    return Err(anyhow!(
                        "found unexpected value. Expected: tuple, Found: {val:?}"
                    ));
                }
            },
            None => {
                for v in &mut values {
                    v.push(None);
                }
            }
        }
    }

    let mut arrays = Vec::with_capacity(fields.len());

    for (sol_type, arr_vals) in fields.iter().zip(values.into_iter()) {
        arrays.push(to_arrow(
            sol_type,
            arr_vals,
            allow_decode_fail,
            large_int_as_binary,
        )?);
    }

    let fields = arrays
        .iter()
        .enumerate()
        .map(|(i, arr)| Field::new(format!("param{i}"), arr.data_type().clone(), true))
        .collect::<Vec<_>>();
    let schema = Arc::new(Schema::new(fields));

    let batch = RecordBatch::try_new(schema, arrays).context("construct record batch")?;

    Ok(Arc::new(StructArray::from(batch)))
}

fn to_bool(sol_values: &[Option<DynSolValue>]) -> Result<Arc<dyn Array>> {
    let mut builder = builder::BooleanBuilder::new();

    for val in sol_values {
        match val {
            Some(val) => match val {
                DynSolValue::Bool(b) => {
                    builder.append_value(*b);
                }
                _ => {
                    return Err(anyhow!(
                        "found unexpected value. Expected: bool, Found: {val:?}"
                    ));
                }
            },
            None => {
                builder.append_null();
            }
        }
    }

    Ok(Arc::new(builder.finish()))
}

fn to_binary(sol_values: &[Option<DynSolValue>]) -> Result<Arc<dyn Array>> {
    let mut builder = builder::BinaryBuilder::new();

    for val in sol_values {
        match val {
            Some(val) => match val {
                DynSolValue::Bytes(data) => {
                    builder.append_value(data);
                }
                DynSolValue::FixedBytes(data, _) => {
                    builder.append_value(data);
                }
                DynSolValue::Address(data) => {
                    builder.append_value(data);
                }
                DynSolValue::Uint(v, _) => {
                    builder.append_value(v.to_be_bytes::<32>());
                }
                DynSolValue::Int(v, _) => {
                    builder.append_value(v.to_be_bytes::<32>());
                }
                _ => {
                    return Err(anyhow!(
                        "found unexpected value. Expected a binary type, Found: {val:?}"
                    ));
                }
            },
            None => {
                builder.append_null();
            }
        }
    }

    Ok(Arc::new(builder.finish()))
}

fn to_string(sol_values: &[Option<DynSolValue>]) -> Result<Arc<dyn Array>> {
    let mut builder = builder::StringBuilder::new();

    for val in sol_values {
        match val {
            Some(val) => match val {
                DynSolValue::String(s) => {
                    builder.append_value(s);
                }
                _ => {
                    return Err(anyhow!(
                        "found unexpected value. Expected string, Found: {val:?}"
                    ));
                }
            },
            None => {
                builder.append_null();
            }
        }
    }

    Ok(Arc::new(builder.finish()))
}

/// Like [`to_struct`] but accepts `(name, components)` per field so that
/// tuple sub-fields are named from their [`Param`] components rather than
/// falling back to `param0`, `param1`, …
pub(crate) fn to_struct_named(
    fields: &[DynSolType],
    named: &[(&str, &[Param])],
    sol_values: Vec<Option<DynSolValue>>,
    allow_decode_fail: bool,
    large_int_as_binary: bool,
) -> Result<Arc<dyn Array>> {
    if fields.is_empty() {
        return Ok(Arc::new(StructArray::new_empty_fields(
            sol_values.len(),
            None,
        )));
    }

    let mut per_field = vec![Vec::with_capacity(sol_values.len()); fields.len()];

    for val in sol_values {
        match val {
            Some(DynSolValue::Tuple(inner_vals)) => {
                if per_field.len() != inner_vals.len() {
                    let exp = per_field.len();
                    let got = inner_vals.len();
                    return Err(anyhow!(
                        "found unexpected length tuple value. Expected: {exp}, Found: {got}"
                    ));
                }
                for (col, v) in per_field.iter_mut().zip(inner_vals) {
                    col.push(Some(v));
                }
            }
            None => {
                for col in &mut per_field {
                    col.push(None);
                }
            }
            Some(other) => {
                return Err(anyhow!(
                    "found unexpected value. Expected: tuple, Found: {other:?}"
                ));
            }
        }
    }

    let mut arrays = Vec::with_capacity(fields.len());
    for (sol_type, (field_vals, &(_, comps))) in
        fields.iter().zip(per_field.into_iter().zip(named.iter()))
    {
        let arr = match (sol_type, comps) {
            (DynSolType::Tuple(sub_fields), comps) if !comps.is_empty() => {
                let sub_named: Vec<(&str, &[Param])> = comps
                    .iter()
                    .map(|c| (c.name.as_str(), c.components.as_slice()))
                    .collect();
                to_struct_named(
                    sub_fields,
                    &sub_named,
                    field_vals,
                    allow_decode_fail,
                    large_int_as_binary,
                )?
            }
            _ => to_arrow(sol_type, field_vals, allow_decode_fail, large_int_as_binary)?,
        };
        arrays.push(arr);
    }

    let schema_fields = named
        .iter()
        .enumerate()
        .zip(arrays.iter())
        .map(|((i, &(name, _)), arr)| {
            let n = if name.is_empty() {
                format!("param{i}")
            } else {
                name.to_string()
            };
            Field::new(n, arr.data_type().clone(), true)
        })
        .collect::<Vec<_>>();

    let batch = RecordBatch::try_new(Arc::new(Schema::new(schema_fields)), arrays)
        .context("construct record batch")?;
    Ok(Arc::new(StructArray::from(batch)))
}

/// Decode topic column values from binary to Arrow arrays.
pub(crate) fn decode_topic<I: OffsetSizeTrait>(
    sol_type: &DynSolType,
    col: &GenericBinaryArray<I>,
    allow_decode_fail: bool,
    large_int_as_binary: bool,
    arrays: &mut Vec<Arc<dyn Array>>,
) -> Result<()> {
    let mut decoded = Vec::<Option<DynSolValue>>::with_capacity(col.len());

    for blob in col {
        match blob {
            Some(blob) => match sol_type.abi_decode(blob) {
                Ok(data) => decoded.push(Some(data)),
                Err(e) if allow_decode_fail => {
                    log::debug!("failed to decode a topic: {e}");
                    decoded.push(None);
                }
                Err(e) => {
                    return Err(anyhow!("failed to decode a topic: {e}"));
                }
            },
            None => decoded.push(None),
        }
    }

    arrays.push(
        to_arrow(sol_type, decoded, allow_decode_fail, large_int_as_binary)
            .context("map topic to arrow")?,
    );

    Ok(())
}

/// Decode body to Arrow arrays. Use [`EventParam`] component names so inner tuple
/// fields are named (e.g. `sharesDelta`) instead of positional (`param0`).
pub(crate) fn decode_body_named<I: OffsetSizeTrait>(
    body_sol_type: &DynSolType,
    body_params: &[&EventParam],
    body_col: &GenericBinaryArray<I>,
    allow_decode_fail: bool,
    large_int_as_binary: bool,
    arrays: &mut Vec<Arc<dyn Array>>,
) -> Result<()> {
    let mut body_decoded = Vec::<Option<DynSolValue>>::with_capacity(body_col.len());

    for blob in body_col {
        match blob {
            Some(blob) => match body_sol_type.abi_decode_sequence(blob) {
                Ok(data) => body_decoded.push(Some(data)),
                Err(e) if allow_decode_fail => {
                    log::debug!("failed to decode body: {e}");
                    body_decoded.push(None);
                }
                Err(e) => return Err(anyhow!("failed to decode body: {e}")),
            },
            None => body_decoded.push(None),
        }
    }

    let named: Vec<(&str, &[Param])> = body_params
        .iter()
        .map(|p| (p.name.as_str(), p.components.as_slice()))
        .collect();

    let body_sol_types = match body_sol_type {
        DynSolType::Tuple(f) => f.as_slice(),
        _ => return Err(anyhow!("body_sol_type must be DynSolType::Tuple")),
    };

    let body_array = to_struct_named(
        body_sol_types,
        &named,
        body_decoded,
        allow_decode_fail,
        large_int_as_binary,
    )
    .context("build body struct")?;

    let arr = body_array
        .as_any()
        .downcast_ref::<StructArray>()
        .context("expected struct array from to_struct_named")?;

    for f in arr.columns() {
        arrays.push(f.clone());
    }
    Ok(())
}

/// Build a boolean mask comparing a topic0 column against a selector.
pub(crate) fn build_topic0_mask(
    col: &dyn arrow::array::Array,
    selector: &[u8],
) -> Result<BooleanArray> {
    use arrow::array::{BinaryArray, LargeBinaryArray};

    if col.data_type() == &DataType::Binary {
        let arr = col
            .as_any()
            .downcast_ref::<BinaryArray>()
            .context("downcast topic0 to BinaryArray")?;
        Ok(arr
            .iter()
            .map(|v| v.map(|b| b == selector))
            .collect::<BooleanArray>())
    } else if col.data_type() == &DataType::LargeBinary {
        let arr = col
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .context("downcast topic0 to LargeBinaryArray")?;
        Ok(arr
            .iter()
            .map(|v| v.map(|b| b == selector))
            .collect::<BooleanArray>())
    } else {
        Err(anyhow!(
            "unexpected data type for topic0 column: {:?}",
            col.data_type()
        ))
    }
}
