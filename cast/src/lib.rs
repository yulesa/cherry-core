//! # tiders-cast
//!
//! Type casting and encoding/decoding utilities for Arrow columns and schemas.
//!
//! Provides batch-level and column-level operations for:
//! - **Type casting**: Convert columns between Arrow data types by name or by source type.
//! - **Hex encoding/decoding**: Binary columns to/from hex strings (with optional `0x` prefix).
//! - **Base58 encoding/decoding**: Binary columns to/from Base58 strings (Bitcoin alphabet).
//! - **U256 conversion**: Between `Decimal256(76,0)` and big-endian binary representations.

#![allow(clippy::manual_div_ceil)]

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use arrow::{
    array::{
        builder, make_array, Array, BinaryArray, Decimal256Array, FixedSizeListArray,
        GenericBinaryArray, GenericStringArray, Int32Array, LargeBinaryArray, OffsetSizeTrait,
        RecordBatch, StructArray,
    },
    buffer::NullBuffer,
    compute::CastOptions,
    datatypes::{DataType, Field, Schema},
};

/// Casts columns according to given (column name, target data type) pairs.
///
/// Returns error if casting a row fails and `allow_cast_fail` is set to `false`.
/// Writes `null` to output if casting a row fails and `allow_cast_fail` is set to `true`.
pub fn cast<S: AsRef<str>>(
    map: &[(S, DataType)],
    data: &RecordBatch,
    allow_cast_fail: bool,
) -> Result<RecordBatch> {
    let schema = cast_schema(map, data.schema_ref()).context("cast schema")?;

    let mut arrays = Vec::with_capacity(data.num_columns());

    let cast_opt = CastOptions {
        safe: allow_cast_fail,
        ..Default::default()
    };

    for (col, field) in data.columns().iter().zip(data.schema_ref().fields().iter()) {
        let cast_target = map.iter().find(|x| x.0.as_ref() == field.name());

        let col = match cast_target {
            Some(tgt) => {
                // allow precision loss for decimal types into floating point types
                if matches!(
                    col.data_type(),
                    DataType::Decimal256(..) | DataType::Decimal128(..)
                ) && tgt.1.is_floating()
                {
                    let string_col =
                        arrow::compute::cast_with_options(col, &DataType::Utf8, &cast_opt)
                            .with_context(|| {
                                format!(
                            "Failed when casting column '{}' to string as intermediate step",
                            field.name()
                        )
                            })?;
                    Arc::new(
                        arrow::compute::cast_with_options(&string_col, &tgt.1, &cast_opt)
                            .with_context(|| {
                                format!(
                                    "Failed when casting column '{}' to {:?}",
                                    field.name(),
                                    tgt.1
                                )
                            })?,
                    )
                } else {
                    Arc::new(
                        arrow::compute::cast_with_options(col, &tgt.1, &cast_opt).with_context(
                            || {
                                format!(
                                    "Failed when casting column '{}' from {:?} to {:?}",
                                    field.name(),
                                    col.data_type(),
                                    tgt.1
                                )
                            },
                        )?,
                    )
                }
            }
            None => col.clone(),
        };

        arrays.push(col);
    }

    let batch = RecordBatch::try_new(Arc::new(schema), arrays).context("construct record batch")?;

    Ok(batch)
}

/// Casts column types according to given (column name, target data type) pairs.
pub fn cast_schema<S: AsRef<str>>(map: &[(S, DataType)], schema: &Schema) -> Result<Schema> {
    let mut fields = schema.fields().to_vec();

    for f in &mut fields {
        let cast_target = map.iter().find(|x| x.0.as_ref() == f.name());

        if let Some(tgt) = cast_target {
            *f = Arc::new(Field::new(f.name(), tgt.1.clone(), f.is_nullable()));
        }
    }

    Ok(Schema::new(fields))
}

/// Casts all columns with from_type to to_type.
///
/// Returns error if casting a row fails and `allow_cast_fail` is set to `false`.
/// Writes `null` to output if casting a row fails and `allow_cast_fail` is set to `true`.
pub fn cast_by_type(
    data: &RecordBatch,
    from_type: &DataType,
    to_type: &DataType,
    allow_cast_fail: bool,
) -> Result<RecordBatch> {
    let schema =
        cast_schema_by_type(data.schema_ref(), from_type, to_type).context("cast schema")?;

    let mut arrays = Vec::with_capacity(data.num_columns());

    let cast_opt = CastOptions {
        safe: allow_cast_fail,
        ..Default::default()
    };

    for (col, field) in data.columns().iter().zip(data.schema_ref().fields().iter()) {
        let col = if col.data_type() == from_type {
            // allow precision loss for decimal types into floating point types
            if matches!(
                col.data_type(),
                DataType::Decimal256(..) | DataType::Decimal128(..)
            ) && to_type.is_floating()
            {
                let string_col = arrow::compute::cast_with_options(col, &DataType::Utf8, &cast_opt)
                    .with_context(|| {
                        format!(
                            "Failed when casting_by_type column '{}' to string as intermediate step",
                            field.name()
                        )
                    })?;
                Arc::new(
                    arrow::compute::cast_with_options(&string_col, to_type, &cast_opt)
                        .with_context(|| {
                            format!(
                                "Failed when casting_by_type column '{}' to {:?}",
                                field.name(),
                                to_type
                            )
                        })?,
                )
            } else {
                Arc::new(
                    arrow::compute::cast_with_options(col, to_type, &cast_opt).with_context(
                        || {
                            format!(
                                "Failed when casting_by_type column '{}' to {:?}",
                                field.name(),
                                to_type
                            )
                        },
                    )?,
                )
            }
        } else {
            col.clone()
        };

        arrays.push(col);
    }

    let batch = RecordBatch::try_new(Arc::new(schema), arrays).context("construct record batch")?;

    Ok(batch)
}

/// Casts columns with from_type to to_type
pub fn cast_schema_by_type(
    schema: &Schema,
    from_type: &DataType,
    to_type: &DataType,
) -> Result<Schema> {
    let mut fields = schema.fields().to_vec();

    for f in &mut fields {
        if f.data_type() == from_type {
            *f = Arc::new(Field::new(f.name(), to_type.clone(), f.is_nullable()));
        }
    }

    Ok(Schema::new(fields))
}

#[expect(
    clippy::unwrap_used,
    reason = "downcast is guaranteed by prior data type check"
)]
/// Encodes all Binary and LargeBinary columns in the batch to Base58 (Bitcoin alphabet) strings.
pub fn base58_encode(data: &RecordBatch) -> Result<RecordBatch> {
    let schema = schema_binary_to_string(data.schema_ref());
    let mut columns = Vec::<Arc<dyn Array>>::with_capacity(data.columns().len());

    for col in data.columns() {
        if col.data_type() == &DataType::Binary {
            columns.push(Arc::new(base58_encode_column(
                col.as_any().downcast_ref::<BinaryArray>().unwrap(),
            )));
        } else if col.data_type() == &DataType::LargeBinary {
            columns.push(Arc::new(base58_encode_column(
                col.as_any().downcast_ref::<LargeBinaryArray>().unwrap(),
            )));
        } else {
            columns.push(col.clone());
        }
    }

    RecordBatch::try_new(Arc::new(schema), columns).context("construct arrow batch")
}

/// Encodes a single binary column to Base58 strings.
pub fn base58_encode_column<I: OffsetSizeTrait>(
    col: &GenericBinaryArray<I>,
) -> GenericStringArray<I> {
    let mut arr = builder::GenericStringBuilder::<I>::with_capacity(
        col.len(),
        (col.value_data().len() + 2) * 2,
    );

    for v in col {
        match v {
            Some(v) => {
                let v = bs58::encode(v)
                    .with_alphabet(bs58::Alphabet::BITCOIN)
                    .into_string();
                arr.append_value(v);
            }
            None => arr.append_null(),
        }
    }

    arr.finish()
}

#[expect(
    clippy::unwrap_used,
    reason = "downcast is guaranteed by prior data type check"
)]
/// Encodes all Binary and LargeBinary columns in the batch to hex strings.
///
/// When `PREFIXED` is `true`, output strings include the `0x` prefix.
pub fn hex_encode<const PREFIXED: bool>(data: &RecordBatch) -> Result<RecordBatch> {
    let schema = schema_binary_to_string(data.schema_ref());
    let mut columns = Vec::<Arc<dyn Array>>::with_capacity(data.columns().len());

    for col in data.columns() {
        if col.data_type() == &DataType::Binary {
            columns.push(Arc::new(hex_encode_column::<PREFIXED, i32>(
                col.as_any().downcast_ref::<BinaryArray>().unwrap(),
            )));
        } else if col.data_type() == &DataType::LargeBinary {
            columns.push(Arc::new(hex_encode_column::<PREFIXED, i64>(
                col.as_any().downcast_ref::<LargeBinaryArray>().unwrap(),
            )));
        } else {
            columns.push(col.clone());
        }
    }

    RecordBatch::try_new(Arc::new(schema), columns).context("construct arrow batch")
}

/// Encodes a single binary column to hex strings.
///
/// When `PREFIXED` is `true`, output strings include the `0x` prefix.
pub fn hex_encode_column<const PREFIXED: bool, I: OffsetSizeTrait>(
    col: &GenericBinaryArray<I>,
) -> GenericStringArray<I> {
    let mut arr = builder::GenericStringBuilder::<I>::with_capacity(
        col.len(),
        (col.value_data().len() + 2) * 2,
    );

    for v in col {
        match v {
            Some(v) => {
                // TODO: avoid allocation here and use a scratch buffer to encode hex into or write to arrow buffer
                // directly somehow.
                let v = if PREFIXED {
                    format!("0x{}", faster_hex::hex_string(v))
                } else {
                    faster_hex::hex_string(v)
                };

                arr.append_value(v);
            }
            None => arr.append_null(),
        }
    }

    arr.finish()
}

/// Converts binary fields to string in the schema
///
/// Intended to be used with encode hex functions
pub fn schema_binary_to_string(schema: &Schema) -> Schema {
    let mut fields = Vec::<Arc<Field>>::with_capacity(schema.fields().len());

    for f in schema.fields() {
        if f.data_type() == &DataType::Binary {
            fields.push(Arc::new(Field::new(
                f.name().clone(),
                DataType::Utf8,
                f.is_nullable(),
            )));
        } else if f.data_type() == &DataType::LargeBinary {
            fields.push(Arc::new(Field::new(
                f.name().clone(),
                DataType::LargeUtf8,
                f.is_nullable(),
            )));
        } else {
            fields.push(f.clone());
        }
    }

    Schema::new(fields)
}

/// Converts decimal256 fields to binary in the schema
///
/// Intended to be used with u256_to_binary function
pub fn schema_decimal256_to_binary(schema: &Schema) -> Schema {
    let mut fields = Vec::<Arc<Field>>::with_capacity(schema.fields().len());

    for f in schema.fields() {
        if f.data_type() == &DataType::Decimal256(76, 0) {
            fields.push(Arc::new(Field::new(
                f.name().clone(),
                DataType::Binary,
                f.is_nullable(),
            )));
        } else {
            fields.push(f.clone());
        }
    }

    Schema::new(fields)
}

/// Decodes a Base58-encoded string column to binary.
pub fn base58_decode_column<I: OffsetSizeTrait>(
    col: &GenericStringArray<I>,
) -> Result<GenericBinaryArray<I>> {
    let mut arr =
        builder::GenericBinaryBuilder::<I>::with_capacity(col.len(), col.value_data().len() / 2);

    for v in col {
        match v {
            // TODO: this should be optimized by removing allocations if needed
            Some(v) => {
                let v = bs58::decode(v)
                    .with_alphabet(bs58::Alphabet::BITCOIN)
                    .into_vec()
                    .context("bs58 decode")?;
                arr.append_value(v);
            }
            None => arr.append_null(),
        }
    }

    Ok(arr.finish())
}

/// Decodes a hex-encoded string column to binary.
///
/// When `PREFIXED` is `true`, expects and strips the `0x` prefix from each value.
pub fn hex_decode_column<const PREFIXED: bool, I: OffsetSizeTrait>(
    col: &GenericStringArray<I>,
) -> Result<GenericBinaryArray<I>> {
    let mut arr =
        builder::GenericBinaryBuilder::<I>::with_capacity(col.len(), col.value_data().len() / 2);

    for v in col {
        match v {
            // TODO: this should be optimized by removing allocations if needed
            Some(v) => {
                let v = v.as_bytes();
                let v = if PREFIXED {
                    v.get(2..).context("index into prefix hex encoded value")?
                } else {
                    v
                };

                let len = v.len();
                let mut dst = vec![0; (len + 1) / 2];

                faster_hex::hex_decode(v, &mut dst).context("hex decode")?;

                arr.append_value(dst);
            }
            None => arr.append_null(),
        }
    }

    Ok(arr.finish())
}

/// Converts a big-endian binary column (up to 32 bytes) to Decimal256(76,0).
pub fn u256_column_from_binary<I: OffsetSizeTrait>(
    col: &GenericBinaryArray<I>,
) -> Result<Decimal256Array> {
    let mut arr = builder::Decimal256Builder::with_capacity(col.len());

    for v in col {
        match v {
            Some(v) => {
                let num = ruint::aliases::U256::try_from_be_slice(v).context("parse ruint u256")?;
                let num = alloy_primitives::I256::try_from(num)
                    .with_context(|| format!("u256 to i256. val was {num}"))?;

                let val = arrow::datatypes::i256::from_be_bytes(num.to_be_bytes::<32>());
                arr.append_value(val);
            }
            None => arr.append_null(),
        }
    }

    Ok(arr
        .with_precision_and_scale(76, 0)
        .context("set precision and scale for Decimal256")?
        .finish())
}

/// Converts a Decimal256(76,0) column to trimmed big-endian binary.
pub fn u256_column_to_binary(col: &Decimal256Array) -> Result<BinaryArray> {
    let mut arr = builder::BinaryBuilder::with_capacity(col.len(), col.len() * 32);

    for v in col {
        match v {
            Some(v) => {
                let num = alloy_primitives::I256::from_be_bytes::<32>(v.to_be_bytes());
                let num = ruint::aliases::U256::try_from(num).context("convert i256 to u256")?;
                arr.append_value(num.to_be_bytes_trimmed_vec());
            }
            None => {
                arr.append_null();
            }
        }
    }

    Ok(arr.finish())
}

/// Converts all Decimal256 (U256) columns in the batch to big endian binary values
#[expect(
    clippy::unwrap_used,
    reason = "downcast is guaranteed by prior data type check"
)]
pub fn u256_to_binary(data: &RecordBatch) -> Result<RecordBatch> {
    let schema = schema_decimal256_to_binary(data.schema_ref());
    let mut columns = Vec::<Arc<dyn Array>>::with_capacity(data.columns().len());

    for (i, col) in data.columns().iter().enumerate() {
        if col.data_type() == &DataType::Decimal256(76, 0) {
            let col = col.as_any().downcast_ref::<Decimal256Array>().unwrap();
            let x = u256_column_to_binary(col)
                .with_context(|| format!("col {} to binary", data.schema().fields()[i].name()))?;
            columns.push(Arc::new(x));
        } else {
            columns.push(col.clone());
        }
    }

    RecordBatch::try_new(Arc::new(schema), columns).context("construct arrow batch")
}

/// Flattens all `Struct` columns in a RecordBatch into top-level columns,
/// dot-joining names as it recurses.
///
/// - `Struct(a, b)` with name `foo` → columns `foo.a`, `foo.b`
/// - `FixedSizeList(n, Struct(...))` with name `foo` → columns `foo.0`, `foo.1`, … `foo.n-1`,
///   each further expanded if the struct has fields
/// - `List(Struct(...))` with name `foo` → single `foo` column serialised to UTF-8 string
/// - Any other type is emitted unchanged
///
/// When two expanded names collide the second occurrence is renamed `name_1`, the third
/// `name_2`, and so on (the first occurrence keeps its original name).
pub fn flatten_record_batch(batch: &RecordBatch) -> Result<RecordBatch> {
    let mut out_fields: Vec<Arc<Field>> = Vec::new();
    let mut out_arrays: Vec<Arc<dyn Array>> = Vec::new();

    for (field, col) in batch.schema().fields().iter().zip(batch.columns()) {
        expand_column(field.name(), col, &mut out_fields, &mut out_arrays)?;
    }

    resolve_name_collisions(&mut out_fields);

    RecordBatch::try_new(Arc::new(Schema::new(out_fields)), out_arrays)
        .context("construct flattened batch")
}

/// Schema-only mirror of [`flatten_record_batch`].
pub fn flatten_schema(schema: &Schema) -> Schema {
    let mut out_fields: Vec<Arc<Field>> = Vec::new();

    for field in schema.fields() {
        expand_field(field.name(), field.data_type(), &mut out_fields);
    }

    resolve_name_collisions(&mut out_fields);

    Schema::new(out_fields)
}

fn expand_column(
    name: &str,
    col: &Arc<dyn Array>,
    out_fields: &mut Vec<Arc<Field>>,
    out_arrays: &mut Vec<Arc<dyn Array>>,
) -> Result<()> {
    match col.data_type() {
        DataType::Struct(inner_fields) => {
            let struct_arr = col
                .as_any()
                .downcast_ref::<StructArray>()
                .context("downcast to StructArray")?;
            for (i, inner_field) in inner_fields.iter().enumerate() {
                let child = struct_arr.column(i).clone();
                let child = propagate_nulls(col.nulls(), child);
                expand_column(
                    &format!("{}.{}", name, inner_field.name()),
                    &child,
                    out_fields,
                    out_arrays,
                )?;
            }
        }
        DataType::FixedSizeList(inner_field, n)
            if matches!(inner_field.data_type(), DataType::Struct(_)) =>
        {
            let n = usize::try_from(*n).context("FixedSizeList size must be non-negative")?;
            let list_arr = col
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .context("downcast to FixedSizeListArray")?;
            let values = list_arr.values();
            let num_rows = list_arr.len();

            for i in 0..n {
                let indices: Int32Array = (0..num_rows)
                    .map(|r| i32::try_from(r * n + i).context("index overflows i32"))
                    .collect::<Result<Vec<_>>>()?
                    .into();
                let element = arrow::compute::take(values.as_ref(), &indices, None)
                    .context("take element from FixedSizeList")?;
                let element = propagate_nulls(list_arr.nulls(), element);
                expand_column(&format!("{name}.{i}"), &element, out_fields, out_arrays)?;
            }
        }
        DataType::List(inner_field) if matches!(inner_field.data_type(), DataType::Struct(_)) => {
            let str_col = arrow::compute::cast_with_options(
                col.as_ref(),
                &DataType::Utf8,
                &CastOptions {
                    safe: true,
                    ..Default::default()
                },
            )
            .context("cast List<Struct> to Utf8")?;
            out_fields.push(Arc::new(Field::new(name, DataType::Utf8, true)));
            out_arrays.push(str_col);
        }
        _ => {
            out_fields.push(Arc::new(Field::new(name, col.data_type().clone(), true)));
            out_arrays.push(col.clone());
        }
    }
    Ok(())
}

fn expand_field(name: &str, dtype: &DataType, out: &mut Vec<Arc<Field>>) {
    match dtype {
        DataType::Struct(inner_fields) => {
            for f in inner_fields {
                expand_field(&format!("{}.{}", name, f.name()), f.data_type(), out);
            }
        }
        DataType::FixedSizeList(inner_field, n)
            if matches!(inner_field.data_type(), DataType::Struct(_)) =>
        {
            for i in 0..usize::try_from(*n).unwrap_or(0) {
                expand_field(&format!("{name}.{i}"), inner_field.data_type(), out);
            }
        }
        DataType::List(inner_field) if matches!(inner_field.data_type(), DataType::Struct(_)) => {
            out.push(Arc::new(Field::new(name, DataType::Utf8, true)));
        }
        _ => {
            out.push(Arc::new(Field::new(name, dtype.clone(), true)));
        }
    }
}

/// Merges the parent's null buffer into a child array so that rows null in the
/// parent are also null in the child.
fn propagate_nulls(parent_nulls: Option<&NullBuffer>, col: Arc<dyn Array>) -> Arc<dyn Array> {
    let Some(parent_nulls) = parent_nulls else {
        return col;
    };
    let merged = NullBuffer::union(Some(parent_nulls), col.nulls());
    let null_count = merged.as_ref().map_or(0, NullBuffer::null_count);
    let data = col.into_data();
    // SAFETY: only the null buffer changes; values and offsets are untouched.
    let new_data = unsafe {
        data.into_builder()
            .null_bit_buffer(merged.map(|nb| nb.into_inner().into_inner()))
            .null_count(null_count)
            .build_unchecked()
    };
    make_array(new_data)
}

/// Renames duplicate field names: the first occurrence keeps its name, subsequent
/// ones become `name_1`, `name_2`, etc.
fn resolve_name_collisions(fields: &mut [Arc<Field>]) {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for f in fields.iter() {
        *counts.entry(f.name().clone()).or_insert(0) += 1;
    }
    let mut seen: HashMap<String, usize> = HashMap::new();
    for field in fields.iter_mut() {
        let name = field.name().clone();
        if *counts.get(&name).unwrap_or(&0) > 1 {
            let idx = seen.entry(name.clone()).or_insert(0);
            if *idx > 0 {
                let new_name = format!("{name}_{idx}");
                *field = Arc::new(Field::new(
                    new_name,
                    field.data_type().clone(),
                    field.is_nullable(),
                ));
            }
            *idx += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::DataType;
    use std::fs::File;

    #[test]
    #[ignore]
    fn test_cast() {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let builder =
            ParquetRecordBatchReaderBuilder::try_new(File::open("data.parquet").unwrap()).unwrap();
        let mut reader = builder.build().unwrap();
        let table = reader.next().unwrap().unwrap();

        let type_mappings = vec![
            ("amount0In", DataType::Decimal128(15, 0)),
            ("amount1In", DataType::Float32),
            ("amount0Out", DataType::Float64),
            ("amount1Out", DataType::Decimal128(38, 0)),
            ("timestamp", DataType::Int64),
        ];

        let result = cast(&type_mappings, &table, true).unwrap();

        // Save the filtered instructions to a new parquet file
        let mut file = File::create("result.parquet").unwrap();
        let mut writer =
            parquet::arrow::ArrowWriter::try_new(&mut file, result.schema(), None).unwrap();
        writer.write(&result).unwrap();
        writer.close().unwrap();
    }
}
