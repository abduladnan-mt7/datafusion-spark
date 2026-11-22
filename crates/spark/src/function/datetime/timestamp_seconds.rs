// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use arrow::array::{ArrayRef, AsArray, PrimitiveArray, TimestampMicrosecondArray};
use arrow::datatypes::{
    DataType, Decimal128Type, Field, FieldRef, Float64Type, Int64Type, TimeUnit,
    TimestampMicrosecondType,
};
use arrow::error::ArrowError;
use datafusion_common::config::ConfigOptions;
use datafusion_common::utils::take_function_args;
use datafusion_common::{Result, internal_err, plan_err};
use datafusion_expr::{
    ColumnarValue, Documentation, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF,
    ScalarUDFImpl, Signature, Volatility,
};
use datafusion_functions::utils::make_scalar_function;
use datafusion_macros::user_doc;

use crate::function::implicit_cast::{SparkInputType, SparkInputTypes};

const MICROS_PER_SECOND: i64 = 1_000_000;
const MICROS_SCALE: i8 = 6;

const LONG_OVERFLOW_ERROR: &str = "long overflow";
const DECIMAL_OVERFLOW_ERROR: &str = "Overflow";
const DECIMAL_ROUNDING_ERROR: &str = "Rounding necessary";

/// Spark-compatible `timestamp_seconds` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#timestamp_seconds>
#[user_doc(
    doc_section(label = "Time and Date Functions"),
    description = "Returns the timestamp that is `seconds` seconds after 1970-01-01 00:00:00 UTC. \
        Negative values precede the epoch. Fractional seconds are kept to microsecond precision. \
        Integer input that overflows the microsecond count is an error. Decimal input must convert \
        to a whole number of microseconds exactly: more than six fractional digits or an \
        out-of-range value is an error. FLOAT and DOUBLE input is truncated toward zero to whole \
        microseconds and saturates at the BIGINT range; NaN and infinities return NULL. The result \
        has the session time zone (`datafusion.execution.time_zone`, default UTC), which affects \
        display, not the stored microsecond count. Returns NULL for NULL input.",
    syntax_example = "timestamp_seconds(seconds)",
    sql_example = r#"```sql
> SELECT timestamp_seconds(1230219000);
2008-12-25T15:30:00Z
> SELECT timestamp_seconds(1230219000.123);
2008-12-25T15:30:00.123Z
> SELECT timestamp_seconds(NULL);
NULL
```"#,
    argument(
        name = "seconds",
        description = "Numeric count of seconds since the UTC epoch: TINYINT, SMALLINT, INT, BIGINT, DECIMAL, FLOAT or DOUBLE. Arrow UInt8, UInt16 and UInt32 inputs are also accepted. NULL returns NULL."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkTimestampSeconds {
    input_types: SparkInputTypes,
    timezone: Arc<str>,
}

impl Default for SparkTimestampSeconds {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkTimestampSeconds {
    pub fn new() -> Self {
        Self::new_with_config(&ConfigOptions::default())
    }

    pub fn new_with_config(config: &ConfigOptions) -> Self {
        Self {
            input_types: SparkInputTypes::check_only(
                vec![SparkInputType::Numeric],
                Volatility::Stable,
            ),
            timezone: config
                .execution
                .time_zone
                .as_ref()
                .map_or("UTC", |tz| tz.as_str())
                .into(),
        }
    }

    fn return_data_type(&self) -> DataType {
        DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::clone(&self.timezone)))
    }
}

impl ScalarUDFImpl for SparkTimestampSeconds {
    fn name(&self) -> &str {
        "timestamp_seconds"
    }

    fn signature(&self) -> &Signature {
        self.input_types.signature()
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        let types = self.input_types.coerce(self.name(), arg_types)?;
        let coerced = match &types[0] {
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32 => DataType::Int64,
            DataType::Float16 | DataType::Float32 | DataType::Float64 => {
                DataType::Float64
            }
            DataType::Decimal32(precision, scale)
            | DataType::Decimal64(precision, scale)
            | DataType::Decimal128(precision, scale) => {
                DataType::Decimal128(*precision, *scale)
            }
            other => {
                return plan_err!(
                    "[DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE] The first parameter of \
                     `timestamp_seconds` requires a numeric type that Spark supports, \
                     however it has the type \"{other}\". SQLSTATE: 42K09"
                );
            }
        };
        Ok(vec![coerced])
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let [input] = take_function_args(self.name(), args.arg_fields)?;
        let nullable = input.is_nullable() || input.data_type().is_floating();
        Ok(Arc::new(Field::new(
            self.name(),
            self.return_data_type(),
            nullable,
        )))
    }

    fn with_updated_config(&self, config: &ConfigOptions) -> Option<ScalarUDF> {
        Some(ScalarUDF::from(Self::new_with_config(config)))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let timezone = Arc::clone(&self.timezone);
        make_scalar_function(
            move |args: &[ArrayRef]| {
                let [seconds] = take_function_args("timestamp_seconds", args)?;
                Ok(Arc::new(
                    seconds_to_micros(seconds)?.with_timezone(Arc::clone(&timezone)),
                ) as ArrayRef)
            },
            vec![],
        )(&args.args)
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

fn seconds_to_micros(seconds: &ArrayRef) -> Result<TimestampMicrosecondArray> {
    let micros = match seconds.data_type() {
        DataType::Int64 => seconds
            .as_primitive::<Int64Type>()
            .try_unary::<_, Int64Type, _>(|s| {
                s.checked_mul(MICROS_PER_SECOND).ok_or_else(|| {
                    ArrowError::ComputeError(LONG_OVERFLOW_ERROR.to_string())
                })
            })?,
        DataType::Float64 => seconds
            .as_primitive::<Float64Type>()
            .iter()
            .map(|s| s.and_then(float_seconds_to_micros))
            .collect::<PrimitiveArray<Int64Type>>(),
        DataType::Decimal128(_, scale) => {
            let scale = *scale;
            seconds
                .as_primitive::<Decimal128Type>()
                .try_unary::<_, Int64Type, _>(|unscaled| {
                    decimal_seconds_to_micros(unscaled, scale)
                })?
        }
        other => {
            return internal_err!(
                "timestamp_seconds received an uncoerced argument of type {other}"
            );
        }
    };
    Ok(micros.reinterpret_cast::<TimestampMicrosecondType>())
}

/// Spark's `(seconds * MICROS_PER_SECOND).toLong`, which truncates toward zero and saturates.
fn float_seconds_to_micros(seconds: f64) -> Option<i64> {
    seconds
        .is_finite()
        .then_some((seconds * MICROS_PER_SECOND as f64) as i64)
}

/// Spark's `BigDecimal.multiply(MICROS_PER_SECOND).longValueExact()`, including the order in
/// which it reports `Rounding necessary` and `Overflow`.
fn decimal_seconds_to_micros(unscaled: i128, scale: i8) -> Result<i64, ArrowError> {
    let overflow = || ArrowError::ComputeError(DECIMAL_OVERFLOW_ERROR.to_string());
    let rounding = || ArrowError::ComputeError(DECIMAL_ROUNDING_ERROR.to_string());
    if unscaled == 0 {
        return Ok(0);
    }
    let micros_scale = i32::from(scale) - i32::from(MICROS_SCALE);
    let micros = if micros_scale <= 0 {
        10_i128
            .checked_pow(micros_scale.unsigned_abs())
            .and_then(|factor| unscaled.checked_mul(factor))
            .ok_or_else(overflow)?
    } else {
        let integer_digits = decimal_digits(unscaled) - micros_scale;
        if integer_digits <= 0 {
            return Err(rounding());
        }
        if integer_digits > 19 {
            return Err(overflow());
        }
        let divisor = 10_i128.pow(micros_scale.unsigned_abs());
        if unscaled % divisor != 0 {
            return Err(rounding());
        }
        unscaled / divisor
    };
    i64::try_from(micros).map_err(|_| overflow())
}

fn decimal_digits(value: i128) -> i32 {
    value
        .unsigned_abs()
        .checked_ilog10()
        .map_or(1, |digits| digits as i32 + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn return_field_tracks_session_timezone_and_nullability() -> Result<()> {
        for timezone in [None, Some("America/Los_Angeles")] {
            let mut config = ConfigOptions::default();
            config.execution.time_zone = timezone.map(str::parse).transpose()?;
            let function = SparkTimestampSeconds::new()
                .with_updated_config(&config)
                .unwrap();
            for (input_type, nullable, expected_nullable) in [
                (DataType::Int64, false, false),
                (DataType::Int64, true, true),
                (DataType::Decimal128(13, 3), false, false),
                (DataType::Float64, false, true),
            ] {
                let input = Arc::new(Field::new("seconds", input_type, nullable));
                let field = function.inner().return_field_from_args(ReturnFieldArgs {
                    arg_fields: &[input],
                    scalar_arguments: &[None],
                })?;
                assert_eq!(field.is_nullable(), expected_nullable);
                assert_eq!(
                    field.data_type(),
                    &DataType::Timestamp(
                        TimeUnit::Microsecond,
                        Some(timezone.unwrap_or("UTC").into()),
                    ),
                );
            }
        }
        Ok(())
    }

    #[test]
    fn float_seconds_truncate_toward_zero_and_saturate() {
        assert_eq!(float_seconds_to_micros(1.5), Some(1_500_000));
        assert_eq!(float_seconds_to_micros(-0.0000015), Some(-1));
        assert_eq!(float_seconds_to_micros(-0.0), Some(0));
        assert_eq!(float_seconds_to_micros(1e300), Some(i64::MAX));
        assert_eq!(float_seconds_to_micros(-1e300), Some(i64::MIN));
        assert_eq!(float_seconds_to_micros(f64::NAN), None);
        assert_eq!(float_seconds_to_micros(f64::INFINITY), None);
        assert_eq!(float_seconds_to_micros(f64::NEG_INFINITY), None);
    }

    #[test]
    fn decimal_seconds_convert_exactly() {
        let micros = |unscaled, scale| {
            decimal_seconds_to_micros(unscaled, scale).map_err(|e| e.to_string())
        };
        assert_eq!(micros(1230219000123, 3), Ok(1230219000123000));
        assert_eq!(micros(-15, 1), Ok(-1_500_000));
        assert_eq!(micros(1234560, 7), Ok(123456));
        assert_eq!(micros(0, 38), Ok(0));
        assert_eq!(micros(9223372036854, 0), Ok(9223372036854000000));
        assert!(
            micros(1234567, 7)
                .unwrap_err()
                .contains(DECIMAL_ROUNDING_ERROR)
        );
        assert!(micros(1, 38).unwrap_err().contains(DECIMAL_ROUNDING_ERROR));
        assert!(
            micros(9223372036855, 0)
                .unwrap_err()
                .contains(DECIMAL_OVERFLOW_ERROR)
        );
        assert!(
            micros(i128::MAX / 10, 0)
                .unwrap_err()
                .contains(DECIMAL_OVERFLOW_ERROR)
        );
    }
}
