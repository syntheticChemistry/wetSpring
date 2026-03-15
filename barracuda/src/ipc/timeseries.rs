// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-spring time series exchange format.
//!
//! Implements the `ecoPrimals/time-series/v1` schema from wateringHole
//! `CROSS_SPRING_DATA_FLOW_STANDARD.md` for structured data exchange
//! between springs via `capability.call`.

use serde_json::{Value, json};

use crate::ipc::protocol::RpcError;

/// Schema identifier for the cross-spring time series format.
pub const SCHEMA: &str = "ecoPrimals/time-series/v1";

/// Build a time series payload conforming to the cross-spring standard.
///
/// Returns a JSON object with `schema`, `variable`, `unit`, `source`,
/// `timestamps`, and `values` fields.
#[must_use]
pub fn build_time_series(
    variable: &str,
    unit: &str,
    timestamps: &[String],
    values: &[f64],
    experiment: Option<&str>,
) -> Value {
    json!({
        "schema": SCHEMA,
        "variable": variable,
        "unit": unit,
        "source": {
            "spring": crate::PRIMAL_NAME,
            "experiment": experiment.unwrap_or(""),
            "capability": format!("science.{variable}"),
        },
        "timestamps": timestamps,
        "values": values,
    })
}

/// Parse an incoming time series payload, validating the schema version.
///
/// # Errors
///
/// Returns `RpcError::invalid_params` if schema is missing or wrong version.
pub fn parse_time_series(params: &Value) -> Result<TimeSeriesData, RpcError> {
    let schema = params
        .get("schema")
        .and_then(Value::as_str)
        .unwrap_or("");

    if schema != SCHEMA {
        return Err(RpcError::invalid_params(format!(
            "expected schema '{SCHEMA}', got '{schema}'"
        )));
    }

    let variable = params
        .get("variable")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();

    let unit = params
        .get("unit")
        .and_then(Value::as_str)
        .unwrap_or("dimensionless")
        .to_string();

    let timestamps: Vec<String> = params
        .get("timestamps")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(Value::as_str)
                .map(String::from)
                .collect()
        })
        .unwrap_or_default();

    let values: Vec<f64> = params
        .get("values")
        .and_then(Value::as_array)
        .map(|arr| arr.iter().filter_map(Value::as_f64).collect())
        .unwrap_or_default();

    let source_spring = params
        .get("source")
        .and_then(|s| s.get("spring"))
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();

    Ok(TimeSeriesData {
        variable,
        unit,
        timestamps,
        values,
        source_spring,
    })
}

/// Parsed cross-spring time series data.
#[derive(Debug)]
pub struct TimeSeriesData {
    /// Variable name (`snake_case`).
    pub variable: String,
    /// SI or documented unit.
    pub unit: String,
    /// ISO 8601 UTC timestamps.
    pub timestamps: Vec<String>,
    /// f64 values (same length as timestamps).
    pub values: Vec<f64>,
    /// Source spring identifier.
    pub source_spring: String,
}

/// Handle `science.timeseries` — analyze incoming cross-spring time series.
///
/// Computes basic statistics (mean, variance, trend) on the values.
///
/// # Errors
///
/// Returns `RpcError::invalid_params` if `time_series` is missing or empty.
#[expect(clippy::cast_precision_loss)]
pub fn handle_timeseries(params: &Value) -> Result<Value, RpcError> {
    let data = extract_ts_data(params)?;
    let values = &data.values;

    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;

    let trend = if values.len() >= 2 {
        let last = values[values.len() - 1];
        let first = values[0];
        if first.abs() > f64::EPSILON {
            (last - first) / first
        } else {
            0.0
        }
    } else {
        0.0
    };

    Ok(json!({
        "variable": data.variable,
        "unit": data.unit,
        "source_spring": data.source_spring,
        "n_points": values.len(),
        "mean": mean,
        "variance": variance,
        "trend_pct": trend * 100.0,
        "min": values.iter().copied().fold(f64::INFINITY, f64::min),
        "max": values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    }))
}

/// Handle `science.timeseries_diversity` — compute diversity on time series
/// values interpreted as community abundances.
///
/// # Errors
///
/// Returns `RpcError::invalid_params` if `time_series` is missing or empty.
pub fn handle_timeseries_diversity(params: &Value) -> Result<Value, RpcError> {
    use crate::bio::diversity;

    let data = extract_ts_data(params)?;
    let values = &data.values;

    Ok(json!({
        "variable": data.variable,
        "source_spring": data.source_spring,
        "shannon": diversity::shannon(values),
        "simpson": diversity::simpson(values),
        "pielou": diversity::pielou_evenness(values),
        "observed": diversity::observed_features(values),
        "chao1": diversity::chao1(values),
    }))
}

/// Extract and validate a `TimeSeriesData` from incoming params.
fn extract_ts_data(params: &Value) -> Result<TimeSeriesData, RpcError> {
    let data = params
        .get("time_series")
        .ok_or_else(|| RpcError::invalid_params("missing time_series object"))
        .and_then(parse_time_series)?;

    if data.values.is_empty() {
        return Err(RpcError::invalid_params("time_series values array is empty"));
    }

    Ok(data)
}

/// Build a diversity time series for outbound exchange (wetSpring → other springs).
#[must_use]
pub fn build_diversity_series(
    timestamps: &[String],
    shannon_values: &[f64],
    experiment: Option<&str>,
) -> Value {
    build_time_series(
        "microbial_diversity_shannon",
        "nats",
        timestamps,
        shannon_values,
        experiment,
    )
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn build_and_parse_roundtrip() {
        let ts = ["2026-01-01T00:00:00Z".to_string()];
        let vals = [3.14];
        let payload = build_time_series("shannon", "nats", &ts, &vals, Some("exp001"));
        let parsed = parse_time_series(&payload).unwrap();
        assert_eq!(parsed.variable, "shannon");
        assert_eq!(parsed.unit, "nats");
        assert_eq!(parsed.values, vec![3.14]);
        assert_eq!(parsed.source_spring, crate::PRIMAL_NAME);
    }

    #[test]
    fn parse_wrong_schema_errors() {
        let bad = json!({"schema": "wrong/v1", "values": [1.0]});
        let err = parse_time_series(&bad).unwrap_err();
        assert_eq!(err.code, -32602);
    }

    #[test]
    fn handle_timeseries_basic() {
        let ts = build_time_series(
            "soil_moisture_vol",
            "m3/m3",
            &["t1".into(), "t2".into()],
            &[0.32, 0.29],
            None,
        );
        let params = json!({"time_series": ts});
        let result = handle_timeseries(&params).unwrap();
        assert_eq!(result["n_points"], 2);
        assert!(result["mean"].as_f64().unwrap() > 0.0);
    }

    #[test]
    fn handle_timeseries_diversity_basic() {
        let ts = build_time_series(
            "abundance",
            "counts",
            &["t1".into()],
            &[10.0, 20.0, 30.0],
            None,
        );
        let params = json!({"time_series": ts});
        let result = handle_timeseries_diversity(&params).unwrap();
        assert!(result.get("shannon").is_some());
    }

    #[test]
    fn build_diversity_series_schema() {
        let ts = build_diversity_series(&["t1".into()], &[2.5], Some("exp372"));
        assert_eq!(ts["schema"], SCHEMA);
        assert_eq!(ts["variable"], "microbial_diversity_shannon");
    }

    #[test]
    fn handle_timeseries_missing_data() {
        let err = handle_timeseries(&json!({})).unwrap_err();
        assert_eq!(err.code, -32602);
    }
}
