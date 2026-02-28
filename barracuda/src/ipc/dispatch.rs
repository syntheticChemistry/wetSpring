// SPDX-License-Identifier: AGPL-3.0-or-later
//! Method dispatch: routes JSON-RPC method calls to barracuda library functions.
//!
//! Each capability maps directly to existing, validated barracuda functions.
//! No math is duplicated — the dispatch layer is purely structural.
//!
//! When `--features gpu` is active, diversity and Anderson methods route
//! through `ToadStool`'s GPU primitives (`FusedMapReduceF64`, Lanczos, etc.)
//! with automatic CPU fallback below the dispatch threshold or on GPU error.

use serde_json::Value;

use crate::ipc::handlers;
use crate::ipc::protocol::RpcError;

// Re-export for tests
pub use handlers::{extract_f64_array, extract_string_array};

/// Dispatch a JSON-RPC method call to the appropriate barracuda handler.
///
/// Returns `Ok(result_value)` on success, `Err((code, message))` on failure.
///
/// # Errors
///
/// Returns JSON-RPC error codes:
/// - `-32601`: Method not found
/// - `-32602`: Invalid params
/// - `-32000` to `-32099`: Server errors (NCBI, GPU, etc.)
pub fn dispatch(method: &str, params: &Value) -> Result<Value, RpcError> {
    match method {
        "health.check" => handlers::handle_health(),
        "science.diversity" => handlers::handle_diversity(params),
        "science.qs_model" => handlers::handle_qs_model(params),
        "science.ncbi_fetch" => handlers::handle_ncbi_fetch(params),
        "science.anderson" => handlers::handle_anderson(params),
        "science.full_pipeline" => handlers::handle_full_pipeline(params),
        _ => Err(RpcError::method_not_found(method)),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn dispatch_health_check() {
        let result = dispatch("health.check", &json!({})).unwrap();
        assert_eq!(result["status"], "healthy");
        assert_eq!(result["primal"], "wetspring");
    }

    #[test]
    fn dispatch_unknown_method() {
        let err = dispatch("nonexistent.method", &json!({})).unwrap_err();
        assert_eq!(err.code, -32601);
    }

    #[test]
    fn diversity_all_metrics() {
        let params = json!({"counts": [10.0, 20.0, 30.0, 40.0]});
        let result = dispatch("science.diversity", &params).unwrap();
        assert!(result.get("shannon").is_some());
        assert!(result.get("simpson").is_some());
        assert!(result.get("chao1").is_some());
        assert!(result.get("observed").is_some());
        assert!(result.get("pielou").is_some());
    }

    #[test]
    fn diversity_specific_metric() {
        let params = json!({"counts": [10.0, 20.0], "metrics": ["shannon"]});
        let result = dispatch("science.diversity", &params).unwrap();
        assert!(result.get("shannon").is_some());
        assert!(result.get("simpson").is_none());
    }

    #[test]
    fn diversity_empty_counts() {
        let params = json!({"counts": []});
        let err = dispatch("science.diversity", &params).unwrap_err();
        assert_eq!(err.code, -32602);
    }

    #[test]
    fn diversity_missing_counts() {
        let err = dispatch("science.diversity", &json!({})).unwrap_err();
        assert_eq!(err.code, -32602);
    }

    #[test]
    fn diversity_with_bray_curtis() {
        let params = json!({
            "counts": [10.0, 20.0, 30.0],
            "counts_b": [15.0, 25.0, 35.0],
        });
        let result = dispatch("science.diversity", &params).unwrap();
        assert!(result.get("bray_curtis").is_some());
    }

    #[test]
    fn qs_model_default_scenario() {
        let result = dispatch("science.qs_model", &json!({})).unwrap();
        assert!(result.get("t_end").is_some());
        assert!(result.get("steps").is_some());
        assert!(result.get("final_state").is_some());
        assert!(result.get("peak_biofilm").is_some());
    }

    #[test]
    fn qs_model_high_density() {
        let params = json!({"scenario": "high_density", "dt": 0.05});
        let result = dispatch("science.qs_model", &params).unwrap();
        let t_end = result["t_end"].as_f64().unwrap();
        assert!(t_end > 0.0);
    }

    #[test]
    fn qs_model_unknown_scenario() {
        let params = json!({"scenario": "imaginary"});
        let err = dispatch("science.qs_model", &params).unwrap_err();
        assert_eq!(err.code, -32602);
    }

    #[test]
    fn full_pipeline_with_counts() {
        let params = json!({"counts": [5.0, 10.0, 15.0]});
        let result = dispatch("science.full_pipeline", &params).unwrap();
        assert!(result.get("diversity").is_some());
        assert!(result.get("qs_model").is_some());
        assert_eq!(result["pipeline"], "complete");
    }

    #[test]
    fn full_pipeline_no_counts() {
        let result = dispatch("science.full_pipeline", &json!({})).unwrap();
        assert!(result.get("diversity").is_none());
        assert!(result.get("qs_model").is_some());
    }

    #[test]
    fn health_lists_capabilities() {
        let result = dispatch("health.check", &json!({})).unwrap();
        let caps = result["capabilities"].as_array().unwrap();
        assert!(caps.len() >= 5);
    }

    #[test]
    fn extract_f64_array_valid() {
        let params = json!({"data": [1.0, 2.0, 3.0]});
        let arr = extract_f64_array(&params, "data").unwrap();
        assert_eq!(arr, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn extract_f64_array_missing() {
        let err = extract_f64_array(&json!({}), "data").unwrap_err();
        assert_eq!(err.code, -32602);
    }

    #[test]
    fn extract_string_array_present() {
        let params = json!({"m": ["a", "b"]});
        let arr = extract_string_array(&params, "m");
        assert_eq!(arr, vec!["a", "b"]);
    }

    #[test]
    fn extract_string_array_missing() {
        let arr = extract_string_array(&json!({}), "m");
        assert!(arr.is_empty());
    }
}
