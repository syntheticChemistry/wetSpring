// SPDX-License-Identifier: AGPL-3.0-or-later
//! Method dispatch: routes JSON-RPC method calls to barracuda library functions.
//!
//! Each capability maps directly to existing, validated barracuda functions.
//! No math is duplicated — the dispatch layer is purely structural.
//!
//! When `--features gpu` is active, diversity and Anderson methods route
//! through barraCuda's GPU primitives (`FusedMapReduceF64`, Lanczos, etc.)
//! with automatic CPU fallback below the dispatch threshold or on GPU error.

use serde_json::Value;

use crate::ipc::handlers;
use crate::ipc::protocol::RpcError;
use crate::ipc::{provenance, timeseries};

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
        "health.liveness" => handlers::handle_health_liveness(),
        "health.readiness" => handlers::handle_health_readiness(),
        "capability.list" => handlers::handle_capability_list(),

        // Core science capabilities
        "science.diversity" => handlers::handle_diversity(params),
        "science.qs_model" => handlers::handle_qs_model(params),
        "science.ncbi_fetch" => handlers::handle_ncbi_fetch(params),
        "science.anderson" => handlers::handle_anderson(params),
        "science.full_pipeline" => handlers::handle_full_pipeline(params),

        // Expanded science capabilities
        "science.kinetics" => handlers::handle_kinetics(params),
        "science.alignment" => handlers::handle_alignment(params),
        "science.taxonomy" => handlers::handle_taxonomy(params),
        "science.phylogenetics" => handlers::handle_phylogenetics(params),
        "science.nmf" => handlers::handle_nmf(params),

        // Cross-spring time series
        "science.timeseries" => timeseries::handle_timeseries(params),
        "science.timeseries_diversity" => timeseries::handle_timeseries_diversity(params),

        // Provenance trio integration
        "provenance.begin" => provenance::handle_provenance_begin(params),
        "provenance.record" => provenance::handle_provenance_record(params),
        "provenance.complete" => provenance::handle_provenance_complete(params),

        // Brain (attention, urgency)
        "brain.observe" => handlers::handle_brain_observe(params),
        "brain.attention" => handlers::handle_brain_attention(params),
        "brain.urgency" => handlers::handle_brain_urgency(params),

        // AI assist (Squirrel)
        "ai.ecology_interpret" => handlers::handle_ai_ecology_interpret(params),

        _ => Err(RpcError::method_not_found(method)),
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "test module: assertions use unwrap for clarity"
)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn dispatch_health_check() {
        let result = dispatch("health.check", &json!({})).unwrap();
        assert_eq!(result["status"], "healthy");
        assert_eq!(result["primal"], crate::ipc::primal_names::SELF);
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

    #[test]
    fn brain_observe_valid() {
        let outputs: Vec<f64> = (0..36).map(|i| f64::from(i) * 0.01).collect();
        let params = json!({
            "sample_id": "test-s1",
            "shannon": 3.2,
            "simpson": 0.85,
            "evenness": 0.7,
            "chao1": 150.0,
            "bray_curtis_mean": 0.4,
            "anderson_w": 5.0,
            "anderson_phase": 0.3,
            "amr_load": 0.1,
            "head_outputs": outputs,
        });
        let result = dispatch("brain.observe", &params).unwrap();
        assert_eq!(result["status"], "observed");
        assert_eq!(result["sample_id"], "test-s1");
        assert!(result.get("attention").is_some());
        assert!(result.get("urgency").is_some());
        assert!(result["observation_count"].as_u64().unwrap() >= 1);
    }

    #[test]
    fn brain_observe_insufficient_heads() {
        let params = json!({
            "shannon": 2.0,
            "head_outputs": [0.1, 0.2],
        });
        let err = dispatch("brain.observe", &params).unwrap_err();
        assert_eq!(err.code, -32602);
    }

    #[test]
    fn brain_attention_no_prior() {
        let result = dispatch("brain.attention", &json!({})).unwrap();
        assert!(result.get("attention").is_some());
        assert!(result.get("urgency").is_some());
    }

    #[test]
    fn brain_urgency_no_prior() {
        let result = dispatch("brain.urgency", &json!({})).unwrap();
        let urgency = result["urgency"].as_f64().unwrap();
        assert!(urgency >= 0.0);
        assert!(result.get("attention").is_some());
    }

    #[test]
    fn brain_observe_then_attention() {
        let outputs: Vec<f64> = vec![0.5; 36];
        let params = json!({
            "sample_id": "seq-check",
            "shannon": 4.0,
            "head_outputs": outputs,
        });
        dispatch("brain.observe", &params).unwrap();

        let att = dispatch("brain.attention", &json!({})).unwrap();
        assert!(att["observation_count"].as_u64().unwrap() >= 1);
    }

    #[test]
    fn capability_list_returns_all_domains() {
        let result = dispatch("capability.list", &json!({})).unwrap();
        assert_eq!(result["primal"], crate::ipc::primal_names::SELF);
        assert_eq!(result["domain"], "ecology");

        let domains = result["domains"].as_array().unwrap();
        assert_eq!(domains.len(), 16);

        let domain_names: Vec<&str> = domains.iter().filter_map(|d| d["name"].as_str()).collect();
        assert!(domain_names.contains(&"ecology.diversity"));
        assert!(domain_names.contains(&"health"));
        assert!(domain_names.contains(&"provenance"));
        assert!(domain_names.contains(&"brain"));
        assert!(domain_names.contains(&"metrics"));
        assert!(domain_names.contains(&"ecology.ai_assist"));
    }

    #[test]
    fn capability_list_methods_total() {
        let result = dispatch("capability.list", &json!({})).unwrap();
        let total_methods: usize = result["domains"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|d| d["methods"].as_array())
            .map(Vec::len)
            .sum();
        assert_eq!(total_methods, 23);
    }

    #[test]
    fn dispatch_health_liveness() {
        let result = dispatch("health.liveness", &json!({})).unwrap();
        assert_eq!(result["alive"], true);
        assert_eq!(result["primal"], crate::ipc::primal_names::SELF);
    }

    #[test]
    fn dispatch_health_readiness() {
        let result = dispatch("health.readiness", &json!({})).unwrap();
        assert_eq!(result["ready"], true);
        assert_eq!(result["status"], "healthy");
        assert_eq!(result["primal"], crate::ipc::primal_names::SELF);
        let subs = &result["subsystems"];
        assert_eq!(subs["math"], true);
        assert_eq!(subs["ipc"], true);
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn unknown_method_returns_not_found(
                method in "[a-z]{1,8}\\.[a-z]{1,8}_[a-z]{1,8}",
            ) {
                let known = [
                    "health.check", "health.liveness", "health.readiness",
                    "capability.list", "science.diversity", "science.qs_model",
                    "science.anderson", "science.kinetics", "science.alignment",
                    "science.taxonomy", "science.phylogenetics", "science.nmf",
                    "science.ncbi_fetch", "science.full_pipeline",
                    "science.timeseries", "science.timeseries_diversity",
                    "provenance.begin", "provenance.record", "provenance.complete",
                    "brain.observe", "brain.attention", "brain.urgency",
                    "ai.ecology_interpret",
                ];
                if !known.contains(&method.as_str()) {
                    let err = dispatch(&method, &json!({})).unwrap_err();
                    prop_assert_eq!(err.code, -32601);
                }
            }

            #[test]
            fn dispatch_never_panics(
                method in "\\PC{0,64}",
            ) {
                let _ = dispatch(&method, &json!({}));
            }
        }
    }
}
