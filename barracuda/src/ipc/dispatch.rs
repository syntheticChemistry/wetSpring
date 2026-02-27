// SPDX-License-Identifier: AGPL-3.0-or-later
//! Method dispatch: routes JSON-RPC method calls to barracuda library functions.
//!
//! Each capability maps directly to existing, validated barracuda functions.
//! No math is duplicated — the dispatch layer is purely structural.
//!
//! When `--features gpu` is active, diversity and Anderson methods route
//! through `ToadStool`'s GPU primitives (`FusedMapReduceF64`, Lanczos, etc.)
//! with automatic CPU fallback below the dispatch threshold or on GPU error.

use serde_json::{json, Value};

use crate::bio::diversity;
use crate::bio::qs_biofilm::{self, N_PARAMS, QsBiofilmParams};

#[cfg(feature = "gpu")]
use std::sync::OnceLock;

#[cfg(feature = "gpu")]
use crate::bio::diversity_gpu;
#[cfg(feature = "gpu")]
use crate::gpu::GpuF64;

#[cfg(feature = "gpu")]
static GPU_CTX: OnceLock<Option<GpuF64>> = OnceLock::new();

#[cfg(feature = "gpu")]
fn try_gpu() -> Option<&'static GpuF64> {
    GPU_CTX
        .get_or_init(|| {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .ok()?;
            rt.block_on(GpuF64::new()).ok()
        })
        .as_ref()
        .filter(|g| !g.is_lost())
}

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
pub fn dispatch(method: &str, params: &Value) -> Result<Value, (i64, String)> {
    match method {
        "health.check" => handle_health(),
        "science.diversity" => handle_diversity(params),
        "science.qs_model" => handle_qs_model(params),
        "science.ncbi_fetch" => handle_ncbi_fetch(params),
        "science.anderson" => handle_anderson(params),
        "science.full_pipeline" => handle_full_pipeline(params),
        _ => Err((-32601, format!("method not found: {method}"))),
    }
}

fn handle_health() -> Result<Value, (i64, String)> {
    #![allow(clippy::unnecessary_wraps)]

    #[cfg(feature = "gpu")]
    let substrate = if try_gpu().is_some() {
        "gpu"
    } else if GPU_CTX.get().is_some_and(|g| g.as_ref().is_some_and(GpuF64::is_lost)) {
        "gpu_lost"
    } else {
        "cpu"
    };
    #[cfg(not(feature = "gpu"))]
    let substrate = "cpu";

    Ok(json!({
        "status": "healthy",
        "primal": "wetspring",
        "version": env!("CARGO_PKG_VERSION"),
        "substrate": substrate,
        "capabilities": [
            "science.diversity",
            "science.anderson",
            "science.qs_model",
            "science.ncbi_fetch",
            "science.full_pipeline",
            "metrics.snapshot",
        ],
    }))
}

fn handle_diversity(params: &Value) -> Result<Value, (i64, String)> {
    let counts = extract_f64_array(params, "counts")?;
    if counts.is_empty() {
        return Err((-32602, "counts array is empty".into()));
    }

    let metrics = extract_string_array(params, "metrics");
    let compute_all = metrics.is_empty() || metrics.iter().any(|m| m == "all");

    let mut result = serde_json::Map::new();

    #[cfg(feature = "gpu")]
    let use_gpu = try_gpu().filter(|g| counts.len() >= g.dispatch_threshold());

    if compute_all || metrics.iter().any(|m| m == "shannon") {
        #[cfg(feature = "gpu")]
        let val = use_gpu
            .and_then(|g| diversity_gpu::shannon_gpu(g, &counts).ok())
            .unwrap_or_else(|| diversity::shannon(&counts));
        #[cfg(not(feature = "gpu"))]
        let val = diversity::shannon(&counts);
        result.insert("shannon".into(), json!(val));
    }
    if compute_all || metrics.iter().any(|m| m == "simpson") {
        #[cfg(feature = "gpu")]
        let val = use_gpu
            .and_then(|g| diversity_gpu::simpson_gpu(g, &counts).ok())
            .unwrap_or_else(|| diversity::simpson(&counts));
        #[cfg(not(feature = "gpu"))]
        let val = diversity::simpson(&counts);
        result.insert("simpson".into(), json!(val));
    }
    if compute_all || metrics.iter().any(|m| m == "chao1") {
        result.insert("chao1".into(), json!(diversity::chao1(&counts)));
    }
    if compute_all || metrics.iter().any(|m| m == "observed") {
        #[cfg(feature = "gpu")]
        let val = use_gpu
            .and_then(|g| diversity_gpu::observed_features_gpu(g, &counts).ok())
            .unwrap_or_else(|| diversity::observed_features(&counts));
        #[cfg(not(feature = "gpu"))]
        let val = diversity::observed_features(&counts);
        result.insert("observed".into(), json!(val));
    }
    if compute_all || metrics.iter().any(|m| m == "pielou") {
        #[cfg(feature = "gpu")]
        let val = use_gpu
            .and_then(|g| diversity_gpu::pielou_evenness_gpu(g, &counts).ok())
            .unwrap_or_else(|| diversity::pielou_evenness(&counts));
        #[cfg(not(feature = "gpu"))]
        let val = diversity::pielou_evenness(&counts);
        result.insert("pielou".into(), json!(val));
    }

    if let Some(b_counts) = params.get("counts_b").and_then(Value::as_array) {
        let b: Vec<f64> = b_counts.iter().filter_map(Value::as_f64).collect();
        if b.len() == counts.len() {
            result.insert(
                "bray_curtis".into(),
                json!(diversity::bray_curtis(&counts, &b)),
            );
        }
    }

    #[cfg(feature = "gpu")]
    {
        let substrate = if use_gpu.is_some() { "gpu" } else { "cpu" };
        result.insert("substrate".into(), json!(substrate));
    }

    Ok(Value::Object(result))
}

fn handle_qs_model(params: &Value) -> Result<Value, (i64, String)> {
    let scenario = params
        .get("scenario")
        .and_then(Value::as_str)
        .unwrap_or("standard_growth");
    let dt = params.get("dt").and_then(Value::as_f64).unwrap_or(0.01);

    let qs_params = if let Some(flat) = params.get("params").and_then(Value::as_array) {
        let flat_vec: Vec<f64> = flat.iter().filter_map(Value::as_f64).collect();
        if flat_vec.len() < N_PARAMS {
            return Err((
                -32602,
                format!(
                    "params array needs {N_PARAMS} values, got {}",
                    flat_vec.len()
                ),
            ));
        }
        QsBiofilmParams::from_flat(&flat_vec)
    } else {
        QsBiofilmParams::default()
    };

    let result = match scenario {
        "standard_growth" => qs_biofilm::scenario_standard_growth(&qs_params, dt),
        "high_density" => qs_biofilm::scenario_high_density(&qs_params, dt),
        "hapr_mutant" => qs_biofilm::scenario_hapr_mutant(&qs_params, dt),
        "dgc_overexpression" => qs_biofilm::scenario_dgc_overexpression(&qs_params, dt),
        _ => return Err((-32602, format!("unknown scenario: {scenario}"))),
    };

    let peak_biofilm = result
        .states()
        .filter_map(|s| s.get(4).copied())
        .fold(0.0_f64, f64::max);

    Ok(json!({
        "t_end": result.t.last().copied().unwrap_or(0.0),
        "steps": result.steps,
        "final_state": result.y_final,
        "peak_biofilm": peak_biofilm,
    }))
}

fn handle_ncbi_fetch(params: &Value) -> Result<Value, (i64, String)> {
    let db = params
        .get("db")
        .and_then(Value::as_str)
        .unwrap_or("nucleotide");
    let id = params
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| (-32602, "missing required param: id".to_string()))?;

    let api_key = params
        .get("api_key")
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .or_else(crate::ncbi::api_key)
        .unwrap_or_default();

    // Three-tier routing: biomeOS → NestGate → sovereign HTTP
    crate::ncbi::nestgate::fetch_tiered(db, id, &api_key)
        .map(|fasta| {
            let source = if crate::ncbi::nestgate::discover_biomeos_socket().is_some() {
                "biomeos"
            } else if crate::ncbi::nestgate::is_enabled()
                && crate::ncbi::nestgate::discover_socket().is_some()
            {
                "nestgate"
            } else {
                "sovereign"
            };
            json!({"fasta": fasta, "source": source})
        })
        .map_err(|e| (-32000, format!("NCBI fetch failed: {e}")))
}

#[allow(clippy::unnecessary_wraps)]
fn handle_anderson(params: &Value) -> Result<Value, (i64, String)> {
    #[cfg(not(feature = "gpu"))]
    {
        let _ = params;
        Err((
            -32001,
            "science.anderson requires GPU feature (--features gpu)".into(),
        ))
    }

    #[cfg(feature = "gpu")]
    {
        use barracuda::spectral::{
            GOE_R, POISSON_R, anderson_3d, lanczos, lanczos_eigenvalues, level_spacing_ratio,
        };

        let l = params
            .get("lattice_size")
            .and_then(Value::as_u64)
            .unwrap_or(8) as usize;
        let w = params
            .get("disorder")
            .and_then(Value::as_f64)
            .unwrap_or(16.5);
        let seed = params
            .get("seed")
            .and_then(Value::as_u64)
            .unwrap_or(42);

        let n = l * l * l;
        let mat = anderson_3d(l, l, l, w, seed);
        let tri = lanczos(&mat, n, seed);
        let eigs = lanczos_eigenvalues(&tri);
        let r = level_spacing_ratio(&eigs);
        let midpoint = f64::midpoint(GOE_R, POISSON_R);
        let regime = if r > midpoint { "extended" } else { "localized" };

        Ok(json!({
            "status": "computed",
            "substrate": "gpu",
            "lattice_size": l,
            "disorder": w,
            "n_sites": n,
            "level_spacing_ratio": r,
            "regime": regime,
            "goe_r": GOE_R,
            "poisson_r": POISSON_R,
        }))
    }
}

fn handle_full_pipeline(params: &Value) -> Result<Value, (i64, String)> {
    let mut pipeline_result = serde_json::Map::new();

    // Stage 1: Diversity (if counts provided)
    if params.get("counts").is_some_and(Value::is_array) {
        let diversity = handle_diversity(params)?;
        pipeline_result.insert("diversity".into(), diversity);
    }

    // Stage 2: QS model
    let qs_params = if params.get("scenario").is_some() || params.get("dt").is_some() {
        params.clone()
    } else {
        json!({"scenario": "standard_growth"})
    };
    let qs_result = handle_qs_model(&qs_params)?;
    pipeline_result.insert("qs_model".into(), qs_result);

    // Stage 3: Anderson (non-fatal if GPU unavailable)
    match handle_anderson(params) {
        Ok(anderson) => {
            pipeline_result.insert("anderson".into(), anderson);
        }
        Err(_) => {
            pipeline_result.insert(
                "anderson".into(),
                json!({"status": "skipped", "reason": "gpu not available"}),
            );
        }
    }

    pipeline_result.insert("pipeline".into(), json!("complete"));
    Ok(Value::Object(pipeline_result))
}

/// Extract an `f64` array from a JSON params object by key.
fn extract_f64_array(params: &Value, key: &str) -> Result<Vec<f64>, (i64, String)> {
    let arr = params
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| (-32602, format!("missing or invalid param: {key}")))?;

    Ok(arr.iter().filter_map(Value::as_f64).collect())
}

/// Extract a string array from a JSON params object by key.
/// Returns empty vec if key is missing.
fn extract_string_array(params: &Value, key: &str) -> Vec<String> {
    params
        .get(key)
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(Value::as_str)
                .map(String::from)
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_health_check() {
        let result = dispatch("health.check", &json!({})).unwrap();
        assert_eq!(result["status"], "healthy");
        assert_eq!(result["primal"], "wetspring");
    }

    #[test]
    fn dispatch_unknown_method() {
        let err = dispatch("nonexistent.method", &json!({})).unwrap_err();
        assert_eq!(err.0, -32601);
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
        assert_eq!(err.0, -32602);
    }

    #[test]
    fn diversity_missing_counts() {
        let err = dispatch("science.diversity", &json!({})).unwrap_err();
        assert_eq!(err.0, -32602);
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
        assert_eq!(err.0, -32602);
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
        assert_eq!(err.0, -32602);
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
