// SPDX-License-Identifier: AGPL-3.0-or-later
//! JSON-RPC method handlers for science capabilities.
//!
//! Each handler maps a method + params to barracuda library calls.
//! The dispatch layer in `dispatch.rs` routes method names here.

use serde_json::{Value, json};

use crate::bio::diversity;
use crate::bio::qs_biofilm::{self, N_PARAMS, QsBiofilmParams};
use crate::ipc::protocol::RpcError;

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

/// Health/readiness probe.
pub fn handle_health() -> Result<Value, RpcError> {
    #![allow(clippy::unnecessary_wraps)]

    #[cfg(feature = "gpu")]
    let substrate = if try_gpu().is_some() {
        "gpu"
    } else if GPU_CTX
        .get()
        .is_some_and(|g| g.as_ref().is_some_and(GpuF64::is_lost))
    {
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

/// Alpha diversity metrics (Shannon, Simpson, Chao1, etc.).
pub fn handle_diversity(params: &Value) -> Result<Value, RpcError> {
    let counts = extract_f64_array(params, "counts")?;
    if counts.is_empty() {
        return Err(RpcError::invalid_params("counts array is empty"));
    }

    let metrics = extract_string_array(params, "metrics");
    let compute_all = metrics.is_empty() || metrics.iter().any(|m| m == "all");

    let mut result = serde_json::Map::new();

    #[cfg(feature = "gpu")]
    let use_gpu = try_gpu().filter(|g| counts.len() >= g.dispatch_threshold());

    #[cfg(feature = "gpu")]
    {
        insert_shannon_if_requested(&mut result, compute_all, &metrics, &counts, use_gpu);
        insert_simpson_if_requested(&mut result, compute_all, &metrics, &counts, use_gpu);
        insert_observed_if_requested(&mut result, compute_all, &metrics, &counts, use_gpu);
        insert_pielou_if_requested(&mut result, compute_all, &metrics, &counts, use_gpu);
    }
    #[cfg(not(feature = "gpu"))]
    {
        insert_shannon_if_requested(&mut result, compute_all, &metrics, &counts);
        insert_simpson_if_requested(&mut result, compute_all, &metrics, &counts);
        insert_observed_if_requested(&mut result, compute_all, &metrics, &counts);
        insert_pielou_if_requested(&mut result, compute_all, &metrics, &counts);
    }
    insert_chao1_if_requested(&mut result, compute_all, &metrics, &counts);
    insert_bray_curtis_if_present(&mut result, params, &counts);

    #[cfg(feature = "gpu")]
    {
        let substrate = if use_gpu.is_some() { "gpu" } else { "cpu" };
        result.insert("substrate".into(), json!(substrate));
    }

    Ok(Value::Object(result))
}

#[cfg(feature = "gpu")]
fn insert_shannon_if_requested(
    result: &mut serde_json::Map<String, Value>,
    compute_all: bool,
    metrics: &[String],
    counts: &[f64],
    use_gpu: Option<&'static GpuF64>,
) {
    if !compute_all && !metrics.iter().any(|m| m == "shannon") {
        return;
    }
    let val = use_gpu
        .and_then(|g| diversity_gpu::shannon_gpu(g, counts).ok())
        .unwrap_or_else(|| diversity::shannon(counts));
    result.insert("shannon".into(), json!(val));
}

#[cfg(not(feature = "gpu"))]
fn insert_shannon_if_requested(
    result: &mut serde_json::Map<String, Value>,
    compute_all: bool,
    metrics: &[String],
    counts: &[f64],
) {
    if !compute_all && !metrics.iter().any(|m| m == "shannon") {
        return;
    }
    result.insert("shannon".into(), json!(diversity::shannon(counts)));
}

#[cfg(feature = "gpu")]
fn insert_simpson_if_requested(
    result: &mut serde_json::Map<String, Value>,
    compute_all: bool,
    metrics: &[String],
    counts: &[f64],
    use_gpu: Option<&'static GpuF64>,
) {
    if !compute_all && !metrics.iter().any(|m| m == "simpson") {
        return;
    }
    let val = use_gpu
        .and_then(|g| diversity_gpu::simpson_gpu(g, counts).ok())
        .unwrap_or_else(|| diversity::simpson(counts));
    result.insert("simpson".into(), json!(val));
}

#[cfg(not(feature = "gpu"))]
fn insert_simpson_if_requested(
    result: &mut serde_json::Map<String, Value>,
    compute_all: bool,
    metrics: &[String],
    counts: &[f64],
) {
    if !compute_all && !metrics.iter().any(|m| m == "simpson") {
        return;
    }
    result.insert("simpson".into(), json!(diversity::simpson(counts)));
}

fn insert_chao1_if_requested(
    result: &mut serde_json::Map<String, Value>,
    compute_all: bool,
    metrics: &[String],
    counts: &[f64],
) {
    if !compute_all && !metrics.iter().any(|m| m == "chao1") {
        return;
    }
    result.insert("chao1".into(), json!(diversity::chao1(counts)));
}

#[cfg(feature = "gpu")]
fn insert_observed_if_requested(
    result: &mut serde_json::Map<String, Value>,
    compute_all: bool,
    metrics: &[String],
    counts: &[f64],
    use_gpu: Option<&'static GpuF64>,
) {
    if !compute_all && !metrics.iter().any(|m| m == "observed") {
        return;
    }
    let val = use_gpu
        .and_then(|g| diversity_gpu::observed_features_gpu(g, counts).ok())
        .unwrap_or_else(|| diversity::observed_features(counts));
    result.insert("observed".into(), json!(val));
}

#[cfg(not(feature = "gpu"))]
fn insert_observed_if_requested(
    result: &mut serde_json::Map<String, Value>,
    compute_all: bool,
    metrics: &[String],
    counts: &[f64],
) {
    if !compute_all && !metrics.iter().any(|m| m == "observed") {
        return;
    }
    result.insert(
        "observed".into(),
        json!(diversity::observed_features(counts)),
    );
}

#[cfg(feature = "gpu")]
fn insert_pielou_if_requested(
    result: &mut serde_json::Map<String, Value>,
    compute_all: bool,
    metrics: &[String],
    counts: &[f64],
    use_gpu: Option<&'static GpuF64>,
) {
    if !compute_all && !metrics.iter().any(|m| m == "pielou") {
        return;
    }
    let val = use_gpu
        .and_then(|g| diversity_gpu::pielou_evenness_gpu(g, counts).ok())
        .unwrap_or_else(|| diversity::pielou_evenness(counts));
    result.insert("pielou".into(), json!(val));
}

#[cfg(not(feature = "gpu"))]
fn insert_pielou_if_requested(
    result: &mut serde_json::Map<String, Value>,
    compute_all: bool,
    metrics: &[String],
    counts: &[f64],
) {
    if !compute_all && !metrics.iter().any(|m| m == "pielou") {
        return;
    }
    result.insert("pielou".into(), json!(diversity::pielou_evenness(counts)));
}

fn insert_bray_curtis_if_present(
    result: &mut serde_json::Map<String, Value>,
    params: &Value,
    counts: &[f64],
) {
    if let Some(b_counts) = params.get("counts_b").and_then(Value::as_array) {
        let b: Vec<f64> = b_counts.iter().filter_map(Value::as_f64).collect();
        if b.len() == counts.len() {
            result.insert(
                "bray_curtis".into(),
                json!(diversity::bray_curtis(counts, &b)),
            );
        }
    }
}

/// QS/c-di-GMP biofilm ODE integration.
pub fn handle_qs_model(params: &Value) -> Result<Value, RpcError> {
    let scenario = params
        .get("scenario")
        .and_then(Value::as_str)
        .unwrap_or("standard_growth");
    let dt = params.get("dt").and_then(Value::as_f64).unwrap_or(0.01);

    let qs_params = if let Some(flat) = params.get("params").and_then(Value::as_array) {
        let flat_vec: Vec<f64> = flat.iter().filter_map(Value::as_f64).collect();
        if flat_vec.len() < N_PARAMS {
            return Err(RpcError::invalid_params(format!(
                "params array needs {N_PARAMS} values, got {}",
                flat_vec.len()
            )));
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
        _ => {
            return Err(RpcError::invalid_params(format!(
                "unknown scenario: {scenario}"
            )));
        }
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

/// NCBI sequence retrieval.
pub fn handle_ncbi_fetch(params: &Value) -> Result<Value, RpcError> {
    let db = params
        .get("db")
        .and_then(Value::as_str)
        .unwrap_or("nucleotide");
    let id = params
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| RpcError::invalid_params("missing required param: id"))?;

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
        .map_err(|e| RpcError::server_error(-32000, format!("NCBI fetch failed: {e}")))
}

/// Anderson spectral analysis (GPU-preferred).
#[allow(clippy::unnecessary_wraps)]
pub fn handle_anderson(params: &Value) -> Result<Value, RpcError> {
    #[cfg(not(feature = "gpu"))]
    {
        let _ = params;
        Err(RpcError::server_error(
            -32001,
            "science.anderson requires GPU feature (--features gpu)",
        ))
    }

    #[cfg(feature = "gpu")]
    {
        use barracuda::spectral::{
            GOE_R, POISSON_R, anderson_3d, lanczos, lanczos_eigenvalues, level_spacing_ratio,
        };

        let l_raw = params
            .get("lattice_size")
            .and_then(Value::as_u64)
            .unwrap_or(8);
        let l = usize::try_from(l_raw).unwrap_or(8);
        let w = params
            .get("disorder")
            .and_then(Value::as_f64)
            .unwrap_or(16.5);
        let seed = params.get("seed").and_then(Value::as_u64).unwrap_or(42);

        let n = l * l * l;
        let mat = anderson_3d(l, l, l, w, seed);
        let tri = lanczos(&mat, n, seed);
        let eigs = lanczos_eigenvalues(&tri);
        let r = level_spacing_ratio(&eigs);
        let midpoint = f64::midpoint(GOE_R, POISSON_R);
        let regime = if r > midpoint {
            "extended"
        } else {
            "localized"
        };

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

/// Full pipeline: diversity + QS model + Anderson (non-fatal).
pub fn handle_full_pipeline(params: &Value) -> Result<Value, RpcError> {
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
///
/// # Errors
///
/// Returns `RpcError::invalid_params` if the key is missing or the value is not an array.
pub fn extract_f64_array(params: &Value, key: &str) -> Result<Vec<f64>, RpcError> {
    let arr = params
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| RpcError::invalid_params(format!("missing or invalid param: {key}")))?;

    Ok(arr.iter().filter_map(Value::as_f64).collect())
}

/// Extract a string array from a JSON params object by key.
/// Returns empty vec if key is missing.
pub fn extract_string_array(params: &Value, key: &str) -> Vec<String> {
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
