// SPDX-License-Identifier: AGPL-3.0-or-later
#![expect(clippy::missing_errors_doc)]
//! JSON-RPC method handlers for science capabilities.
//!
//! Each handler maps a method + params to barracuda library calls.
//! The dispatch layer in [`super::dispatch`] routes method names here.
//!
//! # Sub-modules
//!
//! | Module    | Handlers                                              |
//! |-----------|-------------------------------------------------------|
//! | `brain`   | `brain.observe`, `brain.attention`, `brain.urgency`   |
//! | `science` | `science.diversity`, `science.qs_model`, Anderson, …  |

mod brain;
mod expanded;
mod science;

pub use brain::{handle_brain_attention, handle_brain_observe, handle_brain_urgency};
pub use expanded::{
    handle_alignment, handle_kinetics, handle_nmf, handle_phylogenetics, handle_taxonomy,
};
pub use science::{
    handle_anderson, handle_diversity, handle_full_pipeline, handle_ncbi_fetch, handle_qs_model,
};

use serde_json::{Value, json};

use crate::ipc::protocol::RpcError;

/// Capabilities advertised by this primal, derived from the dispatch table.
pub const CAPABILITIES: &[&str] = &[
    "science.diversity",
    "science.anderson",
    "science.qs_model",
    "science.ncbi_fetch",
    "science.full_pipeline",
    "science.kinetics",
    "science.alignment",
    "science.taxonomy",
    "science.phylogenetics",
    "science.nmf",
    "science.timeseries",
    "science.timeseries_diversity",
    "provenance.begin",
    "provenance.record",
    "provenance.complete",
    "brain.observe",
    "brain.attention",
    "brain.urgency",
    "metrics.snapshot",
];

#[cfg(feature = "gpu")]
use std::sync::OnceLock;

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
        "primal": crate::PRIMAL_NAME,
        "version": env!("CARGO_PKG_VERSION"),
        "substrate": substrate,
        "capabilities": CAPABILITIES,
    }))
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
