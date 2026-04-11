// SPDX-License-Identifier: AGPL-3.0-or-later
#![expect(
    clippy::missing_errors_doc,
    reason = "IPC handlers: error documentation at module level"
)]
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

mod ai;
mod alignment;
mod anderson;
mod brain;
mod data_fetch;
mod drug;
mod expanded;
mod gonzales;
mod kinetics;
mod phylogenetics;
mod science;
mod taxonomy;
mod vault_ipc;

pub use ai::handle_ai_ecology_interpret;
pub use brain::{handle_brain_attention, handle_brain_observe, handle_brain_urgency};
pub use data_fetch::{handle_chembl_fetch, handle_pubchem_fetch, handle_register_table};
pub use expanded::{
    handle_alignment, handle_kinetics, handle_nmf, handle_phylogenetics, handle_taxonomy,
};
pub use gonzales::{
    handle_biome_atlas, handle_cross_species, handle_disorder_sweep, handle_dose_response,
    handle_hormesis, handle_pk_decay, handle_tissue_lattice,
};
pub use science::{
    handle_anderson, handle_diversity, handle_full_pipeline, handle_ncbi_fetch, handle_qs_model,
};
pub use vault_ipc::{handle_vault_consent_verify, handle_vault_retrieve, handle_vault_store};

use serde_json::{Value, json};

const DEPLOY_GRAPH_NAME: &str = "wetspring_science_nucleus.toml";

use crate::ipc::protocol::RpcError;

/// Capabilities advertised by this primal, derived from the dispatch table.
///
/// Kept in sync with [`crate::ipc::capability_domains::DOMAINS`] — the domain
/// registry is the source of truth; this list adds `capability.list` (the
/// introspection method that is not itself a domain capability).
pub const CAPABILITIES: &[&str] = &[
    "capability.list",
    "health.check",
    "health.liveness",
    "health.readiness",
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
    "science.gonzales.dose_response",
    "science.gonzales.pk_decay",
    "science.gonzales.tissue_lattice",
    "science.anderson.biome_atlas",
    "science.anderson.disorder_sweep",
    "science.anderson.hormesis",
    "science.anderson.cross_species",
    "provenance.begin",
    "provenance.record",
    "provenance.complete",
    "brain.observe",
    "brain.attention",
    "brain.urgency",
    "metrics.snapshot",
    "ai.ecology_interpret",
    "data.fetch.chembl",
    "data.fetch.pubchem",
    "data.fetch.register_table",
    "vault.store",
    "vault.retrieve",
    "vault.consent.verify",
    "composition.science_health",
    "composition.tower_health",
    "composition.node_health",
    "composition.nest_health",
    "composition.nucleus_health",
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

/// Capability listing per Spring-as-Niche Deployment Standard.
///
/// Returns all domains with their methods, descriptions, and GPU status.
pub fn handle_capability_list() -> Result<Value, RpcError> {
    use crate::ipc::capability_domains::DOMAINS;

    let domains: Vec<Value> = DOMAINS
        .iter()
        .map(|d| {
            json!({
                "name": d.name,
                "description": d.description,
                "methods": d.methods,
            })
        })
        .collect();

    let mut response = json!({
        "primal": crate::PRIMAL_NAME,
        "version": env!("CARGO_PKG_VERSION"),
        "domain": crate::ipc::capability_domains::DOMAIN,
        "capabilities": CAPABILITIES,
        "domains": domains,
    });

    #[cfg(feature = "json")]
    if let Some(obj) = response.as_object_mut() {
        obj.insert(
            "operation_dependencies".to_string(),
            crate::niche::operation_dependencies(),
        );
        obj.insert("cost_estimates".to_string(), crate::niche::cost_estimates());
        obj.insert(
            "semantic_mappings".to_string(),
            crate::niche::ecology_semantic_mappings(),
        );
    }

    Ok(response)
}

/// Health/readiness probe (legacy — delegates to full readiness).
pub fn handle_health() -> Result<Value, RpcError> {
    handle_health_readiness()
}

/// Minimal liveness probe — confirms the process is alive and responsive.
///
/// No subsystem checks — just an immediate `{"alive": true}` response.
/// Used by biomeOS/sweetGrass orchestrators for fast keep-alive polling.
pub fn handle_health_liveness() -> Result<Value, RpcError> {
    Ok(json!({
        "alive": true,
        "primal": crate::PRIMAL_NAME,
    }))
}

/// Deep readiness probe — checks subsystem health before reporting ready.
///
/// Reports per-subsystem status (`gpu`, `ipc`, `math`) so orchestrators
/// can make routing decisions (e.g. degrade to CPU-only if GPU is lost).
pub fn handle_health_readiness() -> Result<Value, RpcError> {
    #[cfg(feature = "gpu")]
    let (substrate, gpu_ready) = if try_gpu().is_some() {
        ("gpu", true)
    } else if GPU_CTX
        .get()
        .is_some_and(|g| g.as_ref().is_some_and(GpuF64::is_lost))
    {
        ("gpu_lost", false)
    } else {
        ("cpu", false)
    };
    #[cfg(not(feature = "gpu"))]
    let (substrate, gpu_ready) = ("cpu", false);

    let subsystems = json!({
        "math": true,
        "gpu": gpu_ready,
        "ipc": true,
    });

    Ok(json!({
        "ready": true,
        "status": "healthy",
        "primal": crate::PRIMAL_NAME,
        "version": env!("CARGO_PKG_VERSION"),
        "substrate": substrate,
        "capabilities": CAPABILITIES,
        "subsystems": subsystems,
    }))
}

/// Cross-spring composition health check following primalSpring's
/// `composition.*_health` pattern. Reports subsystem readiness so
/// remote springs or primalSpring validators can verify this NUCLEUS
/// is operational before attempting ionic bond interactions.
pub fn handle_composition_science_health(_params: &Value) -> Result<Value, RpcError> {
    #[cfg(feature = "gpu")]
    let gpu_status = if try_gpu().is_some() {
        "available"
    } else {
        "unavailable"
    };
    #[cfg(not(feature = "gpu"))]
    let gpu_status = "not_compiled";

    let subsystems = json!({
        "ipc": true,
        "math": true,
        "gpu": gpu_status,
        "provenance_trio": "deferred_check",
        "nestgate": "deferred_check",
    });

    let science_domains = json!([
        "gonzales_dermatitis",
        "anderson_localization",
        "tissue_geometry",
        "pharmacokinetics",
        "hormesis",
        "cross_species",
    ]);

    Ok(json!({
        "healthy": true,
        "spring": "wetSpring",
        "deploy_graph": DEPLOY_GRAPH_NAME,
        "version": env!("CARGO_PKG_VERSION"),
        "subsystems": subsystems,
        "science_domains": science_domains,
        "capabilities_count": CAPABILITIES.len(),
        "bonding_support": ["Covalent", "Ionic"],
    }))
}

/// Canonical composition health: Tower (BearDog + Songbird).
///
/// Probes the security and discovery primals via Neural API to report
/// whether the Tower atomic layer is operational.
pub fn handle_composition_tower_health(_params: &Value) -> Result<Value, RpcError> {
    let beardog = probe_capability("security");
    let songbird = probe_capability("discovery");
    let healthy = beardog.is_some() && songbird.is_some();

    Ok(json!({
        "healthy": healthy,
        "atomic": "Tower",
        "spring": "wetSpring",
        "components": {
            "beardog": beardog.unwrap_or(json!("unreachable")),
            "songbird": songbird.unwrap_or(json!("unreachable")),
        },
    }))
}

/// Canonical composition health: Node (Tower + ToadStool).
pub fn handle_composition_node_health(_params: &Value) -> Result<Value, RpcError> {
    let beardog = probe_capability("security");
    let toadstool = probe_capability("compute");
    let healthy = beardog.is_some() && toadstool.is_some();

    Ok(json!({
        "healthy": healthy,
        "atomic": "Node",
        "spring": "wetSpring",
        "components": {
            "beardog": beardog.unwrap_or(json!("unreachable")),
            "toadstool": toadstool.unwrap_or(json!("unreachable")),
        },
    }))
}

/// Canonical composition health: Nest (Tower + NestGate).
pub fn handle_composition_nest_health(_params: &Value) -> Result<Value, RpcError> {
    let beardog = probe_capability("security");
    let nestgate = probe_capability("storage");
    let healthy = beardog.is_some() && nestgate.is_some();

    Ok(json!({
        "healthy": healthy,
        "atomic": "Nest",
        "spring": "wetSpring",
        "components": {
            "beardog": beardog.unwrap_or(json!("unreachable")),
            "nestgate": nestgate.unwrap_or(json!("unreachable")),
        },
    }))
}

/// Canonical composition health: full NUCLEUS.
///
/// Aggregates Tower + Node + Nest + provenance trio status.
pub fn handle_composition_nucleus_health(_params: &Value) -> Result<Value, RpcError> {
    let beardog = probe_capability("security");
    let songbird = probe_capability("discovery");
    let toadstool = probe_capability("compute");
    let nestgate = probe_capability("storage");
    let rhizocrypt = probe_capability("provenance");
    let loamspine = probe_capability("ledger");
    let sweetgrass = probe_capability("attribution");

    let tower_ok = beardog.is_some() && songbird.is_some();
    let node_ok = tower_ok && toadstool.is_some();
    let nest_ok = beardog.is_some() && nestgate.is_some();
    let trio_ok = rhizocrypt.is_some() && loamspine.is_some() && sweetgrass.is_some();
    let nucleus_ok = tower_ok && nest_ok;

    Ok(json!({
        "healthy": nucleus_ok,
        "atomic": "NUCLEUS",
        "spring": "wetSpring",
        "tiers": {
            "tower": tower_ok,
            "node": node_ok,
            "nest": nest_ok,
            "provenance_trio": trio_ok,
        },
        "components": {
            "beardog": beardog.unwrap_or(json!("unreachable")),
            "songbird": songbird.unwrap_or(json!("unreachable")),
            "toadstool": toadstool.unwrap_or(json!("unreachable")),
            "nestgate": nestgate.unwrap_or(json!("unreachable")),
            "rhizocrypt": rhizocrypt.unwrap_or(json!("unreachable")),
            "loamspine": loamspine.unwrap_or(json!("unreachable")),
            "sweetgrass": sweetgrass.unwrap_or(json!("unreachable")),
        },
    }))
}

/// Probe a capability domain via Neural API capability.discover.
fn probe_capability(domain: &str) -> Option<Value> {
    let family_id = std::env::var("FAMILY_ID").ok()?;
    let runtime = std::env::var("XDG_RUNTIME_DIR")
        .unwrap_or_else(|_| std::env::temp_dir().to_string_lossy().into_owned());
    let socket_path = std::path::PathBuf::from(runtime)
        .join("biomeos")
        .join(format!("neural-api-{family_id}.sock"));
    if !socket_path.exists() {
        return None;
    }

    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::UnixStream;

    let mut stream = UnixStream::connect(&socket_path).ok()?;
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(3)))
        .ok();

    let request = json!({
        "jsonrpc": "2.0",
        "method": "capability.discover",
        "params": { "domain": domain },
        "id": 1,
    });
    let mut line = serde_json::to_string(&request).ok()?;
    line.push('\n');
    stream.write_all(line.as_bytes()).ok()?;
    stream.flush().ok()?;

    let mut reader = BufReader::new(stream);
    let mut resp_line = String::new();
    reader.read_line(&mut resp_line).ok()?;

    let resp: Value = serde_json::from_str(resp_line.trim()).ok()?;
    resp.get("result").cloned()
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
