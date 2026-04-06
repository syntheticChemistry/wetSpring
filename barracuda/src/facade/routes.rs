// SPDX-License-Identifier: AGPL-3.0-or-later
//! Axum route handlers for the science facade.

use axum::extract::Query;
use axum::http::StatusCode;
use axum::response::Json;
use serde::Deserialize;
use serde_json::{Value, json};

use super::{ipc_client, provenance, shaping};

const MAX_POINTS: u64 = 1000;
const MAX_DOSE: f64 = 10_000.0;
const MAX_DISORDER: f64 = 100.0;
const MAX_DAYS: f64 = 365.0;

fn clamp_u64(v: u64, max: u64) -> u64 {
    v.min(max)
}

fn clamp_f64(v: f64, max: f64) -> f64 {
    v.min(max).max(0.0)
}

/// Query parameters for the IC50 dose-response endpoint.
#[derive(Deserialize)]
pub struct DoseResponseParams {
    /// Number of dose points (capped at 1000).
    pub n_points: Option<u64>,
    /// Maximum dose in nM (capped at 10000).
    pub dose_max: Option<f64>,
    /// Hill coefficient (0.1..10.0).
    pub hill_n: Option<f64>,
}

/// Query parameters for the PK decay endpoint.
#[derive(Deserialize)]
pub struct PkDecayParams {
    /// Number of time points (capped at 1000).
    pub n_points: Option<u64>,
    /// Maximum time in days (capped at 365).
    pub t_max_days: Option<f64>,
}

/// Query parameters for the tissue lattice endpoint.
#[derive(Deserialize)]
pub struct TissueLatticeParams {
    /// Base disorder W (capped at 100).
    pub disorder: Option<f64>,
    /// Number of severity profiles (max 6).
    pub n_profiles: Option<u64>,
    /// Random seed for lattice generation.
    pub seed: Option<u64>,
}

/// Query parameters for the hormesis endpoint.
#[derive(Deserialize)]
pub struct HormesisParams {
    /// Stimulatory amplitude.
    pub amplitude: Option<f64>,
    /// Half-max stimulatory dose.
    pub k_stim: Option<f64>,
    /// Stimulatory Hill coefficient.
    pub n_stim: Option<f64>,
    /// Half-max inhibitory dose.
    pub k_inh: Option<f64>,
    /// Inhibitory Hill coefficient.
    pub n_inh: Option<f64>,
    /// Number of dose points (capped at 1000).
    pub n_points: Option<u64>,
    /// Maximum dose (capped at 10000).
    pub dose_max: Option<f64>,
}

/// Query parameters for the cross-species endpoint.
#[derive(Deserialize)]
pub struct CrossSpeciesParams {}

/// Health check — bypasses Dark Forest gate.
pub async fn health() -> Json<Value> {
    let ipc_ok = ipc_client::call("health.check", &json!({})).is_ok();
    Json(json!({
        "status": if ipc_ok { "ok" } else { "degraded" },
        "facade": "wetspring-science-facade",
        "version": env!("CARGO_PKG_VERSION"),
        "wetspring_ipc": if ipc_ok { "connected" } else { "unreachable" },
    }))
}

/// IC50 dose-response sweep for 6 cytokine pathways.
pub async fn dose_response(
    Query(params): Query<DoseResponseParams>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let rpc_params = json!({
        "n_points": clamp_u64(params.n_points.unwrap_or(50), MAX_POINTS),
        "dose_max": clamp_f64(params.dose_max.unwrap_or(500.0), MAX_DOSE),
        "hill_n": params.hill_n.unwrap_or(1.0).max(0.1).min(10.0),
    });

    let result = ipc_client::call("science.gonzales.dose_response", &rpc_params)
        .map_err(ipc_error)?;

    let node = shaping::shape_dose_response(&result);
    let prov = provenance::envelope("science.gonzales.dose_response", &rpc_params, &result);

    Ok(Json(shaping::scenario_envelope(
        "Gonzales Dose-Response (Live)",
        "IC50 dose-response for 6 cytokine pathways — live from wetSpring",
        vec![node],
        &prov,
    )))
}

/// Lokivetmab pharmacokinetic decay profiles.
pub async fn pk_decay(
    Query(params): Query<PkDecayParams>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let rpc_params = json!({
        "n_points": clamp_u64(params.n_points.unwrap_or(100), MAX_POINTS),
        "t_max_days": clamp_f64(params.t_max_days.unwrap_or(56.0), MAX_DAYS),
    });

    let result = ipc_client::call("science.gonzales.pk_decay", &rpc_params)
        .map_err(ipc_error)?;

    let node = shaping::shape_pk_decay(&result);
    let prov = provenance::envelope("science.gonzales.pk_decay", &rpc_params, &result);

    Ok(Json(shaping::scenario_envelope(
        "Lokivetmab PK Decay (Live)",
        "Pharmacokinetic decay profiles — live from wetSpring",
        vec![node],
        &prov,
    )))
}

/// Anderson tissue lattice with skin-layer geometry.
pub async fn tissue_lattice(
    Query(params): Query<TissueLatticeParams>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let rpc_params = json!({
        "disorder": clamp_f64(params.disorder.unwrap_or(10.0), MAX_DISORDER),
        "n_profiles": clamp_u64(params.n_profiles.unwrap_or(6), 6),
        "seed": params.seed.unwrap_or(42),
    });

    let result = ipc_client::call("science.gonzales.tissue_lattice", &rpc_params)
        .map_err(ipc_error)?;

    let node = shaping::shape_tissue_lattice(&result);
    let prov = provenance::envelope("science.gonzales.tissue_lattice", &rpc_params, &result);

    Ok(Json(shaping::scenario_envelope(
        "Tissue Geometry (Live)",
        "AD severity profiles with Anderson disorder mapping — live from wetSpring",
        vec![node],
        &prov,
    )))
}

/// Biphasic hormesis dose-response with Anderson disorder mapping.
pub async fn hormesis(
    Query(params): Query<HormesisParams>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let rpc_params = json!({
        "amplitude": params.amplitude.unwrap_or(0.3),
        "k_stim": params.k_stim.unwrap_or(10.0),
        "n_stim": params.n_stim.unwrap_or(2.0),
        "k_inh": params.k_inh.unwrap_or(100.0),
        "n_inh": params.n_inh.unwrap_or(2.0),
        "n_points": clamp_u64(params.n_points.unwrap_or(100), MAX_POINTS),
        "dose_max": clamp_f64(params.dose_max.unwrap_or(200.0), MAX_DOSE),
    });

    let result = ipc_client::call("science.anderson.hormesis", &rpc_params)
        .map_err(ipc_error)?;

    let node = shaping::shape_hormesis(&result);
    let prov = provenance::envelope("science.anderson.hormesis", &rpc_params, &result);

    Ok(Json(shaping::scenario_envelope(
        "Hormesis (Live)",
        "Biphasic dose-response with Anderson disorder mapping — live from wetSpring",
        vec![node],
        &prov,
    )))
}

/// Cross-species tissue geometry comparison.
pub async fn cross_species(
    Query(_params): Query<CrossSpeciesParams>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let rpc_params = json!({});

    let result = ipc_client::call("science.anderson.cross_species", &rpc_params)
        .map_err(ipc_error)?;

    let node = shaping::shape_cross_species(&result);
    let prov = provenance::envelope("science.anderson.cross_species", &rpc_params, &result);

    Ok(Json(shaping::scenario_envelope(
        "Cross-Species Comparison (Live)",
        "Tissue geometry across dog, cat, human, horse, mouse — live from wetSpring",
        vec![node],
        &prov,
    )))
}

/// 28-biome Anderson QS atlas.
pub async fn biome_atlas() -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let rpc_params = json!({});
    let result = ipc_client::call("science.anderson.biome_atlas", &rpc_params)
        .map_err(ipc_error)?;

    let prov = provenance::envelope("science.anderson.biome_atlas", &rpc_params, &result);

    Ok(Json(json!({
        "name": "Biome Atlas (Live)",
        "description": "28-biome Anderson QS atlas — live from wetSpring",
        "version": env!("CARGO_PKG_VERSION"),
        "mode": "live",
        "domain": "anderson_atlas",
        "result": result,
        "provenance": prov,
    })))
}

/// Anderson localization finite-size scaling sweep.
pub async fn disorder_sweep() -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let rpc_params = json!({});
    let result = ipc_client::call("science.anderson.disorder_sweep", &rpc_params)
        .map_err(ipc_error)?;

    let prov = provenance::envelope("science.anderson.disorder_sweep", &rpc_params, &result);

    Ok(Json(json!({
        "name": "Disorder Sweep (Live)",
        "description": "Anderson localization finite-size scaling — live from wetSpring",
        "version": env!("CARGO_PKG_VERSION"),
        "mode": "live",
        "domain": "anderson_disorder",
        "result": result,
        "provenance": prov,
    })))
}

/// Provenance query endpoint — look up a result by content hash.
pub async fn provenance_query(
    axum::extract::Path(result_id): axum::extract::Path<String>,
) -> Json<Value> {
    if let Some(tier3) = provenance::try_tier3(&result_id) {
        Json(tier3)
    } else {
        Json(json!({
            "error": "provenance_unavailable",
            "message": "Provenance trio not reachable or result_id not found",
            "result_id": result_id,
        }))
    }
}

/// Full dashboard — combines all Gonzales + Anderson endpoints.
pub async fn full_dashboard() -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let dr = ipc_client::call("science.gonzales.dose_response", &json!({}))
        .map_err(ipc_error)?;
    let pk = ipc_client::call("science.gonzales.pk_decay", &json!({}))
        .map_err(ipc_error)?;
    let tissue = ipc_client::call("science.gonzales.tissue_lattice", &json!({}))
        .map_err(ipc_error)?;
    let horm = ipc_client::call("science.anderson.hormesis", &json!({}))
        .map_err(ipc_error)?;
    let xs = ipc_client::call("science.anderson.cross_species", &json!({}))
        .map_err(ipc_error)?;

    let nodes = vec![
        shaping::shape_dose_response(&dr),
        shaping::shape_pk_decay(&pk),
        shaping::shape_tissue_lattice(&tissue),
        shaping::shape_hormesis(&horm),
        shaping::shape_cross_species(&xs),
    ];

    let prov = provenance::envelope("science.gonzales.full_dashboard", &json!({}), &json!({
        "endpoints": ["dose_response", "pk_decay", "tissue_lattice", "hormesis", "cross_species"],
    }));

    Ok(Json(shaping::scenario_envelope(
        "Full Gonzales Dashboard (Live)",
        "Complete Gonzales dermatitis + Anderson science — live from wetSpring",
        nodes,
        &prov,
    )))
}

// ── petalTongue grammar rendering ─────────────────────────────────────

/// Grammar-rendered dose-response (SVG from petalTongue).
pub async fn grammar_dose_response(
    Query(params): Query<DoseResponseParams>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let rpc_params = json!({
        "n_points": clamp_u64(params.n_points.unwrap_or(50), MAX_POINTS),
        "dose_max": clamp_f64(params.dose_max.unwrap_or(500.0), MAX_DOSE),
        "hill_n": params.hill_n.unwrap_or(1.0).max(0.1).min(10.0),
    });
    let result = ipc_client::call("science.gonzales.dose_response", &rpc_params)
        .map_err(ipc_error)?;
    let (grammar, data) = super::grammar::dose_response_grammar(&result);
    grammar_response("gonzales_dose_response", &grammar, &data, "health")
}

/// Grammar-rendered PK decay (SVG from petalTongue).
pub async fn grammar_pk_decay(
    Query(params): Query<PkDecayParams>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let rpc_params = json!({
        "n_points": clamp_u64(params.n_points.unwrap_or(100), MAX_POINTS),
        "t_max_days": clamp_f64(params.t_max_days.unwrap_or(56.0), MAX_DAYS),
    });
    let result = ipc_client::call("science.gonzales.pk_decay", &rpc_params)
        .map_err(ipc_error)?;
    let (grammar, data) = super::grammar::pk_decay_grammar(&result);
    grammar_response("gonzales_pk", &grammar, &data, "health")
}

/// Grammar-rendered tissue lattice (SVG from petalTongue).
pub async fn grammar_tissue_lattice(
    Query(params): Query<TissueLatticeParams>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let rpc_params = json!({
        "disorder": clamp_f64(params.disorder.unwrap_or(10.0), MAX_DISORDER),
        "n_profiles": clamp_u64(params.n_profiles.unwrap_or(6), 6),
        "seed": params.seed.unwrap_or(42),
    });
    let result = ipc_client::call("science.gonzales.tissue_lattice", &rpc_params)
        .map_err(ipc_error)?;
    let (grammar, data) = super::grammar::tissue_lattice_grammar(&result);
    grammar_response("tissue_lattice", &grammar, &data, "health")
}

/// Grammar-rendered hormesis (SVG from petalTongue).
pub async fn grammar_hormesis(
    Query(params): Query<HormesisParams>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let rpc_params = json!({
        "amplitude": params.amplitude.unwrap_or(0.3),
        "k_stim": params.k_stim.unwrap_or(10.0),
        "n_stim": params.n_stim.unwrap_or(2.0),
        "k_inh": params.k_inh.unwrap_or(100.0),
        "n_inh": params.n_inh.unwrap_or(2.0),
        "n_points": clamp_u64(params.n_points.unwrap_or(100), MAX_POINTS),
        "dose_max": clamp_f64(params.dose_max.unwrap_or(200.0), MAX_DOSE),
    });
    let result = ipc_client::call("science.anderson.hormesis", &rpc_params)
        .map_err(ipc_error)?;
    let (grammar, data) = super::grammar::hormesis_grammar(&result);
    grammar_response("hormesis", &grammar, &data, "health")
}

/// Grammar-rendered cross-species (SVG from petalTongue).
pub async fn grammar_cross_species(
    Query(_params): Query<CrossSpeciesParams>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let result = ipc_client::call("science.anderson.cross_species", &json!({}))
        .map_err(ipc_error)?;
    let (grammar, data) = super::grammar::cross_species_grammar(&result);
    grammar_response("cross_species", &grammar, &data, "health")
}

fn grammar_response(
    id: &str,
    grammar: &Value,
    data: &[Value],
    domain: &str,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    match super::grammar::render_grammar(grammar, data, domain) {
        Some(render_result) => Ok(Json(json!({
            "id": id,
            "renderer": "petaltongue",
            "modality": render_result.get("modality").unwrap_or(&json!("svg")),
            "svg": render_result.get("output"),
            "scene_nodes": render_result.get("scene_nodes"),
            "total_primitives": render_result.get("total_primitives"),
            "tufte_report": render_result.get("tufte_report"),
            "grammar": grammar,
        }))),
        None => Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "error": "petaltongue_unavailable",
                "message": "petalTongue RPC not reachable — use Plotly.js renderer",
                "fallback": "plotly",
            })),
        )),
    }
}

/// Validation chain endpoint: returns the full paper-to-code-to-primal chain.
///
/// `GET /api/v1/validation/chain/:paper_id`
pub async fn validation_chain(
    axum::extract::Path(paper_id): axum::extract::Path<String>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    const REGISTRY: &str = include_str!("../../data/reference_registry.json");

    let registry: Value =
        serde_json::from_str(REGISTRY).unwrap_or(json!({"error": "registry parse failure"}));

    let paper = registry
        .get("papers")
        .and_then(|p| p.get(&paper_id))
        .ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(json!({
                    "error": "paper_not_found",
                    "paper_id": paper_id,
                    "available": registry
                        .get("papers")
                        .and_then(Value::as_object)
                        .map(|o| o.keys().collect::<Vec<_>>())
                        .unwrap_or_default(),
                })),
            )
        })?;

    let live_computation = if paper_id == "gonzales_2014" {
        let result = ipc_client::call("science.gonzales.dose_response", &json!({}))
            .map_err(ipc_error)?;

        let ic50s: Vec<Value> = result
            .get("curves")
            .and_then(Value::as_array)
            .map(|curves| {
                curves
                    .iter()
                    .filter_map(|c| {
                        Some(json!({
                            "pathway": c.get("pathway")?,
                            "ic50_nm": c.get("ic50_nm")?,
                        }))
                    })
                    .collect()
            })
            .unwrap_or_default();

        let prov = provenance::envelope(
            "science.gonzales.dose_response",
            &json!({}),
            &result,
        );

        Some(json!({
            "method": "science.gonzales.dose_response",
            "ic50_values": ic50s,
            "provenance": prov,
        }))
    } else {
        None
    };

    let published = paper.get("tables");
    let rust_validation = paper.get("rust_validation");
    let guidestone = paper.get("guidestone");
    let python_baseline = paper.get("python_baseline");

    let mut chain = json!({
        "paper_id": paper_id,
        "doi": paper.get("doi"),
        "title": paper.get("title"),
        "journal": paper.get("journal"),
        "year": paper.get("year"),
        "chain": {
            "source": {
                "status": "verified",
                "doi": paper.get("doi"),
                "tables": published,
            },
            "python_baseline": python_baseline.map(|p| json!({
                "status": if p.get("hash").is_some() && !p["hash"].is_null() { "hashed" } else { "pending_hash" },
                "path": p.get("path"),
                "hash": p.get("hash"),
            })),
            "rust_validation": rust_validation.map(|r| json!({
                "status": "verified",
                "binary": r.get("binary"),
                "checks": r.get("checks"),
                "result": r.get("status"),
            })),
            "guidestone": guidestone.map(|g| json!({
                "status": "verified",
                "binary": g.get("binary"),
                "checks": g.get("checks"),
                "result": g.get("status"),
            })),
            "nucleus_composition": live_computation.map(|lc| json!({
                "status": "live",
                "computation": lc,
            })),
        },
    });

    let prov = provenance::tier1("validation.chain", &json!({"paper_id": paper_id}), &chain);
    chain["provenance"] = prov;

    Ok(Json(chain))
}

/// System composition endpoint: returns deploy graph, primal versions,
/// capabilities, bonding metadata, graph validation, and circuit breaker
/// status for reproducibility and ionic discovery.
///
/// `GET /api/v1/system/composition`
pub async fn system_composition() -> Json<Value> {
    const DEPLOY_GRAPH: &str =
        include_str!("../../../graphs/wetspring_science_nucleus.toml");
    const REPRODUCTION_MANIFEST: &str =
        include_str!("../../data/reproduction_manifest.toml");
    const BONDING_METADATA_RAW: &str =
        include_str!("../../data/bonding_metadata.json");

    let bonding: Value = serde_json::from_str(BONDING_METADATA_RAW)
        .unwrap_or(json!({"error": "bonding metadata parse failure"}));

    let ipc_ok = ipc_client::call("health.check", &json!({})).is_ok();

    let capabilities_result = ipc_client::call("capability.list", &json!({}));
    let capabilities = capabilities_result
        .as_ref()
        .ok()
        .and_then(|v| v.get("capabilities").cloned())
        .unwrap_or(json!([]));

    let composition_health = ipc_client::call("composition.science_health", &json!({}))
        .unwrap_or(json!({"status": "unavailable"}));

    let graph_validation = super::graph_validate::validate_graph(DEPLOY_GRAPH).to_json();
    let breaker = provenance::breaker_status();

    let prov = provenance::tier1(
        "system.composition",
        &json!({}),
        &json!({"query": "composition"}),
    );

    Json(json!({
        "system": "wetspring_science_nucleus",
        "version": env!("CARGO_PKG_VERSION"),
        "deploy_graph": DEPLOY_GRAPH,
        "reproduction_manifest": REPRODUCTION_MANIFEST,
        "bonding": bonding,
        "status": {
            "facade": "online",
            "wetspring_ipc": if ipc_ok { "connected" } else { "unreachable" },
            "composition_health": composition_health,
            "graph_validation": graph_validation,
            "trio_circuit_breaker": breaker,
        },
        "capabilities": capabilities,
        "reproduction": {
            "fetch_command": "cd plasmidBin && ./fetch.sh --tag v0.7.0",
            "deploy_command": "biomeos deploy --graph graphs/wetspring_science_nucleus.toml",
            "plasmid_bin_url": "https://github.com/ecoPrimals/plasmidBin",
        },
        "provenance": prov,
    }))
}

fn ipc_error(msg: String) -> (StatusCode, Json<Value>) {
    (
        StatusCode::BAD_GATEWAY,
        Json(json!({
            "error": "ipc_error",
            "message": msg,
        })),
    )
}
