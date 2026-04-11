// SPDX-License-Identifier: AGPL-3.0-or-later
//! External data ingestion — pure primal composition, no fallbacks.
//!
//! wetSpring is a science consumer. All infrastructure is primal composition:
//!
//! - **NestGate** (Nest atomic): TLS, content-addressed storage, caching
//! - **BearDog** (Tower atomic): authorization, consent tokens
//! - **Provenance Trio**: session tracking, Merkle roots, semantic braids
//! - **biomeOS**: capability routing between primals
//!
//! If a primal is not running, the fetch **fails with a gap report** that
//! identifies the missing capability. Gaps are handed to primalSpring for
//! evolution — springs never work around missing primals.

use serde_json::{Value, json};

use crate::ipc::protocol::RpcError;
use crate::ipc::provenance as trio;

// ── Public handlers ─────────────────────────────────────────────────────

/// Fetch JAK inhibitor panel data from ChEMBL via primal composition.
///
/// biomeOS routes to NestGate which handles TLS to `ebi.ac.uk`, content-
/// addresses, caches, and returns the payload. wetSpring never opens a
/// network connection.
pub fn handle_chembl_fetch(params: &Value) -> Result<Value, RpcError> {
    let chembl_id = params
        .get("chembl_id")
        .and_then(Value::as_str)
        .unwrap_or("CHEMBL2103874");

    let url = format!(
        "https://www.ebi.ac.uk/chembl/api/data/activity.json?molecule_chembl_id={chembl_id}&limit=100"
    );
    let cache_key = format!("data:chembl:{chembl_id}");

    let session = trio::begin_session(&format!("data.fetch.chembl:{chembl_id}"));

    match fetch_via_composition(&url, &cache_key) {
        Ok((data, content_hash, tier)) => {
            let _ = trio::record_step(
                &session.id,
                &json!({
                    "step": "fetch",
                    "source": tier,
                    "url": url,
                    "content_hash": content_hash,
                }),
            );
            let completion = trio::complete_session(&session.id);

            Ok(json!({
                "chembl_id": chembl_id,
                "data": data,
                "content_hash": content_hash,
                "provenance": completion,
                "source": tier,
            }))
        }
        Err(gap) => {
            let _ = trio::record_step(
                &session.id,
                &json!({
                    "step": "gap_detected",
                    "gaps": gap.missing,
                }),
            );
            let completion = trio::complete_session(&session.id);

            Ok(json!({
                "gap_report": true,
                "chembl_id": chembl_id,
                "url": url,
                "missing_primals": gap.missing,
                "action": "hand to primalSpring for primal evolution",
                "provenance": completion,
                "provenance_session": session.id,
            }))
        }
    }
}

/// Fetch compound data from PubChem via primal composition.
///
/// biomeOS routes to NestGate which handles TLS to
/// `pubchem.ncbi.nlm.nih.gov`. wetSpring never opens a network connection.
pub fn handle_pubchem_fetch(params: &Value) -> Result<Value, RpcError> {
    let aid = params
        .get("aid")
        .and_then(Value::as_str)
        .or_else(|| params.get("assay_id").and_then(Value::as_str))
        .ok_or_else(|| RpcError::invalid_params("missing required param: aid or assay_id"))?;

    let url = format!("https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/JSON");
    let cache_key = format!("data:pubchem:{aid}");

    let session = trio::begin_session(&format!("data.fetch.pubchem:{aid}"));

    match fetch_via_composition(&url, &cache_key) {
        Ok((data, content_hash, tier)) => {
            let _ = trio::record_step(
                &session.id,
                &json!({
                    "step": "fetch",
                    "source": tier,
                    "url": url,
                    "content_hash": content_hash,
                }),
            );
            let completion = trio::complete_session(&session.id);

            Ok(json!({
                "aid": aid,
                "data": data,
                "content_hash": content_hash,
                "provenance": completion,
                "source": tier,
            }))
        }
        Err(gap) => {
            let _ = trio::record_step(
                &session.id,
                &json!({
                    "step": "gap_detected",
                    "gaps": gap.missing,
                }),
            );
            let completion = trio::complete_session(&session.id);

            Ok(json!({
                "gap_report": true,
                "aid": aid,
                "url": url,
                "missing_primals": gap.missing,
                "action": "hand to primalSpring for primal evolution",
                "provenance": completion,
                "provenance_session": session.id,
            }))
        }
    }
}

/// Register a published paper table as a reference data point.
///
/// BLAKE3-hashes the canonical JSON representation (science computation),
/// stores via NestGate (primal composition), and wraps in a provenance
/// session.
pub fn handle_register_table(params: &Value) -> Result<Value, RpcError> {
    let doi = params
        .get("doi")
        .and_then(Value::as_str)
        .ok_or_else(|| RpcError::invalid_params("missing required param: doi"))?;

    let table_id = params
        .get("table_id")
        .and_then(Value::as_str)
        .unwrap_or("table_1");

    let values = params
        .get("values")
        .ok_or_else(|| RpcError::invalid_params("missing required param: values"))?;

    let session = trio::begin_session(&format!("data.register:{doi}:{table_id}"));

    let canonical = json!({
        "doi": doi,
        "table_id": table_id,
        "values": values,
    });
    let content_hash = blake3_hash_json(&canonical);

    let _ = trio::record_step(
        &session.id,
        &json!({
            "step": "register",
            "doi": doi,
            "table_id": table_id,
            "content_hash": content_hash,
        }),
    );

    let completion = trio::complete_session(&session.id);

    nestgate_store(&format!("ref:{doi}:{table_id}"), &canonical, &content_hash);

    Ok(json!({
        "doi": doi,
        "table_id": table_id,
        "content_hash": content_hash,
        "provenance": completion,
        "status": "registered",
    }))
}

// ── Primal composition routing ──────────────────────────────────────────

/// Structured report of which primals are missing for a given operation.
struct GapReport {
    missing: Vec<Value>,
}

/// Attempt to fetch external data via primal composition only.
///
/// Tier 1: biomeOS `capability.call("storage", "fetch_external")` — full
///         composition routing through NestGate.
/// Tier 2: NestGate direct `storage.retrieve` — cache hit without biomeOS.
///
/// No fallbacks. Missing primals produce a [`GapReport`].
fn fetch_via_composition(
    url: &str,
    cache_key: &str,
) -> Result<(Value, String, &'static str), GapReport> {
    let mut gaps: Vec<Value> = Vec::new();

    let Some(socket) = trio::neural_api_socket() else {
        gaps.push(json!({
            "primal": "biomeOS",
            "capability": "Neural API socket",
            "required_for": "capability.call routing",
            "deploy": "start biomeOS orchestrator",
        }));
        gaps.push(json!({
            "primal": "NestGate",
            "capability": "storage.fetch_external",
            "required_for": "TLS fetch + content-addressed caching",
            "deploy": "start NestGate with fetch_external capability",
        }));
        return Err(GapReport { missing: gaps });
    };

    // Tier 1: biomeOS → NestGate fetch_external (NestGate owns TLS)
    match fetch_external_via_biomeos(&socket, url, cache_key) {
        Some((data, hash)) => return Ok((data, hash, "nestgate_via_biomeos")),
        None => {
            gaps.push(json!({
                "primal": "NestGate",
                "capability": "storage.fetch_external",
                "required_for": "TLS fetch of external URL",
                "url": url,
                "note": "biomeOS socket found but fetch_external failed — NestGate may not be registered or URL unreachable",
            }));
        }
    }

    // Tier 2: NestGate direct cache retrieve
    match nestgate_cache_retrieve(&socket, cache_key) {
        Some((data, hash)) => return Ok((data, hash, "nestgate_cache")),
        None => {
            gaps.push(json!({
                "primal": "NestGate",
                "capability": "storage.retrieve",
                "required_for": "cached data retrieval",
                "cache_key": cache_key,
                "note": "no cached data — NestGate needs to fetch_external first",
            }));
        }
    }

    Err(GapReport { missing: gaps })
}

/// Tier 1: biomeOS → NestGate `fetch_external`. NestGate handles TLS.
fn fetch_external_via_biomeos(
    socket: &std::path::Path,
    url: &str,
    cache_key: &str,
) -> Option<(Value, String)> {
    let request = json!({
        "jsonrpc": "2.0",
        "method": "capability.call",
        "params": {
            "capability": "storage",
            "operation": "fetch_external",
            "args": {
                "url": url,
                "cache_key": cache_key,
                "content_address": true,
            },
        },
        "id": 1,
    });

    let resp = send_rpc(socket, &request)?;
    let result = resp.get("result")?;
    if result.get("error").is_some() {
        return None;
    }
    let data_str = result.get("data").and_then(Value::as_str)?;
    let hash = result
        .get("content_hash")
        .and_then(Value::as_str)
        .map_or_else(|| blake3_hash_str(data_str), String::from);
    let parsed: Value = serde_json::from_str(data_str).ok()?;
    Some((parsed, hash))
}

/// Tier 2: NestGate `storage.retrieve` for cached data.
fn nestgate_cache_retrieve(socket: &std::path::Path, cache_key: &str) -> Option<(Value, String)> {
    let request = json!({
        "jsonrpc": "2.0",
        "method": "capability.call",
        "params": {
            "capability": "storage",
            "operation": "retrieve",
            "args": { "key": cache_key },
        },
        "id": 1,
    });

    let resp = send_rpc(socket, &request)?;
    let result = resp.get("result")?;
    let data_str = result.get("data").and_then(Value::as_str)?;
    if data_str == "not_found" || data_str.is_empty() {
        return None;
    }
    let hash = blake3_hash_str(data_str);
    let parsed: Value = serde_json::from_str(data_str).ok()?;
    Some((parsed, hash))
}

// ── Transport + hashing (not infrastructure — just primal IPC wire) ─────

/// Send a JSON-RPC request to a primal Unix socket.
fn send_rpc(socket: &std::path::Path, request: &Value) -> Option<Value> {
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::UnixStream;

    let mut stream = UnixStream::connect(socket).ok()?;
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(10)))
        .ok();
    let mut line = serde_json::to_string(request).ok()?;
    line.push('\n');
    stream.write_all(line.as_bytes()).ok()?;
    stream.flush().ok()?;

    let mut reader = BufReader::new(stream);
    let mut resp = String::new();
    reader.read_line(&mut resp).ok()?;
    serde_json::from_str(&resp).ok()
}

fn blake3_hash_str(input: &str) -> String {
    blake3::hash(input.as_bytes()).to_hex().to_string()
}

fn blake3_hash_json(val: &Value) -> String {
    let bytes = serde_json::to_vec(val).unwrap_or_default();
    blake3::hash(&bytes).to_hex().to_string()
}

/// Best-effort store via NestGate `capability.call("storage", "store")`.
fn nestgate_store(key: &str, data: &Value, content_hash: &str) {
    let Some(socket) = trio::neural_api_socket() else {
        return;
    };

    let data_str = serde_json::to_string(data).unwrap_or_default();
    let request = json!({
        "jsonrpc": "2.0",
        "method": "capability.call",
        "params": {
            "capability": "storage",
            "operation": "store",
            "args": {
                "key": key,
                "data": data_str,
                "content_hash": content_hash,
            },
        },
        "id": 1,
    });

    let _ = send_rpc(&socket, &request);
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "test module: assertions use unwrap for clarity"
)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn chembl_fetch_gap_report_when_no_primals() {
        let result = handle_chembl_fetch(&json!({})).unwrap();
        assert_eq!(result["chembl_id"], "CHEMBL2103874");
        assert_eq!(result["gap_report"], true);
        let missing = result["missing_primals"].as_array().unwrap();
        assert!(
            !missing.is_empty(),
            "must report at least one missing primal"
        );
        let primal_names: Vec<&str> = missing
            .iter()
            .filter_map(|g| g["primal"].as_str())
            .collect();
        assert!(
            primal_names.contains(&"biomeOS") || primal_names.contains(&"NestGate"),
            "gap must name the missing primal, got: {primal_names:?}"
        );
    }

    #[test]
    fn pubchem_fetch_gap_report_when_no_primals() {
        let result = handle_pubchem_fetch(&json!({"aid": "1234"})).unwrap();
        assert_eq!(result["aid"], "1234");
        assert_eq!(result["gap_report"], true);
        assert!(!result["missing_primals"].as_array().unwrap().is_empty());
    }

    #[test]
    fn pubchem_fetch_requires_aid() {
        let err = handle_pubchem_fetch(&json!({})).unwrap_err();
        assert_eq!(err.code, -32602);
    }

    #[test]
    fn register_table_requires_doi() {
        let err = handle_register_table(&json!({"values": [1, 2]})).unwrap_err();
        assert_eq!(err.code, -32602);
    }

    #[test]
    fn register_table_requires_values() {
        let err = handle_register_table(&json!({"doi": "10.1111/test"})).unwrap_err();
        assert_eq!(err.code, -32602);
    }

    #[test]
    fn register_table_success() {
        let result = handle_register_table(&json!({
            "doi": "10.1111/jvp.12065",
            "table_id": "table_1",
            "values": {
                "JAK1": 10.0,
                "IL-31": 71.0,
            },
        }))
        .unwrap();
        assert_eq!(result["status"], "registered");
        assert!(result["content_hash"].as_str().is_some());
    }

    #[test]
    fn gap_report_includes_action() {
        let result = handle_chembl_fetch(&json!({})).unwrap();
        if result["gap_report"] == true {
            assert!(
                result["action"].as_str().unwrap().contains("primalSpring"),
                "gap action must reference primalSpring"
            );
        }
    }

    #[test]
    fn blake3_hash_deterministic() {
        let v = json!({"a": 1, "b": 2});
        assert_eq!(blake3_hash_json(&v), blake3_hash_json(&v));
    }
}
