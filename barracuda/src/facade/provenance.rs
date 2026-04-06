// SPDX-License-Identifier: AGPL-3.0-or-later
//! Progressive provenance metadata for facade responses.
//!
//! Three tiers:
//! - Tier 1 (always): guideStone version, commit hash, content hash, timestamp
//! - Tier 2 (trio reachable): rhizoCrypt session, loamSpine commit, sweetGrass braid
//! - Tier 3 (full RootPulse): PROV-O export, Merkle inclusion proof, verify links
//!
//! NestGate integration: computation results are cached via `storage.store` with
//! family-scoped keys so repeated identical queries skip recomputation.

use serde_json::{Value, json};

const GUIDESTONE_VERSION: &str = "wetspring-gonzales-guideStone v0.1.0";
const GUIDESTONE_CHECKS: &str = "29/29 PASS";
const WETSPRING_VERSION: &str = env!("CARGO_PKG_VERSION");

const REPRODUCTION_MANIFEST: &str = include_str!("../../data/reproduction_manifest.toml");
const DEPLOY_GRAPH: &str = "wetspring_science_nucleus.toml";
const PLASMID_FETCH_TAG: &str = "v0.7.0";

/// Tier 1 provenance: always available, computed locally.
///
/// Includes a reproduction block so any result can be recreated locally
/// by fetching the pinned primal versions and deploying the same graph.
pub fn tier1(method: &str, params: &Value, result: &Value) -> Value {
    let content_hash = blake3_hash_json(result);
    let timestamp = now_iso8601();
    let endpoint = format!("/api/v1/science/{}", method.replace('.', "/"));
    json!({
        "tier": 1,
        "guidestone": {
            "version": GUIDESTONE_VERSION,
            "validation": GUIDESTONE_CHECKS,
        },
        "wetspring": {
            "version": WETSPRING_VERSION,
            "commit": option_env!("GIT_COMMIT_HASH").unwrap_or("dev"),
        },
        "computation": {
            "method": method,
            "params": params,
            "timestamp": timestamp,
            "content_hash": content_hash,
        },
        "reproduction": {
            "plasmid_manifest": REPRODUCTION_MANIFEST,
            "deploy_graph": DEPLOY_GRAPH,
            "fetch_command": format!("cd plasmidBin && ./fetch.sh --tag {PLASMID_FETCH_TAG}"),
            "deploy_command": format!("biomeos deploy --graph graphs/{DEPLOY_GRAPH}"),
            "recompute": {
                "method": method,
                "params": params,
                "endpoint": endpoint,
            },
        },
    })
}

/// Attempt Tier 2 provenance via the provenance trio.
///
/// Calls `provenance.begin`, `provenance.record`, `provenance.complete` via
/// the Neural API socket. Returns `None` if the trio is unreachable.
pub fn try_tier2(method: &str, params: &Value, result_hash: &str) -> Option<Value> {
    let neural_socket = neural_api_socket()?;

    let session = call_neural(&neural_socket, "provenance.begin", &json!({
        "source": "science_facade",
        "method": method,
    }))?;

    let session_id = session.get("session_id")?.as_str()?;

    call_neural(&neural_socket, "provenance.record", &json!({
        "session_id": session_id,
        "step": {
            "method": method,
            "params": params,
            "result_hash": result_hash,
        },
    }))?;

    let completion = call_neural(&neural_socket, "provenance.complete", &json!({
        "session_id": session_id,
    }))?;

    Some(json!({
        "tier": 2,
        "rhizocrypt_session": session_id,
        "loamspine_commit": completion.get("commit_id"),
        "sweetgrass_braid": completion.get("braid_id"),
        "merkle_root": completion.get("merkle_root"),
    }))
}

/// Attempt Tier 3: PROV-O export and Merkle inclusion proof.
pub fn try_tier3(braid_id: &str) -> Option<Value> {
    let neural_socket = neural_api_socket()?;

    let provo = call_neural(&neural_socket, "capability.call", &json!({
        "capability": "provenance",
        "operation": "export_provo",
        "args": { "braid_id": braid_id },
    }))?;

    Some(json!({
        "tier": 3,
        "prov_o": provo,
        "verify_url": format!("https://lab.primals.eco/api/v1/provenance/verify/{braid_id}"),
    }))
}

/// Structure a computation result as a Novel Ferment Transcript vertex.
///
/// Each API response becomes a DAG vertex in the gAIa commons:
/// - `vertex_id`: BLAKE3 of (method, params, result_hash, timestamp)
/// - `parent_vertices`: input data sources (DOI hashes, fetch hashes)
/// - `agents`: who/what contributed (wetSpring, paper authors, hardware)
/// - `derivation`: structural link to source data (Paper 20 Section 5)
/// - `license`: scyBorg triple (AGPL-3.0 / ORC / CC-BY-SA 4.0)
pub fn ferment_vertex(
    method: &str,
    params: &Value,
    result_hash: &str,
    parent_vertices: &[&str],
) -> Value {
    let timestamp = now_iso8601();

    let vertex_input = format!("{method}:{params}:{result_hash}:{timestamp}");
    let vertex_id = blake3::hash(vertex_input.as_bytes()).to_hex().to_string();

    json!({
        "nft_vertex": {
            "vertex_id": vertex_id,
            "parent_vertices": parent_vertices,
            "timestamp": timestamp,
            "agents": [
                {
                    "type": "primal",
                    "id": "wetSpring",
                    "version": WETSPRING_VERSION,
                },
                {
                    "type": "computation",
                    "id": method,
                    "substrate": option_env!("COMPUTE_SUBSTRATE").unwrap_or("cpu"),
                },
            ],
            "derivation": {
                "method": method,
                "params_hash": blake3_hash_json(params),
                "result_hash": result_hash,
                "deploy_graph": DEPLOY_GRAPH,
            },
            "license": {
                "code": "AGPL-3.0-or-later",
                "data_model": "ORC",
                "content": "CC-BY-SA-4.0",
            },
            "gaia_context": {
                "commons_eligible": true,
                "attribution_chain": true,
                "description": "Novel Ferment Transcript vertex — value from verifiable history, not scarcity",
            },
        },
    })
}

/// Build a provenance envelope combining all reachable tiers + NFT vertex.
pub fn envelope(method: &str, params: &Value, result: &Value) -> Value {
    let t1 = tier1(method, params, result);
    let content_hash = t1["computation"]["content_hash"]
        .as_str()
        .unwrap_or("")
        .to_string();

    let t2 = try_tier2(method, params, &content_hash);
    let t3 = t2
        .as_ref()
        .and_then(|v| v["sweetgrass_braid"].as_str())
        .and_then(try_tier3);

    let vertex = ferment_vertex(method, params, &content_hash, &[]);

    let mut prov = t1;
    if let Some(tier2) = t2 {
        prov["tier2"] = tier2;
        prov["tier"] = json!(2);
    }
    if let Some(tier3) = t3 {
        prov["tier3"] = tier3;
        prov["tier"] = json!(3);
    }
    prov["nft_vertex"] = vertex["nft_vertex"].clone();
    prov
}

fn blake3_hash_json(val: &Value) -> String {
    let bytes = serde_json::to_vec(val).unwrap_or_default();
    let hash = blake3::hash(&bytes);
    hash.to_hex().to_string()
}

fn now_iso8601() -> String {
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs();
    format!("{secs}")
}

fn neural_api_socket() -> Option<std::path::PathBuf> {
    let family_id = std::env::var("FAMILY_ID").ok()?;
    let runtime = std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/tmp".into());
    let path = std::path::PathBuf::from(runtime)
        .join("biomeos")
        .join(format!("neural-api-{family_id}.sock"));
    if path.exists() {
        Some(path)
    } else {
        None
    }
}

fn call_neural(socket_path: &std::path::Path, method: &str, params: &Value) -> Option<Value> {
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::UnixStream;

    let mut stream = UnixStream::connect(socket_path).ok()?;
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(5)))
        .ok();

    let request = json!({
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
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

// ── NestGate storage integration ──────────────────────────────────────

/// Build a NestGate key for caching computation results.
///
/// Format: `science:{method}:{param_hash}` where param_hash is a truncated
/// BLAKE3 of the canonical JSON params.
pub fn nestgate_key(method: &str, params: &Value) -> String {
    let hash = blake3_hash_json(params);
    let short = &hash[..16];
    format!("science:{method}:{short}")
}

/// Try to retrieve a cached result from NestGate.
pub fn nestgate_get(method: &str, params: &Value) -> Option<Value> {
    let socket = neural_api_socket()?;
    let key = nestgate_key(method, params);

    let result = call_neural(&socket, "capability.call", &json!({
        "capability": "storage",
        "operation": "retrieve",
        "args": { "key": key },
    }))?;

    let data_str = result.get("data").and_then(Value::as_str)?;
    serde_json::from_str(data_str).ok()
}

/// Store a computation result in NestGate for caching.
pub fn nestgate_put(method: &str, params: &Value, result: &Value) {
    let Some(socket) = neural_api_socket() else { return };
    let key = nestgate_key(method, params);
    let data = serde_json::to_string(result).unwrap_or_default();
    let content_hash = blake3_hash_json(result);

    let _ = call_neural(&socket, "capability.call", &json!({
        "capability": "storage",
        "operation": "store",
        "args": {
            "key": key,
            "data": data,
            "content_hash": content_hash,
            "metadata": {
                "method": method,
                "params": params,
                "stored_at": now_iso8601(),
            },
        },
    }));
}

/// Check if a NestGate key exists.
pub fn nestgate_exists(key: &str) -> bool {
    let Some(socket) = neural_api_socket() else { return false };
    call_neural(&socket, "capability.call", &json!({
        "capability": "storage",
        "operation": "exists",
        "args": { "key": key },
    }))
    .and_then(|r| r.get("exists").and_then(Value::as_bool))
    .unwrap_or(false)
}
