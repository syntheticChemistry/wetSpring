// SPDX-License-Identifier: AGPL-3.0-or-later
//! Progressive provenance metadata for facade responses.
//!
//! Three tiers:
//! - Tier 1 (always): guideStone version, content hash, timestamp, witnesses
//! - Tier 2 (trio reachable): rhizoCrypt DAG, loamSpine commit, sweetGrass braid
//! - Tier 3 (full RootPulse): PROV-O export, Merkle proof, verify links
//!
//! **Witness model** (trio v2, April 2026): provenance events are self-describing
//! `WireWitnessRef`-shaped JSON. Each witness carries `kind`, `encoding`,
//! `algorithm`, `tier`, and `context` — the trio is algo-agnostic and never
//! interprets evidence payloads.
//!
//! Circuit breaker: epoch-based breaker prevents hammering the trio when it's
//! partially reachable. After `BREAKER_THRESHOLD` consecutive failures the
//! breaker opens for `BREAKER_COOLDOWN_SECS`.
//!
//! NestGate integration: computation results are cached via `storage.store` with
//! family-scoped keys so repeated identical queries skip recomputation.

use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{Value, json};

const GUIDESTONE_VERSION: &str = "wetspring-gonzales-guideStone v0.1.0";
const GUIDESTONE_CHECKS: &str = "29/29 PASS";
const WETSPRING_VERSION: &str = env!("CARGO_PKG_VERSION");

const REPRODUCTION_MANIFEST: &str = include_str!("../../data/reproduction_manifest.toml");
const DEPLOY_GRAPH: &str = "wetspring_science_nucleus.toml";
const PLASMID_FETCH_TAG: &str = "v0.7.0";

// ── Circuit breaker (epoch-based, pattern from primalSpring) ──────────

const BREAKER_THRESHOLD: u64 = 3;
const BREAKER_COOLDOWN_SECS: u64 = 30;

/// Consecutive failure count for trio calls.
static TRIO_FAILURES: AtomicU64 = AtomicU64::new(0);
/// Unix timestamp when the breaker opened (0 = closed).
static TRIO_BREAKER_OPENED: AtomicU64 = AtomicU64::new(0);

fn trio_breaker_open() -> bool {
    let opened_at = TRIO_BREAKER_OPENED.load(Ordering::Relaxed);
    if opened_at == 0 {
        return false;
    }
    let now = now_epoch();
    if now.saturating_sub(opened_at) >= BREAKER_COOLDOWN_SECS {
        TRIO_BREAKER_OPENED.store(0, Ordering::Relaxed);
        TRIO_FAILURES.store(0, Ordering::Relaxed);
        false
    } else {
        true
    }
}

fn trio_record_success() {
    TRIO_FAILURES.store(0, Ordering::Relaxed);
    TRIO_BREAKER_OPENED.store(0, Ordering::Relaxed);
}

fn trio_record_failure() {
    let prev = TRIO_FAILURES.fetch_add(1, Ordering::Relaxed);
    if prev + 1 >= BREAKER_THRESHOLD {
        TRIO_BREAKER_OPENED.store(now_epoch(), Ordering::Relaxed);
    }
}

// ── WireWitnessRef builders ───────────────────────────────────────────

fn now_nanos() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

/// Build a hash observation witness (content hash of data at rest).
fn witness_hash(evidence_hex: &str, context: &str) -> Value {
    json!({
        "agent": format!("wetspring:pipeline:{WETSPRING_VERSION}"),
        "kind": "hash",
        "evidence": evidence_hex,
        "witnessed_at": now_nanos(),
        "encoding": "hex",
        "tier": "open",
        "context": context,
    })
}

/// Build a pipeline checkpoint witness (computation step marker).
fn witness_checkpoint(context: &str) -> Value {
    json!({
        "agent": format!("wetspring:pipeline:{WETSPRING_VERSION}"),
        "kind": "checkpoint",
        "evidence": "",
        "witnessed_at": now_nanos(),
        "encoding": "none",
        "tier": "open",
        "context": context,
    })
}

/// Build a BearDog signature witness from a `crypto.sign_ed25519` response.
///
/// Used when BearDog is live to sign dehydrated Merkle roots (Tier 2 step 3).
#[allow(dead_code)]
fn witness_signature(evidence_base64: &str, context: &str) -> Value {
    json!({
        "agent": "beardog:gate",
        "kind": "signature",
        "evidence": evidence_base64,
        "witnessed_at": now_nanos(),
        "encoding": "base64",
        "algorithm": "ed25519",
        "tier": "local",
        "context": context,
    })
}

/// Tier 1 provenance: always available, computed locally.
///
/// Includes a reproduction block and a `witnesses` array containing a hash
/// observation of the result (kind: "hash", encoding: "hex") and a
/// checkpoint for the computation step.
pub fn tier1(method: &str, params: &Value, result: &Value) -> Value {
    let content_hash = blake3_hash_json(result);
    let timestamp = now_iso8601();
    let endpoint = format!("/api/v1/science/{}", method.replace('.', "/"));

    let witnesses = vec![
        witness_hash(&content_hash, &format!("tier1:result_hash:{method}")),
        witness_checkpoint(&format!("tier1:computation:{method}")),
    ];

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
        "witnesses": witnesses,
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
/// Uses `capability.call` routing through Neural API:
/// 1. `dag` domain → `create_session` (rhizoCrypt)
/// 2. `dag` domain → `event.append` (record computation step)
/// 3. `dag` domain → `dehydrate` (Merkle root)
/// 4. `commit` domain → `session` (loamSpine permanent commit)
/// 5. `provenance` domain → `create_braid` (sweetGrass attribution)
///
/// Returns `None` if the trio is unreachable or the circuit breaker is open.
pub fn try_tier2(method: &str, params: &Value, result_hash: &str) -> Option<Value> {
    if trio_breaker_open() {
        return None;
    }

    let result = try_tier2_inner(method, params, result_hash);
    if result.is_some() {
        trio_record_success();
    } else {
        trio_record_failure();
    }
    result
}

fn try_tier2_inner(method: &str, params: &Value, result_hash: &str) -> Option<Value> {
    let neural_socket = neural_api_socket()?;

    // 1. rhizoCrypt: create DAG session
    let session = capability_call(
        &neural_socket,
        "dag",
        "create_session",
        &json!({
            "metadata": { "type": "science", "name": method },
            "session_type": { "Experiment": { "spring_id": "wetspring" } },
            "description": format!("wetspring:{method}"),
        }),
    )?;

    let session_id = session
        .get("session_id")
        .or_else(|| session.get("id"))
        .and_then(Value::as_str)?;

    // 2. rhizoCrypt: append computation event with witnesses
    let computation_witness = witness_hash(result_hash, &format!("tier2:computation:{method}"));
    let params_hash = blake3_hash_json(params);
    let ingest_witness = witness_hash(&params_hash, &format!("tier2:params_hash:{method}"));

    capability_call(
        &neural_socket,
        "dag",
        "event.append",
        &json!({
            "session_id": session_id,
            "event": {
                "method": method,
                "params_hash": params_hash,
                "result_hash": result_hash,
                "witnesses": [computation_witness, ingest_witness],
            },
        }),
    )?;

    // 3. rhizoCrypt: dehydrate → Merkle root
    let dehydration = capability_call(
        &neural_socket,
        "dag",
        "dehydrate",
        &json!({ "session_id": session_id }),
    )?;

    let merkle_root = dehydration
        .get("merkle_root")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned();

    // Collect witnesses from dehydration (trio v2: field is now `witnesses`)
    let dehydration_witnesses = dehydration
        .get("witnesses")
        .cloned()
        .unwrap_or(json!([]));

    // 4. loamSpine: permanent commit
    let commit_result = capability_call(
        &neural_socket,
        "commit",
        "session",
        &json!({
            "summary": dehydration,
            "content_hash": merkle_root,
        }),
    )?;

    let commit_id = commit_result
        .get("commit_id")
        .or_else(|| commit_result.get("entry_id"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned();

    // 5. sweetGrass: attribution braid
    let braid_result = capability_call(
        &neural_socket,
        "provenance",
        "create_braid",
        &json!({
            "commit_ref": commit_id,
            "agents": [{
                "did": "did:key:wetspring",
                "role": "computation",
                "contribution": 1.0,
            }],
        }),
    );

    let braid_id = braid_result
        .as_ref()
        .and_then(|r| r.get("braid_id").or_else(|| r.get("id")))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned();

    Some(json!({
        "tier": 2,
        "rhizocrypt_session": session_id,
        "loamspine_commit": commit_id,
        "sweetgrass_braid": braid_id,
        "merkle_root": merkle_root,
        "witnesses": dehydration_witnesses,
    }))
}

/// Attempt Tier 3: PROV-O export and Merkle inclusion proof.
///
/// Respects the circuit breaker — if the trio is already flagged as
/// unreachable the call returns `None` immediately.
pub fn try_tier3(braid_id: &str) -> Option<Value> {
    if trio_breaker_open() {
        return None;
    }

    let result = try_tier3_inner(braid_id);
    if result.is_some() {
        trio_record_success();
    } else {
        trio_record_failure();
    }
    result
}

fn try_tier3_inner(braid_id: &str) -> Option<Value> {
    let neural_socket = neural_api_socket()?;

    let provo = capability_call(
        &neural_socket,
        "provenance",
        "export_provo",
        &json!({ "braid_id": braid_id }),
    )?;

    Some(json!({
        "tier": 3,
        "prov_o": provo,
        "verify_url": format!("https://lab.primals.eco/api/v1/provenance/verify/{braid_id}"),
    }))
}

/// Circuit breaker status for diagnostic endpoints.
pub fn breaker_status() -> Value {
    let failures = TRIO_FAILURES.load(Ordering::Relaxed);
    let opened_at = TRIO_BREAKER_OPENED.load(Ordering::Relaxed);
    let is_open = trio_breaker_open();

    json!({
        "open": is_open,
        "consecutive_failures": failures,
        "threshold": BREAKER_THRESHOLD,
        "cooldown_secs": BREAKER_COOLDOWN_SECS,
        "opened_at_epoch": opened_at,
    })
}

/// Structure a computation result as a Novel Ferment Transcript vertex.
///
/// Each API response becomes a DAG vertex in the gAIa commons. Includes
/// a checkpoint witness for the vertex creation event.
pub fn ferment_vertex(
    method: &str,
    params: &Value,
    result_hash: &str,
    parent_vertices: &[&str],
) -> Value {
    let timestamp = now_iso8601();

    let vertex_input = format!("{method}:{params}:{result_hash}:{timestamp}");
    let vertex_id = blake3::hash(vertex_input.as_bytes()).to_hex().to_string();

    let vertex_witness = witness_checkpoint(
        &format!("nft:vertex_created:{vertex_id}")
    );

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
            "witnesses": [vertex_witness],
        },
    })
}

/// Build a provenance envelope combining all reachable tiers + NFT vertex.
///
/// Collects witnesses from all tiers into a unified `witnesses` array.
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
        .filter(|s| !s.is_empty())
        .and_then(try_tier3);

    let vertex = ferment_vertex(method, params, &content_hash, &[]);

    let mut all_witnesses: Vec<Value> = t1["witnesses"]
        .as_array()
        .cloned()
        .unwrap_or_default();

    if let Some(ref tier2) = t2 {
        if let Some(t2w) = tier2["witnesses"].as_array() {
            all_witnesses.extend(t2w.iter().cloned());
        }
    }

    if let Some(nft_w) = vertex["nft_vertex"]["witnesses"].as_array() {
        all_witnesses.extend(nft_w.iter().cloned());
    }

    let mut prov = t1;
    if let Some(tier2) = t2 {
        prov["trio"] = json!({
            "rhizocrypt_session": tier2["rhizocrypt_session"],
            "loamspine_commit": tier2["loamspine_commit"],
            "sweetgrass_braid": tier2["sweetgrass_braid"],
            "merkle_root": tier2["merkle_root"],
        });
        prov["tier"] = json!(2);
    }
    if let Some(tier3) = t3 {
        prov["tier3"] = tier3;
        prov["tier"] = json!(3);
    }
    prov["witnesses"] = json!(all_witnesses);
    prov["nft_vertex"] = vertex["nft_vertex"].clone();
    prov
}

fn blake3_hash_json(val: &Value) -> String {
    let bytes = serde_json::to_vec(val).unwrap_or_default();
    let hash = blake3::hash(&bytes);
    hash.to_hex().to_string()
}

fn now_epoch() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn now_iso8601() -> String {
    format!("{}", now_epoch())
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

/// Route a call through Neural API `capability.call` with domain + operation.
///
/// This is the canonical routing pattern from `SPRING_PROVENANCE_PATTERN.md`:
/// the domain maps to a primal's `by_capability` and the Neural API discovers
/// the provider and dispatches.
fn capability_call(
    socket_path: &std::path::Path,
    domain: &str,
    operation: &str,
    args: &Value,
) -> Option<Value> {
    call_neural(socket_path, "capability.call", &json!({
        "capability": domain,
        "operation": operation,
        "args": args,
    }))
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
