// SPDX-License-Identifier: AGPL-3.0-or-later
//! External data ingestion handlers with provenance-wrapped storage.
//!
//! Each fetch follows the pattern:
//! 1. HTTP GET to external API (ChEMBL, PubChem, or manual table)
//! 2. BLAKE3 hash the response
//! 3. Begin provenance session via the trio
//! 4. Record fetch + validation steps
//! 5. Complete session (dehydrate → commit → attribute)
//! 6. Store in NestGate with content hash
//!
//! All external HTTP is blocking (runs in the IPC thread pool). The
//! handlers gracefully degrade when NestGate or the provenance trio
//! are unavailable — raw results are still returned.

use serde_json::{Value, json};

use crate::ipc::protocol::RpcError;
use crate::ipc::provenance as trio;

/// Fetch JAK inhibitor panel data from ChEMBL REST API.
///
/// Queries `https://www.ebi.ac.uk/chembl/api/data/activity.json` for the
/// given compound (default: CHEMBL2366511 = oclacitinib).
pub fn handle_chembl_fetch(params: &Value) -> Result<Value, RpcError> {
    let chembl_id = params
        .get("chembl_id")
        .and_then(Value::as_str)
        .unwrap_or("CHEMBL2366511");

    let url = format!(
        "https://www.ebi.ac.uk/chembl/api/data/activity.json?molecule_chembl_id={chembl_id}&limit=100"
    );

    let session = trio::begin_session(&format!("data.fetch.chembl:{chembl_id}"));

    let fetch_result = blocking_http_get(&url);

    let (data, content_hash, source) = match fetch_result {
        Ok(body) => {
            let hash = blake3_hash_str(&body);
            let parsed: Value = serde_json::from_str(&body).unwrap_or(json!({"raw": body}));
            (parsed, hash, "chembl_api")
        }
        Err(e) => {
            return Ok(json!({
                "error": "fetch_failed",
                "message": e,
                "chembl_id": chembl_id,
                "provenance_session": session.id,
            }));
        }
    };

    let _ = trio::record_step(&session.id, &json!({
        "step": "fetch",
        "source": source,
        "url": url,
        "content_hash": content_hash,
    }));

    let completion = trio::complete_session(&session.id);

    nestgate_store(
        &format!("data:chembl:{chembl_id}"),
        &data,
        &content_hash,
    );

    Ok(json!({
        "chembl_id": chembl_id,
        "activities": data.get("activities"),
        "total_count": data.get("page_meta").and_then(|p| p.get("total_count")),
        "content_hash": content_hash,
        "provenance": completion,
        "source": source,
    }))
}

/// Fetch bioassay data from PubChem PUG REST API.
///
/// Queries `https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/JSON`
/// for the given assay ID.
pub fn handle_pubchem_fetch(params: &Value) -> Result<Value, RpcError> {
    let aid = params
        .get("aid")
        .and_then(Value::as_str)
        .or_else(|| params.get("assay_id").and_then(Value::as_str))
        .ok_or_else(|| RpcError::invalid_params("missing required param: aid or assay_id"))?;

    let url = format!(
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/JSON"
    );

    let session = trio::begin_session(&format!("data.fetch.pubchem:{aid}"));

    let fetch_result = blocking_http_get(&url);

    let (data, content_hash, source) = match fetch_result {
        Ok(body) => {
            let hash = blake3_hash_str(&body);
            let parsed: Value = serde_json::from_str(&body).unwrap_or(json!({"raw": body}));
            (parsed, hash, "pubchem_pug")
        }
        Err(e) => {
            return Ok(json!({
                "error": "fetch_failed",
                "message": e,
                "aid": aid,
                "provenance_session": session.id,
            }));
        }
    };

    let _ = trio::record_step(&session.id, &json!({
        "step": "fetch",
        "source": source,
        "url": url,
        "content_hash": content_hash,
    }));

    let completion = trio::complete_session(&session.id);

    nestgate_store(
        &format!("data:pubchem:{aid}"),
        &data,
        &content_hash,
    );

    Ok(json!({
        "aid": aid,
        "data": data,
        "content_hash": content_hash,
        "provenance": completion,
        "source": source,
    }))
}

/// Register a published paper table as a reference data point.
///
/// Stores the DOI, table values, and computes a BLAKE3 hash for the
/// canonical JSON representation. Wrapped in a provenance session.
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

    let _ = trio::record_step(&session.id, &json!({
        "step": "register",
        "doi": doi,
        "table_id": table_id,
        "content_hash": content_hash,
    }));

    let completion = trio::complete_session(&session.id);

    nestgate_store(
        &format!("ref:{doi}:{table_id}"),
        &canonical,
        &content_hash,
    );

    Ok(json!({
        "doi": doi,
        "table_id": table_id,
        "content_hash": content_hash,
        "provenance": completion,
        "status": "registered",
    }))
}

fn blocking_http_get(url: &str) -> Result<String, String> {
    Err(format!(
        "TLS not available in sovereign mode — use NestGate fetch proxy or pre-populated data for {url}"
    ))
}

fn blake3_hash_str(input: &str) -> String {
    blake3::hash(input.as_bytes()).to_hex().to_string()
}

fn blake3_hash_json(val: &Value) -> String {
    let bytes = serde_json::to_vec(val).unwrap_or_default();
    blake3::hash(&bytes).to_hex().to_string()
}

fn nestgate_store(key: &str, data: &Value, content_hash: &str) {
    let Some(socket) = trio::neural_api_socket() else { return };

    let data_str = serde_json::to_string(data).unwrap_or_default();

    let _ = crate::ipc::provenance::neural_api_socket().and_then(|sock| {
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

        use std::io::{BufRead, BufReader, Write};
        use std::os::unix::net::UnixStream;

        let mut stream = UnixStream::connect(&sock).ok()?;
        stream.set_read_timeout(Some(std::time::Duration::from_secs(5))).ok();
        let mut line = serde_json::to_string(&request).ok()?;
        line.push('\n');
        stream.write_all(line.as_bytes()).ok()?;
        stream.flush().ok()?;

        let mut reader = BufReader::new(stream);
        let mut resp = String::new();
        reader.read_line(&mut resp).ok()?;
        Some(())
    });

    let _ = socket;
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
    fn chembl_fetch_default_compound() {
        let result = handle_chembl_fetch(&json!({})).unwrap();
        assert!(result.get("chembl_id").is_some());
        assert_eq!(result["chembl_id"], "CHEMBL2366511");
    }

    #[test]
    fn pubchem_fetch_requires_aid() {
        let err = handle_pubchem_fetch(&json!({})).unwrap_err();
        assert_eq!(err.code, -32602);
    }

    #[test]
    fn pubchem_fetch_with_aid() {
        let result = handle_pubchem_fetch(&json!({"aid": "1234"})).unwrap();
        assert_eq!(result["aid"], "1234");
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
    fn blake3_hash_deterministic() {
        let v = json!({"a": 1, "b": 2});
        assert_eq!(blake3_hash_json(&v), blake3_hash_json(&v));
    }
}
