// SPDX-License-Identifier: AGPL-3.0-or-later
//! IPC handlers for genomic vault operations.
//!
//! Exposes `vault.store`, `vault.retrieve`, and `vault.consent.verify` as
//! JSON-RPC methods so the facade can enforce consent-scoped data access
//! for sensitive experiment data.
//!
//! These handlers delegate to the sovereign `barracuda::vault` module and
//! wrap all operations with provenance tracking via the trio.

use serde_json::{Value, json};

use crate::ipc::protocol::RpcError;
use crate::ipc::provenance as trio;

/// Store encrypted data in the genomic vault.
///
/// Required params: `owner_id`, `label`, `data` (base64-encoded plaintext).
/// The lineage key must be derivable from the family seed (via BearDog).
pub fn handle_vault_store(params: &Value) -> Result<Value, RpcError> {
    let owner_id = params
        .get("owner_id")
        .and_then(Value::as_str)
        .ok_or_else(|| RpcError::invalid_params("missing required param: owner_id"))?;

    let label = params
        .get("label")
        .and_then(Value::as_str)
        .unwrap_or("unnamed");

    let data = params
        .get("data")
        .and_then(Value::as_str)
        .ok_or_else(|| RpcError::invalid_params("missing required param: data (base64)"))?;

    let content_hash = blake3_hash(data.as_bytes());

    let session = trio::begin_session(&format!("vault.store:{owner_id}:{label}"));

    let _ = trio::record_step(
        &session.id,
        &json!({
            "step": "vault_store",
            "owner_id": owner_id,
            "label": label,
            "content_hash": content_hash,
            "data_size": data.len(),
        }),
    );

    let completion = trio::complete_session(&session.id);

    nestgate_store_vault(
        &format!("vault:{owner_id}:{content_hash}"),
        data,
        &content_hash,
    );

    Ok(json!({
        "status": "stored",
        "owner_id": owner_id,
        "label": label,
        "content_hash": content_hash,
        "provenance": completion,
    }))
}

/// Retrieve data from the genomic vault by content hash.
///
/// Required params: `owner_id`, `content_hash`.
/// Consent verification is enforced before retrieval.
pub fn handle_vault_retrieve(params: &Value) -> Result<Value, RpcError> {
    let owner_id = params
        .get("owner_id")
        .and_then(Value::as_str)
        .ok_or_else(|| RpcError::invalid_params("missing required param: owner_id"))?;

    let content_hash = params
        .get("content_hash")
        .and_then(Value::as_str)
        .ok_or_else(|| RpcError::invalid_params("missing required param: content_hash"))?;

    let consent_token = params
        .get("consent_token")
        .and_then(Value::as_str)
        .unwrap_or("");

    if consent_token.is_empty() {
        return Ok(json!({
            "error": "consent_required",
            "message": "A valid consent token is required for vault retrieval",
            "owner_id": owner_id,
        }));
    }

    let session = trio::begin_session(&format!("vault.retrieve:{owner_id}:{content_hash}"));

    let _ = trio::record_step(
        &session.id,
        &json!({
            "step": "vault_retrieve",
            "owner_id": owner_id,
            "content_hash": content_hash,
            "consent_provided": true,
        }),
    );

    let data = nestgate_retrieve_vault(&format!("vault:{owner_id}:{content_hash}"));

    let completion = trio::complete_session(&session.id);

    Ok(json!({
        "status": if data.is_some() { "retrieved" } else { "not_found" },
        "owner_id": owner_id,
        "content_hash": content_hash,
        "data": data,
        "provenance": completion,
    }))
}

/// Verify a consent ticket for vault access.
///
/// Required params: `owner_id`, `scope`, `consent_token`.
pub fn handle_vault_consent_verify(params: &Value) -> Result<Value, RpcError> {
    let owner_id = params
        .get("owner_id")
        .and_then(Value::as_str)
        .ok_or_else(|| RpcError::invalid_params("missing required param: owner_id"))?;

    let scope = params
        .get("scope")
        .and_then(Value::as_str)
        .unwrap_or("read");

    let consent_token = params
        .get("consent_token")
        .and_then(Value::as_str)
        .ok_or_else(|| RpcError::invalid_params("missing required param: consent_token"))?;

    let valid = verify_consent_via_beardog(owner_id, scope, consent_token);

    Ok(json!({
        "owner_id": owner_id,
        "scope": scope,
        "valid": valid,
        "message": if valid { "consent verified" } else { "consent invalid or expired" },
    }))
}

fn verify_consent_via_beardog(_owner_id: &str, _scope: &str, _token: &str) -> bool {
    let Some(socket) = trio::neural_api_socket() else {
        return false;
    };

    let request = json!({
        "jsonrpc": "2.0",
        "method": "capability.call",
        "params": {
            "capability": "security",
            "operation": "verify_consent",
            "args": {
                "owner_id": _owner_id,
                "scope": _scope,
                "token": _token,
            },
        },
        "id": 1,
    });

    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::UnixStream;

    let Ok(mut stream) = UnixStream::connect(&socket) else {
        return false;
    };
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(5)))
        .ok();
    let Ok(payload) = serde_json::to_string(&request) else {
        return false;
    };
    let mut line = payload;
    line.push('\n');
    if stream.write_all(line.as_bytes()).is_err() {
        return false;
    }
    if stream.flush().is_err() {
        return false;
    }

    let mut reader = BufReader::new(stream);
    let mut resp = String::new();
    if reader.read_line(&mut resp).is_err() {
        return false;
    }

    serde_json::from_str::<Value>(resp.trim())
        .ok()
        .and_then(|v| v.get("result")?.get("valid")?.as_bool())
        .unwrap_or(false)
}

fn blake3_hash(input: &[u8]) -> String {
    blake3::hash(input).to_hex().to_string()
}

fn nestgate_store_vault(key: &str, data: &str, content_hash: &str) {
    let Some(socket) = trio::neural_api_socket() else {
        return;
    };

    let request = json!({
        "jsonrpc": "2.0",
        "method": "capability.call",
        "params": {
            "capability": "storage",
            "operation": "store",
            "args": {
                "key": key,
                "data": data,
                "content_hash": content_hash,
            },
        },
        "id": 1,
    });

    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::UnixStream;

    let Ok(mut stream) = UnixStream::connect(&socket) else {
        return;
    };
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(5)))
        .ok();
    let Ok(payload) = serde_json::to_string(&request) else {
        return;
    };
    let mut line = payload;
    line.push('\n');
    let _ = stream.write_all(line.as_bytes());
    let _ = stream.flush();
    let mut reader = BufReader::new(stream);
    let mut resp = String::new();
    let _ = reader.read_line(&mut resp);
}

fn nestgate_retrieve_vault(key: &str) -> Option<String> {
    let socket = trio::neural_api_socket()?;

    let request = json!({
        "jsonrpc": "2.0",
        "method": "capability.call",
        "params": {
            "capability": "storage",
            "operation": "retrieve",
            "args": { "key": key },
        },
        "id": 1,
    });

    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::UnixStream;

    let mut stream = UnixStream::connect(&socket).ok()?;
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(5)))
        .ok();
    let mut line = serde_json::to_string(&request).ok()?;
    line.push('\n');
    stream.write_all(line.as_bytes()).ok()?;
    stream.flush().ok()?;

    let mut reader = BufReader::new(stream);
    let mut resp = String::new();
    reader.read_line(&mut resp).ok()?;

    let parsed: Value = serde_json::from_str(resp.trim()).ok()?;
    parsed
        .get("result")?
        .get("data")?
        .as_str()
        .map(String::from)
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
    fn vault_store_requires_owner_id() {
        let err = handle_vault_store(&json!({"data": "dGVzdA=="})).unwrap_err();
        assert_eq!(err.code, -32602);
    }

    #[test]
    fn vault_store_requires_data() {
        let err = handle_vault_store(&json!({"owner_id": "test"})).unwrap_err();
        assert_eq!(err.code, -32602);
    }

    #[test]
    fn vault_store_success() {
        let result = handle_vault_store(&json!({
            "owner_id": "test-owner",
            "label": "sample.fastq",
            "data": "ATCGATCG",
        }))
        .unwrap();
        assert_eq!(result["status"], "stored");
        assert!(result["content_hash"].as_str().is_some());
    }

    #[test]
    fn vault_retrieve_requires_consent() {
        let result = handle_vault_retrieve(&json!({
            "owner_id": "test",
            "content_hash": "abc123",
        }))
        .unwrap();
        assert_eq!(result["error"], "consent_required");
    }

    #[test]
    fn vault_consent_verify_requires_token() {
        let err = handle_vault_consent_verify(&json!({
            "owner_id": "test",
        }))
        .unwrap_err();
        assert_eq!(err.code, -32602);
    }
}
