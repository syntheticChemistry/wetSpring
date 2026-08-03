// SPDX-License-Identifier: AGPL-3.0-or-later
//! JSON-RPC 2.0 message shapes, RPC errors, and structured capability payloads.
//!
//! Holds the typed surface of requests, errors, and [`CapabilityInfo`] (multi-format
//! `capability.list` results). Complements [`super::protocol`] helpers for
//! normalization, framing, and response formatting.

use serde_json::Value;

/// JSON-RPC 2.0 error with numeric code and human-readable message.
#[derive(Debug, Clone, thiserror::Error)]
#[error("[{code}] {message}")]
pub struct RpcError {
    /// Standard JSON-RPC error code (e.g. `-32601` method not found).
    pub code: i64,
    /// Human-readable description.
    pub message: String,
}

impl RpcError {
    /// Method not found (`-32601`).
    #[must_use]
    pub fn method_not_found(method: &str) -> Self {
        Self {
            code: -32601,
            message: format!("method not found: {method}"),
        }
    }

    /// Invalid method parameters (`-32602`).
    #[must_use]
    pub fn invalid_params(msg: impl Into<String>) -> Self {
        Self {
            code: -32602,
            message: msg.into(),
        }
    }

    /// Application-level server error (`-32000` to `-32099`).
    #[must_use]
    pub fn server_error(code: i64, msg: impl Into<String>) -> Self {
        Self {
            code,
            message: msg.into(),
        }
    }
}

/// Parse failure including the request `id` for correlating the error response.
#[derive(Debug)]
pub struct ParseError {
    /// Request ID (or `Value::Null` if unparseable).
    pub id: Value,
    /// The underlying RPC error.
    pub error: RpcError,
}

/// A parsed JSON-RPC 2.0 request.
#[derive(Debug)]
pub struct Request {
    /// The method name (e.g., `"science.diversity"`).
    pub method: String,
    /// Parameters object (may be empty `{}`).
    pub params: Value,
    /// Request ID for correlating responses.
    pub id: Value,
}

/// Returns `true` if the JSON-RPC input is a notification (Request without "id").
///
/// Per JSON-RPC 2.0 Section 4.1: a Notification has no "id" member (key absent).
/// Note: `"id": null` is NOT a notification — it is a request and must get a response.
/// Detection checks whether the "id" key EXISTS in the top-level object.
///
/// Returns `false` for invalid JSON, non-objects, or when "id" is present.
#[must_use]
pub fn is_notification(raw_json: &str) -> bool {
    let val: Value = match serde_json::from_str(raw_json.trim()) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let Some(obj) = val.as_object() else {
        return false;
    };
    !obj.contains_key("id")
}

/// Returns `true` if the trimmed input is a JSON-RPC batch (array of requests).
#[must_use]
pub fn is_batch(raw_json: &str) -> bool {
    raw_json.trim().starts_with('[')
}

/// Splits a JSON-RPC batch array into individual request strings.
///
/// Handles nested objects and arrays correctly. Returns an empty vec for invalid
/// JSON or non-array input. Each element is serialized back to a JSON string.
///
/// # Errors
///
/// Returns empty `Vec` if input is not a valid JSON array.
#[must_use]
pub fn parse_batch(raw_json: &str) -> Vec<String> {
    let val: Value = match serde_json::from_str(raw_json.trim()) {
        Ok(v) => v,
        Err(_) => return vec![],
    };
    let Some(arr) = val.as_array() else {
        return vec![];
    };
    arr.iter()
        .filter_map(|v| serde_json::to_string(v).ok())
        .collect()
}

/// Extract an RPC error message from a raw JSON-RPC response string.
///
/// Returns `Some((code, message))` if the response contains an `"error"` field
/// with `"code"` and `"message"` subfields. Returns `None` for success responses
/// or unparseable strings.
///
/// Centralizes the JSON-RPC error extraction pattern used by Songbird, `NestGate`,
/// and provenance clients (healthSpring V29 pattern).
#[must_use]
pub fn extract_rpc_error(response: &str) -> Option<(i64, String)> {
    let val: Value = serde_json::from_str(response).ok()?;
    let err = val.get("error")?;
    let code = err.get("code")?.as_i64()?;
    let message = err.get("message")?.as_str()?.to_string();
    Some((code, message))
}

/// Extract the `"result"` value from a JSON-RPC success response.
///
/// Returns `Some(result_value)` for success responses, `None` for error
/// responses or unparseable strings. Complements [`extract_rpc_error`].
///
/// Follows the healthSpring V29 centralized extraction pattern — callers
/// use `extract_rpc_result` + `extract_rpc_error` instead of ad-hoc
/// `serde_json::from_str` + `val["result"]` everywhere.
#[must_use]
pub fn extract_rpc_result(response: &str) -> Option<Value> {
    let val: Value = serde_json::from_str(response).ok()?;
    val.get("result").cloned()
}

/// Extracted capability information from a `capability.list` response.
///
/// Supports four formats (airSpring/sweetGrass 4-format standard):
/// - Format A: `"capabilities": ["a","b"]` — flat string list
/// - Format B: `"domains": [{"name":"x","methods":["a"]}]` — structured domains
/// - Format C: `"method_info": {"a": {"description":"...", "cost":"low"}}` — per-method metadata
/// - Format D: `"semantic_mappings": {"domain.op": "method.name"}` — semantic aliases
#[derive(Debug, Clone, Default)]
pub struct CapabilityInfo {
    /// Flat capability list (Format A).
    pub capabilities: Vec<String>,
    /// Structured domain groupings (Format B), if present.
    pub domains: Vec<CapabilityDomain>,
    /// Per-method metadata (Format C), if present.
    pub method_info: Vec<MethodInfo>,
    /// Semantic domain→method mappings (Format D), if present.
    pub semantic_mappings: Vec<(String, String)>,
    /// Primal name, if present.
    pub primal: Option<String>,
    /// Primal version, if present.
    pub version: Option<String>,
}

/// Per-method metadata from Format C.
#[derive(Debug, Clone)]
pub struct MethodInfo {
    /// Method name (e.g. `"science.diversity"`).
    pub method: String,
    /// Human-readable description.
    pub description: String,
    /// Cost estimate (`"low"`, `"medium"`, `"high"`), if provided.
    pub cost: Option<String>,
}

/// A structured capability domain from Format B.
#[derive(Debug, Clone)]
pub struct CapabilityDomain {
    /// Domain name (e.g. `"ecology.diversity"`).
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// IPC methods in this domain.
    pub methods: Vec<String>,
}

/// Extract capabilities from a `capability.list` JSON-RPC response.
///
/// Parses all four capability formats (airSpring/sweetGrass standard):
/// - Format A: flat `"capabilities"` array
/// - Format B: structured `"domains"` array
/// - Format C: per-method `"method_info"` object
/// - Format D: `"semantic_mappings"` object (domain alias → method name)
///
/// Returns `None` if the response is not valid JSON or has no result.
#[must_use]
pub fn extract_capabilities(response: &str) -> Option<CapabilityInfo> {
    let val: Value = serde_json::from_str(response).ok()?;
    let result = val.get("result")?;

    let primal = result
        .get("primal")
        .and_then(Value::as_str)
        .map(str::to_string);
    let version = result
        .get("version")
        .and_then(Value::as_str)
        .map(str::to_string);

    let capabilities = result
        .get("capabilities")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();

    let domains = result
        .get("domains")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|d| {
                    let name = d.get("name")?.as_str()?.to_string();
                    let description = d
                        .get("description")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    let methods = d
                        .get("methods")
                        .and_then(Value::as_array)
                        .map(|m| {
                            m.iter()
                                .filter_map(Value::as_str)
                                .map(str::to_string)
                                .collect()
                        })
                        .unwrap_or_default();
                    Some(CapabilityDomain {
                        name,
                        description,
                        methods,
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    let method_info = result
        .get("method_info")
        .and_then(Value::as_object)
        .map(|obj| {
            obj.iter()
                .map(|(method, info)| MethodInfo {
                    method: method.clone(),
                    description: info
                        .get("description")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                    cost: info.get("cost").and_then(Value::as_str).map(str::to_string),
                })
                .collect()
        })
        .unwrap_or_default();

    let semantic_mappings = result
        .get("semantic_mappings")
        .and_then(Value::as_object)
        .map(|obj| {
            obj.iter()
                .filter_map(|(alias, method)| Some((alias.clone(), method.as_str()?.to_string())))
                .collect()
        })
        .unwrap_or_default();

    Some(CapabilityInfo {
        capabilities,
        domains,
        method_info,
        semantic_mappings,
        primal,
        version,
    })
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test assertions")]
mod tests {
    use super::*;

    #[test]
    fn rpc_error_method_not_found() {
        let e = RpcError::method_not_found("foo.bar");
        assert_eq!(e.code, -32601);
        assert!(e.message.contains("foo.bar"));
        assert!(e.to_string().contains("-32601"));
    }

    #[test]
    fn rpc_error_invalid_params() {
        let e = RpcError::invalid_params("missing field");
        assert_eq!(e.code, -32602);
        assert!(e.message.contains("missing field"));
    }

    #[test]
    fn rpc_error_server_error() {
        let e = RpcError::server_error(-32001, "internal");
        assert_eq!(e.code, -32001);
        assert_eq!(e.message, "internal");
    }

    #[test]
    fn is_notification_true_for_no_id() {
        assert!(is_notification(r#"{"jsonrpc":"2.0","method":"ping"}"#));
    }

    #[test]
    fn is_notification_false_when_id_present() {
        assert!(!is_notification(
            r#"{"jsonrpc":"2.0","method":"ping","id":1}"#
        ));
    }

    #[test]
    fn is_notification_false_for_null_id() {
        assert!(!is_notification(
            r#"{"jsonrpc":"2.0","method":"ping","id":null}"#
        ));
    }

    #[test]
    fn is_notification_false_for_invalid_json() {
        assert!(!is_notification("not json"));
    }

    #[test]
    fn is_batch_true_for_array() {
        assert!(is_batch(r#"[{"jsonrpc":"2.0","method":"a","id":1}]"#));
    }

    #[test]
    fn is_batch_false_for_object() {
        assert!(!is_batch(r#"{"jsonrpc":"2.0","method":"a","id":1}"#));
    }

    #[test]
    fn parse_batch_splits_array() {
        let batch = r#"[{"method":"a","id":1},{"method":"b","id":2}]"#;
        let items = parse_batch(batch);
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn parse_batch_empty_for_non_array() {
        assert!(parse_batch(r#"{"method":"a"}"#).is_empty());
    }

    #[test]
    fn parse_batch_empty_for_invalid_json() {
        assert!(parse_batch("not json").is_empty());
    }

    #[test]
    fn extract_rpc_error_success_response_returns_none() {
        let resp = r#"{"jsonrpc":"2.0","result":"ok","id":1}"#;
        assert!(extract_rpc_error(resp).is_none());
    }

    #[test]
    fn extract_rpc_error_error_response_returns_code_and_message() {
        let resp =
            r#"{"jsonrpc":"2.0","error":{"code":-32601,"message":"method not found"},"id":1}"#;
        let (code, msg) = extract_rpc_error(resp).unwrap();
        assert_eq!(code, -32601);
        assert_eq!(msg, "method not found");
    }

    #[test]
    fn extract_rpc_result_success() {
        let resp = r#"{"jsonrpc":"2.0","result":{"data":42},"id":1}"#;
        let result = extract_rpc_result(resp).unwrap();
        assert_eq!(result["data"], 42);
    }

    #[test]
    fn extract_rpc_result_error_returns_none() {
        let resp = r#"{"jsonrpc":"2.0","error":{"code":-32601,"message":"nope"},"id":1}"#;
        assert!(extract_rpc_result(resp).is_none());
    }

    #[test]
    fn extract_capabilities_all_formats() {
        let resp = r#"{"jsonrpc":"2.0","result":{
            "primal":"wetspring","version":"0.1.0",
            "capabilities":["science.diversity","science.kmer"],
            "domains":[{"name":"ecology","description":"Ecology domain","methods":["diversity","kmer"]}],
            "method_info":{"diversity":{"description":"Shannon diversity","cost":"low"}},
            "semantic_mappings":{"ecology.diversity":"science.diversity"}
        },"id":1}"#;
        let info = extract_capabilities(resp).unwrap();
        assert_eq!(info.primal.as_deref(), Some("wetspring"));
        assert_eq!(info.version.as_deref(), Some("0.1.0"));
        assert_eq!(info.capabilities.len(), 2);
        assert_eq!(info.domains.len(), 1);
        assert_eq!(info.domains[0].name, "ecology");
        assert_eq!(info.method_info.len(), 1);
        assert_eq!(info.method_info[0].method, "diversity");
        assert_eq!(info.semantic_mappings.len(), 1);
    }

    #[test]
    fn extract_capabilities_returns_none_for_error_response() {
        let resp = r#"{"jsonrpc":"2.0","error":{"code":-1,"message":"fail"},"id":1}"#;
        assert!(extract_capabilities(resp).is_none());
    }
}
