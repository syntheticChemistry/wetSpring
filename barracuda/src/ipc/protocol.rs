// SPDX-License-Identifier: AGPL-3.0-or-later
//! JSON-RPC 2.0 protocol types and serialization for the Primal IPC Protocol.

use std::borrow::Cow;
use std::fmt;

use serde_json::Value;

/// Normalize a JSON-RPC method name to the bare `{domain}.{operation}` form (barraCuda v0.3.7+).
///
/// Strips legacy product prefixes (`wetspring.`, `barracuda.`) when present so clients using
/// either prefixed or bare names reach the same dispatch table. Repeated prefixes are stripped
/// until stable (e.g. `wetspring.barracuda.science.diversity` → `science.diversity`).
#[must_use]
pub fn normalize_method(method: &str) -> Cow<'_, str> {
    let mut s = method;
    loop {
        let next = s
            .strip_prefix("wetspring.")
            .or_else(|| s.strip_prefix("barracuda."));
        match next {
            Some(rest) => s = rest,
            None => break,
        }
    }
    if s == method {
        Cow::Borrowed(method)
    } else {
        Cow::Owned(s.to_string())
    }
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

/// JSON-RPC 2.0 error with numeric code and human-readable message.
#[derive(Debug, Clone)]
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

impl fmt::Display for RpcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.code, self.message)
    }
}

impl std::error::Error for RpcError {}

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

/// Parse a single JSON-RPC 2.0 request from a newline-delimited string.
///
/// # Errors
///
/// Returns a [`ParseError`] on failure, following JSON-RPC error codes:
/// - `-32700`: Parse error (malformed JSON)
/// - `-32600`: Invalid request (missing required fields)
pub fn parse_request(line: &str) -> Result<Request, ParseError> {
    let val: Value = serde_json::from_str(line.trim()).map_err(|e| ParseError {
        id: Value::Null,
        error: RpcError {
            code: -32700,
            message: format!("parse error: {e}"),
        },
    })?;

    let id = val.get("id").cloned().unwrap_or(Value::Null);

    let jsonrpc = val.get("jsonrpc").and_then(Value::as_str);
    if jsonrpc != Some("2.0") {
        return Err(ParseError {
            id,
            error: RpcError {
                code: -32600,
                message: "invalid or missing jsonrpc version".into(),
            },
        });
    }

    let method = val
        .get("method")
        .and_then(Value::as_str)
        .ok_or_else(|| ParseError {
            id: id.clone(),
            error: RpcError {
                code: -32600,
                message: "missing method field".into(),
            },
        })?
        .to_string();

    let params = val
        .get("params")
        .cloned()
        .unwrap_or_else(|| Value::Object(serde_json::Map::new()));

    Ok(Request { method, params, id })
}

/// Format a JSON-RPC 2.0 success response.
#[must_use]
pub fn success_response(id: &Value, result: &Value) -> String {
    let resp = serde_json::json!({
        "jsonrpc": "2.0",
        "result": result,
        "id": id,
    });
    serde_json::to_string(&resp).unwrap_or_else(|_| {
        r#"{"jsonrpc":"2.0","error":{"code":-32603,"message":"serialize error"},"id":null}"#
            .to_string()
    })
}

/// Format a JSON-RPC 2.0 error response.
#[must_use]
pub fn error_response(id: &Value, code: i64, message: &str) -> String {
    let resp = serde_json::json!({
        "jsonrpc": "2.0",
        "error": {"code": code, "message": message},
        "id": id,
    });
    serde_json::to_string(&resp).unwrap_or_else(|_| {
        format!(r#"{{"jsonrpc":"2.0","error":{{"code":{code},"message":"{message}"}},"id":null}}"#)
    })
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
#[expect(
    clippy::unwrap_used,
    reason = "test module: assertions use unwrap for clarity"
)]
mod tests {
    use super::*;

    #[test]
    fn normalize_method_bare_name_unchanged() {
        assert_eq!(normalize_method("science.diversity").as_ref(), "science.diversity");
        assert_eq!(normalize_method("metrics.snapshot").as_ref(), "metrics.snapshot");
    }

    #[test]
    fn normalize_method_strips_wetspring_prefix() {
        assert_eq!(
            normalize_method("wetspring.science.diversity").as_ref(),
            "science.diversity"
        );
    }

    #[test]
    fn normalize_method_strips_barracuda_prefix() {
        assert_eq!(
            normalize_method("barracuda.science.diversity").as_ref(),
            "science.diversity"
        );
    }

    #[test]
    fn normalize_method_strips_chained_prefixes() {
        assert_eq!(
            normalize_method("wetspring.barracuda.science.diversity").as_ref(),
            "science.diversity"
        );
    }

    #[test]
    fn normalize_method_no_false_strip_on_substring() {
        assert_eq!(
            normalize_method("wetspring_science.diversity").as_ref(),
            "wetspring_science.diversity"
        );
    }

    #[test]
    fn parse_valid_request() {
        let line = r#"{"jsonrpc":"2.0","method":"health.check","params":{},"id":1}"#;
        let req = parse_request(line).unwrap();
        assert_eq!(req.method, "health.check");
        assert_eq!(req.id, serde_json::json!(1));
    }

    #[test]
    fn parse_request_missing_method() {
        let line = r#"{"jsonrpc":"2.0","params":{},"id":1}"#;
        let err = parse_request(line).unwrap_err();
        assert_eq!(err.error.code, -32600);
    }

    #[test]
    fn parse_request_bad_json() {
        let err = parse_request("not json").unwrap_err();
        assert_eq!(err.error.code, -32700);
    }

    #[test]
    fn parse_request_wrong_version() {
        let line = r#"{"jsonrpc":"1.0","method":"test","id":1}"#;
        let err = parse_request(line).unwrap_err();
        assert_eq!(err.error.code, -32600);
    }

    #[test]
    fn parse_request_default_params() {
        let line = r#"{"jsonrpc":"2.0","method":"health.check","id":1}"#;
        let req = parse_request(line).unwrap();
        assert!(req.params.is_object());
    }

    #[test]
    fn success_response_format() {
        let resp = success_response(&serde_json::json!(1), &serde_json::json!({"ok": true}));
        let val: Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(val["jsonrpc"], "2.0");
        assert_eq!(val["id"], 1);
        assert_eq!(val["result"]["ok"], true);
    }

    #[test]
    fn error_response_format() {
        let resp = error_response(&serde_json::json!(2), -32601, "method not found");
        let val: Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(val["jsonrpc"], "2.0");
        assert_eq!(val["error"]["code"], -32601);
        assert_eq!(val["error"]["message"], "method not found");
    }

    #[test]
    fn parse_request_null_id() {
        let line = r#"{"jsonrpc":"2.0","method":"test","params":{}}"#;
        let req = parse_request(line).unwrap();
        assert!(req.id.is_null());
    }

    #[test]
    fn extract_rpc_error_from_error_response() {
        let resp =
            r#"{"jsonrpc":"2.0","error":{"code":-32601,"message":"method not found"},"id":1}"#;
        let (code, msg) = extract_rpc_error(resp).unwrap();
        assert_eq!(code, -32601);
        assert_eq!(msg, "method not found");
    }

    #[test]
    fn extract_rpc_error_from_success_response() {
        let resp = r#"{"jsonrpc":"2.0","result":{"status":"ok"},"id":1}"#;
        assert!(extract_rpc_error(resp).is_none());
    }

    #[test]
    fn extract_rpc_error_from_malformed() {
        assert!(extract_rpc_error("not json").is_none());
        assert!(extract_rpc_error("").is_none());
    }

    #[test]
    fn extract_capabilities_flat_format() {
        let resp = r#"{"jsonrpc":"2.0","result":{"primal":"wetspring","version":"0.2.0","capabilities":["science.diversity","science.anderson"]},"id":1}"#;
        let info = extract_capabilities(resp).unwrap();
        assert_eq!(info.primal.as_deref(), Some("wetspring"));
        assert_eq!(info.version.as_deref(), Some("0.2.0"));
        assert_eq!(info.capabilities.len(), 2);
        assert!(info.domains.is_empty());
    }

    #[test]
    fn extract_capabilities_dual_format() {
        let resp = r#"{"jsonrpc":"2.0","result":{"primal":"wetspring","capabilities":["science.diversity"],"domains":[{"name":"ecology.diversity","description":"Alpha diversity","methods":["science.diversity"]}]},"id":1}"#;
        let info = extract_capabilities(resp).unwrap();
        assert_eq!(info.capabilities.len(), 1);
        assert_eq!(info.domains.len(), 1);
        assert_eq!(info.domains[0].name, "ecology.diversity");
        assert_eq!(info.domains[0].methods, vec!["science.diversity"]);
    }

    #[test]
    fn extract_capabilities_no_result() {
        let resp = r#"{"jsonrpc":"2.0","error":{"code":-1,"message":"fail"},"id":1}"#;
        assert!(extract_capabilities(resp).is_none());
    }

    #[test]
    fn extract_capabilities_empty_response() {
        assert!(extract_capabilities("").is_none());
        assert!(extract_capabilities("not json").is_none());
    }

    #[test]
    fn extract_capabilities_minimal() {
        let resp = r#"{"jsonrpc":"2.0","result":{},"id":1}"#;
        let info = extract_capabilities(resp).unwrap();
        assert!(info.capabilities.is_empty());
        assert!(info.domains.is_empty());
        assert!(info.method_info.is_empty());
        assert!(info.semantic_mappings.is_empty());
        assert!(info.primal.is_none());
    }

    #[test]
    fn extract_capabilities_format_c_method_info() {
        let resp = r#"{"jsonrpc":"2.0","result":{"primal":"wetspring","method_info":{"science.diversity":{"description":"Alpha diversity","cost":"low"},"science.anderson":{"description":"Spectral"}}},"id":1}"#;
        let info = extract_capabilities(resp).unwrap();
        assert_eq!(info.method_info.len(), 2);
        let div = info
            .method_info
            .iter()
            .find(|m| m.method == "science.diversity")
            .unwrap();
        assert_eq!(div.description, "Alpha diversity");
        assert_eq!(div.cost.as_deref(), Some("low"));
        let and = info
            .method_info
            .iter()
            .find(|m| m.method == "science.anderson")
            .unwrap();
        assert!(and.cost.is_none());
    }

    #[test]
    fn extract_capabilities_format_d_semantic_mappings() {
        let resp = r#"{"jsonrpc":"2.0","result":{"primal":"wetspring","semantic_mappings":{"ecology.alpha_diversity":"science.diversity","ecology.spectral":"science.anderson"}},"id":1}"#;
        let info = extract_capabilities(resp).unwrap();
        assert_eq!(info.semantic_mappings.len(), 2);
        assert!(
            info.semantic_mappings
                .iter()
                .any(|(k, v)| k == "ecology.alpha_diversity" && v == "science.diversity")
        );
    }

    #[test]
    fn extract_rpc_result_success() {
        let expected_shannon = 4.0_f64.ln();
        let resp =
            format!(r#"{{"jsonrpc":"2.0","result":{{"shannon":{expected_shannon}}},"id":1}}"#);
        let result = extract_rpc_result(&resp).unwrap();
        assert!(
            (result["shannon"].as_f64().unwrap() - expected_shannon).abs()
                < crate::tolerances::IPC_JSON_ROUNDTRIP
        );
    }

    #[test]
    fn extract_rpc_result_error_response() {
        let resp =
            r#"{"jsonrpc":"2.0","error":{"code":-32601,"message":"method not found"},"id":1}"#;
        assert!(extract_rpc_result(resp).is_none());
    }

    #[test]
    fn extract_rpc_result_malformed() {
        assert!(extract_rpc_result("not json").is_none());
        assert!(extract_rpc_result("").is_none());
    }

    #[test]
    fn is_notification_id_absent() {
        let req = r#"{"jsonrpc":"2.0","method":"health.check","params":{}}"#;
        assert!(is_notification(req));
    }

    #[test]
    fn is_notification_id_null_is_not_notification() {
        let req = r#"{"jsonrpc":"2.0","method":"health.check","params":{},"id":null}"#;
        assert!(!is_notification(req));
    }

    #[test]
    fn is_notification_id_present() {
        let req = r#"{"jsonrpc":"2.0","method":"health.check","params":{},"id":1}"#;
        assert!(!is_notification(req));
    }

    #[test]
    fn is_notification_nested_id_ignored() {
        let req = r#"{"jsonrpc":"2.0","method":"x","params":{"id":"nested"},"id":2}"#;
        assert!(!is_notification(req));
    }

    #[test]
    fn is_batch_array() {
        assert!(is_batch("[1,2,3]"));
        assert!(is_batch("  [{}]  "));
    }

    #[test]
    fn is_batch_not_array() {
        assert!(!is_batch(r#"{"jsonrpc":"2.0","method":"x","id":1}"#));
    }

    #[test]
    fn parse_batch_empty_returns_empty() {
        let batch = "[]";
        let elems = parse_batch(batch);
        assert!(elems.is_empty());
    }

    #[test]
    fn parse_batch_single() {
        let batch = r#"[{"jsonrpc":"2.0","method":"health.check","params":{},"id":1}]"#;
        let elems = parse_batch(batch);
        assert_eq!(elems.len(), 1);
        let req = parse_request(&elems[0]).unwrap();
        assert_eq!(req.method, "health.check");
        assert_eq!(req.id, serde_json::json!(1));
    }

    #[test]
    fn parse_batch_multiple() {
        let batch =
            r#"[{"jsonrpc":"2.0","method":"a","id":1},{"jsonrpc":"2.0","method":"b","id":2}]"#;
        let elems = parse_batch(batch);
        assert_eq!(elems.len(), 2);
    }

    #[test]
    fn parse_batch_nested() {
        let batch = r#"[{"params":{"nested":[1,2,3]},"method":"x","id":1}]"#;
        let elems = parse_batch(batch);
        assert_eq!(elems.len(), 1);
        assert!(elems[0].contains("nested"));
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn parse_request_never_panics(input in "\\PC{0,512}") {
                let _ = parse_request(&input);
            }

            #[test]
            fn extract_capabilities_never_panics(input in "\\PC{0,512}") {
                let _ = extract_capabilities(&input);
            }

            #[test]
            fn is_notification_never_panics(input in "\\PC{0,512}") {
                let _ = is_notification(&input);
            }

            #[test]
            fn extract_rpc_error_never_panics(input in "\\PC{0,512}") {
                let _ = extract_rpc_error(&input);
            }

            #[test]
            fn valid_request_roundtrip(
                method in "[a-z]{1,10}\\.[a-z]{1,10}",
                id in 1..1000i64,
            ) {
                let json = format!(
                    r#"{{"jsonrpc":"2.0","method":"{method}","params":{{}},"id":{id}}}"#
                );
                let req = parse_request(&json).unwrap();
                prop_assert_eq!(req.method, method);
                prop_assert_eq!(req.id, serde_json::json!(id));
            }

            #[test]
            fn success_response_always_valid_json(
                id in 1..1000i64,
                key in "[a-z]{1,10}",
                val in proptest::num::f64::NORMAL,
            ) {
                let resp = success_response(
                    &serde_json::json!(id),
                    &serde_json::json!({ key: val }),
                );
                let parsed: serde_json::Value = serde_json::from_str(&resp).unwrap();
                prop_assert_eq!(parsed["jsonrpc"].as_str(), Some("2.0"));
                prop_assert_eq!(&parsed["id"], &serde_json::json!(id));
                prop_assert!(parsed.get("result").is_some());
            }

            #[test]
            fn capability_extraction_preserves_primal_name(
                primal in "[a-z]{3,12}",
                version in "[0-9]{1,2}\\.[0-9]{1,2}\\.[0-9]{1,2}",
            ) {
                let resp = format!(
                    r#"{{"jsonrpc":"2.0","result":{{"primal":"{primal}","version":"{version}","capabilities":[]}},"id":1}}"#
                );
                let info = extract_capabilities(&resp).unwrap();
                prop_assert_eq!(info.primal.as_deref(), Some(primal.as_str()));
                prop_assert_eq!(info.version.as_deref(), Some(version.as_str()));
            }
        }
    }
}
