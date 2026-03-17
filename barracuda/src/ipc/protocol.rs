// SPDX-License-Identifier: AGPL-3.0-or-later
//! JSON-RPC 2.0 protocol types and serialization for the Primal IPC Protocol.

use std::fmt;

use serde_json::Value;

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

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;

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
}
