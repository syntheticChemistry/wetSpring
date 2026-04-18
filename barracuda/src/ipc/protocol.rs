// SPDX-License-Identifier: AGPL-3.0-or-later
//! JSON-RPC 2.0 protocol types and serialization for the Primal IPC Protocol.

pub use super::message::*;

use std::borrow::Cow;

use serde_json::Value;

use crate::primal_names::LEGACY_SELF_METHOD_PREFIX;

/// Normalize a JSON-RPC method name to the bare `{domain}.{operation}` form (barraCuda v0.3.12+).
///
/// Strips legacy product prefixes (`wetspring.`, `barracuda.`) when present so clients using
/// either prefixed or bare names reach the same dispatch table. Repeated prefixes are stripped
/// until stable (e.g. `wetspring.barracuda.science.diversity` → `science.diversity`).
#[must_use]
pub fn normalize_method(method: &str) -> Cow<'_, str> {
    let mut s = method;
    loop {
        let next = s
            .strip_prefix(LEGACY_SELF_METHOD_PREFIX)
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

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "test module: assertions use unwrap for clarity"
)]
mod tests {
    use super::*;

    #[test]
    fn normalize_method_bare_name_unchanged() {
        assert_eq!(
            normalize_method("science.diversity").as_ref(),
            "science.diversity"
        );
        assert_eq!(
            normalize_method("metrics.snapshot").as_ref(),
            "metrics.snapshot"
        );
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
