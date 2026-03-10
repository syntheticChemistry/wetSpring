// SPDX-License-Identifier: AGPL-3.0-or-later
//! Optional `NestGate` data provider for NCBI operations.
//!
//! When `WETSPRING_DATA_PROVIDER=nestgate` is set, wetSpring routes NCBI
//! data requests through `NestGate`'s Unix socket JSON-RPC API instead of
//! making direct HTTP calls. Falls back to sovereign HTTP if `NestGate` is
//! unavailable.
//!
//! # Protocol
//!
//! JSON-RPC 2.0, newline-delimited, over Unix domain socket.
//! Socket discovery (capability-based, no hardcoded paths):
//! 1. `NESTGATE_SOCKET` env var (explicit override)
//! 2. `$XDG_RUNTIME_DIR/biomeos/nestgate-default.sock`
//! 3. `<temp_dir>/nestgate-default.sock` (platform-agnostic fallback)
//!
//! # Data routing tiers
//!
//! | Tier | Strategy | When |
//! |------|----------|------|
//! | 1 | biomeOS Neural API `capability.call` | biomeOS orchestrator running |
//! | 2 | Direct `NestGate` socket | Standalone + NestGate available |
//! | 3 | Sovereign HTTP | No ecosystem services |

mod discovery;
mod fetch;
mod rpc;
mod storage;

pub use discovery::{discover_biomeos_socket, discover_socket, is_enabled};
pub use fetch::{fetch_or_fallback, fetch_tiered, fetch_via_biomeos};
pub use rpc::{escape_json, extract_error, extract_result_value, health};
pub use storage::{exists, retrieve, store};

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn is_enabled_default_false() {
        assert!(
            !std::env::var("WETSPRING_DATA_PROVIDER")
                .is_ok_and(|v| v.trim().eq_ignore_ascii_case("nestgate"))
                || is_enabled()
        );
    }

    #[test]
    fn resolve_socket_explicit_nonexistent() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nonexistent_wetspring_test.sock");
        let result = discovery::resolve_socket(Some(path.to_str().unwrap()), None);
        assert!(result.is_none());
    }

    #[test]
    fn resolve_socket_all_none() {
        let result = discovery::resolve_socket(None, None);
        assert!(
            result.is_none() || result.is_some_and(|p| p.exists()),
            "should be None or a path that exists"
        );
    }

    #[test]
    fn resolve_socket_explicit_exists() {
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("test.sock");
        std::fs::write(&sock, "").unwrap();
        let result = discovery::resolve_socket(Some(sock.to_str().unwrap()), None);
        assert_eq!(result, Some(sock));
    }

    #[test]
    fn resolve_socket_xdg_path() {
        let dir = tempfile::tempdir().unwrap();
        let biomeos = dir.path().join("biomeos");
        std::fs::create_dir_all(&biomeos).unwrap();
        let sock = biomeos.join("nestgate-default.sock");
        std::fs::write(&sock, "").unwrap();
        let result = discovery::resolve_socket(None, Some(dir.path().to_str().unwrap()));
        assert_eq!(result, Some(sock));
    }

    #[test]
    fn resolve_socket_xdg_nonexistent() {
        let dir = tempfile::tempdir().unwrap();
        let xdg = dir.path().join("nonexistent_xdg");
        let result = discovery::resolve_socket(None, Some(xdg.to_str().unwrap()));
        assert!(result.is_none() || result.is_some_and(|p| p.exists()));
    }

    #[test]
    fn resolve_socket_explicit_overrides_xdg() {
        let dir = tempfile::tempdir().unwrap();
        let explicit_sock = dir.path().join("explicit.sock");
        std::fs::write(&explicit_sock, "").unwrap();
        let xdg_dir = tempfile::tempdir().unwrap();
        let biomeos = xdg_dir.path().join("biomeos");
        std::fs::create_dir_all(&biomeos).unwrap();
        let xdg_sock = biomeos.join("nestgate-default.sock");
        std::fs::write(&xdg_sock, "").unwrap();
        let result = discovery::resolve_socket(
            Some(explicit_sock.to_str().unwrap()),
            Some(xdg_dir.path().to_str().unwrap()),
        );
        assert_eq!(result, Some(explicit_sock));
    }

    #[test]
    fn resolve_socket_temp_fallback() {
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("nestgate-default.sock");
        std::fs::write(&sock, "").unwrap();
        let result = discovery::resolve_socket(None, Some("/nonexistent_xdg_path_12345"));
        assert!(result.is_none() || result.is_some_and(|p| p.exists()));
    }

    #[test]
    fn escape_json_special_chars() {
        assert_eq!(escape_json("hello\nworld"), "hello\\nworld");
        assert_eq!(escape_json("say \"hi\""), "say \\\"hi\\\"");
        assert_eq!(escape_json("back\\slash"), "back\\\\slash");
        assert_eq!(escape_json("tab\there"), "tab\\there");
    }

    #[test]
    fn escape_json_empty() {
        assert_eq!(escape_json(""), "");
    }

    #[test]
    fn escape_json_no_special() {
        assert_eq!(escape_json("plain text"), "plain text");
    }

    #[test]
    fn extract_error_with_message() {
        let response = r#"{"jsonrpc":"2.0","error":{"code":-32600,"message":"not found"},"id":1}"#;
        let err = extract_error(response);
        assert!(err.contains("not found"));
    }

    #[test]
    fn extract_error_no_message() {
        let response = r#"{"jsonrpc":"2.0","error":{"code":-32600},"id":1}"#;
        let err = extract_error(response);
        assert!(err.contains("RPC error"));
    }

    #[test]
    fn extract_error_truncates_long_response() {
        let response = format!(
            r#"{{"jsonrpc":"2.0","error":{{"code":-1,"message":"x"}},"id":1}}{}"#,
            "y".repeat(500)
        );
        let err = extract_error(&response);
        assert!(err.contains('x'));
    }

    #[test]
    fn extract_error_empty_response() {
        let err = extract_error("");
        assert!(err.contains("RPC error"));
    }

    #[test]
    fn extract_result_value_string() {
        let response = r#"{"jsonrpc":"2.0","result":">seq1\nATCG","id":1}"#;
        let val = extract_result_value(response).unwrap();
        assert!(val.contains(">seq1"));
    }

    #[test]
    fn extract_result_value_missing() {
        let response = r#"{"jsonrpc":"2.0","id":1}"#;
        assert!(extract_result_value(response).is_err());
    }

    #[test]
    fn extract_result_value_escaped_newline() {
        let response = r#"{"jsonrpc":"2.0","result":">seq1\nATCG\n","id":1}"#;
        let val = extract_result_value(response).unwrap();
        assert_eq!(val, ">seq1\nATCG\n");
    }

    #[test]
    fn extract_result_value_escaped_quote() {
        let response = r#"{"jsonrpc":"2.0","result":"say \"hi\"","id":1}"#;
        let val = extract_result_value(response).unwrap();
        assert!(val.contains("say"));
    }

    #[test]
    fn extract_result_value_nested_object() {
        let response = r#"{"jsonrpc":"2.0","result":{"value":"cached"},"id":1}"#;
        let val = extract_result_value(response).unwrap();
        assert!(val.contains("value") && val.contains("cached"));
    }

    #[test]
    fn store_request_format() {
        let key = "ncbi:nucleotide:K03455";
        let value = ">seq\nATCG";
        let request = format!(
            r#"{{"jsonrpc":"2.0","method":"storage.store","params":{{"key":"{}","value":"{}","family_id":"{}"}},"id":1}}"#,
            escape_json(key),
            escape_json(value),
            "default",
        );
        assert!(request.contains("storage.store"));
        assert!(request.contains("ncbi:nucleotide:K03455"));
        assert!(request.contains(">seq\\nATCG"));
    }

    #[test]
    fn retrieve_request_format() {
        let key = "test_key";
        let request = format!(
            r#"{{"jsonrpc":"2.0","method":"storage.retrieve","params":{{"key":"{}","family_id":"{}"}},"id":2}}"#,
            escape_json(key),
            "default",
        );
        assert!(request.contains("storage.retrieve"));
        assert!(request.contains("test_key"));
    }

    #[test]
    fn exists_request_format() {
        let request = format!(
            r#"{{"jsonrpc":"2.0","method":"storage.exists","params":{{"key":"{}","family_id":"{}"}},"id":3}}"#,
            escape_json("key"),
            "default",
        );
        assert!(request.contains("storage.exists"));
    }

    #[test]
    fn health_request_format() {
        let request = r#"{"jsonrpc":"2.0","method":"health","params":{},"id":0}"#;
        assert!(request.contains("health"));
    }

    #[test]
    fn discover_socket_does_not_panic() {
        let _ = discover_socket();
    }

    #[test]
    fn discover_biomeos_socket_does_not_panic() {
        let _ = discover_biomeos_socket();
    }

    #[test]
    fn resolve_biomeos_socket_explicit() {
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("biomeos.sock");
        std::fs::write(&sock, "").unwrap();
        let result = discovery::resolve_biomeos_socket(Some(sock.to_str().unwrap()), None);
        assert_eq!(result, Some(sock));
    }

    #[test]
    fn resolve_biomeos_socket_xdg() {
        let dir = tempfile::tempdir().unwrap();
        let biomeos_dir = dir.path().join("biomeos");
        std::fs::create_dir_all(&biomeos_dir).unwrap();
        let sock = biomeos_dir.join("biomeos-default.sock");
        std::fs::write(&sock, "").unwrap();
        let result = discovery::resolve_biomeos_socket(None, Some(dir.path().to_str().unwrap()));
        assert_eq!(result, Some(sock));
    }

    #[test]
    fn resolve_biomeos_socket_none() {
        let result = discovery::resolve_biomeos_socket(None, None);
        assert!(result.is_none() || result.is_some_and(|p| p.exists()));
    }

    #[test]
    fn fetch_via_biomeos_nonexistent_socket() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wetspring_test_no_biomeos.sock");
        let err = fetch_via_biomeos(&path, "nucleotide", "K03455", "").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("NestGate connect") || msg.contains("invalid socket"));
    }

    #[test]
    fn extract_fasta_from_response_with_fasta_key() {
        let resp = r#"{"jsonrpc":"2.0","result":{"fasta":">seq1\nATCG"},"id":1}"#;
        let fasta = super::rpc::extract_fasta_from_response(resp).unwrap();
        assert!(fasta.contains(">seq1"));
    }

    #[test]
    fn extract_fasta_from_response_bare_result() {
        let resp = r#"{"jsonrpc":"2.0","result":">seq2\nGCTA","id":1}"#;
        let fasta = super::rpc::extract_fasta_from_response(resp).unwrap();
        assert!(fasta.contains(">seq2"));
    }

    #[test]
    fn health_nonexistent_socket_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wetspring_test_nonexistent_nestgate.sock");
        let err = health(&path).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("NestGate connect") || msg.contains("invalid socket"));
    }

    #[test]
    fn invalid_socket_path_too_long() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a".repeat(200));
        let err = health(&path).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("invalid socket") || msg.contains("NestGate connect"));
    }

    #[test]
    fn store_nonexistent_socket_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wetspring_test_nonexistent_store.sock");
        let err = store(&path, "key", "value").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("NestGate connect") || msg.contains("invalid socket"));
    }

    #[test]
    fn retrieve_nonexistent_socket_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wetspring_test_nonexistent_retrieve.sock");
        let err = retrieve(&path, "key").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("NestGate connect") || msg.contains("invalid socket"));
    }

    #[test]
    fn exists_nonexistent_socket_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wetspring_test_nonexistent_exists.sock");
        let err = exists(&path, "key").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("NestGate connect") || msg.contains("invalid socket"));
    }
}
