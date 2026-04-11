// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![cfg(feature = "ipc")]
#![expect(
    clippy::expect_used,
    reason = "integration test: expect for IPC setup/parse clarity — failures are test infra bugs"
)]
//! End-to-end IPC integration test: spin up a real Unix socket server,
//! send JSON-RPC 2.0 requests for science handlers, and verify responses.
//!
//! The server unit tests (server.rs) cover protocol mechanics (batching,
//! notifications, error codes). These integration tests cover science
//! handler round-trips that exercise the full math pipeline through IPC.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::time::Duration;

use wetspring_barracuda::ipc::Server;
use wetspring_barracuda::tolerances;

fn rpc_roundtrip(socket_path: &std::path::Path, request: &str) -> serde_json::Value {
    let stream = UnixStream::connect(socket_path).expect("connect to server");
    stream.set_read_timeout(Some(Duration::from_secs(10))).ok();

    let mut writer = std::io::BufWriter::new(&stream);
    writer.write_all(request.as_bytes()).expect("write request");
    writer.write_all(b"\n").expect("write newline");
    writer.flush().expect("flush");

    let mut reader = BufReader::new(&stream);
    let mut response = String::new();
    reader.read_line(&mut response).expect("read response");
    serde_json::from_str(response.trim()).expect("parse JSON response")
}

fn with_server<F: FnOnce(&std::path::Path)>(test_name: &str, f: F) {
    let dir = tempfile::tempdir().expect("tempdir");
    let socket = dir.path().join(format!("{test_name}.sock"));

    let server = Server::bind(&socket).expect("bind server");
    let path = server.socket_path().to_path_buf();

    std::thread::spawn(move || server.run());
    std::thread::sleep(Duration::from_millis(50));

    f(&path);
}

#[test]
fn diversity_science_roundtrip() {
    with_server("diversity", |socket| {
        let resp = rpc_roundtrip(
            socket,
            r#"{"jsonrpc":"2.0","method":"science.diversity","params":{"counts":[0.25,0.25,0.25,0.25]},"id":1}"#,
        );

        assert_eq!(resp["jsonrpc"], "2.0");
        assert_eq!(resp["id"], 1);
        assert!(resp.get("error").is_none(), "unexpected error: {resp}");

        let result = &resp["result"];
        let shannon = result["shannon"].as_f64().expect("shannon value");
        let expected = 4.0_f64.ln();
        assert!(
            (shannon - expected).abs() < tolerances::PYTHON_PARITY,
            "Shannon(uniform 4) = {shannon}, expected ln(4) ≈ {expected}"
        );

        let simpson = result["simpson"].as_f64().expect("simpson value");
        assert!(
            (simpson - 0.75).abs() < tolerances::PYTHON_PARITY,
            "Simpson(uniform 4) = {simpson}, expected 0.75"
        );
    });
}

#[test]
fn health_readiness_reports_subsystems() {
    with_server("readiness", |socket| {
        let resp = rpc_roundtrip(
            socket,
            r#"{"jsonrpc":"2.0","method":"health.readiness","params":{},"id":2}"#,
        );

        assert_eq!(resp["jsonrpc"], "2.0");
        let result = &resp["result"];
        assert_eq!(result["ready"], true);
        assert_eq!(result["primal"], "wetspring");
        assert!(result["subsystems"]["math"].as_bool().unwrap_or(false));
        assert!(result["capabilities"].as_array().is_some());
    });
}

#[test]
fn capability_list_includes_all_domains() {
    with_server("cap_list", |socket| {
        let resp = rpc_roundtrip(
            socket,
            r#"{"jsonrpc":"2.0","method":"capability.list","params":{},"id":3}"#,
        );

        let result = &resp["result"];
        assert_eq!(result["primal"], "wetspring");

        let methods = result["methods"]
            .as_array()
            .expect("methods flat array (Wire Standard L2)");
        assert!(
            methods.len() >= 41,
            "expected 41+ methods, got {}",
            methods.len()
        );

        let provided = result["provided_capabilities"]
            .as_array()
            .expect("provided_capabilities array (Wire Standard L3)");
        assert!(provided.len() >= 15, "expected 15+ capability groups");

        let types: Vec<&str> = provided.iter().filter_map(|d| d["type"].as_str()).collect();
        assert!(types.contains(&"ecology.diversity"));
        assert!(types.contains(&"ecology.anderson"));
        assert!(types.contains(&"health"));
        assert!(types.contains(&"provenance"));
        assert!(types.contains(&"brain"));

        assert!(
            result["consumed_capabilities"].as_array().is_some(),
            "consumed_capabilities declared (Wire Standard L3)"
        );
    });
}

#[test]
fn composition_science_health_roundtrip() {
    with_server("comp_science", |socket| {
        let resp = rpc_roundtrip(
            socket,
            r#"{"jsonrpc":"2.0","method":"composition.science_health","params":{},"id":10}"#,
        );

        assert_eq!(resp["jsonrpc"], "2.0");
        assert!(resp.get("error").is_none(), "unexpected error: {resp}");

        let result = &resp["result"];
        assert_eq!(result["healthy"], true);
        assert_eq!(result["spring"], "wetSpring");
        assert!(result["deploy_graph"].as_str().is_some());
        assert!(result["subsystems"]["ipc"].as_bool().unwrap_or(false));
        assert!(result["subsystems"]["math"].as_bool().unwrap_or(false));
        assert!(result["science_domains"].as_array().is_some());
        assert!(result["capabilities_count"].as_u64().unwrap_or(0) > 30);
    });
}

#[test]
fn composition_nucleus_health_roundtrip() {
    with_server("comp_nucleus", |socket| {
        let resp = rpc_roundtrip(
            socket,
            r#"{"jsonrpc":"2.0","method":"composition.nucleus_health","params":{},"id":11}"#,
        );

        assert_eq!(resp["jsonrpc"], "2.0");
        assert!(resp.get("error").is_none(), "unexpected error: {resp}");

        let result = &resp["result"];
        assert_eq!(result["atomic"], "NUCLEUS");
        assert_eq!(result["spring"], "wetSpring");
        assert!(result["tiers"].is_object());
        assert!(result["components"].is_object());
        assert!(
            result["components"]["beardog"].is_string()
                || result["components"]["beardog"].is_object()
        );
    });
}

#[test]
fn composition_tower_health_roundtrip() {
    with_server("comp_tower", |socket| {
        let resp = rpc_roundtrip(
            socket,
            r#"{"jsonrpc":"2.0","method":"composition.tower_health","params":{},"id":12}"#,
        );

        let result = &resp["result"];
        assert_eq!(result["atomic"], "Tower");
        assert_eq!(result["spring"], "wetSpring");
        assert!(result.get("healthy").is_some());
        assert!(result["components"].is_object());
    });
}

#[test]
fn vault_store_roundtrip() {
    with_server("vault_store", |socket| {
        let resp = rpc_roundtrip(
            socket,
            r#"{"jsonrpc":"2.0","method":"vault.store","params":{"key":"test","value":"data"},"id":13}"#,
        );

        assert_eq!(resp["jsonrpc"], "2.0");
        let has_result = resp.get("result").is_some();
        let has_error = resp.get("error").is_some();
        assert!(
            has_result || has_error,
            "vault.store should return result or error"
        );
    });
}

#[test]
fn data_fetch_roundtrip() {
    with_server("data_fetch", |socket| {
        let resp = rpc_roundtrip(
            socket,
            r#"{"jsonrpc":"2.0","method":"data.fetch.chembl","params":{"chembl_id":"CHEMBL25"},"id":14}"#,
        );

        assert_eq!(resp["jsonrpc"], "2.0");
        let has_result = resp.get("result").is_some();
        let has_error = resp.get("error").is_some();
        assert!(
            has_result || has_error,
            "data.fetch.chembl should return result or error"
        );
    });
}

#[test]
fn qs_model_roundtrip() {
    with_server("qs_model", |socket| {
        let resp = rpc_roundtrip(
            socket,
            r#"{"jsonrpc":"2.0","method":"science.qs_model","params":{"scenario":"standard_growth","dt":0.01},"id":4}"#,
        );

        assert_eq!(resp["jsonrpc"], "2.0");
        assert!(
            resp.get("error").is_none(),
            "QS model should succeed: {resp}"
        );
    });
}
