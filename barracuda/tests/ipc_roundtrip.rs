// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![cfg(feature = "ipc")]
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
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .ok();

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
            r#"{"jsonrpc":"2.0","method":"science.diversity","params":{"abundances":[0.25,0.25,0.25,0.25]},"id":1}"#,
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

        let domains = result["domains"].as_array().expect("domains array");
        assert!(
            domains.len() >= 15,
            "expected 15+ domains, got {}",
            domains.len()
        );

        let domain_names: Vec<&str> = domains
            .iter()
            .filter_map(|d| d["name"].as_str())
            .collect();
        assert!(domain_names.contains(&"ecology.diversity"));
        assert!(domain_names.contains(&"ecology.anderson"));
        assert!(domain_names.contains(&"health"));
        assert!(domain_names.contains(&"provenance"));
        assert!(domain_names.contains(&"brain"));
    });
}

#[test]
fn qs_model_roundtrip() {
    with_server("qs_model", |socket| {
        let resp = rpc_roundtrip(
            socket,
            r#"{"jsonrpc":"2.0","method":"science.qs_model","params":{"scenario":"standard","dt":0.01},"id":4}"#,
        );

        assert_eq!(resp["jsonrpc"], "2.0");
        assert!(
            resp.get("error").is_none(),
            "QS model should succeed: {resp}"
        );
    });
}
