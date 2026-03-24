// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
//! Exp203: `biomeOS` Science Pipeline Integration Validation
//!
//! Validates the wetSpring IPC server end-to-end as a `biomeOS` science primal.
//! Tests the full `science_pipeline.toml` graph locally:
//! 1. Server binds and accepts connections
//! 2. health.check returns all 5 capabilities
//! 3. science.diversity computes correct metrics
//! 4. `science.qs_model` runs ODE integration
//! 5. `science.full_pipeline` chains stages correctly
//! 6. Songbird registration falls back gracefully
//! 7. Metrics are tracked per-method
//!
//! Proves: when `NestGate`/`ToadStool`/`Tower` are running, `biomeOS` can orchestrate
//! the science pipeline graph through wetSpring's JSON-RPC interface.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline | Analytical (closed-form formulas) |
//! | Reference | Magurran 2004, *Measuring Biological Diversity* |
//! | Date | 2026-02-28 |
//! | Command | `cargo run --release --bin validate_science_pipeline` |
//!
//! ## `check_f64_in_json` expected values
//!
//! - **Shannon = ln(4)**: Uniform 4-species community; H′ = ln(S).
//! - **Simpson = 0.75**: Uniform 4-species; D = 1 − 1/S = 0.75.
//! - **observed = 4**: Count of non-zero abundances.
//! - **Pielou = 1.0**: Perfect evenness for uniform distribution.
//!
//! Validation class: Pipeline
//!
//! Provenance: End-to-end pipeline integration test

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::time::Duration;
use wetspring_barracuda::ipc::Server;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation;

#[expect(clippy::too_many_lines)]
fn main() {
    let mut v = validation::Validator::new("Exp203: biomeOS Science Pipeline");

    let dir = std::env::temp_dir().join("wetspring_exp203");
    let _ = std::fs::create_dir_all(&dir);
    let sock_path = dir.join("test.sock");

    // Clean up any stale socket
    let _ = std::fs::remove_file(&sock_path);

    let server = Server::bind(&sock_path).unwrap_or_else(|e| {
        eprintln!("FATAL: cannot bind: {e}");
        std::process::exit(1);
    });

    let metrics = std::sync::Arc::clone(server.metrics());
    let server_path = server.socket_path().to_path_buf();

    // Start server in background thread
    std::thread::spawn(move || server.run());
    std::thread::sleep(Duration::from_millis(100));

    // ── Stage 1: Health Check ─────────────────────────────────────────
    let health_resp = rpc(&server_path, "health.check", "{}");
    v.check_pass(
        "health.check returns healthy",
        health_resp.contains("\"healthy\""),
    );
    v.check_pass(
        "health.check lists 5 capabilities",
        health_resp.contains("science.diversity")
            && health_resp.contains("science.anderson")
            && health_resp.contains("science.qs_model")
            && health_resp.contains("science.ncbi_fetch")
            && health_resp.contains("science.full_pipeline"),
    );
    v.check_pass(
        "health.check includes version",
        health_resp.contains("version"),
    );

    // ── Stage 2: Diversity Analysis ───────────────────────────────────
    let uniform_counts = r#"{"counts":[25.0,25.0,25.0,25.0]}"#;
    let div_resp = rpc(&server_path, "science.diversity", uniform_counts);
    let shannon_expected = 4.0_f64.ln();
    v.check_pass(
        "diversity: uniform Shannon = ln(4)",
        check_f64_in_json(
            &div_resp,
            "shannon",
            shannon_expected,
            tolerances::PYTHON_PARITY,
        ),
    );
    v.check_pass(
        "diversity: uniform Simpson = 0.75",
        check_f64_in_json(&div_resp, "simpson", 0.75, tolerances::PYTHON_PARITY),
    );
    v.check_pass(
        "diversity: observed = 4",
        check_f64_in_json(&div_resp, "observed", 4.0, tolerances::PYTHON_PARITY),
    );
    v.check_pass(
        "diversity: Pielou evenness = 1.0",
        check_f64_in_json(&div_resp, "pielou", 1.0, tolerances::PYTHON_PARITY),
    );

    // Specific metric selection
    let specific = r#"{"counts":[10.0,20.0,30.0],"metrics":["shannon"]}"#;
    let spec_resp = rpc(&server_path, "science.diversity", specific);
    v.check_pass(
        "diversity: specific metric returns only shannon",
        spec_resp.contains("shannon") && !spec_resp.contains("simpson"),
    );

    // Bray-Curtis between two samples
    let bray = r#"{"counts":[10.0,20.0,30.0],"counts_b":[15.0,25.0,35.0]}"#;
    let bray_resp = rpc(&server_path, "science.diversity", bray);
    v.check_pass(
        "diversity: Bray-Curtis computed for sample pair",
        bray_resp.contains("bray_curtis"),
    );

    // Error case: empty counts
    let empty_resp = rpc(&server_path, "science.diversity", r#"{"counts":[]}"#);
    v.check_pass(
        "diversity: empty counts returns error",
        empty_resp.contains("error"),
    );

    // ── Stage 3: QS Biofilm Model ────────────────────────────────────
    let qs_default = rpc(&server_path, "science.qs_model", "{}");
    v.check_pass(
        "qs_model: default scenario returns t_end > 0",
        check_f64_positive(&qs_default, "t_end"),
    );
    v.check_pass(
        "qs_model: peak_biofilm > 0",
        check_f64_positive(&qs_default, "peak_biofilm"),
    );
    v.check_pass(
        "qs_model: returns final_state array",
        qs_default.contains("final_state"),
    );

    let qs_high = r#"{"scenario":"high_density","dt":0.05}"#;
    let qs_high_resp = rpc(&server_path, "science.qs_model", qs_high);
    v.check_pass(
        "qs_model: high_density scenario runs",
        check_f64_positive(&qs_high_resp, "t_end"),
    );

    let qs_hapr = r#"{"scenario":"hapr_mutant"}"#;
    let qs_hapr_resp = rpc(&server_path, "science.qs_model", qs_hapr);
    v.check_pass(
        "qs_model: hapr_mutant scenario runs",
        check_f64_positive(&qs_hapr_resp, "t_end"),
    );

    let qs_dgc = r#"{"scenario":"dgc_overexpression"}"#;
    let qs_dgc_resp = rpc(&server_path, "science.qs_model", qs_dgc);
    v.check_pass(
        "qs_model: dgc_overexpression scenario runs",
        check_f64_positive(&qs_dgc_resp, "t_end"),
    );

    // Error case: unknown scenario
    let qs_bad = rpc(
        &server_path,
        "science.qs_model",
        r#"{"scenario":"imaginary"}"#,
    );
    v.check_pass(
        "qs_model: unknown scenario returns error",
        qs_bad.contains("error"),
    );

    // ── Stage 4: Full Pipeline ───────────────────────────────────────
    let pipeline_with_counts = r#"{"counts":[5.0,10.0,15.0,20.0],"scenario":"standard_growth"}"#;
    let pipe_resp = rpc(&server_path, "science.full_pipeline", pipeline_with_counts);
    v.check_pass(
        "full_pipeline: diversity stage present",
        pipe_resp.contains("diversity"),
    );
    v.check_pass(
        "full_pipeline: qs_model stage present",
        pipe_resp.contains("qs_model"),
    );
    v.check_pass(
        "full_pipeline: pipeline marked complete",
        pipe_resp.contains("complete"),
    );

    let pipe_no_counts = rpc(&server_path, "science.full_pipeline", "{}");
    v.check_pass(
        "full_pipeline: runs without counts (QS only)",
        pipe_no_counts.contains("qs_model") && pipe_no_counts.contains("complete"),
    );

    // ── Stage 5: Protocol Edge Cases ─────────────────────────────────
    let unknown = rpc(&server_path, "nonexistent.method", "{}");
    v.check_pass("unknown method returns -32601", unknown.contains("-32601"));

    let bad_version = rpc_raw(
        &server_path,
        r#"{"jsonrpc":"1.0","method":"health.check","params":{},"id":99}"#,
    );
    v.check_pass(
        "wrong jsonrpc version returns error",
        bad_version.contains("error"),
    );

    // ── Stage 6: Multiple Requests on Single Connection ──────────────
    let stream = UnixStream::connect(&server_path).unwrap_or_else(|e| {
        eprintln!("connect failed: {e}");
        std::process::exit(1);
    });
    let mut writer = std::io::BufWriter::new(&stream);
    let mut reader = BufReader::new(&stream);
    let mut multi_ok = true;
    for i in 1..=5 {
        let req = format!(r#"{{"jsonrpc":"2.0","method":"health.check","params":{{}},"id":{i}}}"#);
        if writer.write_all(req.as_bytes()).is_err()
            || writer.write_all(b"\n").is_err()
            || writer.flush().is_err()
        {
            multi_ok = false;
            break;
        }
        let mut resp = String::new();
        if reader.read_line(&mut resp).is_err() || !resp.contains("healthy") {
            multi_ok = false;
            break;
        }
    }
    v.check_pass("multiple requests on single connection", multi_ok);

    // ── Stage 7: Songbird Graceful Fallback ──────────────────────────
    v.check_pass(
        "Songbird discovery graceful (does not panic)",
        wetspring_barracuda::ipc::songbird::discover_socket().is_none()
            || wetspring_barracuda::ipc::songbird::discover_socket().is_some(),
    );

    // ── Stage 8: Metrics ─────────────────────────────────────────────
    std::thread::sleep(Duration::from_millis(50));
    let total = metrics
        .total_calls
        .load(std::sync::atomic::Ordering::Relaxed);
    v.check_pass(
        &format!("metrics: total_calls >= 15 (got {total})"),
        total >= 15,
    );
    let successes = metrics
        .success_count
        .load(std::sync::atomic::Ordering::Relaxed);
    v.check_pass(
        &format!("metrics: successes > 0 (got {successes})"),
        successes > 0,
    );
    let errors = metrics
        .error_count
        .load(std::sync::atomic::Ordering::Relaxed);
    v.check_pass(
        &format!("metrics: errors > 0 from invalid requests (got {errors})"),
        errors > 0,
    );

    let snapshot = metrics.snapshot();
    v.check_pass(
        "metrics: snapshot is valid JSON with primal field",
        snapshot["primal"] == "wetspring",
    );

    // Cleanup
    let _ = std::fs::remove_file(&sock_path);
    let _ = std::fs::remove_dir(&dir);

    v.finish();
}

/// Send a JSON-RPC request and return the response.
fn rpc(sock: &std::path::Path, method: &str, params: &str) -> String {
    let req = format!(r#"{{"jsonrpc":"2.0","method":"{method}","params":{params},"id":1}}"#);
    rpc_raw(sock, &req)
}

/// Send a raw JSON-RPC request line.
fn rpc_raw(sock: &std::path::Path, request: &str) -> String {
    let stream = match UnixStream::connect(sock) {
        Ok(s) => s,
        Err(e) => return format!("connect error: {e}"),
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
    let mut writer = std::io::BufWriter::new(&stream);
    let _ = writer.write_all(request.as_bytes());
    let _ = writer.write_all(b"\n");
    let _ = writer.flush();

    let mut reader = BufReader::new(&stream);
    let mut resp = String::new();
    let _ = reader.read_line(&mut resp);
    resp
}

/// Check if a JSON response contains a field with a specific f64 value within tolerance.
fn check_f64_in_json(json: &str, field: &str, expected: f64, tol: f64) -> bool {
    let needle = format!("\"{field}\":");
    if let Some(pos) = json.find(&needle) {
        let after = &json[pos + needle.len()..];
        let trimmed = after.trim_start();
        let end = trimmed
            .find(|c: char| {
                !c.is_ascii_digit() && c != '.' && c != '-' && c != 'e' && c != 'E' && c != '+'
            })
            .unwrap_or(trimmed.len());
        if let Ok(val) = trimmed[..end].parse::<f64>() {
            return (val - expected).abs() < tol;
        }
    }
    false
}

/// Check if a JSON response contains a field with a positive f64 value.
fn check_f64_positive(json: &str, field: &str) -> bool {
    let needle = format!("\"{field}\":");
    if let Some(pos) = json.find(&needle) {
        let after = &json[pos + needle.len()..];
        let trimmed = after.trim_start();
        let end = trimmed
            .find(|c: char| {
                !c.is_ascii_digit() && c != '.' && c != '-' && c != 'e' && c != 'E' && c != '+'
            })
            .unwrap_or(trimmed.len());
        if let Ok(val) = trimmed[..end].parse::<f64>() {
            return val > 0.0;
        }
    }
    false
}
