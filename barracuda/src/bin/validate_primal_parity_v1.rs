// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::print_stdout,
    reason = "validation binary: stdout is the output medium"
)]
#![expect(
    clippy::expect_used,
    reason = "validation binary: expect is the pass/fail mechanism"
)]
#![expect(
    clippy::too_many_lines,
    reason = "validation binary: sequential primal parity checks in single main()"
)]
//! # Exp403: Primal Parity — Live NUCLEUS IPC vs Local Rust Baselines
//!
//! **Level 5: The Primal Proof.**
//!
//! This is the Tier 2 (IPC-WIRED) validation harness. It calls barraCuda,
//! NestGate, Squirrel, and BearDog primals **over live UDS sockets** and
//! compares results against local Rust baselines. When a primal socket is
//! absent, the check SKIPs (honest degradation for CI).
//!
//! The three-tier pattern (primalSpring composition guidance):
//!   Tier 1: LOCAL_CAPABILITIES — in-process dispatch (Exp401/402)
//!   Tier 2: IPC-WIRED — live primal calls with check_skip ← THIS BINARY
//!   Tier 3: FULL NUCLEUS — proto-nucleate from plasmidBin ecobins
//!
//! ## Downstream Manifest (validation_capabilities)
//!
//! From `primalSpring/graphs/downstream/downstream_manifest.toml`:
//!   tensor.matmul, stats.mean, compute.dispatch,
//!   storage.store, storage.retrieve, inference.complete, crypto.hash
//!
//! ## Domains
//!
//! | Domain | What |
//! |--------|------|
//! | D01 | barraCuda math parity (stats.mean, tensor.matmul, stats.std_dev) |
//! | D02 | barraCuda infrastructure (health, capabilities, identity) |
//! | D03 | NestGate storage parity (storage.store, storage.retrieve) |
//! | D04 | Squirrel AI (inference.complete) |
//! | D05 | BearDog crypto (crypto.hash) |
//! | D06 | toadStool compute (compute.dispatch) |
//!
//! ## Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline | Local Rust math + Python baselines |
//! | Script | `validate_primal_parity_v1.rs` |
//! | Date | 2026-04-17 |
//! | Command | `cargo run --features ipc --bin validate_primal_parity_v1` |

use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::time::Duration;
use wetspring_barracuda::ipc::discover;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

const RPC_TIMEOUT: Duration = Duration::from_secs(10);

fn rpc_call(socket: &Path, method: &str, params: &Value) -> Result<Value, String> {
    let request = json!({
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": 1
    });
    let request_line = serde_json::to_string(&request).map_err(|e| format!("serialize: {e}"))?;

    let stream =
        UnixStream::connect(socket).map_err(|e| format!("connect {}: {e}", socket.display()))?;
    stream.set_read_timeout(Some(RPC_TIMEOUT)).ok();
    stream.set_write_timeout(Some(RPC_TIMEOUT)).ok();

    let mut writer = std::io::BufWriter::new(&stream);
    writer
        .write_all(request_line.as_bytes())
        .map_err(|e| format!("write: {e}"))?;
    writer
        .write_all(b"\n")
        .map_err(|e| format!("write newline: {e}"))?;
    writer.flush().map_err(|e| format!("flush: {e}"))?;

    let mut reader = BufReader::new(&stream);
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .map_err(|e| format!("read: {e}"))?;

    if line.is_empty() {
        return Err("empty response from primal".to_string());
    }

    let response: Value = serde_json::from_str(&line).map_err(|e| format!("parse: {e}"))?;

    if let Some(err) = response.get("error") {
        return Err(format!("RPC error: {err}"));
    }

    response
        .get("result")
        .cloned()
        .ok_or_else(|| "no result field in response".to_string())
}

fn check_skip(label: &str, socket: Option<&PathBuf>, primal: &str) -> bool {
    if socket.is_none() {
        println!("    [SKIP] {label} — {primal} socket not found");
        return false;
    }
    true
}

fn main() {
    let mut v = Validator::new("Exp403: Primal Parity — Live NUCLEUS IPC vs Local Rust");

    let barracuda_socket = discover::discover_primal("barracuda");
    let nestgate_socket = discover::discover_primal("nestgate");
    let squirrel_socket = discover::discover_primal("squirrel");
    let beardog_socket = discover::discover_primal("beardog");
    let toadstool_socket = discover::discover_primal("toadstool");

    println!("  Socket discovery:");
    println!(
        "    barraCuda:  {}",
        barracuda_socket
            .as_ref()
            .map_or("not found (SKIP)", |p| p.to_str().unwrap_or("?"))
    );
    println!(
        "    NestGate:   {}",
        nestgate_socket
            .as_ref()
            .map_or("not found (SKIP)", |p| p.to_str().unwrap_or("?"))
    );
    println!(
        "    Squirrel:   {}",
        squirrel_socket
            .as_ref()
            .map_or("not found (SKIP)", |p| p.to_str().unwrap_or("?"))
    );
    println!(
        "    BearDog:    {}",
        beardog_socket
            .as_ref()
            .map_or("not found (SKIP)", |p| p.to_str().unwrap_or("?"))
    );
    println!(
        "    toadStool:  {}",
        toadstool_socket
            .as_ref()
            .map_or("not found (SKIP)", |p| p.to_str().unwrap_or("?"))
    );

    let mut any_primal_found = false;

    // ═══════════════════════════════════════════════════════════════
    // D01: barraCuda Math Parity (validation_capabilities: stats.mean, tensor.matmul)
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D01: barraCuda Math Parity ═══");

    if check_skip("barraCuda math", barracuda_socket.as_ref(), "barracuda") {
        any_primal_found = true;
        let sock = barracuda_socket.as_ref().expect("checked above");

        // D01a: stats.mean — validation_capability from downstream manifest
        let local_mean = 30.0; // mean([10, 20, 30, 40, 50]) = 150/5
        match rpc_call(sock, "stats.mean", &json!({"data": [10.0, 20.0, 30.0, 40.0, 50.0]})) {
            Ok(result) => {
                let ipc_mean = result["value"]
                    .as_f64()
                    .or_else(|| result["mean"].as_f64())
                    .or_else(|| result.as_f64())
                    .unwrap_or(f64::NAN);
                v.check(
                    "primal parity: stats.mean([10,20,30,40,50]) IPC vs local",
                    ipc_mean,
                    local_mean,
                    tolerances::ANALYTICAL_F64,
                );
            }
            Err(e) => {
                println!("    [FAIL] stats.mean dispatch error: {e}");
                v.check_pass("primal parity: stats.mean reachable", false);
            }
        }

        // D01b: stats.std_dev
        // std_dev([10,20,30,40,50]) = sqrt(200) = 10*sqrt(2)
        let local_std = (200.0_f64).sqrt();
        match rpc_call(
            sock,
            "stats.std_dev",
            &json!({"data": [10.0, 20.0, 30.0, 40.0, 50.0]}),
        ) {
            Ok(result) => {
                let ipc_std = result["value"]
                    .as_f64()
                    .or_else(|| result["std_dev"].as_f64())
                    .or_else(|| result.as_f64())
                    .unwrap_or(f64::NAN);
                v.check(
                    "primal parity: stats.std_dev IPC vs local",
                    ipc_std,
                    local_std,
                    tolerances::ANALYTICAL_F64,
                );
            }
            Err(e) => {
                println!("    [FAIL] stats.std_dev dispatch error: {e}");
                v.check_pass("primal parity: stats.std_dev reachable", false);
            }
        }

        // D01c: stats.weighted_mean
        match rpc_call(
            sock,
            "stats.weighted_mean",
            &json!({"data": [1.0, 2.0, 3.0], "weights": [0.5, 0.3, 0.2]}),
        ) {
            Ok(result) => {
                let ipc_wm = result["value"]
                    .as_f64()
                    .or_else(|| result.as_f64())
                    .unwrap_or(f64::NAN);
                let local_wm = 3.0_f64.mul_add(0.2, 1.0_f64.mul_add(0.5, 2.0 * 0.3));
                v.check(
                    "primal parity: stats.weighted_mean IPC vs local",
                    ipc_wm,
                    local_wm,
                    tolerances::ANALYTICAL_F64,
                );
            }
            Err(e) => {
                println!("    [FAIL] stats.weighted_mean dispatch error: {e}");
                v.check_pass("primal parity: stats.weighted_mean reachable", false);
            }
        }

        // D01d: tensor.matmul — validation_capability from downstream manifest
        match rpc_call(
            sock,
            "tensor.matmul",
            &json!({
                "a": {"data": [1.0, 2.0, 3.0, 4.0], "shape": [2, 2]},
                "b": {"data": [5.0, 6.0, 7.0, 8.0], "shape": [2, 2]}
            }),
        ) {
            Ok(result) => {
                let data = result["data"]
                    .as_array()
                    .or_else(|| result["result"].as_array())
                    .or_else(|| result.as_array());
                if let Some(arr) = data {
                    let vals: Vec<f64> = arr.iter().filter_map(Value::as_f64).collect();
                    let expected = [19.0, 22.0, 43.0, 50.0];
                    if vals.len() == 4 {
                        for (i, (&got, &exp)) in vals.iter().zip(expected.iter()).enumerate() {
                            v.check(
                                &format!("primal parity: tensor.matmul[{i}] IPC vs local"),
                                got,
                                exp,
                                tolerances::ANALYTICAL_F64,
                            );
                        }
                    } else {
                        v.check_pass("primal parity: tensor.matmul shape correct", false);
                    }
                } else {
                    v.check_pass("primal parity: tensor.matmul returns data array", false);
                }
            }
            Err(e) => {
                println!("    [FAIL] tensor.matmul dispatch error: {e}");
                v.check_pass("primal parity: tensor.matmul reachable", false);
            }
        }

        // D01e: rng.uniform (barraCuda RNG service)
        match rpc_call(sock, "rng.uniform", &json!({"n": 5, "low": 0.0, "high": 1.0})) {
            Ok(result) => {
                let samples = result["values"]
                    .as_array()
                    .or_else(|| result["data"].as_array())
                    .or_else(|| result.as_array());
                if let Some(arr) = samples {
                    let all_in_range = arr.iter().all(|v| {
                        v.as_f64()
                            .is_some_and(|x| (0.0..=1.0).contains(&x))
                    });
                    v.check_pass("primal parity: rng.uniform values in [0,1]", all_in_range);
                    v.check_pass("primal parity: rng.uniform returns 5 values", arr.len() == 5);
                } else {
                    v.check_pass("primal parity: rng.uniform returns array", false);
                }
            }
            Err(e) => {
                println!("    [FAIL] rng.uniform dispatch error: {e}");
                v.check_pass("primal parity: rng.uniform reachable", false);
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // D02: barraCuda Infrastructure (health, capabilities, identity)
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D02: barraCuda Infrastructure ═══");

    if check_skip("barraCuda infra", barracuda_socket.as_ref(), "barracuda") {
        let sock = barracuda_socket.as_ref().expect("checked above");

        match rpc_call(sock, "health.liveness", &json!({})) {
            Ok(result) => {
                v.check_pass(
                    "primal infra: barraCuda health.liveness alive",
                    result["alive"].as_bool() == Some(true),
                );
            }
            Err(e) => {
                println!("    [FAIL] health.liveness: {e}");
                v.check_pass("primal infra: barraCuda reachable", false);
            }
        }

        match rpc_call(sock, "capabilities.list", &json!({})) {
            Ok(result) => {
                let methods = result["methods"]
                    .as_array()
                    .or_else(|| result.as_array());
                v.check_pass(
                    "primal infra: barraCuda capabilities.list returns methods",
                    methods.is_some_and(|m| !m.is_empty()),
                );
            }
            Err(e) => {
                println!("    [FAIL] capabilities.list: {e}");
                v.check_pass("primal infra: capabilities.list reachable", false);
            }
        }

        match rpc_call(sock, "identity.get", &json!({})) {
            Ok(result) => {
                v.check_pass(
                    "primal infra: barraCuda identity.get returns primal name",
                    result["primal"].as_str().is_some()
                        || result["name"].as_str().is_some(),
                );
            }
            Err(e) => {
                println!("    [FAIL] identity.get: {e}");
                v.check_pass("primal infra: identity.get reachable", false);
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // D03: NestGate Storage (validation_capabilities: storage.store, storage.retrieve)
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D03: NestGate Storage Parity ═══");

    if check_skip("NestGate storage", nestgate_socket.as_ref(), "nestgate") {
        any_primal_found = true;
        let sock = nestgate_socket.as_ref().expect("checked above");

        let test_key = "wetspring_primal_parity_test";
        let test_value = json!({"experiment": "Exp403", "timestamp": "2026-04-17"});

        match rpc_call(
            sock,
            "storage.store",
            &json!({"key": test_key, "value": test_value}),
        ) {
            Ok(result) => {
                v.check_pass(
                    "primal parity: storage.store accepted",
                    result.get("ok").is_some()
                        || result.get("stored").is_some()
                        || result.get("key").is_some(),
                );

                match rpc_call(sock, "storage.retrieve", &json!({"key": test_key})) {
                    Ok(ret) => {
                        v.check_pass(
                            "primal parity: storage.retrieve returns value",
                            ret.get("value").is_some() || ret.get("data").is_some(),
                        );
                    }
                    Err(e) => {
                        println!("    [FAIL] storage.retrieve: {e}");
                        v.check_pass("primal parity: storage.retrieve reachable", false);
                    }
                }
            }
            Err(e) => {
                println!("    [FAIL] storage.store: {e}");
                v.check_pass("primal parity: storage.store reachable", false);
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // D04: Squirrel AI (validation_capability: inference.complete)
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D04: Squirrel AI ═══");

    if check_skip("Squirrel AI", squirrel_socket.as_ref(), "squirrel") {
        any_primal_found = true;
        let sock = squirrel_socket.as_ref().expect("checked above");

        match rpc_call(
            sock,
            "inference.complete",
            &json!({"prompt": "What is Shannon diversity?", "max_tokens": 32}),
        ) {
            Ok(result) => {
                v.check_pass(
                    "primal parity: inference.complete returns text",
                    result["text"].as_str().is_some()
                        || result["completion"].as_str().is_some()
                        || result["content"].as_str().is_some(),
                );
            }
            Err(e) => {
                println!("    [FAIL] inference.complete: {e}");
                v.check_pass("primal parity: inference.complete reachable", false);
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // D05: BearDog Crypto (validation_capability: crypto.hash)
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D05: BearDog Crypto ═══");

    if check_skip("BearDog crypto", beardog_socket.as_ref(), "beardog") {
        any_primal_found = true;
        let sock = beardog_socket.as_ref().expect("checked above");

        match rpc_call(
            sock,
            "crypto.hash",
            &json!({"data": "wetspring_primal_parity_test", "algorithm": "blake3"}),
        ) {
            Ok(result) => {
                v.check_pass(
                    "primal parity: crypto.hash returns hash",
                    result["hash"].as_str().is_some()
                        || result["digest"].as_str().is_some(),
                );
            }
            Err(e) => {
                println!("    [FAIL] crypto.hash: {e}");
                v.check_pass("primal parity: crypto.hash reachable", false);
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // D06: toadStool Compute (validation_capability: compute.dispatch)
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D06: toadStool Compute ═══");

    if check_skip("toadStool compute", toadstool_socket.as_ref(), "toadstool") {
        any_primal_found = true;
        let sock = toadstool_socket.as_ref().expect("checked above");

        match rpc_call(sock, "compute.dispatch", &json!({"op": "noop"})) {
            Ok(result) => {
                v.check_pass(
                    "primal parity: compute.dispatch accepted",
                    result.is_object(),
                );
            }
            Err(e) => {
                println!("    [FAIL] compute.dispatch: {e}");
                v.check_pass("primal parity: compute.dispatch reachable", false);
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════

    let (passed, total) = v.counts();

    if total == 0 && !any_primal_found {
        println!("\n  No primal sockets discovered — all checks SKIPPED.");
        println!("  Set BARRACUDA_SOCKET, NESTGATE_SOCKET, etc. or deploy NUCLEUS.");
        println!("  Exit 2 = skipped (not a failure).");
        std::process::exit(2);
    }

    if total == 0 {
        println!("\n  Primals discovered but no checks ran. Exit 2.");
        std::process::exit(2);
    }

    println!("\n  Primal proof: {passed}/{total} checks across live NUCLEUS IPC");
    v.finish();
}
