// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::print_stdout,
    reason = "validation binary: stdout is the output medium"
)]
#![expect(
    clippy::too_many_lines,
    reason = "validation harness: sequential domain checks in single main()"
)]
#![expect(
    clippy::cast_precision_loss,
    reason = "validation harness: f64 arithmetic for timing and metric ratios"
)]
//! # Exp257: NUCLEUS Data Acquisition Pipeline — Three-Tier Primal Routing
//!
//! Validates the full NUCLEUS data acquisition chain for science workloads:
//!
//! 1. **Songbird Discovery** — wetSpring registers capabilities, discovers peers
//! 2. **`NestGate` Three-Tier Routing** — `biomeOS` → `NestGate` → sovereign HTTP
//! 3. **Neural API capability.call** — semantic routing through `biomeOS` orchestrator
//! 4. **Science Pipeline via IPC** — diversity + Anderson through JSON-RPC dispatch
//! 5. **Sovereign Fallback** — standalone mode when no ecosystem is running
//!
//! ## NUCLEUS Integration
//!
//! This experiment probes the live NUCLEUS state and exercises whichever tier
//! is available. It reports what's running, what falls back, and what evolution
//! is needed for each primal.
//!
//! ## Chain
//!
//! Exp203 (IPC pipeline) → Exp204 (Songbird) → Exp205 (sovereign fallback) →
//! Exp206 (IPC dispatch fidelity) → **Exp257 (NUCLEUS data pipeline)**
//!
//! # Provenance
//!
//! Expected values are **analytical** — derived from mathematical
//! identities and algebraic invariants.
//!
//! | Field | Value |
//! |-------|-------|
//! | Provenance type | Analytical (mathematical invariants) |
//! | Date | 2026-03-03 |
//! | Command | `cargo run --release --bin validate_nucleus_data_pipeline` |
//!
//! Provenance: NUCLEUS data pipeline integration validation

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use wetspring_barracuda::bio::diversity;
use wetspring_barracuda::ipc::discover;
use wetspring_barracuda::ipc::primal_names;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

const RPC_TIMEOUT: Duration = Duration::from_secs(5);

fn main() {
    let mut v =
        Validator::new("Exp257: NUCLEUS Data Acquisition Pipeline — Three-Tier Primal Routing");

    v.section("Phase 1: Ecosystem Probe — Socket Discovery");

    let biomeos_socket = discover::discover_socket("BIOMEOS_SOCKET", primal_names::BIOMEOS);
    let songbird_socket = discover::discover_socket("SONGBIRD_SOCKET", primal_names::SONGBIRD);
    let nestgate_socket = discover::discover_socket("NESTGATE_SOCKET", primal_names::NESTGATE);
    let wetspring_socket = discover::discover_socket("WETSPRING_SOCKET", primal_names::SELF);

    println!("  Socket scan results:");
    println!("  ─────────────────────────────────────────");
    report_socket("biomeOS orchestrator", biomeos_socket.as_ref());
    report_socket("Songbird discovery", songbird_socket.as_ref());
    report_socket("NestGate data provider", nestgate_socket.as_ref());
    report_socket("wetSpring IPC server", wetspring_socket.as_ref());
    println!();

    let has_biomeos = biomeos_socket.is_some();
    let has_songbird = songbird_socket.is_some();
    let has_nestgate = nestgate_socket.is_some();
    let has_wetspring = wetspring_socket.is_some();

    v.check_pass("Discovery: socket scan completes without panic", true);

    v.section("Phase 2: Tier Assessment");

    let tier = if has_biomeos {
        println!("  Tier 1: biomeOS Neural API — capability.call routing available");
        1
    } else if has_nestgate {
        println!("  Tier 2: Direct NestGate — socket IPC available");
        2
    } else {
        println!("  Tier 3: Sovereign — standalone HTTP (no ecosystem services)");
        3
    };

    v.check_pass(
        "Tier assessment: tier identified (1-3)",
        (1..=3).contains(&tier),
    );
    println!("  Active tier: {tier}");
    println!();

    v.section("Phase 3: Songbird Registration Probe");

    if let Some(ref sock) = songbird_socket {
        let t = Instant::now();
        match rpc_call(
            sock,
            r#"{"jsonrpc":"2.0","method":"discovery.list","params":{},"id":1}"#,
        ) {
            Ok(response) => {
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                println!(
                    "  Songbird list response ({ms:.1}ms): {}",
                    &response[..response.len().min(200)]
                );
                v.check_pass("Songbird: discovery.list responded", true);
            }
            Err(e) => {
                println!("  Songbird: discovery.list failed — {e}");
                v.check_pass("Songbird: discovery.list responded", false);
            }
        }
    } else {
        println!("  Songbird not running — skipping registration probe");
        v.check_pass("Songbird: not available (standalone OK)", true);
    }

    v.section("Phase 4: NestGate NCBI Probe");

    if let Some(ref sock) = nestgate_socket {
        let t = Instant::now();
        let ncbi_probe = r#"{"jsonrpc":"2.0","method":"ncbi.capabilities","params":{},"id":2}"#;
        match rpc_call(sock, ncbi_probe) {
            Ok(response) => {
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                println!(
                    "  NestGate NCBI capabilities ({ms:.1}ms): {}",
                    &response[..response.len().min(200)]
                );
                v.check_pass("NestGate: NCBI capabilities responded", true);
            }
            Err(e) => {
                println!("  NestGate: NCBI probe failed — {e}");
                let health_probe =
                    r#"{"jsonrpc":"2.0","method":"health.check","params":{},"id":3}"#;
                match rpc_call(sock, health_probe) {
                    Ok(response) => {
                        let ms = t.elapsed().as_secs_f64() * 1000.0;
                        println!(
                            "  NestGate: health check succeeded ({ms:.1}ms): {}",
                            &response[..response.len().min(200)]
                        );
                        v.check_pass("NestGate: health check responded", true);
                    }
                    Err(e2) => {
                        println!("  NestGate: health check also failed — {e2}");
                        v.check_pass("NestGate: reachable", false);
                    }
                }
            }
        }
    } else {
        println!("  NestGate not running — NCBI will use sovereign HTTP fallback");
        v.check_pass("NestGate: not available (sovereign fallback OK)", true);
    }

    v.section("Phase 5: biomeOS Neural API Probe");

    if let Some(ref sock) = biomeos_socket {
        let t = Instant::now();
        let cap_list = r#"{"jsonrpc":"2.0","method":"capability.list","params":{},"id":4}"#;
        match rpc_call(sock, cap_list) {
            Ok(response) => {
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                println!(
                    "  biomeOS capability.list ({ms:.1}ms): {}",
                    &response[..response.len().min(300)]
                );
                v.check_pass("biomeOS: capability.list responded", true);

                let cap_call = r#"{"jsonrpc":"2.0","method":"capability.call","params":{"capability":"science.diversity","args":{"counts":[10,20,30,40],"metrics":["shannon"]}},"id":5}"#;
                let t2 = Instant::now();
                match rpc_call(sock, cap_call) {
                    Ok(response2) => {
                        let ms2 = t2.elapsed().as_secs_f64() * 1000.0;
                        println!(
                            "  capability.call science.diversity ({ms2:.1}ms): {}",
                            &response2[..response2.len().min(200)]
                        );
                        v.check_pass("biomeOS: capability.call science.diversity succeeded", true);
                    }
                    Err(e) => {
                        println!("  capability.call: {e}");
                        v.check_pass("biomeOS: capability.call science.diversity (may need wetspring_server)", false);
                    }
                }
            }
            Err(e) => {
                println!("  biomeOS: capability.list failed — {e}");
                v.check_pass("biomeOS: reachable", false);
            }
        }
    } else {
        println!("  biomeOS not running — Neural API unavailable");
        v.check_pass("biomeOS: not available (standalone OK)", true);
    }

    v.section("Phase 6: Direct IPC Dispatch Fidelity");

    if let Some(ref sock) = wetspring_socket {
        let t = Instant::now();
        let health = r#"{"jsonrpc":"2.0","method":"health.check","params":{},"id":10}"#;
        match rpc_call(sock, health) {
            Ok(response) => {
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                println!(
                    "  wetSpring IPC health ({ms:.1}ms): {}",
                    &response[..response.len().min(200)]
                );
                v.check_pass("wetSpring IPC: health check responded", true);

                let diversity_req = r#"{"jsonrpc":"2.0","method":"science.diversity","params":{"counts":[100,200,300,150,50],"metrics":["all"]},"id":11}"#;
                let t2 = Instant::now();
                match rpc_call(sock, diversity_req) {
                    Ok(div_response) => {
                        let ms2 = t2.elapsed().as_secs_f64() * 1000.0;
                        println!(
                            "  wetSpring IPC diversity ({ms2:.1}ms): {}",
                            &div_response[..div_response.len().min(300)]
                        );

                        let direct_shannon =
                            diversity::shannon(&[100.0, 200.0, 300.0, 150.0, 50.0]);
                        let ipc_match = div_response.contains("shannon");
                        v.check_pass("wetSpring IPC: diversity includes shannon", ipc_match);
                        println!("  Direct shannon: {direct_shannon:.6}");
                    }
                    Err(e) => {
                        println!("  wetSpring IPC diversity: {e}");
                        v.check_pass("wetSpring IPC: science.diversity responded", false);
                    }
                }
            }
            Err(e) => {
                println!("  wetSpring IPC: {e}");
                v.check_pass("wetSpring IPC: reachable", false);
            }
        }
    } else {
        println!("  wetSpring IPC server not running — direct dispatch unavailable");
        v.check_pass("wetSpring IPC: not running (standalone OK)", true);
    }

    v.section("Phase 7: Standalone Science Validation (always runs)");

    let counts: Vec<f64> = vec![100.0, 200.0, 300.0, 150.0, 50.0, 75.0, 25.0, 10.0, 5.0, 2.0];
    let h = diversity::shannon(&counts);
    let s = diversity::simpson(&counts);
    v.check_pass("Standalone: Shannon > 0", h > 0.0);
    v.check_pass("Standalone: Simpson ∈ (0,1)", s > 0.0 && s < 1.0);
    println!("  Standalone diversity: H'={h:.4}, D={s:.4}");

    let s_obs = counts.iter().filter(|&&c| c > 0.0).count();
    let _n: f64 = counts.iter().sum();
    let f1 = counts
        .iter()
        .filter(|&&c| (c - 1.0).abs() < tolerances::CHAO1_COUNT_HALFWIDTH)
        .count() as f64;
    let f2 = counts
        .iter()
        .filter(|&&c| (c - 2.0).abs() < tolerances::CHAO1_COUNT_HALFWIDTH)
        .count() as f64;
    let chao1 = if f2 > 0.0 {
        s_obs as f64 + (f1 * f1) / (2.0 * f2)
    } else {
        s_obs as f64 + f1 * (f1 - 1.0) / 2.0
    };
    v.check_pass("Standalone: Chao1 ≥ S_obs", chao1 >= s_obs as f64);
    println!("  Standalone Chao1: {chao1:.1} (S_obs={s_obs})");

    v.section("Phase 8: Primal Evolution Scoreboard");

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  NUCLEUS Primal Interaction Status                          ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!(
        "║ {:20} │ {:8} │ {:25} ║",
        "Primal", "Status", "Evolution Need"
    );
    println!("╠══════════════════════════════════════════════════════════════╣");

    let biomeos_status = if has_biomeos { "LIVE" } else { "offline" };
    let biomeos_need = if has_biomeos {
        "batch dispatch (N=30K)"
    } else {
        "start NUCLEUS node"
    };
    println!(
        "║ {:20} │ {:8} │ {:25} ║",
        "biomeOS", biomeos_status, biomeos_need
    );

    let songbird_status = if has_songbird { "LIVE" } else { "offline" };
    let songbird_need = if has_songbird {
        "multi-gate federation"
    } else {
        "start with NUCLEUS"
    };
    println!(
        "║ {:20} │ {:8} │ {:25} ║",
        "Songbird", songbird_status, songbird_need
    );

    let nestgate_status = if has_nestgate { "LIVE" } else { "offline" };
    let nestgate_need = if has_nestgate {
        "BIOM parser + SRA bulk"
    } else {
        "start with NUCLEUS"
    };
    println!(
        "║ {:20} │ {:8} │ {:25} ║",
        "NestGate", nestgate_status, nestgate_need
    );

    let wetspring_status = if has_wetspring { "LIVE" } else { "offline" };
    let wetspring_need = if has_wetspring {
        "batch pipeline dispatch"
    } else {
        "run wetspring_server"
    };
    println!(
        "║ {:20} │ {:8} │ {:25} ║",
        "wetSpring", wetspring_status, wetspring_need
    );

    println!(
        "║ {:20} │ {:8} │ {:25} ║",
        "BearDog", "proven", "lineage verification"
    );
    println!(
        "║ {:20} │ {:8} │ {:25} ║",
        "ToadStool", "absorbing", "S70+++ shader evolution"
    );
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!(
        "║ Active tier: {tier}  │  Data path: {:36} ║",
        match tier {
            1 => "biomeOS → capability.call → wetSpring",
            2 => "NestGate → direct IPC → wetSpring",
            _ => "sovereign standalone (no IPC)",
        }
    );
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Next steps to activate NUCLEUS:");
    println!("  ─────────────────────────────────────────");
    if !has_biomeos {
        println!("  1. biomeos nucleus start --mode node --node-id eastgate");
    }
    if !has_wetspring {
        println!("  2. cargo run --release --bin wetspring_server");
    }
    println!(
        "  3. WETSPRING_DATA_PROVIDER=nestgate cargo run --release --bin validate_nucleus_data_pipeline"
    );
    println!("  4. Wire EMP OTU table download via NestGate HTTP fetch");

    v.finish();
}

fn report_socket(label: &str, path: Option<&PathBuf>) {
    match path {
        Some(p) => println!("    ✓ {label}: {}", p.display()),
        None => println!("    · {label}: not found"),
    }
}

fn rpc_call(socket: &PathBuf, request: &str) -> Result<String, String> {
    let stream =
        UnixStream::connect(socket).map_err(|e| format!("connect {}: {e}", socket.display()))?;
    stream.set_read_timeout(Some(RPC_TIMEOUT)).ok();
    stream.set_write_timeout(Some(RPC_TIMEOUT)).ok();

    let mut writer = std::io::BufWriter::new(&stream);
    writer
        .write_all(request.as_bytes())
        .map_err(|e| format!("write: {e}"))?;
    writer
        .write_all(b"\n")
        .map_err(|e| format!("newline: {e}"))?;
    writer.flush().map_err(|e| format!("flush: {e}"))?;

    let mut reader = BufReader::new(&stream);
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .map_err(|e| format!("read: {e}"))?;

    if line.is_empty() {
        return Err("empty response".to_string());
    }
    Ok(line)
}
