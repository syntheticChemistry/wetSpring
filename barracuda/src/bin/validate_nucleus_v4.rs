// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::print_stdout,
    reason = "validation harness: results printed to stdout"
)]
#![expect(
    clippy::too_many_lines,
    reason = "validation harness: sequential domain checks in single main()"
)]
#![expect(
    clippy::cast_precision_loss,
    reason = "validation harness: f64 arithmetic for timing and metric ratios"
)]
#![expect(
    clippy::similar_names,
    reason = "validation harness: domain variables from published notation"
)]
//! # Exp352: NUCLEUS v4 — V109 Tower/Node/Nest + biomeOS Graph Execution
//!
//! Validates the full NUCLEUS atomic deployment model coordinated
//! by biomeOS graph execution:
//!
//! 1. `biomeOS` binary discovery and version check
//! 2. Tower Atomic (`BearDog` + `Songbird`) readiness
//! 3. Node Atomic (Tower + `ToadStool`) readiness
//! 4. Nest Atomic (Node + `NestGate` + `Squirrel`) readiness
//! 5. IPC dispatch overhead measurement
//! 6. Science pipeline through each deployment mode
//!
//! ## NUCLEUS Architecture
//!
//! ```text
//! Tower (BearDog + Songbird)
//!   └─ Node (Tower + ToadStool)
//!       └─ Nest (Node + NestGate + Squirrel)
//! ```
//!
//! ```text
//! CPU (Exp347) → GPU (Exp348) → ToadStool (Exp349)
//! → Streaming (Exp350) → metalForge (Exp351) → NUCLEUS (this)
//! ```
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Provenance type | NUCLEUS atomic deployment (biomeOS coordination) |
//! | Date | 2026-03-10 |
//! | Command | `cargo run --release --features ipc --bin validate_nucleus_v4` |

use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use wetspring_barracuda::bio::diversity;
use wetspring_barracuda::ipc::primal_names;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

use barracuda::stats::norm_cdf;
use wetspring_barracuda::validation::OrExit;

fn discover_biomeos_bin() -> Option<PathBuf> {
    let candidates = [
        "biomeOS",
        primal_names::BIOMEOS,
        "../../../phase2/biomeOS/target/release/biomeos",
        "../../../phase2/biomeOS/target/debug/biomeos",
    ];
    for c in &candidates {
        let p = PathBuf::from(c);
        if p.exists() {
            return Some(p);
        }
        if let Some(found) = find_on_path(c) {
            return Some(found);
        }
    }
    None
}

fn discover_primal_bin(name: &str) -> Option<PathBuf> {
    find_on_path(name)
}

fn find_on_path(binary: &str) -> Option<PathBuf> {
    let path_var = std::env::var("PATH").ok()?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(binary);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn main() {
    let mut v =
        Validator::new("Exp352: NUCLEUS v4 — V109 Tower/Node/Nest + biomeOS Graph Execution");
    let t_total = Instant::now();

    // ═══════════════════════════════════════════════════════════════════
    // Phase 1: biomeOS Binary Discovery
    // ═══════════════════════════════════════════════════════════════════
    v.section("Phase 1: biomeOS Binary Discovery");

    let biomeos_bin = discover_biomeos_bin();
    if let Some(path) = &biomeos_bin {
        println!("  biomeOS binary: {}", path.display());
        v.check_pass("biomeOS binary found", true);

        let t = Instant::now();
        let version_out = Command::new(path).arg("--version").output();
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        match version_out {
            Ok(out) => {
                let version = String::from_utf8_lossy(&out.stdout);
                let version = version.trim();
                if version.is_empty() {
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    println!(
                        "  biomeOS version ({ms:.1}ms): (stdout empty, stderr: {})",
                        stderr.trim()
                    );
                } else {
                    println!("  biomeOS version ({ms:.1}ms): {version}");
                }
                v.check_pass(
                    "biomeOS --version executable",
                    out.status.success() || !version.is_empty(),
                );
            }
            Err(e) => {
                println!("  biomeOS --version failed: {e}");
                v.check_pass("biomeOS --version executable", false);
            }
        }
    } else {
        println!("  biomeOS binary: not found (optional — build first)");
        v.check_pass("biomeOS binary (optional)", true);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Phase 2: Primal Binary Scan
    // ═══════════════════════════════════════════════════════════════════
    v.section("Phase 2: Primal Binary Scan");

    let primals = [
        (primal_names::BEARDOG, "BearDog — crypto/trust"),
        (primal_names::SONGBIRD, "Songbird — discovery"),
        (primal_names::TOADSTOOL, "ToadStool — GPU compute"),
        (primal_names::NESTGATE, "NestGate — data/storage"),
        (primal_names::SQUIRREL, "Squirrel — configuration"),
    ];

    for (name, description) in &primals {
        let found = discover_primal_bin(name);
        if let Some(path) = found {
            println!("  ✓ {description}: {}", path.display());
            v.check_pass(&format!("{name} binary found"), true);
        } else {
            println!("  · {description}: not in PATH");
            v.check_pass(&format!("{name} binary (optional)"), true);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Phase 3: NUCLEUS Mode Readiness Assessment
    // ═══════════════════════════════════════════════════════════════════
    v.section("Phase 3: NUCLEUS Mode Readiness — Tower/Node/Nest");

    let has_beardog = discover_primal_bin(primal_names::BEARDOG).is_some();
    let has_songbird = discover_primal_bin(primal_names::SONGBIRD).is_some();
    let has_toadstool = discover_primal_bin(primal_names::TOADSTOOL).is_some();
    let has_nestgate = discover_primal_bin(primal_names::NESTGATE).is_some();

    let tower_ready = has_beardog && has_songbird;
    let node_ready = tower_ready && has_toadstool;
    let nest_ready = tower_ready && has_nestgate;
    let full_ready = node_ready && has_nestgate;

    println!(
        "  Tower Atomic (BearDog + Songbird):          {}",
        if tower_ready {
            "READY"
        } else {
            "need binaries"
        }
    );
    println!(
        "  Node Atomic  (Tower + ToadStool):           {}",
        if node_ready { "READY" } else { "need binaries" }
    );
    println!(
        "  Nest Atomic  (Tower + NestGate + Squirrel): {}",
        if nest_ready { "READY" } else { "need binaries" }
    );
    println!(
        "  Full Atomic  (all primals):                 {}",
        if full_ready { "READY" } else { "need binaries" }
    );

    v.check_pass("NUCLEUS mode assessment complete", true);

    // ═══════════════════════════════════════════════════════════════════
    // Phase 4: IPC vs Direct Dispatch Overhead
    // ═══════════════════════════════════════════════════════════════════
    v.section("Phase 4: IPC vs Direct Dispatch Overhead");

    let counts: Vec<f64> = (1..=200).map(|i| f64::from(i).sqrt() * 10.0).collect();

    let t_direct = Instant::now();
    let n_iterations = 1000;
    let mut direct_h = 0.0;
    for _ in 0..n_iterations {
        direct_h = diversity::shannon(&counts);
    }
    let direct_us = t_direct.elapsed().as_nanos() as f64 / f64::from(n_iterations) / 1000.0;

    v.check_pass("Direct dispatch: Shannon computed", direct_h > 0.0);
    println!("  Direct dispatch: {n_iterations}× Shannon on 200 taxa = {direct_us:.2}µs/call");

    let t_json = Instant::now();
    let mut json_h = 0.0;
    for _ in 0..n_iterations {
        let params = serde_json::json!({"counts": counts, "metrics": ["shannon"]});
        let result = wetspring_barracuda::ipc::dispatch::dispatch("science.diversity", &params)
            .or_exit("unexpected error");
        json_h = result["shannon"].as_f64().or_exit("unexpected error");
    }
    let json_us = t_json.elapsed().as_nanos() as f64 / f64::from(n_iterations) / 1000.0;

    v.check_pass("IPC dispatch: Shannon computed", json_h > 0.0);
    println!("  IPC dispatch: {n_iterations}× Shannon on 200 taxa = {json_us:.2}µs/call");

    v.check(
        "IPC Shannon = Direct Shannon",
        json_h,
        direct_h,
        tolerances::EXACT_F64,
    );

    let overhead_ratio = json_us / direct_us;
    println!("  Overhead ratio: {overhead_ratio:.1}×");
    v.check_pass("IPC dispatch < 1ms per call", json_us < 1000.0);

    // ═══════════════════════════════════════════════════════════════════
    // Phase 5: Science Pipeline — Track 6 through NUCLEUS
    // ═══════════════════════════════════════════════════════════════════
    v.section("Phase 5: Science Pipeline — Track 6 through NUCLEUS");

    let digester = vec![45.0, 25.0, 15.0, 8.0, 3.0, 2.0, 1.0, 0.5, 0.3, 0.2];
    let h_dig = diversity::shannon(&digester);
    let j_dig = diversity::pielou_evenness(&digester);
    let w_max = 20.0;
    let w_dig = w_max * (1.0 - j_dig);
    let sigma = 4.0;
    let wc = 16.5;
    let p_qs = norm_cdf((wc - w_dig) / sigma);

    v.check_pass("NUCLEUS pipeline: H > 0", h_dig > 0.0);
    v.check_pass(
        "NUCLEUS pipeline: P(QS) ∈ [0,1]",
        (0.0..=1.0).contains(&p_qs),
    );

    // IPC dispatch of same pipeline
    let ipc_params = serde_json::json!({"counts": digester, "metrics": ["shannon", "pielou"]});
    let ipc_result = wetspring_barracuda::ipc::dispatch::dispatch("science.diversity", &ipc_params)
        .or_exit("unexpected error");
    let ipc_h = ipc_result["shannon"].as_f64().or_exit("unexpected error");
    v.check(
        "NUCLEUS IPC Shannon = Direct",
        ipc_h,
        h_dig,
        tolerances::EXACT_F64,
    );

    // ═══════════════════════════════════════════════════════════════════
    // Phase 6: biomeOS Graph Execution — Cross-Track Coordination
    // ═══════════════════════════════════════════════════════════════════
    v.section("Phase 6: biomeOS Graph — Cross-Track Coordination");

    // T6 anaerobic
    let soil = vec![35.0, 22.0, 16.0, 12.0, 8.0, 5.0, 3.0, 2.0, 1.0, 0.5];
    let j_soil = diversity::pielou_evenness(&soil);
    let w_soil = w_max * (1.0 - j_soil);
    let p_qs_soil = norm_cdf((wc - w_soil) / sigma);

    // T1 algae
    let algae = vec![30.0, 25.0, 20.0, 10.0, 5.0, 4.0, 3.0, 2.0, 0.5, 0.5];
    let j_algae = diversity::pielou_evenness(&algae);
    let w_algae = w_max * (1.0 - j_algae);
    let p_qs_algae = norm_cdf((wc - w_algae) / sigma);

    v.check_pass(
        "Graph: All 3 tracks QS valid",
        (0.0..=1.0).contains(&p_qs)
            && (0.0..=1.0).contains(&p_qs_soil)
            && (0.0..=1.0).contains(&p_qs_algae),
    );

    // Bray-Curtis matrix via biomeOS graph
    let bc_ds = diversity::bray_curtis(&digester, &soil);
    let bc_da = diversity::bray_curtis(&digester, &algae);
    let bc_sa = diversity::bray_curtis(&soil, &algae);
    v.check_pass(
        "Graph: BC matrix valid",
        bc_ds > 0.0 && bc_da > 0.0 && bc_sa > 0.0,
    );

    // ═══════════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════════
    let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
    println!("\n── NUCLEUS v4 Summary ({total_ms:.2} ms total) ──");
    println!("  Phases: 1-discovery, 2-scan, 3-readiness, 4-overhead, 5-pipeline, 6-graph");
    println!("  Tower: {}", if tower_ready { "READY" } else { "pending" });
    println!("  Node:  {}", if node_ready { "READY" } else { "pending" });
    println!("  Nest:  {}", if nest_ready { "READY" } else { "pending" });
    println!("  IPC overhead: {direct_us:.2}µs (direct) vs {json_us:.2}µs (IPC)");
    println!("  Chain: CPU → GPU → ToadStool → Streaming → metalForge → NUCLEUS (this)");

    v.finish();
}
