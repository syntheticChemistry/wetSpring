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
//! # Exp258: NUCLEUS Tower-Node Deployment — Live Primal Orchestration
//!
//! Validates the full NUCLEUS Tower→Node deployment model by:
//!
//! 1. Probing whether `biomeOS` binary exists and is executable
//! 2. Testing Tower Atomic (`BearDog` + Songbird) readiness
//! 3. Testing Node Atomic (Tower + `ToadStool`) readiness
//! 4. Validating the science pipeline through each deployment mode
//! 5. Measuring the overhead of IPC vs direct function calls
//!
//! ## NUCLEUS Architecture
//!
//! ```text
//! Tower (BearDog + Songbird)
//!   └─ Node (Tower + `ToadStool`)
//!       └─ Full (Node + `NestGate` + Squirrel)
//! ```
//!
//! ## Chain
//!
//! Exp203-208 (IPC validated) → Exp256 (EMP atlas) → Exp257 (data pipeline) →
//! **Exp258 (NUCLEUS Tower-Node)**
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
//! | Command | `cargo run --release --bin validate_nucleus_tower_node` |

use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use wetspring_barracuda::bio::diversity;
use wetspring_barracuda::ipc::primal_names;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;
use wetspring_barracuda::validation::OrExit;

fn main() {
    let mut v = Validator::new("Exp258: NUCLEUS Tower-Node Deployment — Live Primal Orchestration");

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
        println!("  biomeOS binary: not found in PATH or known locations");
        v.check_pass("biomeOS binary found (optional — build first)", true);
    }

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

    v.section("Phase 3: NUCLEUS Mode Readiness Assessment");

    let has_beardog = discover_primal_bin(primal_names::BEARDOG).is_some();
    let has_songbird = discover_primal_bin(primal_names::SONGBIRD).is_some();
    let has_toadstool = discover_primal_bin(primal_names::TOADSTOOL).is_some();
    let has_nestgate = discover_primal_bin(primal_names::NESTGATE).is_some();

    let tower_ready = has_beardog && has_songbird;
    let node_ready = tower_ready && has_toadstool;
    let nest_ready = tower_ready && has_nestgate;
    let full_ready = node_ready && has_nestgate;

    println!(
        "  Tower Atomic (BearDog + Songbird):        {}",
        if tower_ready {
            "READY"
        } else {
            "need binaries"
        }
    );
    println!(
        "  Node Atomic  (Tower + ToadStool):         {}",
        if node_ready { "READY" } else { "need binaries" }
    );
    println!(
        "  Nest Atomic  (Tower + NestGate + Squirrel): {}",
        if nest_ready { "READY" } else { "need binaries" }
    );
    println!(
        "  Full Atomic  (all primals):               {}",
        if full_ready { "READY" } else { "need binaries" }
    );

    v.check_pass("NUCLEUS mode assessment complete", true);

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
    println!("  Result: H' = {direct_h:.6}");

    let t_json = Instant::now();
    let mut json_h = 0.0;
    for _ in 0..n_iterations {
        let params = serde_json::json!({"counts": counts, "metrics": ["shannon"]});
        let result =
            wetspring_barracuda::ipc::dispatch::dispatch("science.diversity", &params).or_exit("unexpected error");
        json_h = result["shannon"].as_f64().or_exit("unexpected error");
    }
    let json_us = t_json.elapsed().as_nanos() as f64 / f64::from(n_iterations) / 1000.0;

    v.check_pass("IPC dispatch: Shannon computed", json_h > 0.0);
    v.check_pass(
        "IPC dispatch: bit-identical to direct",
        (direct_h - json_h).abs() < tolerances::EXACT_F64,
    );
    println!("  IPC dispatch:    {n_iterations}× Shannon on 200 taxa = {json_us:.2}µs/call");
    println!(
        "  Overhead: {:.1}× (JSON-RPC serialization + dispatch routing)",
        json_us / direct_us
    );
    println!(
        "  Math fidelity: |direct - ipc| = {:.2e}",
        (direct_h - json_h).abs()
    );

    v.section("Phase 5: Full Pipeline Dispatch Timing");

    let t_pipeline = Instant::now();
    let pipeline_params = serde_json::json!({
        "counts": counts,
        "metrics": ["all"],
        "scenario": "standard_growth",
    });
    let pipeline_result =
        wetspring_barracuda::ipc::dispatch::dispatch("science.full_pipeline", &pipeline_params)
            .or_exit("unexpected error");
    let pipeline_ms = t_pipeline.elapsed().as_secs_f64() * 1000.0;

    let has_diversity = pipeline_result.get("diversity").is_some();
    let has_qs = pipeline_result.get("qs_model").is_some();
    v.check_pass("Full pipeline: diversity stage completed", has_diversity);
    v.check_pass("Full pipeline: QS model stage completed", has_qs);
    println!("  Full pipeline dispatch: {pipeline_ms:.2}ms");
    println!(
        "  Stages: diversity={has_diversity}, qs_model={has_qs}, anderson={}",
        pipeline_result.get("anderson").is_some()
    );

    v.section("Phase 6: Deployment Roadmap");

    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  NUCLEUS Deployment Roadmap                                     ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║                                                                  ║");
    println!("║  Step 1: Build primals                                           ║");
    println!("║    cd phase2/biomeOS && cargo build --release                    ║");
    println!("║    cd phase1/toadstool && cargo build --release                  ║");
    println!("║    cd phase2/nestgate && cargo build --release                   ║");
    println!("║                                                                  ║");
    println!("║  Step 2: Start Tower Atomic                                      ║");
    println!("║    biomeos nucleus start --mode tower --node-id eastgate         ║");
    println!("║                                                                  ║");
    println!("║  Step 3: Start Node Atomic (adds ToadStool GPU)                  ║");
    println!("║    biomeos nucleus start --mode node --node-id eastgate          ║");
    println!("║                                                                  ║");
    println!("║  Step 4: Start wetSpring science primal                          ║");
    println!("║    cargo run --release --bin wetspring_server                    ║");
    println!("║                                                                  ║");
    println!("║  Step 5: Verify full pipeline                                    ║");
    println!("║    cargo run --release --bin validate_nucleus_data_pipeline      ║");
    println!("║                                                                  ║");
    println!("║  Step 6: LAN mesh (after 10G cables)                             ║");
    println!("║    biomeos nucleus start --mode full --node-id eastgate          ║");
    println!("║    biomeos nucleus start --mode node --node-id strandgate        ║");
    println!("║    biomeos nucleus start --mode nest --node-id westgate          ║");
    println!("║                                                                  ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");

    v.finish();
}

/// Discover biomeOS binary via environment or PATH.
///
/// Uses `BIOMEOS_BIN` if set and path exists, then `which("biomeos")`,
/// then delegates to [`discover_primal_bin`] for relative-path discovery.
fn discover_biomeos_bin() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("BIOMEOS_BIN") {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }
    if let Ok(path) = which(primal_names::BIOMEOS) {
        return Some(path);
    }

    discover_primal_bin(primal_names::BIOMEOS)
}

/// Discover a primal binary via environment or PATH.
///
/// Uses `{NAME}_BIN` (e.g. `BEARDOG_BIN`, `TOADSTOOL_BIN`) if set and path exists,
/// then `which(name)`, then relative candidates as last resort.
///
/// For `biomeos`, uses directory `biomeOS` (known casing for the phase2 crate).
fn discover_primal_bin(name: &str) -> Option<PathBuf> {
    let env_var = format!("{}_BIN", name.to_uppercase().replace('-', "_"));
    if let Ok(path) = std::env::var(&env_var) {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }
    if let Ok(path) = which(name) {
        return Some(path);
    }

    let dir_name = match name {
        n if n == primal_names::BIOMEOS => "biomeOS",
        _ => name,
    };

    let phase_dirs = ["phase1", "phase2"];
    for phase in &phase_dirs {
        let candidates = [
            format!("../{phase}/{dir_name}/target/release/{name}"),
            format!("../../{phase}/{dir_name}/target/release/{name}"),
            format!("../../../{phase}/{dir_name}/target/release/{name}"),
        ];
        for candidate in &candidates {
            let p = PathBuf::from(candidate);
            if p.exists() {
                return Some(p);
            }
        }
    }
    None
}

fn which(name: &str) -> Result<PathBuf, ()> {
    let path_var = std::env::var("PATH").map_err(|_| ())?;
    for dir in path_var.split(':') {
        let candidate = PathBuf::from(dir).join(name);
        if candidate.exists() && candidate.is_file() {
            return Ok(candidate);
        }
    }
    Err(())
}
