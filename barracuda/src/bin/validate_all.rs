// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]

//! Meta-runner: executes all core CPU validation binaries.
//!
//! Runs a curated set of validators that cover the major wetSpring domains
//! and prints a summary. Exit 0 if all pass, 1 if any fail, 2 if no
//! validators could be found.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Type | Meta-validation |
//! | Date | 2026-03-23 |
//! | Command | `cargo run --bin validate_all` |

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::time::Instant;

use wetspring_barracuda::validation::{DomainResult, OrExit, SKIP_CODE, print_domain_summary};

/// Curated CPU-only validators (default features — no `--features gpu`).
const BINARIES: &[&str] = &[
    "validate_diversity",
    "validate_fastq",
    "validate_alignment",
    "validate_hmm",
    "validate_gillespie",
    "validate_cooperation",
    "validate_barracuda_cpu",
    "validate_newick_parse",
    "validate_bootstrap",
    "validate_pfas",
    "validate_bistable",
    "validate_neighbor_joining",
];

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .or_exit("CARGO_MANIFEST_DIR has no parent (expected barracuda crate directory)")
        .to_path_buf()
}

fn main() -> ExitCode {
    if BINARIES.is_empty() {
        eprintln!("validate_all: no validators configured (empty BINARIES)");
        return ExitCode::from(SKIP_CODE);
    }

    let root = workspace_root();
    println!(
        "wetSpring validate_all — {} core CPU binaries (workspace: {})\n",
        BINARIES.len(),
        root.display()
    );

    let wall = Instant::now();
    let mut domains: Vec<DomainResult> = Vec::with_capacity(BINARIES.len());
    let mut failures: Vec<(&str, String, String)> = Vec::new();

    for &name in BINARIES {
        let start = Instant::now();
        let output = Command::new("cargo")
            .current_dir(&root)
            .args(["run", "--bin", name, "-p", "wetspring-barracuda"])
            .output()
            .or_exit(&format!("spawn cargo for {name}"));
        let ms = start.elapsed().as_secs_f64() * 1e3;

        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        let ok = output.status.success();

        domains.push(DomainResult {
            name,
            spring: None,
            ms,
            checks: u32::from(ok),
        });

        if ok {
            println!("  PASS  {name} ({ms:.1} ms)");
        } else {
            let code = output
                .status
                .code()
                .map_or_else(|| "signal".to_string(), |c| c.to_string());
            eprintln!("  FAIL  {name} (exit {code}, {ms:.1} ms)");
            failures.push((name, stdout, stderr));
        }
    }

    print_domain_summary("wetSpring validate_all — core CPU suite", &domains);

    let total = domains.len();
    let passed = domains.iter().filter(|d| d.checks > 0).count();
    let failed = total - passed;
    println!(
        "\nvalidate_all: {passed}/{} passed, {failed} failed, wall {:?}",
        BINARIES.len(),
        wall.elapsed()
    );

    if !failures.is_empty() {
        eprintln!("\n── Failed binary output ──");
        for (name, out, err) in &failures {
            eprintln!("\n>>> {name} (stdout)");
            for line in out.lines() {
                eprintln!("    {line}");
            }
            eprintln!(">>> {name} (stderr)");
            for line in err.lines() {
                if !line.contains("Compiling") && !line.contains("Finished") {
                    eprintln!("    {line}");
                }
            }
        }
        return ExitCode::FAILURE;
    }

    ExitCode::SUCCESS
}
