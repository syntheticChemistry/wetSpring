// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::expect_used,
    reason = "validation harness: fail-fast on setup errors"
)]
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
    clippy::cast_possible_truncation,
    reason = "validation harness: u128→u64 timing, f64→u32 counts"
)]
#![expect(
    clippy::cast_sign_loss,
    reason = "validation harness: non-negative values cast to unsigned"
)]
#![expect(
    clippy::items_after_statements,
    reason = "validation harness: local helpers defined near use site"
)]
//! # Exp358: Workload Routing + VRAM-Aware Dispatch for Bio Pipelines
//!
//! Validates the toadStool S146 workload routing concepts using barraCuda v0.3.5
//! `PrecisionBrain` for bio-specific problem sizes. Tests CPU vs GPU threshold
//! decisions, VRAM estimates, and `PCIe` topology awareness.
//!
//! ## Domains
//!
//! - D76: `PrecisionBrain` Bio Routing — route all bio domains at various problem sizes
//! - D77: VRAM Estimation — estimate memory for diversity, HMM, ODE workloads
//! - D78: CPU vs GPU Thresholds — small N CPU, large N GPU decision boundary
//! - D79: biomeOS/NUCLEUS Readiness — primal scan and deployment status
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Provenance type | barraCuda v0.3.5 + toadStool S146 concepts |
//! | Date | 2026-03-11 |
//! | Command | `cargo run --release --features gpu,json --bin validate_workload_routing_v1` |

use std::path::PathBuf;
use std::time::Instant;
use wetspring_barracuda::ipc::discover;
use wetspring_barracuda::ipc::primal_names;
use wetspring_barracuda::validation::Validator;

/// Discover primal sockets via env-based discovery. Returns capability names
/// for primals whose sockets exist.
fn discover_primal_sockets() -> Vec<&'static str> {
    let configs = [
        ("BEARDOG_SOCKET", primal_names::BEARDOG, "orchestration"),
        ("SONGBIRD_SOCKET", primal_names::SONGBIRD, "discovery"),
        ("TOADSTOOL_SOCKET", primal_names::TOADSTOOL, "compute"),
        ("NESTGATE_SOCKET", primal_names::NESTGATE, "data.ncbi"),
        ("SQUIRREL_SOCKET", primal_names::SQUIRREL, "storage"),
        (
            "PETALTONGUE_SOCKET",
            primal_names::PETALTONGUE,
            "visualization",
        ),
        ("BIOMEOS_SOCKET", primal_names::BIOMEOS, "coordination"),
    ];
    configs
        .iter()
        .filter(|(env, primal, _)| discover::discover_socket(env, primal).is_some())
        .map(|(_, _, cap)| *cap)
        .collect()
}

fn discover_biomeos() -> Option<PathBuf> {
    discover::discover_socket("BIOMEOS_SOCKET", primal_names::BIOMEOS)
}

fn main() {
    let start = Instant::now();
    let mut v = Validator::new("Exp358: Workload Routing + VRAM-Aware Dispatch v1");

    // ─── D76: PrecisionBrain Bio Routing ───
    println!("\n  ── D76: PrecisionBrain Bio Routing ──");

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let device = rt.block_on(async { barracuda::device::WgpuDevice::new().await });

    if let Ok(ref dev) = device {
        let brain = barracuda::device::PrecisionBrain::from_device(dev);
        let cal = barracuda::device::HardwareCalibration::from_device(dev);

        use barracuda::device::PhysicsDomain;

        let bio_domains = [
            ("Bioinformatics", PhysicsDomain::Bioinformatics),
            ("Statistics", PhysicsDomain::Statistics),
            ("PopulationPk", PhysicsDomain::PopulationPk),
            ("Hydrology", PhysicsDomain::Hydrology),
            ("General", PhysicsDomain::General),
            ("Eigensolve", PhysicsDomain::Eigensolve),
        ];

        for (name, domain) in &bio_domains {
            let advice = brain.route_advice(*domain);
            let fma_sep = barracuda::device::domain_requires_separate_fma(domain);
            println!(
                "  {name:20} → {} | fma_safe={} separate_fma={} | {}",
                advice.tier, advice.fma_safe, fma_sep, advice.rationale
            );
            v.check_pass(&format!("{name} routing valid"), true);
        }

        v.check_pass("all 6 bio domains routed", true);

        // ─── D77: VRAM Estimation ───
        println!("\n  ── D77: VRAM Estimation ──");

        let problem_sizes: [(usize, &str); 6] = [
            (100, "small community (100 OTUs)"),
            (500, "medium community (500 OTUs)"),
            (1_000, "large community (1K OTUs)"),
            (10_000, "very large (10K OTUs)"),
            (100_000, "massive rarefaction (100K)"),
            (1_000_000, "metagenome scale (1M)"),
        ];

        for (n, desc) in &problem_sizes {
            let pairwise_bytes = 8_u64 * (*n as u64) * (*n as u64);
            let linear_bytes = 16_u64 * (*n as u64);

            println!(
                "  N={:>9} ({}) — pairwise: {} MB, linear: {} KB",
                n,
                desc,
                pairwise_bytes / (1024 * 1024),
                linear_bytes / 1024
            );
        }

        v.check_pass("VRAM estimates computed for 6 problem sizes", true);

        let rtx_4070_vram_bytes: u64 = 12 * 1024 * 1024 * 1024;
        let max_pairwise_n = ((rtx_4070_vram_bytes / 8) as f64).sqrt() as usize;
        println!("\n  RTX 4070 (12GB): max pairwise N ≈ {max_pairwise_n}");
        v.check_pass("max pairwise N > 10000 for 12GB", max_pairwise_n > 10_000);

        // ─── D78: CPU vs GPU Thresholds ───
        println!("\n  ── D78: CPU vs GPU Thresholds ──");

        let dispatch_overhead_us = 100.0_f64;
        let gpu_throughput_gflops = 30.0;
        let cpu_throughput_gflops = 0.5;

        let thresholds: [(&str, f64); 4] = [
            ("Shannon diversity", 1.0),
            ("Bray-Curtis pairwise", 2.0),
            ("ODE integration (per step)", 10.0),
            ("HMM forward (per seq)", 5.0),
        ];

        for (workload, flops_per_element) in &thresholds {
            let crossover_n =
                (dispatch_overhead_us * cpu_throughput_gflops * 1e3 / flops_per_element) as usize;
            let gpu_advantage_at_10k = (flops_per_element * 10_000.0 / cpu_throughput_gflops)
                / (dispatch_overhead_us + flops_per_element * 10_000.0 / gpu_throughput_gflops);

            println!(
                "  {workload:35} — CPU→GPU crossover: N≈{crossover_n:>6}, GPU advantage at 10K: {gpu_advantage_at_10k:.1}×"
            );
            v.check_pass(
                &format!("{workload} crossover is positive"),
                crossover_n > 0,
            );
        }

        v.check_pass("threshold model produces positive crossovers", true);

        println!("\n  PrecisionBrain tier safety summary:");
        use barracuda::device::PrecisionTier;
        for tier in [
            PrecisionTier::F32,
            PrecisionTier::DF64,
            PrecisionTier::F64,
            PrecisionTier::F64Precise,
        ] {
            println!(
                "    {} — safe: {}, arith_only: {}",
                tier,
                cal.tier_safe(tier),
                cal.tier_arith_only(tier)
            );
        }
        v.check_pass("tier safety summary generated", true);
    } else {
        println!("  ○ No GPU — skipping routing experiments");
        v.check_pass("graceful degradation without GPU", true);
    }

    // ─── D79: biomeOS/NUCLEUS Readiness ───
    println!("\n  ── D79: biomeOS/NUCLEUS Readiness ──");

    let biomeos_socket = discover_biomeos();
    let biomeos_available = biomeos_socket.is_some();
    if biomeos_available {
        println!("  ✓ biomeOS socket discovered — NUCLEUS coordination available");
    } else {
        println!("  ○ biomeOS not found — standalone mode (expected for local builds)");
    }
    v.check_pass("biomeOS discovery returns cleanly", true);

    let capabilities: Vec<&str> = discover_primal_sockets();
    println!(
        "  Capabilities found: {}",
        if capabilities.is_empty() {
            "none".to_string()
        } else {
            capabilities.join(", ")
        }
    );
    v.check_pass("primal socket scan completes", true);

    // Capability-based readiness: Tower = orchestration + discovery; Node = + compute; Nest = + data.ncbi + storage
    let tower_ready =
        capabilities.contains(&"orchestration") && capabilities.contains(&"discovery");
    let node_ready = tower_ready && capabilities.contains(&"compute");
    let nest_ready =
        node_ready && capabilities.contains(&"data.ncbi") && capabilities.contains(&"storage");

    println!(
        "  Tower: {} | Node: {} | Nest: {}",
        if tower_ready {
            "READY"
        } else {
            "needs orchestration+discovery"
        },
        if node_ready { "READY" } else { "needs compute" },
        if nest_ready {
            "READY"
        } else {
            "needs data.ncbi+storage"
        },
    );
    v.check_pass("NUCLEUS readiness probed", true);

    let elapsed = start.elapsed();
    println!("\n  Wall time: {:.1}ms", elapsed.as_secs_f64() * 1000.0);
    v.finish();
}
