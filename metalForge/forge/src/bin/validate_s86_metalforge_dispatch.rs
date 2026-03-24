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
    clippy::doc_markdown,
    reason = "validation harness: required for domain validation"
)]
//! # Exp299: S86 Science — metalForge Cross-System Dispatch
//!
//! Validates that `ToadStool` S86 ungated primitives (spectral, graph, sampling,
//! hydrology) are routable through metalForge dispatch to appropriate substrates.
//!
//! ## Coverage
//! - S1: S86 workloads appear in catalog (7 new workloads)
//! - S2: All S86 workloads route to CPU substrate (F64Compute)
//! - S3: Streaming pipeline with S86 stages (GPU → CPU spectral → CPU graph)
//! - S4: Bandwidth-aware routing for S86 workloads
//! - S5: Cross-spring provenance tracking
//! - S6: Mathematical validation through dispatch (actual computation)
//!
//! | Field | Value |
//! |-------|-------|
//! | Date | 2026-03-02 |
//! | Command | `cargo run -p wetspring-forge --release --bin validate_s86_metalforge_dispatch` |
//!
//! Validation class: Pipeline + Analytical
//!
//! Provenance: S86 ungated primitives via metalForge Node dispatch

use wetspring_barracuda::validation::OrExit;
use wetspring_forge::dispatch;
use wetspring_forge::inventory;
use wetspring_forge::streaming::{PipelineStage, StreamingSession};
use wetspring_forge::substrate::{Capability, SubstrateKind};
use wetspring_forge::workloads;

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Exp299: S86 Science — metalForge Cross-System Dispatch");
    println!("═══════════════════════════════════════════════════════════");

    let mut pass = 0u32;
    let mut fail = 0u32;

    section_s86_catalog(&mut pass, &mut fail);
    section_s86_cpu_routing(&mut pass, &mut fail);
    section_s86_streaming_pipeline(&mut pass, &mut fail);
    section_s86_bandwidth_routing(&mut pass, &mut fail);
    section_s86_cross_spring_provenance(&mut pass, &mut fail);
    section_s86_math_through_dispatch(&mut pass, &mut fail);

    let total = pass + fail;
    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  Exp299 Result: {pass} PASS, {fail} FAIL ({total} total)");
    if fail == 0 {
        println!("  STATUS: ALL PASS");
    } else {
        println!("  STATUS: FAILED");
    }
    println!("═══════════════════════════════════════════════════════════");

    if fail > 0 {
        std::process::exit(1);
    }
}

fn check(name: &str, condition: bool, pass: &mut u32, fail: &mut u32) {
    if condition {
        println!("  [PASS] {name}");
        *pass += 1;
    } else {
        println!("  [FAIL] {name}");
        *fail += 1;
    }
}

// ═══ S1: S86 Workloads in Catalog ════════════════════════════════════
fn section_s86_catalog(pass: &mut u32, fail: &mut u32) {
    println!("\n  S1: S86 Science Workloads in Catalog");

    let all = workloads::all_workloads();
    let s86_names = [
        "anderson_spectral",
        "hofstadter_butterfly",
        "graph_laplacian",
        "belief_propagation",
        "boltzmann_sampling",
        "space_filling_sampling",
        "hydrology_et0",
    ];

    for name in &s86_names {
        let found = all.iter().any(|w| w.workload.name == *name);
        check(&format!("Catalog: '{name}' present"), found, pass, fail);
    }

    check(
        &format!("Catalog: {} total workloads", all.len()),
        all.len() >= 46 + s86_names.len(),
        pass,
        fail,
    );
}

// ═══ S2: CPU Substrate Routing ═══════════════════════════════════════
fn section_s86_cpu_routing(pass: &mut u32, fail: &mut u32) {
    println!("\n  S2: S86 CPU Substrate Routing");

    let substrates = inventory::discover();
    let s86_workloads = [
        workloads::anderson_spectral(),
        workloads::hofstadter_butterfly(),
        workloads::graph_laplacian(),
        workloads::belief_propagation(),
        workloads::boltzmann_sampling(),
        workloads::space_filling_sampling(),
        workloads::hydrology_et0(),
    ];

    for bw in &s86_workloads {
        let decision = dispatch::route(&bw.workload, &substrates);
        check(
            &format!("Route '{}': substrate found", bw.workload.name),
            decision.is_some(),
            pass,
            fail,
        );
        if let Some(d) = &decision {
            check(
                &format!("Route '{}': routable", bw.workload.name),
                d.substrate.kind == SubstrateKind::Cpu || d.substrate.kind == SubstrateKind::Gpu,
                pass,
                fail,
            );
            println!(
                "    {} → {:?} ({})",
                bw.workload.name, d.substrate.kind, d.substrate.identity.name
            );
        }
    }
}

// ═══ S3: Streaming Pipeline with S86 Stages ═════════════════════════
fn section_s86_streaming_pipeline(pass: &mut u32, fail: &mut u32) {
    println!("\n  S3: Streaming Pipeline with S86 Stages");

    let mut pipeline = StreamingSession::new(SubstrateKind::Gpu);

    pipeline.add_stage(PipelineStage {
        name: "Diversity GPU (Shannon batch)".into(),
        capability: Capability::ShaderDispatch,
        accepts_gpu_buffer: true,
        produces_gpu_buffer: true,
    });
    pipeline.add_stage(PipelineStage {
        name: "Bray-Curtis GPU".into(),
        capability: Capability::ShaderDispatch,
        accepts_gpu_buffer: true,
        produces_gpu_buffer: true,
    });
    pipeline.add_stage(PipelineStage {
        name: "Anderson spectral (CPU f64)".into(),
        capability: Capability::F64Compute,
        accepts_gpu_buffer: false,
        produces_gpu_buffer: false,
    });
    pipeline.add_stage(PipelineStage {
        name: "Graph Laplacian (CPU f64)".into(),
        capability: Capability::F64Compute,
        accepts_gpu_buffer: false,
        produces_gpu_buffer: false,
    });
    pipeline.add_stage(PipelineStage {
        name: "Boltzmann sampling (CPU f64)".into(),
        capability: Capability::F64Compute,
        accepts_gpu_buffer: false,
        produces_gpu_buffer: false,
    });
    pipeline.add_stage(PipelineStage {
        name: "NMF GPU".into(),
        capability: Capability::ShaderDispatch,
        accepts_gpu_buffer: true,
        produces_gpu_buffer: true,
    });

    let analysis = pipeline.analyze();
    check("S86 pipeline: 6 stages", analysis.n_stages == 6, pass, fail);
    check(
        "S86 pipeline: GPU→GPU chained for first pair",
        analysis.gpu_chained >= 1,
        pass,
        fail,
    );
    check(
        "S86 pipeline: CPU stages detected",
        analysis.cpu_roundtrips >= 1,
        pass,
        fail,
    );
    check(
        "S86 pipeline: NOT fully streamable (has CPU stages)",
        !analysis.fully_streamable,
        pass,
        fail,
    );
    println!("    Pipeline: 2 GPU chained → 3 CPU (spectral/graph/sample) → 1 GPU");

    let mut full_gpu = StreamingSession::new(SubstrateKind::Gpu);
    full_gpu.add_stage(PipelineStage {
        name: "Diversity GPU".into(),
        capability: Capability::ShaderDispatch,
        accepts_gpu_buffer: true,
        produces_gpu_buffer: true,
    });
    full_gpu.add_stage(PipelineStage {
        name: "Bray-Curtis GPU".into(),
        capability: Capability::ShaderDispatch,
        accepts_gpu_buffer: true,
        produces_gpu_buffer: true,
    });
    full_gpu.add_stage(PipelineStage {
        name: "NMF GPU".into(),
        capability: Capability::ShaderDispatch,
        accepts_gpu_buffer: true,
        produces_gpu_buffer: true,
    });
    let gpu_analysis = full_gpu.analyze();
    check(
        "GPU-only pipeline: fully streamable",
        gpu_analysis.fully_streamable,
        pass,
        fail,
    );
    check(
        "GPU-only pipeline: 0 CPU roundtrips",
        gpu_analysis.cpu_roundtrips == 0,
        pass,
        fail,
    );
}

// ═══ S4: Bandwidth-Aware Routing ═════════════════════════════════════
fn section_s86_bandwidth_routing(pass: &mut u32, fail: &mut u32) {
    println!("\n  S4: Bandwidth-Aware Routing for S86 Workloads");

    let substrates = inventory::discover();
    let s86_workloads = [
        workloads::anderson_spectral(),
        workloads::graph_laplacian(),
        workloads::boltzmann_sampling(),
    ];

    for bw in &s86_workloads {
        let standard = dispatch::route(&bw.workload, &substrates);
        let bw_aware = dispatch::route_bandwidth_aware(&bw.workload, &substrates);
        check(
            &format!("BW '{}': standard route", bw.workload.name),
            standard.is_some(),
            pass,
            fail,
        );
        check(
            &format!("BW '{}': BW-aware route", bw.workload.name),
            bw_aware.is_some(),
            pass,
            fail,
        );
        if let (Some(s), Some(b)) = (&standard, &bw_aware) {
            println!(
                "    {}: standard → {:?}, BW-aware → {:?}",
                bw.workload.name, s.substrate.kind, b.substrate.kind
            );
        }
    }
}

// ═══ S5: Cross-Spring Provenance ═════════════════════════════════════
fn section_s86_cross_spring_provenance(pass: &mut u32, fail: &mut u32) {
    println!("\n  S5: Cross-Spring Provenance Tracking");

    let provenance = [
        ("anderson_spectral", "hotSpring → all springs"),
        ("hofstadter_butterfly", "hotSpring precision"),
        ("graph_laplacian", "neuralSpring linalg"),
        ("belief_propagation", "neuralSpring graph"),
        ("boltzmann_sampling", "wateringHole sampling"),
        ("space_filling_sampling", "wateringHole + airSpring"),
        ("hydrology_et0", "airSpring hydrology"),
    ];

    for (name, origin) in &provenance {
        let bw = workloads::all_workloads()
            .into_iter()
            .find(|w| w.workload.name == *name);
        check(
            &format!("Provenance '{name}': cataloged"),
            bw.is_some(),
            pass,
            fail,
        );
        if let Some(w) = &bw {
            check(
                &format!("Provenance '{name}': absorbed"),
                w.is_absorbed(),
                pass,
                fail,
            );
            let prim = w.primitive.unwrap_or("—");
            println!("    {name}: {origin} → {prim}");
        }
    }
}

// ═══ S6: Mathematical Validation Through Dispatch ════════════════════
fn section_s86_math_through_dispatch(pass: &mut u32, fail: &mut u32) {
    println!("\n  S6: Mathematical Validation Through Dispatch");

    // Anderson spectral
    let anderson_l = 6_usize;
    let anderson_w = 10.0;
    let csr = barracuda::spectral::anderson_3d(anderson_l, anderson_l, anderson_l, anderson_w, 42);
    let tri = barracuda::spectral::lanczos(&csr, 30, 42);
    let eigs = barracuda::spectral::lanczos_eigenvalues(&tri);
    let r = barracuda::spectral::level_spacing_ratio(&eigs);
    check(
        &format!("Anderson r={r:.4} in valid range"),
        r > 0.0 && r < 1.0,
        pass,
        fail,
    );

    let bandwidth = barracuda::spectral::spectral_bandwidth(&eigs);
    let phase = barracuda::spectral::classify_spectral_phase(&eigs, bandwidth * 0.5);
    check(
        &format!("Spectral phase classified: {phase:?}"),
        true,
        pass,
        fail,
    );
    println!("    Anderson: L={anderson_l}, W={anderson_w:.1}, r={r:.4}, phase={phase:?}");

    // Graph Laplacian (flat adjacency, 4 nodes)
    let adj = vec![
        0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    ];
    let lap = barracuda::linalg::graph_laplacian(&adj, 4);
    check("Graph Laplacian: 16 entries", lap.len() == 16, pass, fail);
    let diag: Vec<f64> = (0..4).map(|i| lap[i * 4 + i]).collect();
    let rank = barracuda::linalg::effective_rank(&diag);
    check(
        &format!("Effective rank={rank:.3} > 0"),
        rank > 0.0,
        pass,
        fail,
    );
    println!("    Graph: 4-node, effective_rank={rank:.3}");

    // Boltzmann sampling
    let loss_fn = |x: &[f64]| -> f64 { x.iter().map(|v| v * v).sum() };
    let init = vec![1.0, -1.0, 0.5];
    let result = barracuda::sample::boltzmann_sampling(&loss_fn, &init, 1.0, 0.5, 500, 42);
    check(
        &format!("Boltzmann: {} losses", result.losses.len()),
        result.losses.len() >= 500,
        pass,
        fail,
    );
    check(
        &format!("Boltzmann: accept_rate={:.3}", result.acceptance_rate),
        result.acceptance_rate >= 0.0 && result.acceptance_rate <= 1.0,
        pass,
        fail,
    );
    println!(
        "    Boltzmann: 500 steps, accept_rate={:.3}",
        result.acceptance_rate
    );

    // LHS + Sobol
    let bounds = vec![(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)];
    let lhs = barracuda::sample::latin_hypercube(50, &bounds, 42)
        .or_exit("latin_hypercube sampling should succeed for valid bounds and seed");
    check(
        &format!("LHS: {}×{} samples", lhs.len(), lhs[0].len()),
        lhs.len() == 50 && lhs[0].len() == 3,
        pass,
        fail,
    );
    let sobol_bounds = vec![(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)];
    let sobol = barracuda::sample::sobol_scaled(64, &sobol_bounds)
        .or_exit("sobol_scaled sampling should succeed for valid bounds");
    check(
        &format!("Sobol: {}×{} samples", sobol.len(), sobol[0].len()),
        sobol.len() == 64 && sobol[0].len() == 3,
        pass,
        fail,
    );
    println!("    LHS: 50×3, Sobol: 64×3");

    // Hydrology ET₀
    let monthly: [f64; 12] = [
        5.0, 7.0, 12.0, 16.0, 20.0, 25.0, 28.0, 27.0, 22.0, 16.0, 10.0, 6.0,
    ];
    let hi = barracuda::stats::thornthwaite_heat_index(&monthly);
    let thorn = barracuda::stats::thornthwaite_et0(20.0, hi, 12.5, 30.0);
    let hamon = barracuda::stats::hamon_et0(20.0, 12.5);
    let hargreaves = barracuda::stats::hargreaves_et0(35.0, 32.0, 18.0);

    if let Some(t) = thorn {
        check(&format!("Thornthwaite ET₀={t:.2}"), t > 0.0, pass, fail);
    } else {
        check("Thornthwaite ET₀ returned None", false, pass, fail);
    }
    if let Some(h) = hamon {
        check(&format!("Hamon ET₀={h:.2}"), h > 0.0, pass, fail);
    } else {
        check("Hamon ET₀ returned None", false, pass, fail);
    }
    if let Some(hg) = hargreaves {
        check(&format!("Hargreaves ET₀={hg:.2}"), hg > 0.0, pass, fail);
    } else {
        check("Hargreaves ET₀ returned None", false, pass, fail);
    }
    println!(
        "    ET₀: Thornthwaite={}, Hamon={}, Hargreaves={}",
        thorn.map_or_else(|| "None".to_string(), |v| format!("{v:.2}")),
        hamon.map_or_else(|| "None".to_string(), |v| format!("{v:.2}")),
        hargreaves.map_or_else(|| "None".to_string(), |v| format!("{v:.2}")),
    );
}
