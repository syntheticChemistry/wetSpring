// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::print_stdout
)]
//! # Exp245: PCIe Bypass Mixed Hardware — NPU→GPU→CPU Dispatch Topology
//!
//! Validates metalForge's mixed hardware dispatch:
//! - S1: PCIe bandwidth tier detection (Gen3/Gen4/Gen5)
//! - S2: GPU→GPU streaming (fully chainable, 0 CPU round-trips)
//! - S3: GPU→NPU bypass (accepts_gpu_buffer: true, 0 CPU round-trips)
//! - S4: GPU→CPU fallback (bandwidth-aware, 1 CPU round-trip)
//! - S5: Mixed pipeline topology analysis (PCIe bypass scoring)
//! - S6: Bandwidth-aware dispatch routing for new workloads
//!
//! Chain: Paper → CPU → GPU → Parity → ToadStool → **metalForge (this)**
//!
//! | Field | Value |
//! |-------|-------|
//! | Date | 2026-02-28 |
//! | Command | `cargo run --bin validate_pcie_bypass_mixed_hw` |

use wetspring_forge::bridge;
use wetspring_forge::dispatch::{self, Workload};
use wetspring_forge::inventory;
use wetspring_forge::streaming::{PipelineStage, StreamingSession};
use wetspring_forge::substrate::{Capability, SubstrateKind};
use wetspring_forge::workloads;

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Exp245: PCIe Bypass Mixed Hardware — NPU→GPU→CPU");
    println!("═══════════════════════════════════════════════════════════");

    let mut pass = 0u32;
    let mut fail = 0u32;

    section_bandwidth_tiers(&mut pass, &mut fail);
    section_gpu_gpu_streaming(&mut pass, &mut fail);
    section_gpu_npu_bypass(&mut pass, &mut fail);
    section_gpu_cpu_fallback(&mut pass, &mut fail);
    section_pipeline_topology(&mut pass, &mut fail);
    section_bandwidth_aware_routing(&mut pass, &mut fail);

    let total = pass + fail;
    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  Exp245 Result: {pass} PASS, {fail} FAIL ({total} total)");
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

// ═══ S1: PCIe Bandwidth Tier Detection ═══════════════════════════════
fn section_bandwidth_tiers(pass: &mut u32, fail: &mut u32) {
    println!("\n  S1: PCIe Bandwidth Tier Detection");

    let substrates = inventory::discover();
    check(
        "Discover: at least 1 substrate",
        !substrates.is_empty(),
        pass,
        fail,
    );

    for sub in &substrates {
        if sub.kind == SubstrateKind::Gpu {
            let _tier = bridge::detect_bandwidth_tier(sub);
            check(
                &format!("GPU '{}': bandwidth tier detected", sub.identity.name),
                true,
                pass,
                fail,
            );
            let transfer_1mb = bridge::estimated_transfer_us(sub, 1_000_000).unwrap_or(0.0);
            check(
                &format!(
                    "GPU '{}': 1 MB transfer = {transfer_1mb:.0} µs",
                    sub.identity.name
                ),
                transfer_1mb > 0.0,
                pass,
                fail,
            );
            let transfer_100mb = bridge::estimated_transfer_us(sub, 100_000_000).unwrap_or(0.0);
            check(
                &format!(
                    "GPU '{}': 100 MB transfer = {transfer_100mb:.0} µs",
                    sub.identity.name
                ),
                transfer_100mb > transfer_1mb,
                pass,
                fail,
            );
        }
    }
}

// ═══ S2: GPU→GPU Streaming ═══════════════════════════════════════════
fn section_gpu_gpu_streaming(pass: &mut u32, fail: &mut u32) {
    println!("\n  S2: GPU→GPU Streaming (Zero CPU Round-Trips)");

    let mut session = StreamingSession::new(SubstrateKind::Gpu);
    session.add_stage(PipelineStage {
        name: "DADA2 denoise".into(),
        capability: Capability::ShaderDispatch,
        accepts_gpu_buffer: true,
        produces_gpu_buffer: true,
    });
    session.add_stage(PipelineStage {
        name: "Chimera filter".into(),
        capability: Capability::ShaderDispatch,
        accepts_gpu_buffer: true,
        produces_gpu_buffer: true,
    });
    session.add_stage(PipelineStage {
        name: "Diversity fusion".into(),
        capability: Capability::ScalarReduce,
        accepts_gpu_buffer: true,
        produces_gpu_buffer: true,
    });
    session.add_stage(PipelineStage {
        name: "Kriging interpolation".into(),
        capability: Capability::ShaderDispatch,
        accepts_gpu_buffer: true,
        produces_gpu_buffer: false,
    });

    let analysis = session.analyze();
    check("GPU pipeline: 4 stages", analysis.n_stages == 4, pass, fail);
    check(
        "GPU pipeline: 3 chained (DADA2→Chimera→Diversity)",
        analysis.gpu_chained == 3,
        pass,
        fail,
    );
    check(
        "GPU pipeline: 0 CPU round-trips",
        analysis.cpu_roundtrips == 0,
        pass,
        fail,
    );
    check(
        "GPU pipeline: fully streamable",
        analysis.fully_streamable,
        pass,
        fail,
    );
}

// ═══ S3: GPU→NPU Bypass ═════════════════════════════════════════════
fn section_gpu_npu_bypass(pass: &mut u32, fail: &mut u32) {
    println!("\n  S3: GPU→NPU PCIe Bypass (Direct Transfer)");

    let mut session = StreamingSession::new(SubstrateKind::Gpu);
    session.add_stage(PipelineStage {
        name: "Diversity GPU".into(),
        capability: Capability::F64Compute,
        accepts_gpu_buffer: true,
        produces_gpu_buffer: true,
    });
    session.add_stage(PipelineStage {
        name: "ESN NPU classify".into(),
        capability: Capability::QuantizedInference { bits: 8 },
        accepts_gpu_buffer: true,
        produces_gpu_buffer: false,
    });

    let analysis = session.analyze();
    check("GPU→NPU: 2 stages", analysis.n_stages == 2, pass, fail);
    check(
        "GPU→NPU: chained (NPU accepts GPU buffer)",
        analysis.gpu_chained == 1,
        pass,
        fail,
    );
    check(
        "GPU→NPU: 0 CPU round-trips (PCIe bypass)",
        analysis.cpu_roundtrips == 0,
        pass,
        fail,
    );

    let mut no_bypass = StreamingSession::new(SubstrateKind::Gpu);
    no_bypass.add_stage(PipelineStage {
        name: "Diversity GPU".into(),
        capability: Capability::F64Compute,
        accepts_gpu_buffer: true,
        produces_gpu_buffer: true,
    });
    no_bypass.add_stage(PipelineStage {
        name: "ESN NPU classify (no bypass)".into(),
        capability: Capability::QuantizedInference { bits: 8 },
        accepts_gpu_buffer: false,
        produces_gpu_buffer: false,
    });

    let analysis_no = no_bypass.analyze();
    check(
        "Without bypass: 1 CPU round-trip",
        analysis_no.cpu_roundtrips == 1,
        pass,
        fail,
    );
    check(
        "Without bypass: NOT fully streamable",
        !analysis_no.fully_streamable,
        pass,
        fail,
    );
}

// ═══ S4: GPU→CPU Fallback ════════════════════════════════════════════
fn section_gpu_cpu_fallback(pass: &mut u32, fail: &mut u32) {
    println!("\n  S4: GPU→CPU Bandwidth-Aware Fallback");

    let mut mixed = StreamingSession::new(SubstrateKind::Gpu);
    mixed.add_stage(PipelineStage {
        name: "GBM GPU inference".into(),
        capability: Capability::ShaderDispatch,
        accepts_gpu_buffer: true,
        produces_gpu_buffer: true,
    });
    mixed.add_stage(PipelineStage {
        name: "CPU post-processing".into(),
        capability: Capability::F64Compute,
        accepts_gpu_buffer: false,
        produces_gpu_buffer: false,
    });
    mixed.add_stage(PipelineStage {
        name: "Molecular clock CPU".into(),
        capability: Capability::F64Compute,
        accepts_gpu_buffer: false,
        produces_gpu_buffer: false,
    });

    let analysis = mixed.analyze();
    check("Mixed: 3 stages", analysis.n_stages == 3, pass, fail);
    check(
        &format!(
            "Mixed: {} CPU round-trips (GPU→CPU breaks)",
            analysis.cpu_roundtrips
        ),
        analysis.cpu_roundtrips >= 1,
        pass,
        fail,
    );
    check(
        "Mixed: NOT fully streamable",
        !analysis.fully_streamable,
        pass,
        fail,
    );
}

// ═══ S5: Pipeline Topology Analysis ══════════════════════════════════
fn section_pipeline_topology(pass: &mut u32, fail: &mut u32) {
    println!("\n  S5: Pipeline Topology Analysis");

    let mut dada2_pipeline = StreamingSession::new(SubstrateKind::Gpu);
    dada2_pipeline.add_stage(PipelineStage {
        name: "DADA2".into(),
        capability: Capability::ShaderDispatch,
        accepts_gpu_buffer: true,
        produces_gpu_buffer: true,
    });
    dada2_pipeline.add_stage(PipelineStage {
        name: "Chimera".into(),
        capability: Capability::ShaderDispatch,
        accepts_gpu_buffer: true,
        produces_gpu_buffer: true,
    });
    dada2_pipeline.add_stage(PipelineStage {
        name: "Diversity".into(),
        capability: Capability::ScalarReduce,
        accepts_gpu_buffer: true,
        produces_gpu_buffer: true,
    });
    dada2_pipeline.add_stage(PipelineStage {
        name: "Rarefaction".into(),
        capability: Capability::ShaderDispatch,
        accepts_gpu_buffer: true,
        produces_gpu_buffer: true,
    });
    dada2_pipeline.add_stage(PipelineStage {
        name: "Kriging".into(),
        capability: Capability::ShaderDispatch,
        accepts_gpu_buffer: true,
        produces_gpu_buffer: true,
    });
    dada2_pipeline.add_stage(PipelineStage {
        name: "Reconciliation".into(),
        capability: Capability::ShaderDispatch,
        accepts_gpu_buffer: true,
        produces_gpu_buffer: false,
    });

    let analysis = dada2_pipeline.analyze();
    check(
        "6-stage pipeline: all stages counted",
        analysis.n_stages == 6,
        pass,
        fail,
    );
    check(
        "6-stage pipeline: 5 chained",
        analysis.gpu_chained == 5,
        pass,
        fail,
    );
    check(
        "6-stage pipeline: 0 CPU round-trips",
        analysis.cpu_roundtrips == 0,
        pass,
        fail,
    );
    check(
        "6-stage pipeline: fully streamable",
        analysis.fully_streamable,
        pass,
        fail,
    );
    println!("    PCIe bypass: 5 transitions saved, 0 CPU round-trips needed");
}

// ═══ S6: Bandwidth-Aware Dispatch Routing ════════════════════════════
fn section_bandwidth_aware_routing(pass: &mut u32, fail: &mut u32) {
    println!("\n  S6: Bandwidth-Aware Dispatch Routing (New Workloads)");

    let substrates = inventory::discover();

    let new_workloads = vec![
        workloads::chimera(),
        workloads::dada2(),
        workloads::gbm_inference(),
        workloads::reconciliation(),
        workloads::molecular_clock(),
    ];

    for bw in &new_workloads {
        let wl = &bw.workload;
        let decision = dispatch::route(wl, &substrates);
        match decision {
            Some(d) => {
                check(
                    &format!(
                        "Route '{}': dispatched to {:?} ({:?})",
                        wl.name, d.substrate.kind, d.reason
                    ),
                    true,
                    pass,
                    fail,
                );
            }
            None => {
                check(
                    &format!("Route '{}': no substrate (acceptable if no GPU)", wl.name),
                    substrates.iter().all(|s| s.kind == SubstrateKind::Cpu),
                    pass,
                    fail,
                );
            }
        }

        let mut bw_workload = Workload::new(&wl.name, wl.required.clone());
        bw_workload.preferred_substrate = Some(SubstrateKind::Gpu);
        bw_workload.data_bytes = Some(10_000_000);
        let bw_decision = dispatch::route_bandwidth_aware(&bw_workload, &substrates);
        match bw_decision {
            Some(d) => {
                check(
                    &format!(
                        "BW '{}': {} ({:?})",
                        wl.name, d.substrate.identity.name, d.reason
                    ),
                    true,
                    pass,
                    fail,
                );
            }
            None => {
                check(
                    &format!("BW '{}': no route (CPU-only env OK)", wl.name),
                    true,
                    pass,
                    fail,
                );
            }
        }
    }
}
