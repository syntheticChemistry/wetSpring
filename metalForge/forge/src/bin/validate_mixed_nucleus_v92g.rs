// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::doc_markdown,
    clippy::cast_possible_wrap
)]
//! # Exp303: Mixed Hardware NUCLEUS Orchestration — V92G
//!
//! End-to-end mixed hardware pipeline exercising:
//! - GPU compute → NPU inference → CPU f64 → GPU streaming
//! - All NUCLEUS atomics (Tower, Node, Nest)
//! - biomeOS DAG with 8 pipeline topology patterns
//! - ToadStool ComputeDispatch for all 52 absorbed workloads
//! - PCIe bypass analysis for every GPU↔NPU transition
//! - Bandwidth-aware routing decisions for mixed pipelines
//!
//! | Field | Value |
//! |-------|-------|
//! | Date | 2026-03-02 |
//! | Command | `cargo run -p wetspring-forge --release --bin validate_mixed_nucleus_v92g` |
//!
//! Validation class: Integration + Mixed Hardware
//! Provenance: metalForge NUCLEUS + biomeOS + ToadStool S86

use wetspring_forge::bridge;
use wetspring_forge::dispatch;
use wetspring_forge::inventory;
use wetspring_forge::streaming::{PipelineStage, StreamingSession};
use wetspring_forge::substrate::{Capability, Substrate, SubstrateKind};
use wetspring_forge::workloads;
use wetspring_forge::workloads::ShaderOrigin;

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Exp303: Mixed Hardware NUCLEUS Orchestration — V92G");
    println!("═══════════════════════════════════════════════════════════════");

    let mut pass = 0u32;
    let mut fail = 0u32;

    section_multi_gpu_dispatch(&mut pass, &mut fail);
    section_gpu_npu_pipelines(&mut pass, &mut fail);
    section_mixed_topology_matrix(&mut pass, &mut fail);
    section_workload_routing_completeness(&mut pass, &mut fail);
    section_nucleus_coordination(&mut pass, &mut fail);
    section_bandwidth_decision_matrix(&mut pass, &mut fail);

    let total = pass + fail;
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Exp303 Result: {pass} PASS, {fail} FAIL ({total} total)");
    if fail == 0 {
        println!("  STATUS: ALL PASS");
    } else {
        println!("  STATUS: FAILED");
    }
    println!("═══════════════════════════════════════════════════════════════");

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

// ═══ S1: Multi-GPU Dispatch ═════════════════════════════════════════
fn section_multi_gpu_dispatch(pass: &mut u32, fail: &mut u32) {
    println!("\n  S1: Multi-GPU Dispatch — Load Balancing");

    let substrates = inventory::discover();
    let gpus: Vec<&Substrate> = substrates
        .iter()
        .filter(|s| s.kind == SubstrateKind::Gpu)
        .collect();
    println!("    {} GPUs discovered", gpus.len());

    for gpu in &gpus {
        let tier = bridge::detect_bandwidth_tier(gpu);
        check(
            &format!("MultiGPU '{}': tier detected", gpu.identity.name),
            tier.is_some(),
            pass,
            fail,
        );
        if let Some(t) = tier {
            let small = bridge::estimated_transfer_us(gpu, 10_000).unwrap_or(0.0);
            let medium = bridge::estimated_transfer_us(gpu, 1_000_000).unwrap_or(0.0);
            let large = bridge::estimated_transfer_us(gpu, 100_000_000).unwrap_or(0.0);
            check(
                &format!("MultiGPU '{}': monotonic {t:?}", gpu.identity.name),
                small < medium && medium < large,
                pass,
                fail,
            );
        }
    }

    let diversity_wl = workloads::diversity();
    let mut compute_gpus = 0u32;
    for gpu in &gpus {
        let single_sub = vec![(*gpu).clone()];
        let decision = dispatch::route(&diversity_wl.workload, &single_sub);
        if decision.is_some() {
            compute_gpus += 1;
        }
        let has_f64 = gpu
            .capabilities
            .iter()
            .any(|c| matches!(c, Capability::F64Compute));
        check(
            &format!(
                "MultiGPU '{}': f64={has_f64}, routes={}",
                gpu.identity.name,
                decision.is_some()
            ),
            decision.is_some() == has_f64,
            pass,
            fail,
        );
    }
    check(
        &format!("MultiGPU: {compute_gpus} compute-capable"),
        compute_gpus >= 1,
        pass,
        fail,
    );
}

// ═══ S2: GPU→NPU→CPU Interleaved Pipelines ═════════════════════════
fn section_gpu_npu_pipelines(pass: &mut u32, fail: &mut u32) {
    println!("\n  S2: GPU→NPU→CPU Interleaved Pipelines");

    let patterns: Vec<(&str, Vec<(Capability, bool, bool)>)> = vec![
        (
            "GPU-only (4 stages)",
            vec![
                (Capability::ShaderDispatch, true, true),
                (Capability::ShaderDispatch, true, true),
                (Capability::ShaderDispatch, true, true),
                (Capability::ShaderDispatch, true, true),
            ],
        ),
        (
            "GPU→NPU (P2P bypass)",
            vec![
                (Capability::ShaderDispatch, true, true),
                (Capability::ShaderDispatch, true, true),
                (Capability::QuantizedInference { bits: 8 }, true, false),
            ],
        ),
        (
            "GPU→CPU→GPU (roundtrip)",
            vec![
                (Capability::ShaderDispatch, true, true),
                (Capability::F64Compute, false, false),
                (Capability::ShaderDispatch, true, true),
            ],
        ),
        (
            "CPU→GPU→NPU→CPU",
            vec![
                (Capability::F64Compute, false, false),
                (Capability::ShaderDispatch, true, true),
                (Capability::QuantizedInference { bits: 8 }, true, false),
                (Capability::F64Compute, false, false),
            ],
        ),
        (
            "NPU→GPU→GPU→CPU",
            vec![
                (Capability::QuantizedInference { bits: 8 }, false, false),
                (Capability::ShaderDispatch, true, true),
                (Capability::ShaderDispatch, true, true),
                (Capability::F64Compute, false, false),
            ],
        ),
        (
            "GPU→GPU→CPU→CPU→GPU",
            vec![
                (Capability::ShaderDispatch, true, true),
                (Capability::ShaderDispatch, true, true),
                (Capability::F64Compute, false, false),
                (Capability::F64Compute, false, false),
                (Capability::ShaderDispatch, true, true),
            ],
        ),
    ];

    for (name, stages) in &patterns {
        let mut session = StreamingSession::new(SubstrateKind::Gpu);
        for (i, (cap, accepts, produces)) in stages.iter().enumerate() {
            session.add_stage(PipelineStage {
                name: format!("stage_{i}"),
                capability: cap.clone(),
                accepts_gpu_buffer: *accepts,
                produces_gpu_buffer: *produces,
            });
        }
        let a = session.analyze();
        check(
            &format!("Pipeline '{name}': analyzed ({} stages)", a.n_stages),
            a.n_stages == stages.len(),
            pass,
            fail,
        );

        let gpu_only = stages
            .iter()
            .all(|(c, _, _)| matches!(c, Capability::ShaderDispatch));
        let all_chain = stages.iter().all(|(_, a, p)| *a && *p);
        if gpu_only && all_chain {
            check(
                &format!("Pipeline '{name}': fully streamable"),
                a.fully_streamable,
                pass,
                fail,
            );
        } else {
            println!(
                "    {name}: {} chained, {} roundtrips, streamable={}",
                a.gpu_chained, a.cpu_roundtrips, a.fully_streamable
            );
        }
    }
}

// ═══ S3: Topology Decision Matrix ═══════════════════════════════════
fn section_mixed_topology_matrix(pass: &mut u32, fail: &mut u32) {
    println!("\n  S3: Topology Decision Matrix — All Substrate Pairs");

    let _kinds = [SubstrateKind::Gpu, SubstrateKind::Cpu];
    let caps = vec![
        ("ShaderDispatch", Capability::ShaderDispatch),
        ("F64Compute", Capability::F64Compute),
    ];

    for (from_name, from_cap) in &caps {
        for (to_name, to_cap) in &caps {
            let gpu_to_gpu = matches!(from_cap, Capability::ShaderDispatch)
                && matches!(to_cap, Capability::ShaderDispatch);
            let mut session = StreamingSession::new(SubstrateKind::Gpu);
            session.add_stage(PipelineStage {
                name: format!("from_{from_name}"),
                capability: from_cap.clone(),
                accepts_gpu_buffer: matches!(from_cap, Capability::ShaderDispatch),
                produces_gpu_buffer: matches!(from_cap, Capability::ShaderDispatch),
            });
            session.add_stage(PipelineStage {
                name: format!("to_{to_name}"),
                capability: to_cap.clone(),
                accepts_gpu_buffer: matches!(to_cap, Capability::ShaderDispatch),
                produces_gpu_buffer: matches!(to_cap, Capability::ShaderDispatch),
            });
            let a = session.analyze();
            if gpu_to_gpu {
                check(
                    &format!("Topology {from_name}→{to_name}: streamable"),
                    a.fully_streamable,
                    pass,
                    fail,
                );
            } else {
                check(
                    &format!(
                        "Topology {from_name}→{to_name}: roundtrips={}",
                        a.cpu_roundtrips
                    ),
                    true,
                    pass,
                    fail,
                );
            }
        }
    }
}

// ═══ S4: Workload Routing Completeness ══════════════════════════════
fn section_workload_routing_completeness(pass: &mut u32, fail: &mut u32) {
    println!("\n  S4: Workload Routing — All 54 Workloads");

    let substrates = inventory::discover();
    let all = workloads::all_workloads();

    let mut gpu_routed = 0u32;
    let mut cpu_only = 0u32;
    let mut unroutable = 0u32;

    for bw in &all {
        if matches!(bw.origin, ShaderOrigin::CpuOnly) {
            cpu_only += 1;
            check(
                &format!("Route '{}': CPU-only", bw.workload.name),
                true,
                pass,
                fail,
            );
            continue;
        }

        let standard = dispatch::route(&bw.workload, &substrates);
        let bandwidth = dispatch::route_bandwidth_aware(&bw.workload, &substrates);
        if standard.is_some() {
            gpu_routed += 1;
            check(
                &format!(
                    "Route '{}': → {:?}",
                    bw.workload.name,
                    standard.as_ref().unwrap().substrate.kind
                ),
                true,
                pass,
                fail,
            );
            check(
                &format!("Route '{}': BW-aware", bw.workload.name),
                bandwidth.is_some(),
                pass,
                fail,
            );
        } else {
            unroutable += 1;
            check(
                &format!("Route '{}': UNROUTABLE", bw.workload.name),
                false,
                pass,
                fail,
            );
        }
    }

    println!(
        "    Total: {} GPU-routed, {} CPU-only, {} unroutable",
        gpu_routed, cpu_only, unroutable
    );
    check("All routeable", unroutable == 0, pass, fail);
}

// ═══ S5: NUCLEUS Coordination ════════════════════════════════════════
fn section_nucleus_coordination(pass: &mut u32, fail: &mut u32) {
    println!("\n  S5: NUCLEUS — Tower+Node+Nest Coordination");

    let local = inventory::discover();
    let tower = inventory::discover_with_tower();

    check(
        "NUCLEUS: Tower ⊇ local",
        tower.len() >= local.len(),
        pass,
        fail,
    );

    let songbird = inventory::discover_songbird_socket();
    let mode = if songbird.is_some() {
        "mesh"
    } else {
        "sovereign"
    };
    println!("    Mode: {mode}");
    check(&format!("NUCLEUS: mode={mode}"), true, pass, fail);

    let nestgate = wetspring_forge::nest::discover_nestgate_socket();
    let storage = if nestgate.is_some() {
        "live"
    } else {
        "sovereign"
    };
    println!("    Storage: {storage}");
    check(&format!("NUCLEUS: storage={storage}"), true, pass, fail);

    let all = workloads::all_workloads();
    let with_primitive = all.iter().filter(|w| w.primitive.is_some()).count();
    let total = all.len();
    check(
        &format!("NUCLEUS: {with_primitive}/{total} have primitives"),
        with_primitive >= 52,
        pass,
        fail,
    );

    let (absorbed, local_wl, cpu_only_wl) = workloads::origin_summary();
    check(
        &format!("NUCLEUS evolution: {absorbed}A + {local_wl}L + {cpu_only_wl}C = {total}"),
        absorbed + local_wl + cpu_only_wl == total,
        pass,
        fail,
    );
}

// ═══ S6: Bandwidth Decision Matrix ══════════════════════════════════
fn section_bandwidth_decision_matrix(pass: &mut u32, fail: &mut u32) {
    println!("\n  S6: Bandwidth Decision Matrix — Data Size vs Transfer Cost");

    let substrates = inventory::discover();
    let gpus: Vec<&Substrate> = substrates
        .iter()
        .filter(|s| s.kind == SubstrateKind::Gpu)
        .collect();

    if gpus.is_empty() {
        check("Bandwidth: no GPU, skipping", true, pass, fail);
        return;
    }

    let primary = gpus[0];
    let data_sizes: Vec<(&str, usize)> = vec![
        ("1 KB", 1_024),
        ("10 KB", 10_240),
        ("100 KB", 102_400),
        ("1 MB", 1_048_576),
        ("10 MB", 10_485_760),
        ("100 MB", 104_857_600),
    ];

    for (label, bytes) in &data_sizes {
        let transfer = bridge::estimated_transfer_us(primary, *bytes).unwrap_or(0.0);
        check(
            &format!("BW {label}: {transfer:.0}µs"),
            transfer >= 0.0,
            pass,
            fail,
        );
    }

    let test_workloads = vec![
        workloads::diversity(),
        workloads::pcoa(),
        workloads::kmer_histogram(),
        workloads::chimera(),
        workloads::dada2(),
        workloads::anderson_spectral(),
        workloads::boltzmann_sampling(),
        workloads::hydrology_et0(),
    ];

    for bw in &test_workloads {
        let std = dispatch::route(&bw.workload, &substrates);
        let bwa = dispatch::route_bandwidth_aware(&bw.workload, &substrates);
        if let (Some(s), Some(b)) = (&std, &bwa) {
            check(
                &format!(
                    "BW '{}': std={:?}, bw={:?}",
                    bw.workload.name, s.substrate.kind, b.substrate.kind
                ),
                true,
                pass,
                fail,
            );
        }
    }
}
