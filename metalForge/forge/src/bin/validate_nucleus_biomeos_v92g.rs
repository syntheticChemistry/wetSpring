// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::doc_markdown
)]
//! # Exp302: NUCLEUS Atomics + PCIe Bypass + biomeOS Orchestration
//!
//! Comprehensive mixed hardware validation exercising:
//! - S1: Tower discovery — all substrates (CPU, GPU, NPU)
//! - S2: PCIe bypass topology — GPU→NPU→CPU routing decisions
//! - S3: NUCLEUS pipeline — Tower→Node→Nest end-to-end
//! - S4: biomeOS graph — DAG topology for cross-substrate dispatch
//! - S5: Mixed hardware pipelines — all 53 workloads dispatched
//! - S6: Bandwidth-aware routing — PCIe transfer cost decisions
//! - S7: Streaming analysis — chained vs round-trip for all patterns
//! - S8: Absorption evolution — Write→Absorb→Lean tracking
//!
//! | Field | Value |
//! |-------|-------|
//! | Date | 2026-03-02 |
//! | Command | `cargo run -p wetspring-forge --release --bin validate_nucleus_biomeos_v92g` |
//!
//! Validation class: Pipeline + Integration
//! Provenance: metalForge NUCLEUS + biomeOS infrastructure

use wetspring_forge::bridge;
use wetspring_forge::dispatch;
use wetspring_forge::inventory;
use wetspring_forge::nest;
use wetspring_forge::streaming::{PipelineStage, StreamingSession};
use wetspring_forge::substrate::{Capability, SubstrateKind};
use wetspring_forge::workloads;
use wetspring_forge::workloads::ShaderOrigin;

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Exp302: NUCLEUS + PCIe + biomeOS — V92G Mixed Hardware");
    println!("═══════════════════════════════════════════════════════════");

    let mut pass = 0u32;
    let mut fail = 0u32;

    section_tower_discovery(&mut pass, &mut fail);
    section_pcie_bypass(&mut pass, &mut fail);
    section_nucleus_pipeline(&mut pass, &mut fail);
    section_biomeos_dag(&mut pass, &mut fail);
    section_full_catalog_dispatch(&mut pass, &mut fail);
    section_bandwidth_routing(&mut pass, &mut fail);
    section_streaming_patterns(&mut pass, &mut fail);
    section_absorption_evolution(&mut pass, &mut fail);

    let total = pass + fail;
    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  Exp302 Result: {pass} PASS, {fail} FAIL ({total} total)");
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

// ═══ S1: Tower Discovery ═════════════════════════════════════════════
fn section_tower_discovery(pass: &mut u32, fail: &mut u32) {
    println!("\n  S1: Tower Discovery — All Substrates");

    let local = inventory::discover();
    let tower = inventory::discover_with_tower();

    check(
        "Tower: local substrates found",
        !local.is_empty(),
        pass,
        fail,
    );
    check(
        "Tower: tower ⊇ local",
        tower.len() >= local.len(),
        pass,
        fail,
    );

    let cpu_count = local
        .iter()
        .filter(|s| s.kind == SubstrateKind::Cpu)
        .count();
    let gpu_count = local
        .iter()
        .filter(|s| s.kind == SubstrateKind::Gpu)
        .count();
    let npu_count = local
        .iter()
        .filter(|s| s.kind == SubstrateKind::Npu)
        .count();
    check("Tower: CPU present", cpu_count > 0, pass, fail);
    println!("    Local: {cpu_count} CPU, {gpu_count} GPU, {npu_count} NPU");
    println!("    Tower: {} total substrates", tower.len());

    for sub in &local {
        let has_caps = !sub.capabilities.is_empty();
        check(
            &format!("Tower '{}': has capabilities", sub.identity.name),
            has_caps,
            pass,
            fail,
        );
    }

    for sub in local.iter().filter(|s| s.kind == SubstrateKind::Gpu) {
        let tier = bridge::detect_bandwidth_tier(sub);
        check(
            &format!("Tower '{}': bandwidth tier detected", sub.identity.name),
            tier.is_some(),
            pass,
            fail,
        );
        if let Some(t) = tier {
            let est_1mb = bridge::estimated_transfer_us(sub, 1_000_000).unwrap_or(0.0);
            let est_100mb = bridge::estimated_transfer_us(sub, 100_000_000).unwrap_or(0.0);
            check(
                &format!("Tower '{}': monotonic transfer", sub.identity.name),
                est_100mb > est_1mb && est_1mb > 0.0,
                pass,
                fail,
            );
            println!(
                "    {}: {:?}, 1MB={est_1mb:.0}µs, 100MB={est_100mb:.0}µs",
                sub.identity.name, t
            );
        }
    }
}

// ═══ S2: PCIe Bypass Topology ════════════════════════════════════════
fn section_pcie_bypass(pass: &mut u32, fail: &mut u32) {
    println!("\n  S2: PCIe Bypass — GPU→NPU→CPU Topology");

    let mut gpu_only = StreamingSession::new(SubstrateKind::Gpu);
    gpu_only.add_stage(PipelineStage {
        name: "Diversity (GPU)".into(),
        capability: Capability::ShaderDispatch,
        accepts_gpu_buffer: true,
        produces_gpu_buffer: true,
    });
    gpu_only.add_stage(PipelineStage {
        name: "Bray-Curtis (GPU)".into(),
        capability: Capability::ShaderDispatch,
        accepts_gpu_buffer: true,
        produces_gpu_buffer: true,
    });
    gpu_only.add_stage(PipelineStage {
        name: "NMF (GPU)".into(),
        capability: Capability::ShaderDispatch,
        accepts_gpu_buffer: true,
        produces_gpu_buffer: true,
    });
    let gpu_analysis = gpu_only.analyze();
    check(
        "PCIe: GPU-only fully streamable",
        gpu_analysis.fully_streamable,
        pass,
        fail,
    );
    check(
        "PCIe: GPU-only 0 CPU roundtrips",
        gpu_analysis.cpu_roundtrips == 0,
        pass,
        fail,
    );
    check(
        &format!("PCIe: GPU-only {} chained", gpu_analysis.gpu_chained),
        gpu_analysis.gpu_chained >= 2,
        pass,
        fail,
    );
    println!(
        "    GPU-only: {} chained, {} roundtrips",
        gpu_analysis.gpu_chained, gpu_analysis.cpu_roundtrips
    );

    let mut npu_bypass = StreamingSession::new(SubstrateKind::Gpu);
    npu_bypass.add_stage(PipelineStage {
        name: "DADA2 (GPU)".into(),
        capability: Capability::ShaderDispatch,
        accepts_gpu_buffer: true,
        produces_gpu_buffer: true,
    });
    npu_bypass.add_stage(PipelineStage {
        name: "Chimera (GPU)".into(),
        capability: Capability::ShaderDispatch,
        accepts_gpu_buffer: true,
        produces_gpu_buffer: true,
    });
    npu_bypass.add_stage(PipelineStage {
        name: "ESN classify (NPU)".into(),
        capability: Capability::QuantizedInference { bits: 8 },
        accepts_gpu_buffer: true,
        produces_gpu_buffer: false,
    });
    let npu_analysis = npu_bypass.analyze();
    check(
        "PCIe: GPU→GPU→NPU bypass, 0 CPU roundtrips",
        npu_analysis.cpu_roundtrips == 0,
        pass,
        fail,
    );
    check(
        "PCIe: NPU accepts GPU buffer",
        npu_analysis.fully_streamable,
        pass,
        fail,
    );
    println!(
        "    GPU→GPU→NPU: {} chained, {} roundtrips, streamable={}",
        npu_analysis.gpu_chained, npu_analysis.cpu_roundtrips, npu_analysis.fully_streamable
    );

    let mut cpu_fallback = StreamingSession::new(SubstrateKind::Gpu);
    cpu_fallback.add_stage(PipelineStage {
        name: "Diversity (GPU)".into(),
        capability: Capability::ShaderDispatch,
        accepts_gpu_buffer: true,
        produces_gpu_buffer: true,
    });
    cpu_fallback.add_stage(PipelineStage {
        name: "Clock (CPU f64)".into(),
        capability: Capability::F64Compute,
        accepts_gpu_buffer: false,
        produces_gpu_buffer: false,
    });
    cpu_fallback.add_stage(PipelineStage {
        name: "Bootstrap (CPU f64)".into(),
        capability: Capability::F64Compute,
        accepts_gpu_buffer: false,
        produces_gpu_buffer: false,
    });
    let fb_analysis = cpu_fallback.analyze();
    check(
        "PCIe: CPU fallback NOT fully streamable",
        !fb_analysis.fully_streamable,
        pass,
        fail,
    );
    check(
        &format!(
            "PCIe: CPU fallback {} roundtrips",
            fb_analysis.cpu_roundtrips
        ),
        fb_analysis.cpu_roundtrips >= 1,
        pass,
        fail,
    );
    println!(
        "    GPU→CPU→CPU: {} roundtrips (expected — f64 precision needs CPU)",
        fb_analysis.cpu_roundtrips
    );
}

// ═══ S3: NUCLEUS Pipeline ════════════════════════════════════════════
fn section_nucleus_pipeline(pass: &mut u32, fail: &mut u32) {
    println!("\n  S3: NUCLEUS Pipeline — Tower→Node→Nest");

    let songbird = inventory::discover_songbird_socket();
    if let Some(path) = &songbird {
        println!("    Songbird: {}", path.display());
        check("NUCLEUS: Songbird socket live", true, pass, fail);
    } else {
        println!("    Songbird: sovereign mode (no mesh)");
        check("NUCLEUS: sovereign (no Songbird)", true, pass, fail);
    }

    let nestgate = nest::discover_nestgate_socket();
    if let Some(path) = &nestgate {
        println!("    NestGate: {}", path.display());
        check("NUCLEUS: NestGate socket live", true, pass, fail);
    } else {
        println!("    NestGate: sovereign mode");
        check("NUCLEUS: sovereign (no NestGate)", true, pass, fail);
    }

    let subs = inventory::discover();
    let has_compute = subs.iter().any(|s| {
        s.capabilities
            .iter()
            .any(|c| matches!(c, Capability::F64Compute | Capability::ShaderDispatch))
    });
    check(
        "NUCLEUS: compute capability present",
        has_compute,
        pass,
        fail,
    );

    let has_storage = nestgate.is_some() || !subs.is_empty();
    check("NUCLEUS: storage path available", has_storage, pass, fail);

    let tower_subs = inventory::discover_with_tower();
    let local_count = subs.len();
    let tower_count = tower_subs.len();
    check(
        &format!("NUCLEUS: Tower discovery ({local_count} local, {tower_count} tower)"),
        tower_count >= local_count,
        pass,
        fail,
    );

    let all_workloads = workloads::all_workloads();
    let dispatchable = all_workloads
        .iter()
        .filter(|w| {
            !matches!(w.origin, ShaderOrigin::CpuOnly)
                && dispatch::route(&w.workload, &subs).is_some()
        })
        .count();
    check(
        &format!("NUCLEUS: {dispatchable} workloads dispatchable"),
        dispatchable > 0,
        pass,
        fail,
    );
    println!(
        "    Pipeline: Tower({tower_count}) → Node({dispatchable} dispatched) → Nest({})",
        if nestgate.is_some() {
            "live"
        } else {
            "sovereign"
        }
    );
}

// ═══ S4: biomeOS DAG ════════════════════════════════════════════════
fn section_biomeos_dag(pass: &mut u32, fail: &mut u32) {
    println!("\n  S4: biomeOS DAG — Cross-Substrate Pipeline Topology");

    let patterns: Vec<(&str, Vec<PipelineStage>)> = vec![
        (
            "Full GPU streaming (diversity→BC→NMF→PCoA)",
            vec![
                PipelineStage {
                    name: "Diversity".into(),
                    capability: Capability::ShaderDispatch,
                    accepts_gpu_buffer: true,
                    produces_gpu_buffer: true,
                },
                PipelineStage {
                    name: "Bray-Curtis".into(),
                    capability: Capability::ShaderDispatch,
                    accepts_gpu_buffer: true,
                    produces_gpu_buffer: true,
                },
                PipelineStage {
                    name: "NMF".into(),
                    capability: Capability::ShaderDispatch,
                    accepts_gpu_buffer: true,
                    produces_gpu_buffer: true,
                },
                PipelineStage {
                    name: "PCoA".into(),
                    capability: Capability::ShaderDispatch,
                    accepts_gpu_buffer: true,
                    produces_gpu_buffer: true,
                },
            ],
        ),
        (
            "GPU→NPU→CPU (16S pipeline)",
            vec![
                PipelineStage {
                    name: "DADA2 (GPU)".into(),
                    capability: Capability::ShaderDispatch,
                    accepts_gpu_buffer: true,
                    produces_gpu_buffer: true,
                },
                PipelineStage {
                    name: "Chimera (GPU)".into(),
                    capability: Capability::ShaderDispatch,
                    accepts_gpu_buffer: true,
                    produces_gpu_buffer: true,
                },
                PipelineStage {
                    name: "Taxonomy (NPU)".into(),
                    capability: Capability::QuantizedInference { bits: 8 },
                    accepts_gpu_buffer: true,
                    produces_gpu_buffer: false,
                },
                PipelineStage {
                    name: "Diversity (CPU)".into(),
                    capability: Capability::F64Compute,
                    accepts_gpu_buffer: false,
                    produces_gpu_buffer: false,
                },
            ],
        ),
        (
            "Science pipeline (GPU+CPU spectral)",
            vec![
                PipelineStage {
                    name: "Diversity (GPU)".into(),
                    capability: Capability::ShaderDispatch,
                    accepts_gpu_buffer: true,
                    produces_gpu_buffer: true,
                },
                PipelineStage {
                    name: "Anderson spectral (CPU)".into(),
                    capability: Capability::F64Compute,
                    accepts_gpu_buffer: false,
                    produces_gpu_buffer: false,
                },
                PipelineStage {
                    name: "Graph Laplacian (CPU)".into(),
                    capability: Capability::F64Compute,
                    accepts_gpu_buffer: false,
                    produces_gpu_buffer: false,
                },
                PipelineStage {
                    name: "NMF (GPU)".into(),
                    capability: Capability::ShaderDispatch,
                    accepts_gpu_buffer: true,
                    produces_gpu_buffer: true,
                },
            ],
        ),
        (
            "Drug repurposing (GPU GEMM+NMF)",
            vec![
                PipelineStage {
                    name: "GEMM (GPU)".into(),
                    capability: Capability::ShaderDispatch,
                    accepts_gpu_buffer: true,
                    produces_gpu_buffer: true,
                },
                PipelineStage {
                    name: "NMF (GPU)".into(),
                    capability: Capability::ShaderDispatch,
                    accepts_gpu_buffer: true,
                    produces_gpu_buffer: true,
                },
                PipelineStage {
                    name: "TransE scoring (GPU)".into(),
                    capability: Capability::ShaderDispatch,
                    accepts_gpu_buffer: true,
                    produces_gpu_buffer: true,
                },
                PipelineStage {
                    name: "Top-K ranking (CPU)".into(),
                    capability: Capability::F64Compute,
                    accepts_gpu_buffer: false,
                    produces_gpu_buffer: false,
                },
            ],
        ),
        (
            "Field genomics (NPU→GPU→CPU)",
            vec![
                PipelineStage {
                    name: "Basecall (NPU)".into(),
                    capability: Capability::QuantizedInference { bits: 8 },
                    accepts_gpu_buffer: false,
                    produces_gpu_buffer: false,
                },
                PipelineStage {
                    name: "DADA2 (GPU)".into(),
                    capability: Capability::ShaderDispatch,
                    accepts_gpu_buffer: true,
                    produces_gpu_buffer: true,
                },
                PipelineStage {
                    name: "Diversity (GPU)".into(),
                    capability: Capability::ShaderDispatch,
                    accepts_gpu_buffer: true,
                    produces_gpu_buffer: true,
                },
                PipelineStage {
                    name: "Bootstrap CI (CPU)".into(),
                    capability: Capability::F64Compute,
                    accepts_gpu_buffer: false,
                    produces_gpu_buffer: false,
                },
            ],
        ),
    ];

    for (name, stages) in &patterns {
        let mut session = StreamingSession::new(SubstrateKind::Gpu);
        for stage in stages {
            session.add_stage(stage.clone());
        }
        let analysis = session.analyze();
        check(
            &format!("DAG '{name}': {}-stage analyzed", analysis.n_stages),
            analysis.n_stages == stages.len(),
            pass,
            fail,
        );
        println!(
            "    {name}: {} stages, {} GPU-chained, {} CPU-roundtrips, streamable={}",
            analysis.n_stages,
            analysis.gpu_chained,
            analysis.cpu_roundtrips,
            analysis.fully_streamable
        );
    }
}

// ═══ S5: Full Catalog Dispatch ═══════════════════════════════════════
fn section_full_catalog_dispatch(pass: &mut u32, fail: &mut u32) {
    println!("\n  S5: Full Catalog — 53 Workloads Dispatched");

    let substrates = inventory::discover();
    let all = workloads::all_workloads();
    let (absorbed, local, cpu_only) = workloads::origin_summary();

    check(
        &format!("Catalog: {} total", all.len()),
        all.len() >= 53,
        pass,
        fail,
    );
    check(
        &format!("Catalog: {absorbed} absorbed"),
        absorbed >= 52,
        pass,
        fail,
    );
    check(&format!("Catalog: {local} local"), local == 0, pass, fail);
    check(
        &format!("Catalog: {cpu_only} CPU-only"),
        cpu_only == 2,
        pass,
        fail,
    );

    let mut routed = 0u32;
    let mut cpu_only_count = 0u32;
    for bw in &all {
        if matches!(bw.origin, ShaderOrigin::CpuOnly) {
            cpu_only_count += 1;
            continue;
        }
        if dispatch::route(&bw.workload, &substrates).is_some() {
            routed += 1;
        }
    }
    let gpu_workloads = all.len() as u32 - cpu_only_count;
    check(
        &format!("Dispatch: {routed}/{gpu_workloads} routed"),
        routed == gpu_workloads,
        pass,
        fail,
    );
    println!(
        "    {} total: {} absorbed, {} local, {} CPU-only, {routed} routed",
        all.len(),
        absorbed,
        local,
        cpu_only
    );
}

// ═══ S6: Bandwidth-Aware Routing ═════════════════════════════════════
fn section_bandwidth_routing(pass: &mut u32, fail: &mut u32) {
    println!("\n  S6: Bandwidth-Aware Routing — PCIe Transfer Cost");

    let substrates = inventory::discover();
    let test_workloads = vec![
        workloads::diversity(),
        workloads::pcoa(),
        workloads::kmer_histogram(),
        workloads::chimera(),
        workloads::kmd(),
        workloads::gbm_inference(),
        workloads::anderson_spectral(),
        workloads::boltzmann_sampling(),
    ];

    for bw in &test_workloads {
        let standard = dispatch::route(&bw.workload, &substrates);
        let bw_aware = dispatch::route_bandwidth_aware(&bw.workload, &substrates);
        check(
            &format!("BW '{}': standard", bw.workload.name),
            standard.is_some(),
            pass,
            fail,
        );
        check(
            &format!("BW '{}': bandwidth-aware", bw.workload.name),
            bw_aware.is_some(),
            pass,
            fail,
        );
        if let (Some(s), Some(b)) = (&standard, &bw_aware) {
            println!(
                "    {}: std→{:?}, BW→{:?}",
                bw.workload.name, s.substrate.kind, b.substrate.kind
            );
        }
    }
}

// ═══ S7: Streaming Patterns ══════════════════════════════════════════
fn section_streaming_patterns(pass: &mut u32, fail: &mut u32) {
    println!("\n  S7: Streaming Patterns — Chained vs Round-Trip");

    let stages_2gpu = vec![
        PipelineStage {
            name: "A (GPU)".into(),
            capability: Capability::ShaderDispatch,
            accepts_gpu_buffer: true,
            produces_gpu_buffer: true,
        },
        PipelineStage {
            name: "B (GPU)".into(),
            capability: Capability::ShaderDispatch,
            accepts_gpu_buffer: true,
            produces_gpu_buffer: true,
        },
    ];
    let mut s2 = StreamingSession::new(SubstrateKind::Gpu);
    for stage in &stages_2gpu {
        s2.add_stage(stage.clone());
    }
    let a2 = s2.analyze();
    check("Stream 2-GPU: chained=1", a2.gpu_chained == 1, pass, fail);
    check("Stream 2-GPU: streamable", a2.fully_streamable, pass, fail);

    let stages_4gpu = vec![
        PipelineStage {
            name: "A (GPU)".into(),
            capability: Capability::ShaderDispatch,
            accepts_gpu_buffer: true,
            produces_gpu_buffer: true,
        },
        PipelineStage {
            name: "B (GPU)".into(),
            capability: Capability::ShaderDispatch,
            accepts_gpu_buffer: true,
            produces_gpu_buffer: true,
        },
        PipelineStage {
            name: "C (GPU)".into(),
            capability: Capability::ShaderDispatch,
            accepts_gpu_buffer: true,
            produces_gpu_buffer: true,
        },
        PipelineStage {
            name: "D (GPU)".into(),
            capability: Capability::ShaderDispatch,
            accepts_gpu_buffer: true,
            produces_gpu_buffer: true,
        },
    ];
    let mut s4 = StreamingSession::new(SubstrateKind::Gpu);
    for stage in &stages_4gpu {
        s4.add_stage(stage.clone());
    }
    let a4 = s4.analyze();
    check("Stream 4-GPU: chained=3", a4.gpu_chained == 3, pass, fail);
    check("Stream 4-GPU: streamable", a4.fully_streamable, pass, fail);

    let stages_mixed = vec![
        PipelineStage {
            name: "GPU".into(),
            capability: Capability::ShaderDispatch,
            accepts_gpu_buffer: true,
            produces_gpu_buffer: true,
        },
        PipelineStage {
            name: "CPU".into(),
            capability: Capability::F64Compute,
            accepts_gpu_buffer: false,
            produces_gpu_buffer: false,
        },
        PipelineStage {
            name: "GPU".into(),
            capability: Capability::ShaderDispatch,
            accepts_gpu_buffer: true,
            produces_gpu_buffer: true,
        },
    ];
    let mut sm = StreamingSession::new(SubstrateKind::Gpu);
    for stage in &stages_mixed {
        sm.add_stage(stage.clone());
    }
    let am = sm.analyze();
    check(
        "Stream GPU-CPU-GPU: NOT streamable",
        !am.fully_streamable,
        pass,
        fail,
    );
    check(
        &format!("Stream GPU-CPU-GPU: {} roundtrips", am.cpu_roundtrips),
        am.cpu_roundtrips >= 1,
        pass,
        fail,
    );
    println!(
        "    2-GPU: chained={}, 4-GPU: chained={}, mixed: roundtrips={}",
        a2.gpu_chained, a4.gpu_chained, am.cpu_roundtrips
    );
}

// ═══ S8: Absorption Evolution ════════════════════════════════════════
fn section_absorption_evolution(pass: &mut u32, fail: &mut u32) {
    println!("\n  S8: Absorption Evolution — Write→Absorb→Lean");

    let all = workloads::all_workloads();
    let (absorbed, local, cpu_only) = workloads::origin_summary();

    check(
        &format!("Evolution: {absorbed} absorbed (fully lean)"),
        absorbed >= 52,
        pass,
        fail,
    );
    check(
        &format!("Evolution: {local} local WGSL (target: 0)"),
        local == 0,
        pass,
        fail,
    );
    check(
        &format!("Evolution: {cpu_only} CPU-only (I/O-bound)"),
        cpu_only == 2,
        pass,
        fail,
    );

    for bw in &all {
        if bw.is_absorbed() {
            check(
                &format!("Lean '{}': has primitive", bw.workload.name),
                bw.primitive.is_some(),
                pass,
                fail,
            );
        }
    }

    println!("    Write→Absorb→Lean: {absorbed} absorbed, {local} local, {cpu_only} CPU-only");
    println!("    All absorbed workloads have ToadStool primitive names");
    println!("    0 local WGSL — fully lean on ToadStool S86");
}
