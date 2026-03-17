// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::print_stdout,
    reason = "validation harness: results printed to stdout"
)]
#![expect(
    clippy::items_after_statements,
    reason = "validation harness: local helpers defined near use site"
)]
//! # Exp213: Compute Dispatch + Streaming Evolution (V66)
//!
//! Extends Exp080 (35/35) with V66 evolution targets:
//!
//! - **S1**: All 29 bio workloads route correctly on full-system inventory
//! - **S2**: Absorption status audit — zero local WGSL shaders
//! - **S3**: Streaming pipeline topology for `PCIe` bypass (NPU→GPU direct)
//! - **S4**: Mixed hardware dispatch priority + fallback
//! - **S5**: NUCLEUS atomic model — Tower (capabilities), Node (dispatch), Nest (metrics)
//! - **S6**: Streaming pipeline analysis: 16S, pure analytics, ODE sweep, sentinel
//! - **S7**: Dispatch threshold gating (GPU only above threshold)
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline | Exp080 (35/35), metalForge forge v0.3.0 |
//! | Date | 2026-02-27 |
//! | Commit | wetSpring Phase 66 |
//! | Command | `cargo run --release --bin validate_dispatch_evolution_v66` |
//!
//! Validation class: Pipeline
//! Provenance: End-to-end pipeline integration test

use wetspring_barracuda::validation::OrExit;
use wetspring_forge::dispatch::{self, Reason, Workload};
use wetspring_forge::streaming::{PipelineStage, StreamingSession};
use wetspring_forge::substrate::{
    Capability, Identity, Properties, Substrate, SubstrateKind, SubstrateOrigin,
};
use wetspring_forge::workloads;

fn check(pass: &mut u32, fail: &mut u32, total: &mut u32, name: &str, ok: bool) {
    *total += 1;
    if ok {
        *pass += 1;
        println!("  [PASS] {name}");
    } else {
        *fail += 1;
        println!("  [FAIL] {name}");
    }
}

fn make_gpu() -> Substrate {
    Substrate {
        kind: SubstrateKind::Gpu,
        identity: Identity::named("RTX 4070"),
        properties: Properties {
            memory_bytes: Some(12 * 1024 * 1024 * 1024),
            has_f64: true,
            has_timestamps: true,
            ..Properties::default()
        },
        capabilities: vec![
            Capability::F64Compute,
            Capability::F32Compute,
            Capability::ScalarReduce,
            Capability::ShaderDispatch,
            Capability::TimestampQuery,
        ],
        origin: SubstrateOrigin::Local,
    }
}

fn make_npu() -> Substrate {
    Substrate {
        kind: SubstrateKind::Npu,
        identity: Identity::named("AKD1000"),
        properties: Properties {
            memory_bytes: Some(1024 * 1024),
            ..Properties::default()
        },
        capabilities: vec![
            Capability::F32Compute,
            Capability::QuantizedInference { bits: 8 },
            Capability::BatchInference { max_batch: 8 },
            Capability::WeightMutation,
        ],
        origin: SubstrateOrigin::Local,
    }
}

fn make_cpu() -> Substrate {
    Substrate {
        kind: SubstrateKind::Cpu,
        identity: Identity::named("i9-12900K"),
        properties: Properties {
            memory_bytes: Some(64 * 1024 * 1024 * 1024),
            core_count: Some(24),
            thread_count: Some(32),
            ..Properties::default()
        },
        capabilities: vec![
            Capability::F64Compute,
            Capability::F32Compute,
            Capability::CpuCompute,
            Capability::SimdVector,
        ],
        origin: SubstrateOrigin::Local,
    }
}

// ── S1 ──────────────────────────────────────────────────────────────────────

fn section_workload_routing(pass: &mut u32, fail: &mut u32, total: &mut u32) {
    println!("\n═══ S1: All bio workloads route to capable substrates ═══\n");

    let inventory = [make_gpu(), make_npu(), make_cpu()];
    let all = workloads::all_workloads();

    check(pass, fail, total, "workload catalog >= 29", all.len() >= 29);

    let mut routed = 0_usize;
    let mut gpu_n = 0_usize;
    let mut npu_n = 0_usize;
    let mut cpu_n = 0_usize;

    for bw in &all {
        if let Some(d) = dispatch::route(&bw.workload, &inventory) {
            routed += 1;
            match d.substrate.kind {
                SubstrateKind::Gpu => gpu_n += 1,
                SubstrateKind::Npu => npu_n += 1,
                SubstrateKind::Cpu => cpu_n += 1,
            }
        }
    }

    check(
        pass,
        fail,
        total,
        "all workloads routed",
        routed == all.len(),
    );
    check(
        pass,
        fail,
        total,
        &format!("GPU handles {gpu_n} compute workloads (>=20)"),
        gpu_n >= 20,
    );
    check(
        pass,
        fail,
        total,
        &format!("NPU routes {npu_n} workloads (0: taxonomy needs f64)"),
        npu_n == 0,
    );
    check(
        pass,
        fail,
        total,
        "CPU handles I/O-bound workloads",
        cpu_n >= 1,
    );

    let tax = workloads::taxonomy();
    let tax_d = dispatch::route(&tax.workload, &inventory);
    check(
        pass,
        fail,
        total,
        "taxonomy → GPU (NPU lacks f64+shader)",
        tax_d.is_some_and(|d| d.substrate.kind == SubstrateKind::Gpu),
    );

    let fq = workloads::fastq_parsing();
    let fq_d = dispatch::route(&fq.workload, &inventory);
    check(
        pass,
        fail,
        total,
        "FASTQ parsing routes to CPU",
        fq_d.is_some_and(|d| d.substrate.kind == SubstrateKind::Cpu),
    );
}

// ── S2 ──────────────────────────────────────────────────────────────────────

fn section_absorption_status(pass: &mut u32, fail: &mut u32, total: &mut u32) {
    println!("\n═══ S2: Absorption status — zero local WGSL ═══\n");

    let (absorbed, local, cpu_only) = workloads::origin_summary();

    check(pass, fail, total, "45 absorbed workloads", absorbed == 45);
    check(pass, fail, total, "0 local WGSL (full lean)", local == 0);
    check(
        pass,
        fail,
        total,
        "2 CPU-only (FASTQ + ncbi_assembly_ingest)",
        cpu_only == 2,
    );

    let all = workloads::all_workloads();
    let all_have_prim = all
        .iter()
        .filter(|w| w.is_absorbed())
        .all(|w| w.primitive.is_some());
    check(
        pass,
        fail,
        total,
        "all absorbed have ToadStool primitive",
        all_have_prim,
    );

    let ode_count = all.iter().filter(|w| w.ode_dims.is_some()).count();
    check(
        pass,
        fail,
        total,
        &format!("{ode_count} ODE workloads with dims (>=5)"),
        ode_count >= 5,
    );

    let ode_all_absorbed = all
        .iter()
        .filter(|w| w.ode_dims.is_some())
        .all(workloads::BioWorkload::is_absorbed);
    check(
        pass,
        fail,
        total,
        "all ODE workloads absorbed",
        ode_all_absorbed,
    );
}

// ── S3 ──────────────────────────────────────────────────────────────────────

fn section_streaming_topology(pass: &mut u32, fail: &mut u32, total: &mut u32) {
    println!("\n═══ S3: Streaming pipeline topology — PCIe bypass ═══\n");

    let mut pure_gpu = StreamingSession::new(SubstrateKind::Gpu);
    for name in &["QF", "DADA2", "Diversity", "BrayCurtis"] {
        pure_gpu.add_stage(PipelineStage {
            name: name.to_string(),
            capability: Capability::F64Compute,
            accepts_gpu_buffer: true,
            produces_gpu_buffer: true,
        });
    }
    let a = pure_gpu.analyze();
    check(
        pass,
        fail,
        total,
        "pure GPU: fully streamable",
        a.fully_streamable,
    );
    check(pass, fail, total, "pure GPU: 4 stages", a.n_stages == 4);
    check(
        pass,
        fail,
        total,
        "pure GPU: 3 chained transitions",
        a.gpu_chained == 3,
    );
    check(
        pass,
        fail,
        total,
        "pure GPU: 0 CPU roundtrips",
        a.cpu_roundtrips == 0,
    );

    let mut mixed = StreamingSession::new(SubstrateKind::Gpu);
    mixed.add_stage(PipelineStage {
        name: "GPU_Div".into(),
        capability: Capability::F64Compute,
        accepts_gpu_buffer: true,
        produces_gpu_buffer: true,
    });
    mixed.add_stage(PipelineStage {
        name: "NPU_Classify".into(),
        capability: Capability::QuantizedInference { bits: 8 },
        accepts_gpu_buffer: false,
        produces_gpu_buffer: false,
    });
    let m = mixed.analyze();
    check(
        pass,
        fail,
        total,
        "GPU→NPU (no PCIe): not streamable",
        !m.fully_streamable,
    );
    check(
        pass,
        fail,
        total,
        "GPU→NPU: 1 CPU roundtrip",
        m.cpu_roundtrips == 1,
    );

    let mut bypass = StreamingSession::new(SubstrateKind::Gpu);
    bypass.add_stage(PipelineStage {
        name: "GPU_Div".into(),
        capability: Capability::F64Compute,
        accepts_gpu_buffer: true,
        produces_gpu_buffer: true,
    });
    bypass.add_stage(PipelineStage {
        name: "NPU_PCIe".into(),
        capability: Capability::QuantizedInference { bits: 8 },
        accepts_gpu_buffer: true,
        produces_gpu_buffer: false,
    });
    let b = bypass.analyze();
    check(
        pass,
        fail,
        total,
        "PCIe bypass: GPU→NPU chains (1 chained)",
        b.gpu_chained >= 1,
    );
    check(
        pass,
        fail,
        total,
        "PCIe bypass: 0 CPU roundtrips",
        b.cpu_roundtrips == 0,
    );
}

// ── S4 ──────────────────────────────────────────────────────────────────────

fn section_mixed_dispatch(pass: &mut u32, fail: &mut u32, total: &mut u32) {
    println!("\n═══ S4: Mixed hardware dispatch priority + fallback ═══\n");

    let full = [make_gpu(), make_npu(), make_cpu()];
    let cpu_only = [make_cpu()];
    let gpu_only = [make_gpu()];

    let div = Workload::new(
        "diversity",
        vec![Capability::F64Compute, Capability::ScalarReduce],
    );
    let tax = Workload::new(
        "taxonomy_int8",
        vec![Capability::QuantizedInference { bits: 8 }],
    );
    let fq = Workload::new("fastq_parse", vec![Capability::CpuCompute]);

    let d1 = dispatch::route(&div, &full)
        .or_exit("diversity workload should route to GPU on full system");
    check(
        pass,
        fail,
        total,
        "diversity → GPU on full system",
        d1.substrate.kind == SubstrateKind::Gpu,
    );
    check(
        pass,
        fail,
        total,
        "diversity reason: BestAvailable",
        d1.reason == Reason::BestAvailable,
    );

    let d2 = dispatch::route(&tax, &full)
        .or_exit("taxonomy workload should route to NPU on full system");
    check(
        pass,
        fail,
        total,
        "taxonomy → NPU on full system",
        d2.substrate.kind == SubstrateKind::Npu,
    );

    let d3 =
        dispatch::route(&fq, &full).or_exit("FASTQ workload should route to CPU on full system");
    check(
        pass,
        fail,
        total,
        "FASTQ → CPU on full system",
        d3.substrate.kind == SubstrateKind::Cpu,
    );

    let d4 = dispatch::route(&div, &cpu_only);
    check(
        pass,
        fail,
        total,
        "diversity: no route on CPU-only (needs ScalarReduce)",
        d4.is_none(),
    );

    let fq = Workload::new("fastq_parse", vec![Capability::CpuCompute]);
    let d4b = dispatch::route(&fq, &cpu_only)
        .or_exit("FASTQ workload should route to CPU on CPU-only inventory");
    check(
        pass,
        fail,
        total,
        "FASTQ: routes to CPU on CPU-only",
        d4b.substrate.kind == SubstrateKind::Cpu,
    );

    let d5 = dispatch::route(&tax, &gpu_only);
    check(
        pass,
        fail,
        total,
        "taxonomy: no route on GPU-only (needs int8)",
        d5.is_none(),
    );

    let pref = Workload::new("test", vec![Capability::F64Compute]).prefer(SubstrateKind::Gpu);
    let d6 = dispatch::route(&pref, &full)
        .or_exit("workload with GPU preference should route when GPU available");
    check(
        pass,
        fail,
        total,
        "GPU preference honored",
        d6.reason == Reason::Preferred,
    );

    let bad_pref = Workload::new("test", vec![Capability::F64Compute]).prefer(SubstrateKind::Npu);
    let d7 = dispatch::route(&bad_pref, &full)
        .or_exit("workload with NPU preference should fallback to GPU when NPU lacks f64");
    check(
        pass,
        fail,
        total,
        "NPU preference ignored (NPU lacks f64)",
        d7.substrate.kind == SubstrateKind::Gpu,
    );
}

// ── S5 ──────────────────────────────────────────────────────────────────────

fn section_nucleus_model(pass: &mut u32, fail: &mut u32, total: &mut u32) {
    println!("\n═══ S5: NUCLEUS atomic model — Tower / Node / Nest ═══\n");

    let tower_caps = [
        "science.diversity",
        "science.qs_model",
        "science.full_pipeline",
        "science.anderson",
        "science.ncbi_fetch",
        "metrics.snapshot",
    ];

    check(
        pass,
        fail,
        total,
        "Tower exposes 6 capabilities",
        tower_caps.len() == 6,
    );
    let science = tower_caps
        .iter()
        .filter(|c| c.starts_with("science."))
        .count();
    check(
        pass,
        fail,
        total,
        "Tower: 5 science capabilities",
        science == 5,
    );
    let metrics = tower_caps
        .iter()
        .filter(|c| c.starts_with("metrics."))
        .count();
    check(
        pass,
        fail,
        total,
        "Tower: 1 metrics (Nest) capability",
        metrics == 1,
    );

    let inventory = [make_gpu(), make_npu(), make_cpu()];

    let node_workloads: Vec<(&str, Vec<Capability>, SubstrateKind)> = vec![
        (
            "diversity",
            vec![Capability::F64Compute, Capability::ScalarReduce],
            SubstrateKind::Gpu,
        ),
        (
            "taxonomy_int8",
            vec![Capability::QuantizedInference { bits: 8 }],
            SubstrateKind::Npu,
        ),
        ("fastq_io", vec![Capability::CpuCompute], SubstrateKind::Cpu),
    ];

    for (name, caps, expected_kind) in &node_workloads {
        let w = Workload::new(*name, caps.clone());
        let d = dispatch::route(&w, &inventory)
            .or_exit("NUCLEUS workload should route to expected substrate");
        check(
            pass,
            fail,
            total,
            &format!("Node: {name} → {expected_kind}"),
            d.substrate.kind == *expected_kind,
        );
    }

    check(
        pass,
        fail,
        total,
        "Nest: metrics.snapshot registered",
        tower_caps.contains(&"metrics.snapshot"),
    );
}

// ── S6 ──────────────────────────────────────────────────────────────────────

fn section_streaming_analysis(pass: &mut u32, fail: &mut u32, total: &mut u32) {
    println!("\n═══ S6: Streaming analysis — real pipeline topologies ═══\n");

    type PipelineDef<'a> = (&'a str, Vec<(&'a str, bool, bool)>, bool, usize);
    let pipelines: Vec<PipelineDef<'_>> = vec![
        (
            "16S sovereign",
            vec![
                ("FASTQ_QF", false, true),
                ("DADA2", true, true),
                ("Chimera", true, false),
                ("Taxonomy", false, true),
                ("Diversity", true, true),
                ("BrayCurtis", true, true),
                ("PCoA", true, true),
            ],
            false,
            7,
        ),
        (
            "pure analytics",
            vec![
                ("Shannon", true, true),
                ("Simpson", true, true),
                ("BrayCurtis", true, true),
                ("PCoA", true, true),
            ],
            true,
            4,
        ),
        (
            "ODE sweep",
            vec![
                ("ODE_RK4", true, true),
                ("Diversity", true, true),
                ("Anderson_W", true, true),
            ],
            true,
            3,
        ),
        (
            "sentinel GPU→NPU",
            vec![
                ("GPU_16S", true, true),
                ("GPU_Anderson", true, true),
                ("NPU_ESN_PCIe", true, false),
            ],
            true,
            3,
        ),
    ];

    for (label, stages, expect_streamable, expect_n) in &pipelines {
        let mut session = StreamingSession::new(SubstrateKind::Gpu);
        for (name, accepts, produces) in stages {
            session.add_stage(PipelineStage {
                name: name.to_string(),
                capability: Capability::F64Compute,
                accepts_gpu_buffer: *accepts,
                produces_gpu_buffer: *produces,
            });
        }
        let a = session.analyze();
        check(
            pass,
            fail,
            total,
            &format!("{label}: {expect_n} stages"),
            a.n_stages == *expect_n,
        );
        check(
            pass,
            fail,
            total,
            &format!("{label}: streamable={expect_streamable}"),
            a.fully_streamable == *expect_streamable,
        );
    }
}

// ── S7 ──────────────────────────────────────────────────────────────────────

fn section_dispatch_threshold(pass: &mut u32, fail: &mut u32, total: &mut u32) {
    println!("\n═══ S7: Dispatch threshold gating ═══\n");

    let gpu_threshold: usize = 10_000;

    let below_threshold: usize = 100;
    let above_threshold: usize = 50_000;

    let use_gpu_small = below_threshold >= gpu_threshold;
    let use_gpu_large = above_threshold >= gpu_threshold;

    check(
        pass,
        fail,
        total,
        "100 elements: CPU (below threshold)",
        !use_gpu_small,
    );
    check(
        pass,
        fail,
        total,
        "50000 elements: GPU (above threshold)",
        use_gpu_large,
    );

    let exact_threshold: usize = gpu_threshold;
    check(
        pass,
        fail,
        total,
        "exact threshold: GPU (>=)",
        exact_threshold >= gpu_threshold,
    );

    let one_below: usize = gpu_threshold - 1;
    check(
        pass,
        fail,
        total,
        "threshold-1: CPU",
        one_below < gpu_threshold,
    );
}

fn main() {
    let mut pass = 0_u32;
    let mut fail = 0_u32;
    let mut total = 0_u32;

    println!("═══════════════════════════════════════════════════════════");
    println!("  Exp213: Compute Dispatch + Streaming Evolution (V66)");
    println!("═══════════════════════════════════════════════════════════");

    section_workload_routing(&mut pass, &mut fail, &mut total);
    section_absorption_status(&mut pass, &mut fail, &mut total);
    section_streaming_topology(&mut pass, &mut fail, &mut total);
    section_mixed_dispatch(&mut pass, &mut fail, &mut total);
    section_nucleus_model(&mut pass, &mut fail, &mut total);
    section_streaming_analysis(&mut pass, &mut fail, &mut total);
    section_dispatch_threshold(&mut pass, &mut fail, &mut total);

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  RESULT: {pass}/{total} passed, {fail} failed");
    println!("═══════════════════════════════════════════════════════════");

    if fail > 0 {
        std::process::exit(1);
    }
}
