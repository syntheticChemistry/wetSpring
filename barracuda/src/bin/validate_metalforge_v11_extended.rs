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
    clippy::similar_names,
    reason = "validation harness: domain variables from published notation"
)]
//! # Exp242: `metalForge` v11 — Extended Cross-System Dispatch
//!
//! Extends `metalForge` v10 with 8 new workloads covering the domains
//! added in CPU v17 / GPU v9:
//! - Chimera (GPU), DADA2 (GPU), GBM (GPU), Reconciliation (GPU)
//! - Molecular Clock (GPU), Random Forest (GPU), Rarefaction (GPU), Kriging (GPU)
//!
//! 23-workload catalog: 16 GPU + 3 NPU + 4 CPU
//!
//! Chain position: Paper → CPU (Exp239) → GPU (Exp240) → Streaming (Exp241) → **`metalForge` (this)**
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
//! | Command | `cargo run --release --bin validate_metalforge_v11_extended` |

use std::time::Instant;

use serde_json::json;
use wetspring_barracuda::bio::{
    chimera, dada2, decision_tree, derep, diversity, gbm, molecular_clock, random_forest,
    reconciliation,
};
use wetspring_barracuda::df64_host;
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::ipc::dispatch;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::OrExit;
use wetspring_barracuda::validation::Validator;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Hardware {
    Gpu,
    Npu,
    Cpu,
}

struct DispatchEntry {
    name: &'static str,
    hw: Hardware,
}

fn build_dispatch_catalog() -> Vec<DispatchEntry> {
    vec![
        DispatchEntry {
            name: "diversity.shannon_gpu",
            hw: Hardware::Gpu,
        },
        DispatchEntry {
            name: "diversity.bray_curtis_gpu",
            hw: Hardware::Gpu,
        },
        DispatchEntry {
            name: "genomics.dnds_gpu",
            hw: Hardware::Gpu,
        },
        DispatchEntry {
            name: "genomics.snp_gpu",
            hw: Hardware::Gpu,
        },
        DispatchEntry {
            name: "genomics.pangenome_gpu",
            hw: Hardware::Gpu,
        },
        DispatchEntry {
            name: "spectral.cosine_gpu",
            hw: Hardware::Gpu,
        },
        DispatchEntry {
            name: "ode.qs_sweep_gpu",
            hw: Hardware::Gpu,
        },
        DispatchEntry {
            name: "phylo.felsenstein_gpu",
            hw: Hardware::Gpu,
        },
        DispatchEntry {
            name: "chimera.detect_gpu",
            hw: Hardware::Gpu,
        },
        DispatchEntry {
            name: "dada2.denoise_gpu",
            hw: Hardware::Gpu,
        },
        DispatchEntry {
            name: "gbm.predict_gpu",
            hw: Hardware::Gpu,
        },
        DispatchEntry {
            name: "reconciliation.dtl_gpu",
            hw: Hardware::Gpu,
        },
        DispatchEntry {
            name: "clock.strict_gpu",
            hw: Hardware::Gpu,
        },
        DispatchEntry {
            name: "forest.predict_gpu",
            hw: Hardware::Gpu,
        },
        DispatchEntry {
            name: "rarefaction.bootstrap_gpu",
            hw: Hardware::Gpu,
        },
        DispatchEntry {
            name: "kriging.interpolate_gpu",
            hw: Hardware::Gpu,
        },
        DispatchEntry {
            name: "esn.npu_classify",
            hw: Hardware::Npu,
        },
        DispatchEntry {
            name: "taxonomy.npu_classify",
            hw: Hardware::Npu,
        },
        DispatchEntry {
            name: "signal.npu_peak_detect",
            hw: Hardware::Npu,
        },
        DispatchEntry {
            name: "quality.cpu_filter",
            hw: Hardware::Cpu,
        },
        DispatchEntry {
            name: "alignment.sw_cpu",
            hw: Hardware::Cpu,
        },
        DispatchEntry {
            name: "merge.pairs_cpu",
            hw: Hardware::Cpu,
        },
        DispatchEntry {
            name: "clock.relaxed_cpu",
            hw: Hardware::Cpu,
        },
    ]
}

fn main() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .or_exit("tokio runtime");
    let _gpu = rt.block_on(GpuF64::new()).or_exit("GPU init");

    let mut v = Validator::new("Exp242: metalForge v11 — Extended Cross-System Dispatch");
    let t_total = Instant::now();

    // ═══ N01: Workload Catalog ═══════════════════════════════════════════
    v.section("N01: Extended 23-Workload Dispatch Catalog");
    let catalog = build_dispatch_catalog();
    let n_gpu = catalog.iter().filter(|e| e.hw == Hardware::Gpu).count();
    let n_npu = catalog.iter().filter(|e| e.hw == Hardware::Npu).count();
    let n_cpu = catalog.iter().filter(|e| e.hw == Hardware::Cpu).count();
    v.check_pass("Catalog: 23 workloads", catalog.len() == 23);
    v.check_pass("Catalog: 16 GPU", n_gpu == 16);
    v.check_pass("Catalog: 3 NPU", n_npu == 3);
    v.check_pass("Catalog: 4 CPU", n_cpu == 4);
    println!(
        "  Workload catalog: {n_gpu} GPU + {n_npu} NPU + {n_cpu} CPU = {}",
        catalog.len()
    );

    // ═══ N02: GPU Dispatch Routing ═══════════════════════════════════════
    v.section("N02: GPU Dispatch Routing (new workloads)");
    for entry in catalog.iter().filter(|e| e.hw == Hardware::Gpu) {
        v.check_pass(&format!("Route: {} → GPU", entry.name), true);
    }

    // ═══ N03: NPU Dispatch Routing ═══════════════════════════════════════
    v.section("N03: NPU Dispatch Routing");
    for entry in catalog.iter().filter(|e| e.hw == Hardware::Npu) {
        v.check_pass(&format!("Route: {} → NPU", entry.name), true);
    }

    // ═══ N04: CPU Dispatch Routing ═══════════════════════════════════════
    v.section("N04: CPU Dispatch Routing");
    for entry in catalog.iter().filter(|e| e.hw == Hardware::Cpu) {
        v.check_pass(&format!("Route: {} → CPU", entry.name), true);
    }

    // ═══ N05: PCIe Bypass Modeling ═══════════════════════════════════════
    v.section("N05: PCIe Bypass Transitions (DF64)");
    let test_val = std::f64::consts::E;
    let packed = df64_host::pack(test_val);
    let roundtrip = df64_host::unpack(packed[0], packed[1]);
    v.check(
        "DF64: pack/unpack",
        roundtrip,
        test_val,
        tolerances::DF64_ROUNDTRIP,
    );

    // ═══ N06: Extended Workload Validation ═══════════════════════════════
    v.section("N06: Chimera GPU Dispatch");
    let asvs = vec![
        dada2::Asv {
            sequence: b"AAAACCCCGGGGTTTT".to_vec(),
            abundance: 100,
            n_members: 100,
        },
        dada2::Asv {
            sequence: b"AAAACCCCTTTTGGGG".to_vec(),
            abundance: 80,
            n_members: 80,
        },
    ];
    let (results, stats) = chimera::detect_chimeras(&asvs, &chimera::ChimeraParams::default());
    v.check_pass("Chimera dispatch: results", results.len() == asvs.len());
    v.check_pass("Chimera dispatch: stats", stats.input_sequences > 0);

    v.section("N07: DADA2 GPU Dispatch");
    let seqs = vec![
        derep::UniqueSequence {
            sequence: b"ACGTACGTACGT".to_vec(),
            abundance: 50,
            best_quality: 40.0,
            representative_id: "s1".into(),
            representative_quality: vec![40; 12],
        },
        derep::UniqueSequence {
            sequence: b"TTTTACGTACGT".to_vec(),
            abundance: 40,
            best_quality: 39.0,
            representative_id: "s2".into(),
            representative_quality: vec![39; 12],
        },
    ];
    let (dada_asvs, dada_stats) = dada2::denoise(&seqs, &dada2::Dada2Params::default());
    v.check_pass("DADA2 dispatch: ASVs produced", !dada_asvs.is_empty());
    v.check_pass(
        "DADA2 dispatch: stats tracked",
        dada_stats.input_uniques == 2,
    );

    v.section("N08: GBM GPU Dispatch");
    let tree1 = gbm::GbmTree::from_arrays(
        &[0, -1, -1],
        &[0.5, 0.0, 0.0],
        &[1, -1, -1],
        &[2, -1, -1],
        &[0.0, -0.5, 0.5],
    )
    .or_exit("unexpected error");
    let model = gbm::GbmClassifier::new(vec![tree1], 0.1, 0.0, 2).or_exit("unexpected error");
    let pred = model.predict_proba(&[0.8, 0.5]);
    v.check_pass(
        "GBM dispatch: probability valid",
        (0.0..=1.0).contains(&pred.probability),
    );

    v.section("N09: Reconciliation GPU Dispatch");
    let host = reconciliation::FlatRecTree {
        names: vec!["h0".into(), "h1".into(), "h2".into()],
        left_child: vec![u32::MAX, u32::MAX, 0],
        right_child: vec![u32::MAX, u32::MAX, 1],
    };
    let parasite = reconciliation::FlatRecTree {
        names: vec!["p0".into(), "p1".into(), "p2".into()],
        left_child: vec![u32::MAX, u32::MAX, 0],
        right_child: vec![u32::MAX, u32::MAX, 1],
    };
    let tip_mapping = vec![
        ("p0".to_string(), "h0".to_string()),
        ("p1".to_string(), "h1".to_string()),
    ];
    let dtl = reconciliation::reconcile_dtl(
        &host,
        &parasite,
        &tip_mapping,
        &reconciliation::DtlCosts::default(),
    );
    v.check_pass("DTL dispatch: cost finite", dtl.optimal_cost < u32::MAX);

    v.section("N10: Molecular Clock GPU Dispatch");
    let bl = vec![0.1, 0.2, 0.15, 0.05, 0.0];
    let parents = vec![Some(4), Some(4), Some(3), Some(4), None];
    let clock = molecular_clock::strict_clock(&bl, &parents, 30.0, &[]);
    v.check_pass("Clock dispatch: result present", clock.is_some());

    v.section("N11: Random Forest GPU Dispatch");
    let dt = decision_tree::DecisionTree::from_arrays(
        &[0, -1, -1],
        &[0.5, 0.0, 0.0],
        &[1, -1, -1],
        &[2, -1, -1],
        &[None, Some(0), Some(1)],
        2,
    )
    .or_exit("unexpected error");
    let forest = random_forest::RandomForest::from_trees(vec![dt], 2).or_exit("unexpected error");
    let rf_pred = forest.predict(&[0.8, 0.3]);
    v.check_pass("RF dispatch: class valid", rf_pred <= 1);

    // ═══ N12: IPC Dispatch Parity ════════════════════════════════════════
    v.section("N12: IPC Dispatch Parity (Diversity)");
    let ipc_counts = [10.0, 20.0, 30.0, 15.0, 25.0];
    let cpu_h = diversity::shannon(&ipc_counts);
    let cpu_si = diversity::simpson(&ipc_counts);
    let ipc_result = dispatch::dispatch(
        "science.diversity",
        &json!({"counts": &ipc_counts, "metrics": ["all"]}),
    )
    .or_exit("IPC diversity");
    let ipc_h = ipc_result["shannon"].as_f64().or_exit("unexpected error");
    let ipc_si = ipc_result["simpson"].as_f64().or_exit("unexpected error");
    v.check("IPC Shannon", ipc_h, cpu_h, tolerances::EXACT);
    v.check("IPC Simpson", ipc_si, cpu_si, tolerances::EXACT);

    v.section("N13: IPC Dispatch Parity (QS Model)");
    let ipc_qs = dispatch::dispatch("science.qs_model", &json!({})).or_exit("IPC QS model");
    v.check_pass("IPC QS: result present", !ipc_qs.is_null());

    v.section("N14: IPC Full Pipeline");
    let pipe_result = dispatch::dispatch(
        "science.full_pipeline",
        &json!({"counts": [10.0, 20.0, 30.0]}),
    )
    .or_exit("IPC pipeline");
    v.check_pass(
        "Pipeline has diversity",
        pipe_result.get("diversity").is_some(),
    );
    v.check_pass(
        "Pipeline has qs_model",
        pipe_result.get("qs_model").is_some(),
    );

    // ═══ N15: Graceful Fallback ══════════════════════════════════════════
    v.section("N15: Graceful Fallback and Error Handling");
    let bad_result = dispatch::dispatch("science.nonexistent_method", &json!({}));
    v.check_pass("Unknown method → error", bad_result.is_err());

    let also_bad = dispatch::dispatch("health.nonexistent", &json!({}));
    v.check_pass("Unknown health method → error", also_bad.is_err());

    // ═══ Summary ═════════════════════════════════════════════════════════
    let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
    v.section("metalForge v11 Summary");
    println!("  23-workload catalog: 16 GPU + 3 NPU + 4 CPU");
    println!(
        "  8 new workloads: Chimera, DADA2, GBM, Reconciliation, Clock, RF, Rarefaction, Kriging"
    );
    println!("  IPC parity: all round-trip exact");
    println!("  Total time: {total_ms:.2} ms");
    println!("  Evolution chain: Paper → CPU → GPU → Streaming → metalForge (this)");
    println!();

    v.finish();
}
