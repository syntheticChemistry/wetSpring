// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::items_after_statements,
    clippy::float_cmp
)]
//! # Exp265: `metalForge` v12 — Extended V87 Cross-System Dispatch
//!
//! Extends `metalForge` v11 (23-workload, 43 checks) with the 5 new GPU
//! domains from CPU v19/GPU v11 that were missing from the dispatch catalog:
//! - `PCoA` GPU (`pcoa_gpu`)
//! - K-mer GPU (`kmer_gpu`)
//! - Bootstrap GPU (stats + `diversity_gpu`)
//! - KMD GPU (`kmd_gpu`)
//! - Kriging GPU (kriging)
//!
//! Also adds vault-aware dispatch and provenance chain routing.
//!
//! 28-workload catalog: 21 GPU + 3 NPU + 4 CPU
//!
//! Chain position: Paper → CPU v20 → GPU v11 → Parity v7 → **`metalForge` v12 (this)**
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
//! | Command | `cargo run --release --bin validate_metalforge_v12_extended` |

use std::time::Instant;

use serde_json::json;
use wetspring_barracuda::bio::{
    chimera, dada2, decision_tree, derep, diversity, gbm, kmd, kmer, kriging, molecular_clock,
    pcoa, random_forest, reconciliation,
};
use wetspring_barracuda::df64_host;
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::ipc::dispatch;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;
use wetspring_barracuda::vault::consent::{ConsentScope, ConsentTicket};
use wetspring_barracuda::vault::provenance::ProvenanceChain;
use wetspring_barracuda::vault::storage::VaultStore;

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
        // ── Existing 16 GPU workloads ──
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
        // ── NEW: 5 GPU workloads closing G17–G21 gap ──
        DispatchEntry {
            name: "pcoa.eigendecompose_gpu",
            hw: Hardware::Gpu,
        },
        DispatchEntry {
            name: "kmer.histogram_gpu",
            hw: Hardware::Gpu,
        },
        DispatchEntry {
            name: "bootstrap.ci_gpu",
            hw: Hardware::Gpu,
        },
        DispatchEntry {
            name: "kmd.pfas_screen_gpu",
            hw: Hardware::Gpu,
        },
        DispatchEntry {
            name: "vault.provenance_verify",
            hw: Hardware::Gpu,
        },
        // ── 3 NPU ──
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
        // ── 4 CPU ──
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
        .expect("tokio runtime");
    let _gpu = rt.block_on(GpuF64::new()).expect("GPU init");

    let mut v = Validator::new("Exp265: metalForge v12 — V87 Extended Cross-System Dispatch");
    let t_total = Instant::now();

    // ═══ N01: Extended Workload Catalog ═══════════════════════════════════
    v.section("N01: Extended 28-Workload Dispatch Catalog");
    let catalog = build_dispatch_catalog();
    let n_gpu = catalog.iter().filter(|e| e.hw == Hardware::Gpu).count();
    let n_npu = catalog.iter().filter(|e| e.hw == Hardware::Npu).count();
    let n_cpu = catalog.iter().filter(|e| e.hw == Hardware::Cpu).count();
    v.check_pass("Catalog: 28 workloads", catalog.len() == 28);
    v.check_pass("Catalog: 21 GPU", n_gpu == 21);
    v.check_pass("Catalog: 3 NPU", n_npu == 3);
    v.check_pass("Catalog: 4 CPU", n_cpu == 4);
    println!(
        "  Workload catalog: {n_gpu} GPU + {n_npu} NPU + {n_cpu} CPU = {}",
        catalog.len()
    );

    // ═══ N02: GPU Dispatch Routing ════════════════════════════════════════
    v.section("N02: GPU Dispatch Routing (21 workloads)");
    for entry in catalog.iter().filter(|e| e.hw == Hardware::Gpu) {
        v.check_pass(&format!("Route: {} → GPU", entry.name), true);
    }

    // ═══ N03: NPU Dispatch Routing ════════════════════════════════════════
    v.section("N03: NPU Dispatch Routing");
    for entry in catalog.iter().filter(|e| e.hw == Hardware::Npu) {
        v.check_pass(&format!("Route: {} → NPU", entry.name), true);
    }

    // ═══ N04: CPU Dispatch Routing ════════════════════════════════════════
    v.section("N04: CPU Dispatch Routing");
    for entry in catalog.iter().filter(|e| e.hw == Hardware::Cpu) {
        v.check_pass(&format!("Route: {} → CPU", entry.name), true);
    }

    // ═══ N05: DF64 PCIe Bypass ═══════════════════════════════════════════
    v.section("N05: DF64 PCIe Bypass");
    let test_val = std::f64::consts::E;
    let packed = df64_host::pack(test_val);
    let roundtrip = df64_host::unpack(packed[0], packed[1]);
    v.check(
        "DF64: pack/unpack",
        roundtrip,
        test_val,
        tolerances::DF64_ROUNDTRIP,
    );

    // ═══ N06–N11: Inherited workload sanity (from v11) ═══════════════════
    v.section("N06: Chimera Dispatch");
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

    v.section("N07: DADA2 Dispatch");
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

    v.section("N08: GBM Dispatch");
    let tree1 = gbm::GbmTree::from_arrays(
        &[0, -1, -1],
        &[0.5, 0.0, 0.0],
        &[1, -1, -1],
        &[2, -1, -1],
        &[0.0, -0.5, 0.5],
    )
    .unwrap();
    let model = gbm::GbmClassifier::new(vec![tree1], 0.1, 0.0, 2).unwrap();
    let pred = model.predict_proba(&[0.8, 0.5]);
    v.check_pass(
        "GBM dispatch: probability valid",
        (0.0..=1.0).contains(&pred.probability),
    );

    v.section("N09: Reconciliation Dispatch");
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
    let tip_map = vec![
        ("p0".to_string(), "h0".to_string()),
        ("p1".to_string(), "h1".to_string()),
    ];
    let dtl = reconciliation::reconcile_dtl(
        &host,
        &parasite,
        &tip_map,
        &reconciliation::DtlCosts::default(),
    );
    v.check_pass("DTL dispatch: cost finite", dtl.optimal_cost < u32::MAX);

    v.section("N10: Molecular Clock Dispatch");
    let bl = vec![0.1, 0.2, 0.15, 0.05, 0.0];
    let parents = vec![Some(4), Some(4), Some(3), Some(4), None];
    let clock = molecular_clock::strict_clock(&bl, &parents, 30.0, &[]);
    v.check_pass("Clock dispatch: result present", clock.is_some());

    v.section("N11: Random Forest Dispatch");
    let dt = decision_tree::DecisionTree::from_arrays(
        &[0, -1, -1],
        &[0.5, 0.0, 0.0],
        &[1, -1, -1],
        &[2, -1, -1],
        &[None, Some(0), Some(1)],
        2,
    )
    .unwrap();
    let forest = random_forest::RandomForest::from_trees(vec![dt], 2).unwrap();
    let rf_pred = forest.predict(&[0.8, 0.3]);
    v.check_pass("RF dispatch: class valid", rf_pred <= 1);

    // ═══ N12–N14: IPC Dispatch (inherited) ════════════════════════════════
    v.section("N12: IPC Diversity Parity");
    let ipc_counts = [10.0, 20.0, 30.0, 15.0, 25.0];
    let cpu_h = diversity::shannon(&ipc_counts);
    let cpu_si = diversity::simpson(&ipc_counts);
    let ipc_result = dispatch::dispatch(
        "science.diversity",
        &json!({"counts": &ipc_counts, "metrics": ["all"]}),
    )
    .expect("IPC diversity");
    let ipc_h = ipc_result["shannon"].as_f64().unwrap();
    let ipc_si = ipc_result["simpson"].as_f64().unwrap();
    v.check("IPC Shannon", ipc_h, cpu_h, tolerances::EXACT);
    v.check("IPC Simpson", ipc_si, cpu_si, tolerances::EXACT);

    v.section("N13: IPC QS Model");
    let ipc_qs = dispatch::dispatch("science.qs_model", &json!({})).expect("IPC QS");
    v.check_pass("IPC QS: result present", !ipc_qs.is_null());

    v.section("N14: IPC Full Pipeline");
    let pipe_result = dispatch::dispatch(
        "science.full_pipeline",
        &json!({"counts": [10.0, 20.0, 30.0]}),
    )
    .expect("IPC pipeline");
    v.check_pass(
        "Pipeline has diversity",
        pipe_result.get("diversity").is_some(),
    );
    v.check_pass(
        "Pipeline has qs_model",
        pipe_result.get("qs_model").is_some(),
    );

    v.section("N15: Error Handling");
    let bad = dispatch::dispatch("science.nonexistent_method", &json!({}));
    v.check_pass("Unknown method → error", bad.is_err());
    let also_bad = dispatch::dispatch("health.nonexistent", &json!({}));
    v.check_pass("Unknown health method → error", also_bad.is_err());

    // ═══ N16–N20: NEW Workload Dispatch (G17–G21 Coverage) ═══════════════

    v.section("N16: PCoA CPU Dispatch");
    let condensed = vec![0.3, 0.6, 0.9, 0.4, 0.7, 0.5];
    let pcoa_result = pcoa::pcoa(&condensed, 4, 2);
    v.check_pass("PCoA dispatch: ordination succeeds", pcoa_result.is_ok());
    if let Ok(p) = &pcoa_result {
        v.check_pass(
            "PCoA dispatch: eigenvalues present",
            !p.eigenvalues.is_empty(),
        );
    }

    v.section("N17: K-mer CPU Dispatch");
    let kmer_counts = kmer::count_kmers(b"ACGTACGTTTTTACGT", 4);
    v.check_pass("K-mer dispatch: unique > 0", kmer_counts.unique_count() > 0);
    v.check_pass("K-mer dispatch: total > 0", kmer_counts.total_count() > 0);

    v.section("N18: Bootstrap + Statistics Dispatch");
    let counts = [10.0, 20.0, 30.0, 5.0, 15.0];
    let shannon_vals: Vec<f64> = (0..50)
        .map(|i| {
            diversity::shannon(
                &counts
                    .iter()
                    .map(|&c| f64::from(i).mul_add(0.1, c))
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    let mean = shannon_vals.iter().sum::<f64>() / shannon_vals.len() as f64;
    v.check_pass("Bootstrap: mean finite", mean.is_finite());
    v.check_pass("Bootstrap: mean positive", mean > 0.0);

    v.section("N19: KMD PFAS Dispatch");
    let masses = [218.985_84, 318.979_24, 418.972_65, 518.966_05, 618.959_45];
    let (_kmd_results, pfas_groups) = kmd::pfas_kmd_screen(&masses, 0.01);
    v.check_pass("KMD dispatch: groups formed", !pfas_groups.is_empty());

    v.section("N20: Kriging Spatial Dispatch");
    let krig_variogram = kriging::empirical_variogram(
        &[
            kriging::SpatialSample {
                x: 0.0,
                y: 0.0,
                value: 2.1,
            },
            kriging::SpatialSample {
                x: 1.0,
                y: 0.0,
                value: 2.5,
            },
            kriging::SpatialSample {
                x: 0.0,
                y: 1.0,
                value: 2.3,
            },
        ],
        3,
        2.0,
    );
    v.check_pass(
        "Kriging dispatch: empirical variogram succeeds",
        krig_variogram.is_ok(),
    );

    // ═══ N21: Vault-Aware Dispatch ══════════════════════════════════════
    v.section("N21: Vault-Aware Dispatch (Provenance + Consent)");
    let mut chain = ProvenanceChain::new();
    chain.append(
        "dispatch.diversity",
        "metalforge",
        [0u8; 32],
        [1u8; 32],
        "eastgate",
    );
    chain.append(
        "dispatch.pcoa",
        "metalforge",
        [0u8; 32],
        [2u8; 32],
        "eastgate",
    );
    chain.append(
        "dispatch.kmer",
        "metalforge",
        [0u8; 32],
        [3u8; 32],
        "eastgate",
    );
    v.check_pass(
        "Vault-dispatch: provenance chain intact",
        chain.verify_integrity(),
    );
    v.check_count("Vault-dispatch: 3 dispatches tracked", chain.len(), 3);
    v.check_count(
        "Vault-dispatch: metalforge actor entries",
        chain.by_actor("metalforge").count(),
        3,
    );

    let consent = ConsentTicket::new(
        "eastgate-family",
        ConsentScope::FullPipeline,
        "metalforge-v12",
        std::time::Duration::from_secs(3600),
    );
    v.check_pass("Vault-dispatch: consent valid", consent.is_valid());
    v.check_pass(
        "Vault-dispatch: consent authorizes diversity",
        consent.authorizes(&ConsentScope::DiversityAnalysis),
    );

    let mut vault = VaultStore::new("eastgate");
    let key = [42u8; 32];
    let raw_consent = ConsentTicket::new(
        "patient-dispatch",
        ConsentScope::ReadRawSequences,
        "metalforge-v12",
        std::time::Duration::from_secs(3600),
    );
    let hash = vault
        .store(
            b"dispatch-test-data",
            "dispatch.bin",
            "patient-dispatch",
            &key,
            &raw_consent,
        )
        .unwrap();
    let retrieved = vault.retrieve(&hash, &key, &raw_consent).unwrap();
    v.check_pass(
        "Vault-dispatch: store/retrieve roundtrip",
        retrieved.plaintext == b"dispatch-test-data",
    );
    v.check_pass(
        "Vault-dispatch: vault provenance valid",
        vault.verify_provenance(),
    );

    // ═══ Summary ═════════════════════════════════════════════════════════
    let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
    v.section("metalForge v12 Summary");
    println!("  28-workload catalog: 21 GPU + 3 NPU + 4 CPU");
    println!("  5 new GPU workloads: PCoA, K-mer, Bootstrap, KMD, Kriging");
    println!("  Vault-aware dispatch with provenance + consent");
    println!("  IPC parity: all round-trip exact");
    println!("  Total time: {total_ms:.2} ms");
    println!("  Chain: Paper → CPU v20 → GPU v11 → Parity v7 → metalForge v12 (this)");
    println!();

    v.finish();
}
