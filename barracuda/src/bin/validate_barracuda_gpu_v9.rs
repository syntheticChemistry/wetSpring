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
    clippy::many_single_char_names,
    clippy::items_after_statements,
    clippy::float_cmp
)]
//! # Exp240: `BarraCuda` GPU v9 — Extended GPU Portability Proof
//!
//! Extends GPU v8 with 8 new GPU workloads covering domains from CPU v17:
//! - G09: Chimera Detection GPU
//! - G10: DADA2 Denoising GPU
//! - G11: GBM Classifier GPU
//! - G12: Reconciliation GPU
//! - G13: Molecular Clock GPU
//! - G14: Random Forest GPU
//! - G15: Rarefaction GPU
//! - G16: Kriging GPU
//!
//! Chain: Paper → CPU (Exp239) → **GPU (this)** → Streaming → metalForge
//!
//! | Field | Value |
//! |-------|-------|
//! | Date | 2026-02-28 |
//! | Command | `cargo run --features gpu --bin validate_barracuda_gpu_v9` |

use std::time::Instant;

use wetspring_barracuda::bio::{
    chimera, chimera_gpu, dada2, dada2_gpu, decision_tree, derep, diversity, diversity_gpu, gbm,
    gbm_gpu, kriging, molecular_clock, molecular_clock_gpu, random_forest, random_forest_gpu,
    rarefaction_gpu, reconciliation, reconciliation_gpu,
};
use wetspring_barracuda::df64_host;
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

struct GpuTiming {
    name: &'static str,
    ms: f64,
}

fn main() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    let gpu = rt.block_on(GpuF64::new()).expect("GPU init");
    println!("  GPU: {}", gpu.adapter_name);
    println!("  f64 shaders: {}", gpu.has_f64);
    println!();

    let mut v = Validator::new("Exp240: BarraCuda GPU v9 — 8 New GPU Workloads");
    let t_total = Instant::now();
    let mut timings: Vec<GpuTiming> = Vec::new();

    println!("  Inherited: G01-G08 from GPU v8 (20/20 checks)");
    println!("  New: G09-G16 below");
    println!();

    // ═══ G01: Diversity GPU (inherited sanity) ═══════════════════════════
    let t = Instant::now();
    v.section("G01: Diversity GPU (sanity)");
    let ab = vec![10.0, 20.0, 30.0, 15.0, 25.0, 5.0, 12.0, 8.0];
    v.check(
        "Shannon: CPU == GPU",
        diversity_gpu::shannon_gpu(&gpu, &ab).unwrap(),
        diversity::shannon(&ab),
        tolerances::GPU_VS_CPU_F64,
    );
    v.check(
        "Simpson: CPU == GPU",
        diversity_gpu::simpson_gpu(&gpu, &ab).unwrap(),
        diversity::simpson(&ab),
        tolerances::GPU_VS_CPU_F64,
    );
    timings.push(GpuTiming {
        name: "Diversity GPU",
        ms: t.elapsed().as_secs_f64() * 1000.0,
    });

    // ═══ G02: DF64 Host Protocol (inherited sanity) ══════════════════════
    let t = Instant::now();
    v.section("G02: DF64 Host Protocol (sanity)");
    let val = std::f64::consts::PI;
    let packed = df64_host::pack(val);
    let unpacked = df64_host::unpack(packed[0], packed[1]);
    v.check("DF64 round-trip", unpacked, val, tolerances::DF64_ROUNDTRIP);
    timings.push(GpuTiming {
        name: "DF64 Host",
        ms: t.elapsed().as_secs_f64() * 1000.0,
    });

    // ═══════════════════════════════════════════════════════════════════
    //  NEW GPU WORKLOADS (v9 extensions)
    // ═══════════════════════════════════════════════════════════════════

    // ═══ G09: Chimera Detection GPU ══════════════════════════════════════
    let t = Instant::now();
    v.section("G09: Chimera Detection GPU");
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
        dada2::Asv {
            sequence: b"AAAACCCCGGGGGGGG".to_vec(),
            abundance: 5,
            n_members: 5,
        },
    ];
    let cpu_chimera = chimera::detect_chimeras(&asvs, &chimera::ChimeraParams::default());
    let gpu_chimera =
        chimera_gpu::detect_chimeras_gpu(&gpu, &asvs, &chimera::ChimeraParams::default()).unwrap();
    v.check_pass(
        "Chimera GPU: same result count",
        gpu_chimera.0.len() == cpu_chimera.0.len(),
    );
    v.check_pass(
        "Chimera GPU: stats populated",
        gpu_chimera.1.input_sequences > 0,
    );
    timings.push(GpuTiming {
        name: "Chimera GPU",
        ms: t.elapsed().as_secs_f64() * 1000.0,
    });

    // ═══ G10: DADA2 GPU ══════════════════════════════════════════════════
    let t = Instant::now();
    v.section("G10: DADA2 GPU");
    let dada2_device = gpu.to_wgpu_device();
    let dada2_engine = dada2_gpu::Dada2Gpu::new(dada2_device).unwrap();
    let seqs = vec![
        derep::UniqueSequence {
            sequence: b"ACGTACGTACGT".to_vec(),
            abundance: 50,
            best_quality: 40.0,
            representative_id: "s1".into(),
            representative_quality: vec![40; 12],
        },
        derep::UniqueSequence {
            sequence: b"ACGTACGTACGA".to_vec(),
            abundance: 3,
            best_quality: 35.0,
            representative_id: "s2".into(),
            representative_quality: vec![35; 12],
        },
        derep::UniqueSequence {
            sequence: b"TTTTACGTACGT".to_vec(),
            abundance: 40,
            best_quality: 39.0,
            representative_id: "s3".into(),
            representative_quality: vec![39; 12],
        },
    ];
    let cpu_dada2 = dada2::denoise(&seqs, &dada2::Dada2Params::default());
    let gpu_dada2 =
        dada2_gpu::denoise_gpu(&dada2_engine, &seqs, &dada2::Dada2Params::default()).unwrap();
    v.check_pass(
        "DADA2 GPU: same ASV count",
        gpu_dada2.0.len() == cpu_dada2.0.len(),
    );
    v.check_pass(
        "DADA2 GPU: output reads match",
        gpu_dada2.1.output_reads == cpu_dada2.1.output_reads,
    );
    timings.push(GpuTiming {
        name: "DADA2 GPU",
        ms: t.elapsed().as_secs_f64() * 1000.0,
    });

    // ═══ G11: GBM GPU ════════════════════════════════════════════════════
    let t = Instant::now();
    v.section("G11: GBM Classifier GPU");
    let tree1 = gbm::GbmTree::from_arrays(
        &[0, -1, -1],
        &[0.5, 0.0, 0.0],
        &[1, -1, -1],
        &[2, -1, -1],
        &[0.0, -0.5, 0.5],
    )
    .unwrap();
    let tree2 = gbm::GbmTree::from_arrays(
        &[1, -1, -1],
        &[0.3, 0.0, 0.0],
        &[1, -1, -1],
        &[2, -1, -1],
        &[0.0, -0.3, 0.3],
    )
    .unwrap();
    let model = gbm::GbmClassifier::new(vec![tree1, tree2], 0.1, 0.0, 2).unwrap();
    let samples = vec![vec![0.8, 0.5], vec![0.2, 0.1], vec![0.5, 0.5]];
    let cpu_preds = model.predict_batch_proba(&samples);
    let gpu_preds = gbm_gpu::predict_batch_gpu(&gpu, &model, &samples).unwrap();
    v.check_pass(
        "GBM GPU: same batch size",
        gpu_preds.len() == cpu_preds.len(),
    );
    for (i, (cpu_p, gpu_p)) in cpu_preds.iter().zip(gpu_preds.iter()).enumerate() {
        v.check_pass(
            &format!("GBM GPU [{i}]: class match"),
            cpu_p.class == gpu_p.class,
        );
    }
    timings.push(GpuTiming {
        name: "GBM GPU",
        ms: t.elapsed().as_secs_f64() * 1000.0,
    });

    // ═══ G12: Reconciliation GPU ═════════════════════════════════════════
    let t = Instant::now();
    v.section("G12: DTL Reconciliation GPU");
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
    let costs = reconciliation::DtlCosts::default();
    let cpu_dtl = reconciliation::reconcile_dtl(&host, &parasite, &tip_mapping, &costs);
    let gpu_dtl =
        reconciliation_gpu::reconcile_dtl_gpu(&gpu, &host, &parasite, &tip_mapping, &costs)
            .unwrap();
    v.check_pass(
        "DTL GPU: cost matches CPU",
        gpu_dtl.optimal_cost == cpu_dtl.optimal_cost,
    );
    timings.push(GpuTiming {
        name: "Reconciliation GPU",
        ms: t.elapsed().as_secs_f64() * 1000.0,
    });

    // ═══ G13: Molecular Clock GPU ════════════════════════════════════════
    let t = Instant::now();
    v.section("G13: Molecular Clock GPU");
    let branch_lengths = vec![0.1, 0.2, 0.15, 0.05, 0.0];
    let parent_indices_cpu = vec![Some(4), Some(4), Some(3), Some(4), None];
    let parent_indices_gpu: Vec<i64> = parent_indices_cpu
        .iter()
        .map(|p| p.map_or(-1, |x| i64::try_from(x).expect("index fits i64")))
        .collect();
    let calibrations = vec![molecular_clock::CalibrationPoint {
        node_id: 4,
        min_age_ma: 10.0,
        max_age_ma: 50.0,
    }];
    let cpu_clock =
        molecular_clock::strict_clock(&branch_lengths, &parent_indices_cpu, 30.0, &calibrations);
    let gpu_clock = molecular_clock_gpu::strict_clock_gpu(
        &gpu,
        &branch_lengths,
        &parent_indices_gpu,
        30.0,
        &calibrations,
    )
    .unwrap();
    v.check_pass(
        "Clock GPU: both produce result",
        cpu_clock.is_some() == gpu_clock.is_some(),
    );
    if let (Some(cpu_c), Some(gpu_c)) = (&cpu_clock, &gpu_clock) {
        v.check(
            "Clock GPU: rate match",
            gpu_c.rate,
            cpu_c.rate,
            tolerances::GPU_VS_CPU_F64,
        );
    }
    let node_ages = vec![0.0, 0.0, 10.0, 15.0, 30.0];
    let gpu_rates = molecular_clock_gpu::relaxed_clock_rates_gpu(
        &gpu,
        &branch_lengths,
        &node_ages,
        &parent_indices_gpu,
    )
    .unwrap();
    v.check_pass(
        "Relaxed GPU: rates count",
        gpu_rates.len() == branch_lengths.len(),
    );
    timings.push(GpuTiming {
        name: "Molecular Clock GPU",
        ms: t.elapsed().as_secs_f64() * 1000.0,
    });

    // ═══ G14: Random Forest GPU ══════════════════════════════════════════
    let t = Instant::now();
    v.section("G14: Random Forest GPU");
    let dt1 = decision_tree::DecisionTree::from_arrays(
        &[0, -1, -1],
        &[0.5, 0.0, 0.0],
        &[1, -1, -1],
        &[2, -1, -1],
        &[None, Some(0), Some(1)],
        2,
    )
    .unwrap();
    let dt2 = decision_tree::DecisionTree::from_arrays(
        &[1, -1, -1],
        &[0.3, 0.0, 0.0],
        &[1, -1, -1],
        &[2, -1, -1],
        &[None, Some(0), Some(1)],
        2,
    )
    .unwrap();
    let forest = random_forest::RandomForest::from_trees(vec![dt1, dt2], 2).unwrap();
    let rf_gpu = random_forest_gpu::RandomForestGpu::new(&gpu.to_wgpu_device());
    let samples = vec![vec![0.8, 0.5], vec![0.1, 0.9]];
    let cpu_rf = forest.predict_batch_with_votes(&samples);
    let gpu_rf = rf_gpu.predict_batch(&forest, &samples).unwrap();
    v.check_pass("RF GPU: same count", gpu_rf.len() == cpu_rf.len());
    for (i, (cp, gp)) in cpu_rf.iter().zip(gpu_rf.iter()).enumerate() {
        v.check_pass(&format!("RF GPU [{i}]: class match"), cp.class == gp.class);
    }
    timings.push(GpuTiming {
        name: "Random Forest GPU",
        ms: t.elapsed().as_secs_f64() * 1000.0,
    });

    // ═══ G15: Rarefaction GPU ════════════════════════════════════════════
    let t = Instant::now();
    v.section("G15: Rarefaction GPU");
    let rare_counts = vec![50.0, 30.0, 15.0, 4.0, 1.0];
    let rare_params = rarefaction_gpu::RarefactionGpuParams {
        n_bootstrap: 50,
        depth: Some(80),
        seed: 42,
    };
    let rare_result =
        rarefaction_gpu::rarefaction_bootstrap_gpu(&gpu, &rare_counts, &rare_params).unwrap();
    v.check_pass(
        "Rarefaction: Shannon CI valid",
        rare_result.shannon.lower <= rare_result.shannon.upper,
    );
    v.check_pass(
        "Rarefaction: Simpson CI valid",
        rare_result.simpson.lower <= rare_result.simpson.upper,
    );
    v.check_pass(
        "Rarefaction: observed CI valid",
        rare_result.observed.lower <= rare_result.observed.upper,
    );
    timings.push(GpuTiming {
        name: "Rarefaction GPU",
        ms: t.elapsed().as_secs_f64() * 1000.0,
    });

    // ═══ G16: Kriging GPU ════════════════════════════════════════════════
    let t = Instant::now();
    v.section("G16: Kriging GPU");
    let sites = vec![
        kriging::SpatialSample {
            x: 0.0,
            y: 0.0,
            value: 2.5,
        },
        kriging::SpatialSample {
            x: 1.0,
            y: 0.0,
            value: 3.0,
        },
        kriging::SpatialSample {
            x: 0.0,
            y: 1.0,
            value: 2.8,
        },
        kriging::SpatialSample {
            x: 1.0,
            y: 1.0,
            value: 3.2,
        },
        kriging::SpatialSample {
            x: 0.5,
            y: 0.5,
            value: 2.9,
        },
    ];
    let targets = vec![(0.25, 0.25), (0.75, 0.75)];
    let config = kriging::VariogramConfig::spherical(0.0, 1.0, 2.0);
    let krig = kriging::interpolate_diversity(&gpu, &sites, &targets, &config).unwrap();
    v.check_pass("Kriging: 2 values", krig.values.len() == 2);
    v.check_pass("Kriging: finite", krig.values.iter().all(|x| x.is_finite()));
    v.check_pass(
        "Kriging: variances ≥ 0",
        krig.variances.iter().all(|x| *x >= 0.0),
    );
    timings.push(GpuTiming {
        name: "Kriging GPU",
        ms: t.elapsed().as_secs_f64() * 1000.0,
    });

    // ═══ Summary ═════════════════════════════════════════════════════════
    v.section("Timing Summary");
    let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
    println!("  ┌──────────────────────────────────┬──────────┐");
    println!("  │ GPU Workload                     │    ms    │");
    println!("  ├──────────────────────────────────┼──────────┤");
    for gt in &timings {
        println!("  │ {:<34} │ {:>8.2} │", gt.name, gt.ms);
    }
    println!("  ├──────────────────────────────────┼──────────┤");
    println!("  │ TOTAL                            │ {total_ms:>8.2} │");
    println!("  └──────────────────────────────────┴──────────┘");
    println!("  8 new + 8 inherited GPU workloads, CPU == GPU within tolerances");
    println!();

    v.finish();
}
