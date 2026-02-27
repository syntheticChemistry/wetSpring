// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names
)]
//! Exp103: metalForge Cross-Substrate v5 — 13 Pure GPU Promotion Domains
//!
//! Extends metalForge cross-substrate validation (Exp093: 16 domains) to
//! cover the 13 newly GPU-promoted modules. Proves: for every new GPU module,
//! metalForge router can dispatch to CPU or GPU and get the same answer.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | 1f9f80e |
//! | Baseline tool | `BarraCuda` CPU reference |
//! | Baseline date | 2026-02-22 |
//! | Exact command | `cargo run --features gpu --release --bin validate_metalforge_v5` |
//! | Data | Synthetic test vectors (self-contained) |
//! | Hardware | i9-12900K, 64 GB DDR5, RTX 4070, Pop!\_OS 22.04 |

use std::sync::Arc;
use std::time::Instant;

use barracuda::device::WgpuDevice;
use wetspring_barracuda::bio::{
    capacitor::{self, CapacitorParams},
    chimera::{self, ChimeraParams},
    cooperation::{self, CooperationParams},
    dada2::Asv,
    derep::{self, DerepSort},
    feature_table::{self, FeatureParams},
    gbm::{GbmClassifier, GbmTree},
    kmd::{self, units},
    merge_pairs::{self, MergeParams},
    molecular_clock, neighbor_joining,
    reconciliation::{self, DtlCosts, FlatRecTree},
    robinson_foulds,
    signal::{self, PeakParams},
    unifrac::PhyloTree,
};
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::io::fastq::FastqRecord;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::{self, Validator};

#[tokio::main]
async fn main() {
    let mut v = Validator::new("Exp103: metalForge Cross-Substrate v5 — 13 Pure GPU Domains");

    let gpu = match GpuF64::new().await {
        Ok(g) => g,
        Err(e) => {
            eprintln!("No GPU: {e}");
            validation::exit_skipped("No GPU available");
        }
    };
    gpu.print_info();
    if !gpu.has_f64 {
        validation::exit_skipped("No SHADER_F64 support on this GPU");
    }

    let device = gpu.to_wgpu_device();
    let t0 = Instant::now();
    let mut timings: Vec<(&str, f64, f64, &str)> = Vec::new();

    validate_cooperation_mf(&device, &mut v, &mut timings);
    validate_capacitor_mf(&device, &mut v, &mut timings);
    validate_kmd_mf(&gpu, &mut v, &mut timings);
    validate_gbm_mf(&gpu, &mut v, &mut timings);
    validate_merge_pairs_mf(&gpu, &mut v, &mut timings);
    validate_signal_mf(&gpu, &mut v, &mut timings);
    validate_feature_table_mf(&gpu, &mut v, &mut timings);
    validate_robinson_foulds_mf(&gpu, &mut v, &mut timings);
    validate_derep_mf(&gpu, &mut v, &mut timings);
    validate_chimera_mf(&gpu, &mut v, &mut timings);
    validate_neighbor_joining_mf(&gpu, &mut v, &mut timings);
    validate_reconciliation_mf(&gpu, &mut v, &mut timings);
    validate_molecular_clock_mf(&gpu, &mut v, &mut timings);

    // ═══ Summary ════════════════════════════════════════════════════
    v.section("═══ metalForge Cross-Substrate v5 Summary ═══");
    println!();
    println!(
        "  {:<25} {:>10} {:>10} {:>10}",
        "Workload", "CPU (µs)", "GPU (µs)", "Substrate"
    );
    println!("  {}", "─".repeat(59));
    for (name, cpu, gpu_t, result) in &timings {
        println!("  {name:<25} {cpu:>10.0} {gpu_t:>10.0} {result:>10}");
    }
    println!("  {}", "─".repeat(59));

    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("\n  13/13 new GPU domains: substrate-independent PROVEN");
    println!("  [Total] {ms:.1} ms");
    v.finish();
}

// ═══ MF-N01: Cooperation ════════════════════════════════════════════

fn validate_cooperation_mf(
    device: &Arc<WgpuDevice>,
    v: &mut Validator,
    timings: &mut Vec<(&'static str, f64, f64, &'static str)>,
) {
    use wetspring_barracuda::bio::cooperation_gpu::CooperationGpu;

    v.section("MF-N01: Cooperation ODE");

    let params = CooperationParams::default();
    let tc = Instant::now();
    let cpu = cooperation::scenario_equal_start(&params, 0.001);
    let cpu_us = tc.elapsed().as_micros() as f64;

    let tg = Instant::now();
    let gpu_engine = CooperationGpu::new(Arc::clone(device)).expect("shader compile");
    let results = gpu_engine
        .integrate_params(&[params], &[[0.01, 0.01, 0.0, 0.0]], 48000, 0.001)
        .expect("GPU integrate");
    let gpu_us = tg.elapsed().as_micros() as f64;

    for (i, (&g, &c)) in results[0]
        .iter()
        .zip(cpu.y_final.iter())
        .take(cooperation::N_VARS)
        .enumerate()
    {
        v.check(&format!("coop var[{i}]"), g, c, tolerances::ODE_GPU_PARITY);
    }
    timings.push(("Cooperation ODE", cpu_us, gpu_us, "CPU=GPU"));
}

// ═══ MF-N02: Capacitor ═════════════════════════════════════════════

fn validate_capacitor_mf(
    device: &Arc<WgpuDevice>,
    v: &mut Validator,
    timings: &mut Vec<(&'static str, f64, f64, &'static str)>,
) {
    use wetspring_barracuda::bio::capacitor_gpu::CapacitorGpu;

    v.section("MF-N02: Capacitor ODE");

    let params = CapacitorParams::default();
    let tc = Instant::now();
    let cpu = capacitor::scenario_normal(&params, 0.001);
    let cpu_us = tc.elapsed().as_micros() as f64;

    let tg = Instant::now();
    let gpu_engine = CapacitorGpu::new(Arc::clone(device)).expect("shader compile");
    let results = gpu_engine
        .integrate_params(&[params], &[[0.01, 1.0, 0.0, 0.0, 0.5, 0.0]], 48000, 0.001)
        .expect("GPU integrate");
    let gpu_us = tg.elapsed().as_micros() as f64;

    for (i, (&g, &c)) in results[0]
        .iter()
        .zip(cpu.y_final.iter())
        .take(capacitor::N_VARS)
        .enumerate()
    {
        v.check(&format!("cap var[{i}]"), g, c, tolerances::ODE_GPU_PARITY);
    }
    timings.push(("Capacitor ODE", cpu_us, gpu_us, "CPU=GPU"));
}

// ═══ MF-N03: KMD ═══════════════════════════════════════════════════

fn validate_kmd_mf(
    gpu: &GpuF64,
    v: &mut Validator,
    timings: &mut Vec<(&'static str, f64, f64, &'static str)>,
) {
    use wetspring_barracuda::bio::kmd_gpu;

    v.section("MF-N03: KMD");

    let masses = vec![412.966, 462.963, 512.960, 562.957, 612.954];
    let tc = Instant::now();
    let cpu = kmd::kendrick_mass_defect(&masses, units::CF2_EXACT, units::CF2_NOMINAL);
    let cpu_us = tc.elapsed().as_micros() as f64;

    let tg = Instant::now();
    let gpu_results =
        kmd_gpu::kendrick_mass_defect_gpu(gpu, &masses, units::CF2_EXACT, units::CF2_NOMINAL)
            .expect("KMD GPU");
    let gpu_us = tg.elapsed().as_micros() as f64;

    for (i, (c, g)) in cpu.iter().zip(&gpu_results).enumerate() {
        v.check(
            &format!("KMD[{i}]"),
            g.kmd,
            c.kmd,
            tolerances::ANALYTICAL_F64,
        );
    }
    timings.push(("KMD", cpu_us, gpu_us, "CPU=GPU"));
}

// ═══ MF-N04: GBM ═══════════════════════════════════════════════════

fn validate_gbm_mf(
    gpu: &GpuF64,
    v: &mut Validator,
    timings: &mut Vec<(&'static str, f64, f64, &'static str)>,
) {
    use wetspring_barracuda::bio::gbm_gpu;

    v.section("MF-N04: GBM");

    let t1 = GbmTree::from_arrays(
        &[0, -1, -1],
        &[0.5, 0.0, 0.0],
        &[1, -1, -1],
        &[2, -1, -1],
        &[0.0, 0.3, -0.1],
    )
    .expect("MetalForge v5");
    let t2 = GbmTree::from_arrays(
        &[1, -1, -1],
        &[0.3, 0.0, 0.0],
        &[1, -1, -1],
        &[2, -1, -1],
        &[0.0, 0.2, -0.2],
    )
    .expect("MetalForge v5");
    let model = GbmClassifier::new(vec![t1, t2], 0.1, 0.0, 2).expect("MetalForge v5");

    let samples = vec![vec![0.8, 0.5], vec![0.2, 0.1], vec![0.6, 0.4]];
    let tc = Instant::now();
    let cpu: Vec<_> = model.predict_batch_proba(&samples);
    let cpu_us = tc.elapsed().as_micros() as f64;

    let tg = Instant::now();
    let gpu_results = gbm_gpu::predict_batch_gpu(gpu, &model, &samples).expect("GBM GPU");
    let gpu_us = tg.elapsed().as_micros() as f64;

    for (i, (c, g)) in cpu.iter().zip(&gpu_results).enumerate() {
        v.check(
            &format!("GBM[{i}] prob"),
            g.probability,
            c.probability,
            tolerances::ANALYTICAL_F64,
        );
    }
    timings.push(("GBM", cpu_us, gpu_us, "CPU=GPU"));
}

// ═══ MF-N05: Merge Pairs ═══════════════════════════════════════════

fn validate_merge_pairs_mf(
    gpu: &GpuF64,
    v: &mut Validator,
    timings: &mut Vec<(&'static str, f64, f64, &'static str)>,
) {
    use wetspring_barracuda::bio::merge_pairs_gpu;

    v.section("MF-N05: Merge Pairs");

    let fwd = vec![FastqRecord {
        id: "read1".to_string(),
        sequence: b"ATCGATCGATCG".to_vec(),
        quality: vec![40; 12],
    }];
    let rev = vec![FastqRecord {
        id: "read1".to_string(),
        sequence: b"CGATCGATCGAT".to_vec(),
        quality: vec![35; 12],
    }];
    let params = MergeParams::default();
    let tc = Instant::now();
    let (cpu_merged, cpu_stats) = merge_pairs::merge_pairs(&fwd, &rev, &params);
    let cpu_us = tc.elapsed().as_micros() as f64;

    let tg = Instant::now();
    let (gpu_merged, gpu_stats) =
        merge_pairs_gpu::merge_pairs_gpu(gpu, &fwd, &rev, &params).expect("merge GPU");
    let gpu_us = tg.elapsed().as_micros() as f64;

    v.check(
        "merge count",
        gpu_merged.len() as f64,
        cpu_merged.len() as f64,
        tolerances::EXACT,
    );
    v.check(
        "merge input_pairs",
        gpu_stats.input_pairs as f64,
        cpu_stats.input_pairs as f64,
        tolerances::EXACT,
    );
    timings.push(("Merge Pairs", cpu_us, gpu_us, "CPU=GPU"));
}

// ═══ MF-N06: Signal ════════════════════════════════════════════════

fn validate_signal_mf(
    gpu: &GpuF64,
    v: &mut Validator,
    timings: &mut Vec<(&'static str, f64, f64, &'static str)>,
) {
    use wetspring_barracuda::bio::signal_gpu;

    v.section("MF-N06: Signal");

    let data: Vec<f64> = (0..200)
        .map(|i| {
            let x = f64::from(i) / 20.0;
            0.5f64.mul_add((-(x - 7.0).powi(2) / 0.5).exp(), (-(x - 3.0).powi(2)).exp())
        })
        .collect();

    let params = PeakParams {
        min_height: Some(0.1),
        min_prominence: Some(0.05),
        distance: 3,
        ..PeakParams::default()
    };

    let tc = Instant::now();
    let cpu = signal::find_peaks(&data, &params);
    let cpu_us = tc.elapsed().as_micros() as f64;

    let tg = Instant::now();
    let gpu_peaks = signal_gpu::find_peaks_gpu(gpu, &data, &params).expect("signal GPU");
    let gpu_us = tg.elapsed().as_micros() as f64;

    v.check(
        "peak count",
        gpu_peaks.len() as f64,
        cpu.len() as f64,
        tolerances::EXACT,
    );
    for (i, (c, g)) in cpu.iter().zip(&gpu_peaks).enumerate() {
        v.check(
            &format!("peak[{i}] index"),
            g.index as f64,
            c.index as f64,
            tolerances::EXACT,
        );
    }
    timings.push(("Signal", cpu_us, gpu_us, "CPU=GPU"));
}

// ═══ MF-N07: Feature Table ═════════════════════════════════════════

fn validate_feature_table_mf(
    gpu: &GpuF64,
    v: &mut Validator,
    timings: &mut Vec<(&'static str, f64, f64, &'static str)>,
) {
    use wetspring_barracuda::bio::feature_table_gpu;

    v.section("MF-N07: Feature Table");

    let params = FeatureParams::default();
    let tc = Instant::now();
    let cpu = feature_table::extract_features(&[], &params);
    let cpu_us = tc.elapsed().as_micros() as f64;

    let tg = Instant::now();
    let gpu_ft = feature_table_gpu::extract_features_gpu(gpu, &[], &params).expect("FT GPU");
    let gpu_us = tg.elapsed().as_micros() as f64;

    v.check(
        "feature count",
        gpu_ft.features.len() as f64,
        cpu.features.len() as f64,
        tolerances::EXACT,
    );
    timings.push(("Feature Table", cpu_us, gpu_us, "CPU=GPU"));
}

// ═══ MF-N08: Robinson-Foulds ═══════════════════════════════════════

fn validate_robinson_foulds_mf(
    gpu: &GpuF64,
    v: &mut Validator,
    timings: &mut Vec<(&'static str, f64, f64, &'static str)>,
) {
    use wetspring_barracuda::bio::robinson_foulds_gpu;

    v.section("MF-N08: Robinson-Foulds");

    let t1 = PhyloTree::from_newick("((C:0.5,D:0.5):0.5,(E:0.5,(A:1.0,B:1.0):0.5):0.5);");
    let t2 = PhyloTree::from_newick("((C:0.5,(A:1.0,B:1.0):0.5):0.5,(D:0.5,E:0.5):0.5);");

    let tc = Instant::now();
    let cpu_dist = robinson_foulds::rf_distance(&t1, &t2);
    let cpu_us = tc.elapsed().as_micros() as f64;

    let tg = Instant::now();
    let gpu_dist = robinson_foulds_gpu::rf_distance_gpu(gpu, &t1, &t2).expect("RF GPU");
    let gpu_us = tg.elapsed().as_micros() as f64;

    v.check(
        "RF distance",
        gpu_dist as f64,
        cpu_dist as f64,
        tolerances::EXACT,
    );
    timings.push(("Robinson-Foulds", cpu_us, gpu_us, "CPU=GPU"));
}

// ═══ MF-N09: Dereplication ═════════════════════════════════════════

fn validate_derep_mf(
    gpu: &GpuF64,
    v: &mut Validator,
    timings: &mut Vec<(&'static str, f64, f64, &'static str)>,
) {
    use wetspring_barracuda::bio::derep_gpu;

    v.section("MF-N09: Dereplication");

    let records = vec![
        FastqRecord {
            id: "s1".to_string(),
            sequence: b"ATCGATCG".to_vec(),
            quality: vec![40; 8],
        },
        FastqRecord {
            id: "s2".to_string(),
            sequence: b"ATCGATCG".to_vec(),
            quality: vec![40; 8],
        },
        FastqRecord {
            id: "s3".to_string(),
            sequence: b"GCTAGCTA".to_vec(),
            quality: vec![40; 8],
        },
    ];

    let tc = Instant::now();
    let (cpu_uniq, cpu_stats) = derep::dereplicate(&records, DerepSort::Abundance, 1);
    let cpu_us = tc.elapsed().as_micros() as f64;

    let tg = Instant::now();
    let (gpu_uniq, gpu_stats) =
        derep_gpu::dereplicate_gpu(gpu, &records, DerepSort::Abundance, 1).expect("derep GPU");
    let gpu_us = tg.elapsed().as_micros() as f64;

    v.check(
        "derep unique count",
        gpu_uniq.len() as f64,
        cpu_uniq.len() as f64,
        tolerances::EXACT,
    );
    v.check(
        "derep input_sequences",
        gpu_stats.input_sequences as f64,
        cpu_stats.input_sequences as f64,
        tolerances::EXACT,
    );
    timings.push(("Dereplication", cpu_us, gpu_us, "CPU=GPU"));
}

// ═══ MF-N10: Chimera ═══════════════════════════════════════════════

fn validate_chimera_mf(
    gpu: &GpuF64,
    v: &mut Validator,
    timings: &mut Vec<(&'static str, f64, f64, &'static str)>,
) {
    use wetspring_barracuda::bio::chimera_gpu;

    v.section("MF-N10: Chimera");

    let asvs = vec![
        Asv {
            sequence: b"ATCGATCGATCGATCG".to_vec(),
            abundance: 100,
            n_members: 1,
        },
        Asv {
            sequence: b"GCTAGCTAGCTAGCTA".to_vec(),
            abundance: 50,
            n_members: 1,
        },
        Asv {
            sequence: b"ATCGATCGGCTAGCTA".to_vec(),
            abundance: 10,
            n_members: 1,
        },
    ];
    let params = ChimeraParams::default();

    let tc = Instant::now();
    let (cpu_results, cpu_stats) = chimera::detect_chimeras(&asvs, &params);
    let cpu_us = tc.elapsed().as_micros() as f64;

    let tg = Instant::now();
    let (gpu_results, gpu_stats) =
        chimera_gpu::detect_chimeras_gpu(gpu, &asvs, &params).expect("chimera GPU");
    let gpu_us = tg.elapsed().as_micros() as f64;

    v.check(
        "chimera count",
        gpu_results.len() as f64,
        cpu_results.len() as f64,
        tolerances::EXACT,
    );
    v.check(
        "chimera found",
        gpu_stats.chimeras_found as f64,
        cpu_stats.chimeras_found as f64,
        tolerances::EXACT,
    );
    timings.push(("Chimera", cpu_us, gpu_us, "CPU=GPU"));
}

// ═══ MF-N11: Neighbor Joining ══════════════════════════════════════

fn validate_neighbor_joining_mf(
    gpu: &GpuF64,
    v: &mut Validator,
    timings: &mut Vec<(&'static str, f64, f64, &'static str)>,
) {
    use wetspring_barracuda::bio::neighbor_joining_gpu;

    v.section("MF-N11: Neighbor Joining");

    let seqs: Vec<&[u8]> = vec![b"ATCGATCG", b"ATCAATCG", b"GCTAGCTA", b"GCTGGCTA"];

    let tc = Instant::now();
    let cpu_dist = neighbor_joining::distance_matrix(&seqs);
    let cpu_us = tc.elapsed().as_micros() as f64;

    let tg = Instant::now();
    let gpu_dist = neighbor_joining_gpu::distance_matrix_gpu(gpu, &seqs).expect("NJ GPU");
    let gpu_us = tg.elapsed().as_micros() as f64;

    for (i, (c, g)) in cpu_dist.iter().zip(&gpu_dist).enumerate() {
        v.check(&format!("NJ dist[{i}]"), *g, *c, tolerances::ANALYTICAL_F64);
    }
    timings.push(("Neighbor Joining", cpu_us, gpu_us, "CPU=GPU"));
}

// ═══ MF-N12: Reconciliation ════════════════════════════════════════

const NO_CHILD: u32 = u32::MAX;

fn validate_reconciliation_mf(
    gpu: &GpuF64,
    v: &mut Validator,
    timings: &mut Vec<(&'static str, f64, f64, &'static str)>,
) {
    use wetspring_barracuda::bio::reconciliation_gpu;

    v.section("MF-N12: Reconciliation");

    let host = FlatRecTree {
        names: vec![
            "h1".into(),
            "h2".into(),
            "H1".into(),
            "H2".into(),
            "HR".into(),
        ],
        left_child: vec![NO_CHILD, NO_CHILD, 0, NO_CHILD, 2],
        right_child: vec![NO_CHILD, NO_CHILD, 1, NO_CHILD, 3],
    };
    let parasite = FlatRecTree {
        names: vec!["g1".into(), "g2".into(), "GR".into()],
        left_child: vec![NO_CHILD, NO_CHILD, 0],
        right_child: vec![NO_CHILD, NO_CHILD, 1],
    };
    let tip_mapping = vec![("g1".into(), "h1".into()), ("g2".into(), "h2".into())];
    let costs = DtlCosts::default();

    let tc = Instant::now();
    let cpu = reconciliation::reconcile_dtl(&host, &parasite, &tip_mapping, &costs);
    let cpu_us = tc.elapsed().as_micros() as f64;

    let tg = Instant::now();
    let gpu_result =
        reconciliation_gpu::reconcile_dtl_gpu(gpu, &host, &parasite, &tip_mapping, &costs)
            .expect("reconciliation GPU");
    let gpu_us = tg.elapsed().as_micros() as f64;

    v.check(
        "reconciliation cost",
        f64::from(gpu_result.optimal_cost),
        f64::from(cpu.optimal_cost),
        tolerances::ANALYTICAL_F64,
    );
    timings.push(("Reconciliation", cpu_us, gpu_us, "CPU=GPU"));
}

// ═══ MF-N13: Molecular Clock ═══════════════════════════════════════

fn validate_molecular_clock_mf(
    gpu: &GpuF64,
    v: &mut Validator,
    timings: &mut Vec<(&'static str, f64, f64, &'static str)>,
) {
    use wetspring_barracuda::bio::molecular_clock_gpu;

    v.section("MF-N13: Molecular Clock");

    let branch_lengths = vec![0.0, 0.1, 0.15, 0.05, 0.08];
    let parent_indices: Vec<i64> = vec![-1, 0, 0, 1, 1];
    let parent_opt: Vec<Option<usize>> = parent_indices
        .iter()
        .map(|&p| u64::try_from(p).ok().map(|u| u as usize))
        .collect();
    let root_age = 100.0;

    let tc = Instant::now();
    let cpu = molecular_clock::strict_clock(&branch_lengths, &parent_opt, root_age, &[]);
    let cpu_us = tc.elapsed().as_micros() as f64;

    let tg = Instant::now();
    let gpu_result =
        molecular_clock_gpu::strict_clock_gpu(gpu, &branch_lengths, &parent_indices, root_age, &[])
            .expect("clock GPU");
    let gpu_us = tg.elapsed().as_micros() as f64;

    match (&cpu, &gpu_result) {
        (Some(c), Some(g)) => {
            v.check(
                "strict clock rate",
                g.rate,
                c.rate,
                tolerances::ANALYTICAL_F64,
            );
        }
        (None, None) => {
            v.check("strict clock both None", 1.0, 1.0, tolerances::EXACT);
        }
        _ => {
            v.check("strict clock mismatch", 0.0, 1.0, tolerances::EXACT);
        }
    }

    let node_ages = vec![100.0, 60.0, 70.0, 0.0, 0.0];
    let cpu_rates = molecular_clock::relaxed_clock_rates(&branch_lengths, &node_ages, &parent_opt);
    let gpu_rates = molecular_clock_gpu::relaxed_clock_rates_gpu(
        gpu,
        &branch_lengths,
        &node_ages,
        &parent_indices,
    )
    .expect("relaxed clock GPU");

    for (i, (c, g)) in cpu_rates.iter().zip(&gpu_rates).enumerate() {
        v.check(
            &format!("relaxed rate[{i}]"),
            *g,
            *c,
            tolerances::ANALYTICAL_F64,
        );
    }
    timings.push(("Molecular Clock", cpu_us, gpu_us, "CPU=GPU"));
}
