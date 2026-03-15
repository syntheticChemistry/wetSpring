// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! Validate all 13 pure GPU promotion modules against CPU baselines.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | 1f9f80e |
//! | Baseline tool | `BarraCuda` CPU (sovereign Rust reference) |
//! | Baseline date | 2026-02-22 |
//! | Exact command | `cargo run --release --features gpu --bin validate_pure_gpu_complete` |
//! | Data | Synthetic test vectors (self-contained) |
//! | Hardware | i9-12900K, 64 GB DDR5, RTX 4070, Pop!\_OS 22.04 |
//!
//! Validation class: GPU-parity
//! Provenance: CPU reference implementation in `barracuda::bio`

use std::sync::Arc;

use barracuda::device::WgpuDevice;
use wetspring_barracuda::bio::{
    capacitor::{self, CapacitorParams},
    cooperation::{self, CooperationParams},
    derep::{self, DerepSort},
    feature_table::{self, FeatureParams},
    gbm::{GbmClassifier, GbmTree},
    kmd::{self, units},
    merge_pairs::{self, MergeParams},
    molecular_clock, neighbor_joining,
    reconciliation::{self, DtlCosts, FlatRecTree},
    robinson_foulds,
    signal::{self, PeakParams},
};
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::{self, Validator};

#[tokio::main]
async fn main() {
    let mut v = Validator::new("Pure GPU Promotion: All 13 Modules vs CPU Baselines");

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

    validate_cooperation_gpu(&device, &mut v);
    validate_capacitor_gpu(&device, &mut v);
    validate_kmd_gpu(&gpu, &mut v);
    validate_gbm_gpu(&gpu, &mut v);
    validate_merge_pairs_gpu(&gpu, &mut v);
    validate_signal_gpu(&gpu, &mut v);
    validate_feature_table_gpu(&gpu, &mut v);
    validate_robinson_foulds_gpu(&gpu, &mut v);
    validate_derep_gpu(&gpu, &mut v);
    validate_chimera_gpu(&gpu, &mut v);
    validate_neighbor_joining_gpu(&gpu, &mut v);
    validate_reconciliation_gpu(&gpu, &mut v);
    validate_molecular_clock_gpu(&gpu, &mut v);

    v.finish();
}

fn validate_cooperation_gpu(device: &Arc<WgpuDevice>, v: &mut Validator) {
    use wetspring_barracuda::bio::cooperation_gpu::CooperationGpu;

    v.section("M01: Cooperation ODE GPU");

    let params = CooperationParams::default();
    let cpu = cooperation::scenario_equal_start(&params, tolerances::ODE_DEFAULT_DT);

    let gpu_engine = CooperationGpu::new(Arc::clone(device)).expect("shader compile");
    let results = gpu_engine
        .integrate_params(
            &[params],
            &[[0.01, 0.01, 0.0, 0.0]],
            48000,
            tolerances::ODE_DEFAULT_DT,
        )
        .expect("GPU integrate");

    for (i, (&g, &c)) in results[0]
        .iter()
        .zip(cpu.y_final.iter())
        .take(cooperation::N_VARS)
        .enumerate()
    {
        v.check(
            &format!("coop var[{i}] CPU≈GPU"),
            g,
            c,
            tolerances::ODE_GPU_PARITY,
        );
    }
}

fn validate_capacitor_gpu(device: &Arc<WgpuDevice>, v: &mut Validator) {
    use wetspring_barracuda::bio::capacitor_gpu::CapacitorGpu;

    v.section("M02: Capacitor ODE GPU");

    let params = CapacitorParams::default();
    let cpu = capacitor::scenario_normal(&params, tolerances::ODE_DEFAULT_DT);

    let gpu_engine = CapacitorGpu::new(Arc::clone(device)).expect("shader compile");
    let results = gpu_engine
        .integrate_params(
            &[params],
            &[[0.01, 1.0, 0.0, 0.0, 0.5, 0.0]],
            48000,
            tolerances::ODE_DEFAULT_DT,
        )
        .expect("GPU integrate");

    for (i, (&g, &c)) in results[0]
        .iter()
        .zip(cpu.y_final.iter())
        .take(capacitor::N_VARS)
        .enumerate()
    {
        v.check(
            &format!("cap var[{i}] CPU≈GPU"),
            g,
            c,
            tolerances::ODE_GPU_PARITY,
        );
    }
}

fn validate_kmd_gpu(gpu: &GpuF64, v: &mut Validator) {
    use wetspring_barracuda::bio::kmd_gpu;

    v.section("M03: KMD GPU");

    let masses = vec![412.966, 462.963, 512.960, 562.957, 612.954];
    let cpu = kmd::kendrick_mass_defect(&masses, units::CF2_EXACT, units::CF2_NOMINAL);
    let gpu_results =
        kmd_gpu::kendrick_mass_defect_gpu(gpu, &masses, units::CF2_EXACT, units::CF2_NOMINAL)
            .expect("KMD GPU");

    for (i, (c, g)) in cpu.iter().zip(&gpu_results).enumerate() {
        v.check(
            &format!("KMD[{i}] CPU≈GPU"),
            g.kmd,
            c.kmd,
            tolerances::ANALYTICAL_F64,
        );
    }
}

fn validate_gbm_gpu(gpu: &GpuF64, v: &mut Validator) {
    use wetspring_barracuda::bio::gbm_gpu;

    v.section("M04: GBM GPU");

    let t1 = GbmTree::from_arrays(
        &[0, -1, -1],
        &[0.5, 0.0, 0.0],
        &[1, -1, -1],
        &[2, -1, -1],
        &[0.0, 0.3, -0.1],
    )
    .expect("pure GPU complete");
    let t2 = GbmTree::from_arrays(
        &[1, -1, -1],
        &[0.3, 0.0, 0.0],
        &[1, -1, -1],
        &[2, -1, -1],
        &[0.0, 0.2, -0.2],
    )
    .expect("pure GPU complete");
    let model = GbmClassifier::new(vec![t1, t2], 0.1, 0.0, 2).expect("pure GPU complete");

    let samples = vec![vec![0.8, 0.5], vec![0.2, 0.1], vec![0.6, 0.4]];
    let cpu: Vec<_> = model.predict_batch_proba(&samples);
    let gpu_results = gbm_gpu::predict_batch_gpu(gpu, &model, &samples).expect("GBM GPU");

    for (i, (c, g)) in cpu.iter().zip(&gpu_results).enumerate() {
        v.check(
            &format!("GBM[{i}] prob CPU≈GPU"),
            g.probability,
            c.probability,
            tolerances::ANALYTICAL_F64,
        );
    }
}

fn validate_merge_pairs_gpu(gpu: &GpuF64, v: &mut Validator) {
    use wetspring_barracuda::bio::merge_pairs_gpu;
    use wetspring_barracuda::io::fastq::FastqRecord;

    v.section("M05: Merge Pairs GPU");

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
    let (cpu_merged, cpu_stats) = merge_pairs::merge_pairs(&fwd, &rev, &params);
    let (gpu_merged, gpu_stats) =
        merge_pairs_gpu::merge_pairs_gpu(gpu, &fwd, &rev, &params).expect("merge GPU");

    v.check(
        "merge count CPU≈GPU",
        gpu_merged.len() as f64,
        cpu_merged.len() as f64,
        tolerances::EXACT,
    );
    v.check(
        "merge stats input_pairs CPU≈GPU",
        gpu_stats.input_pairs as f64,
        cpu_stats.input_pairs as f64,
        tolerances::EXACT,
    );
}

fn validate_signal_gpu(gpu: &GpuF64, v: &mut Validator) {
    use wetspring_barracuda::bio::signal_gpu;

    v.section("M06: Signal GPU");

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

    let cpu = signal::find_peaks(&data, &params);
    let gpu_peaks = signal_gpu::find_peaks_gpu(gpu, &data, &params).expect("signal GPU");

    v.check(
        "peak count CPU≈GPU",
        gpu_peaks.len() as f64,
        cpu.len() as f64,
        tolerances::EXACT,
    );
    for (i, (c, g)) in cpu.iter().zip(&gpu_peaks).enumerate() {
        v.check(
            &format!("peak[{i}] index CPU≈GPU"),
            g.index as f64,
            c.index as f64,
            tolerances::EXACT,
        );
    }
}

fn validate_feature_table_gpu(gpu: &GpuF64, v: &mut Validator) {
    use wetspring_barracuda::bio::feature_table_gpu;

    v.section("M07: Feature Table GPU");

    let params = FeatureParams::default();
    let cpu = feature_table::extract_features(&[], &params);
    let gpu_ft = feature_table_gpu::extract_features_gpu(gpu, &[], &params).expect("FT GPU");

    v.check(
        "feature count CPU≈GPU (empty)",
        gpu_ft.features.len() as f64,
        cpu.features.len() as f64,
        tolerances::EXACT,
    );
}

fn validate_robinson_foulds_gpu(gpu: &GpuF64, v: &mut Validator) {
    use wetspring_barracuda::bio::robinson_foulds_gpu;
    use wetspring_barracuda::bio::unifrac::PhyloTree;

    v.section("M08: Robinson-Foulds GPU");

    let t1 = PhyloTree::from_newick("((C:0.5,D:0.5):0.5,(E:0.5,(A:1.0,B:1.0):0.5):0.5);");
    let t2 = PhyloTree::from_newick("((C:0.5,(A:1.0,B:1.0):0.5):0.5,(D:0.5,E:0.5):0.5);");

    let cpu_dist = robinson_foulds::rf_distance(&t1, &t2);
    let gpu_dist = robinson_foulds_gpu::rf_distance_gpu(gpu, &t1, &t2).expect("RF GPU");

    v.check(
        "RF distance CPU≈GPU",
        gpu_dist as f64,
        cpu_dist as f64,
        tolerances::EXACT,
    );
}

fn validate_derep_gpu(gpu: &GpuF64, v: &mut Validator) {
    use wetspring_barracuda::bio::derep_gpu;
    use wetspring_barracuda::io::fastq::FastqRecord;

    v.section("M09: Dereplication GPU");

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

    let (cpu_uniq, cpu_stats) = derep::dereplicate(&records, DerepSort::Abundance, 1);
    let (gpu_uniq, gpu_stats) =
        derep_gpu::dereplicate_gpu(gpu, &records, DerepSort::Abundance, 1).expect("derep GPU");

    v.check(
        "derep unique count CPU≈GPU",
        gpu_uniq.len() as f64,
        cpu_uniq.len() as f64,
        tolerances::EXACT,
    );
    v.check(
        "derep input_sequences CPU≈GPU",
        gpu_stats.input_sequences as f64,
        cpu_stats.input_sequences as f64,
        tolerances::EXACT,
    );
}

fn validate_chimera_gpu(gpu: &GpuF64, v: &mut Validator) {
    use wetspring_barracuda::bio::chimera::{self, ChimeraParams};
    use wetspring_barracuda::bio::chimera_gpu;
    use wetspring_barracuda::bio::dada2::Asv;

    v.section("M10: Chimera GPU (upgraded)");

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
    let (cpu_results, cpu_stats) = chimera::detect_chimeras(&asvs, &params);
    let (gpu_results, gpu_stats) =
        chimera_gpu::detect_chimeras_gpu(gpu, &asvs, &params).expect("chimera GPU");

    v.check(
        "chimera count CPU≈GPU",
        gpu_results.len() as f64,
        cpu_results.len() as f64,
        tolerances::EXACT,
    );
    v.check(
        "chimera found CPU≈GPU",
        gpu_stats.chimeras_found as f64,
        cpu_stats.chimeras_found as f64,
        tolerances::EXACT,
    );
}

fn validate_neighbor_joining_gpu(gpu: &GpuF64, v: &mut Validator) {
    use wetspring_barracuda::bio::neighbor_joining_gpu;

    v.section("M11: Neighbor Joining GPU");

    let seqs: Vec<&[u8]> = vec![b"ATCGATCG", b"ATCAATCG", b"GCTAGCTA", b"GCTGGCTA"];
    let cpu_dist = neighbor_joining::distance_matrix(&seqs);
    let gpu_dist = neighbor_joining_gpu::distance_matrix_gpu(gpu, &seqs).expect("NJ distance GPU");

    for (i, (c, g)) in cpu_dist.iter().zip(&gpu_dist).enumerate() {
        v.check(
            &format!("NJ dist[{i}] CPU≈GPU"),
            *g,
            *c,
            tolerances::ANALYTICAL_F64,
        );
    }
}

const NO_CHILD: u32 = u32::MAX;

fn validate_reconciliation_gpu(gpu: &GpuF64, v: &mut Validator) {
    use wetspring_barracuda::bio::reconciliation_gpu;

    v.section("M12: Reconciliation GPU");

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

    let cpu = reconciliation::reconcile_dtl(&host, &parasite, &tip_mapping, &costs);
    let gpu_result =
        reconciliation_gpu::reconcile_dtl_gpu(gpu, &host, &parasite, &tip_mapping, &costs)
            .expect("reconciliation GPU");

    v.check(
        "reconciliation cost CPU≈GPU",
        f64::from(gpu_result.optimal_cost),
        f64::from(cpu.optimal_cost),
        tolerances::ANALYTICAL_F64,
    );
}

fn validate_molecular_clock_gpu(gpu: &GpuF64, v: &mut Validator) {
    use wetspring_barracuda::bio::molecular_clock_gpu;

    v.section("M13: Molecular Clock GPU");

    let branch_lengths = vec![0.0, 0.1, 0.15, 0.05, 0.08];
    let parent_indices: Vec<i64> = vec![-1, 0, 0, 1, 1];
    let parent_opt: Vec<Option<usize>> = parent_indices
        .iter()
        .map(|&p| if p < 0 { None } else { Some(p as usize) })
        .collect();
    let root_age = 100.0;

    let cpu = molecular_clock::strict_clock(&branch_lengths, &parent_opt, root_age, &[]);
    let gpu_result =
        molecular_clock_gpu::strict_clock_gpu(gpu, &branch_lengths, &parent_indices, root_age, &[])
            .expect("clock GPU");

    match (&cpu, &gpu_result) {
        (Some(c), Some(g)) => {
            v.check(
                "strict clock rate CPU≈GPU",
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
            &format!("relaxed rate[{i}] CPU≈GPU"),
            *g,
            *c,
            tolerances::ANALYTICAL_F64,
        );
    }
}
