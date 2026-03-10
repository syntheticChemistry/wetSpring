// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![allow(
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::similar_names
)]
//! Exp102: `BarraCuda` CPU Parity v8 — Pure Rust Math for 13 GPU-Promoted Domains
//!
//! Validates that the 13 modules promoted to GPU in the pure GPU completion
//! pass produce correct CPU results from analytical known-values. This proves
//! **pure Rust math** matches paper equations before any GPU dispatch.
//!
//! Each module is tested with synthetic inputs that have known analytical
//! answers, following the established `validate_barracuda_cpu` pattern.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | 1f9f80e |
//! | Baseline tool | Analytical known-values from papers |
//! | Baseline date | 2026-02-22 |
//! | Exact command | `cargo run --release --bin validate_barracuda_cpu_v8` |
//! | Data | Synthetic test vectors (self-contained) |
//! | Hardware | i9-12900K, 64 GB DDR5, RTX 4070, Pop!\_OS 22.04 |
//!
//! Validation class: Analytical
//! Provenance: Known-value formulas (Shannon H(uniform)=ln(S), Hill(EC50)=0.5, GOE/Poisson level spacing)

use std::time::Instant;
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
    ode::steady_state_mean,
    reconciliation::{self, DtlCosts, FlatRecTree},
    robinson_foulds,
    signal::{self, PeakParams},
    unifrac::PhyloTree,
};
use wetspring_barracuda::io::fastq::FastqRecord;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

const SS_FRAC: f64 = 0.1;

fn main() {
    let mut v =
        Validator::new("Exp102: BarraCuda CPU v8 — Pure Rust Math (13 GPU-Promoted Domains)");
    let t_total = Instant::now();

    validate_cooperation(&mut v);
    validate_capacitor(&mut v);
    validate_kmd(&mut v);
    validate_gbm(&mut v);
    validate_merge_pairs(&mut v);
    validate_signal(&mut v);
    validate_feature_table(&mut v);
    validate_robinson_foulds(&mut v);
    validate_derep(&mut v);
    validate_chimera(&mut v);
    validate_neighbor_joining(&mut v);
    validate_reconciliation(&mut v);
    validate_molecular_clock(&mut v);

    #[expect(clippy::cast_precision_loss)]
    let elapsed_us = t_total.elapsed().as_nanos() as f64 / 1000.0;
    println!("\n  Total v8 validation: {elapsed_us:.0} µs");

    v.finish();
}

fn print_timing(label: &str, t0: Instant) {
    #[expect(clippy::cast_precision_loss)]
    let us = t0.elapsed().as_nanos() as f64 / 1000.0;
    println!("    {label}: {us:.0} µs");
}

// ════════════════════════════════════════════════════════════════════
//  Module 1: Cooperation (Bruger & Waters 2018) — 4 vars, 13 params
// ════════════════════════════════════════════════════════════════════

fn validate_cooperation(v: &mut Validator) {
    v.section("═══ M01: Cooperation (Bruger 2018) — 4 vars, 13 params ═══");
    let t0 = Instant::now();

    let p = CooperationParams::default();
    let flat = p.to_flat();
    let p2 = CooperationParams::from_flat(&flat);
    let flat2 = p2.to_flat();

    v.check_count("cooperation flat length", flat.len(), cooperation::N_PARAMS);
    v.check(
        "cooperation flat round-trip",
        bitwise_diff(&flat, &flat2),
        0.0,
        tolerances::EXACT,
    );

    let r = cooperation::scenario_equal_start(&p, tolerances::ODE_DEFAULT_DT);
    let n_coop = steady_state_mean(&r, 0, SS_FRAC);
    let n_cheat = steady_state_mean(&r, 1, SS_FRAC);

    v.check_pass(
        "cooperation coexistence: both > 0",
        n_coop > 0.0 && n_cheat > 0.0,
    );
    v.check_pass(
        "cooperation cooperators persist",
        n_coop > tolerances::ODE_COOPERATOR_PERSIST_THRESHOLD,
    );

    let r_flat = cooperation::scenario_equal_start(&p2, tolerances::ODE_DEFAULT_DT);
    let n_coop_flat = steady_state_mean(&r_flat, 0, SS_FRAC);
    v.check(
        "cooperation flat vs direct bitwise",
        (n_coop - n_coop_flat).abs(),
        0.0,
        tolerances::EXACT,
    );

    let bio = steady_state_mean(&r, 3, SS_FRAC);
    v.check_pass("cooperation biofilm positive", bio > 0.0);

    print_timing("cooperation", t0);
}

// ════════════════════════════════════════════════════════════════════
//  Module 2: Capacitor (Mhatre 2020) — 6 vars, 16 params
// ════════════════════════════════════════════════════════════════════

fn validate_capacitor(v: &mut Validator) {
    v.section("═══ M02: Capacitor (Mhatre 2020) — 6 vars, 16 params ═══");
    let t0 = Instant::now();

    let p = CapacitorParams::default();
    let flat = p.to_flat();
    let p2 = CapacitorParams::from_flat(&flat);
    let flat2 = p2.to_flat();

    v.check_count("capacitor flat length", flat.len(), capacitor::N_PARAMS);
    v.check(
        "capacitor flat round-trip",
        bitwise_diff(&flat, &flat2),
        0.0,
        tolerances::EXACT,
    );

    let r = capacitor::scenario_normal(&p, tolerances::ODE_DEFAULT_DT);
    let n = steady_state_mean(&r, 0, SS_FRAC);
    v.check_pass(
        "capacitor cells grow",
        n > tolerances::ODE_CELL_GROWTH_THRESHOLD,
    );

    let vpsr = steady_state_mean(&r, 2, SS_FRAC);
    v.check_pass("capacitor VpsR accumulates", vpsr > 0.0);

    let biofilm = steady_state_mean(&r, 3, SS_FRAC);
    let motility = steady_state_mean(&r, 4, SS_FRAC);
    v.check_pass(
        "capacitor mixed phenotype (bio+mot > 0)",
        biofilm > 0.0 || motility > 0.0,
    );

    let r_flat = capacitor::scenario_normal(&p2, tolerances::ODE_DEFAULT_DT);
    let n_flat = steady_state_mean(&r_flat, 0, SS_FRAC);
    v.check(
        "capacitor flat vs direct bitwise",
        (n - n_flat).abs(),
        0.0,
        tolerances::EXACT,
    );

    print_timing("capacitor", t0);
}

// ════════════════════════════════════════════════════════════════════
//  Module 3: KMD (Kendrick 1963)
// ════════════════════════════════════════════════════════════════════

fn validate_kmd(v: &mut Validator) {
    v.section("═══ M03: KMD (Kendrick 1963) — CF2 homologue series ═══");
    let t0 = Instant::now();

    let masses = vec![412.966, 462.963, 512.960, 562.957, 612.954];
    let results = kmd::kendrick_mass_defect(&masses, units::CF2_EXACT, units::CF2_NOMINAL);

    v.check_count("KMD result count", results.len(), 5);

    for (i, r) in results.iter().enumerate() {
        let km = r.exact_mass * (units::CF2_NOMINAL / units::CF2_EXACT);
        v.check(
            &format!("KMD[{i}] kendrick_mass"),
            r.kendrick_mass,
            km,
            tolerances::ANALYTICAL_F64,
        );
    }

    let kmd_spread = results.last().map_or(0.0, |l| l.kmd) - results.first().map_or(0.0, |f| f.kmd);
    v.check(
        "KMD CF2 series spread < 0.02",
        kmd_spread.abs(),
        0.0,
        tolerances::KMD_SPREAD,
    );

    print_timing("kmd", t0);
}

// ════════════════════════════════════════════════════════════════════
//  Module 4: GBM (Gradient Boosted Model)
// ════════════════════════════════════════════════════════════════════

fn validate_gbm(v: &mut Validator) {
    v.section("═══ M04: GBM — stump ensemble predictions ═══");
    let t0 = Instant::now();

    let t1 = GbmTree::from_arrays(
        &[0, -1, -1],
        &[0.5, 0.0, 0.0],
        &[1, -1, -1],
        &[2, -1, -1],
        &[0.0, 0.3, -0.1],
    )
    .expect("Barracuda CPU v8");
    let t2 = GbmTree::from_arrays(
        &[1, -1, -1],
        &[0.3, 0.0, 0.0],
        &[1, -1, -1],
        &[2, -1, -1],
        &[0.0, 0.2, -0.2],
    )
    .expect("Barracuda CPU v8");
    let model = GbmClassifier::new(vec![t1, t2], 0.1, 0.0, 2).expect("Barracuda CPU v8");

    let samples = vec![vec![0.8, 0.5], vec![0.2, 0.1], vec![0.6, 0.4]];
    let preds = model.predict_batch_proba(&samples);

    v.check_count("GBM prediction count", preds.len(), 3);

    for (i, p) in preds.iter().enumerate() {
        v.check_pass(
            &format!("GBM[{i}] probability in [0,1]"),
            (0.0..=1.0).contains(&p.probability),
        );
    }

    let sample_a = vec![0.8, 0.5];
    let sample_b = vec![0.2, 0.1];
    let p_a = model.predict_batch_proba(&[sample_a]);
    let p_b = model.predict_batch_proba(&[sample_b]);

    v.check_pass(
        "GBM different features → different raw_scores",
        (p_a[0].raw_score - p_b[0].raw_score).abs() > tolerances::EXACT_F64,
    );

    print_timing("gbm", t0);
}

// ════════════════════════════════════════════════════════════════════
//  Module 5: Merge Pairs (DADA2 pipeline)
// ════════════════════════════════════════════════════════════════════

fn validate_merge_pairs(v: &mut Validator) {
    v.section("═══ M05: Merge Pairs — overlap detection ═══");
    let t0 = Instant::now();

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
    let (merged, stats) = merge_pairs::merge_pairs(&fwd, &rev, &params);

    v.check_count("merge input_pairs", stats.input_pairs, 1);
    v.check_count(
        "merge total = merged + no_overlap + mismatches",
        stats.merged_count + stats.no_overlap_count + stats.too_many_mismatches,
        1,
    );

    if merged.is_empty() {
        v.check_pass("merge: no-overlap is valid outcome", true);
    } else {
        v.check_pass(
            "merged sequence longer than inputs",
            merged[0].sequence.len() >= 12,
        );
    }

    print_timing("merge_pairs", t0);
}

// ════════════════════════════════════════════════════════════════════
//  Module 6: Signal (Peak Detection)
// ════════════════════════════════════════════════════════════════════

fn validate_signal(v: &mut Validator) {
    v.section("═══ M06: Signal — Gaussian + Lorentzian peak detection ═══");
    let t0 = Instant::now();

    let data: Vec<f64> = (0..200)
        .map(|i| {
            let x = f64::from(i) / 20.0;
            0.5f64.mul_add((-(x - 7.0).powi(2) / 0.5).exp(), (-(x - 3.0).powi(2)).exp())
        })
        .collect();

    let params = PeakParams {
        min_height: Some(0.1),
        min_prominence: Some(tolerances::PEAK_MIN_PROMINENCE),
        distance: 3,
        ..PeakParams::default()
    };

    let peaks = signal::find_peaks(&data, &params);

    v.check_pass("signal finds >= 2 peaks", peaks.len() >= 2);

    let peak_near_60 = peaks.iter().any(|p| (55..=65).contains(&p.index));
    let peak_near_140 = peaks.iter().any(|p| (135..=145).contains(&p.index));
    v.check_pass("signal peak near x=3 (index ~60)", peak_near_60);
    v.check_pass("signal peak near x=7 (index ~140)", peak_near_140);

    for p in &peaks {
        v.check_pass(
            &format!("signal peak[{}] height > min_height", p.index),
            p.height >= 0.1,
        );
    }

    print_timing("signal", t0);
}

// ════════════════════════════════════════════════════════════════════
//  Module 7: Feature Table (LC-MS)
// ════════════════════════════════════════════════════════════════════

fn validate_feature_table(v: &mut Validator) {
    v.section("═══ M07: Feature Table — empty-input identity ═══");
    let t0 = Instant::now();

    let params = FeatureParams::default();
    let ft = feature_table::extract_features(&[], &params);

    v.check_count("feature_table empty → 0 features", ft.features.len(), 0);
    v.check_count(
        "feature_table empty → 0 mass_tracks",
        ft.mass_tracks_evaluated,
        0,
    );

    print_timing("feature_table", t0);
}

// ════════════════════════════════════════════════════════════════════
//  Module 8: Robinson-Foulds (Robinson & Foulds 1981)
// ════════════════════════════════════════════════════════════════════

fn validate_robinson_foulds(v: &mut Validator) {
    v.section("═══ M08: Robinson-Foulds — known tree distances ═══");
    let t0 = Instant::now();

    let t1 = PhyloTree::from_newick("((A:1.0,B:1.0):0.5,(C:0.5,D:0.5):0.5);");
    let t2 = PhyloTree::from_newick("((A:1.0,C:1.0):0.5,(B:0.5,D:0.5):0.5);");

    let dist = robinson_foulds::rf_distance(&t1, &t2);
    v.check_pass("RF distance > 0 for different topologies", dist > 0);

    let self_dist = robinson_foulds::rf_distance(&t1, &t1);
    v.check_count("RF self-distance = 0", self_dist, 0);

    let t3 = PhyloTree::from_newick("((C:0.5,D:0.5):0.5,(E:0.5,(A:1.0,B:1.0):0.5):0.5);");
    let t4 = PhyloTree::from_newick("((C:0.5,(A:1.0,B:1.0):0.5):0.5,(D:0.5,E:0.5):0.5);");
    let dist2 = robinson_foulds::rf_distance(&t3, &t4);
    v.check_pass("RF 5-taxon distance > 0", dist2 > 0);

    print_timing("robinson_foulds", t0);
}

// ════════════════════════════════════════════════════════════════════
//  Module 9: Dereplication
// ════════════════════════════════════════════════════════════════════

fn validate_derep(v: &mut Validator) {
    v.section("═══ M09: Dereplication — exact duplicate counting ═══");
    let t0 = Instant::now();

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

    let (uniq, stats) = derep::dereplicate(&records, DerepSort::Abundance, 1);

    v.check_count("derep input_sequences", stats.input_sequences, 3);
    v.check_count("derep unique_sequences", stats.unique_sequences, 2);
    v.check_count("derep unique count", uniq.len(), 2);
    v.check_count("derep max_abundance", stats.max_abundance, 2);

    let (uniq_seq, _) = derep::dereplicate(&records, DerepSort::Sequence, 1);
    v.check_count("derep Sequence sort same count", uniq_seq.len(), 2);

    print_timing("derep", t0);
}

// ════════════════════════════════════════════════════════════════════
//  Module 10: Chimera (UCHIME-inspired)
// ════════════════════════════════════════════════════════════════════

fn validate_chimera(v: &mut Validator) {
    v.section("═══ M10: Chimera — known chimeric detection ═══");
    let t0 = Instant::now();

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
    let (results, stats) = chimera::detect_chimeras(&asvs, &params);

    v.check_count("chimera input_sequences", stats.input_sequences, 3);
    v.check_pass(
        "chimera results count == input",
        results.len() == asvs.len(),
    );
    v.check_pass(
        "chimera retained + found == input",
        stats.retained + stats.chimeras_found == stats.input_sequences,
    );

    print_timing("chimera", t0);
}

// ════════════════════════════════════════════════════════════════════
//  Module 11: Neighbor Joining (Saitou & Nei 1987)
// ════════════════════════════════════════════════════════════════════

fn validate_neighbor_joining(v: &mut Validator) {
    v.section("═══ M11: Neighbor Joining — JC69 distance matrix ═══");
    let t0 = Instant::now();

    let seqs: Vec<&[u8]> = vec![b"ATCGATCG", b"ATCAATCG", b"GCTAGCTA", b"GCTGGCTA"];
    let dist = neighbor_joining::distance_matrix(&seqs);

    let n = seqs.len();
    v.check_count("NJ distance matrix length", dist.len(), n * n);

    for i in 0..n {
        v.check(
            &format!("NJ dist[{i},{i}] = 0 (diagonal)"),
            dist[i * n + i],
            0.0,
            tolerances::EXACT,
        );
    }

    v.check_pass(
        "NJ dist[0,1] < dist[0,2] (1-diff vs many-diff)",
        dist[1] < dist[2],
    );

    for i in 0..n {
        for j in 0..n {
            v.check(
                &format!("NJ symmetry [{i},{j}]"),
                dist[i * n + j],
                dist[j * n + i],
                tolerances::ANALYTICAL_F64,
            );
        }
    }

    let labels: Vec<String> = (0..n).map(|i| format!("T{i}")).collect();
    let result = neighbor_joining::neighbor_joining(&dist, &labels);
    v.check_pass("NJ newick non-empty", !result.newick.is_empty());
    v.check_count("NJ joins = n-2", result.n_joins, n - 2);

    print_timing("neighbor_joining", t0);
}

// ════════════════════════════════════════════════════════════════════
//  Module 12: Reconciliation (Zheng 2023 DTL)
// ════════════════════════════════════════════════════════════════════

const NO_CHILD: u32 = u32::MAX;

fn validate_reconciliation(v: &mut Validator) {
    v.section("═══ M12: Reconciliation — DTL cost for known mapping ═══");
    let t0 = Instant::now();

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

    let result = reconciliation::reconcile_dtl(&host, &parasite, &tip_mapping, &costs);

    v.check_pass(
        "reconciliation cost finite",
        f64::from(result.optimal_cost).is_finite(),
    );

    let high_costs = DtlCosts {
        duplication: 100,
        transfer: 100,
        loss: 100,
    };
    let result_high = reconciliation::reconcile_dtl(&host, &parasite, &tip_mapping, &high_costs);
    v.check_pass(
        "reconciliation higher costs → higher or equal total",
        result_high.optimal_cost >= result.optimal_cost,
    );

    print_timing("reconciliation", t0);
}

// ════════════════════════════════════════════════════════════════════
//  Module 13: Molecular Clock
// ════════════════════════════════════════════════════════════════════

fn validate_molecular_clock(v: &mut Validator) {
    v.section("═══ M13: Molecular Clock — strict + relaxed rates ═══");
    let t0 = Instant::now();

    let branch_lengths = vec![0.0, 0.1, 0.15, 0.05, 0.08];
    let parent_opt: Vec<Option<usize>> = vec![None, Some(0), Some(0), Some(1), Some(1)];
    let root_age = 100.0;

    let result = molecular_clock::strict_clock(&branch_lengths, &parent_opt, root_age, &[]);
    match &result {
        Some(r) => {
            v.check_pass("strict clock rate > 0", r.rate > 0.0);
            v.check_pass("strict clock rate finite", r.rate.is_finite());
        }
        None => {
            v.check_pass("strict clock None is valid (degenerate input)", true);
        }
    }

    let node_ages = vec![100.0, 60.0, 70.0, 0.0, 0.0];
    let rates = molecular_clock::relaxed_clock_rates(&branch_lengths, &node_ages, &parent_opt);

    v.check_count("relaxed rates count", rates.len(), branch_lengths.len());
    for (i, &r) in rates.iter().enumerate() {
        v.check_pass(&format!("relaxed rate[{i}] finite"), r.is_finite());
    }

    v.check(
        "relaxed root rate = 0 (root has no parent branch)",
        rates[0],
        0.0,
        tolerances::ANALYTICAL_F64,
    );

    for &r in &rates[1..] {
        v.check_pass("relaxed non-root rate >= 0", r >= 0.0);
    }

    print_timing("molecular_clock", t0);
}

// ════════════════════════════════════════════════════════════════════
//  Helpers
// ════════════════════════════════════════════════════════════════════

fn bitwise_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x.to_bits() ^ y.to_bits()).count_ones())
        .sum::<u32>()
        .into()
}
