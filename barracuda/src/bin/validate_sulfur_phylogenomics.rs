// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(clippy::similar_names)]
//! Exp053 — Mateos & Anderson 2023: Sulfur-cycling enzyme phylogenomics.
//!
//! Validates DTL reconciliation, molecular clock, and Robinson-Foulds
//! primitives against synthetic gene/species tree pairs modeled after
//! the sulfur enzyme evolution analysis in Mateos et al. 2023.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Paper | Mateos et al. (2023) Science Advances 9:eade4847 |
//! | DOI | 10.1126/sciadv.ade4847 |
//! | Faculty | R. Anderson (Carleton College) |
//! | Baseline script | `scripts/mateos2023_sulfur_phylogenomics.py` |
//! | Baseline date | 2026-02-20 |
//! | Data | Figshare project 144267 (synthetic proxy for validation) |

use wetspring_barracuda::bio::{
    molecular_clock::{self, CalibrationPoint},
    reconciliation::{self, DtlCosts, FlatRecTree},
    robinson_foulds,
    unifrac::PhyloTree,
};
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

fn main() {
    let mut v = Validator::new("Exp053: Mateos 2023 Sulfur Enzyme Phylogenomics");

    validate_molecular_clock(&mut v);
    validate_dtl_reconciliation(&mut v);
    validate_rf_distance(&mut v);
    validate_python_parity(&mut v);

    v.finish();
}

fn validate_molecular_clock(v: &mut Validator) {
    v.section("── Molecular clock (strict) ──");

    // 5-node tree matching Python baseline
    let branch_lengths = vec![0.0, 0.1, 0.2, 0.05, 0.05];
    let parents: Vec<Option<usize>> = vec![None, Some(0), Some(0), Some(1), Some(1)];
    let root_age = 3000.0; // 3 Gya

    let result = molecular_clock::strict_clock(&branch_lengths, &parents, root_age, &[]).unwrap();

    v.check(
        "Clock rate positive",
        f64::from(u8::from(result.rate > 0.0)),
        1.0,
        0.0,
    );
    v.check(
        "Root age = 3000 Ma",
        result.node_ages[0],
        3000.0,
        tolerances::ANALYTICAL_F64,
    );
    v.check(
        "All node ages non-negative",
        f64::from(u8::from(result.node_ages.iter().all(|&a| a >= 0.0))),
        1.0,
        0.0,
    );

    let monotonic = (1..5).all(|i| match parents[i] {
        Some(p) => result.node_ages[p] > result.node_ages[i],
        None => true,
    });
    v.check(
        "Parent ages > child ages",
        f64::from(u8::from(monotonic)),
        1.0,
        0.0,
    );

    // Calibration satisfied
    let cals = vec![CalibrationPoint {
        node_id: 0,
        min_age_ma: 2500.0,
        max_age_ma: 3500.0,
    }];
    let result_cal =
        molecular_clock::strict_clock(&branch_lengths, &parents, root_age, &cals).unwrap();
    v.check(
        "Calibration constraint satisfied",
        f64::from(u8::from(result_cal.calibrations_satisfied)),
        1.0,
        0.0,
    );

    // Calibration violated
    let bad_cals = vec![CalibrationPoint {
        node_id: 0,
        min_age_ma: 5000.0,
        max_age_ma: 6000.0,
    }];
    let result_bad =
        molecular_clock::strict_clock(&branch_lengths, &parents, root_age, &bad_cals).unwrap();
    v.check(
        "Calibration violation detected",
        f64::from(u8::from(!result_bad.calibrations_satisfied)),
        1.0,
        0.0,
    );

    // Relaxed clock rates on strict tree → CV ≈ 0
    let rates = molecular_clock::relaxed_clock_rates(&branch_lengths, &result.node_ages, &parents);
    let positive_rates: Vec<f64> = rates.iter().copied().filter(|&r| r > 0.0).collect();
    let cv = molecular_clock::rate_variation_cv(&positive_rates);
    v.check(
        "Strict tree rate CV ≈ 0",
        f64::from(u8::from(cv < 1e-10)),
        1.0,
        0.0,
    );
}

fn validate_dtl_reconciliation(v: &mut Validator) {
    v.section("── DTL reconciliation ──");

    // Congruent trees → cost = 0
    let species = FlatRecTree {
        names: vec!["C".into(), "D".into(), "A".into(), "B".into(), "R".into()],
        left_child: vec![u32::MAX, u32::MAX, 0, u32::MAX, 2],
        right_child: vec![u32::MAX, u32::MAX, 1, u32::MAX, 3],
    };
    let gene = FlatRecTree {
        names: vec!["c".into(), "d".into(), "a".into(), "b".into(), "r".into()],
        left_child: vec![u32::MAX, u32::MAX, 0, u32::MAX, 2],
        right_child: vec![u32::MAX, u32::MAX, 1, u32::MAX, 3],
    };
    let tip_map = vec![
        ("c".into(), "C".into()),
        ("d".into(), "D".into()),
        ("a".into(), "A".into()),
        ("b".into(), "B".into()),
    ];
    let costs = DtlCosts::default();

    let result = reconciliation::reconcile_dtl(&species, &gene, &tip_map, &costs);
    v.check(
        "Congruent: optimal cost = 0",
        f64::from(result.optimal_cost),
        0.0,
        0.0,
    );

    // Incongruent trees → cost > 0
    let gene_inc = FlatRecTree {
        names: vec!["b".into(), "c".into(), "bc".into(), "a".into(), "r".into()],
        left_child: vec![u32::MAX, u32::MAX, 0, u32::MAX, 2],
        right_child: vec![u32::MAX, u32::MAX, 1, u32::MAX, 3],
    };
    let tip_map_inc = vec![
        ("b".into(), "B".into()),
        ("c".into(), "C".into()),
        ("a".into(), "A".into()),
    ];

    let species_3 = FlatRecTree {
        names: vec!["A".into(), "B".into(), "AB".into(), "C".into(), "R".into()],
        left_child: vec![u32::MAX, u32::MAX, 0, u32::MAX, 2],
        right_child: vec![u32::MAX, u32::MAX, 1, u32::MAX, 3],
    };

    let result_inc = reconciliation::reconcile_dtl(&species_3, &gene_inc, &tip_map_inc, &costs);
    v.check(
        "Incongruent: optimal cost > 0",
        f64::from(u8::from(result_inc.optimal_cost > 0)),
        1.0,
        0.0,
    );

    // Deterministic
    let result2 = reconciliation::reconcile_dtl(&species, &gene, &tip_map, &costs);
    v.check(
        "DTL reconciliation deterministic",
        f64::from(u8::from(result.optimal_cost == result2.optimal_cost)),
        1.0,
        0.0,
    );
}

fn validate_rf_distance(v: &mut Validator) {
    v.section("── Robinson-Foulds distance ──");

    let tree_a = PhyloTree::from_newick("((A:0.1,B:0.2):0.3,(C:0.4,D:0.5):0.6);");
    let tree_b = PhyloTree::from_newick("((A:0.1,B:0.2):0.3,(C:0.4,D:0.5):0.6);");
    let rf = robinson_foulds::rf_distance(&tree_a, &tree_b);
    v.check_count("RF(same topology) = 0", rf, 0);

    let tree_c = PhyloTree::from_newick("((A:0.1,C:0.2):0.3,(B:0.4,D:0.5):0.6);");
    let rf2 = robinson_foulds::rf_distance(&tree_a, &tree_c);
    v.check(
        "RF(different topology) > 0",
        f64::from(u8::from(rf2 > 0)),
        1.0,
        0.0,
    );

    let rf_norm = robinson_foulds::rf_distance_normalized(&tree_a, &tree_c);
    v.check(
        "RF normalized in [0,1]",
        f64::from(u8::from((0.0..=1.0).contains(&rf_norm))),
        1.0,
        0.0,
    );
}

fn validate_python_parity(v: &mut Validator) {
    v.section("── Python baseline parity ──");

    let branch_lengths = vec![0.0, 0.1, 0.2, 0.05, 0.05];
    let parents: Vec<Option<usize>> = vec![None, Some(0), Some(0), Some(1), Some(1)];

    let py_rate = 6.666_666_666_667e-5;
    let result = molecular_clock::strict_clock(&branch_lengths, &parents, 3000.0, &[]).unwrap();

    v.check("Python: clock rate", result.rate, py_rate, 1e-14);
    v.check(
        "Python: root age",
        result.node_ages[0],
        3000.0,
        tolerances::ANALYTICAL_F64,
    );
}
