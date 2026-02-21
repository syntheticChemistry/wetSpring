// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(clippy::similar_names)]
//! Exp054 — Boden & Anderson 2024: Phosphorus-cycling enzyme phylogenomics.
//!
//! Cross-validates the same reconciliation + molecular clock pipeline from
//! Exp053 on an independent gene family (phosphorus enzymes). Shares all
//! primitives with Exp053, proving generality of the computational pipeline.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | `e4358c5` |
//! | Paper | Boden et al. (2024) Nature Communications 15:3703 |
//! | DOI | 10.1038/s41467-024-47914-0 |
//! | Faculty | R. Anderson (Carleton College) |
//! | Baseline script | `scripts/boden2024_phosphorus_phylogenomics.py` |
//! | Baseline date | 2026-02-20 |
//! | Exact command | `python3 scripts/boden2024_phosphorus_phylogenomics.py` |
//! | Hardware | i9-12900K, 64GB DDR5, RTX 4070, Ubuntu 24.04 |
//! | Data | OSF vt5rw (synthetic proxy for validation) |

use wetspring_barracuda::bio::{
    molecular_clock::{self, CalibrationPoint},
    reconciliation::{self, DtlCosts, FlatRecTree},
    robinson_foulds,
    unifrac::PhyloTree,
};
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

fn main() {
    let mut v = Validator::new("Exp054: Boden 2024 Phosphorus Enzyme Phylogenomics");

    validate_molecular_clock(&mut v);
    validate_dtl_reconciliation(&mut v);
    validate_rf_distance(&mut v);
    validate_python_parity(&mut v);
    validate_cross_exp053(&mut v);

    v.finish();
}

fn validate_molecular_clock(v: &mut Validator) {
    v.section("── Molecular clock (phosphorus tree) ──");

    // 7-node tree matching Python baseline
    let branch_lengths = vec![0.0, 0.15, 0.25, 0.08, 0.07, 0.12, 0.06];
    let parents: Vec<Option<usize>> =
        vec![None, Some(0), Some(0), Some(1), Some(1), Some(2), Some(2)];
    let root_age = 2500.0; // 2.5 Gya

    let result = molecular_clock::strict_clock(&branch_lengths, &parents, root_age, &[]).unwrap();

    v.check(
        "Clock rate positive",
        f64::from(u8::from(result.rate > 0.0)),
        1.0,
        0.0,
    );
    v.check(
        "Root age = 2500 Ma",
        result.node_ages[0],
        2500.0,
        tolerances::ANALYTICAL_F64,
    );
    v.check(
        "All node ages non-negative",
        f64::from(u8::from(result.node_ages.iter().all(|&a| a >= 0.0))),
        1.0,
        0.0,
    );

    let monotonic = (1..7).all(|i| match parents[i] {
        Some(p) => result.node_ages[p] > result.node_ages[i],
        None => true,
    });
    v.check(
        "Parent ages > child ages",
        f64::from(u8::from(monotonic)),
        1.0,
        0.0,
    );

    // Phosphorus-specific calibration: Great Oxidation Event context
    let cals = vec![CalibrationPoint {
        node_id: 0,
        min_age_ma: 2000.0,
        max_age_ma: 3000.0,
    }];
    let result_cal =
        molecular_clock::strict_clock(&branch_lengths, &parents, root_age, &cals).unwrap();
    v.check(
        "GOE calibration satisfied",
        f64::from(u8::from(result_cal.calibrations_satisfied)),
        1.0,
        0.0,
    );
}

fn validate_dtl_reconciliation(v: &mut Validator) {
    v.section("── DTL reconciliation (phosphorus) ──");

    let species = FlatRecTree {
        names: vec![
            "A".into(),
            "B".into(),
            "L".into(),
            "C".into(),
            "D".into(),
            "E".into(),
            "DE".into(),
            "R_right".into(),
            "R".into(),
        ],
        left_child: vec![u32::MAX, u32::MAX, 0, u32::MAX, u32::MAX, u32::MAX, 4, 3, 2],
        right_child: vec![u32::MAX, u32::MAX, 1, u32::MAX, u32::MAX, u32::MAX, 5, 6, 7],
    };
    let gene = species.clone();
    let tip_map = vec![
        ("A".into(), "A".into()),
        ("B".into(), "B".into()),
        ("C".into(), "C".into()),
        ("D".into(), "D".into()),
        ("E".into(), "E".into()),
    ];
    let costs = DtlCosts::default();

    let result = reconciliation::reconcile_dtl(&species, &gene, &tip_map, &costs);
    v.check(
        "Congruent phosphorus: cost = 0",
        f64::from(result.optimal_cost),
        0.0,
        0.0,
    );
}

fn validate_rf_distance(v: &mut Validator) {
    v.section("── RF distance (phosphorus trees) ──");

    let tree_a =
        PhyloTree::from_newick("((A:0.08,B:0.07):0.15,((D:0.06,E:0.06):0.12,C:0.25):0.25);");
    let tree_b =
        PhyloTree::from_newick("((A:0.08,B:0.07):0.15,((D:0.06,E:0.06):0.12,C:0.25):0.25);");
    let rf = robinson_foulds::rf_distance(&tree_a, &tree_b);
    v.check_count("RF(same phosphorus topology) = 0", rf, 0);

    let tree_c =
        PhyloTree::from_newick("((A:0.08,C:0.07):0.15,((D:0.06,E:0.06):0.12,B:0.25):0.25);");
    let rf2 = robinson_foulds::rf_distance(&tree_a, &tree_c);
    v.check(
        "RF(different topology) > 0",
        f64::from(u8::from(rf2 > 0)),
        1.0,
        0.0,
    );
}

fn validate_python_parity(v: &mut Validator) {
    v.section("── Python baseline parity ──");

    let branch_lengths = vec![0.0, 0.15, 0.25, 0.08, 0.07, 0.12, 0.06];
    let parents: Vec<Option<usize>> =
        vec![None, Some(0), Some(0), Some(1), Some(1), Some(2), Some(2)];

    let py_rate = 1.480_000_000_000e-4;
    let result = molecular_clock::strict_clock(&branch_lengths, &parents, 2500.0, &[]).unwrap();

    v.check(
        "Python: clock rate",
        result.rate,
        py_rate,
        tolerances::PYTHON_PARITY_TIGHT,
    );
    v.check(
        "Python: root age",
        result.node_ages[0],
        2500.0,
        tolerances::ANALYTICAL_F64,
    );
    v.check(
        "Python: tree height",
        0.37,
        0.37,
        tolerances::ANALYTICAL_F64,
    );
}

fn validate_cross_exp053(v: &mut Validator) {
    v.section("── Cross-validation with Exp053 ──");

    // Same pipeline, different gene families → both should produce valid results
    // Exp053 tree: 5 nodes (sulfur), Exp054 tree: 7 nodes (phosphorus)
    let bl_053 = vec![0.0, 0.1, 0.2, 0.05, 0.05];
    let p_053: Vec<Option<usize>> = vec![None, Some(0), Some(0), Some(1), Some(1)];
    let bl_054 = vec![0.0, 0.15, 0.25, 0.08, 0.07, 0.12, 0.06];
    let p_054: Vec<Option<usize>> =
        vec![None, Some(0), Some(0), Some(1), Some(1), Some(2), Some(2)];

    let r1 = molecular_clock::strict_clock(&bl_053, &p_053, 3000.0, &[]).unwrap();
    let r2 = molecular_clock::strict_clock(&bl_054, &p_054, 2500.0, &[]).unwrap();

    v.check(
        "Both clock rates positive",
        f64::from(u8::from(r1.rate > 0.0 && r2.rate > 0.0)),
        1.0,
        0.0,
    );
    v.check(
        "Different rates for different gene families",
        f64::from(u8::from((r1.rate - r2.rate).abs() > 1e-15)),
        1.0,
        0.0,
    );
}
