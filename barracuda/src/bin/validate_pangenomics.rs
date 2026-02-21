// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exp056 — Moulana & Anderson 2020: Sulfurovum pangenomics.
//!
//! Validates pangenome analysis, gene presence-absence partitioning,
//! `Heap's` law fitting, and enrichment testing (hypergeometric +
//! `Benjamini-Hochberg` FDR).
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | `e4358c5` |
//! | Paper | Moulana et al. (2020) mSystems 5:e00673-19 |
//! | DOI | 10.1128/mSystems.00673-19 |
//! | Faculty | R. Anderson (Carleton College) |
//! | BioProjects | PRJNA283159, PRJEB5293 |
//! | Baseline script | `scripts/moulana2020_pangenomics.py` |
//! | Baseline date | 2026-02-20 |
//! | Exact command | `python3 scripts/moulana2020_pangenomics.py` |
//! | Hardware | i9-12900K, 64GB DDR5, RTX 4070, Ubuntu 24.04 |

use wetspring_barracuda::bio::pangenome::{self, GeneCluster};
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

fn main() {
    let mut v = Validator::new("Exp056: Moulana 2020 Pangenomics Validation");

    validate_core_accessory_unique(&mut v);
    validate_heaps_law(&mut v);
    validate_enrichment(&mut v);
    validate_bh_correction(&mut v);
    validate_python_parity(&mut v);

    v.finish();
}

fn sample_clusters() -> Vec<GeneCluster> {
    vec![
        GeneCluster {
            id: "core1".into(),
            presence: vec![true, true, true],
        },
        GeneCluster {
            id: "core2".into(),
            presence: vec![true, true, true],
        },
        GeneCluster {
            id: "acc1".into(),
            presence: vec![true, true, false],
        },
        GeneCluster {
            id: "unique1".into(),
            presence: vec![true, false, false],
        },
        GeneCluster {
            id: "unique2".into(),
            presence: vec![false, false, true],
        },
    ]
}

fn validate_core_accessory_unique(v: &mut Validator) {
    v.section("── Core / accessory / unique partitioning ──");

    let clusters = sample_clusters();
    let result = pangenome::analyze(&clusters, 3);

    v.check_count("Core genome size = 2", result.core_size, 2);
    v.check_count("Accessory genome size = 1", result.accessory_size, 1);
    v.check_count("Unique genome size = 2", result.unique_size, 2);
    v.check_count("Total = 5", result.total_size, 5);
    v.check_count("N genomes = 3", result.n_genomes, 3);

    // All-core genome
    let all_core = vec![
        GeneCluster {
            id: "g1".into(),
            presence: vec![true, true],
        },
        GeneCluster {
            id: "g2".into(),
            presence: vec![true, true],
        },
    ];
    let result = pangenome::analyze(&all_core, 2);
    v.check_count("All-core: core = 2", result.core_size, 2);
    v.check_count("All-core: accessory = 0", result.accessory_size, 0);
    v.check_count("All-core: unique = 0", result.unique_size, 0);
}

fn validate_heaps_law(v: &mut Validator) {
    v.section("── Heap's law (pangenome openness) ──");

    let clusters = sample_clusters();
    let result = pangenome::analyze(&clusters, 3);

    v.check(
        "Heap's alpha computed",
        f64::from(u8::from(result.heaps_alpha.is_some())),
        1.0,
        0.0,
    );

    // Large open pangenome: many unique genes per genome
    let open_clusters: Vec<GeneCluster> = (0..20)
        .map(|i| {
            let presence = if i < 3 {
                vec![true; 5] // core
            } else {
                let mut p = vec![false; 5];
                p[i % 5] = true; // unique to one genome
                p
            };
            GeneCluster {
                id: format!("g{i}"),
                presence,
            }
        })
        .collect();
    let open_result = pangenome::analyze(&open_clusters, 5);
    v.check(
        "Open pangenome: unique > core",
        f64::from(u8::from(open_result.unique_size > open_result.core_size)),
        1.0,
        0.0,
    );
}

fn validate_enrichment(v: &mut Validator) {
    v.section("── Enrichment testing (hypergeometric) ──");

    // Enriched: 8 of 10 drawn from population of 100 with 20 successes
    let p_enriched = pangenome::hypergeometric_pvalue(8, 10, 20, 100);
    v.check(
        "Enriched: p < 0.05",
        f64::from(u8::from(p_enriched < 0.05)),
        1.0,
        0.0,
    );

    // Not enriched: 2 of 10 (expected ≈ 2)
    let p_not = pangenome::hypergeometric_pvalue(2, 10, 20, 100);
    v.check(
        "Not enriched: p ≈ 1.0",
        f64::from(u8::from((p_not - 1.0).abs() < 0.01)),
        1.0,
        0.0,
    );

    // Edge: zero population
    let p_zero = pangenome::hypergeometric_pvalue(0, 0, 0, 0);
    v.check("Zero population: p = 1.0", p_zero, 1.0, 0.0);
}

fn validate_bh_correction(v: &mut Validator) {
    v.section("── Benjamini-Hochberg FDR ──");

    let pvals = [0.01, 0.04, 0.03, 0.5];
    let adj = pangenome::benjamini_hochberg(&pvals);

    v.check(
        "BH: all adjusted in [0,1]",
        f64::from(u8::from(adj.iter().all(|&p| (0.0..=1.0).contains(&p)))),
        1.0,
        0.0,
    );
    v.check(
        "BH: adjusted[0] <= adjusted[3]",
        f64::from(u8::from(adj[0] <= adj[3])),
        1.0,
        0.0,
    );
    v.check(
        "BH: smallest raw → smallest adjusted",
        f64::from(u8::from(adj[0] < adj[3])),
        1.0,
        0.0,
    );

    // Empty
    let empty = pangenome::benjamini_hochberg(&[]);
    v.check_count("BH empty: 0 results", empty.len(), 0);
}

fn validate_python_parity(v: &mut Validator) {
    v.section("── Python baseline parity ──");

    let clusters = sample_clusters();
    let result = pangenome::analyze(&clusters, 3);

    v.check_count("Python: core = 2", result.core_size, 2);
    v.check_count("Python: accessory = 1", result.accessory_size, 1);
    v.check_count("Python: unique = 2", result.unique_size, 2);
    v.check_count("Python: total = 5", result.total_size, 5);

    let py_enriched = 3.263_440e-7;
    let rust_enriched = pangenome::hypergeometric_pvalue(8, 10, 20, 100);
    v.check(
        "Python: enriched p-value",
        rust_enriched,
        py_enriched,
        tolerances::PYTHON_PVALUE,
    );

    let pvals = [0.01, 0.04, 0.03, 0.5];
    let adj = pangenome::benjamini_hochberg(&pvals);
    v.check("Python: BH adj[0]", adj[0], 0.04, tolerances::PYTHON_PARITY);
    v.check("Python: BH adj[3]", adj[3], 0.5, tolerances::PYTHON_PARITY);
}
