// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exp036 — RF distances on real `PhyNetPy` DEFJ gene trees.
//!
//! # Provenance
//!
//! | Item            | Value                                                     |
//! |-----------------|-----------------------------------------------------------|
//! | Baseline commit | `e4358c5`                                                 |
//! | Baseline script | `scripts/phynetpy_rf_baseline.py`                         |
//! | Baseline output | `experiments/results/036_phynetpy_rf/python_baseline.json` |
//! | Data source     | `NakhlehLab/PhyNetPy` DEFJ/ (1,160 gene trees)           |
//! | Date            | 2026-02-20                                                |
//! | Exact command   | `python3 scripts/phynetpy_rf_baseline.py`                 |
//! | Hardware        | i9-12900K, 64GB DDR5, RTX 4070, Ubuntu 24.04              |
//!
//! Validates that Rust RF distance module produces identical results to the
//! Python baseline on real phylogenetic gene trees with 25 leaves each.

use wetspring_barracuda::bio::robinson_foulds::{rf_distance, rf_distance_normalized};
use wetspring_barracuda::bio::unifrac::PhyloTree;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

// First 3 gene trees from DEFJ/10Genes/withOG/E/g10/n3/t20/r1
const TREE_0: &str = "(01oA:0.02989072,((((02xA:0.00130491,01xA:0.00130491):0.00712181,03xA:0.00842672):0.00312531,03aA:0.01155203):0.00201325,(((01zA:0.00274507,02zA:0.00274507):0.00522019,((01aA:0.00093992,02aA:0.00093992):0.00675429,((02yA:0.00072649,(01yA:5.535e-05,03yA:5.535e-05):0.00067114):0.00259255,03zA:0.00331904):0.00437517):0.00027105):0.00467598,((03bA:0.00933968,01yB:0.00933968):0.00247815,(((((03yB:0.000662,02yB:0.000662):0.00261864,(03zB:0.00286729,02zB:0.00286729):0.00041335):0.00438422,(02bA:0.00096306,01bA:0.00096306):0.0067018):0.00129576,01zB:0.00896062):0.000275069999999999,(02xB:0.00626878,(03xB:0.00155405,01xB:0.00155405):0.00471473):0.00296691):0.00258214):0.00082341):0.000924040000000001):0.01632544);";

const TREE_1: &str = "(((02xB:0.00735262,(01xB:0.00043667,03xB:0.00043667):0.00691595):0.00542853,(((01aA:0.00209739,(02aA:0.0013554,03aA:0.0013554):0.00074199):0.00496948,((01zA:0.0045122,03yA:0.0045122):0.00017167,((03zA:0.00129095,02zA:0.00129095):0.00297682,(01yA:0.00212511,02yA:0.00212511):0.00214266):0.0004161):0.002383):0.00352198,(01xA:0.00325405,(02xA:0.00055501,03xA:0.00055501):0.00269904):0.0073348):0.0021923):0.01662885,(01oA:0.02930846,(02zB:0.0125142,((01zB:0.00201593,03zB:0.00201593):0.00609511,(((01yB:0.00157776,02yB:0.00157776):0.00293639,03yB:0.00451415):0.00208403,(02bA:0.00037418,(03bA:0.00010048,01bA:0.00010048):0.0002737):0.006224):0.00151286):0.00440316):0.01679426):0.000101539999999997);";

const TREE_2: &str = "((((((01aA:0.00087835,02aA:0.00087835):0.00033491,03aA:0.00121326):0.00814283,((03zA:0.00034167,02zA:0.00034167):0.005471,(01zA:0.00496283,((03yA:0.00062903,02yA:0.00062903):0.00114376,01yA:0.00177279):0.00319004):0.00084984):0.00354342):0.00530192,((01zB:0.00588153,((02zB:0.00170969,03zB:0.00170969):0.00255941,((02yB:0.00017335,03yB:0.00017335):0.001219,01yB:0.00139235):0.00287675):0.00161243):0.00507387,((03bA:0.0001473,02bA:0.0001473):0.00113639,01bA:0.00128369):0.00967171):0.00370261):0.00920802,01oA:0.02386603):0.00211073,((02xA:0.0102362,(03xA:0.00272858,01xA:0.00272858):0.00750762):0.00212711,((03xB:0.00037747,02xB:0.00037747):0.00799987,01xB:0.00837734):0.00398597):0.01361345);";

fn main() {
    let mut v = Validator::new("Exp036: PhyNetPy Gene Tree RF Distances");

    let t0 = PhyloTree::from_newick(TREE_0);
    let t1 = PhyloTree::from_newick(TREE_1);
    let t2 = PhyloTree::from_newick(TREE_2);

    // ── Section 1: Leaf counts ──────────────────────────────────
    v.section("── Leaf counts (25-leaf gene trees) ──");
    let n0 = t0
        .nodes
        .iter()
        .filter(|n| n.children.is_empty() && !n.label.is_empty())
        .count();
    let n1 = t1
        .nodes
        .iter()
        .filter(|n| n.children.is_empty() && !n.label.is_empty())
        .count();
    let n2 = t2
        .nodes
        .iter()
        .filter(|n| n.children.is_empty() && !n.label.is_empty())
        .count();
    v.check_count("tree_0 leaves", n0, 25);
    v.check_count("tree_1 leaves", n1, 25);
    v.check_count("tree_2 leaves", n2, 25);

    // ── Section 2: RF vs Python baseline ────────────────────────
    v.section("── RF distance vs Python baseline ──");
    v.check_count("RF(tree_0, tree_1)", rf_distance(&t0, &t1), 38);
    v.check_count("RF(tree_0, tree_2)", rf_distance(&t0, &t2), 32);
    v.check_count("RF(tree_1, tree_2)", rf_distance(&t1, &t2), 26);

    // ── Section 3: Symmetry ─────────────────────────────────────
    v.section("── RF symmetry ──");
    v.check_count(
        "RF(0,1) == RF(1,0)",
        rf_distance(&t0, &t1),
        rf_distance(&t1, &t0),
    );
    v.check_count(
        "RF(0,2) == RF(2,0)",
        rf_distance(&t0, &t2),
        rf_distance(&t2, &t0),
    );
    v.check_count(
        "RF(1,2) == RF(2,1)",
        rf_distance(&t1, &t2),
        rf_distance(&t2, &t1),
    );

    // ── Section 4: Identity ─────────────────────────────────────
    v.section("── RF self-distance = 0 ──");
    v.check_count("RF(tree_0, tree_0)", rf_distance(&t0, &t0), 0);
    v.check_count("RF(tree_1, tree_1)", rf_distance(&t1, &t1), 0);

    // ── Section 5: Normalized RF ────────────────────────────────
    v.section("── Normalized RF ──");
    let nrf_self = rf_distance_normalized(&t0, &t0);
    v.check("nRF(self)", nrf_self, 0.0, tolerances::ANALYTICAL_F64);
    let nrf_01 = rf_distance_normalized(&t0, &t1);
    v.check(
        "nRF(0,1) in [0,1]",
        nrf_01.clamp(0.0, 1.0),
        nrf_01,
        tolerances::ANALYTICAL_F64,
    );

    // ── Section 6: Determinism ──────────────────────────────────
    v.section("── Determinism ──");
    let d1 = rf_distance(&t0, &t1);
    let d2 = rf_distance(&t0, &t1);
    v.check_count("deterministic_rerun", d1, d2);

    // ── Section 7: Triangle inequality ──────────────────────────
    v.section("── Triangle inequality ──");
    let d01 = rf_distance(&t0, &t1);
    let d02 = rf_distance(&t0, &t2);
    let d12 = rf_distance(&t1, &t2);
    let triangle_holds = d01 <= d02 + d12 && d02 <= d01 + d12 && d12 <= d01 + d02;
    v.check_count("triangle_inequality", usize::from(triangle_holds), 1);

    v.finish();
}
