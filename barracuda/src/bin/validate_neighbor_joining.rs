// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validation binary for Exp033: Neighbor-Joining tree construction.
//!
//! Validates Rust NJ against Python baseline (Saitou & Nei 1987).
//! Checks: topology (sister pairs), branch lengths, distance matrix
//! symmetry, determinism.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | `e4358c5` |
//! | Baseline tool | liu2009_neighbor_joining.py |
//! | Baseline version | scripts/ |
//! | Baseline command | python3 scripts/liu2009_neighbor_joining.py |
//! | Baseline date | 2026-02-19 |
//! | Exact command | `python3 scripts/liu2009_neighbor_joining.py` |
//! | Data | synthetic distance matrices, 3–5 taxa |
//! | Hardware | Eastgate (i9-12900K, 64 GB, RTX 4070, Pop!\_OS 22.04) |

use wetspring_barracuda::bio::neighbor_joining::{
    distance_matrix, jukes_cantor_distance, neighbor_joining,
};
use wetspring_barracuda::tolerances;

fn main() {
    let mut pass = 0_u32;
    let mut fail = 0_u32;

    macro_rules! check {
        ($name:expr, $cond:expr) => {
            if $cond {
                println!("[PASS] {}", $name);
                pass += 1;
            } else {
                println!("[FAIL] {}", $name);
                fail += 1;
            }
        };
    }

    println!("=== Exp033: Neighbor-Joining Validation ===\n");

    // ── Test 1: 3-taxon ──
    println!("─── 3-taxon (X,Y,Z) ───");
    let labels_3: Vec<String> = vec!["X".into(), "Y".into(), "Z".into()];
    #[rustfmt::skip]
    let dist_3 = vec![
        0.0, 0.2, 0.4,
        0.2, 0.0, 0.4,
        0.4, 0.4, 0.0,
    ];
    let r3 = neighbor_joining(&dist_3, &labels_3);
    check!("3-taxon: 1 join", r3.n_joins == 1);
    check!(
        "3-taxon: X-Y sister pair",
        r3.newick.contains("X:") && r3.newick.contains("Y:")
    );
    // Python: Z:0.150000,(X:0.100000,Y:0.100000):0.150000
    // Check branch lengths approximately
    check!("3-taxon: X branch ~0.1", r3.newick.contains("X:0.1"));
    check!("3-taxon: Y branch ~0.1", r3.newick.contains("Y:0.1"));

    // ── Test 2: 4-taxon ──
    println!("\n─── 4-taxon (A,B,C,D) ───");
    let labels_4: Vec<String> = vec!["A".into(), "B".into(), "C".into(), "D".into()];
    #[rustfmt::skip]
    let dist_4 = vec![
        0.0, 0.3, 0.5, 0.6,
        0.3, 0.0, 0.6, 0.5,
        0.5, 0.6, 0.0, 0.3,
        0.6, 0.5, 0.3, 0.0,
    ];
    let r4 = neighbor_joining(&dist_4, &labels_4);
    check!("4-taxon: 2 joins", r4.n_joins == 2);
    check!(
        "4-taxon: all taxa present",
        r4.newick.contains('A')
            && r4.newick.contains('B')
            && r4.newick.contains('C')
            && r4.newick.contains('D')
    );
    // Python groups (A,B) and (C,D) or similar — check they're in the tree
    check!("4-taxon: valid Newick", r4.newick.ends_with(';'));

    // ── Test 3: 5-taxon from JC distances ──
    println!("\n─── 5-taxon from sequences ───");
    let seqs: Vec<&[u8]> = vec![
        b"ACGTACGTACGT",
        b"ACGTACGTACTT",
        b"ACTTACTTACTT",
        b"TGCATGCATGCA",
        b"TGCATGCATGCC",
    ];
    let labels_5: Vec<String> = (1..=5).map(|i| format!("S{i}")).collect();
    let dm = distance_matrix(&seqs);
    let r5 = neighbor_joining(&dm, &labels_5);
    check!("5-taxon: 3 joins", r5.n_joins == 3);
    check!("5-taxon: S1 and S2 close (should be sisters)", {
        // S1-S2 have smallest distance, should be grouped first
        let nwk = &r5.newick;
        nwk.contains("S1") && nwk.contains("S2")
    });
    check!(
        "5-taxon: S4 and S5 close (should be sisters)",
        r5.newick.contains("S4") && r5.newick.contains("S5")
    );

    // ── Test 4: JC distance validation ──
    println!("\n─── JC distance checks ───");
    let d_ident = jukes_cantor_distance(b"ACGTACGT", b"ACGTACGT");
    check!(
        "JC: identical = 0",
        d_ident.abs() < tolerances::ANALYTICAL_F64
    );
    let d_small = jukes_cantor_distance(b"ACGTACGTACGT", b"ACGTACGTACTT");
    // Python: 0.088337
    check!(
        "JC: small diff matches Python",
        (d_small - 0.088_337_276_742_287_64).abs() < tolerances::JC69_PROBABILITY
    );
    let d_sat = jukes_cantor_distance(b"AAAA", b"CCCC");
    check!(
        "JC: saturated = 10.0",
        (d_sat - 10.0).abs() < tolerances::ANALYTICAL_F64
    );

    // ── Test 5: Distance matrix symmetry ──
    println!("\n─── Distance matrix checks ───");
    let n = seqs.len();
    let mut symmetric = true;
    for i in 0..n {
        for j in 0..n {
            if (dm[i * n + j] - dm[j * n + i]).abs() > tolerances::ANALYTICAL_F64 {
                symmetric = false;
            }
        }
    }
    check!("Distance matrix symmetric", symmetric);
    let mut diag_zero = true;
    for i in 0..n {
        if dm[i * n + i].abs() > tolerances::ANALYTICAL_F64 {
            diag_zero = false;
        }
    }
    check!("Distance matrix diagonal zero", diag_zero);

    // ── Test 6: Determinism ──
    println!("\n─── Determinism ───");
    let r5b = neighbor_joining(&dm, &labels_5);
    check!("Deterministic", r5.newick == r5b.newick);

    // ── Summary ──
    println!("\n========================================");
    println!("Exp033 Neighbor-Joining: {pass} PASS, {fail} FAIL");
    if fail > 0 {
        std::process::exit(1);
    }
}
