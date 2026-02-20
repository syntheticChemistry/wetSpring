// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validation binary for Exp034: DTL Reconciliation.
//!
//! Validates Rust DTL reconciliation against Python baseline
//! (Zheng et al. 2023, Bansal et al. 2012).

use wetspring_barracuda::bio::reconciliation::{
    reconcile_batch, reconcile_dtl, DtlCosts, DtlEvent, FlatRecTree,
};

const NO_CHILD: u32 = u32::MAX;

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

    let costs = DtlCosts::default();
    println!("=== Exp034: DTL Reconciliation Validation ===\n");

    // ── Test 1: Congruent 2-leaf trees (zero cost) ──
    println!("─── Congruent trees ───");
    let host_2 = FlatRecTree {
        names: vec!["H_A".into(), "H_B".into(), "H_AB".into()],
        left_child: vec![NO_CHILD, NO_CHILD, 0],
        right_child: vec![NO_CHILD, NO_CHILD, 1],
    };
    let para_2 = FlatRecTree {
        names: vec!["P_A".into(), "P_B".into(), "P_AB".into()],
        left_child: vec![NO_CHILD, NO_CHILD, 0],
        right_child: vec![NO_CHILD, NO_CHILD, 1],
    };
    let tip_map_cong = vec![("P_A".into(), "H_A".into()), ("P_B".into(), "H_B".into())];
    let r1 = reconcile_dtl(&host_2, &para_2, &tip_map_cong, &costs);
    // Python: cost=0, host=H_AB
    check!("Congruent: cost=0", r1.optimal_cost == 0);
    check!("Congruent: mapped to H_AB", r1.optimal_host == "H_AB");
    check!(
        "Congruent: root event is speciation",
        r1.event_table[2 * 3 + 2] == DtlEvent::Speciation
    );

    // ── Test 2: Duplication scenario (4-leaf host) ──
    println!("\n─── Duplication scenario ───");
    let host_4 = FlatRecTree {
        names: vec![
            "H_A".into(),
            "H_B".into(),
            "H_AB".into(),
            "H_C".into(),
            "H_D".into(),
            "H_CD".into(),
            "H_root".into(),
        ],
        left_child: vec![NO_CHILD, NO_CHILD, 0, NO_CHILD, NO_CHILD, 3, 2],
        right_child: vec![NO_CHILD, NO_CHILD, 1, NO_CHILD, NO_CHILD, 4, 5],
    };
    let para_dup = FlatRecTree {
        names: vec![
            "P_1".into(),
            "P_2".into(),
            "P_12".into(),
            "P_3".into(),
            "P_root".into(),
        ],
        left_child: vec![NO_CHILD, NO_CHILD, 0, NO_CHILD, 2],
        right_child: vec![NO_CHILD, NO_CHILD, 1, NO_CHILD, 3],
    };
    let tip_map_dup = vec![
        ("P_1".into(), "H_A".into()),
        ("P_2".into(), "H_A".into()),
        ("P_3".into(), "H_C".into()),
    ];
    let r2 = reconcile_dtl(&host_4, &para_dup, &tip_map_dup, &costs);
    // Python: cost=4, host=H_root
    check!("Duplication: cost matches Python (4)", r2.optimal_cost == 4);
    check!("Duplication: mapped to H_root", r2.optimal_host == "H_root");
    check!("Duplication: cost > 0", r2.optimal_cost > 0);

    // ── Test 3: Simple loss scenario ──
    println!("\n─── Loss/co-speciation scenario ───");
    let host_3 = FlatRecTree {
        names: vec![
            "H_A".into(),
            "H_B".into(),
            "H_AB".into(),
            "H_C".into(),
            "H_root".into(),
        ],
        left_child: vec![NO_CHILD, NO_CHILD, 0, NO_CHILD, 2],
        right_child: vec![NO_CHILD, NO_CHILD, 1, NO_CHILD, 3],
    };
    let para_loss = FlatRecTree {
        names: vec!["P_A".into(), "P_C".into(), "P_root".into()],
        left_child: vec![NO_CHILD, NO_CHILD, 0],
        right_child: vec![NO_CHILD, NO_CHILD, 1],
    };
    let tip_map_loss = vec![("P_A".into(), "H_A".into()), ("P_C".into(), "H_C".into())];
    let r3 = reconcile_dtl(&host_3, &para_loss, &tip_map_loss, &costs);
    // Python: cost=1, host=H_root
    check!("Loss: cost matches Python (1)", r3.optimal_cost == 1);
    check!("Loss: mapped to H_root", r3.optimal_host == "H_root");

    // ── Test 4: DP table dimensions ──
    println!("\n─── DP table checks ───");
    check!(
        "Table dimensions correct (congruent)",
        r1.cost_table.len() == 3 * 3 && r1.event_table.len() == 3 * 3
    );
    check!(
        "Table dimensions correct (duplication)",
        r2.cost_table.len() == 5 * 7 && r2.event_table.len() == 5 * 7
    );

    // ── Test 5: Batch reconciliation ──
    println!("\n─── Batch reconciliation ───");
    let batch_results = reconcile_batch(
        &host_2,
        &[(&para_2, &tip_map_cong), (&para_2, &tip_map_cong)],
        &costs,
    );
    check!("Batch: returns 2 results", batch_results.len() == 2);
    check!(
        "Batch: both congruent = 0",
        batch_results[0].optimal_cost == 0 && batch_results[1].optimal_cost == 0
    );

    // ── Test 6: Determinism ──
    println!("\n─── Determinism ───");
    let r2b = reconcile_dtl(&host_4, &para_dup, &tip_map_dup, &costs);
    check!("Deterministic (cost)", r2.optimal_cost == r2b.optimal_cost);
    check!("Deterministic (host)", r2.optimal_host == r2b.optimal_host);

    // ── Summary ──
    println!("\n========================================");
    println!("Exp034 DTL Reconciliation: {pass} PASS, {fail} FAIL");
    if fail > 0 {
        std::process::exit(1);
    }
}
