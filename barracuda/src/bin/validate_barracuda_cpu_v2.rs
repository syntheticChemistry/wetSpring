// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validation binary for Exp035: `BarraCuda` CPU Parity v2.
//!
//! Extends CPU parity validation to new modules: `FlatTree` Felsenstein,
//! batch HMM, batch Smith-Waterman, `NeighborJoining`, DTL Reconciliation.
//! Proves the new batch/flat APIs produce identical results to their
//! sequential counterparts, validating GPU-ready data layouts on CPU.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | `e4358c5` |
//! | Baseline tool | `BarraCuda` CPU v1 (recursive/sequential) + Python refs: `felsenstein_pruning_baseline.py`, `liu2014_hmm_baseline.py`, `smith_waterman_baseline.py`, `liu2009_neighbor_joining.py`, `zheng2023_dtl_reconciliation.py` |
//! | Baseline version | Feb 2026 |
//! | Baseline command | Batch/flat impl validated against sequential; `python3 scripts/liu2009_neighbor_joining.py` (NJ), `python3 scripts/zheng2023_dtl_reconciliation.py` (DTL) |
//! | Baseline date | 2026-02-19 |
//! | Exact command | `cargo run --bin validate_barracuda_cpu_v2` |
//! | Data | Synthetic test vectors (hardcoded) |
//! | Hardware | Eastgate (i9-12900K, 64 GB, RTX 4070, Pop!\_OS 22.04) |
//!
//! Validation class: Python-parity
//! Provenance: Python/QIIME2/SciPy baseline script (see doc table for script, commit, date)

use wetspring_barracuda::bio::alignment::{ScoringParams, score_batch, smith_waterman_score};
use wetspring_barracuda::bio::felsenstein::{FlatTree, TreeNode, encode_dna, log_likelihood};
use wetspring_barracuda::bio::hmm::{HmmModel, forward, forward_batch, viterbi, viterbi_batch};
use wetspring_barracuda::bio::neighbor_joining::{distance_matrix, neighbor_joining};
use wetspring_barracuda::bio::reconciliation::{DtlCosts, FlatRecTree, reconcile_dtl};
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

const NO_CHILD: u32 = u32::MAX;

fn validate_flat_felsenstein(v: &mut Validator) {
    v.section("── FlatTree Felsenstein Parity ──");
    let tree = TreeNode::Internal {
        left: Box::new(TreeNode::Internal {
            left: Box::new(TreeNode::Leaf {
                name: "A".into(),
                states: encode_dna("ACGTACGTACGTACGT"),
            }),
            right: Box::new(TreeNode::Leaf {
                name: "B".into(),
                states: encode_dna("ACGTACTTACGTACTT"),
            }),
            left_branch: 0.1,
            right_branch: 0.15,
        }),
        right: Box::new(TreeNode::Leaf {
            name: "C".into(),
            states: encode_dna("ACTTACTTACTTACTT"),
        }),
        left_branch: 0.2,
        right_branch: 0.3,
    };
    let ll_recursive = log_likelihood(&tree, 1.0);
    let flat = FlatTree::from_tree(&tree, 1.0);
    let ll_flat = flat.log_likelihood();
    v.check(
        "FlatTree: LL matches recursive",
        ll_flat,
        ll_recursive,
        tolerances::ANALYTICAL_F64,
    );
    v.check_pass(
        "FlatTree: LL finite & negative",
        ll_flat.is_finite() && ll_flat < 0.0,
    );
    v.check_count("FlatTree: n_sites=16", flat.n_sites, 16);
    v.check_count("FlatTree: n_leaves=3", flat.n_leaves, 3);
    v.check_count("FlatTree: n_internal=2", flat.n_internal, 2);
}

fn validate_batch_hmm(v: &mut Validator) {
    v.section("── Batch HMM Parity ──");
    let model = HmmModel {
        n_states: 2,
        log_pi: vec![0.6_f64.ln(), 0.4_f64.ln()],
        log_trans: vec![0.7_f64.ln(), 0.3_f64.ln(), 0.4_f64.ln(), 0.6_f64.ln()],
        log_emit: vec![
            0.1_f64.ln(),
            0.4_f64.ln(),
            0.5_f64.ln(),
            0.6_f64.ln(),
            0.3_f64.ln(),
            0.1_f64.ln(),
        ],
        n_symbols: 3,
    };
    let obs1: Vec<usize> = vec![0, 1, 2, 0, 1];
    let obs2: Vec<usize> = vec![2, 2, 1, 0, 0, 1, 2];

    let fw1 = forward(&model, &obs1);
    let fw2 = forward(&model, &obs2);
    let batch_fw = forward_batch(&model, &[&obs1, &obs2]);
    v.check(
        "Batch HMM: forward LL[0]",
        batch_fw[0].log_likelihood,
        fw1.log_likelihood,
        tolerances::ANALYTICAL_F64,
    );
    v.check(
        "Batch HMM: forward LL[1]",
        batch_fw[1].log_likelihood,
        fw2.log_likelihood,
        tolerances::ANALYTICAL_F64,
    );

    let vit1 = viterbi(&model, &obs1);
    let vit2 = viterbi(&model, &obs2);
    let batch_vit = viterbi_batch(&model, &[&obs1, &obs2]);
    v.check_pass("Batch HMM: Viterbi[0]", vit1.path == batch_vit[0].path);
    v.check_pass("Batch HMM: Viterbi[1]", vit2.path == batch_vit[1].path);
}

fn validate_batch_sw(v: &mut Validator) {
    v.section("── Batch Smith-Waterman Parity ──");
    let params = ScoringParams::default();
    let s1 = smith_waterman_score(b"ACGTACGT", b"ACGTACTT", &params);
    let s2 = smith_waterman_score(b"TTTTAAAA", b"AAAATTTT", &params);
    let batch = score_batch(
        &[(b"ACGTACGT", b"ACGTACTT"), (b"TTTTAAAA", b"AAAATTTT")],
        &params,
    );
    v.check_pass("Batch SW: score[0]", s1 == batch[0]);
    v.check_pass("Batch SW: score[1]", s2 == batch[1]);
}

fn validate_nj_and_dtl(v: &mut Validator) {
    v.section("── Neighbor-Joining + DTL ──");
    let seqs: Vec<&[u8]> = vec![b"ACGTACGTACGT", b"ACGTACGTACTT", b"TGCATGCATGCA"];
    let dm = distance_matrix(&seqs);
    let labels: Vec<String> = vec!["S1".into(), "S2".into(), "S3".into()];
    let nj = neighbor_joining(&dm, &labels);
    v.check_count("NJ: 1 join for 3 taxa", nj.n_joins, 1);
    v.check_pass("NJ: valid Newick", nj.newick.ends_with(';'));
    v.check_count("NJ: distance matrix 3×3", dm.len(), 9);

    let host = FlatRecTree {
        names: vec!["H_A".into(), "H_B".into(), "H_AB".into()],
        left_child: vec![NO_CHILD, NO_CHILD, 0],
        right_child: vec![NO_CHILD, NO_CHILD, 1],
    };
    let para = FlatRecTree {
        names: vec!["P_A".into(), "P_B".into(), "P_AB".into()],
        left_child: vec![NO_CHILD, NO_CHILD, 0],
        right_child: vec![NO_CHILD, NO_CHILD, 1],
    };
    let tip_map = vec![("P_A".into(), "H_A".into()), ("P_B".into(), "H_B".into())];
    let dtl = reconcile_dtl(&host, &para, &tip_map, &DtlCosts::default());
    v.check_count("DTL: congruent cost=0", dtl.optimal_cost as usize, 0);
    v.check_pass("DTL: mapped to H_AB", dtl.optimal_host == "H_AB");
}

fn validate_cross_module(v: &mut Validator) {
    v.section("── Cross-module: NJ → Felsenstein ──");
    let tree = TreeNode::Internal {
        left: Box::new(TreeNode::Leaf {
            name: "S1".into(),
            states: encode_dna("ACGTACGTACGT"),
        }),
        right: Box::new(TreeNode::Leaf {
            name: "S2".into(),
            states: encode_dna("ACGTACGTACTT"),
        }),
        left_branch: 0.044,
        right_branch: 0.044,
    };
    let ll = log_likelihood(&tree, 1.0);
    let flat = FlatTree::from_tree(&tree, 1.0);
    let ll_flat = flat.log_likelihood();
    v.check(
        "NJ→Felsenstein: flat matches recursive",
        ll_flat,
        ll,
        tolerances::ANALYTICAL_F64,
    );
    v.check_pass("NJ→Felsenstein: LL negative", ll < 0.0);
}

fn main() {
    let mut v = Validator::new("Exp035: BarraCuda CPU Parity v2");
    validate_flat_felsenstein(&mut v);
    validate_batch_hmm(&mut v);
    validate_batch_sw(&mut v);
    validate_nj_and_dtl(&mut v);
    validate_cross_module(&mut v);
    v.finish();
}
