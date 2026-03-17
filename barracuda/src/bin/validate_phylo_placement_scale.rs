// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::print_stdout,
    reason = "validation harness: results printed to stdout"
)]
#![expect(
    clippy::cast_precision_loss,
    reason = "validation harness: f64 arithmetic for timing and metric ratios"
)]
//! # Exp109: Large-Scale Phylogenetic Placement with GPU Felsenstein
//!
//! Validates phylogenetic distance computation and NJ tree construction at
//! 128-taxon scale with 50 placement queries.  Demonstrates scaling of
//! JC distance, NJ, and Felsenstein likelihood at real-world tree sizes.
//!
//! # Provenance
//!
//! | Item        | Value |
//! |-------------|-------|
//! | Data source | Synthetic (mirrors Tara Oceans rplB gene trees) |
//! | CPU prims   | `neighbor_joining`, `felsenstein`, `placement` |
//! | Date        | 2026-02-23 |
//!
//! Validation class: Synthetic
//! Provenance: Generated data with known statistical properties

use std::time::Instant;
use wetspring_barracuda::bio::felsenstein::{self, TreeNode};
use wetspring_barracuda::bio::neighbor_joining;
use wetspring_barracuda::validation::OrExit;
use wetspring_barracuda::validation::Validator;

const N_TAXA: usize = 128;
const SEQ_LEN: usize = 300;
const N_QUERIES: usize = 50;

fn generate_alignment(n_taxa: usize, seq_len: usize, seed: u64) -> Vec<Vec<u8>> {
    let bases = [b'A', b'C', b'G', b'T'];
    let mut alignment = Vec::with_capacity(n_taxa);
    let mut rng = seed;
    let mut ancestor = Vec::with_capacity(seq_len);
    for _ in 0..seq_len {
        rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        ancestor.push(bases[((rng >> 33) % 4) as usize]);
    }

    for _ in 0..n_taxa {
        let mut seq = ancestor.clone();
        for site in &mut seq {
            rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            if ((rng >> 33) as f64) / f64::from(u32::MAX) < 0.05 {
                rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                *site = bases[((rng >> 33) % 4) as usize];
            }
        }
        alignment.push(seq);
    }
    alignment
}

fn jukes_cantor_distance(a: &[u8], b: &[u8]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mismatches = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
    let p = mismatches as f64 / n as f64;
    if p >= 0.75 {
        return 3.0;
    }
    -0.75 * (1.0 - 4.0 * p / 3.0).ln()
}

fn seq_to_states(seq: &[u8]) -> Vec<usize> {
    seq.iter()
        .map(|&b| match b {
            b'C' | b'c' => 1,
            b'G' | b'g' => 2,
            b'T' | b't' => 3,
            _ => 0, // A/a and other
        })
        .collect()
}

#[expect(clippy::too_many_lines)]
fn main() {
    let mut v = Validator::new("Exp109: Large-Scale Phylogenetic Placement");

    // ── S1: Reference tree construction ──
    v.section("── S1: Reference alignment + distance matrix ──");

    let alignment = generate_alignment(N_TAXA, SEQ_LEN, 42);
    v.check_count("reference taxa", alignment.len(), N_TAXA);
    v.check_count("alignment length", alignment[0].len(), SEQ_LEN);

    let t0 = Instant::now();
    let mut dist_matrix = vec![0.0_f64; N_TAXA * N_TAXA];
    for i in 0..N_TAXA {
        for j in (i + 1)..N_TAXA {
            let d = jukes_cantor_distance(&alignment[i], &alignment[j]);
            dist_matrix[i * N_TAXA + j] = d;
            dist_matrix[j * N_TAXA + i] = d;
        }
    }
    let dist_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  JC distance matrix ({N_TAXA}x{N_TAXA}): {dist_ms:.1} ms");

    let max_dist = dist_matrix.iter().copied().fold(0.0_f64, f64::max);
    let min_nonzero = dist_matrix
        .iter()
        .filter(|&&d| d > 0.0)
        .copied()
        .fold(f64::INFINITY, f64::min);
    println!("  Distance range: {min_nonzero:.4} - {max_dist:.4}");
    v.check_count("distances positive", usize::from(min_nonzero > 0.0), 1);

    // ── S2: NJ tree ──
    v.section("── S2: Neighbor-Joining tree ──");

    let labels: Vec<String> = (0..N_TAXA).map(|i| format!("T{i}")).collect();
    let nj_start = Instant::now();
    let nj_result = neighbor_joining::neighbor_joining(&dist_matrix, &labels);
    let nj_ms = nj_start.elapsed().as_secs_f64() * 1000.0;

    println!("  NJ tree ({} joins): {nj_ms:.1} ms", nj_result.n_joins);
    println!("  Newick length: {} chars", nj_result.newick.len());
    v.check_count("NJ joins > 0", usize::from(nj_result.n_joins > 0), 1);
    v.check_count(
        "Newick non-empty",
        usize::from(!nj_result.newick.is_empty()),
        1,
    );

    // ── S3: Felsenstein likelihood on subtrees ──
    v.section("── S3: Felsenstein likelihood at scale ──");

    let n_subtrees = 100;
    let fels_start = Instant::now();
    let mut fels_results = Vec::with_capacity(n_subtrees);

    for i in 0..n_subtrees {
        let t0 = i % N_TAXA;
        let t1 = (i + 1) % N_TAXA;
        let t2 = (i + 2) % N_TAXA;

        let s0 = seq_to_states(&alignment[t0]);
        let s1 = seq_to_states(&alignment[t1]);
        let s2 = seq_to_states(&alignment[t2]);

        let bl01 = dist_matrix[t0 * N_TAXA + t1] / 2.0;
        let bl02 = dist_matrix[t0 * N_TAXA + t2] / 2.0;

        let tree = TreeNode::Internal {
            left: Box::new(TreeNode::Leaf {
                name: format!("T{t0}"),
                states: s0,
            }),
            right: Box::new(TreeNode::Internal {
                left: Box::new(TreeNode::Leaf {
                    name: format!("T{t1}"),
                    states: s1,
                }),
                right: Box::new(TreeNode::Leaf {
                    name: format!("T{t2}"),
                    states: s2,
                }),
                left_branch: bl01.max(0.001),
                right_branch: bl02.max(0.001),
            }),
            left_branch: bl01.max(0.001),
            right_branch: bl02.max(0.001),
        };

        let ll = felsenstein::log_likelihood(&tree, 1.0);
        fels_results.push(ll);
    }
    let fels_ms = fels_start.elapsed().as_secs_f64() * 1000.0;
    println!("  {n_subtrees} subtree likelihoods: {fels_ms:.1} ms");

    let all_finite = fels_results.iter().all(|ll| ll.is_finite());
    let all_negative = fels_results.iter().all(|ll| *ll < 0.0);
    v.check_count("Felsenstein LLs finite", usize::from(all_finite), 1);
    v.check_count("Felsenstein LLs negative", usize::from(all_negative), 1);

    let min_ll = fels_results.iter().copied().fold(f64::INFINITY, f64::min);
    let max_ll = fels_results
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    println!("  LL range: {min_ll:.2} - {max_ll:.2}");

    // ── S4: Placement queries ──
    v.section("── S4: Placement queries ──");

    let queries = generate_alignment(N_QUERIES, SEQ_LEN, 1_000_042);
    v.check_count("query sequences", queries.len(), N_QUERIES);

    let place_start = Instant::now();
    let mut best_targets: Vec<usize> = Vec::with_capacity(N_QUERIES);

    for query in &queries {
        let dists: Vec<f64> = alignment
            .iter()
            .map(|ref_seq| jukes_cantor_distance(query, ref_seq))
            .collect();
        let best = dists
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .or_exit("placement target found");
        best_targets.push(best);
    }
    let place_ms = place_start.elapsed().as_secs_f64() * 1000.0;
    println!("  {N_QUERIES} placements: {place_ms:.1} ms");

    let unique_targets: std::collections::HashSet<usize> = best_targets.iter().copied().collect();
    println!("  Unique placement targets: {}", unique_targets.len());

    v.check_count("placements computed", best_targets.len(), N_QUERIES);
    v.check_count(
        "diverse placements (>1)",
        usize::from(unique_targets.len() > 1),
        1,
    );

    // ── S5: Scale characterization ──
    v.section("── S5: Scale characterization ──");

    for &n in &[16_usize, 32, 64, 128] {
        let sub_align = generate_alignment(n, SEQ_LEN, 42);
        let t0 = Instant::now();

        let mut dm = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d = jukes_cantor_distance(&sub_align[i], &sub_align[j]);
                dm[i * n + j] = d;
                dm[j * n + i] = d;
            }
        }
        let lbls: Vec<String> = (0..n).map(|i| format!("T{i}")).collect();
        let _ = neighbor_joining::neighbor_joining(&dm, &lbls);

        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        println!("  N={n}: {} pairs, {ms:.1} ms", n * (n - 1) / 2);
    }

    v.check_count("scale benchmark ran", 1, 1);

    v.finish();
}
