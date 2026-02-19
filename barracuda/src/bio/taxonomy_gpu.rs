// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated taxonomy classification via compact GEMM dispatch.
//!
//! Replaces N×T sequential scoring loops with a single matrix multiply:
//!
//!   scores = Q_compact × T_compact
//!
//! Only k-mers actually present in query sequences are included in the
//! matrices.  For 5 queries with ~250 unique k-mers each, the active
//! set is ~1,000 k-mers out of 65,536 possible (k=8), cutting GPU
//! transfer from ~587 MB to ~13 MB per dispatch — a 45× reduction.
//!
//! Bootstrap confidence is computed by stacking all subsets into Q,
//! so the full classification + bootstrap execute as a single GEMM.

use crate::bio::taxonomy::{
    extract_kmers, ClassifyParams, Classification, NaiveBayesClassifier, TaxRank,
};
use crate::error::{Error, Result};
use crate::gpu::GpuF64;
use barracuda::ops::linalg::gemm_f64::GemmF64;
use std::collections::HashSet;

/// Batch-classify multiple query sequences on GPU via compact GEMM.
///
/// Produces identical classifications to calling
/// [`NaiveBayesClassifier::classify`] on each sequence — the scoring
/// math is the same matrix multiply, just with zero-contributing
/// columns removed.
///
/// # Errors
///
/// Returns [`Error::Gpu`] if the device lacks `SHADER_F64` or GEMM dispatch fails.
pub fn classify_batch_gpu(
    gpu: &GpuF64,
    classifier: &NaiveBayesClassifier,
    sequences: &[&[u8]],
    params: &ClassifyParams,
) -> Result<Vec<Classification>> {
    if !gpu.has_f64 {
        return Err(Error::Gpu("SHADER_F64 required for taxonomy GPU".into()));
    }

    if sequences.is_empty() || classifier.n_taxa() == 0 {
        return Ok(vec![]);
    }

    let n_queries = sequences.len();
    let n_taxa = classifier.n_taxa();
    let ks = classifier.kmer_space();

    if n_queries * n_taxa < 100 {
        return Ok(sequences
            .iter()
            .map(|seq| classifier.classify(seq, params))
            .collect());
    }

    // ── Extract k-mers and find the active k-mer set ────────────────────────
    let k = (ks as f64).log(4.0).round() as usize;
    let query_kmer_lists: Vec<Vec<u64>> = sequences
        .iter()
        .map(|seq| extract_kmers(seq, k))
        .collect();

    // Active set: union of all k-mers across all queries (including duplicates
    // that bootstrap might sample). Since bootstrap samples from each query's
    // k-mer list, the active set is exactly the union of per-query unique k-mers.
    let mut active_set = HashSet::new();
    for kmers in &query_kmer_lists {
        for &kmer in kmers {
            active_set.insert(kmer as usize);
        }
    }
    let mut active_indices: Vec<usize> = active_set.into_iter().collect();
    active_indices.sort_unstable();
    let n_active = active_indices.len();

    // Reverse map: kmer_value → compact column index
    let mut kmer_to_col = vec![0_usize; ks];
    for (col, &ki) in active_indices.iter().enumerate() {
        kmer_to_col[ki] = col;
    }

    let n_boot = params.bootstrap_n;
    let rows_per_query = 1 + n_boot;
    let total_rows = n_queries * rows_per_query;

    // ── Build compact Q matrix: total_rows × n_active ───────────────────────
    let mut q_compact = vec![0.0_f64; total_rows * n_active];

    for (qi, kmers) in query_kmer_lists.iter().enumerate() {
        let base_row = qi * rows_per_query;

        // Row 0: full k-mer counts (compact)
        let row_start = base_row * n_active;
        for &kmer in kmers {
            q_compact[row_start + kmer_to_col[kmer as usize]] += 1.0;
        }

        // Rows 1..=n_boot: bootstrap subsets
        let n_sample = (kmers.len() * 2 / 3).max(1);
        let mut seed: u64 = 42;
        for bi in 0..n_boot {
            let boot_start = (base_row + 1 + bi) * n_active;
            for _ in 0..n_sample {
                seed = seed
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                let idx = (seed >> 33) as usize % kmers.len();
                q_compact[boot_start + kmer_to_col[kmers[idx] as usize]] += 1.0;
            }
        }
    }

    // ── Build compact T^T: n_active × n_taxa ────────────────────────────────
    // dense_log_probs layout: n_taxa × ks (row-major)
    // T^T_compact[col][taxon] = dense_log_probs[taxon * ks + active_indices[col]]
    let dense = classifier.dense_log_probs();
    let mut t_compact = vec![0.0_f64; n_active * n_taxa];
    for (col, &ki) in active_indices.iter().enumerate() {
        for ti in 0..n_taxa {
            t_compact[col * n_taxa + ti] = dense[ti * ks + ki];
        }
    }

    // ── GEMM dispatch: scores = Q_compact × T_compact ───────────────────────
    // A: total_rows × n_active, B: n_active × n_taxa, C: total_rows × n_taxa
    let scores = GemmF64::execute(
        gpu.to_wgpu_device(),
        &q_compact,
        &t_compact,
        total_rows,
        n_active,
        n_taxa,
        1,
    )
    .map_err(|e| Error::Gpu(format!("taxonomy GEMM: {e}")))?;

    // ── Post-process: add log-priors, argmax, bootstrap confidence ──────────
    let log_priors = classifier.log_priors();
    let taxon_labels = classifier.taxon_labels();
    let mut results = Vec::with_capacity(n_queries);

    for qi in 0..n_queries {
        let base_row = qi * rows_per_query;

        let full_row = &scores[base_row * n_taxa..(base_row + 1) * n_taxa];
        let best_taxon = argmax_with_priors(full_row, log_priors);

        let full_lineage = &taxon_labels[best_taxon];
        let n_ranks = TaxRank::all().len();
        let mut rank_votes = vec![0_usize; n_ranks];

        for bi in 0..n_boot {
            let boot_row = base_row + 1 + bi;
            let boot_scores = &scores[boot_row * n_taxa..(boot_row + 1) * n_taxa];
            let boot_taxon = argmax_with_priors(boot_scores, log_priors);
            let boot_lineage = &taxon_labels[boot_taxon];

            for (ri, rank) in TaxRank::all().iter().enumerate() {
                let full_at = full_lineage.at_rank(*rank);
                let boot_at = boot_lineage.at_rank(*rank);
                if full_at.is_some() && full_at == boot_at {
                    rank_votes[ri] += 1;
                }
            }
        }

        let confidence: Vec<f64> = rank_votes
            .iter()
            .map(|&v| v as f64 / n_boot as f64)
            .collect();

        results.push(Classification {
            lineage: taxon_labels[best_taxon].clone(),
            confidence,
            taxon_idx: best_taxon,
        });
    }

    Ok(results)
}

fn argmax_with_priors(scores: &[f64], log_priors: &[f64]) -> usize {
    let mut best = f64::NEG_INFINITY;
    let mut best_idx = 0;
    for (i, (&s, &lp)) in scores.iter().zip(log_priors.iter()).enumerate() {
        let total = s + lp;
        if total > best {
            best = total;
            best_idx = i;
        }
    }
    best_idx
}
