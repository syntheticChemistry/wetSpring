// SPDX-License-Identifier: AGPL-3.0-or-later
//! DADA2 core denoising algorithm — error model, partitioning, EM.

use crate::bio::derep::UniqueSequence;
use crate::special::regularized_gamma_lower;

use super::types::{Asv, Dada2Params, Dada2Stats};

/// Number of canonical nucleotide bases (A, C, G, T).
pub const NUM_BASES: usize = 4;

/// Maximum Phred quality score tracked in the error matrix.
///
/// Illumina instruments report Q0–Q41; Q42 covers edge cases.
/// Matches DADA2 R package `MAX_QUAL` (Callahan et al. 2016, §Methods).
pub const MAX_QUAL: usize = 42;

/// Maximum iterations for error-rate self-consistency loop.
///
/// DADA2 R package `learnErrors()` default convergence rounds.
pub const MAX_ERR_ITERS: usize = 6;

/// Floor for per-substitution error rate.
///
/// Prevents log(0) in Poisson abundance calculations. DADA2 R package
/// `MIN_ERR_RATE` (`dada.cpp:24`).
pub const MIN_ERR: f64 = 1e-7;

/// Ceiling for per-substitution error rate.
///
/// Caps unreliable estimates from low-coverage substitution classes.
/// DADA2 R package `MAX_ERR_RATE` (`dada.cpp:25`).
pub const MAX_ERR: f64 = 0.25;

/// 4×4 error matrix indexed by quality score.
/// `err[from][to][qual]` = P(observing `to` when truth is `from` at quality `qual`).
pub type ErrorModel = [[[f64; MAX_QUAL]; NUM_BASES]; NUM_BASES];

/// Initial sequence partitioning: all sequences assigned to center 0.
fn init_partition(n: usize) -> (Vec<usize>, Vec<usize>) {
    let partition = vec![0; n];
    let centers = vec![0];
    (partition, centers)
}

/// Single EM iteration: E-step (assign), M-step (refine error model), split (find new centers).
/// Returns `true` if centers changed (iteration should continue).
fn em_step(
    seqs: &[&UniqueSequence],
    partition: &mut [usize],
    centers: &mut Vec<usize>,
    err: &mut ErrorModel,
    params: &Dada2Params,
    last_n_centers: usize,
) -> bool {
    assign_to_centers(seqs, centers, err, partition);

    for _ in 0..params.max_err_iterations {
        let new_err = estimate_error_model(seqs, partition, centers);
        let converged = err_model_converged(err, &new_err);
        *err = new_err;
        if converged {
            break;
        }
    }

    let new_centers = find_new_centers(seqs, partition, centers, err, params.omega_a);
    for &c in &new_centers {
        if !centers.contains(&c) {
            centers.push(c);
        }
    }
    centers.sort_unstable();

    centers.len() != last_n_centers
}

/// Initialize error model from Phred quality scores (no prior data).
#[allow(clippy::needless_range_loop)] // 3D array requires indexing by from/to/q
pub fn init_error_model() -> ErrorModel {
    let mut err = [[[0.0_f64; MAX_QUAL]; NUM_BASES]; NUM_BASES];
    for q in 0..MAX_QUAL {
        #[allow(clippy::cast_precision_loss)] // q is 0..42, exact
        let p_err = (10.0_f64).powf(-(q as f64) / 10.0).clamp(MIN_ERR, MAX_ERR);
        for from in 0..NUM_BASES {
            for to in 0..NUM_BASES {
                if from == to {
                    err[from][to][q] = 1.0 - p_err;
                } else {
                    err[from][to][q] = p_err / 3.0;
                }
            }
        }
    }
    err
}

#[allow(clippy::match_same_arms)]
/// Map nucleotide to error-matrix index: A=0, C=1, G=2, T=3.
///
/// Ambiguous/unknown bases (N, IUPAC degenerate) map to 0 (A).
/// This matches the DADA2 R package behavior where non-ACGT bases
/// are treated as A for error-rate indexing — rare in quality-filtered
/// reads and has negligible effect on the error model.
pub const fn base_to_idx(b: u8) -> usize {
    match b {
        b'A' | b'a' => 0,
        b'C' | b'c' => 1,
        b'G' | b'g' => 2,
        b'T' | b't' => 3,
        _ => 0,
    }
}

/// Compute log-probability that `seq` arose from `center` by errors.
fn log_p_error(seq: &UniqueSequence, center: &UniqueSequence, err: &ErrorModel) -> f64 {
    let len = seq.sequence.len().min(center.sequence.len());
    let mut log_p = 0.0_f64;
    for i in 0..len {
        let from = base_to_idx(center.sequence[i]);
        let to = base_to_idx(seq.sequence[i]);
        let q = seq
            .representative_quality
            .get(i)
            .map_or(0, |&v| v.saturating_sub(33) as usize)
            .min(MAX_QUAL - 1);
        let p = err[from][to][q].max(MIN_ERR);
        log_p += p.ln();
    }
    log_p
}

/// Assign each sequence to its most likely center.
fn assign_to_centers(
    seqs: &[&UniqueSequence],
    centers: &[usize],
    err: &ErrorModel,
    partition: &mut [usize],
) {
    for (i, seq) in seqs.iter().enumerate() {
        let mut best_center = centers[0];
        let mut best_log_p = f64::NEG_INFINITY;

        for &c in centers {
            let lp = log_p_error(seq, seqs[c], err);
            if lp > best_log_p {
                best_log_p = lp;
                best_center = c;
            }
        }
        partition[i] = best_center;
    }
}

/// Re-estimate error model from observed substitution patterns.
#[allow(clippy::cast_precision_loss)]
pub fn estimate_error_model(
    seqs: &[&UniqueSequence],
    partition: &[usize],
    _centers: &[usize],
) -> ErrorModel {
    // Count observed transitions
    let mut counts = [[[0.0_f64; MAX_QUAL]; NUM_BASES]; NUM_BASES];
    let mut totals = [[0.0_f64; MAX_QUAL]; NUM_BASES];

    for (i, seq) in seqs.iter().enumerate() {
        let center_idx = partition[i];
        let center = seqs[center_idx];
        let len = seq.sequence.len().min(center.sequence.len());
        let weight = seq.abundance as f64;

        for pos in 0..len {
            let from = base_to_idx(center.sequence[pos]);
            let to = base_to_idx(seq.sequence[pos]);
            let q = seq
                .representative_quality
                .get(pos)
                .map_or(0, |&v| v.saturating_sub(33) as usize)
                .min(MAX_QUAL - 1);

            counts[from][to][q] += weight;
            totals[from][q] += weight;
        }
    }

    let mut err = init_error_model();
    #[allow(clippy::needless_range_loop)] // 3D array requires indexing by from/to/q
    for from in 0..NUM_BASES {
        for q in 0..MAX_QUAL {
            if totals[from][q] > 0.0 {
                for to in 0..NUM_BASES {
                    let rate = counts[from][to][q] / totals[from][q];
                    err[from][to][q] = rate.clamp(MIN_ERR, 1.0 - MIN_ERR);
                }
            }
        }
    }

    // Ensure rows sum to ~1
    #[allow(clippy::needless_range_loop)] // 3D array requires indexing by from/to/q
    for from in 0..NUM_BASES {
        for q in 0..MAX_QUAL {
            let sum: f64 = (0..NUM_BASES).map(|to| err[from][to][q]).sum();
            if sum > 0.0 {
                for to in 0..NUM_BASES {
                    err[from][to][q] /= sum;
                }
            }
        }
    }

    err
}

#[allow(clippy::needless_range_loop)]
pub fn err_model_converged(old: &ErrorModel, new: &ErrorModel) -> bool {
    let mut max_diff = 0.0_f64;
    for from in 0..NUM_BASES {
        for to in 0..NUM_BASES {
            for q in 0..MAX_QUAL {
                let diff = (old[from][to][q] - new[from][to][q]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }
    }
    max_diff < crate::tolerances::DADA2_ERR_CONVERGENCE
}

/// Find sequences that should become new centers (abundance p-value test).
///
/// For each non-center sequence, computes the expected abundance under the
/// error model and tests whether the observed abundance is significantly
/// higher using a Poisson CDF approximation.
#[allow(clippy::cast_precision_loss)]
fn find_new_centers(
    seqs: &[&UniqueSequence],
    partition: &[usize],
    centers: &[usize],
    err: &ErrorModel,
    omega_a: f64,
) -> Vec<usize> {
    let mut new_centers = Vec::new();

    for (i, seq) in seqs.iter().enumerate() {
        if centers.contains(&i) {
            continue;
        }

        let center_idx = partition[i];
        let center = seqs[center_idx];

        // Expected number of reads that would look like seq if they came from center
        let log_p = log_p_error(seq, center, err);
        let lambda = (center.abundance as f64) * log_p.exp();

        if lambda <= 0.0 {
            new_centers.push(i);
            continue;
        }

        let observed = seq.abundance;

        // Poisson survival function: P(X >= observed) where X ~ Poisson(lambda)
        let p_value = poisson_pvalue(observed, lambda);

        if p_value < omega_a {
            new_centers.push(i);
        }
    }

    new_centers
}

/// Upper-tail Poisson p-value: P(X >= k) for X ~ Poisson(lambda).
/// Uses the identity: P(X >= k | Poisson(λ)) = P(k, λ) where P is the
/// regularized lower incomplete gamma function.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn poisson_pvalue(k: usize, lambda: f64) -> f64 {
    if lambda <= 0.0 || k == 0 {
        return 1.0;
    }
    regularized_gamma_lower(k as f64, lambda)
}

/// Build ASV structs from final partition assignments.
fn build_asvs(seqs: &[&UniqueSequence], partition: &[usize], centers: &[usize]) -> Vec<Asv> {
    let mut asvs: Vec<Asv> = centers
        .iter()
        .map(|&c| Asv {
            sequence: seqs[c].sequence.clone(), // ownership transfer: borrowed input requires clone
            abundance: 0,
            n_members: 0,
        })
        .collect();

    for (i, &center_idx) in partition.iter().enumerate() {
        if let Some(asv_pos) = centers.iter().position(|&c| c == center_idx) {
            asvs[asv_pos].abundance += seqs[i].abundance;
            asvs[asv_pos].n_members += 1;
        }
    }

    asvs
}

/// Denoise a set of dereplicated sequences into ASVs.
///
/// This is the main entry point. Takes abundance-sorted `UniqueSequence`s
/// (from `bio::derep::dereplicate`) and returns ASVs.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn denoise(seqs: &[UniqueSequence], params: &Dada2Params) -> (Vec<Asv>, Dada2Stats) {
    let seqs: Vec<&UniqueSequence> = seqs
        .iter()
        .filter(|s| s.abundance >= params.min_abundance && !s.sequence.is_empty())
        .collect();

    let input_uniques = seqs.len();
    let input_reads: usize = seqs.iter().map(|s| s.abundance).sum();

    if seqs.is_empty() {
        return (
            vec![],
            Dada2Stats {
                input_uniques,
                input_reads,
                output_asvs: 0,
                output_reads: 0,
                iterations: 0,
            },
        );
    }

    let mut err = init_error_model();
    let (mut partition, mut centers) = init_partition(seqs.len());

    let mut last_n_centers = 0;
    let mut iters = 0;

    for _ in 0..params.max_iterations {
        iters += 1;
        let centers_changed = em_step(
            &seqs,
            &mut partition,
            &mut centers,
            &mut err,
            params,
            last_n_centers,
        );
        last_n_centers = centers.len();
        if !centers_changed {
            break;
        }
    }

    assign_to_centers(&seqs, &centers, &err, &mut partition);
    let mut asvs = build_asvs(&seqs, &partition, &centers);
    asvs.sort_by(|a, b| b.abundance.cmp(&a.abundance));

    let output_reads: usize = asvs.iter().map(|a| a.abundance).sum();
    let output_asvs = asvs.len();

    (
        asvs,
        Dada2Stats {
            input_uniques,
            input_reads,
            output_asvs,
            output_reads,
            iterations: iters,
        },
    )
}
