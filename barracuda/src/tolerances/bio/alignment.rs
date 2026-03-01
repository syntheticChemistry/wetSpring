// SPDX-License-Identifier: AGPL-3.0-or-later
//! Alignment, ANI, spectral, and HMM tolerances.

/// Cross-species ANI comparison against Python baseline.
///
/// k-mer sampling and gap handling can differ at boundary codons.
/// 1e-4 covers implementation-level variation in ANI estimation.
/// Calibrated: Exp055 (population genomics ANI).
/// Script: `anderson2017_population_genomics.py`. Commit `48fb787`.
pub const ANI_CROSS_SPECIES: f64 = 1e-4;

/// Evolutionary distance estimation (Jukes-Cantor, alignment-derived).
///
/// Distance matrix computation accumulates log/exp corrections over
/// pairwise alignment columns; ~1e-3 covers implementation variation.
/// Calibrated: Exp033 (neighbor joining), Exp034 (DTL reconciliation).
/// Scripts: `liu2009_neighbor_joining.py`, `zheng2023_dtl_reconciliation.py`.
/// Commit `48fb787`.
pub const EVOLUTIONARY_DISTANCE: f64 = 1e-3;

/// Spectral cosine similarity rounding (m/z-shifted comparisons).
///
/// Greedy peak matching with floating-point intensity weighting
/// can differ by ~1e-3 at alignment boundaries.
/// Calibrated: Exp005/006 (mzML spectral matching), Exp042 (`MassBank`).
/// Scripts: `validate_track2.py`, `massbank_spectral_baseline.py`.
/// Commit `48fb787`.
pub const SPECTRAL_COSINE: f64 = 1e-3;

/// HMM forward log-likelihood Python parity.
///
/// HMM forward algorithm accumulates log-sum-exp over T×N states.
/// Python `hmmlearn` and Rust differ by ~1e-7 due to log/exp ordering.
/// 1e-6 covers observed variance for small models (2 states, 5 obs).
/// Validated: `scripts/liu2014_hmm_baseline.py`, commit `e4358c5`.
pub const HMM_FORWARD_PARITY: f64 = 1e-6;

/// HMM mathematical invariant slack: Viterbi log-prob ≤ Forward LL.
///
/// The most-likely path probability (Viterbi) must be ≤ the total
/// probability (Forward) by definition. Numerical noise from
/// log-sum-exp accumulation can cause sub-ULP violations; 1e-10
/// accommodates this without masking real bugs.
/// Validated: Exp026 (Liu 2014), `scripts/liu2014_hmm_baseline.py`,
/// commit `e4358c5`.
pub const HMM_INVARIANT_SLACK: f64 = 1e-10;
