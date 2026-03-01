// SPDX-License-Identifier: AGPL-3.0-or-later
//! DADA2 types — ASV, parameters, statistics.

/// Abundance p-value threshold (DADA2 R package `OMEGA_A`).
const OMEGA_A: f64 = 1e-40;

/// Maximum outer iterations (DADA2 R package `MAX_CONSIST`).
const MAX_DADA_ITERS: usize = 10;

/// Maximum error model refinement rounds (DADA2 R package `learnErrors`).
const MAX_ERR_ITERS: usize = 6;

/// An Amplicon Sequence Variant — the output of denoising.
#[derive(Debug, Clone)]
pub struct Asv {
    /// The denoised sequence (uppercase ACGT).
    pub sequence: Vec<u8>,
    /// Total abundance (sum of all sequences assigned to this ASV).
    pub abundance: usize,
    /// Number of unique sequences merged into this ASV.
    pub n_members: usize,
}

/// Parameters for DADA2 denoising.
#[derive(Debug, Clone)]
pub struct Dada2Params {
    /// Abundance p-value threshold. Sequences with `p < omega_a` are promoted
    /// to new ASVs instead of being absorbed. Default: 1e-40.
    pub omega_a: f64,
    /// Maximum rounds of the partition–refine loop. Default: 10.
    pub max_iterations: usize,
    /// Maximum rounds of error model refinement per partition step. Default: 6.
    pub max_err_iterations: usize,
    /// Minimum abundance for a unique sequence to be considered. Default: 1.
    pub min_abundance: usize,
}

/// Manual impl intentional: all fields use non-zero defaults (DADA2 R package).
impl Default for Dada2Params {
    fn default() -> Self {
        Self {
            omega_a: OMEGA_A,
            max_iterations: MAX_DADA_ITERS,
            max_err_iterations: MAX_ERR_ITERS,
            min_abundance: 1,
        }
    }
}

/// Statistics from a denoising run.
#[derive(Debug, Clone)]
pub struct Dada2Stats {
    /// Number of input unique sequences.
    pub input_uniques: usize,
    /// Total input reads (sum of abundances).
    pub input_reads: usize,
    /// Number of ASVs produced.
    pub output_asvs: usize,
    /// Total output reads (sum of ASV abundances).
    pub output_reads: usize,
    /// Number of partition–refine iterations performed.
    pub iterations: usize,
}
