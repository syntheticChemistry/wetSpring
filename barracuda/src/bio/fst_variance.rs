// SPDX-License-Identifier: AGPL-3.0-or-later
//! FST (Fixation Index) variance decomposition — Weir-Cockerham estimator.
//!
//! Re-exports from `barracuda::ops::bio::fst_variance` for discoverability.
//! CPU-only: computes Wright's F-statistics (FST, FIS, FIT) from allele
//! frequencies and population sample sizes.
//!
//! # References
//!
//! - Weir & Cockerham (1984) Evolution 38:1358-1370

pub use barracuda::ops::bio::fst_variance::{FstResult, fst_variance_decomposition};

#[cfg(test)]
#[cfg(feature = "gpu")]
mod tests {
    use super::*;

    #[test]
    fn api_surface_compiles() {
        fn _assert_result(_: &FstResult) {}
        let _ = fst_variance_decomposition;
    }

    #[test]
    fn cpu_signature_check() {
        let allele_freqs = [0.8, 0.6, 0.3];
        let sample_sizes = [100usize, 100, 100];
        let result = fst_variance_decomposition(&allele_freqs, &sample_sizes);
        assert!(
            result.is_ok(),
            "fst_variance_decomposition should succeed with valid input"
        );
    }
}
