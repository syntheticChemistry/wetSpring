// SPDX-License-Identifier: AGPL-3.0-or-later
//! Synthetic test data generators shared across validation binaries.
//!
//! Eliminates duplicated `make_communities`, `make_abundances`, and
//! `make_sequences` helpers that appeared independently in 6+ binaries.

/// Synthetic community abundance matrix with shifted-sqrt gradient.
///
/// Each cell: `(sqrt(s * n_features + f + 1) + (s + 1) * 0.1).max(0.01)`
///
/// Used by: `validate_pure_gpu_streaming`, `validate_pure_gpu_pipeline`,
/// `benchmark_streaming_vs_roundtrip`.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn make_communities_shift(n_samples: usize, n_features: usize) -> Vec<Vec<f64>> {
    (0..n_samples)
        .map(|s| {
            (0..n_features)
                .map(|f| {
                    let base = ((s * n_features + f + 1) as f64).sqrt();
                    let shift = (s + 1) as f64 * 0.1;
                    (base + shift).max(0.01)
                })
                .collect()
        })
        .collect()
}

/// Synthetic community abundance matrix with gradient scaling.
///
/// Each cell: `(sqrt(s * n_features + f + 1) * (1.0 + gradient)).max(0.01)`
/// where `gradient = (s / n_samples) * 2.0`.
///
/// Used by: `validate_cross_substrate_pipeline`.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn make_communities_gradient(n_samples: usize, n_features: usize) -> Vec<Vec<f64>> {
    (0..n_samples)
        .map(|s| {
            let gradient = (s as f64 / n_samples as f64) * 2.0;
            (0..n_features)
                .map(|f| {
                    let base = ((s * n_features + f + 1) as f64).sqrt();
                    (base * (1.0 + gradient)).max(0.01)
                })
                .collect()
        })
        .collect()
}

/// Synthetic abundance vector for diversity dispatch tests.
///
/// Each element: `(i + 1) * 1.5 + 0.5`.
///
/// Used by: `validate_dispatch_overhead_proof`.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn make_abundances(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i + 1) as f64).mul_add(1.5, 0.5)).collect()
}

/// Synthetic DNA sequences for taxonomy pipeline tests.
///
/// Generates `n` sequences of length `len` using a deterministic ACGT rotation.
///
/// Used by: `validate_pure_gpu_streaming`, `benchmark_streaming_vs_roundtrip`.
#[must_use]
pub fn make_sequences(n: usize, len: usize) -> Vec<Vec<u8>> {
    let bases = [b'A', b'C', b'G', b'T'];
    (0..n)
        .map(|i| (0..len).map(|j| bases[(i * len + j) % 4]).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn communities_shift_dimensions() {
        let c = make_communities_shift(5, 10);
        assert_eq!(c.len(), 5);
        assert!(c.iter().all(|row| row.len() == 10));
        assert!(c.iter().flatten().all(|&v| v >= 0.01));
    }

    #[test]
    fn communities_gradient_dimensions() {
        let c = make_communities_gradient(5, 10);
        assert_eq!(c.len(), 5);
        assert!(c.iter().all(|row| row.len() == 10));
        assert!(c.iter().flatten().all(|&v| v >= 0.01));
    }

    #[test]
    fn abundances_monotonic() {
        let a = make_abundances(10);
        assert_eq!(a.len(), 10);
        for i in 1..a.len() {
            assert!(a[i] > a[i - 1]);
        }
    }

    #[test]
    fn sequences_valid_bases() {
        let seqs = make_sequences(8, 80);
        assert_eq!(seqs.len(), 8);
        assert!(seqs.iter().all(|s| s.len() == 80));
        assert!(
            seqs.iter()
                .flatten()
                .all(|&b| b == b'A' || b == b'C' || b == b'G' || b == b'T')
        );
    }
}
