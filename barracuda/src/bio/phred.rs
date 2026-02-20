// SPDX-License-Identifier: AGPL-3.0-or-later
//! Phred quality score conversion utilities.
//!
//! Centralizes the `Q → P_error` and `P_error → Q` formulas used
//! by [`super::merge_pairs`], [`super::dada2`], and
//! [`super::quality`].

/// Convert a Phred quality score to an error probability.
///
/// `P = 10^(-Q / 10)`
#[must_use]
#[inline]
pub fn phred_to_error_prob(q: f64) -> f64 {
    10_f64.powf(-q / 10.0)
}

/// Convert an error probability to a Phred quality score.
///
/// `Q = -10 * log10(P)`.  Returns 0.0 for `p >= 1.0` and caps at
/// 41.0 (Illumina maximum) for `p <= 0.0`.
#[must_use]
#[inline]
pub fn error_prob_to_phred(p: f64) -> f64 {
    if p <= 0.0 {
        41.0
    } else if p >= 1.0 {
        0.0
    } else {
        -10.0 * p.log10()
    }
}

/// Extract the numeric quality value from a raw Phred+offset byte.
#[must_use]
#[inline]
pub fn decode_qual(raw: u8, offset: u8) -> f64 {
    f64::from(raw.saturating_sub(offset))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_q30() {
        let q = 30.0;
        let p = phred_to_error_prob(q);
        let q2 = error_prob_to_phred(p);
        assert!((q - q2).abs() < 1e-12);
    }

    #[test]
    fn error_prob_boundaries() {
        assert_eq!(error_prob_to_phred(0.0), 41.0);
        assert_eq!(error_prob_to_phred(1.0), 0.0);
        assert_eq!(error_prob_to_phred(2.0), 0.0);
    }

    #[test]
    fn phred_q0_is_one() {
        assert!((phred_to_error_prob(0.0) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn decode_qual_saturates() {
        assert!((decode_qual(33, 33) - 0.0).abs() < f64::EPSILON);
        assert!((decode_qual(63, 33) - 30.0).abs() < f64::EPSILON);
        assert!((decode_qual(0, 33) - 0.0).abs() < f64::EPSILON); // saturating_sub
    }
}
