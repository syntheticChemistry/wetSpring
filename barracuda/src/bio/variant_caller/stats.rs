// SPDX-License-Identifier: AGPL-3.0-or-later
//! Statistical quality models for variant calling.
//!
//! Pure-math functions that compute variant quality scores, binomial
//! p-values, and supporting primitives (`log_gamma`, `log_add_exp`,
//! normal survival). No bio-type dependencies — all inputs are
//! primitive floats/integers.
//!
//! # Models
//!
//! - **Legacy**: flat error rate (0.001) log-likelihood ratio
//! - **Per-base quality (`_bq`)**: mean Phred of alt allele ↦ error rate
//! - **Binomial (WS-11)**: one-sided binomial test against quality-weighted
//!   null error rate, Phred-scaled

use crate::bio::pileup::PileupColumn;

/// Compute quality-weighted frequency for a base index at a pileup column.
///
/// Each base contributes `1 - 10^(-Q/10)` (probability of being correct)
/// instead of a flat 1.0. This down-weights low-quality bases that are
/// likely sequencing errors.
#[expect(
    clippy::cast_precision_loss,
    reason = "Precision: quality sums bounded by coverage"
)]
pub(super) fn quality_weighted_freq(col: &PileupColumn, base_idx: usize) -> f64 {
    let weights: Vec<f64> = (0..4)
        .map(|i| {
            let count = f64::from(col.base_counts[i]);
            if count == 0.0 {
                return 0.0;
            }
            let mean_q = if col.base_counts[i] > 0 {
                col.quality_sums[i] as f64 / count
            } else {
                0.0
            };
            let p_correct = 1.0 - 10.0_f64.powf(-mean_q / 10.0);
            count * p_correct
        })
        .collect();

    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    weights[base_idx] / total
}

/// Compute Phred-scaled variant quality using per-base quality information.
///
/// Uses the actual mean quality of the alternative allele's supporting reads
/// to compute the expected error rate, rather than assuming a flat Q30.
/// Q = sum over alt reads of (-10 * `log10(P_error_per_base)`).
#[expect(
    clippy::cast_precision_loss,
    reason = "Precision: quality sums bounded by coverage"
)]
pub(super) fn variant_quality_bq(col: &PileupColumn, alt_idx: usize, alt_count: u32) -> f64 {
    if col.depth == 0 || alt_count == 0 {
        return 0.0;
    }

    let alt_mean_q = if col.base_counts[alt_idx] > 0 {
        col.quality_sums[alt_idx] as f64 / f64::from(col.base_counts[alt_idx])
    } else {
        0.0
    };

    let p_error = 10.0_f64.powf(-alt_mean_q / 10.0);
    if p_error >= 1.0 {
        return 0.0;
    }

    let alt = f64::from(alt_count);
    let total = f64::from(col.depth);
    let observed_freq = alt / total;

    if observed_freq <= p_error {
        return 0.0;
    }

    let lr = (observed_freq / p_error).log10();
    (alt * lr * 10.0).min(999.0)
}

/// Legacy quality function (flat error rate assumption).
pub(super) fn variant_quality(alt_count: u32, total_depth: u32, _frequency: f64) -> f64 {
    if total_depth == 0 {
        return 0.0;
    }
    let error_rate = 0.001;
    let alt = f64::from(alt_count);
    let total = f64::from(total_depth);
    let observed_freq = alt / total;
    if observed_freq <= error_rate {
        return 0.0;
    }
    let lr = (observed_freq / error_rate).log10();
    (alt * lr * 10.0).min(999.0)
}

// ── Quality-weighted binomial model (WS-11) ─────────────────────

/// Quality-weighted binomial p-value for a variant call.
///
/// Computes the one-sided binomial probability of seeing at least `k`
/// variant-supporting reads out of `n` total, where the null hypothesis
/// error rate `p_err` is a **combined error** from base quality (Phred)
/// and mapping quality (MAPQ):
///
///   P(error) = P_base + P_map - P_base × P_map
///
/// This follows breseq's approach: a non-reference base can arise from
/// either a sequencing error (base quality) OR a mismapped read (MAPQ).
/// When MAPQ data is unavailable (mapq_sums == 0), falls back to base
/// quality only (pre-V205 behavior).
///
/// Returns a Phred-scaled quality score.
#[expect(clippy::cast_precision_loss, reason = "quality sums bounded by coverage")]
pub(super) fn binomial_quality(col: &PileupColumn, alt_idx: usize, alt_count: u32) -> f64 {
    if col.depth == 0 || alt_count == 0 {
        return 0.0;
    }

    let alt_mean_q = if col.base_counts[alt_idx] > 0 {
        col.quality_sums[alt_idx] as f64 / f64::from(col.base_counts[alt_idx])
    } else {
        0.0
    };

    let p_base_err = 10.0_f64.powf(-alt_mean_q / 10.0);

    // Combined error: incorporate MAPQ if available
    let p_err = if col.mapq_sums[alt_idx] > 0 {
        let alt_mean_mapq = col.mean_mapq(alt_idx);
        let p_map_err = 10.0_f64.powf(-alt_mean_mapq / 10.0);
        // P(error) = 1 - (1-P_base)(1-P_map) = P_base + P_map - P_base*P_map
        p_base_err + p_map_err - p_base_err * p_map_err
    } else {
        p_base_err
    };

    if p_err >= 1.0 || p_err <= 0.0 {
        return 0.0;
    }

    let n = col.depth;
    let k = alt_count;

    let log_p = binomial_log_sf(k, n, p_err);

    let quality = -10.0 * log_p / std::f64::consts::LN_10;
    quality.clamp(0.0, 999.0)
}

/// Log survival function: ln(P(X >= k)) for X ~ Binomial(n, p).
///
/// For large n, uses the normal approximation to avoid combinatorial
/// overflow: P(X >= k) ≈ Φ_c((k - 0.5 - np) / sqrt(np(1-p))).
pub(super) fn binomial_log_sf(k: u32, n: u32, p: f64) -> f64 {
    if k == 0 {
        return 0.0;
    }

    let nf = f64::from(n);
    let kf = f64::from(k);

    if kf <= nf * p {
        return 0.0;
    }

    if n > 50 {
        let mu = nf * p;
        let sigma = (nf * p * (1.0 - p)).sqrt();
        if sigma <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let z = (kf - 0.5 - mu) / sigma;
        return log_normal_sf(z);
    }

    let mut log_sum = f64::NEG_INFINITY;
    for i in k..=n {
        let log_pmf = log_binom_pmf(i, n, p);
        log_sum = log_add_exp(log_sum, log_pmf);
    }
    log_sum
}

/// Log of the binomial PMF: ln(C(n,k) * p^k * (1-p)^(n-k)).
pub(super) fn log_binom_pmf(k: u32, n: u32, p: f64) -> f64 {
    let kf = f64::from(k);
    let nf = f64::from(n);
    kf.mul_add(p.ln(), (nf - kf).mul_add((1.0 - p).ln(), log_binom_coeff(n, k)))
}

/// Log binomial coefficient: ln(C(n, k)).
pub(super) fn log_binom_coeff(n: u32, k: u32) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    log_gamma(f64::from(n) + 1.0) - log_gamma(f64::from(k) + 1.0)
        - log_gamma(f64::from(n - k) + 1.0)
}

/// Stirling's log-gamma approximation: ln(Γ(x)).
///
/// Uses the Lanczos approximation for small x, Stirling for large x.
/// Adequate precision for sequencing-depth scale (n < 10^6).
pub(super) fn log_gamma(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 12.0 {
        let x2 = x * x;
        let half_ln_2pi = 0.5 * (2.0 * std::f64::consts::PI).ln();
        return (x - 0.5).mul_add(x.ln(), half_ln_2pi.mul_add(1.0, -x))
            + 1.0 / (12.0 * x)
            - 1.0 / (360.0 * x2 * x);
    }
    let mut log_shift = 0.0;
    let mut z = x;
    #[expect(clippy::while_float, reason = "reduction loop with finite termination")]
    while z < 12.0 {
        log_shift += z.ln();
        z += 1.0;
    }
    log_gamma(z) - log_shift
}

/// Log of the normal survival function: ln(P(Z > z)) for standard normal.
///
/// Uses the rational approximation from Abramowitz & Stegun 26.2.17.
pub(super) fn log_normal_sf(z: f64) -> f64 {
    if z < -8.0 {
        return 0.0;
    }
    if z > 37.0 {
        return f64::NEG_INFINITY;
    }

    if z < 0.0 {
        let p_lower = 1.0 - normal_sf_approx(-z);
        return (1.0 - p_lower).ln();
    }

    let p = normal_sf_approx(z);
    if p <= 0.0 {
        f64::NEG_INFINITY
    } else {
        p.ln()
    }
}

/// Upper-tail probability P(Z > z) for z >= 0 using A&S 26.2.17.
pub(super) fn normal_sf_approx(z: f64) -> f64 {
    let t = 1.0 / 0.231_641_9_f64.mul_add(z, 1.0);
    let d = (1.0 / (2.0 * std::f64::consts::PI)).sqrt() * (-z * z / 2.0).exp();
    let p = d * t
        * t.mul_add(
            t.mul_add(
                t.mul_add(
                    t.mul_add(1.330_274_429, -1.821_255_978),
                    1.781_477_937,
                ),
                -0.356_563_782,
            ),
            0.319_381_530,
        );
    p.max(0.0)
}

/// Numerically stable log(exp(a) + exp(b)).
pub(super) fn log_add_exp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let max = a.max(b);
    max + ((a - max).exp() + (b - max).exp()).ln()
}
