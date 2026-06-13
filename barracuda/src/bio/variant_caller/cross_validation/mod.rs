// SPDX-License-Identifier: AGPL-3.0-or-later
//! breseq polymorphism cross-validation for LTEE data (WS-11).
//!
//! Given a set of known mutations (from breseq GD output) and our sovereign
//! variant caller results, computes concordance statistics:
//!
//! - **Sensitivity** (recall): fraction of breseq mutations also called by us
//! - **Precision**: fraction of our calls confirmed by breseq
//! - **F1 score**: harmonic mean of sensitivity and precision
//! - **Per-type concordance**: broken down by SNP/DEL/INS
//!
//! This module integrates with the [`LteeThresholds`](super::thresholds::LteeThresholds)
//! to validate that generation-aware thresholds improve concordance over fixed
//! thresholds across the LTEE evolutionary trajectory.

#[cfg(test)]
mod tests;

use super::{CalledVariant, CallerConfig, VariantType};
use super::thresholds::LteeThresholds;

/// Concordance statistics from cross-validation.
#[derive(Debug, Clone)]
pub struct ConcordanceStats {
    /// Mutations found by both callers (true positives).
    pub true_positives: usize,
    /// Mutations found only by the reference caller (false negatives — we missed them).
    pub false_negatives: usize,
    /// Mutations found only by our caller (false positives — not in reference).
    pub false_positives: usize,
    /// Position window used for matching (bp).
    pub window: usize,
}

impl ConcordanceStats {
    /// Sensitivity (recall): TP / (TP + FN).
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "mutation counts fit comfortably in f64 mantissa"
    )]
    pub fn sensitivity(&self) -> f64 {
        let denom = self.true_positives + self.false_negatives;
        if denom == 0 {
            return 0.0;
        }
        self.true_positives as f64 / denom as f64
    }

    /// Precision: TP / (TP + FP).
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "mutation counts fit comfortably in f64 mantissa"
    )]
    pub fn precision(&self) -> f64 {
        let denom = self.true_positives + self.false_positives;
        if denom == 0 {
            return 0.0;
        }
        self.true_positives as f64 / denom as f64
    }

    /// F1 score: 2 × (precision × sensitivity) / (precision + sensitivity).
    #[must_use]
    pub fn f1_score(&self) -> f64 {
        let p = self.precision();
        let s = self.sensitivity();
        if p + s == 0.0 {
            return 0.0;
        }
        2.0 * p * s / (p + s)
    }

    /// Total reference mutations (TP + FN).
    #[must_use]
    pub const fn reference_total(&self) -> usize {
        self.true_positives + self.false_negatives
    }

    /// Total sovereign calls (TP + FP).
    #[must_use]
    pub const fn sovereign_total(&self) -> usize {
        self.true_positives + self.false_positives
    }
}

/// Per-type concordance breakdown.
#[derive(Debug, Clone, Default)]
pub struct TypedConcordance {
    /// SNP-type concordance.
    pub snp: ConcordanceStats,
    /// Deletion-type concordance.
    pub del: ConcordanceStats,
    /// Insertion-type concordance.
    pub ins: ConcordanceStats,
    /// Aggregate concordance across all types.
    pub overall: ConcordanceStats,
}

impl Default for ConcordanceStats {
    fn default() -> Self {
        Self {
            true_positives: 0,
            false_negatives: 0,
            false_positives: 0,
            window: 5,
        }
    }
}

/// Cross-validate our sovereign calls against a breseq reference set.
///
/// The `reference` mutations are parsed from a GD file (type, position, new_base).
/// The `sovereign` calls are from our variant caller.
///
/// Uses a position window of `window` bp for matching (default: 5).
#[must_use]
pub fn cross_validate(
    sovereign: &[CalledVariant],
    reference: &[(String, usize, String)],
    window: usize,
) -> TypedConcordance {
    let (snp_ref, del_ref, ins_ref) = partition_reference(reference);
    let (snp_sov, del_sov, ins_sov) = partition_sovereign(sovereign);

    let snp = compute_concordance(&snp_sov, &snp_ref, window);
    let del = compute_concordance(&del_sov, &del_ref, window);
    let ins = compute_concordance(&ins_sov, &ins_ref, window);

    let overall = ConcordanceStats {
        true_positives: snp.true_positives + del.true_positives + ins.true_positives,
        false_negatives: snp.false_negatives + del.false_negatives + ins.false_negatives,
        false_positives: snp.false_positives + del.false_positives + ins.false_positives,
        window,
    };

    TypedConcordance {
        snp,
        del,
        ins,
        overall,
    }
}

/// Cross-validate with generation-aware thresholds and compare against fixed.
///
/// Returns `(gen_aware_concordance, fixed_concordance)` for sensitivity comparison.
#[must_use]
pub fn compare_threshold_strategies(
    sovereign_gen: &[CalledVariant],
    sovereign_fixed: &[CalledVariant],
    reference: &[(String, usize, String)],
    window: usize,
) -> (TypedConcordance, TypedConcordance) {
    let gen_aware = cross_validate(sovereign_gen, reference, window);
    let fixed = cross_validate(sovereign_fixed, reference, window);
    (gen_aware, fixed)
}

/// Compute the effective caller config for a given LTEE generation.
///
/// Applies [`LteeThresholds`] to a permissive base config,
/// producing the generation-appropriate variant calling parameters.
#[must_use]
pub fn config_for_generation(generation: u64) -> CallerConfig {
    let thresholds = LteeThresholds::at_generation(generation);
    let base = CallerConfig {
        quality_weighted: true,
        binomial_quality: true,
        ..CallerConfig::permissive()
    };
    thresholds.apply_to(&base)
}

fn partition_reference(reference: &[(String, usize, String)]) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let mut snp = Vec::new();
    let mut del = Vec::new();
    let mut ins = Vec::new();

    for (vtype, pos, _) in reference {
        match vtype.as_str() {
            "SNP" | "SUB" => snp.push(*pos),
            "DEL" => del.push(*pos),
            "INS" => ins.push(*pos),
            _ => {} // MOB, AMP, CON, INV — skip for now
        }
    }
    (snp, del, ins)
}

fn partition_sovereign(sovereign: &[CalledVariant]) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let mut snp = Vec::new();
    let mut del = Vec::new();
    let mut ins = Vec::new();

    for call in sovereign {
        match call.variant_type {
            VariantType::Snp => snp.push(call.position),
            VariantType::Deletion => del.push(call.position),
            VariantType::Insertion => ins.push(call.position),
        }
    }
    (snp, del, ins)
}

fn compute_concordance(
    sovereign_positions: &[usize],
    reference_positions: &[usize],
    window: usize,
) -> ConcordanceStats {
    let mut ref_matched = vec![false; reference_positions.len()];
    let mut tp = 0;
    let mut fp = 0;

    for &sov_pos in sovereign_positions {
        let found = reference_positions.iter().enumerate().any(|(i, &ref_pos)| {
            if ref_matched[i] {
                return false;
            }
            if sov_pos.abs_diff(ref_pos) <= window {
                ref_matched[i] = true;
                true
            } else {
                false
            }
        });
        if found {
            tp += 1;
        } else {
            fp += 1;
        }
    }

    let fn_count = ref_matched.iter().filter(|&&m| !m).count();

    ConcordanceStats {
        true_positives: tp,
        false_negatives: fn_count,
        false_positives: fp,
        window,
    }
}
