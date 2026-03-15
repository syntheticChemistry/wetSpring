// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bio 36-head Gen 2 layout adapted from hotSpring's physics brain architecture.
//!
//! 6 groups of 6 overlapping heads predict the same bio quantities from different
//! models. Disagreement between groups = epistemic uncertainty = attention escalation.
//!
//! # Cross-spring provenance
//!
//! - Layout concept: hotSpring v0.6.15 (`reservoir.rs` Gen 2 "Developed Organism")
//! - Physics→bio mapping: wetSpring V89 (`CROSS_SPRING_EVOLUTION.md`)
//! - Upstream ESN: barraCuda `MultiHeadEsn` (S79, hotSpring 36-head absorption)

// ── Group A: Anderson-informed (disorder → localization → spectral) ──

/// Anderson disorder W → diversity regime cost.
pub const A0_DIVERSITY_REGIME: usize = 0;
/// Anderson phase → community phase (stable / transitioning / collapsed).
pub const A1_COMMUNITY_PHASE: usize = 1;
/// Anderson `λ_min` → community resilience floor.
pub const A2_RESILIENCE_FLOOR: usize = 2;
/// Anderson anomaly → diversity anomaly.
pub const A3_DIVERSITY_ANOMALY: usize = 3;
/// Anderson thermalization → community equilibration time.
pub const A4_EQUILIBRATION: usize = 4;
/// Anderson priority → sampling priority.
pub const A5_SAMPLING_PRIORITY: usize = 5;

// ── Group B: Diversity-empirical (Shannon/Simpson/Chao1 observables) ──

/// Shannon diversity → regime classification cost.
pub const B0_SHANNON_REGIME: usize = 6;
/// Community phase from diversity trajectory.
pub const B1_DIVERSITY_PHASE: usize = 7;
/// Sampling acceptance (quality filter pass rate).
pub const B2_QUALITY_ACCEPT: usize = 8;
/// Diversity anomaly (novel community state detected).
pub const B3_NOVEL_STATE: usize = 9;
/// Community equilibration from evenness trajectory.
pub const B4_EVENNESS_EQUIL: usize = 10;
/// Sampling priority from richness trajectory.
pub const B5_RICHNESS_PRIORITY: usize = 11;

// ── Group C: Phylogeny-informed (UniFrac, AMR, clade structure) ──

/// `UniFrac` distance → regime transition cost.
pub const C0_UNIFRAC_REGIME: usize = 12;
/// Phylogenetic phase (functional vs taxonomic divergence).
pub const C1_PHYLO_PHASE: usize = 13;
/// AMR gene load → resistance regime.
pub const C2_AMR_REGIME: usize = 14;
/// Phylogenetic anomaly (novel clade emergence).
pub const C3_PHYLO_ANOMALY: usize = 15;
/// Phylogenetic equilibration time.
pub const C4_PHYLO_EQUIL: usize = 16;
/// Phylogenetic sampling priority.
pub const C5_PHYLO_PRIORITY: usize = 17;

// ── Group D: Steering / Control (action targets) ──

/// Next sample recommendation.
pub const D0_NEXT_SAMPLE: usize = 18;
/// Optimal sequencing depth.
pub const D1_SEQ_DEPTH: usize = 19;
/// Optimal rarefaction depth.
pub const D2_RARE_DEPTH: usize = 20;
/// Re-sampling interval.
pub const D3_RESAMPLE_INTERVAL: usize = 21;
/// Alert / flag decision.
pub const D4_ALERT_FLAG: usize = 22;
/// Skip-sample decision.
pub const D5_SKIP_DECISION: usize = 23;

// ── Group E: Brain / Monitor (real-time pipeline stream) ──

/// Pipeline residual ETA (how much compute remains).
pub const E0_RESIDUAL_ETA: usize = 24;
/// Data quality anomaly (bad reads, chimeras).
pub const E1_QUALITY_ANOMALY: usize = 25;
/// Convergence rate (diversity accumulation curve slope).
pub const E2_CONVERGENCE_RATE: usize = 26;
/// Stall detector (no new species across N samples).
pub const E3_STALL_DETECTOR: usize = 27;
/// Divergence detector (community collapse signal).
pub const E4_DIVERGENCE_DETECTOR: usize = 28;
/// Quality forecast (expected output quality given current trajectory).
pub const E5_QUALITY_FORECAST: usize = 29;

// ── Group M: Meta-mixer (cross-group agreement, proxy trust) ──

/// Diversity consensus (Anderson vs empirical vs phylo).
pub const M0_DIVERSITY_CONSENSUS: usize = 30;
/// Phase consensus across all three model families.
pub const M1_PHASE_CONSENSUS: usize = 31;
/// Diversity uncertainty magnitude.
pub const M2_DIVERSITY_UNCERTAINTY: usize = 32;
/// Phase uncertainty magnitude.
pub const M3_PHASE_UNCERTAINTY: usize = 33;
/// Proxy trust (how well Anderson predicts empirical diversity).
pub const M4_PROXY_TRUST: usize = 34;
/// Scalar attention level (urgency).
pub const M5_ATTENTION_LEVEL: usize = 35;

/// Total number of bio heads (Gen 2).
pub const NUM_HEADS: usize = 36;

/// Number of heads per group.
pub const GROUP_SIZE: usize = 6;

/// Group A base index (Anderson-informed).
pub const GROUP_A: usize = 0;
/// Group B base index (Diversity-empirical).
pub const GROUP_B: usize = 6;
/// Group C base index (Phylogeny-informed).
pub const GROUP_C: usize = 12;
/// Group D base index (Steering / Control).
pub const GROUP_D: usize = 18;
/// Group E base index (Brain / Monitor).
pub const GROUP_E: usize = 24;
/// Group M base index (Meta-mixer).
pub const GROUP_M: usize = 30;

/// Regime cost heads across groups (for disagreement: Anderson vs Shannon vs `UniFrac`).
pub const REGIME_HEADS: [usize; 3] = [A0_DIVERSITY_REGIME, B0_SHANNON_REGIME, C0_UNIFRAC_REGIME];
/// Phase heads across groups.
pub const PHASE_HEADS: [usize; 3] = [A1_COMMUNITY_PHASE, B1_DIVERSITY_PHASE, C1_PHYLO_PHASE];
/// Anomaly heads across groups.
pub const ANOMALY_HEADS: [usize; 3] = [A3_DIVERSITY_ANOMALY, B3_NOVEL_STATE, C3_PHYLO_ANOMALY];
/// Priority heads across groups.
pub const PRIORITY_HEADS: [usize; 3] = [
    A5_SAMPLING_PRIORITY,
    B5_RICHNESS_PRIORITY,
    C5_PHYLO_PRIORITY,
];

/// Urgency threshold for escalation from `Healthy` to `Alert`.
///
/// When head-group disagreement urgency exceeds this value, the attention
/// state machine escalates from normal operation to increased monitoring.
/// 0.6 balances sensitivity (catching real anomalies) against false alarms
/// (normal stochastic variation in head outputs).
///
/// Provenance: adapted from hotSpring `AttentionState` physics thresholds,
/// re-calibrated for bio diversity prediction uncertainty ranges.
pub const URGENCY_ESCALATE_ALERT: f64 = 0.6;

/// Urgency threshold for escalation from `Alert` to `Critical`.
///
/// Requires higher disagreement than alert escalation to avoid
/// premature human-review flagging. 0.8 means ≥80% of the maximum
/// possible head-group disagreement.
pub const URGENCY_ESCALATE_CRITICAL: f64 = 0.8;

/// Urgency threshold for de-escalation (recovery to lower state).
///
/// The system only de-escalates when urgency drops below this value,
/// requiring sustained agreement across head groups before relaxing
/// monitoring cadence. 0.3 provides hysteresis against oscillation
/// at boundary urgency levels.
pub const URGENCY_DEESCALATE: f64 = 0.3;

/// Phase label boundary between low (0) and medium (1) regimes.
///
/// Used by [`BioHeadGroupDisagreement::from_outputs`] to discretize
/// continuous head outputs into phase labels for disagreement counting.
/// Matches [`URGENCY_DEESCALATE`] for consistency in the state machine.
pub const PHASE_LABEL_LOW: f64 = 0.3;

/// Phase label boundary between medium (1) and high (2) regimes.
///
/// Matches [`URGENCY_ESCALATE_ALERT`] for consistency.
pub const PHASE_LABEL_HIGH: f64 = 0.6;

/// Bio attention state (adapted from hotSpring `AttentionState`).
///
/// Tracks the overall health of the bio monitoring pipeline.
/// Escalation: `Healthy` → `Alert` → `Critical`.
/// De-escalation requires consecutive normal readings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionState {
    /// All heads agree, no anomalies. Normal operation.
    Healthy,
    /// Moderate disagreement or anomaly detected. Increase monitoring cadence.
    Alert,
    /// High disagreement or divergence. Flag for human review or corrective action.
    Critical,
}

impl AttentionState {
    /// Transition based on urgency score (0.0–1.0).
    #[must_use]
    pub fn transition(self, urgency: f64) -> Self {
        match self {
            Self::Healthy => {
                if urgency > URGENCY_ESCALATE_ALERT {
                    Self::Alert
                } else {
                    Self::Healthy
                }
            }
            Self::Alert => {
                if urgency > URGENCY_ESCALATE_CRITICAL {
                    Self::Critical
                } else if urgency < URGENCY_DEESCALATE {
                    Self::Healthy
                } else {
                    Self::Alert
                }
            }
            Self::Critical => {
                if urgency < URGENCY_DEESCALATE {
                    Self::Alert
                } else {
                    Self::Critical
                }
            }
        }
    }
}

/// Cross-group disagreement signals for bio heads.
///
/// Computed from a full 36-head output vector by comparing overlapping heads
/// across Groups A (Anderson), B (Diversity-empirical), C (Phylogeny).
/// Large disagreement = high epistemic uncertainty = attention escalation.
///
/// # Cross-spring provenance
///
/// Adapted from hotSpring `HeadGroupDisagreement` (reservoir.rs L617-671).
/// Physics heads (CG cost, QCD phase, Potts order) mapped to bio equivalents
/// (diversity regime, community phase, phylogenetic anomaly).
#[derive(Debug, Clone, Default)]
pub struct BioHeadGroupDisagreement {
    /// `max(A0,B0,C0) - min(A0,B0,C0)` — regime cost prediction spread.
    pub delta_regime: f64,
    /// Number of groups disagreeing on phase label (0, 1, or 2).
    pub delta_phase: f64,
    /// `max(A3,B3,C3) - min(A3,B3,C3)` — anomaly score spread.
    pub delta_anomaly: f64,
    /// `max(A5,B5,C5) - min(A5,B5,C5)` — sampling priority spread.
    pub delta_priority: f64,
}

impl BioHeadGroupDisagreement {
    /// Compute disagreement from a full head output vector (length >= [`NUM_HEADS`]).
    ///
    /// Returns default (all zeros) if the output has fewer than 36 heads.
    #[must_use]
    pub fn from_outputs(outputs: &[f64]) -> Self {
        if outputs.len() < NUM_HEADS {
            return Self::default();
        }
        let spread = |indices: &[usize]| -> f64 {
            let vals: Vec<f64> = indices.iter().map(|&i| outputs[i]).collect();
            let max = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let min = vals.iter().copied().fold(f64::INFINITY, f64::min);
            max - min
        };
        let phase_disagree = {
            #[expect(clippy::bool_to_int_with_if)]
            let phase_label = |v: f64| -> u8 {
                if v > PHASE_LABEL_HIGH {
                    2
                } else if v < PHASE_LABEL_LOW {
                    0
                } else {
                    1
                }
            };
            let a = phase_label(outputs[PHASE_HEADS[0]]);
            let b = phase_label(outputs[PHASE_HEADS[1]]);
            let c = phase_label(outputs[PHASE_HEADS[2]]);
            let mut distinct: u8 = 1;
            if b != a {
                distinct += 1;
            }
            if c != a && c != b {
                distinct += 1;
            }
            f64::from(distinct - 1)
        };
        Self {
            delta_regime: spread(&REGIME_HEADS),
            delta_phase: phase_disagree,
            delta_anomaly: spread(&ANOMALY_HEADS),
            delta_priority: spread(&PRIORITY_HEADS),
        }
    }

    /// Scalar urgency score for the attention state machine.
    ///
    /// Weighted combination: regime (40%), phase (30%), anomaly (20%), priority (10%).
    /// Returns 0.0 (full agreement) to 1.0 (maximum disagreement).
    #[must_use]
    pub fn urgency(&self) -> f64 {
        self.delta_regime
            .mul_add(
                0.4,
                self.delta_phase.mul_add(
                    0.3,
                    self.delta_anomaly.mul_add(0.2, self.delta_priority * 0.1),
                ),
            )
            .clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    use crate::tolerances;

    #[test]
    fn head_constants_non_overlapping() {
        let all: Vec<usize> = (0..NUM_HEADS).collect();
        let set: HashSet<usize> = all.iter().copied().collect();
        assert_eq!(all.len(), set.len(), "head indices must be unique");
        assert_eq!(NUM_HEADS, 36);
        assert_eq!(GROUP_SIZE, 6);
    }

    #[test]
    fn disagreement_default_for_short_output() {
        let d = BioHeadGroupDisagreement::from_outputs(&[0.5; 10]);
        assert!((d.delta_regime).abs() < f64::EPSILON);
        assert!((d.delta_phase).abs() < f64::EPSILON);
        assert!((d.delta_anomaly).abs() < f64::EPSILON);
        assert!((d.delta_priority).abs() < f64::EPSILON);
    }

    #[test]
    fn disagreement_zero_when_all_agree() {
        let outputs = vec![0.5; NUM_HEADS];
        let d = BioHeadGroupDisagreement::from_outputs(&outputs);
        assert!((d.delta_regime).abs() < f64::EPSILON);
        assert!((d.delta_anomaly).abs() < f64::EPSILON);
        assert!((d.delta_priority).abs() < f64::EPSILON);
        assert!(d.urgency() < 0.01);
    }

    #[test]
    fn disagreement_nonzero_when_groups_diverge() {
        let mut outputs = vec![0.5; NUM_HEADS];
        outputs[A0_DIVERSITY_REGIME] = 0.1;
        outputs[B0_SHANNON_REGIME] = 0.9;
        outputs[C0_UNIFRAC_REGIME] = 0.5;
        let d = BioHeadGroupDisagreement::from_outputs(&outputs);
        assert!(
            (d.delta_regime - 0.8).abs() < tolerances::ANALYTICAL_LOOSE,
            "regime spread = 0.9 - 0.1 = 0.8"
        );
        assert!(d.urgency() > 0.1);
    }

    #[test]
    fn phase_disagreement_counts_distinct_labels() {
        let mut outputs = vec![0.5; NUM_HEADS];
        outputs[A1_COMMUNITY_PHASE] = 0.1; // label 0
        outputs[B1_DIVERSITY_PHASE] = 0.5; // label 1
        outputs[C1_PHYLO_PHASE] = 0.9; // label 2
        let d = BioHeadGroupDisagreement::from_outputs(&outputs);
        assert!(
            (d.delta_phase - 2.0).abs() < f64::EPSILON,
            "3 distinct labels → 2"
        );
    }

    #[test]
    fn urgency_clamped() {
        let d = BioHeadGroupDisagreement {
            delta_regime: 5.0,
            delta_phase: 5.0,
            delta_anomaly: 5.0,
            delta_priority: 5.0,
        };
        assert!((d.urgency() - 1.0).abs() < f64::EPSILON, "clamped to 1.0");
    }

    #[test]
    fn attention_state_transitions() {
        let s = AttentionState::Healthy;
        assert_eq!(s.transition(0.1), AttentionState::Healthy);
        assert_eq!(s.transition(0.7), AttentionState::Alert);

        let s = AttentionState::Alert;
        assert_eq!(s.transition(0.2), AttentionState::Healthy);
        assert_eq!(s.transition(0.5), AttentionState::Alert);
        assert_eq!(s.transition(0.9), AttentionState::Critical);

        let s = AttentionState::Critical;
        assert_eq!(s.transition(0.9), AttentionState::Critical);
        assert_eq!(s.transition(0.2), AttentionState::Alert);
    }
}
