// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bio observation types and the `BioBrain` adapter.
//!
//! `BioObservation` captures a single metagenomic / environmental sample snapshot.
//! `BioBrain` maintains the attention state machine and converts observations into
//! feature vectors for ESN inference.

use crate::bio::esn::heads::{AttentionState, BioHeadGroupDisagreement, NUM_HEADS};
use crate::cast::f64_usize;

/// A single bio observation (analog to hotSpring `BetaObservation`).
///
/// Captures diversity, phylogenetic, and resistance metrics from one
/// metagenomic sample or sentinel monitoring timepoint.
#[derive(Debug, Clone, Default)]
pub struct BioObservation {
    /// Unique sample identifier.
    pub sample_id: String,
    /// Shannon diversity index (H').
    pub shannon: f64,
    /// Simpson's diversity index (1 - D).
    pub simpson: f64,
    /// Pielou evenness (J').
    pub evenness: f64,
    /// Chao1 richness estimator.
    pub chao1: f64,
    /// Mean Bray-Curtis dissimilarity to reference community.
    pub bray_curtis_mean: f64,
    /// Anderson disorder parameter (W) from spectral analysis of community matrix.
    pub anderson_w: f64,
    /// Anderson phase label: 0.0 (localized), 0.5 (transitioning), 1.0 (extended).
    pub anderson_phase: f64,
    /// Antimicrobial resistance gene load (normalized 0–1).
    pub amr_load: f64,
}

impl BioObservation {
    /// Number of features when converted to a flat vector.
    pub const FEATURE_COUNT: usize = 8;

    /// Convert to a feature vector suitable for ESN input.
    #[must_use]
    pub fn to_features(&self) -> Vec<f64> {
        vec![
            self.shannon,
            self.simpson,
            self.evenness,
            self.chao1,
            self.bray_curtis_mean,
            self.anderson_w,
            self.anderson_phase,
            self.amr_load,
        ]
    }
}

/// Streaming diversity update (analog to hotSpring `CgResidualUpdate`).
///
/// Lightweight message for real-time monitoring: sent from the GPU/CPU compute
/// layer to the NPU/sentinel layer after each sample is processed.
#[derive(Debug, Clone)]
pub struct DiversityUpdate {
    /// Observed species count.
    pub n_species: usize,
    /// Shannon diversity (H').
    pub shannon_h: f64,
    /// Pielou evenness (J').
    pub evenness: f64,
    /// Current attention state after this update.
    pub attention: AttentionState,
}

/// Bio brain adapter managing the attention state machine and observation history.
///
/// Maintains a sliding window of observations and computes head-group disagreement
/// from the most recent 36-head ESN output. The attention state escalates/de-escalates
/// based on disagreement urgency, mirroring hotSpring's brain architecture.
pub struct BioBrain {
    attention: AttentionState,
    last_disagreement: BioHeadGroupDisagreement,
    observation_count: u64,
    window_size: usize,
    recent_urgencies: Vec<f64>,
}

impl BioBrain {
    /// Create a new bio brain with a given urgency sliding-window size.
    #[must_use]
    pub fn new(window_size: usize) -> Self {
        Self {
            attention: AttentionState::Healthy,
            last_disagreement: BioHeadGroupDisagreement::default(),
            observation_count: 0,
            window_size: window_size.max(1),
            recent_urgencies: Vec::new(),
        }
    }

    /// Process a new observation and 36-head ESN output vector.
    ///
    /// Updates disagreement, urgency history, and attention state. Returns
    /// a `DiversityUpdate` suitable for the sentinel monitoring channel.
    pub fn observe(&mut self, obs: &BioObservation, head_outputs: &[f64]) -> DiversityUpdate {
        self.observation_count += 1;
        self.last_disagreement = BioHeadGroupDisagreement::from_outputs(head_outputs);
        let urgency = self.last_disagreement.urgency();

        self.recent_urgencies.push(urgency);
        if self.recent_urgencies.len() > self.window_size {
            self.recent_urgencies.remove(0);
        }

        let smoothed = self.smoothed_urgency();
        self.attention = self.attention.transition(smoothed);

        DiversityUpdate {
            n_species: f64_usize(obs.chao1),
            shannon_h: obs.shannon,
            evenness: obs.evenness,
            attention: self.attention,
        }
    }

    /// Current attention state.
    #[must_use]
    pub const fn attention(&self) -> AttentionState {
        self.attention
    }

    /// Most recent head-group disagreement.
    #[must_use]
    pub const fn last_disagreement(&self) -> &BioHeadGroupDisagreement {
        &self.last_disagreement
    }

    /// Current smoothed urgency (mean of recent window).
    #[must_use]
    pub fn smoothed_urgency(&self) -> f64 {
        if self.recent_urgencies.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent_urgencies.iter().sum();
        let n = self.recent_urgencies.len();
        #[expect(clippy::cast_precision_loss)]
        let divisor = n as f64;
        sum / divisor
    }

    /// Total observations processed.
    #[must_use]
    pub const fn observation_count(&self) -> u64 {
        self.observation_count
    }

    /// Whether the brain is in a healthy state (no alerts).
    #[must_use]
    pub const fn is_healthy(&self) -> bool {
        matches!(self.attention, AttentionState::Healthy)
    }

    /// Whether the brain is in critical state (needs human review).
    #[must_use]
    pub const fn is_critical(&self) -> bool {
        matches!(self.attention, AttentionState::Critical)
    }

    /// Generate a JSON-compatible status snapshot for IPC.
    #[must_use]
    pub fn status_snapshot(&self) -> BrainStatus {
        BrainStatus {
            attention: self.attention,
            urgency: self.smoothed_urgency(),
            observation_count: self.observation_count,
            delta_regime: self.last_disagreement.delta_regime,
            delta_phase: self.last_disagreement.delta_phase,
            delta_anomaly: self.last_disagreement.delta_anomaly,
            delta_priority: self.last_disagreement.delta_priority,
        }
    }

    /// Process a 36-head output without a full observation (useful for
    /// attention-only queries when the bio observation is not yet complete).
    pub fn update_attention(&mut self, head_outputs: &[f64]) {
        self.last_disagreement = BioHeadGroupDisagreement::from_outputs(head_outputs);
        let urgency = self.last_disagreement.urgency();
        self.recent_urgencies.push(urgency);
        if self.recent_urgencies.len() > self.window_size {
            self.recent_urgencies.remove(0);
        }
        self.attention = self.attention.transition(self.smoothed_urgency());
    }

    /// Feature vector size expected by ESN input layer.
    #[must_use]
    pub const fn feature_count() -> usize {
        BioObservation::FEATURE_COUNT
    }

    /// Number of ESN output heads expected.
    #[must_use]
    pub const fn head_count() -> usize {
        NUM_HEADS
    }
}

/// Snapshot of brain status for IPC serialization.
#[derive(Debug, Clone)]
pub struct BrainStatus {
    /// Current attention state.
    pub attention: AttentionState,
    /// Smoothed urgency score.
    pub urgency: f64,
    /// Total observations processed.
    pub observation_count: u64,
    /// Regime disagreement.
    pub delta_regime: f64,
    /// Phase disagreement.
    pub delta_phase: f64,
    /// Anomaly disagreement.
    pub delta_anomaly: f64,
    /// Priority disagreement.
    pub delta_priority: f64,
}
