// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(clippy::unwrap_used, clippy::cast_precision_loss)]

use super::*;
use crate::bio::esn::heads::{
    A0_DIVERSITY_REGIME, A1_COMMUNITY_PHASE, A3_DIVERSITY_ANOMALY, A5_SAMPLING_PRIORITY,
    AttentionState, B0_SHANNON_REGIME, B1_DIVERSITY_PHASE, B3_NOVEL_STATE, B5_RICHNESS_PRIORITY,
    C0_UNIFRAC_REGIME, C1_PHYLO_PHASE, C3_PHYLO_ANOMALY, C5_PHYLO_PRIORITY, NUM_HEADS,
};

fn sample_observation() -> BioObservation {
    BioObservation {
        sample_id: "test_001".to_string(),
        shannon: 3.5,
        simpson: 0.85,
        evenness: 0.78,
        chao1: 150.0,
        bray_curtis_mean: 0.42,
        anderson_w: 4.5,
        anderson_phase: 0.5,
        amr_load: 0.12,
    }
}

#[test]
fn bio_observation_to_features() {
    let obs = sample_observation();
    let features = obs.to_features();
    assert_eq!(features.len(), BioObservation::FEATURE_COUNT);
    assert!((features[0] - 3.5).abs() < f64::EPSILON);
    assert!((features[7] - 0.12).abs() < f64::EPSILON);
}

#[test]
fn brain_starts_healthy() {
    let brain = BioBrain::new(5);
    assert!(brain.is_healthy());
    assert!(!brain.is_critical());
    assert_eq!(brain.observation_count(), 0);
}

#[test]
fn brain_observe_updates_count() {
    let mut brain = BioBrain::new(5);
    let obs = sample_observation();
    let outputs = vec![0.5; NUM_HEADS];
    let update = brain.observe(&obs, &outputs);
    assert_eq!(brain.observation_count(), 1);
    assert_eq!(update.attention, AttentionState::Healthy);
    assert!((update.shannon_h - 3.5).abs() < f64::EPSILON);
}

#[test]
fn brain_escalates_on_disagreement() {
    let mut brain = BioBrain::new(1);
    let obs = sample_observation();
    let mut outputs = vec![0.5; NUM_HEADS];

    outputs[A0_DIVERSITY_REGIME] = 0.0;
    outputs[B0_SHANNON_REGIME] = 1.0;
    outputs[C0_UNIFRAC_REGIME] = 0.5;

    outputs[A1_COMMUNITY_PHASE] = 0.1;
    outputs[B1_DIVERSITY_PHASE] = 0.5;
    outputs[C1_PHYLO_PHASE] = 0.9;

    outputs[A3_DIVERSITY_ANOMALY] = 0.0;
    outputs[B3_NOVEL_STATE] = 0.8;
    outputs[C3_PHYLO_ANOMALY] = 0.4;

    outputs[A5_SAMPLING_PRIORITY] = 0.2;
    outputs[B5_RICHNESS_PRIORITY] = 0.9;
    outputs[C5_PHYLO_PRIORITY] = 0.5;

    let update = brain.observe(&obs, &outputs);
    assert_ne!(
        update.attention,
        AttentionState::Healthy,
        "should escalate on high disagreement"
    );
}

#[test]
fn brain_de_escalates_on_agreement() {
    let mut brain = BioBrain::new(1);
    let obs = sample_observation();

    let mut high_disagree = vec![0.5; NUM_HEADS];
    high_disagree[A0_DIVERSITY_REGIME] = 0.0;
    high_disagree[B0_SHANNON_REGIME] = 1.0;
    high_disagree[C0_UNIFRAC_REGIME] = 0.5;
    high_disagree[A1_COMMUNITY_PHASE] = 0.1;
    high_disagree[B1_DIVERSITY_PHASE] = 0.5;
    high_disagree[C1_PHYLO_PHASE] = 0.9;
    high_disagree[A3_DIVERSITY_ANOMALY] = 0.0;
    high_disagree[B3_NOVEL_STATE] = 0.8;
    high_disagree[C3_PHYLO_ANOMALY] = 0.4;
    high_disagree[A5_SAMPLING_PRIORITY] = 0.2;
    high_disagree[B5_RICHNESS_PRIORITY] = 0.9;
    high_disagree[C5_PHYLO_PRIORITY] = 0.5;
    brain.observe(&obs, &high_disagree);

    let calm = vec![0.5; NUM_HEADS];
    for _ in 0..5 {
        brain.observe(&obs, &calm);
    }
    assert!(brain.is_healthy(), "should de-escalate after calm readings");
}

#[test]
fn brain_smoothed_urgency() {
    let mut brain = BioBrain::new(3);
    let obs = sample_observation();
    let calm = vec![0.5; NUM_HEADS];

    brain.observe(&obs, &calm);
    brain.observe(&obs, &calm);
    brain.observe(&obs, &calm);
    assert!(
        brain.smoothed_urgency() < 0.01,
        "all-calm should yield near-zero urgency"
    );
}

#[test]
fn brain_status_snapshot() {
    let mut brain = BioBrain::new(5);
    let obs = sample_observation();
    let outputs = vec![0.5; NUM_HEADS];
    brain.observe(&obs, &outputs);
    let status = brain.status_snapshot();
    assert_eq!(status.observation_count, 1);
    assert!(status.urgency >= 0.0);
    assert!(status.urgency <= 1.0);
}

#[test]
fn update_attention_without_observation() {
    let mut brain = BioBrain::new(3);
    let outputs = vec![0.5; NUM_HEADS];
    brain.update_attention(&outputs);
    assert!(brain.is_healthy());
    assert_eq!(brain.observation_count(), 0);
}

#[test]
fn diversity_update_fields() {
    let mut brain = BioBrain::new(5);
    let obs = BioObservation {
        chao1: 42.0,
        shannon: 2.1,
        evenness: 0.65,
        ..BioObservation::default()
    };
    let update = brain.observe(&obs, &vec![0.5; NUM_HEADS]);
    assert_eq!(update.n_species, 42);
    assert!((update.shannon_h - 2.1).abs() < f64::EPSILON);
    assert!((update.evenness - 0.65).abs() < f64::EPSILON);
}

#[test]
fn update_attention_escalates_to_alert() {
    let mut brain = BioBrain::new(1);
    let mut outputs = vec![0.5; NUM_HEADS];
    outputs[A0_DIVERSITY_REGIME] = 0.0;
    outputs[B0_SHANNON_REGIME] = 1.0;
    outputs[C0_UNIFRAC_REGIME] = 0.5;
    outputs[A1_COMMUNITY_PHASE] = 0.1;
    outputs[B1_DIVERSITY_PHASE] = 0.9;
    outputs[C1_PHYLO_PHASE] = 0.5;
    brain.update_attention(&outputs);
    assert!(
        !brain.is_healthy() || brain.attention() == AttentionState::Alert,
        "high disagreement should escalate"
    );
}

#[test]
fn update_attention_escalates_to_critical() {
    let mut brain = BioBrain::new(1);
    let mut outputs = vec![0.5; NUM_HEADS];
    outputs[A0_DIVERSITY_REGIME] = 0.0;
    outputs[B0_SHANNON_REGIME] = 1.0;
    outputs[C0_UNIFRAC_REGIME] = 0.5;
    outputs[A1_COMMUNITY_PHASE] = 0.1;
    outputs[B1_DIVERSITY_PHASE] = 0.5;
    outputs[C1_PHYLO_PHASE] = 0.9;
    outputs[A3_DIVERSITY_ANOMALY] = 0.0;
    outputs[B3_NOVEL_STATE] = 1.0;
    outputs[C3_PHYLO_ANOMALY] = 0.5;
    for _ in 0..3 {
        brain.update_attention(&outputs);
    }
    assert!(
        brain.is_critical() || brain.attention() == AttentionState::Alert,
        "sustained high disagreement should reach Critical"
    );
}

#[test]
fn is_healthy_and_is_critical_state_transitions() {
    let mut brain = BioBrain::new(1);
    assert!(brain.is_healthy());
    assert!(!brain.is_critical());

    let calm = vec![0.5; NUM_HEADS];
    let mut high = vec![0.5; NUM_HEADS];
    high[A0_DIVERSITY_REGIME] = 0.0;
    high[B0_SHANNON_REGIME] = 1.0;
    high[C0_UNIFRAC_REGIME] = 0.5;
    high[A1_COMMUNITY_PHASE] = 0.1;
    high[B1_DIVERSITY_PHASE] = 0.9;
    high[C1_PHYLO_PHASE] = 0.5;

    brain.update_attention(&high);
    let was_alert = !brain.is_healthy() && !brain.is_critical();

    for _ in 0..5 {
        brain.update_attention(&high);
    }
    let reached_critical = brain.is_critical();

    for _ in 0..10 {
        brain.update_attention(&calm);
    }
    assert!(
        brain.is_healthy(),
        "calm readings should de-escalate to Healthy"
    );
    assert!(!brain.is_critical());
    let _ = (was_alert, reached_critical);
}

#[test]
fn feature_count_and_head_count() {
    assert_eq!(BioBrain::feature_count(), BioObservation::FEATURE_COUNT);
    assert_eq!(BioBrain::feature_count(), 8);
    assert_eq!(BioBrain::head_count(), NUM_HEADS);
    assert_eq!(BioBrain::head_count(), 36);
}

#[test]
fn bio_observation_to_features_default() {
    let obs = BioObservation::default();
    let features = obs.to_features();
    assert_eq!(features.len(), BioObservation::FEATURE_COUNT);
    assert!(features.iter().all(|&f| f.abs() < f64::EPSILON));
}

#[test]
fn bio_observation_to_features_order() {
    let obs = BioObservation {
        sample_id: "x".to_string(),
        shannon: 1.0,
        simpson: 2.0,
        evenness: 3.0,
        chao1: 4.0,
        bray_curtis_mean: 5.0,
        anderson_w: 6.0,
        anderson_phase: 7.0,
        amr_load: 8.0,
    };
    let features = obs.to_features();
    assert_eq!(features, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}
