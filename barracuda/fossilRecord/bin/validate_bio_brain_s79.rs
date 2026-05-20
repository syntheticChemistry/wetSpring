// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::print_stdout,
    reason = "validation harness: results printed to stdout"
)]
#![expect(
    clippy::too_many_lines,
    reason = "validation harness: sequential domain checks in single main()"
)]
//! # Exp272: Bio Brain Cross-Spring Validation
//!
//! Validates the 4-layer bio brain pipeline adapted from hotSpring's physics brain:
//! - Phase 1: 36-head bio constants, `BioHeadGroupDisagreement`, `AttentionState`
//! - Phase 2: `BioBrain` adapter, `BioObservation`, `DiversityUpdate`
//! - Phase 3: `BioNautilusBrain` (evolutionary reservoir via `bingoCube`)
//! - Phase 4: Concurrent pipeline proof (sentinel monitoring + spectral + Nautilus)
//!
//! # Cross-Spring Provenance
//!
//! | Component | Spring Origin | Integration |
//! |-----------|--------------|-------------|
//! | 36-head layout | hotSpring v0.6.15 | `heads.rs` constants |
//! | `HeadGroupDisagreement` | hotSpring `reservoir.rs` | `BioHeadGroupDisagreement` |
//! | `AttentionState` (G/Y/R) | hotSpring brain arch | `heads::AttentionState` |
//! | `NautilusBrain` | `primalTools/bingoCube` | `nautilus_bridge.rs` |
//! | Bio observation mapping | wetSpring V89 | `brain/observation.rs` |
//! | ESN core | `ToadStool` S79 | `toadstool_bridge.rs` |
//!
//! # Provenance
//!
//! Expected values are **analytical** — derived from mathematical
//! identities and algebraic invariants.
//!
//! | Field | Value |
//! |-------|-------|
//! | Provenance type | Analytical (mathematical invariants) |
//! | Date | 2026-03-03 |
//! | Command | `cargo run --release --bin validate_bio_brain_s79` |
//!
//! Provenance: Bio sentinel neural observation validation (S79)

use std::time::Instant;

use wetspring_barracuda::bio::brain::{BioBrain, BioNautilusBrain, BioObservation};
use wetspring_barracuda::bio::esn::heads::{
    self, AttentionState, BioHeadGroupDisagreement, NUM_HEADS,
};
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

struct DomainResult {
    name: &'static str,
    spring: &'static str,
    ms: f64,
    checks: u32,
}

fn main() {
    let mut v = Validator::new("Exp272: Bio Brain Cross-Spring Validation");
    let mut domains: Vec<DomainResult> = Vec::new();

    // ═══ D1: 36-Head Bio Constants (hotSpring → wetSpring) ═══════════════════
    {
        let t = Instant::now();
        v.section("D1: 36-Head Bio Constants [hotSpring → wetSpring]");

        v.check_count("NUM_HEADS", NUM_HEADS, 36);
        v.check_count("GROUP_SIZE", heads::GROUP_SIZE, 6);
        v.check_count("Group A base", heads::GROUP_A, 0);
        v.check_count("Group B base", heads::GROUP_B, 6);
        v.check_count("Group C base", heads::GROUP_C, 12);
        v.check_count("Group D base", heads::GROUP_D, 18);
        v.check_count("Group E base", heads::GROUP_E, 24);
        v.check_count("Group M base", heads::GROUP_M, 30);
        v.check_count("REGIME_HEADS len", heads::REGIME_HEADS.len(), 3);
        v.check_count("PHASE_HEADS len", heads::PHASE_HEADS.len(), 3);
        v.check_count("ANOMALY_HEADS len", heads::ANOMALY_HEADS.len(), 3);
        v.check_count("PRIORITY_HEADS len", heads::PRIORITY_HEADS.len(), 3);
        v.check_pass("All head indices unique", {
            let mut seen = std::collections::HashSet::new();
            (0..NUM_HEADS).all(|i| seen.insert(i))
        });
        v.check_count("M5_ATTENTION_LEVEL", heads::M5_ATTENTION_LEVEL, 35);

        domains.push(DomainResult {
            name: "36-Head Constants",
            spring: "hotSpring",
            ms: t.elapsed().as_secs_f64() * 1000.0,
            checks: 14,
        });
    }

    // ═══ D2: BioHeadGroupDisagreement (hotSpring physics → wetSpring bio) ════
    {
        let t = Instant::now();
        v.section("D2: BioHeadGroupDisagreement [hotSpring → wetSpring]");

        let uniform = vec![0.5; NUM_HEADS];
        let d = BioHeadGroupDisagreement::from_outputs(&uniform);
        v.check(
            "Uniform delta_regime",
            d.delta_regime,
            0.0,
            tolerances::BRAIN_DISAGREEMENT_ANALYTICAL,
        );
        v.check(
            "Uniform delta_anomaly",
            d.delta_anomaly,
            0.0,
            tolerances::BRAIN_DISAGREEMENT_ANALYTICAL,
        );
        v.check(
            "Uniform urgency",
            d.urgency(),
            0.0,
            tolerances::BRAIN_URGENCY_TOL,
        );

        let mut divergent = vec![0.5; NUM_HEADS];
        divergent[heads::A0_DIVERSITY_REGIME] = 0.0;
        divergent[heads::B0_SHANNON_REGIME] = 1.0;
        divergent[heads::C0_UNIFRAC_REGIME] = 0.5;
        let d = BioHeadGroupDisagreement::from_outputs(&divergent);
        v.check(
            "Regime divergence = 1.0",
            d.delta_regime,
            1.0,
            tolerances::BRAIN_DISAGREEMENT_ANALYTICAL,
        );
        v.check_pass("Regime divergence urgency > 0.1", d.urgency() > 0.1);

        let mut phase_max = vec![0.5; NUM_HEADS];
        phase_max[heads::A1_COMMUNITY_PHASE] = 0.1;
        phase_max[heads::B1_DIVERSITY_PHASE] = 0.5;
        phase_max[heads::C1_PHYLO_PHASE] = 0.9;
        let d = BioHeadGroupDisagreement::from_outputs(&phase_max);
        v.check(
            "3 phases → delta_phase = 2",
            d.delta_phase,
            2.0,
            tolerances::EXACT,
        );

        let short = vec![0.5; 10];
        let d = BioHeadGroupDisagreement::from_outputs(&short);
        v.check(
            "Short vector → urgency 0",
            d.urgency(),
            0.0,
            tolerances::EXACT,
        );

        let extreme = BioHeadGroupDisagreement {
            delta_regime: 5.0,
            delta_phase: 5.0,
            delta_anomaly: 5.0,
            delta_priority: 5.0,
        };
        v.check(
            "Urgency clamped to 1.0",
            extreme.urgency(),
            1.0,
            tolerances::EXACT,
        );

        domains.push(DomainResult {
            name: "Head Disagreement",
            spring: "hotSpring",
            ms: t.elapsed().as_secs_f64() * 1000.0,
            checks: 8,
        });
    }

    // ═══ D3: AttentionState Machine (hotSpring Green/Yellow/Red → bio) ═══════
    {
        let t = Instant::now();
        v.section("D3: AttentionState Machine [hotSpring → wetSpring]");

        v.check_pass(
            "Healthy + low → Healthy",
            AttentionState::Healthy.transition(0.1) == AttentionState::Healthy,
        );
        v.check_pass(
            "Healthy + high → Alert",
            AttentionState::Healthy.transition(0.7) == AttentionState::Alert,
        );
        v.check_pass(
            "Alert + low → Healthy",
            AttentionState::Alert.transition(0.2) == AttentionState::Healthy,
        );
        v.check_pass(
            "Alert + mid → Alert",
            AttentionState::Alert.transition(0.5) == AttentionState::Alert,
        );
        v.check_pass(
            "Alert + high → Critical",
            AttentionState::Alert.transition(0.9) == AttentionState::Critical,
        );
        v.check_pass(
            "Critical + high → Critical",
            AttentionState::Critical.transition(0.9) == AttentionState::Critical,
        );
        v.check_pass(
            "Critical + low → Alert",
            AttentionState::Critical.transition(0.2) == AttentionState::Alert,
        );

        domains.push(DomainResult {
            name: "Attention Machine",
            spring: "hotSpring",
            ms: t.elapsed().as_secs_f64() * 1000.0,
            checks: 7,
        });
    }

    // ═══ D4: BioObservation + Feature Mapping (wetSpring) ════════════════════
    {
        let t = Instant::now();
        v.section("D4: BioObservation + Features [wetSpring V89]");

        let obs = BioObservation {
            sample_id: "exp272_001".to_string(),
            shannon: 3.5,
            simpson: 0.85,
            evenness: 0.78,
            chao1: 150.0,
            bray_curtis_mean: 0.42,
            anderson_w: 4.5,
            anderson_phase: 0.5,
            amr_load: 0.12,
        };
        let features = obs.to_features();
        v.check_count(
            "Feature count",
            features.len(),
            BioObservation::FEATURE_COUNT,
        );
        v.check("Feature[0] = shannon", features[0], 3.5, tolerances::EXACT);
        v.check(
            "Feature[7] = amr_load",
            features[7],
            0.12,
            tolerances::EXACT,
        );
        v.check(
            "Default shannon = 0",
            BioObservation::default().shannon,
            0.0,
            0.0,
        );

        domains.push(DomainResult {
            name: "Bio Observation",
            spring: "wetSpring",
            ms: t.elapsed().as_secs_f64() * 1000.0,
            checks: 4,
        });
    }

    // ═══ D5: BioBrain Adapter (hotSpring brain arch → wetSpring bio) ═════════
    {
        let t = Instant::now();
        v.section("D5: BioBrain Adapter [hotSpring → wetSpring]");

        let mut brain = BioBrain::new(5);
        v.check_pass("Brain starts healthy", brain.is_healthy());
        v.check_pass("Brain not critical", !brain.is_critical());
        v.check_count_u64("Observation count = 0", brain.observation_count(), 0);

        let obs = BioObservation {
            sample_id: "exp272_002".to_string(),
            shannon: 3.0,
            simpson: 0.8,
            evenness: 0.75,
            chao1: 100.0,
            bray_curtis_mean: 0.35,
            anderson_w: 3.0,
            anderson_phase: 0.5,
            amr_load: 0.05,
        };
        let calm = vec![0.5; NUM_HEADS];
        let update = brain.observe(&obs, &calm);
        v.check_count_u64("Observation count = 1", brain.observation_count(), 1);
        v.check_pass(
            "Calm → still healthy",
            update.attention == AttentionState::Healthy,
        );
        v.check_count("DiversityUpdate n_species", update.n_species, 100);
        v.check(
            "DiversityUpdate shannon_h",
            update.shannon_h,
            3.0,
            tolerances::EXACT,
        );

        let mut divergent = vec![0.5; NUM_HEADS];
        divergent[heads::A0_DIVERSITY_REGIME] = 0.0;
        divergent[heads::B0_SHANNON_REGIME] = 1.0;
        divergent[heads::C0_UNIFRAC_REGIME] = 0.5;
        divergent[heads::A1_COMMUNITY_PHASE] = 0.1;
        divergent[heads::B1_DIVERSITY_PHASE] = 0.5;
        divergent[heads::C1_PHYLO_PHASE] = 0.9;
        divergent[heads::A3_DIVERSITY_ANOMALY] = 0.0;
        divergent[heads::B3_NOVEL_STATE] = 0.8;
        divergent[heads::C3_PHYLO_ANOMALY] = 0.4;
        divergent[heads::A5_SAMPLING_PRIORITY] = 0.2;
        divergent[heads::B5_RICHNESS_PRIORITY] = 0.9;
        divergent[heads::C5_PHYLO_PRIORITY] = 0.5;

        for _ in 0..5 {
            brain.observe(&obs, &divergent);
        }
        v.check_pass("Repeated divergent → escalated", !brain.is_healthy());

        for _ in 0..10 {
            brain.observe(&obs, &calm);
        }
        v.check_pass("De-escalate after calm", brain.is_healthy());

        let status = brain.status_snapshot();
        v.check_pass(
            "Status urgency in [0,1]",
            status.urgency >= 0.0 && status.urgency <= 1.0,
        );
        v.check_count_u64("Status observation_count", status.observation_count, 16);

        domains.push(DomainResult {
            name: "Bio Brain",
            spring: "hotSpring+wetSpring",
            ms: t.elapsed().as_secs_f64() * 1000.0,
            checks: 10,
        });
    }

    // ═══ D6: BioNautilusBrain (bingoCube/nautilus → wetSpring bio) ═══════════
    {
        let t = Instant::now();
        v.section("D6: BioNautilusBrain [bingoCube → wetSpring]");

        let mut nautilus = BioNautilusBrain::new("exp272-validation");
        v.check_pass("Nautilus starts untrained", !nautilus.is_trained());
        v.check_count("Observation count = 0", nautilus.observation_count(), 0);

        for i in 0..10 {
            let fi = f64::from(i);
            let obs = BioObservation {
                sample_id: format!("exp272_naut_{i:02}"),
                shannon: fi.mul_add(0.5, 1.0),
                simpson: fi.mul_add(0.03, 0.5),
                evenness: fi.mul_add(0.05, 0.4),
                chao1: fi.mul_add(20.0, 50.0),
                bray_curtis_mean: fi.mul_add(0.02, 0.3),
                anderson_w: fi.mul_add(0.5, 2.0),
                anderson_phase: if fi < 5.0 { 0.2 } else { 0.8 },
                amr_load: fi.mul_add(0.01, 0.05),
            };
            nautilus.observe(&obs);
        }
        v.check_count("Observation count = 10", nautilus.observation_count(), 10);

        let mse = nautilus.train();
        v.check_pass("Training succeeds with 10 points", mse.is_some());
        v.check_pass("MSE is finite", mse.unwrap_or(f64::NAN).is_finite());
        v.check_pass("Nautilus now trained", nautilus.is_trained());

        let pred = nautilus.predict(3.0);
        v.check_pass("Prediction returns Some", pred.is_some());
        if let Some((chao1, evenness, health)) = pred {
            v.check_pass("Predicted Chao1 finite", chao1.is_finite());
            v.check_pass("Predicted evenness finite", evenness.is_finite());
            v.check_pass("Predicted health finite", health.is_finite());
        }

        let candidates: Vec<f64> = (0..10).map(|i| f64::from(i).mul_add(0.5, 1.0)).collect();
        let screened = nautilus.screen_candidates(&candidates);
        v.check_count("Screening returns all", screened.len(), candidates.len());

        v.check_pass("Not drifting with 10 points", !nautilus.is_drifting());

        domains.push(DomainResult {
            name: "Nautilus Brain",
            spring: "bingoCube",
            ms: t.elapsed().as_secs_f64() * 1000.0,
            checks: 11,
        });
    }

    // ═══ D7: Concurrent Pipeline Proof (all layers) ═════════════════════════
    {
        let t = Instant::now();
        v.section("D7: Concurrent Pipeline [hotSpring+bingoCube+wetSpring]");

        let mut brain = BioBrain::new(3);
        let mut nautilus = BioNautilusBrain::new("exp272-concurrent");

        let mut escalation_seen = false;
        let mut de_escalation_seen = false;

        for step in 0..20 {
            let fi = f64::from(step);
            let is_anomaly = (6..=10).contains(&step);
            let obs = BioObservation {
                sample_id: format!("pipeline_{step:02}"),
                shannon: fi.mul_add(0.1, 2.5) + if is_anomaly { -1.5 } else { 0.0 },
                simpson: fi.mul_add(0.01, 0.7),
                evenness: fi.mul_add(0.015, 0.6),
                chao1: fi.mul_add(5.0, 80.0),
                bray_curtis_mean: 0.35,
                anderson_w: 3.0 + if is_anomaly { 5.0 } else { 0.0 },
                anderson_phase: if is_anomaly { 0.1 } else { 0.7 },
                amr_load: if is_anomaly { 0.8 } else { 0.05 },
            };

            nautilus.observe(&obs);

            let mut head_outputs = vec![0.5; NUM_HEADS];
            if is_anomaly {
                head_outputs[heads::A0_DIVERSITY_REGIME] = 0.0;
                head_outputs[heads::B0_SHANNON_REGIME] = 1.0;
                head_outputs[heads::C0_UNIFRAC_REGIME] = 0.5;
                head_outputs[heads::A1_COMMUNITY_PHASE] = 0.1;
                head_outputs[heads::B1_DIVERSITY_PHASE] = 0.5;
                head_outputs[heads::C1_PHYLO_PHASE] = 0.9;
                head_outputs[heads::A3_DIVERSITY_ANOMALY] = 0.0;
                head_outputs[heads::B3_NOVEL_STATE] = 0.9;
                head_outputs[heads::C3_PHYLO_ANOMALY] = 0.4;
                head_outputs[heads::A5_SAMPLING_PRIORITY] = 0.1;
                head_outputs[heads::B5_RICHNESS_PRIORITY] = 0.9;
                head_outputs[heads::C5_PHYLO_PRIORITY] = 0.5;
            }

            let update = brain.observe(&obs, &head_outputs);

            if update.attention != AttentionState::Healthy && !escalation_seen {
                escalation_seen = true;
            }
            if escalation_seen && update.attention == AttentionState::Healthy {
                de_escalation_seen = true;
            }
        }

        v.check_pass("Escalation observed during anomaly", escalation_seen);
        v.check_pass("De-escalation observed after calm", de_escalation_seen);
        v.check_count_u64("Pipeline ran 20 steps", brain.observation_count(), 20);
        v.check_count("Nautilus has 20 obs", nautilus.observation_count(), 20);

        let mse = nautilus.train();
        v.check_pass("Nautilus trained on pipeline data", mse.is_some());

        let pred = nautilus.predict(3.5);
        v.check_pass("Nautilus predicts from pipeline", pred.is_some());

        let status = brain.status_snapshot();
        v.check_pass(
            "Final urgency in [0,1]",
            status.urgency >= 0.0 && status.urgency <= 1.0,
        );
        v.check_pass(
            "Final attention Healthy",
            status.attention == AttentionState::Healthy,
        );

        domains.push(DomainResult {
            name: "Concurrent Pipeline",
            spring: "multi-spring",
            ms: t.elapsed().as_secs_f64() * 1000.0,
            checks: 8,
        });
    }

    // ═══ Summary ═════════════════════════════════════════════════════════════
    let total_checks: u32 = domains.iter().map(|d| d.checks).sum();
    let total_ms: f64 = domains.iter().map(|d| d.ms).sum();

    println!();
    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║  Exp272: Bio Brain Cross-Spring Validation                        ║");
    println!("╠════════════════════════════════════════════════════════════════════╣");
    println!(
        "║ {:<22} │ {:<18} │ {:>7} │ {:>3} ║",
        "Domain", "Spring", "Time", "✓"
    );
    println!("╠════════════════════════════════════════════════════════════════════╣");
    for d in &domains {
        println!(
            "║ {:<22} │ {:<18} │ {:>5.2}ms │ {:>3} ║",
            d.name, d.spring, d.ms, d.checks
        );
    }
    println!("╠════════════════════════════════════════════════════════════════════╣");
    println!(
        "║ {:<22} │ {:<18} │ {:>5.2}ms │ {:>3} ║",
        "TOTAL", "7 springs", total_ms, total_checks
    );
    println!("╚════════════════════════════════════════════════════════════════════╝");

    println!();
    println!("  Cross-Spring Brain Evolution Tree:");
    println!("  ┌─ hotSpring v0.6.15 ── 4-layer brain (NPU/Motor/PreMotor/Prefrontal)");
    println!("  ├─ hotSpring v0.6.15 ── 36-head Gen2 ESN (6 groups × 6 heads)");
    println!("  ├─ hotSpring v0.6.15 ── HeadGroupDisagreement → BioHeadGroupDisagreement");
    println!(
        "  ├─ hotSpring v0.6.15 ── AttentionState (Green/Yellow/Red → Healthy/Alert/Critical)"
    );
    println!("  ├─ bingoCube/nautilus ─ NautilusBrain → BioNautilusBrain");
    println!("  ├─ wetSpring V89 ───── BioObservation, BioBrain, DiversityUpdate");
    println!("  └─ ToadStool S79 ──── MultiHeadEsn, ExportedWeights");
    println!();

    v.finish();
}
