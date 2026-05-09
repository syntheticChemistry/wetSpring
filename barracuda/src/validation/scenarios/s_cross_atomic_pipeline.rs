// SPDX-License-Identifier: AGPL-3.0-or-later
//! Scenario: Cross-Atomic Pipeline — hash → store → retrieve → verify.

use primalspring::composition::CompositionContext;
use primalspring::validation::ValidationResult;

use super::registry::{Scenario, ScenarioMeta, Tier, Track};

/// Cross-atomic pipeline scenario — BearDog hash → NestGate store → retrieve.
pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "cross-atomic-pipeline",
        track: Track::Pipeline,
        tier: Tier::Live,
        provenance_crate: "wetspring_guidestone",
        provenance_date: "2026-05-09",
        description: "BearDog→NestGate cross-atomic integrity pipeline — Tier 2",
    },
    run: run_cross_atomic,
};

fn run_cross_atomic(v: &mut ValidationResult, ctx: &mut CompositionContext) {
    crate::certification::health::validate_cross_atomic(ctx, v);
}
