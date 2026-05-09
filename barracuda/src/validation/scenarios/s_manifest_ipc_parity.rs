// SPDX-License-Identifier: AGPL-3.0-or-later
//! Scenario: Manifest IPC Parity — 15 downstream validation capabilities.

use primalspring::composition::CompositionContext;
use primalspring::validation::ValidationResult;

use super::registry::{Scenario, ScenarioMeta, Tier, Track};

/// Manifest IPC parity scenario — validates 15 downstream capabilities live.
pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "manifest-ipc-parity",
        track: Track::Composition,
        tier: Tier::Live,
        provenance_crate: "wetspring_guidestone",
        provenance_date: "2026-05-09",
        description: "15 downstream manifest capabilities vs live barraCuda IPC — Tier 2",
    },
    run: run_manifest_ipc,
};

fn run_manifest_ipc(v: &mut ValidationResult, ctx: &mut CompositionContext) {
    crate::certification::health::validate_manifest_ipc(ctx, v);
    crate::certification::health::validate_domain_science(ctx, v);
}
