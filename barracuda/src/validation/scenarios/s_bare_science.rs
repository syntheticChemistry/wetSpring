// SPDX-License-Identifier: AGPL-3.0-or-later
//! Scenario: Bare Science — deterministic baselines with no IPC.

use primalspring::composition::CompositionContext;
use primalspring::validation::ValidationResult;

use super::registry::{Scenario, ScenarioMeta, Tier, Track};

/// Bare science scenario — deterministic baselines, tolerances, checksums.
pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "bare-science",
        track: Track::Science,
        tier: Tier::Rust,
        provenance_crate: "wetspring_guidestone",
        provenance_date: "2026-05-09",
        description: "Deterministic science baselines (Shannon, Hill, stats, matmul) — Tier 1",
    },
    run: run_bare_science,
};

fn run_bare_science(v: &mut ValidationResult, _ctx: &mut CompositionContext) {
    crate::certification::bare::validate_bare_science(v);
    crate::certification::bare::validate_tolerance_provenance(v);
    crate::certification::bare::validate_checksums(v);
}
