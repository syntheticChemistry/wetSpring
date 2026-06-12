// SPDX-License-Identifier: AGPL-3.0-or-later
//! Scenario: NUCLEUS Mesh Health — 13-primal liveness + version skew detection.

use primalspring::composition::CompositionContext;
use primalspring::validation::ValidationResult;

use super::registry::{Scenario, ScenarioMeta, Tier, Track};

/// Mesh health scenario — probes all 13 NUCLEUS primals for liveness,
/// version reporting, and version skew. Tier 2 (requires deployed primals).
pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "mesh-health",
        track: Track::Composition,
        tier: Tier::Live,
        provenance_crate: "wetspring_certification",
        provenance_date: "2026-06-12",
        description: "NUCLEUS 13/13 mesh health audit + version skew detection — Tier 2",
    },
    run: run_mesh_health,
};

fn run_mesh_health(v: &mut ValidationResult, _ctx: &mut CompositionContext) {
    crate::certification::health::validate_mesh_health(v);
}
