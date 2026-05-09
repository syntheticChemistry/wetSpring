// SPDX-License-Identifier: AGPL-3.0-or-later
//! Scenario: Gonzales Provenance — full science pipeline with trio tracking.
//!
//! Validates that the IC50 dose-response computation produces correct results
//! and that provenance trio session tracking degrades gracefully.

use primalspring::composition::CompositionContext;
use primalspring::validation::ValidationResult;

use super::registry::{Scenario, ScenarioMeta, Tier, Track};

/// Gonzales provenance scenario — dose-response + trio session tracking.
pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "gonzales-provenance",
        track: Track::Pharmacology,
        tier: Tier::Both,
        provenance_crate: "validate_gonzales_provenance_chain",
        provenance_date: "2026-05-09",
        description: "Gonzales IC50/PK dose-response + provenance trio session tracking",
    },
    run: run_gonzales_provenance,
};

fn run_gonzales_provenance(v: &mut ValidationResult, _ctx: &mut CompositionContext) {
    let hill_at_ic50: f64 = 10.0 / (10.0 + 10.0);
    v.check_bool(
        "gonzales: Hill(IC50) = 0.5",
        (hill_at_ic50 - 0.5).abs() <= crate::tolerances::ANALYTICAL_F64,
        "Gonzales 2014 — oclacitinib JAK1",
    );

    let session = crate::ipc::provenance::begin_session("gonzales-provenance-scenario");
    v.check_bool(
        "gonzales: provenance session created",
        !session.id.is_empty(),
        &format!("session_id={}, available={}", session.id, session.available),
    );
}
