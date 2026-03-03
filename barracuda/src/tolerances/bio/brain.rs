// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bio Brain tolerances: head-group disagreement, urgency, and attention.
//!
//! The 4-layer bio brain pipeline (adapted from hotSpring physics brain)
//! computes disagreement across 36 attention heads grouped into regime,
//! phase, and anomaly tiers. These tolerances cover the analytical checks
//! on uniform/divergent head outputs.

/// Head-group disagreement analytical tolerance.
///
/// `BioHeadGroupDisagreement::from_outputs` on uniform head vectors
/// computes `max - min` within each group. For identical inputs this is
/// pure f64 subtraction of equal values — result should be exactly 0.0.
/// `ANALYTICAL_F64` (1e-12) is sufficient, but we use 1e-10 to match
/// hotSpring's physics brain convention and allow minor group-boundary
/// rounding when head counts are not multiples of 3.
/// Validated: Exp272 (Bio Brain Cross-Spring), Phase 79, 2026-02-25.
pub const BRAIN_DISAGREEMENT_ANALYTICAL: f64 = 1e-10;

/// Bio brain urgency tolerance.
///
/// `urgency()` is a weighted sum of `delta_regime`, `delta_anomaly`, and
/// `delta_phase`, each in [0, 1]. For uniform heads the urgency should be
/// 0.0; this tolerance covers accumulated f64 rounding across the
/// 36-head aggregation (12 heads × 3 groups × weighted sum).
/// Validated: Exp272 (Bio Brain Cross-Spring), Phase 79, 2026-02-25.
pub const BRAIN_URGENCY_TOL: f64 = 0.01;
