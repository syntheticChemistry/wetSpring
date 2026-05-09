// SPDX-License-Identifier: AGPL-3.0-or-later
//! Certification engine for wetSpring — layered composition correctness.
//!
//! Absorbs the guidestone layers into a library module following the
//! primalSpring eukaryotic organelle pattern. The `certify` function
//! drives sequential layer validation with early exit for bare mode.
//!
//! # Layers
//!
//! | Layer | Name | Requires |
//! |-------|------|----------|
//! | 0 | Bare Science Baselines | Nothing |
//! | 1 | Tolerance Provenance | Nothing |
//! | 2 | Checksum Verification | Nothing |
//! | 3 | NUCLEUS Liveness | Deployed primals |
//! | 4 | Manifest IPC Parity | Live barraCuda |
//! | 5 | Domain Science IPC | Live barraCuda |
//! | 6 | Cross-Atomic Pipeline | Live NUCLEUS |

pub mod bare;
pub mod health;

use primalspring::composition::{CompositionContext, validate_liveness};
use primalspring::validation::ValidationResult;

/// Maximum certification layer (inclusive).
pub const MAX_LAYER: u8 = 6;

/// Run layered certification up to `max_layer`.
///
/// Layers 0–2 are bare (no primals needed). Layers 3–6 require a live
/// NUCLEUS deployment. Returns a `ValidationResult` with pass/fail/skip
/// counts and an appropriate exit code.
#[must_use]
pub fn certify(max_layer: u8) -> ValidationResult {
    let mut v = ValidationResult::new("wetSpring Certification — Life Science Composition Proof");
    ValidationResult::print_banner("wetSpring Certification — Life Science Composition Proof");

    // Layer 0: Bare Science Baselines
    v.section("Layer 0: Bare Science Baselines");
    bare::validate_bare_science(&mut v);

    if max_layer == 0 {
        v.finish();
        return v;
    }

    // Layer 1: Tolerance Provenance
    v.section("Layer 1: Tolerance Provenance");
    bare::validate_tolerance_provenance(&mut v);

    if max_layer <= 1 {
        v.finish();
        return v;
    }

    // Layer 2: Checksum Verification
    v.section("Layer 2: Checksum Verification");
    bare::validate_checksums(&mut v);

    if max_layer <= 2 {
        v.finish();
        return v;
    }

    // Layer 3: NUCLEUS Liveness (requires deployed primals)
    v.section("Layer 3: NUCLEUS Liveness");
    let mut ctx = CompositionContext::from_live_discovery_with_fallback();
    let alive = validate_liveness(
        &mut ctx,
        &mut v,
        &["tensor", "security", "storage", "compute", "ai"],
    );

    if alive == 0 {
        tracing::warn!(
            "No NUCLEUS primals discovered — bare properties validated. Deploy from plasmidBin and rerun for full certification."
        );
        v.finish();
        return v;
    }

    if max_layer <= 3 {
        v.finish();
        return v;
    }

    // Layer 4: Manifest IPC Parity
    v.section("Layer 4: Manifest IPC Parity (15 validation_capabilities)");
    health::validate_manifest_ipc(&mut ctx, &mut v);

    if max_layer <= 4 {
        v.finish();
        return v;
    }

    // Layer 5: Domain Science IPC
    v.section("Layer 5: Domain Science IPC");
    health::validate_domain_science(&mut ctx, &mut v);

    if max_layer <= 5 {
        v.finish();
        return v;
    }

    // Layer 6: Cross-Atomic Pipeline
    v.section("Layer 6: Cross-Atomic Pipeline");
    health::validate_cross_atomic(&mut ctx, &mut v);

    v.finish();
    v
}
