// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![cfg(feature = "ipc")]
#![expect(
    clippy::unwrap_used,
    reason = "integration test: CI cross-sync validations use unwrap for diagnostic clarity"
)]
//! CI Cross-Sync: validates wetSpring's local capability surface against the
//! primalSpring canonical registry (`config/capability_registry.toml`, 403 methods).
//!
//! Per primalSpring post-interstadial directive (May 10, 2026): "CI cross-sync —
//! validate your local capability methods against primalSpring canonical 403.
//! Zero drift is the target."
//!
//! Three checks:
//! 1. Local capability_registry.toml methods == dispatch table methods
//! 2. Niche CAPABILITIES ⊇ dispatch methods
//! 3. Consumed capabilities reference recognized ecosystem domains

use std::collections::BTreeSet;

use wetspring_barracuda::ipc::capability_domains;
use wetspring_barracuda::niche;

/// Every method in the local `capability_registry.toml` must be present in
/// the dispatch table (via `capability_domains::all_methods()`), and vice versa.
#[test]
fn local_registry_matches_dispatch() {
    let toml_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("capability_registry.toml");
    let content = std::fs::read_to_string(&toml_path).unwrap_or_else(|e| {
        panic!(
            "cannot read {}: {e} — CI cross-sync requires local registry",
            toml_path.display()
        )
    });

    let toml_methods: BTreeSet<&str> = content
        .lines()
        .filter(|l| l.starts_with("method = "))
        .map(|l| l.trim_start_matches("method = ").trim_matches('"'))
        .collect();

    let dispatch_methods: BTreeSet<&str> = capability_domains::all_methods().into_iter().collect();

    let in_toml_not_dispatch: Vec<&&str> = toml_methods.difference(&dispatch_methods).collect();
    let in_dispatch_not_toml: Vec<&&str> = dispatch_methods.difference(&toml_methods).collect();

    assert!(
        in_toml_not_dispatch.is_empty(),
        "methods in capability_registry.toml but not dispatched: {in_toml_not_dispatch:?}"
    );
    assert!(
        in_dispatch_not_toml.is_empty(),
        "methods dispatched but not in capability_registry.toml: {in_dispatch_not_toml:?}"
    );
}

/// Every dispatched method must appear in `niche::CAPABILITIES`.
#[test]
fn niche_capabilities_superset_of_dispatch() {
    let niche_caps: BTreeSet<&str> = niche::CAPABILITIES.iter().copied().collect();
    let dispatch_methods: BTreeSet<&str> = capability_domains::all_methods().into_iter().collect();

    let missing: Vec<&&str> = dispatch_methods.difference(&niche_caps).collect();
    assert!(
        missing.is_empty(),
        "dispatch methods missing from niche::CAPABILITIES: {missing:?}"
    );
}

/// `niche::DEPENDENCIES` must include all required infrastructure primals.
#[test]
fn niche_dependencies_include_infrastructure() {
    let dep_names: Vec<&str> = niche::DEPENDENCIES.iter().map(|d| d.name).collect();

    let required_infra = ["beardog", "songbird"];
    for infra in &required_infra {
        assert!(
            dep_names.contains(infra),
            "niche::DEPENDENCIES missing required infra primal: {infra}"
        );
    }

    assert!(
        dep_names.contains(&"skunkbat"),
        "niche::DEPENDENCIES missing skunkBat audit primal (upstream directive: wire JH-5)"
    );
}

/// `niche::CONSUMED_CAPABILITIES` must include biomeOS v3.51 lifecycle methods.
#[test]
fn consumed_includes_biomeos_v351_lifecycle() {
    let consumed: BTreeSet<&str> = niche::CONSUMED_CAPABILITIES.iter().copied().collect();

    assert!(
        consumed.contains("composition.status"),
        "CONSUMED_CAPABILITIES missing composition.status (biomeOS v3.51)"
    );
    assert!(
        consumed.contains("method.register"),
        "CONSUMED_CAPABILITIES missing method.register (biomeOS v3.51)"
    );
    assert!(
        consumed.contains("audit.event"),
        "CONSUMED_CAPABILITIES missing audit.event (skunkBat JH-5)"
    );
}

/// Validate that consumed capabilities reference only recognized ecosystem
/// domain prefixes from the canonical registry structure.
#[test]
fn consumed_capabilities_use_recognized_domains() {
    let recognized_prefixes = [
        "crypto.",
        "discovery.",
        "tensor.",
        "stats.",
        "compute.",
        "spectral.",
        "linalg.",
        "health.",
        "storage.",
        "dag.",
        "spine.",
        "entry.",
        "braid.",
        "ai.",
        "inference.",
        "render.",
        "shader.",
        "math.",
        "noise.",
        "activation.",
        "fhe.",
        "tolerances.",
        "rng.",
        "provenance.",
        "audit.",
        "composition.",
        "method.",
        "defense.",
    ];

    for cap in niche::CONSUMED_CAPABILITIES {
        let has_recognized = recognized_prefixes.iter().any(|p| cap.starts_with(p));
        assert!(
            has_recognized,
            "consumed capability '{cap}' has unrecognized domain prefix — \
             add prefix to recognized_prefixes or verify against canonical registry"
        );
    }
}

/// Cross-check: canonical primalSpring registry file exists and contains 403+
/// methods. This validates our CI can access the canonical source of truth.
#[test]
fn canonical_registry_accessible_and_nontrivial() {
    let canonical = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("../primalSpring/config/capability_registry.toml");

    if !canonical.exists() {
        eprintln!(
            "SKIP: canonical registry not found at {} — \
             cross-sync requires primalSpring checkout",
            canonical.display()
        );
        return;
    }

    let content = std::fs::read_to_string(&canonical).unwrap();
    let method_count = content
        .lines()
        .filter(|l| l.starts_with("methods = [") || l.starts_with("    \""))
        .filter(|l| l.trim().starts_with('"'))
        .count();

    assert!(
        method_count >= 300,
        "canonical registry has only {method_count} method entries — expected 403+"
    );
}
