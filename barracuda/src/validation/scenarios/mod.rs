// SPDX-License-Identifier: AGPL-3.0-or-later
//! Two-tier validation scenarios for wetSpring.
//!
//! Each scenario is a self-contained validation function with `ScenarioMeta`
//! provenance, following the primalSpring eukaryotic pattern. Scenarios are
//! registered in [`build_registry`] and executed by the UniBin `validate`
//! subcommand.
//!
//! # Tiers
//!
//! - **Tier 1 (Rust)**: Pure structural validation, no IPC. Safe for CI.
//! - **Tier 2 (Live)**: Requires deployed primals from plasmidBin.

pub mod benchmark_registry;
pub mod registry;

pub use benchmark_registry::{BenchmarkRegistry, BenchmarkScenario};
pub use registry::{Scenario, ScenarioMeta, ScenarioRegistry, Tier, Track};

pub mod s_bare_science;
pub mod s_cross_atomic_pipeline;
pub mod s_gonzales_provenance;
pub mod s_manifest_ipc_parity;

/// Build the complete scenario registry.
#[must_use]
pub fn build_registry() -> ScenarioRegistry {
    let mut r = ScenarioRegistry::new();
    r.register(s_bare_science::SCENARIO);
    r.register(s_manifest_ipc_parity::SCENARIO);
    r.register(s_cross_atomic_pipeline::SCENARIO);
    r.register(s_gonzales_provenance::SCENARIO);
    super::experiments::register_all(&mut r);
    r
}

/// Build the benchmark registry.
#[must_use]
pub fn build_benchmark_registry() -> BenchmarkRegistry {
    BenchmarkRegistry::new()
}
