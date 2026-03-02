// SPDX-License-Identifier: AGPL-3.0-or-later

//! S86 science workloads — spectral, graph, and sampling primitives.
//!
//! These were ungated in `ToadStool` S86 (previously hidden behind
//! `#[cfg(feature = "gpu")]` despite being pure CPU math). They are
//! routable to CPU substrates via metalForge dispatch.

use super::provenance::{BioWorkload, ShaderOrigin};
use crate::substrate::Capability;

/// Anderson spectral analysis — localization eigensolve (CPU f64).
///
/// Generates 3D Anderson lattice, runs Lanczos iteration, computes
/// level spacing statistics. Used by Tracks 1 and 4 (QS-geometry coupling).
#[must_use]
pub fn anderson_spectral() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named("anderson_spectral", vec![Capability::F64Compute])
        .with_primitive("anderson_3d + lanczos + level_spacing_ratio")
}

/// Hofstadter butterfly — quasiperiodic spectral structure (CPU f64).
///
/// Computes the Hofstadter butterfly spectrum (almost-Mathieu operator)
/// for validating quasiperiodic Anderson models.
#[must_use]
pub fn hofstadter_butterfly() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named("hofstadter_butterfly", vec![Capability::F64Compute])
        .with_primitive("hofstadter_butterfly + almost_mathieu")
}

/// Graph Laplacian + effective rank (CPU f64).
///
/// Builds graph Laplacian from adjacency, computes effective rank
/// (Shannon entropy of normalized eigenvalues). Used by NMF and
/// community detection pipelines.
#[must_use]
pub fn graph_laplacian() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named("graph_laplacian", vec![Capability::F64Compute])
        .with_primitive("graph_laplacian + effective_rank")
}

/// Belief propagation on disordered graphs (CPU f64).
///
/// Message-passing algorithm on disordered Laplacian.
/// Used by Anderson localization models for community network analysis.
#[must_use]
pub fn belief_propagation() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named("belief_propagation", vec![Capability::F64Compute])
        .with_primitive("belief_propagation + disordered_laplacian")
}

/// Boltzmann sampling (CPU f64).
///
/// Gibbs/Metropolis MCMC sampling for configuration-space exploration.
/// Cross-spring origin: wateringHole.
#[must_use]
pub fn boltzmann_sampling() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named("boltzmann_sampling", vec![Capability::F64Compute])
        .with_primitive("boltzmann_sampling")
}

/// Latin Hypercube + Sobol sampling (CPU f64).
///
/// Space-filling sampling for parameter sweeps and sensitivity analysis.
/// Cross-spring origin: wateringHole + airSpring.
#[must_use]
pub fn space_filling_sampling() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named("space_filling_sampling", vec![Capability::F64Compute])
        .with_primitive("latin_hypercube + sobol_scaled")
}

/// Hydrology ET₀ (6 methods) — CPU f64 compute.
///
/// Thornthwaite, Makkink, Turc, Hamon, Hargreaves, FAO-56 PM.
/// Cross-spring origin: airSpring.
#[must_use]
pub fn hydrology_et0() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named("hydrology_et0", vec![Capability::F64Compute])
        .with_primitive(
            "thornthwaite_et0 + makkink_et0 + turc_et0 + hamon_et0 + hargreaves_et0 + fao56_pm_et0",
        )
}
