// SPDX-License-Identifier: AGPL-3.0-or-later
//! Benchmark registry — parallel to [`super::registry`] for timing/performance scenarios.

use super::registry::{ScenarioMeta, Tier, Track};
use primalspring::composition::CompositionContext;
use primalspring::validation::ValidationResult;

/// A benchmark scenario: metadata + run function that reports timing data.
pub struct BenchmarkScenario {
    /// Classification and provenance.
    pub meta: ScenarioMeta,
    /// Run function receiving `ValidationResult` and `CompositionContext`.
    pub run: fn(&mut ValidationResult, &mut CompositionContext),
}

/// Registry of all benchmark scenarios.
pub struct BenchmarkRegistry {
    benchmarks: Vec<BenchmarkScenario>,
}

impl BenchmarkRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            benchmarks: Vec::new(),
        }
    }

    /// Register a benchmark.
    pub fn register(&mut self, benchmark: BenchmarkScenario) {
        self.benchmarks.push(benchmark);
    }

    /// All registered benchmarks.
    #[must_use]
    pub fn all(&self) -> &[BenchmarkScenario] {
        &self.benchmarks
    }

    /// Filter benchmarks by tier.
    pub fn filter_by_tier(&self, tier: Tier) -> impl Iterator<Item = &BenchmarkScenario> {
        self.benchmarks
            .iter()
            .filter(move |b| b.meta.tier == tier || b.meta.tier == Tier::Both || tier == Tier::Both)
    }

    /// Filter benchmarks by track.
    pub fn filter_by_track(&self, track: Track) -> impl Iterator<Item = &BenchmarkScenario> {
        self.benchmarks
            .iter()
            .filter(move |b| b.meta.track == track)
    }

    /// Find a benchmark by id.
    #[must_use]
    pub fn find(&self, id: &str) -> Option<&BenchmarkScenario> {
        self.benchmarks.iter().find(|b| b.meta.id == id)
    }

    /// Number of registered benchmarks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.benchmarks.len()
    }

    /// Whether the registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.benchmarks.is_empty()
    }
}

impl Default for BenchmarkRegistry {
    fn default() -> Self {
        Self::new()
    }
}
