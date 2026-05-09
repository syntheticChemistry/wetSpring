// SPDX-License-Identifier: AGPL-3.0-or-later
//! Scenario registry — types for two-tier validation scenarios.
//!
//! Mirrors `primalSpring::validation::scenarios::registry` for ecosystem
//! consistency. Each scenario carries `ScenarioMeta` provenance and a
//! run function that exercises a specific composition behavior.

use primalspring::composition::CompositionContext;
use primalspring::validation::ValidationResult;

/// Validation tier: determines what infrastructure a scenario needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// Tier 1: Pure Rust, no IPC. Safe for CI and bare environments.
    Rust,
    /// Tier 2: Requires deployed primals from plasmidBin.
    Live,
    /// Runs in both tiers.
    Both,
}

impl std::fmt::Display for Tier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rust => write!(f, "rust"),
            Self::Live => write!(f, "live"),
            Self::Both => write!(f, "both"),
        }
    }
}

impl Tier {
    /// Parse a tier string loosely (case-insensitive).
    #[must_use]
    pub fn from_str_loose(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "rust" | "tier1" | "t1" | "structural" => Some(Self::Rust),
            "live" | "tier2" | "t2" | "ipc" => Some(Self::Live),
            "both" | "all" => Some(Self::Both),
            _ => None,
        }
    }
}

/// Scenario track — groups related scenarios by domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Track {
    /// Core science baselines (diversity, stats, alignment).
    Science,
    /// Pharmacology and dose-response (Gonzales papers).
    Pharmacology,
    /// NUCLEUS composition and IPC parity.
    Composition,
    /// Cross-atomic pipeline and provenance.
    Pipeline,
}

impl std::fmt::Display for Track {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Science => write!(f, "science"),
            Self::Pharmacology => write!(f, "pharmacology"),
            Self::Composition => write!(f, "composition"),
            Self::Pipeline => write!(f, "pipeline"),
        }
    }
}

impl Track {
    /// Parse a track string loosely.
    #[must_use]
    pub fn from_str_loose(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "science" | "sci" | "core" => Some(Self::Science),
            "pharmacology" | "pharma" | "gonzales" => Some(Self::Pharmacology),
            "composition" | "comp" | "nucleus" => Some(Self::Composition),
            "pipeline" | "pipe" | "cross-atomic" => Some(Self::Pipeline),
            _ => None,
        }
    }
}

/// Scenario metadata — provenance, classification, and description.
#[derive(Debug, Clone)]
pub struct ScenarioMeta {
    /// Unique scenario identifier (e.g. `"bare-science"`).
    pub id: &'static str,
    /// Which track this scenario belongs to.
    pub track: Track,
    /// Which validation tier this scenario exercises.
    pub tier: Tier,
    /// Original experiment crate/binary for provenance.
    pub provenance_crate: &'static str,
    /// Date of last significant update.
    pub provenance_date: &'static str,
    /// One-line description.
    pub description: &'static str,
}

/// A validation scenario: metadata + run function.
pub struct Scenario {
    /// Classification and provenance.
    pub meta: ScenarioMeta,
    /// Run function. Receives a mutable `ValidationResult` and
    /// `CompositionContext` (the context may be unused for Tier 1).
    pub run: fn(&mut ValidationResult, &mut CompositionContext),
}

/// Registry of all validation scenarios.
pub struct ScenarioRegistry {
    scenarios: Vec<Scenario>,
}

impl ScenarioRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            scenarios: Vec::new(),
        }
    }

    /// Register a scenario.
    pub fn register(&mut self, scenario: Scenario) {
        self.scenarios.push(scenario);
    }

    /// All registered scenarios.
    #[must_use]
    pub fn all(&self) -> &[Scenario] {
        &self.scenarios
    }

    /// Filter scenarios by tier.
    pub fn filter_by_tier(&self, tier: Tier) -> impl Iterator<Item = &Scenario> {
        self.scenarios
            .iter()
            .filter(move |s| s.meta.tier == tier || s.meta.tier == Tier::Both || tier == Tier::Both)
    }

    /// Filter scenarios by track.
    pub fn filter_by_track(&self, track: Track) -> impl Iterator<Item = &Scenario> {
        self.scenarios.iter().filter(move |s| s.meta.track == track)
    }

    /// Find a scenario by id.
    #[must_use]
    pub fn find(&self, id: &str) -> Option<&Scenario> {
        self.scenarios.iter().find(|s| s.meta.id == id)
    }

    /// Number of registered scenarios.
    #[must_use]
    pub fn len(&self) -> usize {
        self.scenarios.len()
    }

    /// Whether the registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.scenarios.is_empty()
    }
}

impl Default for ScenarioRegistry {
    fn default() -> Self {
        Self::new()
    }
}
