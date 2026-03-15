// SPDX-License-Identifier: AGPL-3.0-or-later
//! Capability domain definitions for wetSpring as a biomeOS science primal.
//!
//! Codifies the full capability namespace per the Spring-as-Niche Deployment
//! Standard. Domains span ecology (science), provenance trio, brain sentinel,
//! and server metrics. Each domain maps to a set of IPC methods backed by
//! barracuda library functions.
//!
//! Domain registration is self-knowledge only — wetSpring declares what it
//! provides, not what other primals offer. Cross-primal wiring happens at
//! runtime via Songbird capability-based discovery.

/// The primary capability domain for this primal.
pub const DOMAIN: &str = "ecology";

/// Recognised domain prefixes for validation.
#[cfg(test)]
const VALID_DOMAIN_PREFIXES: &[&str] = &["ecology.", "provenance", "brain", "metrics"];

/// All capability domains this primal registers with Songbird.
///
/// Covers 4 domain families (19 capabilities total):
/// - `ecology.*`   — 12 science capabilities (diversity, ODE, alignment, …)
/// - `provenance`  — 3 provenance-trio lifecycle methods
/// - `brain`       — 3 neural sentinel methods
/// - `metrics`     — 1 server metrics capability
pub const DOMAINS: &[CapabilityDomain] = &[
    // ── ecology (science) ───────────────────────────────────────────
    CapabilityDomain {
        name: "ecology.diversity",
        description: "Alpha/beta diversity metrics (Shannon, Simpson, Chao1, Bray-Curtis)",
        methods: &["science.diversity"],
    },
    CapabilityDomain {
        name: "ecology.qs_model",
        description: "Quorum-sensing biofilm ODE models (Waters 2008, Fernandez 2020)",
        methods: &["science.qs_model"],
    },
    CapabilityDomain {
        name: "ecology.anderson",
        description: "Anderson spectral disorder analysis (2D/3D lattice)",
        methods: &["science.anderson"],
    },
    CapabilityDomain {
        name: "ecology.kinetics",
        description: "Biogas production curve fitting (Gompertz, first-order, Monod, Haldane)",
        methods: &["science.kinetics"],
    },
    CapabilityDomain {
        name: "ecology.alignment",
        description: "Smith-Waterman local sequence alignment",
        methods: &["science.alignment"],
    },
    CapabilityDomain {
        name: "ecology.taxonomy",
        description: "Naive Bayes k-mer classification (RDP-style)",
        methods: &["science.taxonomy"],
    },
    CapabilityDomain {
        name: "ecology.phylogenetics",
        description: "Robinson-Foulds tree distance and phylogenetic placement",
        methods: &["science.phylogenetics"],
    },
    CapabilityDomain {
        name: "ecology.nmf",
        description: "Non-negative Matrix Factorization for drug repurposing",
        methods: &["science.nmf"],
    },
    CapabilityDomain {
        name: "ecology.timeseries",
        description: "Cross-spring time series analysis and diversity tracking",
        methods: &["science.timeseries", "science.timeseries_diversity"],
    },
    CapabilityDomain {
        name: "ecology.ncbi",
        description: "NCBI sequence retrieval (E-search, EFetch, SRA)",
        methods: &["science.ncbi_fetch"],
    },
    CapabilityDomain {
        name: "ecology.pipeline",
        description: "End-to-end 16S amplicon analysis pipeline",
        methods: &["science.full_pipeline"],
    },
    // ── provenance trio ─────────────────────────────────────────────
    CapabilityDomain {
        name: "provenance",
        description: "Provenance-tracked session lifecycle (begin, record, complete)",
        methods: &[
            "provenance.begin",
            "provenance.record",
            "provenance.complete",
        ],
    },
    // ── brain sentinel ──────────────────────────────────────────────
    CapabilityDomain {
        name: "brain",
        description: "Neural observation, attention weighting, and urgency scoring",
        methods: &["brain.observe", "brain.attention", "brain.urgency"],
    },
    // ── metrics ─────────────────────────────────────────────────────
    CapabilityDomain {
        name: "metrics",
        description: "Server metrics snapshot (calls, errors, latency)",
        methods: &["metrics.snapshot"],
    },
];

/// A capability domain grouping related IPC methods.
pub struct CapabilityDomain {
    /// Domain identifier (e.g. `ecology.diversity`).
    pub name: &'static str,
    /// Human-readable description.
    pub description: &'static str,
    /// IPC method names belonging to this domain.
    pub methods: &'static [&'static str],
}

/// Build the Songbird registration payload for this primal's capabilities.
#[must_use]
pub fn registration_domains() -> Vec<(&'static str, &'static str)> {
    DOMAINS
        .iter()
        .map(|d| (d.name, d.description))
        .collect()
}

/// Flat list of every IPC method this primal supports, derived from [`DOMAINS`].
#[must_use]
pub fn all_methods() -> Vec<&'static str> {
    DOMAINS.iter().flat_map(|d| d.methods.iter().copied()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn domain_is_ecology() {
        assert_eq!(DOMAIN, "ecology");
    }

    #[test]
    fn all_domains_have_recognised_prefix() {
        for d in DOMAINS {
            assert!(
                VALID_DOMAIN_PREFIXES.iter().any(|p| d.name.starts_with(p)),
                "domain '{}' does not match any recognised prefix",
                d.name
            );
        }
    }

    #[test]
    fn no_empty_method_lists() {
        for d in DOMAINS {
            assert!(
                !d.methods.is_empty(),
                "domain '{}' has no methods",
                d.name
            );
        }
    }

    #[test]
    fn registration_covers_all_domains() {
        let reg = registration_domains();
        assert_eq!(reg.len(), DOMAINS.len());
    }

    #[test]
    fn domains_cover_all_four_families() {
        let names: Vec<&str> = DOMAINS.iter().map(|d| d.name).collect();
        assert!(names.iter().any(|n| n.starts_with("ecology.")));
        assert!(names.contains(&"provenance"));
        assert!(names.contains(&"brain"));
        assert!(names.contains(&"metrics"));
    }

    #[test]
    fn total_capability_count_matches_registry() {
        assert_eq!(DOMAINS.len(), 14, "14 domains (11 ecology + provenance + brain + metrics)");
        let total_methods: usize = DOMAINS.iter().map(|d| d.methods.len()).sum();
        assert_eq!(total_methods, 19, "19 total capability methods");
    }

    #[test]
    fn all_methods_returns_flat_list() {
        let methods = all_methods();
        assert_eq!(methods.len(), 19);
        assert!(methods.contains(&"science.diversity"));
        assert!(methods.contains(&"provenance.begin"));
        assert!(methods.contains(&"brain.observe"));
        assert!(methods.contains(&"metrics.snapshot"));
    }
}
