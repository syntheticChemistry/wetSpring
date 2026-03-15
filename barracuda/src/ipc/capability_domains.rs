// SPDX-License-Identifier: AGPL-3.0-or-later
//! Capability domain definitions for wetSpring as a biomeOS science primal.
//!
//! Codifies the `ecology.*` capability namespace per the Spring-as-Niche
//! Deployment Standard. Each domain maps to a set of IPC methods backed
//! by barracuda library functions.
//!
//! Domain registration is self-knowledge only — wetSpring declares what it
//! provides, not what other primals offer. Cross-primal wiring happens at
//! runtime via Songbird capability-based discovery.

/// The primary capability domain for this primal.
pub const DOMAIN: &str = "ecology";

/// All capability domains this primal registers with Songbird.
pub const DOMAINS: &[CapabilityDomain] = &[
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn domain_is_ecology() {
        assert_eq!(DOMAIN, "ecology");
    }

    #[test]
    fn all_domains_prefixed_with_ecology() {
        for d in DOMAINS {
            assert!(
                d.name.starts_with("ecology."),
                "domain '{}' must start with 'ecology.'",
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
}
