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
const VALID_DOMAIN_PREFIXES: &[&str] = &[
    "ecology.",
    "health",
    "provenance",
    "brain",
    "metrics",
    "data",
    "vault",
    "composition",
];

/// All capability domains this primal registers with Songbird.
///
/// Covers 8 domain families (41 capabilities total):
/// - `ecology.*`      — 14 science capabilities (diversity, ODE, alignment, AI, …)
/// - `health`         — 3 health probes (check, liveness, readiness)
/// - `provenance`     — 3 provenance-trio lifecycle methods
/// - `brain`          — 3 neural sentinel methods
/// - `metrics`        — 1 server metrics capability
/// - `data`           — 3 data ingestion methods (ChEMBL, PubChem, table)
/// - `vault`          — 3 consent-gated storage methods
/// - `composition`    — 5 NUCLEUS composition health probes
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
    CapabilityDomain {
        name: "ecology.gonzales",
        description: "Gonzales dermatitis: IC50 dose-response, PK decay, tissue lattice (Papers 53-58)",
        methods: &[
            "science.gonzales.dose_response",
            "science.gonzales.pk_decay",
            "science.gonzales.tissue_lattice",
        ],
    },
    CapabilityDomain {
        name: "ecology.anderson_exploration",
        description: "Anderson localization exploration: biome atlas, disorder sweep, hormesis, cross-species",
        methods: &[
            "science.anderson.biome_atlas",
            "science.anderson.disorder_sweep",
            "science.anderson.hormesis",
            "science.anderson.cross_species",
        ],
    },
    // ── health probes ──────────────────────────────────────────────
    CapabilityDomain {
        name: "health",
        description: "Liveness and readiness probes for biomeOS orchestration",
        methods: &["health.check", "health.liveness", "health.readiness"],
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
    // ── AI assist (Squirrel) ────────────────────────────────────────
    CapabilityDomain {
        name: "ecology.ai_assist",
        description: "AI-assisted diversity interpretation and parameter exploration",
        methods: &["ai.ecology_interpret"],
    },
    // ── data ingestion ──────────────────────────────────────────────
    CapabilityDomain {
        name: "data",
        description: "External data ingestion (ChEMBL, PubChem, table registration)",
        methods: &[
            "data.fetch.chembl",
            "data.fetch.pubchem",
            "data.fetch.register_table",
        ],
    },
    // ── vault (consent-gated storage) ───────────────────────────────
    CapabilityDomain {
        name: "vault",
        description: "Consent-gated encrypted storage via NestGate integration",
        methods: &["vault.store", "vault.retrieve", "vault.consent.verify"],
    },
    // ── composition health ──────────────────────────────────────────
    CapabilityDomain {
        name: "composition",
        description: "Cross-spring NUCLEUS composition health probes (Tower, Node, Nest)",
        methods: &[
            "composition.science_health",
            "composition.tower_health",
            "composition.node_health",
            "composition.nest_health",
            "composition.nucleus_health",
        ],
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
    DOMAINS.iter().map(|d| (d.name, d.description)).collect()
}

/// Flat list of every IPC method this primal supports, derived from [`DOMAINS`].
#[must_use]
pub fn all_methods() -> Vec<&'static str> {
    DOMAINS
        .iter()
        .flat_map(|d| d.methods.iter().copied())
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
            assert!(!d.methods.is_empty(), "domain '{}' has no methods", d.name);
        }
    }

    #[test]
    fn registration_covers_all_domains() {
        let reg = registration_domains();
        assert_eq!(reg.len(), DOMAINS.len());
    }

    #[test]
    fn domains_cover_all_eight_families() {
        let names: Vec<&str> = DOMAINS.iter().map(|d| d.name).collect();
        assert!(names.iter().any(|n| n.starts_with("ecology.")));
        assert!(names.contains(&"health"));
        assert!(names.contains(&"provenance"));
        assert!(names.contains(&"brain"));
        assert!(names.contains(&"metrics"));
        assert!(names.contains(&"data"));
        assert!(names.contains(&"vault"));
        assert!(names.contains(&"composition"));
    }

    #[test]
    fn total_capability_count_matches_registry() {
        assert_eq!(
            DOMAINS.len(),
            21,
            "21 domains (13 ecology + health + provenance + brain + metrics + ai_assist + data + vault + composition)"
        );
        let total_methods: usize = DOMAINS.iter().map(|d| d.methods.len()).sum();
        assert_eq!(total_methods, 41, "41 total capability methods");
    }

    #[test]
    fn all_methods_returns_flat_list() {
        let methods = all_methods();
        assert_eq!(methods.len(), 41);
        assert!(methods.contains(&"science.diversity"));
        assert!(methods.contains(&"health.liveness"));
        assert!(methods.contains(&"health.readiness"));
        assert!(methods.contains(&"provenance.begin"));
        assert!(methods.contains(&"brain.observe"));
        assert!(methods.contains(&"metrics.snapshot"));
        assert!(methods.contains(&"ai.ecology_interpret"));
        assert!(methods.contains(&"science.gonzales.dose_response"));
        assert!(methods.contains(&"science.anderson.biome_atlas"));
        assert!(methods.contains(&"science.anderson.hormesis"));
        assert!(methods.contains(&"data.fetch.chembl"));
        assert!(methods.contains(&"vault.store"));
        assert!(methods.contains(&"composition.nucleus_health"));
    }

    /// Every handler-advertised capability must appear in at least one domain.
    ///
    /// `handlers::CAPABILITIES` is the IPC dispatch surface (42 entries).
    /// `DOMAINS` groups them into semantic families (currently 41 methods in
    /// 21 domains plus `capability.list` which is meta-introspection, not a
    /// domain capability). This test catches drift between the two lists.
    #[cfg(feature = "ipc")]
    #[test]
    fn handlers_capabilities_covered_by_domains_or_documented() {
        let domain_methods = all_methods();
        let handler_caps = crate::ipc::handlers::CAPABILITIES;

        let meta_introspection: &[&str] = &["capability.list", "identity.get"];

        let orphan_handler_caps: Vec<&&str> = handler_caps
            .iter()
            .filter(|cap| !domain_methods.contains(cap) && !meta_introspection.contains(cap))
            .collect();

        assert!(
            orphan_handler_caps.is_empty(),
            "handler capabilities not covered by any DOMAIN: {orphan_handler_caps:?} — \
             add these to DOMAINS or document why they are excluded"
        );
    }

    /// Every niche capability must be either in handlers (implemented) or
    /// documented as aspirational (integration/protocol prefixes).
    ///
    /// Orphan niche capabilities:
    /// - `integration.sweetgrass.braid` — aspirational: sweetGrass braid wiring
    /// - `integration.toadstool.performance_surface` — aspirational: toadStool perf data
    /// - `protocol.stream_item` — aspirational: streaming protocol extension
    #[cfg(feature = "ipc")]
    #[test]
    fn niche_capabilities_superset_of_handlers() {
        let niche_caps = crate::niche::CAPABILITIES;
        let handler_caps = crate::ipc::handlers::CAPABILITIES;

        let aspirational_prefixes: &[&str] = &["integration.", "protocol."];

        for hcap in handler_caps {
            assert!(
                niche_caps.contains(hcap),
                "handler capability '{hcap}' missing from niche::CAPABILITIES"
            );
        }

        let orphan_niche: Vec<&&str> = niche_caps
            .iter()
            .filter(|cap| {
                !handler_caps.contains(cap)
                    && !aspirational_prefixes.iter().any(|p| cap.starts_with(p))
            })
            .collect();

        assert!(
            orphan_niche.is_empty(),
            "niche capabilities not in handlers and not aspirational: {orphan_niche:?}"
        );
    }
}
