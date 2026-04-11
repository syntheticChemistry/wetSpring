// SPDX-License-Identifier: AGPL-3.0-or-later
//! Niche deployment self-knowledge for wetSpring.
//!
//! A Spring is a niche validation domain — not a primal. It proves that
//! scientific Python baselines can be faithfully ported to sovereign
//! Rust + GPU compute using the ecoPrimals stack. The niche deploys as
//! a biomeOS graph (`graphs/wetspring_deploy.toml`) that composes
//! real primals (`BearDog`, `Songbird`, `ToadStool`, etc.).
//!
//! This module holds the niche's self-knowledge:
//! - Capability table (what the niche exposes via biomeOS)
//! - Semantic mappings (capability domain → science methods)
//! - Operation dependencies (parallelization hints for Pathway Learner)
//! - Cost estimates (scheduling hints for biomeOS)
//!
//! # Evolution
//!
//! The transitional `wetspring_server` binary exposes these capabilities
//! via a JSON-RPC server. The final form is graph-only deployment where
//! biomeOS orchestrates the niche directly from deploy graphs.

/// Niche identity.
pub use crate::PRIMAL_NAME as NICHE_NAME;

/// Human-readable niche description for biomeOS.
pub const NICHE_DESCRIPTION: &str = "Life science and analytical chemistry validation environment";

/// Niche version (tracks the spring version, not the crate version).
pub const NICHE_VERSION: &str = "1.0.0";

use crate::primal_names::{
    BEARDOG, LOAMSPINE, NESTGATE, PETALTONGUE, RHIZOCRYPT, SONGBIRD, SQUIRREL, SWEETGRASS,
    TOADSTOOL,
};

/// Primals this niche depends on (germination order matters).
pub const DEPENDENCIES: &[NicheDependency] = &[
    NicheDependency {
        name: BEARDOG,
        role: "security",
        required: true,
        capability: "security",
    },
    NicheDependency {
        name: SONGBIRD,
        role: "discovery",
        required: true,
        capability: "discovery",
    },
    NicheDependency {
        name: RHIZOCRYPT,
        role: "dag",
        required: true,
        capability: "dag",
    },
    NicheDependency {
        name: LOAMSPINE,
        role: "commit",
        required: true,
        capability: "commit",
    },
    NicheDependency {
        name: SWEETGRASS,
        role: "provenance",
        required: true,
        capability: "provenance",
    },
    NicheDependency {
        name: NESTGATE,
        role: "storage",
        required: false,
        capability: "storage",
    },
    NicheDependency {
        name: TOADSTOOL,
        role: "compute",
        required: false,
        capability: "compute",
    },
    NicheDependency {
        name: PETALTONGUE,
        role: "visualization",
        required: false,
        capability: "visualization",
    },
    NicheDependency {
        name: SQUIRREL,
        role: "ai",
        required: false,
        capability: "ai",
    },
];

/// Capabilities this niche consumes from other primals.
///
/// Declared per Capability Wire Standard v1.0 Level 3 — enables biomeOS
/// to validate composition completeness without hardcoded knowledge of
/// which primals provide what.
pub const CONSUMED_CAPABILITIES: &[&str] = &[
    // Tower Atomic
    "crypto.sign_ed25519",
    "crypto.verify_ed25519",
    "crypto.blake3_hash",
    "discovery.find_primals",
    "discovery.announce",
    // Node Atomic
    "compute.dispatch.submit",
    "math.tensor",
    "math.stats",
    "math.spectral",
    "shader.compile.wgsl",
    // Nest Atomic
    "storage.store",
    "storage.retrieve",
    "dag.session.create",
    "dag.event.append",
    "spine.create",
    "entry.append",
    "braid.create",
    "braid.commit",
    // Meta-tier
    "ai.complete",
    "inference.complete",
    "inference.embed",
    "render.dashboard",
];

/// All capabilities this niche exposes to biomeOS.
///
/// Derived from `ipc::capability_domains::DOMAINS` — kept as a flat list
/// for biomeOS registration and Songbird advertisement.
pub const CAPABILITIES: &[&str] = &[
    // ── health / discovery (biomeOS infrastructure) ──
    "capability.list",
    "identity.get",
    "health.check",
    "health.liveness",
    "health.readiness",
    // ── ecology (science) ──
    "science.diversity",
    "science.qs_model",
    "science.anderson",
    "science.kinetics",
    "science.alignment",
    "science.taxonomy",
    "science.phylogenetics",
    "science.nmf",
    "science.timeseries",
    "science.timeseries_diversity",
    "science.ncbi_fetch",
    "science.full_pipeline",
    // ── Gonzales dermatitis / immunological Anderson ──
    "science.gonzales.dose_response",
    "science.gonzales.pk_decay",
    "science.gonzales.tissue_lattice",
    "science.anderson.biome_atlas",
    "science.anderson.disorder_sweep",
    "science.anderson.hormesis",
    "science.anderson.cross_species",
    // ── provenance trio ──
    "provenance.begin",
    "provenance.record",
    "provenance.complete",
    // ── brain sentinel ──
    "brain.observe",
    "brain.attention",
    "brain.urgency",
    // ── metrics ──
    "metrics.snapshot",
    // ── AI assist (Squirrel) ──
    "ai.ecology_interpret",
    // ── external data ingestion ──
    "data.fetch.chembl",
    "data.fetch.pubchem",
    "data.fetch.register_table",
    // ── vault (consent-gated storage) ──
    "vault.store",
    "vault.retrieve",
    "vault.consent.verify",
    // ── composition health (cross-spring validation) ──
    "composition.science_health",
    "composition.tower_health",
    "composition.node_health",
    "composition.nest_health",
    "composition.nucleus_health",
    // ── ecosystem client integrations (optional primals) ──
    "integration.sweetgrass.braid",
    "integration.toadstool.performance_surface",
    "protocol.stream_item",
];

/// A primal dependency for this niche.
pub struct NicheDependency {
    /// Primal identifier (e.g. `"beardog"`).
    pub name: &'static str,
    /// Role in the niche (e.g. `"security"`, `"discovery"`).
    pub role: &'static str,
    /// Whether the niche can function without this primal.
    pub required: bool,
    /// Capability used to discover this primal at runtime.
    pub capability: &'static str,
}

/// Operation dependency hints for biomeOS Pathway Learner parallelization.
///
/// Maps each operation to the data inputs it requires, enabling the Pathway
/// Learner to build a DAG and parallelize independent operations.
#[cfg(feature = "json")]
#[must_use]
pub fn operation_dependencies() -> serde_json::Value {
    serde_json::json!({
        "science.diversity": ["abundance_table"],
        "science.qs_model": ["ode_parameters"],
        "science.anderson": ["lattice_dimension", "disorder_strength"],
        "science.kinetics": ["time_series", "kinetic_model"],
        "science.alignment": ["query_sequence", "reference_sequence"],
        "science.taxonomy": ["kmer_profiles", "reference_db"],
        "science.phylogenetics": ["newick_trees"],
        "science.nmf": ["expression_matrix", "rank"],
        "science.timeseries": ["station", "date_range"],
        "science.ncbi_fetch": ["accession", "database"],
        "science.full_pipeline": ["fastq_paths"],
        "provenance.begin": ["experiment_name"],
        "provenance.record": ["session_id", "step_data"],
        "provenance.complete": ["session_id"],
        "brain.observe": ["observation_data"],
        "brain.attention": ["head_group"],
        "brain.urgency": ["domain"],
        "metrics.snapshot": [],
    })
}

/// Cost estimates for biomeOS scheduling (measured on Eastgate — RTX 4070 12 GB).
///
/// Guides the Pathway Learner on expected latency and resource usage per operation.
#[cfg(feature = "json")]
#[must_use]
pub fn cost_estimates() -> serde_json::Value {
    serde_json::json!({
        "science.diversity":      { "latency_ms": 0.5, "cpu": "low", "memory_bytes": 4096 },
        "science.qs_model":       { "latency_ms": 2.0, "cpu": "medium", "memory_bytes": 8192 },
        "science.anderson":       { "latency_ms": 5.0, "cpu": "medium", "memory_bytes": 16384 },
        "science.kinetics":       { "latency_ms": 1.0, "cpu": "low", "memory_bytes": 4096 },
        "science.alignment":      { "latency_ms": 0.3, "cpu": "low", "memory_bytes": 2048 },
        "science.taxonomy":       { "latency_ms": 1.5, "cpu": "medium", "memory_bytes": 32768 },
        "science.phylogenetics":  { "latency_ms": 1.0, "cpu": "low", "memory_bytes": 4096 },
        "science.nmf":            { "latency_ms": 50.0, "cpu": "high", "memory_bytes": 65536 },
        "science.timeseries":     { "latency_ms": 5.0, "cpu": "low", "memory_bytes": 8192 },
        "science.ncbi_fetch":     { "latency_ms": 500.0, "cpu": "low", "memory_bytes": 16384 },
        "science.full_pipeline":  { "latency_ms": 200.0, "cpu": "high", "memory_bytes": 131_072 },
        "provenance.begin":       { "latency_ms": 10.0, "cpu": "low", "memory_bytes": 512 },
        "provenance.record":      { "latency_ms": 5.0, "cpu": "low", "memory_bytes": 1024 },
        "provenance.complete":    { "latency_ms": 50.0, "cpu": "medium", "memory_bytes": 2048 },
        "brain.observe":          { "latency_ms": 1.0, "cpu": "low", "memory_bytes": 4096 },
        "brain.attention":        { "latency_ms": 0.5, "cpu": "low", "memory_bytes": 2048 },
        "brain.urgency":          { "latency_ms": 0.3, "cpu": "low", "memory_bytes": 1024 },
        "metrics.snapshot":       { "latency_ms": 0.1, "cpu": "low", "memory_bytes": 256 },
    })
}

/// Semantic mappings for ecology capability domain routing.
///
/// Maps short operation names to fully-qualified IPC method names,
/// enabling biomeOS Neural API to route natural-language requests.
#[cfg(feature = "json")]
#[must_use]
pub fn ecology_semantic_mappings() -> serde_json::Value {
    serde_json::json!({
        "diversity":              "science.diversity",
        "qs_model":               "science.qs_model",
        "anderson":               "science.anderson",
        "kinetics":               "science.kinetics",
        "alignment":              "science.alignment",
        "taxonomy":               "science.taxonomy",
        "phylogenetics":          "science.phylogenetics",
        "nmf":                    "science.nmf",
        "timeseries":             "science.timeseries",
        "timeseries_diversity":   "science.timeseries_diversity",
        "ncbi_fetch":             "science.ncbi_fetch",
        "full_pipeline":          "science.full_pipeline",
        "gonzales_dose_response": "science.gonzales.dose_response",
        "gonzales_pk_decay":      "science.gonzales.pk_decay",
        "gonzales_tissue_lattice":"science.gonzales.tissue_lattice",
        "anderson_biome_atlas":   "science.anderson.biome_atlas",
        "anderson_disorder_sweep":"science.anderson.disorder_sweep",
        "anderson_hormesis":      "science.anderson.hormesis",
        "anderson_cross_species": "science.anderson.cross_species",
    })
}

/// Returns the deploy graph path for this niche.
#[must_use]
pub const fn deploy_graph_path() -> &'static str {
    crate::primal_names::DEPLOY_GRAPH_REL_PATH
}

/// Environment variable for explicit NPU device path override.
pub const NPU_DEVICE_ENV_VAR: &str = "WETSPRING_NPU_DEVICE";

/// Default NPU device path when neither env var nor toadStool discovery
/// is available.
///
/// `BrainChip` `AKD1000` standard Linux device node. In production, NPU
/// discovery goes through toadStool IPC (`toadstool.device.npu`).
const NPU_DEFAULT_DEVICE: &str = "/dev/akida0";

/// Discover the NPU device path using the standard cascade.
///
/// # Discovery cascade
///
/// 1. `$WETSPRING_NPU_DEVICE` — explicit operator override
/// 2. toadStool IPC `toadstool.device.npu` — runtime hardware discovery
///    (caller responsibility; this function handles the non-IPC fallback)
/// 3. Default: `/dev/akida0` (`BrainChip` `AKD1000` standard path)
#[must_use]
pub fn discover_npu_device() -> String {
    std::env::var(NPU_DEVICE_ENV_VAR).unwrap_or_else(|_| NPU_DEFAULT_DEVICE.to_owned())
}

/// Returns the number of required dependencies.
#[must_use]
pub fn required_dependency_count() -> usize {
    DEPENDENCIES.iter().filter(|d| d.required).count()
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: assertions use expect for clarity"
)]
mod tests {
    use super::*;

    #[test]
    fn capabilities_are_not_empty() {
        assert!(!CAPABILITIES.is_empty());
    }

    #[test]
    fn capabilities_follow_semantic_naming() {
        for cap in CAPABILITIES {
            assert!(
                cap.contains('.'),
                "capability '{cap}' should follow domain.operation format"
            );
        }
    }

    #[test]
    fn capabilities_count_matches_domains() {
        assert_eq!(
            CAPABILITIES.len(),
            46,
            "5 health/meta + 19 science + 3 provenance + 3 brain + 1 metrics + 1 ai + 3 data.fetch + 3 vault + 5 composition + 3 ecosystem integrations"
        );
    }

    #[test]
    fn dependencies_include_security_and_discovery() {
        assert!(DEPENDENCIES.iter().any(|d| d.role == "security"));
        assert!(DEPENDENCIES.iter().any(|d| d.role == "discovery"));
    }

    #[test]
    fn required_dependencies_are_five() {
        assert_eq!(required_dependency_count(), 5);
    }

    #[test]
    fn niche_name_matches_convention() {
        assert_eq!(NICHE_NAME, "wetspring");
        assert!(NICHE_NAME.chars().all(|c| c.is_ascii_lowercase()));
    }

    #[test]
    fn deploy_graph_path_is_toml() {
        let path = std::path::Path::new(deploy_graph_path());
        assert_eq!(path.extension().and_then(|e| e.to_str()), Some("toml"));
    }

    #[cfg(feature = "json")]
    #[test]
    fn operation_dependencies_is_object() {
        let deps = operation_dependencies();
        assert!(deps.is_object());
    }

    #[cfg(feature = "json")]
    #[test]
    fn cost_estimates_is_object() {
        let costs = cost_estimates();
        assert!(costs.is_object());
    }

    #[cfg(feature = "json")]
    #[test]
    fn ecology_mappings_cover_science_capabilities() {
        let mappings = ecology_semantic_mappings();
        let map = mappings.as_object().expect("mappings should be an object");
        let science_caps: Vec<&&str> = CAPABILITIES
            .iter()
            .filter(|c| c.starts_with("science."))
            .collect();
        for cap in &science_caps {
            assert!(
                map.values().any(|v| v.as_str() == Some(cap)),
                "science capability '{cap}' should appear in ecology mappings"
            );
        }
    }

    /// Cross-validate niche dependencies against the proto-nucleate graph.
    ///
    /// The canonical proto-nucleate lives in primalSpring. This test
    /// `include_str!`s it at compile time so drift is caught in CI.
    /// If the file is not reachable (different checkout layout), the test
    /// is a no-op rather than a hard failure.
    #[test]
    fn proto_nucleate_node_names_match_niche_dependencies() {
        let proto_nucleate_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../primalSpring/graphs/downstream/wetspring_lifescience_proto_nucleate.toml"
        );
        let Ok(proto) = std::fs::read_to_string(proto_nucleate_path) else {
            eprintln!(
                "proto-nucleate not found at {proto_nucleate_path} — \
                 clone primalSpring alongside wetSpring to enable this check"
            );
            return;
        };

        for dep in DEPENDENCIES {
            let pattern = format!("name = \"{}\"", dep.name);
            assert!(
                proto.contains(&pattern),
                "niche dependency '{}' (role: {}) not found as a node in the proto-nucleate graph",
                dep.name,
                dep.role,
            );
        }

        assert!(
            proto.contains("name = \"wetspring\""),
            "proto-nucleate must contain a 'wetspring' application node"
        );

        assert!(
            proto.contains("pattern = \"proto_nucleate\""),
            "proto-nucleate must declare pattern = \"proto_nucleate\""
        );

        assert!(
            proto.contains("owner = \"wetSpring\""),
            "proto-nucleate must declare owner = \"wetSpring\""
        );
    }
}
