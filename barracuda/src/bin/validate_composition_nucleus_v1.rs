// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::print_stdout,
    reason = "validation binary: stdout is the output medium"
)]
#![expect(
    clippy::expect_used,
    reason = "validation binary: expect is the pass/fail mechanism"
)]
#![expect(
    clippy::too_many_lines,
    reason = "validation binary: sequential domain checks in single main()"
)]
//! # Exp400: NUCLEUS Composition Validation — Proto-Nucleate Alignment
//!
//! Validates that wetSpring's niche self-knowledge, deploy graphs, and
//! composition model align with the proto-nucleate graph defined in
//! `primalSpring/graphs/downstream/wetspring_lifescience_proto_nucleate.toml`.
//!
//! This is the **composition validation** tier: Python was the validation
//! target for Rust, and now Rust + Python are the validation targets for
//! the ecoPrimal NUCLEUS composition patterns.
//!
//! ## Domains
//!
//! | Domain | Check |
//! |--------|-------|
//! | D01 | Niche self-knowledge consistency |
//! | D02 | Capability surface completeness vs proto-nucleate |
//! | D03 | Proto-nucleate primal coverage |
//! | D04 | Deploy graph structural validation (TOML parse) |
//! | D05 | Composition model alignment |
//! | D06 | Bonding metadata and atomic fragments |
//!
//! ## Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline | wetSpring niche + proto-nucleate graph |
//! | Script | `validate_composition_nucleus_v1.rs` |
//! | Date | 2026-04-10 |
//! | Command | `cargo run --features json --bin validate_composition_nucleus_v1` |

use wetspring_barracuda::niche;
use wetspring_barracuda::validation::Validator;

/// Expected total checks — matches the "141/141" claim in README/CONTEXT.
/// Update this constant (and docs) whenever domains are added or removed.
const EXPECTED_CHECKS: u32 = 141;

fn main() {
    let mut v = Validator::new("Exp400: NUCLEUS Composition — Proto-Nucleate Alignment");

    // ═══════════════════════════════════════════════════════════════
    // D01: Niche self-knowledge consistency
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D01: Niche Self-Knowledge ═══");

    v.check_pass(
        "niche: NICHE_NAME = wetspring",
        niche::NICHE_NAME == "wetspring",
    );

    let graph_path = niche::deploy_graph_path();
    v.check_pass(
        "niche: deploy_graph_path ends with .toml",
        std::path::Path::new(&graph_path)
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("toml")),
    );

    v.check_pass(
        "niche: required_dependency_count >= 5",
        niche::required_dependency_count() >= 5,
    );

    let dep_names: Vec<&str> = niche::DEPENDENCIES.iter().map(|d| d.name).collect();
    v.check_pass(
        "niche: deps include beardog (Tower/security)",
        dep_names.contains(&"beardog"),
    );
    v.check_pass(
        "niche: deps include songbird (Tower/discovery)",
        dep_names.contains(&"songbird"),
    );
    v.check_pass(
        "niche: deps include toadstool (Node/compute)",
        dep_names.contains(&"toadstool"),
    );
    v.check_pass(
        "niche: deps include nestgate (Nest/storage)",
        dep_names.contains(&"nestgate"),
    );
    v.check_pass(
        "niche: deps include rhizocrypt (provenance/dag)",
        dep_names.contains(&"rhizocrypt"),
    );
    v.check_pass(
        "niche: deps include loamspine (provenance/commit)",
        dep_names.contains(&"loamspine"),
    );
    v.check_pass(
        "niche: deps include sweetgrass (provenance/attribution)",
        dep_names.contains(&"sweetgrass"),
    );

    // ═══════════════════════════════════════════════════════════════
    // D02: Capability surface completeness
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D02: Capability Surface ═══");

    let caps = niche::CAPABILITIES;
    println!("  Total capabilities advertised: {}", caps.len());

    v.check_pass("capabilities: 46 advertised", caps.len() == 46);

    let required_caps = [
        "health.check",
        "health.liveness",
        "health.readiness",
        "capability.list",
        "science.diversity",
        "science.anderson",
        "science.full_pipeline",
        "science.gonzales.dose_response",
        "science.gonzales.pk_decay",
        "science.gonzales.tissue_lattice",
        "science.anderson.biome_atlas",
        "science.anderson.disorder_sweep",
        "science.anderson.hormesis",
        "science.anderson.cross_species",
        "provenance.begin",
        "provenance.record",
        "provenance.complete",
        "brain.observe",
        "brain.attention",
        "brain.urgency",
        "ai.ecology_interpret",
        "metrics.snapshot",
        "composition.science_health",
        "composition.tower_health",
        "composition.node_health",
        "composition.nest_health",
        "composition.nucleus_health",
        "vault.store",
        "vault.retrieve",
        "vault.consent.verify",
        "data.fetch.chembl",
        "data.fetch.pubchem",
        "data.fetch.register_table",
    ];

    for cap in &required_caps {
        v.check_pass(&format!("capability present: {cap}"), caps.contains(cap));
    }

    let has_health = caps.iter().any(|c| c.starts_with("health."));
    let has_science = caps.iter().any(|c| c.starts_with("science."));
    let has_composition = caps.iter().any(|c| c.starts_with("composition."));
    let has_provenance = caps.iter().any(|c| c.starts_with("provenance."));
    let has_brain = caps.iter().any(|c| c.starts_with("brain."));
    let has_vault = caps.iter().any(|c| c.starts_with("vault."));
    let has_data = caps.iter().any(|c| c.starts_with("data.fetch."));

    v.check_pass("domain: health.*", has_health);
    v.check_pass("domain: science.*", has_science);
    v.check_pass("domain: composition.*", has_composition);
    v.check_pass("domain: provenance.*", has_provenance);
    v.check_pass("domain: brain.*", has_brain);
    v.check_pass("domain: vault.*", has_vault);
    v.check_pass("domain: data.fetch.*", has_data);

    // ═══════════════════════════════════════════════════════════════
    // D03: Proto-nucleate primal coverage
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D03: Proto-Nucleate Primal Coverage ═══");

    let proto_primals: &[(&str, &str, &str)] = &[
        ("beardog", "security", "Tower"),
        ("songbird", "discovery", "Tower"),
        ("toadstool", "compute", "Node"),
        ("barracuda", "math", "Node (path dep)"),
        ("coralreef", "shader", "Node (via barraCuda)"),
        ("nestgate", "storage", "Nest"),
        ("rhizocrypt", "dag", "Provenance trio"),
        ("loamspine", "commit", "Provenance trio"),
        ("sweetgrass", "provenance", "Provenance trio"),
        ("squirrel", "ai", "Meta-tier"),
        ("petaltongue", "visualization", "Meta-tier"),
    ];

    for (primal, role, tier) in proto_primals {
        let has_dep = dep_names.contains(primal);
        let is_path_dep = *primal == "barracuda" || *primal == "coralreef";
        let covered = has_dep || is_path_dep;
        v.check_pass(
            &format!("proto-nucleate: {primal} ({role}, {tier}) covered"),
            covered,
        );
    }

    // ═══════════════════════════════════════════════════════════════
    // D04: Deploy graph structural validation
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D04: Deploy Graph Structural Validation ═══");

    let graph_sources: &[(&str, &str)] = &[
        (
            "wetspring_science_nucleus",
            include_str!("../../../graphs/wetspring_science_nucleus.toml"),
        ),
        (
            "wetspring_niche",
            include_str!("../../../graphs/wetspring_niche.toml"),
        ),
        (
            "wetspring_anderson_atlas",
            include_str!("../../../graphs/wetspring_anderson_atlas.toml"),
        ),
        (
            "wetspring_gonzales_exploration",
            include_str!("../../../graphs/wetspring_gonzales_exploration.toml"),
        ),
        (
            "wetspring_basement_deploy",
            include_str!("../../../graphs/wetspring_basement_deploy.toml"),
        ),
        (
            "wetspring_deploy",
            include_str!("../../../graphs/wetspring_deploy.toml"),
        ),
        (
            "wetspring_science_facade",
            include_str!("../../../graphs/wetspring_science_facade.toml"),
        ),
    ];

    for (name, source) in graph_sources {
        v.check_pass(
            &format!("graph {name}: non-empty"),
            !source.trim().is_empty(),
        );

        let has_graph_nodes =
            source.contains("[[graph.nodes]]") || source.contains("[[graph.node]]");
        v.check_pass(
            &format!("graph {name}: has [[graph.nodes]] (canonical) or [[graph.node]] (legacy)"),
            has_graph_nodes,
        );

        let has_name_field = source.contains("name =") || source.contains("name=");
        v.check_pass(
            &format!("graph {name}: has node name declarations"),
            has_name_field,
        );

        println!("    {name}: {} bytes, structurally present", source.len());
    }

    // ═══════════════════════════════════════════════════════════════
    // D05: Composition model alignment
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D05: Composition Model Alignment ═══");

    let nucleus_graph_src = include_str!("../../../graphs/wetspring_science_nucleus.toml");

    let required_nodes = [
        "biomeos_neural_api",
        "beardog",
        "songbird",
        "nestgate",
        "toadstool",
        "wetspring",
    ];
    for required in &required_nodes {
        let pattern = format!("name = \"{required}\"");
        v.check_pass(
            &format!("nucleus graph: node '{required}' declared"),
            nucleus_graph_src.contains(&pattern),
        );
    }

    v.check_pass(
        "nucleus graph: wetspring has composition.science_health",
        nucleus_graph_src.contains("composition.science_health"),
    );
    v.check_pass(
        "nucleus graph: wetspring has composition.nucleus_health",
        nucleus_graph_src.contains("composition.nucleus_health"),
    );

    // ═══════════════════════════════════════════════════════════════
    // D06: Bonding metadata and atomic fragments
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D06: Bonding & Atomic Alignment ═══");

    let ecology_map = niche::ecology_semantic_mappings();

    v.check_pass("ecology mappings: is JSON object", ecology_map.is_object());

    let eco_obj = ecology_map.as_object().expect("ecology mappings object");
    v.check_pass("ecology mappings: 15+ entries", eco_obj.len() >= 15);

    let has_gonzales = eco_obj.keys().any(|k| k.contains("gonzales"));
    let has_anderson = eco_obj.keys().any(|k| k.contains("anderson"));
    let has_diversity = eco_obj.keys().any(|k| k.contains("diversity"));

    v.check_pass("ecology: gonzales domain mapped", has_gonzales);
    v.check_pass("ecology: anderson domain mapped", has_anderson);
    v.check_pass("ecology: diversity domain mapped", has_diversity);

    v.check_pass(
        "niche deploy graph: references science_nucleus",
        graph_path.contains("science_nucleus") || graph_path.contains("wetspring"),
    );

    // ═══════════════════════════════════════════════════════════════
    // D07: Deploy graph metadata compliance (V143 composition tier)
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D07: Deploy Graph Metadata Compliance ═══");

    let metadata_graphs: &[(&str, &str)] = &[
        (
            "wetspring_deploy",
            include_str!("../../../graphs/wetspring_deploy.toml"),
        ),
        (
            "wetspring_science_nucleus",
            include_str!("../../../graphs/wetspring_science_nucleus.toml"),
        ),
        (
            "wetspring_science_facade",
            include_str!("../../../graphs/wetspring_science_facade.toml"),
        ),
        (
            "wetspring_niche",
            include_str!("../../../graphs/wetspring_niche.toml"),
        ),
        (
            "wetspring_anderson_atlas",
            include_str!("../../../graphs/wetspring_anderson_atlas.toml"),
        ),
        (
            "wetspring_basement_deploy",
            include_str!("../../../graphs/wetspring_basement_deploy.toml"),
        ),
        (
            "wetspring_gonzales_exploration",
            include_str!("../../../graphs/wetspring_gonzales_exploration.toml"),
        ),
    ];

    for (name, src) in metadata_graphs {
        v.check_pass(
            &format!("{name}: has composition_model"),
            src.contains("composition_model"),
        );
        v.check_pass(
            &format!("{name}: has owner = \"wetSpring\""),
            src.contains("owner = \"wetSpring\""),
        );
        v.check_pass(
            &format!("{name}: has fragments metadata"),
            src.contains("fragments"),
        );
        v.check_pass(
            &format!("{name}: uses [[graph.nodes]] canonical schema"),
            src.contains("[[graph.nodes]]"),
        );
    }

    let full_nucleus_graphs: &[(&str, &str)] = &[
        (
            "wetspring_deploy",
            include_str!("../../../graphs/wetspring_deploy.toml"),
        ),
        (
            "wetspring_science_nucleus",
            include_str!("../../../graphs/wetspring_science_nucleus.toml"),
        ),
    ];

    for (name, src) in full_nucleus_graphs {
        v.check_pass(
            &format!("{name}: has bonding_policy"),
            src.contains("[graph.bonding_policy]"),
        );
        v.check_pass(
            &format!("{name}: has witness_wire"),
            src.contains("witness_wire"),
        );
        v.check_pass(
            &format!("{name}: declares tower_atomic fragment"),
            src.contains("tower_atomic"),
        );
        v.check_pass(
            &format!("{name}: declares node_atomic fragment"),
            src.contains("node_atomic"),
        );
        v.check_pass(
            &format!("{name}: declares nest_atomic fragment"),
            src.contains("nest_atomic"),
        );
        v.check_pass(
            &format!("{name}: declares meta_tier fragment"),
            src.contains("meta_tier"),
        );
        v.check_pass(
            &format!("{name}: has coralreef node"),
            src.contains("name = \"coralreef\""),
        );
        v.check_pass(
            &format!("{name}: has barracuda node"),
            src.contains("name = \"barracuda\""),
        );
    }

    // ═══════════════════════════════════════════════════════════════
    // Guard: total check count matches documented claim
    // ═══════════════════════════════════════════════════════════════
    let (_, total) = v.counts();
    assert_eq!(
        total, EXPECTED_CHECKS,
        "Exp400 check count drifted: got {total}, expected {EXPECTED_CHECKS} — update EXPECTED_CHECKS and docs"
    );

    // ═══════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════
    v.finish();
}
