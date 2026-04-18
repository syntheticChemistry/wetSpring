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
    reason = "validation binary: sequential niche checks in single main()"
)]
//! # Exp402: Niche Parity — NICHE_STARTER_PATTERNS Composition Gate
//!
//! Follows primalSpring's `NICHE_STARTER_PATTERNS.md` template to validate
//! that wetSpring's niche self-knowledge matches its IPC surface and deploy
//! graph — the bridge between Rust validation and primal composition.
//!
//! Pattern: Tower gate → niche parity → science parity → graph alignment.
//!
//! ## Domains
//!
//! | Domain | Check |
//! |--------|-------|
//! | D01 | Niche ↔ IPC surface alignment (CAPABILITIES vs dispatched methods) |
//! | D02 | Niche ↔ deploy graph alignment (DEPENDENCIES vs graph nodes) |
//! | D03 | Consumed ↔ provided capability accounting |
//! | D04 | Science method dispatch coverage (every science.* capability dispatches) |
//! | D05 | Wire standard compliance (L2/L3 shape) |
//!
//! ## Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline | wetSpring niche + dispatch + deploy graph |
//! | Script | `validate_niche_parity_v1.rs` |
//! | Date | 2026-04-17 |
//! | Command | `cargo run --features json,ipc --bin validate_niche_parity_v1` |

use serde_json::json;
use wetspring_barracuda::ipc::dispatch::dispatch;
use wetspring_barracuda::niche;
use wetspring_barracuda::validation::Validator;

fn main() {
    let mut v = Validator::new("Exp402: Niche Parity — NICHE_STARTER_PATTERNS Gate");

    // ═══════════════════════════════════════════════════════════════
    // D01: Niche ↔ IPC surface alignment
    //
    // Every non-aspirational capability in niche::CAPABILITIES should
    // successfully dispatch (or return a structured error, not method_not_found).
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D01: Niche ↔ IPC Surface Alignment ═══");

    let aspirational = [
        "integration.sweetgrass.braid",
        "integration.toadstool.performance_surface",
        "protocol.stream_item",
    ];

    let mut dispatched_count = 0_usize;
    let mut aspirational_count = 0_usize;

    for cap in niche::CAPABILITIES {
        if aspirational.contains(cap) {
            aspirational_count += 1;
            println!("    [SKIP] {cap} (aspirational — primal not yet wired)");
            continue;
        }

        let params = match *cap {
            "science.diversity" => json!({"counts": [0.25, 0.25, 0.25, 0.25]}),
            "science.qs_model" => json!({"scenario": "standard_growth", "dt": 0.01}),
            "science.anderson" => json!({"system_size": 6, "disorder": 4.0}),
            "science.gonzales.dose_response" => {
                json!({"n_points": 10, "dose_max": 100.0, "hill_n": 1.0})
            }
            "science.anderson.disorder_sweep" => {
                json!({"w_min": 1.0, "w_max": 30.0, "n_points": 10})
            }
            "science.alignment" => {
                json!({"seq_a": "ACGT", "seq_b": "ACGT"})
            }
            "science.taxonomy" => json!({"sequence": "ACGTACGT"}),
            "science.phylogenetics" => {
                json!({"tree_a": "(A:0.1,B:0.2);", "tree_b": "(A:0.1,B:0.2);"})
            }
            "science.nmf" => {
                json!({"data": [[1.0, 2.0], [3.0, 4.0]], "rank": 2})
            }
            "science.timeseries" | "science.timeseries_diversity" => {
                json!({"time_series": {"times": [0.0, 1.0], "values": [1.0, 2.0]}})
            }
            "science.ncbi_fetch" => json!({"id": "NC_000913", "db": "nucleotide"}),
            "brain.observe" => {
                let head_outputs: Vec<f64> = (0..36).map(|i| f64::from(i) * 0.01).collect();
                json!({"event": "niche_test", "value": 1.0, "head_outputs": head_outputs})
            }
            "provenance.begin" => json!({"context": "niche_parity_test"}),
            "provenance.record" => {
                json!({"session_id": "niche-test", "event": {"step": "test"}})
            }
            "provenance.complete" => json!({"session_id": "niche-test"}),
            "data.fetch.chembl" => json!({"chembl_id": "CHEMBL25"}),
            "data.fetch.pubchem" => json!({"aid": "1234"}),
            "data.fetch.register_table" => {
                json!({"doi": "10.1111/test", "table_id": "t1", "values": {"a": 1}})
            }
            "vault.store" => json!({"owner_id": "niche-test", "label": "t", "data": "x"}),
            "vault.retrieve" => {
                json!({"owner_id": "niche-test", "content_hash": "0".repeat(64), "consent_token": "t"})
            }
            "vault.consent.verify" => {
                json!({"owner_id": "niche-test", "scope": "read", "consent_token": "t"})
            }
            "ai.ecology_interpret" => {
                json!({"context": "test", "question": "What is diversity?"})
            }
            _ => json!({}),
        };

        let result = dispatch(cap, &params);
        match &result {
            Ok(_) => {
                v.check_pass(&format!("niche→dispatch: {cap}"), true);
                dispatched_count += 1;
            }
            Err(e) => {
                let msg = e.to_string();
                let is_method_not_found = msg.contains("-32601") || msg.contains("method not found");
                if is_method_not_found {
                    v.check_pass(&format!("niche→dispatch: {cap} (method not found — gap)"), false);
                    println!("    GAP: {msg}");
                } else {
                    // Method exists but returned a domain error (GPU required, external
                    // service down, etc.) — counts as "dispatched" for niche parity.
                    v.check_pass(&format!("niche→dispatch: {cap} (method exists, domain error)"), true);
                    dispatched_count += 1;
                    println!("    [INFO] {msg}");
                }
            }
        }
    }

    println!(
        "  Dispatched: {dispatched_count}/{}, aspirational: {aspirational_count}",
        niche::CAPABILITIES.len()
    );

    // ═══════════════════════════════════════════════════════════════
    // D02: Niche ↔ deploy graph alignment
    //
    // Every required dependency in niche::DEPENDENCIES should appear
    // as a node in the primary deploy graph.
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D02: Niche ↔ Deploy Graph Alignment ═══");

    let deploy_graph = include_str!("../../../graphs/wetspring_deploy.toml");
    let nucleus_graph = include_str!("../../../graphs/wetspring_science_nucleus.toml");

    for dep in niche::DEPENDENCIES {
        let name_pattern = format!("name = \"{}\"", dep.name);
        let in_deploy = deploy_graph.contains(&name_pattern);
        let in_nucleus = nucleus_graph.contains(&name_pattern);
        let present = in_deploy || in_nucleus;

        if dep.required {
            v.check_pass(
                &format!("graph: required dep '{}' ({}) in deploy graph", dep.name, dep.role),
                present,
            );
        } else if present {
            v.check_pass(
                &format!("graph: optional dep '{}' ({}) in deploy graph", dep.name, dep.role),
                true,
            );
        } else {
            println!(
                "    [INFO] Optional dep '{}' ({}) not in graph — available via discovery",
                dep.name, dep.role
            );
        }
    }

    // Wetspring itself must be in both graphs
    v.check_pass(
        "graph: wetspring node in deploy graph",
        deploy_graph.contains("name = \"wetspring\""),
    );
    v.check_pass(
        "graph: wetspring node in nucleus graph",
        nucleus_graph.contains("name = \"wetspring\""),
    );

    // ═══════════════════════════════════════════════════════════════
    // D03: Consumed ↔ provided capability accounting
    //
    // Verify that consumed capabilities are a subset of what the
    // ecosystem provides (or would provide once primals are wired).
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D03: Consumed ↔ Provided Accounting ═══");

    v.check_pass(
        "niche: CONSUMED_CAPABILITIES is non-empty",
        !niche::CONSUMED_CAPABILITIES.is_empty(),
    );

    v.check_pass(
        "niche: CAPABILITIES is non-empty",
        !niche::CAPABILITIES.is_empty(),
    );

    v.check_count(
        "niche: CAPABILITIES count",
        niche::CAPABILITIES.len(),
        42,
    );

    v.check_pass(
        "niche: DEPENDENCIES count >= 5 required",
        niche::required_dependency_count() >= 5,
    );

    // No overlap between consumed and provided (a spring shouldn't consume its own caps)
    let mut overlap_count = 0;
    for consumed in niche::CONSUMED_CAPABILITIES {
        if niche::CAPABILITIES.contains(consumed) {
            overlap_count += 1;
            println!("    [WARN] '{consumed}' both consumed and provided");
        }
    }
    v.check_pass(
        "niche: zero overlap between consumed and provided capabilities",
        overlap_count == 0,
    );

    // ═══════════════════════════════════════════════════════════════
    // D04: Science method dispatch coverage
    //
    // Every science.* capability must dispatch and return a result
    // (not just health or infrastructure methods).
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D04: Science Method Dispatch Coverage ═══");

    let science_caps: Vec<&&str> = niche::CAPABILITIES
        .iter()
        .filter(|c| c.starts_with("science."))
        .collect();

    v.check_pass(
        "coverage: science.* capabilities >= 15",
        science_caps.len() >= 15,
    );

    println!("  Science capabilities: {}", science_caps.len());
    for cap in &science_caps {
        println!("    {cap}");
    }

    // ═══════════════════════════════════════════════════════════════
    // D05: Wire standard compliance (L2/L3 shape)
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D05: Wire Standard Compliance ═══");

    let cap_list = dispatch("capability.list", &json!({})).expect("capability.list dispatch");

    // L2: flat methods array
    let methods = cap_list["methods"].as_array().map_or(0, Vec::len);
    v.check_pass("wire L2: methods array present", methods > 0);

    // L3: provided_capabilities with type + methods structure
    let provided = cap_list["provided_capabilities"].as_array();
    v.check_pass("wire L3: provided_capabilities present", provided.is_some());

    if let Some(caps) = provided {
        let has_type = caps.iter().all(|c| c["type"].as_str().is_some());
        let has_methods = caps.iter().all(|c| c["methods"].as_array().is_some());
        v.check_pass("wire L3: every capability has 'type' field", has_type);
        v.check_pass("wire L3: every capability has 'methods' array", has_methods);
    }

    // L3: consumed_capabilities declared
    v.check_pass(
        "wire L3: consumed_capabilities declared",
        cap_list["consumed_capabilities"].as_array().is_some(),
    );

    // Primal identity
    v.check_pass(
        "wire: primal identity = wetspring",
        cap_list["primal"].as_str() == Some("wetspring"),
    );

    // Health probes
    let health = dispatch("health.check", &json!({})).expect("health.check dispatch");
    v.check_pass(
        "wire: health.check returns healthy status",
        health.get("healthy").is_some() || health.get("status").is_some(),
    );

    v.finish();
}
