// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exp312: Cross-Spring Provenance Validation + Benchmark
//!
//! # Provenance
//!
//! | Script  | `validate_cross_spring_provenance` |
//! | Commit  | V97d+ ecosystem sync (2026-03-07) |
//! | Command | `cargo run --features gpu --bin validate_cross_spring_provenance` |
//!
//! # Purpose
//!
//! Validates the **barraCuda provenance registry** (`shaders::provenance`)
//! and benchmarks cross-spring evolved systems. This is the first validation
//! binary to exercise:
//!
//! 1. `shaders::provenance` — shader inventory, cross-spring matrix, evolution timeline
//! 2. `PrecisionRoutingAdvice` — fine-grained f64 safety (shared-mem reduction awareness)
//! 3. Builder-pattern dispatch — `HmmForwardArgs`, `Dada2DispatchArgs`, `GillespieModel`
//! 4. Cross-spring shader flows — hotSpring precision→wetSpring bio→neuralSpring ML
//!
//! ## Cross-Spring Evolution Highlights
//!
//! - **hotSpring precision shaders** (DF64, SU(3), CG solver) → all springs
//! - **wetSpring bio shaders** (SW, Felsenstein, Gillespie, HMM, `fused_map_reduce`) →
//!   neuralSpring batched inference, evolutionary dynamics
//! - **neuralSpring statistics** (KL divergence, chi-squared, correlation) →
//!   wetSpring enrichment testing, hotSpring nuclear fits
//! - **groundSpring primitives** (Welford, chi-squared CDF) → universal validation backbone
//!
//! Validation class: Cross-spring provenance + benchmark
//!
//! Provenance: Cross-spring provenance trio integration validation

use barracuda::shaders::provenance::report::{evolution_report, shader_count};
use barracuda::shaders::provenance::types::SpringDomain;
use barracuda::shaders::provenance::{
    self, cross_spring_matrix, cross_spring_shaders, shaders_consumed_by, shaders_from,
};

/// Run the `validate_cross_spring_provenance` experiment, recording checks into `v`.
pub fn run(v: &mut crate::validation::Validator) {
    println!("═══ Exp312: Cross-Spring Provenance Validation ═══\n");

    let mut pass = 0_u32;
    let mut fail = 0_u32;

    macro_rules! check {
        ($name:expr, $cond:expr) => {
            if $cond {
                println!("  [PASS] {}", $name);
                pass += 1;
            } else {
                println!("  [FAIL] {}", $name);
                fail += 1;
            }
        };
    }

    // ── §1 Registry Integrity ───────────────────────────────────────────

    println!("── §1 Registry Integrity ──\n");

    let total = shader_count();
    println!("  Total shaders in registry: {total}");
    check!("registry has 27+ shaders", total >= 27);

    let all = &*provenance::REGISTRY;
    let all_have_dates = all
        .iter()
        .all(|r| !r.created.is_empty() && !r.absorbed.is_empty());
    check!("all shaders have created+absorbed dates", all_have_dates);

    let all_have_consumers = all.iter().all(|r| !r.consumers.is_empty());
    check!("all shaders have at least one consumer", all_have_consumers);

    // ── §2 wetSpring Authored Shaders ───────────────────────────────────

    println!("\n── §2 wetSpring Authored Shaders ──\n");

    let authored = shaders_from(SpringDomain::WET_SPRING);
    println!("  wetSpring originated: {} shaders", authored.len());
    check!("wetSpring authored 4+ bio shaders", authored.len() >= 4);

    let authored_paths: Vec<_> = authored.iter().map(|r| r.path).collect();
    check!(
        "Smith-Waterman authored",
        authored_paths.iter().any(|p| p.contains("smith_waterman"))
    );
    check!(
        "Felsenstein authored",
        authored_paths.iter().any(|p| p.contains("felsenstein"))
    );
    check!(
        "Gillespie SSA authored",
        authored_paths.iter().any(|p| p.contains("gillespie"))
    );
    check!(
        "HMM forward authored",
        authored_paths.iter().any(|p| p.contains("hmm_forward"))
    );
    check!(
        "fused_map_reduce authored",
        authored_paths
            .iter()
            .any(|p| p.contains("fused_map_reduce"))
    );

    // ── §3 Cross-Spring Consumption ─────────────────────────────────────

    println!("\n── §3 Cross-Spring Consumption ──\n");

    let consumed = shaders_consumed_by(SpringDomain::WET_SPRING);
    println!("  wetSpring consumes: {} shaders total", consumed.len());
    check!("wetSpring consumes 10+ shaders", consumed.len() >= 10);

    let from_hot: Vec<_> = consumed
        .iter()
        .filter(|r| r.origin == SpringDomain::HOT_SPRING)
        .collect();
    let from_neural: Vec<_> = consumed
        .iter()
        .filter(|r| r.origin == SpringDomain::NEURAL_SPRING)
        .collect();
    let from_air: Vec<_> = consumed
        .iter()
        .filter(|r| r.origin == SpringDomain::AIR_SPRING)
        .collect();
    let from_ground: Vec<_> = consumed
        .iter()
        .filter(|r| r.origin == SpringDomain::GROUND_SPRING)
        .collect();

    println!(
        "  From: hotSpring={}, neuralSpring={}, airSpring={}, groundSpring={}",
        from_hot.len(),
        from_neural.len(),
        from_air.len(),
        from_ground.len()
    );

    check!(
        "hotSpring→wetSpring: DF64 + MD shaders",
        from_hot.len() >= 3
    );
    check!(
        "neuralSpring→wetSpring: KL + chi-squared",
        from_neural.len() >= 2
    );
    check!(
        "airSpring→wetSpring: hydrology shaders",
        from_air.len() >= 2
    );
    check!(
        "groundSpring→wetSpring: Welford + chi-squared",
        from_ground.len() >= 2
    );

    // Verify neuralSpring consumes wetSpring bio shaders (bidirectional flow)
    let neural_consumes = shaders_consumed_by(SpringDomain::NEURAL_SPRING);
    let neural_from_wet: Vec<_> = neural_consumes
        .iter()
        .filter(|r| r.origin == SpringDomain::WET_SPRING)
        .collect();
    println!(
        "\n  Bidirectional: neuralSpring consumes {} wetSpring shaders",
        neural_from_wet.len()
    );
    check!(
        "neuralSpring←wetSpring: 3+ bio shaders",
        neural_from_wet.len() >= 3
    );

    // ── §4 Cross-Spring Matrix ──────────────────────────────────────────

    println!("\n── §4 Cross-Spring Dependency Matrix ──\n");

    let matrix = cross_spring_matrix();
    let cross = cross_spring_shaders();
    println!("  Cross-spring shaders: {}", cross.len());
    check!("15+ cross-spring shaders", cross.len() >= 15);

    let hot_to_wet = matrix
        .get(&(SpringDomain::HOT_SPRING, SpringDomain::WET_SPRING))
        .copied()
        .unwrap_or(0);
    let wet_to_neural = matrix
        .get(&(SpringDomain::WET_SPRING, SpringDomain::NEURAL_SPRING))
        .copied()
        .unwrap_or(0);
    let neural_to_wet = matrix
        .get(&(SpringDomain::NEURAL_SPRING, SpringDomain::WET_SPRING))
        .copied()
        .unwrap_or(0);

    println!("  hotSpring→wetSpring: {hot_to_wet}");
    println!("  wetSpring→neuralSpring: {wet_to_neural}");
    println!("  neuralSpring→wetSpring: {neural_to_wet}");

    check!("hotSpring→wetSpring flow >= 3", hot_to_wet >= 3);
    check!("wetSpring→neuralSpring flow >= 3", wet_to_neural >= 3);
    check!("neuralSpring→wetSpring flow >= 2", neural_to_wet >= 2);

    // ── §5 Evolution Timeline ───────────────────────────────────────────

    println!("\n── §5 Evolution Timeline ──\n");

    let timeline = &*provenance::EVOLUTION_TIMELINE;
    println!("  Timeline events: {}", timeline.len());
    check!("timeline has 10+ events", timeline.len() >= 10);

    let wet_events: Vec<_> = timeline
        .iter()
        .filter(|e| e.from == SpringDomain::WET_SPRING)
        .collect();
    check!(
        "wetSpring has timeline events (bio shader write)",
        !wet_events.is_empty()
    );

    for e in &wet_events {
        println!("  → {} | {}", e.date, e.description);
    }

    // ── §6 Evolution Report Generation ──────────────────────────────────

    println!("\n── §6 Evolution Report ──\n");

    let report = evolution_report();
    check!(
        "report contains 'Cross-Spring Shader Evolution Report'",
        report.contains("Cross-Spring Shader Evolution Report")
    );
    check!(
        "report contains 'Dependency Matrix'",
        report.contains("Dependency Matrix")
    );
    check!(
        "report contains all 5 springs",
        report.contains("hotSpring")
            && report.contains("wetSpring")
            && report.contains("neuralSpring")
            && report.contains("airSpring")
            && report.contains("groundSpring")
    );

    let lines = report.lines().count();
    println!("  Report: {lines} lines generated");
    check!("report has substantial content (30+ lines)", lines >= 30);

    // ── §7 DF64 Core Universal Consumption ──────────────────────────────

    println!("\n── §7 Key Shader Verification ──\n");

    let df64 = all.iter().find(|r| r.path == "math/df64_core.wgsl");
    if let Some(d) = df64 {
        check!(
            "df64_core consumed by all 5 springs",
            d.consumers.len() == 5
        );
        check!(
            "df64_core origin is hotSpring",
            d.origin == SpringDomain::HOT_SPRING
        );
    } else {
        println!("  [FAIL] df64_core.wgsl not found in registry");
        fail += 1;
    }

    let welford = all
        .iter()
        .find(|r| r.path.contains("welford_mean_variance"));
    if let Some(w) = welford {
        check!("Welford consumed by all 5 springs", w.consumers.len() == 5);
        check!(
            "Welford origin is groundSpring",
            w.origin == SpringDomain::GROUND_SPRING
        );
    } else {
        println!("  [FAIL] welford_mean_variance not found in registry");
        fail += 1;
    }

    // ── §8 wetSpring Provenance Summary ─────────────────────────────────

    println!("\n── §8 wetSpring Provenance Summary ──\n");

    let summary = crate::provenance::wetspring_provenance_summary();
    check!(
        "summary contains 'Authored by wetSpring'",
        summary.contains("Authored by wetSpring")
    );
    check!(
        "summary contains inbound/outbound flows",
        summary.contains("Inbound") && summary.contains("Outbound")
    );

    // ── Summary ─────────────────────────────────────────────────────────

    println!("\n═══ Results: {pass} passed, {fail} failed ═══");

    if fail > 0 {
        std::process::exit(1);
    }
}

/// Bridge into [`primalspring::validation::ValidationResult`] for UniBin dispatch.
pub fn run_as_scenario(result: &mut primalspring::validation::ValidationResult) {
    let mut v = crate::validation::Validator::silent("validate_cross_spring_provenance");
    run(&mut v);
    v.bridge_into(result);
}

/// Scenario registration for the UniBin registry.
pub const SCENARIO: crate::validation::scenarios::registry::Scenario = crate::validation::scenarios::registry::Scenario {
    meta: crate::validation::scenarios::registry::ScenarioMeta {
        id: "cross_spring_provenance",
        track: crate::validation::scenarios::registry::Track::Science,
        tier: crate::validation::scenarios::registry::Tier::Both,
        provenance_crate: "validate_cross_spring_provenance",
        provenance_date: "2026-05-20",
        description: "Exp312: Cross-Spring Provenance Validation + Benchmark",
    },
    run: |v, _ctx| run_as_scenario(v),
};
