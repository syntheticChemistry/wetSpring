// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
//! Exp329: metalForge + petalTongue visualization integration.
//!
//! Validates hardware inventory, workload dispatch, and NUCLEUS topology
//! scenarios for petalTongue consumption. Uses live-probed substrates.
//!
//! | Domain | Checks |
//! |--------|--------|
//! | MF1 Inventory | Hardware nodes, memory gauges, capability bars |
//! | MF2 Dispatch  | Workload routing bar + coverage gauge |
//! | MF3 NUCLEUS   | Tower→Node→Nest topology, substrate count gauge |
//! | MF4 JSON      | Scenario serialization round-trips |
//!
//! Provenance: metalForge + petalTongue integration validation

use wetspring_barracuda::visualization::{DataChannel, scenario_with_edges_json};
use wetspring_forge::inventory;
use wetspring_forge::visualization;

#[expect(clippy::too_many_lines, reason = "validation binary — linear checks")]
fn main() {
    let start = std::time::Instant::now();
    let mut passed = 0_u32;
    let mut total = 0_u32;

    println!("═══════════════════════════════════════════════════════════");
    println!("  Exp329: metalForge + petalTongue Visualization");
    println!("═══════════════════════════════════════════════════════════");

    let substrates = inventory::discover();

    // ── MF1: Inventory ──
    println!("\nMF1 — Hardware Inventory");
    let (inv, inv_edges) = visualization::inventory_scenario(&substrates);
    check(
        &mut passed,
        &mut total,
        "inventory: has summary node",
        inv.nodes.iter().any(|n| n.id == "summary"),
    );
    check(
        &mut passed,
        &mut total,
        "inventory: ≥2 nodes (at least CPU + summary)",
        inv.nodes.len() >= 2,
    );
    check(
        &mut passed,
        &mut total,
        "inventory: summary has substrate_counts bar",
        {
            inv.nodes.iter().filter(|n| n.id == "summary").any(|n| {
                n.data_channels
                    .iter()
                    .any(|ch| matches!(ch, DataChannel::Bar { id, .. } if id == "substrate_counts"))
            })
        },
    );
    check(
        &mut passed,
        &mut total,
        "inventory: has probe edges",
        !inv_edges.is_empty(),
    );
    check(
        &mut passed,
        &mut total,
        "inventory: domain is default",
        inv.domain == "default",
    );

    // ── MF2: Dispatch ──
    println!("\nMF2 — Workload Dispatch");
    let (disp, _) = visualization::dispatch_scenario(&substrates);
    check(
        &mut passed,
        &mut total,
        "dispatch: has dispatch node",
        disp.nodes.len() == 1,
    );
    check(
        &mut passed,
        &mut total,
        "dispatch: has routable_workloads bar",
        {
            disp.nodes[0]
                .data_channels
                .iter()
                .any(|ch| matches!(ch, DataChannel::Bar { id, .. } if id == "routable_workloads"))
        },
    );
    check(
        &mut passed,
        &mut total,
        "dispatch: has route_coverage gauge",
        {
            disp.nodes[0]
                .data_channels
                .iter()
                .any(|ch| matches!(ch, DataChannel::Gauge { id, .. } if id == "route_coverage"))
        },
    );

    // ── MF3: NUCLEUS ──
    println!("\nMF3 — NUCLEUS Topology");
    let (nuc, nuc_edges) = visualization::nucleus_scenario(&substrates);
    check(
        &mut passed,
        &mut total,
        "nucleus: 3 atomics (tower+node+nest)",
        nuc.nodes.len() == 3,
    );
    check(
        &mut passed,
        &mut total,
        "nucleus: 3 edges (cyclic flow)",
        nuc_edges.len() == 3,
    );
    check(
        &mut passed,
        &mut total,
        "nucleus: tower→node edge",
        nuc_edges
            .iter()
            .any(|e| e.from == "tower" && e.to == "node_atomic"),
    );
    check(
        &mut passed,
        &mut total,
        "nucleus: node→nest edge",
        nuc_edges
            .iter()
            .any(|e| e.from == "node_atomic" && e.to == "nest"),
    );
    check(
        &mut passed,
        &mut total,
        "nucleus: nest→tower edge",
        nuc_edges
            .iter()
            .any(|e| e.from == "nest" && e.to == "tower"),
    );
    check(
        &mut passed,
        &mut total,
        "nucleus: substrate gauge present",
        {
            nuc.nodes.iter().any(|n| n.data_channels.iter().any(|ch| matches!(ch, DataChannel::Gauge { id, .. } if id == "substrates_discovered")))
        },
    );

    // ── MF4: JSON serialization ──
    println!("\nMF4 — JSON Serialization");
    let inv_json = scenario_with_edges_json(&inv, &inv_edges);
    check(
        &mut passed,
        &mut total,
        "inventory JSON: serializes",
        inv_json.is_ok(),
    );
    if let Ok(ref json) = inv_json {
        check(
            &mut passed,
            &mut total,
            "inventory JSON: valid parse",
            serde_json::from_str::<serde_json::Value>(json).is_ok(),
        );
        check(
            &mut passed,
            &mut total,
            "inventory JSON: contains metalforge",
            json.contains("metalforge"),
        );
    }

    let nuc_json = scenario_with_edges_json(&nuc, &nuc_edges);
    check(
        &mut passed,
        &mut total,
        "nucleus JSON: serializes",
        nuc_json.is_ok(),
    );
    if let Ok(ref json) = nuc_json {
        check(
            &mut passed,
            &mut total,
            "nucleus JSON: contains tower",
            json.contains("tower"),
        );
    }

    // ── Summary ──
    let total_ms = start.elapsed().as_millis();
    println!("\n═══════════════════════════════════════════════════════════");
    println!("  Exp329: metalForge + petalTongue: {passed}/{total} checks passed");
    let result = if passed == total { "PASS" } else { "FAIL" };
    println!("  RESULT: {result}");
    println!("═══════════════════════════════════════════════════════════");
    println!("  completed in {total_ms} ms");

    if passed < total {
        std::process::exit(1);
    }
}

fn check(passed: &mut u32, total: &mut u32, label: &str, ok: bool) {
    *total += 1;
    let tag = if ok { "OK" } else { "FAIL" };
    println!("  [{tag}]  {label}");
    if ok {
        *passed += 1;
    }
}
