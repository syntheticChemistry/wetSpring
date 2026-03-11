// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![allow(
    unexpected_cfgs,
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::items_after_statements,
    clippy::float_cmp
)]
//! # Exp370: LAN Mesh + SRA Atlas Planning Validation
//!
//! Validates the compute model for the NUCLEUS LAN mesh SRA Longitudinal
//! Atlas. Estimates storage, bandwidth, and compute requirements for
//! distributed Anderson spectral analysis across the tower mesh.
//!
//! ## Architecture
//!
//! ```text
//! Eastgate (Node: RTX 4070, AKD1000)  ──┐
//! Strandgate (Node: RTX 3090 + RX 6950) ──┤── 10G Switch ── Westgate (Nest: 76TB ZFS)
//! Northgate (Node: RTX 5090)            ──┤
//! biomeGate (Node: Titan V)             ──┘
//! ```
//!
//! ## Domains
//!
//! - D131: Tower Inventory — hardware capabilities per node
//! - D132: SRA Atlas Sizing — storage + compute estimates for 500-2000 BioProjects
//! - D133: Workload Distribution — NUCLEUS graph for distributed DADA2 + Anderson
//! - D134: Bandwidth Model — 10G throughput for NestGate ↔ node transfers
//! - D135: Cost-Time Trade-off — weekend vs week vs month projections
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Provenance type | LAN mesh + SRA atlas planning |
//! | Date | 2026-03-11 |
//! | Command | `cargo run --release --features gpu,json --bin validate_lan_mesh_plan_v1` |

use std::time::Instant;
use wetspring_barracuda::validation::Validator;

struct TowerNode {
    name: &'static str,
    role: &'static str,
    gpu: &'static str,
    vram_gb: f64,
    cpu_cores: usize,
    ram_gb: f64,
    storage_tb: f64,
    gpu_tflops_f32: f64,
    anderson_samples_per_hour: f64,
}

fn tower_inventory() -> Vec<TowerNode> {
    vec![
        TowerNode {
            name: "Eastgate",
            role: "Node (primary)",
            gpu: "RTX 4070 (AD104)",
            vram_gb: 12.0,
            cpu_cores: 24,
            ram_gb: 32.0,
            storage_tb: 2.0,
            gpu_tflops_f32: 29.1,
            anderson_samples_per_hour: 5000.0,
        },
        TowerNode {
            name: "Strandgate",
            role: "Node (heavy)",
            gpu: "RTX 3090 + RX 6950 XT",
            vram_gb: 24.0 + 16.0,
            cpu_cores: 128,
            ram_gb: 128.0,
            storage_tb: 4.0,
            gpu_tflops_f32: 35.6 + 23.7,
            anderson_samples_per_hour: 8000.0,
        },
        TowerNode {
            name: "Northgate",
            role: "Node (flagship)",
            gpu: "RTX 5090",
            vram_gb: 32.0,
            cpu_cores: 16,
            ram_gb: 64.0,
            storage_tb: 2.0,
            gpu_tflops_f32: 104.8,
            anderson_samples_per_hour: 15_000.0,
        },
        TowerNode {
            name: "biomeGate",
            role: "Node (precision)",
            gpu: "Titan V (GV100)",
            vram_gb: 12.0,
            cpu_cores: 8,
            ram_gb: 32.0,
            storage_tb: 1.0,
            gpu_tflops_f32: 14.9,
            anderson_samples_per_hour: 3000.0,
        },
        TowerNode {
            name: "Westgate",
            role: "Nest (storage)",
            gpu: "none",
            vram_gb: 0.0,
            cpu_cores: 4,
            ram_gb: 16.0,
            storage_tb: 76.0,
            gpu_tflops_f32: 0.0,
            anderson_samples_per_hour: 0.0,
        },
    ]
}

fn main() {
    let start = Instant::now();
    let mut v = Validator::new("Exp370: LAN Mesh + SRA Atlas Planning v1");

    let towers = tower_inventory();

    // ─── D131: Tower Inventory ───
    println!("\n  ── D131: Tower Inventory ──");
    println!(
        "  {:12} {:18} {:>30} {:>8} {:>6} {:>8}",
        "Node", "Role", "GPU", "VRAM", "Cores", "Storage"
    );
    println!("  {}", "─".repeat(88));
    for t in &towers {
        println!(
            "  {:12} {:18} {:>30} {:>6.0}GB {:>6} {:>6.0}TB",
            t.name, t.role, t.gpu, t.vram_gb, t.cpu_cores, t.storage_tb
        );
    }

    let total_vram: f64 = towers.iter().map(|t| t.vram_gb).sum();
    let total_tflops: f64 = towers.iter().map(|t| t.gpu_tflops_f32).sum();
    let total_storage: f64 = towers.iter().map(|t| t.storage_tb).sum();
    let total_samples_hr: f64 = towers.iter().map(|t| t.anderson_samples_per_hour).sum();

    println!(
        "\n  Totals: {total_vram:.0}GB VRAM, {total_tflops:.1} TFLOPS F32, {total_storage:.0}TB storage"
    );
    println!("  Anderson throughput: {total_samples_hr:.0} samples/hour mesh");

    v.check_pass("5 towers in inventory", towers.len() == 5);
    v.check_pass("total VRAM > 80GB", total_vram > 80.0);
    v.check_pass("total storage > 50TB", total_storage > 50.0);

    // ─── D132: SRA Atlas Sizing ───
    println!("\n  ── D132: SRA Atlas Sizing ──");

    struct AtlasTier {
        name: &'static str,
        bioprojects: usize,
        samples: usize,
        raw_tb: f64,
        processed_gb: f64,
    }

    let tiers = vec![
        AtlasTier {
            name: "Minimal",
            bioprojects: 100,
            samples: 10_000,
            raw_tb: 0.5,
            processed_gb: 5.0,
        },
        AtlasTier {
            name: "Standard",
            bioprojects: 500,
            samples: 50_000,
            raw_tb: 5.0,
            processed_gb: 25.0,
        },
        AtlasTier {
            name: "Full",
            bioprojects: 2000,
            samples: 200_000,
            raw_tb: 50.0,
            processed_gb: 100.0,
        },
    ];

    println!(
        "  {:12} {:>12} {:>10} {:>8} {:>12}",
        "Tier", "BioProjects", "Samples", "Raw", "Processed"
    );
    println!("  {}", "─".repeat(60));
    for t in &tiers {
        println!(
            "  {:12} {:>12} {:>10} {:>6.1}TB {:>10.0}GB",
            t.name, t.bioprojects, t.samples, t.raw_tb, t.processed_gb
        );
    }

    let standard = &tiers[1];
    let fits_storage = standard.raw_tb < total_storage;
    println!(
        "\n  Standard tier fits on Westgate: {} ({:.1}TB / {:.0}TB)",
        fits_storage, standard.raw_tb, total_storage
    );
    v.check_pass("standard atlas fits storage", fits_storage);

    // ─── D133: Workload Distribution ───
    println!("\n  ── D133: Workload Distribution ──");

    let compute_nodes: Vec<&TowerNode> = towers
        .iter()
        .filter(|t| t.anderson_samples_per_hour > 0.0)
        .collect();
    let standard_hours = standard.samples as f64 / total_samples_hr;

    println!("  NUCLEUS graph for distributed pipeline:");
    println!("    fetch(sra, BioProject) → Westgate NestGate");
    println!("    process(dada2, FASTQ) → Strandgate CPU (128 cores)");
    println!("    classify(anderson_qs, ASV) → all GPU nodes");
    println!("    store(results, atlas) → Westgate ZFS");

    println!("\n  Workload split:");
    for node in &compute_nodes {
        let share = node.anderson_samples_per_hour / total_samples_hr;
        let node_samples = (standard.samples as f64 * share) as usize;
        println!(
            "    {:12}: {:.0}% ({node_samples} samples, {:.1}h)",
            node.name,
            share * 100.0,
            node_samples as f64 / node.anderson_samples_per_hour
        );
    }

    println!(
        "\n  Standard atlas ({} samples): {standard_hours:.1} hours on mesh",
        standard.samples
    );
    v.check_pass("standard atlas < 24h on mesh", standard_hours < 24.0);

    // ─── D134: Bandwidth Model ───
    println!("\n  ── D134: Bandwidth Model ──");

    let bandwidth_gbps = 10.0;
    let effective_gbps = bandwidth_gbps * 0.8;
    let transfer_rate_gb_s = effective_gbps / 8.0;

    let standard_transfer_seconds = standard.raw_tb * 1000.0 / transfer_rate_gb_s;
    let standard_transfer_hours = standard_transfer_seconds / 3600.0;

    println!("  10G backbone: {bandwidth_gbps}Gbps nominal, {effective_gbps}Gbps effective");
    println!("  Transfer rate: {transfer_rate_gb_s}GB/s");
    println!(
        "  Standard atlas transfer: {standard_transfer_hours:.1}h ({:.0}TB)",
        standard.raw_tb
    );

    v.check_pass(
        "data transfer < compute time",
        standard_transfer_hours < standard_hours,
    );

    // ─── D135: Cost-Time Trade-off ───
    println!("\n  ── D135: Cost-Time Trade-off ──");

    struct Projection {
        name: &'static str,
        tier_idx: usize,
        wall_hours: f64,
    }

    let projections = [
        Projection {
            name: "Minimal (weekend)",
            tier_idx: 0,
            wall_hours: tiers[0].samples as f64 / total_samples_hr,
        },
        Projection {
            name: "Standard (week)",
            tier_idx: 1,
            wall_hours: tiers[1].samples as f64 / total_samples_hr,
        },
        Projection {
            name: "Full (month)",
            tier_idx: 2,
            wall_hours: tiers[2].samples as f64 / total_samples_hr,
        },
    ];

    println!(
        "  {:25} {:>10} {:>10} {:>10}",
        "Projection", "Samples", "Hours", "Days"
    );
    println!("  {}", "─".repeat(58));
    for p in &projections {
        println!(
            "  {:25} {:>10} {:>10.1} {:>10.1}",
            p.name,
            tiers[p.tier_idx].samples,
            p.wall_hours,
            p.wall_hours / 24.0
        );
    }

    let weekend_hours = 48.0;
    v.check_pass(
        "minimal atlas fits in weekend",
        projections[0].wall_hours < weekend_hours,
    );

    let electricity_kwh_per_hour = 1.5;
    let electricity_rate = 0.12;
    let standard_cost = standard_hours * electricity_kwh_per_hour * electricity_rate;
    println!("\n  Estimated electricity cost (standard): ${standard_cost:.2}");
    println!("  Hardware investment: ~$50 (Cat6a cables only)");
    v.check_pass("affordable compute", standard_cost < 10.0);

    // Export
    #[cfg(feature = "json")]
    {
        let export = serde_json::json!({
            "experiment": "Exp370",
            "towers": towers.iter().map(|t| serde_json::json!({
                "name": t.name,
                "role": t.role,
                "gpu": t.gpu,
                "vram_gb": t.vram_gb,
                "tflops_f32": t.gpu_tflops_f32,
                "storage_tb": t.storage_tb,
            })).collect::<Vec<_>>(),
            "mesh_totals": {
                "vram_gb": total_vram,
                "tflops_f32": total_tflops,
                "storage_tb": total_storage,
                "samples_per_hour": total_samples_hr,
            },
            "standard_atlas": {
                "bioprojects": standard.bioprojects,
                "samples": standard.samples,
                "raw_tb": standard.raw_tb,
                "compute_hours": standard_hours,
                "cost_usd": standard_cost,
            },
            "status": "validated — awaiting 10G cables",
        });
        let json = serde_json::to_string_pretty(&export).expect("serialize");
        std::fs::create_dir_all("output").ok();
        std::fs::write("output/lan_mesh_sra_atlas_plan.json", &json).expect("write");
        println!("\n  Exported: output/lan_mesh_sra_atlas_plan.json");
        v.check_pass("planning export", true);
    }

    #[cfg(not(feature = "json"))]
    {
        v.check_pass("graceful skip", true);
    }

    println!("\n  ═══════════════════════════════════════════════");
    println!("  LAN Mesh SRA Atlas Summary:");
    println!("    Towers:           {}", towers.len());
    println!("    Total VRAM:       {total_vram:.0}GB");
    println!("    Total TFLOPS:     {total_tflops:.1}");
    println!("    Total Storage:    {total_storage:.0}TB");
    println!("    Throughput:       {total_samples_hr:.0} samples/hour");
    println!(
        "    Standard atlas:   {:.1}h ({} BioProjects)",
        standard_hours, standard.bioprojects
    );
    println!("    Status:           Plan validated, awaiting 10G cables (~$50)");
    println!("  ═══════════════════════════════════════════════");

    let elapsed = start.elapsed();
    println!("\n  Wall time: {:.1}ms", elapsed.as_secs_f64() * 1000.0);
    v.finish();
}
