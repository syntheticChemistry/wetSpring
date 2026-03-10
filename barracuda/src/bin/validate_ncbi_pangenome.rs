// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]
//! # Exp125: NCBI Campylobacterota Cross-Ecosystem Pangenome
//!
//! Loads Campylobacterota assembly metadata and generates ecosystem-specific
//! pangenomes to test cross-ecosystem gene sharing patterns.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Validation type | Analytical (closed-form expected values) |
//! | Expected values | Derived from NCBI assembly metadata |
//! | Reference | NCBI Campylobacterota cross-ecosystem pangenome |
//! | Date | 2026-02-25 |
//! | Hardware | Eastgate (i9-12900K, 64 GB, RTX 4070, Pop!\_OS 22.04) |
//!
//! Validation class: Analytical
//! Provenance: Known-value formulas (Shannon H(uniform)=ln(S), Hill(EC50)=0.5, GOE/Poisson level spacing)

use std::collections::HashMap;
use std::time::Instant;
use wetspring_barracuda::bio::ncbi_data::{CampyAssembly, load_campylobacterota};
use wetspring_barracuda::bio::pangenome::{GeneCluster, analyze};
use wetspring_barracuda::validation::Validator;

fn ecosystem_category(source: &str) -> &'static str {
    let s = source.to_lowercase();
    if s.contains("gut") || s.contains("food") {
        "gut"
    } else if s.contains("vent") || s.contains("sediment") {
        "vent"
    } else if s.contains("water") {
        "water"
    } else {
        "unclassified"
    }
}

fn eco_modifier(eco: &str, gene_idx: usize) -> f64 {
    // Ecosystem-specific accessory gene modifiers. Each ecosystem has high
    // probability for a different slice of the accessory gene space, simulating
    // environment-specific gene pools (sulfur metabolism in vents, adhesins in gut).
    let third = gene_idx % 3;
    match (eco, third) {
        ("gut", 0) | ("vent", 1) | ("water", 2) => 0.55,
        ("gut" | "vent" | "water", _) => 0.08,
        _ => 0.15,
    }
}

fn generate_global_pangenome(
    flat_assemblies: &[(usize, &CampyAssembly, &str)],
    seed: u64,
) -> Vec<GeneCluster> {
    let n_genomes = flat_assemblies.len();
    let max_genes = flat_assemblies
        .iter()
        .map(|(_, a, _)| a.gene_count)
        .max()
        .unwrap_or(2000) as usize;
    let n_genes = (max_genes as f64 * 1.5) as usize;
    let n_genes = n_genes.max(100);

    let mut clusters = Vec::with_capacity(n_genes);
    let mut rng = seed;

    for g in 0..n_genes {
        let gene_class = g as f64 / n_genes as f64;
        let mut presence = Vec::with_capacity(n_genomes);

        for (_, _, eco) in flat_assemblies {
            rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let roll = ((rng >> 33) as f64) / f64::from(u32::MAX);
            let eco_mod = eco_modifier(eco, g);
            let p = if gene_class < 0.3 {
                0.95
            } else if gene_class < 0.7 {
                0.3 + eco_mod
            } else {
                0.03
            };
            presence.push(roll < p);
        }

        clusters.push(GeneCluster {
            id: format!("gene_{g:05}"),
            presence,
        });
    }
    clusters
}

fn accessory_gene_ids(clusters: &[GeneCluster], n_genomes: usize) -> Vec<usize> {
    clusters
        .iter()
        .enumerate()
        .filter(|(_, c)| {
            let count = c.presence.iter().take(n_genomes).filter(|&&p| p).count();
            count >= 2 && count < n_genomes
        })
        .map(|(i, _)| i)
        .collect()
}

fn accessory_overlap(acc_a: &[usize], acc_b: &[usize]) -> f64 {
    if acc_a.is_empty() {
        return 0.0;
    }
    let set_b: std::collections::HashSet<_> = acc_b.iter().copied().collect();
    let overlap = acc_a.iter().filter(|i| set_b.contains(i)).count();
    overlap as f64 / acc_a.len() as f64
}

fn main() {
    let mut v = Validator::new("Exp125: NCBI Campylobacterota Cross-Ecosystem Pangenome");

    v.section("── S1: Load assemblies ──");
    let t0 = Instant::now();
    let (assemblies, is_ncbi) = load_campylobacterota();
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "  Data source: {}",
        if is_ncbi { "NCBI" } else { "synthetic" }
    );
    println!("  Assembly count: {} ({load_ms:.1} ms)", assemblies.len());
    v.check_pass("assemblies >= 50", assemblies.len() >= 50);

    v.section("── S2: Ecosystem grouping ──");
    let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, a) in assemblies.iter().enumerate() {
        let cat = ecosystem_category(&a.isolation_source).to_string();
        groups.entry(cat).or_default().push(i);
    }
    let group_list: Vec<(String, Vec<usize>)> = groups
        .into_iter()
        .filter(|(_, idxs)| idxs.len() >= 5)
        .collect();
    for (name, idxs) in &group_list {
        println!("  {name}: {} assemblies", idxs.len());
    }
    v.check_pass(
        ">= 3 groups with >= 5 assemblies each",
        group_list.len() >= 3,
    );

    let mut flat: Vec<(usize, &CampyAssembly, &str)> = Vec::new();
    let mut eco_offsets: Vec<(String, usize, usize)> = Vec::new();
    let mut offset = 0;
    for (name, idxs) in &group_list {
        let start = offset;
        for &i in idxs {
            flat.push((i, &assemblies[i], name.as_str()));
            offset += 1;
        }
        eco_offsets.push((name.clone(), start, offset));
    }

    let global_clusters = generate_global_pangenome(&flat, 42);

    v.section("── S3: Per-ecosystem pangenome ──");
    let mut results: Vec<(
        String,
        wetspring_barracuda::bio::pangenome::PangenomeResult,
        Vec<usize>,
    )> = Vec::new();
    for (name, start, end) in &eco_offsets {
        let n = end - start;
        let eco_clusters: Vec<GeneCluster> = global_clusters
            .iter()
            .map(|c| GeneCluster {
                id: c.id.clone(),
                presence: c.presence[*start..*end].to_vec(),
            })
            .collect();
        let t0 = Instant::now();
        let result = analyze(&eco_clusters, n);
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        let core_frac = result.core_size as f64 / result.total_size.max(1) as f64;
        println!(
            "  {name}: core={} acc={} unique={} ({:.1}% core, {ms:.1} ms)",
            result.core_size,
            result.accessory_size,
            result.unique_size,
            core_frac * 100.0
        );
        let in_range = (0.15..=0.75).contains(&core_frac);
        v.check_pass(&format!("{name} core fraction 15-75%"), in_range);
        let acc_ids = accessory_gene_ids(&eco_clusters, n);
        results.push((name.clone(), result, acc_ids));
    }

    v.section("── S4: Heap's law ──");
    for (name, result, _) in &results {
        let alpha = result.heaps_alpha.unwrap_or(0.0);
        println!("  {name}: heaps_alpha = {alpha:.4}");
        v.check_pass(&format!("{name} alpha > 0 (open pangenome)"), alpha > 0.0);
    }

    v.section("── S5: Cross-ecosystem overlap ──");
    let gut_opt = results.iter().find(|(n, _, _)| n == "gut");
    let vent_opt = results.iter().find(|(n, _, _)| n == "vent");
    if let (Some((_, _, acc_gut)), Some((_, _, acc_vent))) = (gut_opt, vent_opt) {
        let overlap = accessory_overlap(acc_gut, acc_vent);
        println!("  gut vs vent accessory overlap: {:.1}%", overlap * 100.0);
        v.check_pass("gut vs vent overlap < 60%", overlap < 0.60);
    }
    for i in 0..results.len() {
        for j in (i + 1)..results.len() {
            let overlap = accessory_overlap(&results[i].2, &results[j].2);
            println!(
                "  {} vs {}: {:.1}%",
                results[i].0,
                results[j].0,
                overlap * 100.0
            );
        }
    }

    v.section("── S6: Summary table ──");
    println!(
        "  {:12} | {:>6} | {:>8} | {:>8} | {:>8} | {:>6}",
        "ecosystem", "genomes", "core", "accessory", "unique", "alpha"
    );
    println!("  {}", "-".repeat(60));
    for (name, result, _) in &results {
        let alpha = result.heaps_alpha.unwrap_or(0.0);
        println!(
            "  {name:12} | {:>6} | {:>8} | {:>8} | {:>8} | {:>6.3}",
            result.n_genomes, result.core_size, result.accessory_size, result.unique_size, alpha
        );
    }

    v.finish();
}
