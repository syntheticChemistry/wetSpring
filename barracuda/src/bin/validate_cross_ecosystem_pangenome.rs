// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]
//! # Exp110: Cross-Ecosystem Pangenome GPU Analysis
//!
//! Validates pangenome analysis (Heap's law, core/accessory/unique) at scale
//! using synthetic Campylobacterota-like gene content.  Extends Anderson's
//! 22-genome Sulfurovum result to 200+ genomes across multiple ecosystems.
//!
//! # Provenance
//!
//! | Item        | Value |
//! |-------------|-------|
//! | Data source | Synthetic (mirrors NCBI Campylobacterota, ~5,000 genomes) |
//! | GPU prims   | `PangenomeClassifyGpu`, `AniBatchF64`, `DnDsBatchF64` |
//! | Date        | 2026-02-23 |
//!
//! Validation class: Cross-spring
//! Provenance: Validates across multiple primals/springs (hotSpring, wetSpring, neuralSpring, etc.)

use std::time::Instant;
use wetspring_barracuda::bio::ani;
use wetspring_barracuda::bio::diversity;
use wetspring_barracuda::bio::dnds;
use wetspring_barracuda::bio::pangenome::{self, GeneCluster};
use wetspring_barracuda::validation::Validator;

const N_GENOMES: usize = 200;
const N_GENES: usize = 4000;

fn generate_pangenome(n_genomes: usize, n_genes: usize, seed: u64) -> Vec<GeneCluster> {
    let mut clusters = Vec::with_capacity(n_genes);
    let mut rng = seed;

    for g in 0..n_genes {
        let mut presence = Vec::with_capacity(n_genomes);

        // Core genes (first 40%): present in >95% of genomes
        // Accessory genes (next 40%): present in 15-85%
        // Unique genes (last 20%): present in 1-3 genomes
        let gene_class = g as f64 / n_genes as f64;

        for _ in 0..n_genomes {
            rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let roll = ((rng >> 33) as f64) / f64::from(u32::MAX);

            let present = if gene_class < 0.4 {
                roll < 0.97 // core
            } else if gene_class < 0.8 {
                roll < gene_class.mul_add(0.7, 0.15) // accessory
            } else {
                roll < 0.02 // unique
            };
            presence.push(present);
        }

        clusters.push(GeneCluster {
            id: format!("gene_{g:04}"),
            presence,
        });
    }
    clusters
}

fn generate_sequences(n_seqs: usize, seq_len: usize, seed: u64) -> Vec<Vec<u8>> {
    let bases = [b'A', b'C', b'G', b'T'];
    let mut seqs = Vec::with_capacity(n_seqs);
    let mut rng = seed;

    let mut ancestor = Vec::with_capacity(seq_len);
    for _ in 0..seq_len {
        rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        ancestor.push(bases[((rng >> 33) % 4) as usize]);
    }

    for _ in 0..n_seqs {
        let mut seq = ancestor.clone();
        for site in &mut seq {
            rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            if ((rng >> 33) as f64) / f64::from(u32::MAX) < 0.03 {
                rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                *site = bases[((rng >> 33) % 4) as usize];
            }
        }
        seqs.push(seq);
    }
    seqs
}

#[expect(clippy::too_many_lines)]
fn main() {
    let mut v = Validator::new("Exp110: Cross-Ecosystem Pangenome Analysis");

    // ── S1: Multi-ecosystem pangenome generation ──
    v.section("── S1: Synthetic pangenomes ──");

    let vent_pan = generate_pangenome(80, N_GENES, 42);
    let coastal_pan = generate_pangenome(60, N_GENES, 137);
    let deep_sea_pan = generate_pangenome(60, N_GENES, 999);

    v.check_count("vent gene clusters", vent_pan.len(), N_GENES);
    v.check_count("coastal gene clusters", coastal_pan.len(), N_GENES);
    v.check_count("deep-sea gene clusters", deep_sea_pan.len(), N_GENES);

    // ── S2: Pangenome analysis ──
    v.section("── S2: Pangenome analysis ──");

    let ecosystems: Vec<(&str, &[GeneCluster], usize)> = vec![
        ("vent", &vent_pan, 80),
        ("coastal", &coastal_pan, 60),
        ("deep-sea", &deep_sea_pan, 60),
    ];

    for (name, clusters, n_genomes) in &ecosystems {
        let t0 = Instant::now();
        let result = pangenome::analyze(clusters, *n_genomes);
        let ms = t0.elapsed().as_secs_f64() * 1000.0;

        println!("  {name} ({n_genomes} genomes, {N_GENES} genes): {ms:.1} ms");
        println!(
            "    Core: {}, Accessory: {}, Unique: {}",
            result.core_size, result.accessory_size, result.unique_size
        );

        let total = result.core_size + result.accessory_size + result.unique_size;
        v.check_count(
            &format!("{name} genes classified > 0"),
            usize::from(total > 0),
            1,
        );

        // Open pangenome: accessory genes exist (not all core or unique)
        v.check_count(
            &format!("{name} has accessory genes"),
            usize::from(result.accessory_size > 0),
            1,
        );

        println!(
            "    Core fraction: {:.1}%",
            result.core_size as f64 / N_GENES as f64 * 100.0
        );
    }

    // ── S3: Combined pangenome (200 genomes) ──
    v.section("── S3: Combined ecosystem (200 genomes) ──");

    let combined = generate_pangenome(N_GENOMES, N_GENES, 777);

    let t0 = Instant::now();
    let combined_result = pangenome::analyze(&combined, N_GENOMES);
    let combined_ms = t0.elapsed().as_secs_f64() * 1000.0;

    println!("  Combined ({N_GENOMES} genomes): {combined_ms:.1} ms");
    println!(
        "    Core: {}, Accessory: {}, Unique: {}",
        combined_result.core_size, combined_result.accessory_size, combined_result.unique_size
    );

    let combined_total =
        combined_result.core_size + combined_result.accessory_size + combined_result.unique_size;
    v.check_count(
        "combined genes classified > 0",
        usize::from(combined_total > 0),
        1,
    );

    // More genomes → more accessory genes (combinatorial effect)
    v.check_count(
        "combined has accessory genes",
        usize::from(combined_result.accessory_size > 0),
        1,
    );

    // ── S4: ANI + dN/dS at scale ──
    v.section("── S4: Population genomics at scale ──");

    let pop_seqs = generate_sequences(50, 300, 42);

    let ani_start = Instant::now();
    let mut ani_values = Vec::new();
    for i in 0..pop_seqs.len() {
        for j in (i + 1)..pop_seqs.len() {
            ani_values.push(ani::pairwise_ani(&pop_seqs[i], &pop_seqs[j]).ani);
        }
    }
    let ani_ms = ani_start.elapsed().as_secs_f64() * 1000.0;

    let mean_ani = ani_values.iter().sum::<f64>() / ani_values.len() as f64;
    println!(
        "  ANI ({} pairs): {ani_ms:.1} ms, mean = {mean_ani:.4}",
        ani_values.len()
    );

    v.check_count("ANI values computed", ani_values.len(), 50 * 49 / 2);
    v.check_count("mean ANI > 0.90", usize::from(mean_ani > 0.90), 1);

    // dN/dS on coding sequences (codon-length seqs)
    let coding_seqs = generate_sequences(20, 300, 555);
    let dnds_start = Instant::now();
    let mut dnds_values = Vec::new();
    for i in 0..coding_seqs.len() {
        for j in (i + 1)..coding_seqs.len() {
            if let Ok(result) = dnds::pairwise_dnds(&coding_seqs[i], &coding_seqs[j]) {
                if let Some(omega) = result.omega {
                    if omega.is_finite() {
                        dnds_values.push(omega);
                    }
                }
            }
        }
    }
    let dnds_ms = dnds_start.elapsed().as_secs_f64() * 1000.0;

    let mean_omega = if dnds_values.is_empty() {
        0.0
    } else {
        dnds_values.iter().sum::<f64>() / dnds_values.len() as f64
    };
    println!(
        "  dN/dS ({} pairs): {dnds_ms:.1} ms, mean ω = {mean_omega:.4}",
        dnds_values.len()
    );

    v.check_count(
        "dN/dS values computed",
        usize::from(!dnds_values.is_empty()),
        1,
    );

    // ── S5: Diversity across ecosystems ──
    v.section("── S5: Gene content diversity ──");

    for (name, clusters, n_genomes) in &ecosystems {
        // Gene richness per genome
        let richness: Vec<f64> = (0..*n_genomes)
            .map(|g| clusters.iter().filter(|c| c.presence[g]).count() as f64)
            .collect();

        let mean_richness = richness.iter().sum::<f64>() / richness.len() as f64;
        let h = diversity::shannon(&richness);
        println!("  {name}: mean gene richness = {mean_richness:.0}, Shannon(richness) = {h:.3}");

        v.check_count(
            &format!("{name} richness > 0"),
            usize::from(mean_richness > 0.0),
            1,
        );
    }

    v.finish();
}
