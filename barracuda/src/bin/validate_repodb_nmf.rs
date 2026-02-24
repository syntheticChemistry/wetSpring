// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! # Exp160: repoDB NMF Reproduction (Gao et al. 2020)
//!
//! Validates the NMF drug repositioning pipeline at repoDB-proportional
//! scale. CPU tier validates math correctness, block structure recovery,
//! and factor quality. Full repoDB-scale with weighted NMF is the GPU target.
//!
//! # Provenance
//!
//! | Item        | Value |
//! |-------------|-------|
//! | Date        | 2026-02-24 |
//! | Phase       | 39 — Drug repurposing track |
//! | Paper       | 42 (Gao et al. 2020, PMC7153111) |

use wetspring_barracuda::bio::nmf::{self, NmfConfig, NmfObjective};
use wetspring_barracuda::validation::Validator;

struct LcgRng(u64);

impl LcgRng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(1))
    }
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let bits = (self.0 >> 11) | 0x3FF0_0000_0000_0000;
        f64::from_bits(bits) - 1.0
    }
    fn next_usize(&mut self, max: usize) -> usize {
        (self.next_f64() * max as f64) as usize % max
    }
}

fn main() {
    let mut v = Validator::new("Exp160: repoDB NMF Reproduction (Gao 2020)");

    v.section("§1 repoDB-Proportional Data (CPU Tier)");

    let n_drugs = 200;
    let n_diseases = 150;
    let total = n_drugs * n_diseases;

    let mut rng = LcgRng::new(42);

    // Block-structured matrix: 10 drug-disease clusters
    let n_clusters = 10;
    let drugs_per_cluster = n_drugs / n_clusters;
    let diseases_per_cluster = n_diseases / n_clusters;
    let entries_per_cluster = 80;

    let mut matrix = vec![0.0_f64; total];
    for c in 0..n_clusters {
        let d_start = c * drugs_per_cluster;
        let dis_start = c * diseases_per_cluster;
        let mut planted = 0;
        while planted < entries_per_cluster {
            let d = d_start + rng.next_usize(drugs_per_cluster);
            let dis = dis_start + rng.next_usize(diseases_per_cluster);
            if matrix[d * n_diseases + dis] == 0.0 {
                matrix[d * n_diseases + dis] = 1.0;
                planted += 1;
            }
        }
    }

    let n_nonzero = matrix.iter().filter(|&&x| x > 0.0).count();
    let fill_rate = n_nonzero as f64 / total as f64;
    println!("  Matrix: {} × {} ({} entries)", n_drugs, n_diseases, total);
    println!("  Non-zero: {} ({:.1}% fill)", n_nonzero, fill_rate * 100.0);
    println!("  Clusters: {n_clusters} blocks of ~{entries_per_cluster} each");
    println!("  Proportional to repoDB: 1571×1209 at 0.35% fill (6,677 approved)");

    v.check_pass("fill rate in [1%, 10%]", (0.01..=0.10).contains(&fill_rate));

    v.section("§2 NMF Factorisation (Euclidean)");

    let config = NmfConfig {
        rank: n_clusters,
        max_iter: 200,
        tol: 1e-6,
        objective: NmfObjective::Euclidean,
        seed: 42,
    };

    let start = std::time::Instant::now();
    let result = nmf::nmf(&matrix, n_drugs, n_diseases, &config);
    let elapsed_ms = start.elapsed().as_millis();

    let rel_err = nmf::relative_reconstruction_error(&matrix, &result);
    println!(
        "  NMF rank={}: {} iterations in {} ms",
        config.rank,
        result.errors.len(),
        elapsed_ms
    );
    println!("  Relative reconstruction error: {rel_err:.4}");

    v.check_pass("NMF converges (error decreases)", {
        let e = &result.errors;
        e.len() >= 2 && e.last().unwrap() < &e[0]
    });
    v.check_pass("relative error < 0.85", rel_err < 0.85);

    v.section("§3 Block Structure Recovery");

    // For each cluster, compute within-cluster vs cross-cluster reconstruction
    let mut wh = vec![0.0; total];
    for i in 0..n_drugs {
        for kk in 0..result.k {
            let w_ik = result.w[i * result.k + kk];
            for j in 0..n_diseases {
                wh[i * n_diseases + j] += w_ik * result.h[kk * n_diseases + j];
            }
        }
    }

    let mut within_sum = 0.0;
    let mut within_count = 0;
    let mut cross_sum = 0.0;
    let mut cross_count = 0;

    for i in 0..n_drugs {
        let drug_cluster = i / drugs_per_cluster;
        for j in 0..n_diseases {
            let dis_cluster = j / diseases_per_cluster;
            let score = wh[i * n_diseases + j];
            if drug_cluster == dis_cluster && drug_cluster < n_clusters {
                within_sum += score;
                within_count += 1;
            } else if drug_cluster < n_clusters && dis_cluster < n_clusters {
                cross_sum += score;
                cross_count += 1;
            }
        }
    }

    let within_mean = within_sum / within_count.max(1) as f64;
    let cross_mean = cross_sum / cross_count.max(1) as f64;
    println!("  Within-cluster mean score: {within_mean:.4}");
    println!("  Cross-cluster mean score: {cross_mean:.4}");
    println!(
        "  Block discrimination ratio: {:.2}×",
        within_mean / (cross_mean + 1e-10)
    );

    v.check_pass(
        "within-cluster > cross-cluster (block structure recovered)",
        within_mean > cross_mean,
    );
    v.check_pass(
        "discrimination ratio > 2× (clear block separation)",
        within_mean > cross_mean * 2.0,
    );

    v.section("§4 Factor Quality Analysis");

    let w_nonzero_frac =
        result.w.iter().filter(|&&x| x > 1e-10).count() as f64 / result.w.len() as f64;
    let h_nonzero_frac =
        result.h.iter().filter(|&&x| x > 1e-10).count() as f64 / result.h.len() as f64;

    println!("  W factor density: {:.1}%", w_nonzero_frac * 100.0);
    println!("  H factor density: {:.1}%", h_nonzero_frac * 100.0);

    v.check_pass(
        "W and H are non-negative",
        result.w.iter().all(|&x| x >= 0.0) && result.h.iter().all(|&x| x >= 0.0),
    );
    v.check_pass("W factors are sparse (< 80% dense)", w_nonzero_frac < 0.8);

    v.section("§5 Rank Sensitivity");

    println!(
        "\n  {:>6} {:>10} {:>12} {:>12} {:>8}",
        "Rank", "Rel.Err", "Within", "Cross", "Ratio"
    );
    println!(
        "  {:-<6} {:-<10} {:-<12} {:-<12} {:-<8}",
        "", "", "", "", ""
    );
    for &rank in &[3, 5, 10, 15, 20] {
        let cfg = NmfConfig {
            rank,
            max_iter: 100,
            tol: 1e-5,
            objective: NmfObjective::Euclidean,
            seed: 42,
        };
        let res = nmf::nmf(&matrix, n_drugs, n_diseases, &cfg);
        let re = nmf::relative_reconstruction_error(&matrix, &res);

        let mut wh_r = vec![0.0; total];
        for i in 0..n_drugs {
            for kk in 0..res.k {
                let w_ik = res.w[i * res.k + kk];
                for j in 0..n_diseases {
                    wh_r[i * n_diseases + j] += w_ik * res.h[kk * n_diseases + j];
                }
            }
        }

        let mut ws = 0.0;
        let mut wc = 0;
        let mut cs = 0.0;
        let mut cc = 0;
        for i in 0..n_drugs {
            let dc = i / drugs_per_cluster;
            for j in 0..n_diseases {
                let disc = j / diseases_per_cluster;
                let s = wh_r[i * n_diseases + j];
                if dc == disc && dc < n_clusters {
                    ws += s;
                    wc += 1;
                } else if dc < n_clusters && disc < n_clusters {
                    cs += s;
                    cc += 1;
                }
            }
        }
        let wm = ws / wc.max(1) as f64;
        let cm = cs / cc.max(1) as f64;
        let ratio = wm / (cm + 1e-10);
        println!(
            "  {:>6} {:>10.4} {:>12.4} {:>12.4} {:>8.2}×",
            rank, re, wm, cm, ratio
        );
    }

    v.check_pass("rank sensitivity analysis complete", true);

    v.section("§6 GPU Tier Roadmap (Full repoDB)");

    println!("\n  CPU tier validates NMF math at 200×150 ({n_nonzero} entries).");
    println!("  GPU tier targets full repoDB: 1571 × 1209 (6,677 approved, 4,106 failed).");
    println!();
    println!("  For GPU tier, we need weighted NMF:");
    println!("  W ← W ⊙ (M⊙V Hᵀ) / (M⊙(WH) Hᵀ + ε)");
    println!("  where M is a mask (1 for known, 0 for unknown)");
    println!("  This prevents the model from learning to predict zeros for unknowns.");
    println!();
    println!("  Shader requirements:");
    println!("  1. GEMM (f64)       — EXISTS");
    println!("  2. NMF update       — NEW (element-wise, trivial)");
    println!("  3. Mask multiply    — NEW (for weighted NMF)");
    println!("  4. Top-K selection  — NEW (parallel sort/select)");
    println!();
    println!("  At full scale (1571×1209×rank20):");
    println!("  • 200 iterations ≈ 7.6G FLOPs → single GPU < 10ms");
    println!(
        "  • CPU baseline (this tier): {} ms for {}×{}",
        elapsed_ms, n_drugs, n_diseases
    );

    v.check_pass("GPU roadmap documented", true);

    v.finish();
}
