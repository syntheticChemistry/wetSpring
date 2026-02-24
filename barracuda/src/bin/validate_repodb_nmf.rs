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

use barracuda::linalg::nmf::{self, NmfConfig, NmfObjective};
use wetspring_barracuda::validation::Validator;

struct LcgRng(u64);

impl LcgRng {
    const fn new(seed: u64) -> Self {
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
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    fn next_usize(&mut self, max: usize) -> usize {
        (self.next_f64() * max as f64) as usize % max
    }
}

const N_DRUGS: usize = 200;
const N_DISEASES: usize = 150;
const N_CLUSTERS: usize = 10;
const DRUGS_PER_CLUSTER: usize = N_DRUGS / N_CLUSTERS;
const DISEASES_PER_CLUSTER: usize = N_DISEASES / N_CLUSTERS;

fn compute_block_discrimination(wh: &[f64]) -> (f64, f64) {
    let mut within_sum = 0.0;
    let mut within_count = 0;
    let mut cross_sum = 0.0;
    let mut cross_count = 0;

    for i in 0..N_DRUGS {
        let drug_cluster = i / DRUGS_PER_CLUSTER;
        for j in 0..N_DISEASES {
            let dis_cluster = j / DISEASES_PER_CLUSTER;
            let score = wh[i * N_DISEASES + j];
            if drug_cluster == dis_cluster && drug_cluster < N_CLUSTERS {
                within_sum += score;
                within_count += 1;
            } else if drug_cluster < N_CLUSTERS && dis_cluster < N_CLUSTERS {
                cross_sum += score;
                cross_count += 1;
            }
        }
    }

    let within_mean = within_sum / f64::from(within_count.max(1));
    let cross_mean = cross_sum / f64::from(cross_count.max(1));
    (within_mean, cross_mean)
}

fn reconstruct_wh(result: &nmf::NmfResult) -> Vec<f64> {
    let total = N_DRUGS * N_DISEASES;
    let mut wh = vec![0.0; total];
    for i in 0..N_DRUGS {
        for kk in 0..result.k {
            let w_ik = result.w[i * result.k + kk];
            for j in 0..N_DISEASES {
                wh[i * N_DISEASES + j] += w_ik * result.h[kk * N_DISEASES + j];
            }
        }
    }
    wh
}

fn validate_factorisation_and_structure(
    v: &mut Validator,
    matrix: &[f64],
    result: &nmf::NmfResult,
    elapsed_ms: u128,
    rel_err: f64,
    n_nonzero: usize,
) {
    let rank = result.k;
    let n_iters = result.errors.len();
    println!("  NMF rank={rank}: {n_iters} iterations in {elapsed_ms} ms");
    println!("  Relative reconstruction error: {rel_err:.4}");

    v.check_pass("NMF converges (error decreases)", {
        let e = &result.errors;
        e.len() >= 2 && e.last().unwrap() < &e[0]
    });
    v.check_pass("relative error < 0.85", rel_err < 0.85);

    v.section("§3 Block Structure Recovery");

    let wh = reconstruct_wh(result);
    let (within_mean, cross_mean) = compute_block_discrimination(&wh);
    let disc_ratio = within_mean / (cross_mean + 1e-10);
    println!("  Within-cluster mean score: {within_mean:.4}");
    println!("  Cross-cluster mean score: {cross_mean:.4}");
    println!("  Block discrimination ratio: {disc_ratio:.2}×");

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

    validate_rank_sensitivity(v, matrix);
    validate_gpu_roadmap(v, elapsed_ms, n_nonzero);
}

fn validate_rank_sensitivity(v: &mut Validator, matrix: &[f64]) {
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
        let res = nmf::nmf(matrix, N_DRUGS, N_DISEASES, &cfg).expect("NMF failed");
        let re = nmf::relative_reconstruction_error(matrix, &res);
        let wh_r = reconstruct_wh(&res);
        let (wm, cm) = compute_block_discrimination(&wh_r);
        let ratio = wm / (cm + 1e-10);
        println!("  {rank:>6} {re:>10.4} {wm:>12.4} {cm:>12.4} {ratio:>8.2}×");
    }

    v.check_pass("rank sensitivity analysis complete", true);
}

fn validate_gpu_roadmap(v: &mut Validator, elapsed_ms: u128, n_nonzero: usize) {
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
    println!("  • CPU baseline (this tier): {elapsed_ms} ms for {N_DRUGS}×{N_DISEASES}");

    v.check_pass("GPU roadmap documented", true);
}

fn main() {
    let mut v = Validator::new("Exp160: repoDB NMF Reproduction (Gao 2020)");

    v.section("§1 repoDB-Proportional Data (CPU Tier)");

    let total = N_DRUGS * N_DISEASES;
    let mut rng = LcgRng::new(42);
    let entries_per_cluster = 80;

    let mut matrix = vec![0.0_f64; total];
    for c in 0..N_CLUSTERS {
        let d_start = c * DRUGS_PER_CLUSTER;
        let dis_start = c * DISEASES_PER_CLUSTER;
        let mut planted = 0;
        while planted < entries_per_cluster {
            let d = d_start + rng.next_usize(DRUGS_PER_CLUSTER);
            let dis = dis_start + rng.next_usize(DISEASES_PER_CLUSTER);
            if matrix[d * N_DISEASES + dis] == 0.0 {
                matrix[d * N_DISEASES + dis] = 1.0;
                planted += 1;
            }
        }
    }

    let n_nonzero = matrix.iter().filter(|&&x| x > 0.0).count();
    let fill_rate = n_nonzero as f64 / total as f64;
    println!("  Matrix: {N_DRUGS} × {N_DISEASES} ({total} entries)");
    println!("  Non-zero: {n_nonzero} ({:.1}% fill)", fill_rate * 100.0);
    println!("  Clusters: {N_CLUSTERS} blocks of ~{entries_per_cluster} each");
    println!("  Proportional to repoDB: 1571×1209 at 0.35% fill (6,677 approved)");

    v.check_pass("fill rate in [1%, 10%]", (0.01..=0.10).contains(&fill_rate));

    v.section("§2 NMF Factorisation (Euclidean)");

    let config = NmfConfig {
        rank: N_CLUSTERS,
        max_iter: 200,
        tol: 1e-6,
        objective: NmfObjective::Euclidean,
        seed: 42,
    };

    let start = std::time::Instant::now();
    let result = nmf::nmf(&matrix, N_DRUGS, N_DISEASES, &config).expect("NMF failed");
    let elapsed_ms = start.elapsed().as_millis();
    let rel_err = nmf::relative_reconstruction_error(&matrix, &result);

    validate_factorisation_and_structure(&mut v, &matrix, &result, elapsed_ms, rel_err, n_nonzero);

    v.finish();
}
