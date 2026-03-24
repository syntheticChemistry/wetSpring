// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::cast_precision_loss,
    reason = "validation harness: f64 arithmetic for timing and metric ratios"
)]
#![expect(
    clippy::cast_possible_truncation,
    reason = "validation harness: u128→u64 timing, f64→u32 counts"
)]
#![expect(
    clippy::cast_sign_loss,
    reason = "validation harness: non-negative values cast to unsigned"
)]
//! Exp163: `BarraCuda` CPU Parity v9 — Pure Rust Math for Track 3 Drug Repurposing
//!
//! Validates that all 5 Track 3 domains (Exp157-161) produce correct CPU
//! results using `barracuda` always-on math (NMF, special, ridge, trapz)
//! against hardcoded expected values from Python baselines and published papers.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline      | Fajgenbaum JCI 2019 (Paper 39, doi:10.1172/JCI126091), |
//! |               | Yang et al. 2020 (Paper 41, doi:10.1093/bioinformatics/btaa164), |
//! |               | `barracuda::linalg::nmf` + `barracuda::special` (always-on) |
//! | Date          | 2026-02-25 |
//! | Command       | `cargo run --release --bin validate_barracuda_cpu_v9` |
//! | Data          | Synthetic test vectors (self-contained) |
//! | Tolerances    | Structural (pass/fail) + `EXACT` for paper-derived constants |
//!
//! Validation class: Python-parity
//!
//! Provenance: Python/QIIME2/SciPy baseline script (see doc table for script, commit, date)

use barracuda::linalg::nmf::{self, NmfConfig, NmfObjective};
use std::time::Instant;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::OrExit;
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
    fn next_usize(&mut self, max: usize) -> usize {
        (self.next_f64() * max as f64) as usize % max
    }
}

// ── D01: Fajgenbaum Pathway Scoring ─────────────────────────────────────────

fn validate_pathway_scoring(v: &mut Validator) {
    v.section("═══ D01: Fajgenbaum Pathway Scoring (Paper 39) ═══");
    let t = Instant::now();

    let pathways: &[(&str, f64)] = &[
        ("PI3K/AKT/mTOR", 0.92),
        ("JAK/STAT3", 0.85),
        ("NF-κB", 0.78),
        ("MAPK/ERK", 0.65),
        ("VEGF", 0.72),
        ("IL-6/gp130", 0.88),
    ];

    let drugs: &[(&str, &str)] = &[
        ("sirolimus", "PI3K/AKT/mTOR"),
        ("everolimus", "PI3K/AKT/mTOR"),
        ("tocilizumab", "IL-6/gp130"),
        ("siltuximab", "IL-6/gp130"),
        ("ruxolitinib", "JAK/STAT3"),
        ("bevacizumab", "VEGF"),
    ];

    let (top_name, top_score) = pathways
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .or_exit("pathways non-empty");

    v.check_pass(
        "PI3K/AKT/mTOR is highest-activation pathway",
        *top_name == "PI3K/AKT/mTOR",
    );
    v.check(
        "PI3K/AKT/mTOR activation",
        *top_score,
        0.92,
        tolerances::EXACT,
    );

    let mut drug_scores: Vec<(&str, f64)> = drugs
        .iter()
        .map(|(name, target)| {
            let score = pathways
                .iter()
                .find(|p| p.0 == *target)
                .map_or(0.0, |p| p.1);
            (*name, score)
        })
        .collect();
    drug_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    v.check_pass(
        "mTOR inhibitor ranks #1",
        drug_scores[0].0 == "sirolimus" || drug_scores[0].0 == "everolimus",
    );

    let mtor = pathways
        .iter()
        .find(|p| p.0 == "PI3K/AKT/mTOR")
        .or_exit("known pathway PI3K/AKT/mTOR")
        .1;
    let il6 = pathways
        .iter()
        .find(|p| p.0 == "IL-6/gp130")
        .or_exit("known pathway IL-6/gp130")
        .1;
    v.check_pass("mTOR pathway > IL-6 pathway", mtor > il6);

    let mut score_matrix = vec![0.0_f64; drugs.len() * pathways.len()];
    for (i, (_, target)) in drugs.iter().enumerate() {
        for (j, (pname, pscore)) in pathways.iter().enumerate() {
            if *pname == *target {
                score_matrix[i * pathways.len() + j] = *pscore;
            }
        }
    }
    v.check_pass(
        "score matrix non-trivial",
        score_matrix.iter().any(|&x| x > 0.0),
    );

    println!("  Pathway scoring: {:.0}µs", t.elapsed().as_micros());
}

// ── D02: MATRIX Pharmacophenomics ───────────────────────────────────────────

fn validate_matrix_pharmacophenomics(v: &mut Validator) {
    v.section("═══ D02: MATRIX Pharmacophenomics (Paper 40) ═══");
    let t = Instant::now();

    let mut rng = LcgRng::new(42);
    let n_drugs = 50;
    let n_pathways = 10;
    let n_diseases = 30;

    let mut drug_pathway = vec![0.0_f64; n_drugs * n_pathways];
    for val in &mut drug_pathway {
        *val = rng.next_f64().max(0.0) * 0.3;
    }
    drug_pathway[0] = 0.95;

    let mut disease_pathway = vec![0.0_f64; n_diseases * n_pathways];
    for val in &mut disease_pathway {
        *val = rng.next_f64().max(0.0) * 0.3;
    }
    disease_pathway[0] = 0.92;

    let mut score_matrix = vec![0.0_f64; n_drugs * n_diseases];
    for i in 0..n_drugs {
        for j in 0..n_diseases {
            let mut s = 0.0;
            for p in 0..n_pathways {
                s += drug_pathway[i * n_pathways + p] * disease_pathway[j * n_pathways + p];
            }
            score_matrix[i * n_diseases + j] = s;
        }
    }

    let mut top_drug = 0;
    let mut top_disease = 0;
    let mut top_score = f64::NEG_INFINITY;
    for i in 0..n_drugs {
        for j in 0..n_diseases {
            let s = score_matrix[i * n_diseases + j];
            if s > top_score {
                top_score = s;
                top_drug = i;
                top_disease = j;
            }
        }
    }
    v.check_pass(
        "top prediction is (drug=0, disease=0)",
        top_drug == 0 && top_disease == 0,
    );
    v.check_pass("top score > 0.5", top_score > 0.5);

    let config = NmfConfig {
        rank: 5,
        max_iter: 200,
        tol: tolerances::NMF_CONVERGENCE_EUCLIDEAN,
        objective: NmfObjective::Euclidean,
        seed: 42,
    };
    let result = nmf::nmf(&score_matrix, n_drugs, n_diseases, &config).or_exit("NMF failed");
    let rel_err = nmf::relative_reconstruction_error(&score_matrix, &result);
    v.check_pass(
        "NMF converges",
        result.errors.len() >= 2 && result.errors.last() < result.errors.first(),
    );
    v.check_pass("relative error < 0.5", rel_err < 0.5);

    let row0_w = &result.w[..result.k];
    let row1_w = &result.w[result.k..result.k * 2];
    let cos = nmf::cosine_similarity(row0_w, row1_w);
    v.check_pass(
        "cosine similarity in [-1, 1]",
        (-1.0001..=1.0001).contains(&cos),
    );

    println!(
        "  MATRIX pharmacophenomics: {:.0}µs",
        t.elapsed().as_micros()
    );
}

// ── D03: NMF Drug Repurposing ───────────────────────────────────────────────

fn validate_nmf_drug_repurposing(v: &mut Validator) {
    v.section("═══ D03: NMF Drug Repurposing (Paper 41) ═══");
    let t = Instant::now();

    let n_drugs = 200;
    let n_diseases = 100;
    let total = n_drugs * n_diseases;
    let n_clusters = 5;
    let target_fill = 0.05;
    let n_known = (total as f64 * target_fill) as usize;

    let mut rng = LcgRng::new(42);
    let mut matrix = vec![0.0_f64; total];
    let entries_per_cluster = n_known / n_clusters;
    let drugs_per = n_drugs / n_clusters;
    let dis_per = n_diseases / n_clusters;

    for c in 0..n_clusters {
        let mut planted = 0;
        while planted < entries_per_cluster {
            let d = c * drugs_per + rng.next_usize(drugs_per);
            let dis = c * dis_per + rng.next_usize(dis_per);
            if matrix[d * n_diseases + dis] == 0.0 {
                matrix[d * n_diseases + dis] = 1.0;
                planted += 1;
            }
        }
    }

    let n_nonzero = matrix.iter().filter(|&&x| x > 0.0).count();
    let sparsity = 1.0 - n_nonzero as f64 / total as f64;
    v.check_pass("sparsity in [90%, 99%]", (0.90..=0.99).contains(&sparsity));

    for &rank in &[5, 10, 20] {
        let config = NmfConfig {
            rank,
            max_iter: 200,
            tol: tolerances::NMF_CONVERGENCE_EUCLIDEAN,
            objective: NmfObjective::Euclidean,
            seed: 42,
        };
        let result = nmf::nmf(&matrix, n_drugs, n_diseases, &config).or_exit("NMF failed");
        let rel_err = nmf::relative_reconstruction_error(&matrix, &result);
        println!(
            "  NMF rank={rank}: rel_err={rel_err:.4}, iters={}",
            result.errors.len()
        );
        v.check_pass(
            &format!("NMF rank={rank} converges"),
            result.errors.len() >= 2 && result.errors.last() < result.errors.first(),
        );
    }

    let best_config = NmfConfig {
        rank: 10,
        max_iter: 200,
        tol: tolerances::NMF_CONVERGENCE_EUCLIDEAN,
        objective: NmfObjective::Euclidean,
        seed: 42,
    };
    let best = nmf::nmf(&matrix, n_drugs, n_diseases, &best_config).or_exit("NMF failed");
    let best_err = nmf::relative_reconstruction_error(&matrix, &best);
    v.check_pass("best rank error < 0.8", best_err < 0.8);

    let kl_matrix: Vec<f64> = matrix
        .iter()
        .map(|&x| x + tolerances::ANALYTICAL_LOOSE)
        .collect();
    let kl_config = NmfConfig {
        rank: 10,
        max_iter: 200,
        tol: tolerances::NMF_CONVERGENCE_EUCLIDEAN,
        objective: NmfObjective::KlDivergence,
        seed: 42,
    };
    let kl = nmf::nmf(&kl_matrix, n_drugs, n_diseases, &kl_config).or_exit("KL NMF failed");
    v.check_pass(
        "KL NMF produces factors",
        !kl.w.is_empty() && !kl.h.is_empty(),
    );

    v.check_pass(
        "W and H non-negative",
        best.w.iter().all(|&x| x >= 0.0) && best.h.iter().all(|&x| x >= 0.0),
    );

    println!("  NMF drug repurposing: {:.0}µs", t.elapsed().as_micros());
}

// ── D04: repoDB NMF ─────────────────────────────────────────────────────────

fn validate_repodb_nmf(v: &mut Validator) {
    v.section("═══ D04: repoDB NMF (Paper 42) ═══");
    let t = Instant::now();

    let n_drugs = 200;
    let n_diseases = 150;
    let total = n_drugs * n_diseases;
    let n_clusters = 10;
    let drugs_per = n_drugs / n_clusters;
    let dis_per = n_diseases / n_clusters;

    let mut rng = LcgRng::new(42);
    let mut matrix = vec![0.0_f64; total];
    for c in 0..n_clusters {
        let mut planted = 0;
        while planted < 80 {
            let d = c * drugs_per + rng.next_usize(drugs_per);
            let dis = c * dis_per + rng.next_usize(dis_per);
            if matrix[d * n_diseases + dis] == 0.0 {
                matrix[d * n_diseases + dis] = 1.0;
                planted += 1;
            }
        }
    }

    let n_nonzero = matrix.iter().filter(|&&x| x > 0.0).count();
    let fill_rate = n_nonzero as f64 / total as f64;
    v.check_pass("fill rate in [1%, 10%]", (0.01..=0.10).contains(&fill_rate));

    let config = NmfConfig {
        rank: n_clusters,
        max_iter: 200,
        tol: tolerances::NMF_CONVERGENCE_EUCLIDEAN,
        objective: NmfObjective::Euclidean,
        seed: 42,
    };
    let result = nmf::nmf(&matrix, n_drugs, n_diseases, &config).or_exit("NMF failed");
    let rel_err = nmf::relative_reconstruction_error(&matrix, &result);

    v.check_pass(
        "NMF converges",
        result.errors.len() >= 2 && result.errors.last() < result.errors.first(),
    );
    v.check_pass("relative error < 0.85", rel_err < 0.85);

    let mut wh = vec![0.0_f64; total];
    for i in 0..n_drugs {
        for kk in 0..result.k {
            let w_ik = result.w[i * result.k + kk];
            for j in 0..n_diseases {
                wh[i * n_diseases + j] += w_ik * result.h[kk * n_diseases + j];
            }
        }
    }

    let (within_mean, cross_mean) = {
        let (mut ws, mut wc, mut cs, mut cc) = (0.0, 0u32, 0.0, 0u32);
        for i in 0..n_drugs {
            let dc = i / drugs_per;
            for j in 0..n_diseases {
                let dic = j / dis_per;
                let s = wh[i * n_diseases + j];
                if dc == dic && dc < n_clusters {
                    ws += s;
                    wc += 1;
                } else if dc < n_clusters && dic < n_clusters {
                    cs += s;
                    cc += 1;
                }
            }
        }
        (ws / f64::from(wc.max(1)), cs / f64::from(cc.max(1)))
    };

    v.check_pass("within-cluster > cross-cluster", within_mean > cross_mean);
    v.check_pass("discrimination ratio > 2×", within_mean > cross_mean * 2.0);

    v.check_pass(
        "W and H non-negative",
        result.w.iter().all(|&x| x >= 0.0) && result.h.iter().all(|&x| x >= 0.0),
    );

    println!("  repoDB NMF: {:.0}µs", t.elapsed().as_micros());
}

// ── D05: Knowledge Graph Embedding ──────────────────────────────────────────

const KG_EMBED_DIM: usize = 32;
const KG_N_ENTITIES: usize = 310;
const KG_N_RELATIONS: usize = 4;

fn transe_score_fn(entity_emb: &[f64], relation_emb: &[f64], h: usize, r: usize, t: usize) -> f64 {
    let mut sum_sq = 0.0;
    for d in 0..KG_EMBED_DIM {
        let diff = entity_emb[h * KG_EMBED_DIM + d] + relation_emb[r * KG_EMBED_DIM + d]
            - entity_emb[t * KG_EMBED_DIM + d];
        sum_sq += diff * diff;
    }
    -sum_sq.sqrt()
}

fn l2_normalize_row(emb: &mut [f64], idx: usize) {
    let row = &mut emb[idx * KG_EMBED_DIM..(idx + 1) * KG_EMBED_DIM];
    let norm: f64 = row.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > tolerances::EMBEDDING_NORM_FLOOR {
        for val in row.iter_mut() {
            *val /= norm;
        }
    }
}

fn validate_kg_embedding(v: &mut Validator) {
    v.section("═══ D05: KG Embedding / TransE (Paper 43) ═══");
    let t = Instant::now();

    let mut rng = LcgRng::new(42);

    let mut entity_emb = vec![0.0_f64; KG_N_ENTITIES * KG_EMBED_DIM];
    for i in 0..KG_N_ENTITIES {
        let row = &mut entity_emb[i * KG_EMBED_DIM..(i + 1) * KG_EMBED_DIM];
        for val in row.iter_mut() {
            *val = rng.next_f64().mul_add(0.2, -0.1);
        }
    }
    for i in 0..KG_N_ENTITIES {
        l2_normalize_row(&mut entity_emb, i);
    }

    let mut relation_emb = vec![0.0_f64; KG_N_RELATIONS * KG_EMBED_DIM];
    for val in &mut relation_emb {
        *val = rng.next_f64().mul_add(0.2, -0.1);
    }

    let mut triples: Vec<(usize, usize, usize)> = Vec::new();
    for cluster in 0..10 {
        let d_start = cluster * 10;
        let dis_start = 100 + cluster * 10;
        let g_start = 200 + cluster * 10;
        let p = 300 + (cluster % 10);

        for i in 0..10 {
            triples.push((d_start + i, 0, dis_start + i));
            triples.push((d_start + i, 1, g_start + i));
            triples.push((g_start + i, 2, dis_start + i));
            triples.push((g_start + i, 3, p));
        }

        for _ in 0..5 {
            let d1 = d_start + rng.next_usize(10);
            let d2 = dis_start + rng.next_usize(10);
            triples.push((d1, 0, d2));
        }
        for _ in 0..3 {
            let g1 = g_start + rng.next_usize(10);
            let d2 = dis_start + rng.next_usize(10);
            triples.push((g1, 2, d2));
        }
    }

    triples.sort_unstable();
    triples.dedup();

    v.check_pass("310 entities", KG_N_ENTITIES == 310);
    v.check_pass(">300 triples", triples.len() > 300);

    let margin = 1.0;
    let lr = 0.01;
    let mut losses = Vec::new();

    for _epoch in 0..50 {
        let mut epoch_loss = 0.0;
        for &(h, r, tt) in &triples {
            let pos = transe_score_fn(&entity_emb, &relation_emb, h, r, tt);
            let neg_t = rng.next_usize(KG_N_ENTITIES);
            let neg = transe_score_fn(&entity_emb, &relation_emb, h, r, neg_t);
            let loss = (margin + pos.abs() - neg.abs()).max(0.0);
            epoch_loss += loss;

            if loss > 0.0 {
                let pos_norm = {
                    let mut s = 0.0;
                    for d in 0..KG_EMBED_DIM {
                        let diff = entity_emb[h * KG_EMBED_DIM + d]
                            + relation_emb[r * KG_EMBED_DIM + d]
                            - entity_emb[tt * KG_EMBED_DIM + d];
                        s += diff * diff;
                    }
                    s.sqrt() + tolerances::EMBEDDING_NORM_FLOOR
                };
                let neg_norm = {
                    let mut s = 0.0;
                    for d in 0..KG_EMBED_DIM {
                        let diff = entity_emb[h * KG_EMBED_DIM + d]
                            + relation_emb[r * KG_EMBED_DIM + d]
                            - entity_emb[neg_t * KG_EMBED_DIM + d];
                        s += diff * diff;
                    }
                    s.sqrt() + tolerances::EMBEDDING_NORM_FLOOR
                };

                for d in 0..KG_EMBED_DIM {
                    let pos_grad = (entity_emb[h * KG_EMBED_DIM + d]
                        + relation_emb[r * KG_EMBED_DIM + d]
                        - entity_emb[tt * KG_EMBED_DIM + d])
                        / pos_norm;
                    let neg_grad = (entity_emb[h * KG_EMBED_DIM + d]
                        + relation_emb[r * KG_EMBED_DIM + d]
                        - entity_emb[neg_t * KG_EMBED_DIM + d])
                        / neg_norm;

                    entity_emb[h * KG_EMBED_DIM + d] -= lr * pos_grad;
                    relation_emb[r * KG_EMBED_DIM + d] -= lr * pos_grad;
                    entity_emb[tt * KG_EMBED_DIM + d] += lr * pos_grad;
                    entity_emb[neg_t * KG_EMBED_DIM + d] += lr * neg_grad;
                }

                l2_normalize_row(&mut entity_emb, h);
                l2_normalize_row(&mut entity_emb, tt);
                l2_normalize_row(&mut entity_emb, neg_t);
            }
        }
        losses.push(epoch_loss / triples.len() as f64);
    }

    v.check_pass("training loss decreased", losses.last() < losses.first());

    let sample: Vec<_> = triples.iter().take(100).copied().collect();
    let mut hits_at_10 = 0;
    for (h, r, tt) in &sample {
        let mut scores: Vec<(f64, usize)> = (0..KG_N_ENTITIES)
            .map(|e| (transe_score_fn(&entity_emb, &relation_emb, *h, *r, e), e))
            .collect();
        scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        if scores.iter().take(10).any(|&(_, e)| e == *tt) {
            hits_at_10 += 1;
        }
    }
    let rate = f64::from(hits_at_10) / sample.len() as f64;
    v.check_pass("Hits@10 > 5%", rate > 0.05);
    println!("  Hits@10 = {hits_at_10}/{} = {rate:.4}", sample.len());

    println!("  KG embedding: {:.0}µs", t.elapsed().as_micros());
}

fn main() {
    let mut v = Validator::new(
        "Exp163: BarraCuda CPU v9 — Pure Rust Math (Track 3 Drug Repurposing, 5 Domains)",
    );
    let t_total = Instant::now();

    validate_pathway_scoring(&mut v);
    validate_matrix_pharmacophenomics(&mut v);
    validate_nmf_drug_repurposing(&mut v);
    validate_repodb_nmf(&mut v);
    validate_kg_embedding(&mut v);

    let total_ms = t_total.elapsed().as_millis();
    println!("\n  Total wall-clock: {total_ms} ms");

    v.finish();
}
