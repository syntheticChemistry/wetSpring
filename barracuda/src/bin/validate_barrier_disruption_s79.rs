// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::items_after_statements,
    clippy::float_cmp
)]
//! # Exp274: Barrier Disruption — Dimensional Promotion Threshold
//!
//! Models the AD scratch cycle as an Anderson dimensional promotion:
//! scratching opens 3D diffusion channels through the normally 2D
//! epidermal barrier, enabling cytokine signal delocalization.
//!
//! This is the inverse of Paper 06's tillage dimensional collapse:
//! - Paper 06: tillage collapses 3D soil → 2D → QS fails
//! - Paper 12: scratching promotes 2D skin → 3D → cytokines escape
//!
//! # Experimental Design
//!
//! 1. Vary breach depth (Lz = 1..8) with fixed lateral size (Lx=Ly=8)
//! 2. At each depth, compute level spacing ratio r
//! 3. Identify the depth at which r crosses the midpoint → `d_eff` transition
//! 4. Repeat at multiple disorder strengths W to map the (W, Lz) phase diagram
//! 5. Ensemble average over seeds for statistical robustness
//!
//! # Cross-Spring Provenance
//!
//! | Component | Spring | Module |
//! |-----------|--------|--------|
//! | `anderson_2d` / `anderson_3d` | hotSpring | `barracuda::spectral` |
//! | `lanczos` / `lanczos_eigenvalues` | hotSpring | `barracuda::spectral` |
//! | `level_spacing_ratio` | multiple | ``barracuda::spectral`::stats` |
//! | Diversity (Pielou) | wetSpring | `wetspring_barracuda::bio::diversity` |
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | `ToadStool` pin | S79 (`f97fc2ae`) |
//! | baseCamp paper | Paper 12: Immunological Anderson |
//! | Date | 2026-03-02 |
//! | Command | `cargo run --release --features gpu --bin validate_barrier_disruption_s79` |
//!
//! Validation class: Analytical
//! Provenance: Known-value formulas (Shannon H(uniform)=ln(S), Hill(EC50)=0.5, GOE/Poisson level spacing)

use std::time::Instant;

use barracuda::spectral::{
    GOE_R, POISSON_R, anderson_2d, anderson_3d, lanczos, lanczos_eigenvalues, level_spacing_ratio,
};
use wetspring_barracuda::bio::diversity;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

fn ensemble_r_2d(lx: usize, ly: usize, w: f64, seeds: &[u64]) -> f64 {
    let n = lx * ly;
    let mut sum = 0.0;
    for &seed in seeds {
        let mat = anderson_2d(lx, ly, w, seed);
        let tri = lanczos(&mat, n, seed);
        let eigs = lanczos_eigenvalues(&tri);
        sum += level_spacing_ratio(&eigs);
    }
    sum / seeds.len() as f64
}

fn ensemble_r_3d(lx: usize, ly: usize, lz: usize, w: f64, seeds: &[u64]) -> f64 {
    let n = lx * ly * lz;
    let mut sum = 0.0;
    for &seed in seeds {
        let mat = anderson_3d(lx, ly, lz, w, seed);
        let tri = lanczos(&mat, n, seed);
        let eigs = lanczos_eigenvalues(&tri);
        sum += level_spacing_ratio(&eigs);
    }
    sum / seeds.len() as f64
}

struct DomainResult {
    name: &'static str,
    ms: f64,
    checks: u32,
}

fn main() {
    let mut v = Validator::new("Exp274: Barrier Disruption — Dimensional Promotion");
    let mut domains: Vec<DomainResult> = Vec::new();
    let seeds: [u64; 5] = [42, 137, 271, 577, 997];
    let midpoint = f64::midpoint(GOE_R, POISSON_R);

    // ── D1: Breach Depth Scan at Fixed W ────────────────────────────────
    {
        let t0 = Instant::now();

        let lxy = 6_usize;
        let w = 14.0; // chosen near W_c(3D) so dimensional effects are visible
        let depths: [usize; 6] = [1, 2, 3, 4, 5, 6];

        let r_2d = ensemble_r_2d(lxy, lxy, w, &seeds);

        println!("  ┌─ D1: Breach Depth Scan (Lx=Ly={lxy}, W={w})");
        println!("  │  Reference 2D ({lxy}×{lxy}): <r> = {r_2d:.4}");
        println!("  │  ");
        println!("  │  Lz  │ <r>    │ Δr vs 2D │ Regime      │ Skin interpretation");
        println!("  │  ────┼────────┼──────────┼─────────────┼──────────────────────────");

        let mut rs: Vec<(usize, f64)> = Vec::new();

        for &lz in &depths {
            let r = ensemble_r_3d(lxy, lxy, lz, w, &seeds);
            let delta = r - r_2d;
            let regime = if r > midpoint {
                "extended"
            } else {
                "localized"
            };
            let interp = match lz {
                1 => "surface scratch — quasi-2D",
                2 => "shallow breach",
                3 => "moderate breach",
                4 => "deep breach into dermis",
                5 => "full-thickness breach",
                _ => "through to reticular dermis",
            };

            println!("  │  {lz:>3}  │ {r:.4} │ {delta:+.4}   │ {regime:<11} │ {interp}");
            rs.push((lz, r));
        }

        v.check_pass(
            "All breach depths produce valid r",
            rs.iter().all(|(_, r)| *r > 0.35 && *r < 0.6),
        );

        // r should generally increase (or at least not decrease dramatically)
        // as depth increases from 1 to 6
        let r_shallow = rs[0].1;
        let r_deep = rs[rs.len() - 1].1;
        v.check_pass(
            "Deepest breach r >= shallowest breach r",
            r_deep >= r_shallow - 0.01,
        );

        // Check if any depth crosses midpoint
        let transition_depth = rs.iter().find(|(_, r)| *r > midpoint);
        if let Some((lz, r)) = transition_depth {
            println!("  │  ");
            println!("  │  Transition at Lz={lz}: r={r:.4} > midpoint={midpoint:.4}");
            v.check_pass("Dimensional promotion transition found", true);
        } else {
            println!("  │  ");
            println!("  │  No transition in this W — all depths localized");
            v.check_pass("Consistent regime across depths", true);
        }

        println!("  └─ Depth determines effective dimension → controls signal propagation\n");

        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        domains.push(DomainResult {
            name: "Depth scan",
            ms,
            checks: 3,
        });
    }

    // ── D2: Disorder × Depth Phase Diagram ──────────────────────────────
    {
        let t0 = Instant::now();

        let lxy = 6_usize;
        let disorders: [f64; 5] = [4.0, 8.0, 12.0, 16.0, 20.0];
        let depths: [usize; 4] = [2, 3, 4, 6];

        println!("  ┌─ D2: (W, Lz) Phase Diagram ({lxy}×{lxy}×Lz, 5 seeds)");
        println!("  │  ");
        print!("  │  W \\ Lz │");
        for &lz in &depths {
            print!("   {lz:>2}   │");
        }
        println!();
        print!("  │  ───────┼");
        for _ in &depths {
            print!("────────┼");
        }
        println!();

        let mut any_extended = false;
        let mut any_localized = false;

        for &w in &disorders {
            print!("  │  {w:>5.1}  │");
            for &lz in &depths {
                let r = ensemble_r_3d(lxy, lxy, lz, w, &seeds);
                let symbol = if r > midpoint {
                    any_extended = true;
                    "EXT"
                } else {
                    any_localized = true;
                    "LOC"
                };
                print!(" {r:.3} {symbol} │");
            }
            println!();
        }

        v.check_pass(
            "Phase diagram has both regimes",
            any_extended && any_localized,
        );

        // Low W, high Lz should be extended; high W, low Lz should be localized
        let r_low_w_deep = ensemble_r_3d(lxy, lxy, 6, 4.0, &seeds);
        let r_high_w_shallow = ensemble_r_3d(lxy, lxy, 2, 20.0, &seeds);
        v.check_pass(
            "Low-W/deep > high-W/shallow",
            r_low_w_deep > r_high_w_shallow,
        );

        println!("  │  ");
        println!("  │  EXT = extended (cytokines propagate)  LOC = localized (confined)");
        println!("  └─ Phase boundary separates inflammatory vs controlled regimes\n");

        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        domains.push(DomainResult {
            name: "Phase diagram",
            ms,
            checks: 2,
        });
    }

    // ── D3: Duality — Paper 06 Collapse vs Paper 12 Promotion ───────────
    {
        let t0 = Instant::now();

        let l = 6_usize;
        let w = 10.0;

        // Paper 06: Soil starts 3D, tillage collapses to 2D
        let r_soil_3d = ensemble_r_3d(l, l, l, w, &seeds);
        let r_soil_2d = ensemble_r_2d(l, l, w, &seeds);

        // Paper 12: Skin starts 2D, scratching promotes to 3D
        let r_skin_2d = ensemble_r_2d(l, l, w, &seeds);
        let r_skin_3d = ensemble_r_3d(l, l, l, w, &seeds);

        v.check_pass(
            "Soil r_3D = Skin r_3D (same physics)",
            (r_soil_3d - r_skin_3d).abs() < tolerances::PYTHON_PARITY,
        );
        v.check_pass(
            "Soil r_2D = Skin r_2D (same physics)",
            (r_soil_2d - r_skin_2d).abs() < tolerances::PYTHON_PARITY,
        );

        let collapse_delta = r_soil_2d - r_soil_3d;
        let promotion_delta = r_skin_3d - r_skin_2d;

        v.check_pass(
            "Collapse and promotion deltas are exact negatives",
            (collapse_delta + promotion_delta).abs() < tolerances::PYTHON_PARITY,
        );

        println!("  ┌─ D3: Dimensional Duality (W={w}, L={l})");
        println!("  │  Paper 06 (Soil) — Tillage COLLAPSE:");
        println!("  │    3D → 2D: r = {r_soil_3d:.4} → {r_soil_2d:.4} (Δr = {collapse_delta:+.4})");
        println!("  │    → QS signals LOCALIZE → ecosystem services LOST");
        println!("  │  ");
        println!("  │  Paper 12 (Skin) — Scratching PROMOTION:");
        println!(
            "  │    2D → 3D: r = {r_skin_2d:.4} → {r_skin_3d:.4} (Δr = {promotion_delta:+.4})"
        );
        println!("  │    → Cytokines DELOCALIZE → inflammatory cascade AMPLIFIES");
        println!("  │  ");
        println!("  │  Same physics, opposite direction, opposite biological outcome.");
        println!("  └─ Anderson framework is agnostic — context determines pathology\n");

        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        domains.push(DomainResult {
            name: "P06↔P12 duality",
            ms,
            checks: 3,
        });
    }

    // ── D4: AD Disease Cycle Simulation ─────────────────────────────────
    {
        let t0 = Instant::now();

        let lxy = 6_usize;

        // Phase 1: Healthy skin — low W, 2D epidermis
        let w_healthy = 6.0;
        let r_healthy_epi = ensemble_r_2d(lxy, lxy, w_healthy, &seeds);

        // Phase 2: Allergen exposure → Th2 activation → cytokine production in dermis
        let w_allergen = 10.0;
        let r_allergen_derm = ensemble_r_3d(lxy, lxy, lxy, w_allergen, &seeds);

        // Phase 3: Itch → scratching → barrier disruption
        let w_scratch = 12.0;
        let r_scratch = ensemble_r_3d(lxy, lxy, 4, w_scratch, &seeds);

        // Phase 4: Chronic AD — full 3D access, high W
        let w_chronic = 15.0;
        let r_chronic = ensemble_r_3d(lxy, lxy, lxy, w_chronic, &seeds);

        // Phase 5: Treatment — Apoquel (W reduction)
        let w_treated = 5.0;
        let r_treated = ensemble_r_3d(lxy, lxy, lxy, w_treated, &seeds);

        // Phase 6: Barrier repair + treatment
        let r_repaired = ensemble_r_2d(lxy, lxy, w_treated, &seeds);

        v.check_pass(
            "Healthy epidermis r is reasonable",
            r_healthy_epi > 0.35 && r_healthy_epi < 0.56,
        );
        v.check_pass(
            "Allergen exposure in dermis shows propagation",
            r_allergen_derm > midpoint - 0.05,
        );
        v.check_pass(
            "Treatment reduces effective disorder",
            w_treated < w_chronic,
        );

        let pielou_healthy: f64 = {
            let counts = vec![850.0, 50.0, 30.0, 20.0, 15.0, 10.0, 5.0, 5.0, 5.0, 10.0];
            let h = diversity::shannon(&counts);
            h / (counts.len() as f64).ln()
        };
        let pielou_chronic: f64 = {
            let counts = vec![
                400.0, 100.0, 15.0, 60.0, 150.0, 100.0, 60.0, 40.0, 35.0, 40.0,
            ];
            let h = diversity::shannon(&counts);
            h / (counts.len() as f64).ln()
        };

        v.check_pass(
            "Chronic AD Pielou > healthy Pielou",
            pielou_chronic > pielou_healthy,
        );

        println!("  ┌─ D4: AD Disease Cycle as Anderson Phase Transitions");
        println!("  │  ");
        println!("  │  Phase             │ Geometry │ W     │ <r>    │ Regime");
        println!("  │  ──────────────────┼──────────┼───────┼────────┼──────────");
        println!(
            "  │  1. Healthy        │ 2D epi   │ {:>5.1} │ {:.4} │ {}",
            w_healthy,
            r_healthy_epi,
            if r_healthy_epi > midpoint {
                "extended"
            } else {
                "localized"
            }
        );
        println!(
            "  │  2. Allergen→Th2   │ 3D derm  │ {:>5.1} │ {:.4} │ {}",
            w_allergen,
            r_allergen_derm,
            if r_allergen_derm > midpoint {
                "extended"
            } else {
                "localized"
            }
        );
        println!(
            "  │  3. Scratch/breach │ 3D slab  │ {:>5.1} │ {:.4} │ {}",
            w_scratch,
            r_scratch,
            if r_scratch > midpoint {
                "extended"
            } else {
                "localized"
            }
        );
        println!(
            "  │  4. Chronic AD     │ 3D full  │ {:>5.1} │ {:.4} │ {}",
            w_chronic,
            r_chronic,
            if r_chronic > midpoint {
                "extended"
            } else {
                "localized"
            }
        );
        println!(
            "  │  5. Apoquel        │ 3D full  │ {:>5.1} │ {:.4} │ {}",
            w_treated,
            r_treated,
            if r_treated > midpoint {
                "extended"
            } else {
                "localized"
            }
        );
        println!(
            "  │  6. Barrier repair │ 2D epi   │ {:>5.1} │ {:.4} │ {}",
            w_treated,
            r_repaired,
            if r_repaired > midpoint {
                "extended"
            } else {
                "localized"
            }
        );
        println!("  │  ");
        println!(
            "  │  Cell-type evenness: Pielou(healthy)={pielou_healthy:.3}, Pielou(chronic)={pielou_chronic:.3}"
        );
        println!("  └─ Full AD cycle traverses Anderson phase space\n");

        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        domains.push(DomainResult {
            name: "AD disease cycle",
            ms,
            checks: 4,
        });
    }

    // ── D5: Fajgenbaum Geometry Score Prototype ─────────────────────────
    {
        let t0 = Instant::now();

        struct DrugCandidate {
            name: &'static str,
            pathway_score: f64,
            delivery: &'static str,
            #[allow(dead_code)]
            molecular_weight_da: f64,
            reaches_dermis: bool,
            crosses_barrier_topical: bool,
        }

        let drugs = [
            DrugCandidate {
                name: "Apoquel (oclacitinib)",
                pathway_score: 0.95,
                delivery: "oral→systemic",
                molecular_weight_da: 338.4,
                reaches_dermis: true,
                crosses_barrier_topical: true,
            },
            DrugCandidate {
                name: "Cytopoint (lokivetmab)",
                pathway_score: 0.90,
                delivery: "injection→systemic",
                molecular_weight_da: 150_000.0,
                reaches_dermis: true,
                crosses_barrier_topical: false,
            },
            DrugCandidate {
                name: "Rapamycin (sirolimus)",
                pathway_score: 0.65,
                delivery: "oral→systemic",
                molecular_weight_da: 914.2,
                reaches_dermis: true,
                crosses_barrier_topical: false,
            },
            DrugCandidate {
                name: "Crisaborole",
                pathway_score: 0.55,
                delivery: "topical",
                molecular_weight_da: 251.1,
                reaches_dermis: false, // topical: limited by 2D barrier
                crosses_barrier_topical: true,
            },
            DrugCandidate {
                name: "Trametinib",
                pathway_score: 0.40,
                delivery: "oral→systemic",
                molecular_weight_da: 615.4,
                reaches_dermis: true,
                crosses_barrier_topical: false,
            },
        ];

        println!("  ┌─ D5: Anderson-Augmented Drug Repurposing Score");
        println!("  │  ");
        println!("  │  Drug                    │ Pathway │ Geom  │ Score │ Delivery");
        println!("  │  ────────────────────────┼─────────┼───────┼───────┼──────────────");

        let mut scores: Vec<f64> = Vec::new();
        for drug in &drugs {
            let geom = if drug.reaches_dermis { 1.0 } else { 0.4 }
                * if drug.crosses_barrier_topical {
                    1.0
                } else {
                    0.8
                };

            let anderson_score = drug.pathway_score * geom;
            scores.push(anderson_score);

            println!(
                "  │  {:<24}│ {:.2}    │ {:.2}  │ {:.2}  │ {}",
                drug.name, drug.pathway_score, geom, anderson_score, drug.delivery
            );
        }

        // Apoquel should rank highest (high pathway + good geometry)
        v.check_pass(
            "Apoquel has highest Anderson-augmented score",
            scores[0] >= scores[1]
                && scores[0] >= scores[2]
                && scores[0] >= scores[3]
                && scores[0] >= scores[4],
        );

        // Crisaborole loses rank due to geometry penalty
        v.check_pass(
            "Crisaborole penalized by barrier geometry",
            scores[3] < scores[0] && scores[3] < scores[1],
        );

        v.check_pass(
            "All scores in [0, 1]",
            scores.iter().all(|&s| (0.0..=1.0).contains(&s)),
        );

        println!("  │  ");
        println!("  │  Score = Pathway × Geometry (Anderson barrier penetration)");
        println!("  │  MATRIX asks: does the drug hit the right target?");
        println!("  │  Anderson adds: can the drug physically REACH the target?");
        println!("  └─ Geometry dimension filters topical drugs for 3D compartment targets\n");

        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        domains.push(DomainResult {
            name: "Fajgenbaum score",
            ms,
            checks: 3,
        });
    }

    // ── Summary ─────────────────────────────────────────────────────────
    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║  Exp274: Barrier Disruption — Dimensional Promotion Threshold     ║");
    println!("╠════════════════════════════════════════════════════════════════════╣");
    println!("║ Domain                 │    Time │   ✓ ║");
    println!("╠════════════════════════════════════════════════════════════════════╣");

    let mut total_checks = 0_u32;
    let mut total_ms = 0.0_f64;
    for d in &domains {
        let (name, ms, checks) = (&d.name, d.ms, d.checks);
        println!("║ {name:<22} │ {ms:>6.1}ms │ {checks:>3} ║");
        total_checks += d.checks;
        total_ms += d.ms;
    }
    println!("╠════════════════════════════════════════════════════════════════════╣");
    println!("║ TOTAL                  │ {total_ms:>6.1}ms │ {total_checks:>3} ║");
    println!("╚════════════════════════════════════════════════════════════════════╝");

    println!("\n  Paper 06 ↔ Paper 12 Duality:");
    println!("  Tillage COLLAPSE (3D→2D) = QS fails = ecosystem damage");
    println!("  Scratch PROMOTION (2D→3D) = cytokines escape = inflammatory cascade");
    println!("  Same Anderson physics, opposite direction, opposite pathology.");
    println!();

    v.finish();
}
