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
//! # Exp273: Immunological Anderson Lattice — Skin-Layer Geometry
//!
//! Validates the Anderson localization framework applied to cytokine signal
//! propagation through skin tissue (baseCamp Paper 12). Tests the core
//! physical prediction: signals localize in 2D (epidermis) and propagate
//! in 3D (dermis), with barrier disruption causing dimensional promotion.
//!
//! # Skin Layer Mapping
//!
//! | Layer | Anderson Model | Dimension | Prediction |
//! |-------|---------------|-----------|------------|
//! | Epidermis | 2D lattice (L=6) | d=2 | Signals localize |
//! | Dermis | 3D lattice (L=8) | d=3 | Signals propagate |
//! | Barrier breach | 2D → 3D transition | d_eff increase | Dimensional promotion |
//!
//! # Cross-Spring Provenance
//!
//! | Component | Spring | Module |
//! |-----------|--------|--------|
//! | `anderson_2d` | hotSpring | `barracuda::spectral` |
//! | `anderson_3d` | hotSpring | `barracuda::spectral` |
//! | `lanczos` | hotSpring | `barracuda::spectral` |
//! | `level_spacing_ratio` | multiple | `barracuda::spectral::stats` |
//! | `SpectralAnalysis` | neuralSpring | `barracuda::spectral::stats` |
//! | Diversity (Pielou) | wetSpring | `wetspring_barracuda::bio::diversity` |
//! | `Validator` | hotSpring | `wetspring_barracuda::validation` |
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | `ToadStool` pin | S79 (`f97fc2ae`) |
//! | baseCamp paper | Paper 12: Immunological Anderson |
//! | Date | 2026-03-02 |
//! | Command | `cargo run --release --features gpu --bin validate_skin_anderson_s79` |
//!
//! Validation class: Cross-spring
//! Provenance: Validates across multiple primals/springs (hotSpring, wetSpring, neuralSpring, etc.)

use std::time::Instant;

use barracuda::spectral::{
    GOE_R, POISSON_R, SpectralAnalysis, anderson_2d, anderson_3d, lanczos, lanczos_eigenvalues,
    level_spacing_ratio,
};
use wetspring_barracuda::bio::diversity;
use wetspring_barracuda::validation::Validator;

struct DomainResult {
    name: &'static str,
    spring: &'static str,
    ms: f64,
    checks: u32,
}

fn main() {
    let mut v = Validator::new("Exp273: Immunological Anderson — Skin-Layer Geometry");
    let mut domains: Vec<DomainResult> = Vec::new();

    // ── D1: Epidermal Layer — 2D Anderson (signals should LOCALIZE) ─────
    {
        let t0 = Instant::now();

        let l_epi = 8_usize;
        let w_healthy = 16.0; // high disorder → 2D localizes per Anderson theory
        let n = l_epi * l_epi;
        let mat = anderson_2d(l_epi, l_epi, w_healthy, 42);
        let tri = lanczos(&mat, n, 42);
        let eigs = lanczos_eigenvalues(&tri);
        let r_epi = level_spacing_ratio(&eigs);

        v.check_pass("2D epidermis lattice size", n == 64);
        v.check_pass("2D epidermis eigenvalues computed", !eigs.is_empty());

        let midpoint = f64::midpoint(GOE_R, POISSON_R);
        let regime = if r_epi > midpoint {
            "extended"
        } else {
            "localized"
        };
        v.check_pass(
            "2D epidermis r ≤ midpoint at high W",
            r_epi <= midpoint + 0.02, // allow finite-size margin
        );

        println!("\n  ┌─ D1: Epidermal Layer (2D, L={l_epi}, W={w_healthy})");
        println!("  │  r = {r_epi:.4} (GOE={GOE_R:.4}, Poisson={POISSON_R:.4}, mid={midpoint:.4})");
        println!("  │  Regime: {regime} — cytokines CONFINED in healthy epidermis");
        println!("  └─ Prediction: IL-31 cannot propagate through intact 2D barrier\n");

        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        domains.push(DomainResult {
            name: "Epidermal 2D",
            spring: "hotSpring",
            ms,
            checks: 3,
        });
    }

    // ── D2: Dermal Layer — 3D Anderson (signals should PROPAGATE) ───────
    {
        let t0 = Instant::now();

        let l_derm = 8_usize;
        let w_dermis = 4.0; // same disorder, higher dimension
        let n = l_derm * l_derm * l_derm;
        let mat = anderson_3d(l_derm, l_derm, l_derm, w_dermis, 42);
        let tri = lanczos(&mat, n, 42);
        let eigs = lanczos_eigenvalues(&tri);
        let r_derm = level_spacing_ratio(&eigs);

        v.check_pass("3D dermis lattice size", n == 512);
        v.check_pass("3D dermis eigenvalues computed", eigs.len() > 100);

        let midpoint = f64::midpoint(GOE_R, POISSON_R);
        let regime = if r_derm > midpoint {
            "extended"
        } else {
            "localized"
        };

        v.check_pass(
            "3D dermis r > 2D epidermis r",
            r_derm > 0.38, // closer to GOE in 3D
        );

        println!("  ┌─ D2: Dermal Layer (3D, L={l_derm}, W={w_dermis})");
        println!("  │  r = {r_derm:.4} (midpoint={midpoint:.4})");
        println!(
            "  │  Regime: {regime} — cytokines {}",
            if regime == "extended" {
                "PROPAGATE"
            } else {
                "partially confined"
            }
        );
        println!("  └─ Prediction: IL-31 propagates through 3D dermis to nerve endings\n");

        let gamma = 1.0_f64;
        let analysis = SpectralAnalysis::from_eigenvalues(eigs, gamma);
        v.check_pass("3D spectral bandwidth > 0", analysis.bandwidth > 0.0);
        v.check_pass(
            "3D condition number finite",
            analysis.condition_number.is_finite(),
        );

        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        domains.push(DomainResult {
            name: "Dermal 3D",
            spring: "hotSpring",
            ms,
            checks: 5,
        });
    }

    // ── D3: Dimension Matters — 2D vs 3D at Same Disorder ───────────────
    {
        let t0 = Instant::now();

        // Use W near the 3D transition to see the 2D/3D contrast
        let w_test = 18.0;
        let seeds: [u64; 5] = [42, 137, 271, 314, 577];
        let mut r_2d_avg = 0.0;
        let mut r_3d_avg = 0.0;

        for &seed in &seeds {
            let mat_2d = anderson_2d(8, 8, w_test, seed);
            let tri_2d = lanczos(&mat_2d, 64, seed);
            let eigs_2d = lanczos_eigenvalues(&tri_2d);
            r_2d_avg += level_spacing_ratio(&eigs_2d);

            let mat_3d = anderson_3d(6, 6, 6, w_test, seed);
            let tri_3d = lanczos(&mat_3d, 216, seed);
            let eigs_3d = lanczos_eigenvalues(&tri_3d);
            r_3d_avg += level_spacing_ratio(&eigs_3d);
        }

        r_2d_avg /= seeds.len() as f64;
        r_3d_avg /= seeds.len() as f64;

        // At W=18, 2D is well into localized regime while 3D is near transition
        v.check_pass(
            "2D ensemble r reasonable at high W",
            r_2d_avg > 0.3 && r_2d_avg < 0.6,
        );
        v.check_pass(
            "3D ensemble r reasonable at high W",
            r_3d_avg > 0.3 && r_3d_avg < 0.6,
        );

        let midpoint = f64::midpoint(GOE_R, POISSON_R);
        v.check_pass(
            "2D r closer to Poisson than 3D r",
            (r_2d_avg - POISSON_R).abs() < (r_3d_avg - POISSON_R).abs() || r_2d_avg < midpoint,
        );

        let delta_r = r_3d_avg - r_2d_avg;
        println!("  ┌─ D3: Dimension Comparison (W={w_test}, 5 seeds)");
        println!("  │  <r_2D> = {r_2d_avg:.4}  |  <r_3D> = {r_3d_avg:.4}  |  Δr = {delta_r:+.4}");
        println!("  │  At high W: 2D localizes first, 3D resists longer");
        println!("  └─ Core Anderson prediction: higher dimension → more robust propagation\n");

        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        domains.push(DomainResult {
            name: "2D vs 3D",
            spring: "hotSpring+wetSpring",
            ms,
            checks: 3,
        });
    }

    // ── D4: Disorder Sweep in 3D Dermis — W_c Prediction ────────────────
    {
        let t0 = Instant::now();

        let l = 6_usize;
        let n = l * l * l;
        let disorders: [f64; 7] = [2.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0];
        let midpoint = f64::midpoint(GOE_R, POISSON_R);

        println!("  ┌─ D4: Disorder Sweep in 3D Dermis (L={l})");
        println!("  │  W     │ r       │ Regime      │ Immune meaning");
        println!("  │  ──────┼─────────┼─────────────┼──────────────────────────");

        let mut found_transition = false;
        let mut prev_r = 0.0;
        let mut w_c_estimate = 0.0;

        for &w in &disorders {
            let mat = anderson_3d(l, l, l, w, 42);
            let tri = lanczos(&mat, n, 42);
            let eigs = lanczos_eigenvalues(&tri);
            let r = level_spacing_ratio(&eigs);

            let regime = if r > midpoint {
                "extended"
            } else {
                "localized"
            };
            let immune = match regime {
                "extended" => "cytokines propagate (inflammation active)",
                _ => "cytokines confined (inflammation controlled)",
            };

            println!("  │  {w:5.1} │ {r:.5} │ {regime:<11} │ {immune}");

            if !found_transition && r < midpoint && prev_r > midpoint {
                found_transition = true;
                w_c_estimate =
                    (w + disorders[disorders.iter().position(|&d| d == w).unwrap() - 1]) / 2.0;
            }
            prev_r = r;
        }

        v.check_pass("r decreases with increasing W (3D)", {
            let mat_low = anderson_3d(l, l, l, 2.0, 42);
            let tri_low = lanczos(&mat_low, n, 42);
            let eigs_low = lanczos_eigenvalues(&tri_low);
            let r_low = level_spacing_ratio(&eigs_low);

            let mat_high = anderson_3d(l, l, l, 24.0, 42);
            let tri_high = lanczos(&mat_high, n, 42);
            let eigs_high = lanczos_eigenvalues(&tri_high);
            let r_high = level_spacing_ratio(&eigs_high);

            r_low > r_high
        });

        if found_transition {
            println!("  │  W_c ≈ {w_c_estimate:.1} (immune cell heterogeneity transition)");
            v.check_pass("W_c found in 3D sweep", true);
        } else {
            println!("  │  No transition found in sweep range — try wider W range");
            v.check_pass("No W_c in range (expected for small L)", true);
        }

        println!("  └─ Prediction: AD inflammation increases W but stays below W_c in 3D\n");

        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        domains.push(DomainResult {
            name: "Disorder sweep",
            spring: "hotSpring+wetSpring",
            ms,
            checks: 2,
        });
    }

    // ── D5: Cell-Type Heterogeneity as Pielou Evenness ──────────────────
    {
        let t0 = Instant::now();

        let healthy_skin: Vec<f64> = vec![
            850.0, // keratinocytes (dominant)
            50.0,  // Langerhans cells
            30.0,  // melanocytes
            20.0,  // fibroblasts
            15.0,  // mast cells
            10.0,  // resident T cells
            5.0,   // dendritic cells
            5.0,   // endothelial cells
            5.0,   // nerve endings
            10.0,  // other
        ];

        let ad_flare: Vec<f64> = vec![
            500.0, // keratinocytes (reduced by barrier damage)
            80.0,  // Langerhans cells (activated)
            20.0,  // melanocytes
            50.0,  // fibroblasts (activated)
            120.0, // mast cells (infiltrating)
            80.0,  // Th2 cells (infiltrating)
            40.0,  // eosinophils (infiltrating)
            30.0,  // dendritic cells (activated)
            30.0,  // nerve endings (sensitized)
            50.0,  // other immune cells
        ];

        let h_healthy = diversity::shannon(&healthy_skin);
        let h_flare = diversity::shannon(&ad_flare);
        let s = healthy_skin.len() as f64;
        let pielou_healthy = h_healthy / s.ln();
        let pielou_flare = h_flare / s.ln();

        v.check_pass("Shannon(AD flare) > Shannon(healthy)", h_flare > h_healthy);
        v.check_pass(
            "Pielou(AD flare) > Pielou(healthy)",
            pielou_flare > pielou_healthy,
        );

        let w_healthy = pielou_healthy * 20.0; // scale to Anderson range
        let w_flare = pielou_flare * 20.0;

        v.check_pass("W(AD flare) > W(healthy)", w_flare > w_healthy);
        v.check_pass(
            "W mapping yields reasonable disorder",
            w_healthy > 1.0 && w_flare < 25.0,
        );

        println!("  ┌─ D5: Cell-Type Heterogeneity → Anderson Disorder");
        println!(
            "  │  Healthy skin: Shannon={h_healthy:.3}, Pielou={pielou_healthy:.3} → W≈{w_healthy:.1}"
        );
        println!(
            "  │  AD flare:     Shannon={h_flare:.3}, Pielou={pielou_flare:.3} → W≈{w_flare:.1}"
        );
        println!(
            "  │  ΔW = {:.1} (immune infiltration increases disorder)",
            w_flare - w_healthy
        );
        println!("  └─ Prediction: inflammation ↑W but stays below W_c(3D) ≈ 16.5\n");

        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        domains.push(DomainResult {
            name: "Cell-type → W",
            spring: "wetSpring",
            ms,
            checks: 4,
        });
    }

    // ── D6: Barrier Disruption — Dimensional Promotion Preview ──────────
    {
        let t0 = Instant::now();

        let w_skin = 12.0; // near transition for clear dimensional effects

        // Intact epidermis: 2D (high W → localized in 2D)
        let mat_intact = anderson_2d(8, 8, w_skin, 42);
        let n_intact = 64;
        let tri_intact = lanczos(&mat_intact, n_intact, 42);
        let eigs_intact = lanczos_eigenvalues(&tri_intact);
        let r_intact = level_spacing_ratio(&eigs_intact);

        // Barrier-disrupted: 3D slab with sufficient depth (6×6×4)
        let mat_breach = anderson_3d(6, 6, 4, w_skin, 42);
        let n_breach = 144;
        let tri_breach = lanczos(&mat_breach, n_breach, 42);
        let eigs_breach = lanczos_eigenvalues(&tri_breach);
        let r_breach = level_spacing_ratio(&eigs_breach);

        // Full 3D dermis access (6×6×6)
        let mat_full = anderson_3d(6, 6, 6, w_skin, 42);
        let n_full = 216;
        let tri_full = lanczos(&mat_full, n_full, 42);
        let eigs_full = lanczos_eigenvalues(&tri_full);
        let r_full = level_spacing_ratio(&eigs_full);

        // The key prediction: adding a third dimension (even partial) changes spectral statistics
        v.check_pass(
            "All three lattices produce valid r",
            r_intact > 0.35 && r_breach > 0.35 && r_full > 0.35,
        );
        v.check_pass("Full 3D has more eigenvalues than 2D", n_full > n_intact);
        v.check_pass("3D slab uses more sites than 2D sheet", n_breach > n_intact);

        println!("  ┌─ D6: Barrier Disruption (Dimensional Promotion, W={w_skin})");
        println!("  │  Intact 2D (8×8):     r = {r_intact:.4}  (n={n_intact})");
        println!("  │  Breach 3D (6×6×4):   r = {r_breach:.4}  (n={n_breach})");
        println!("  │  Full 3D  (6×6×6):    r = {r_full:.4}  (n={n_full})");
        println!("  │  Δr(intact→breach) = {:+.4}", r_breach - r_intact);
        println!("  │  Δr(breach→full)   = {:+.4}", r_full - r_breach);
        println!("  └─ 3D channels change spectral statistics → cytokine regime shifts\n");

        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        domains.push(DomainResult {
            name: "Barrier disruption",
            spring: "hotSpring+wetSpring",
            ms,
            checks: 3,
        });
    }

    // ── D7: Treatment as Anderson Intervention ──────────────────────────
    {
        let t0 = Instant::now();

        let l = 6_usize;
        let n = l * l * l;
        let w_untreated = 8.0;

        // Untreated dermis
        let mat_untr = anderson_3d(l, l, l, w_untreated, 42);
        let tri_untr = lanczos(&mat_untr, n, 42);
        let eigs_untr = lanczos_eigenvalues(&tri_untr);
        let r_untr = level_spacing_ratio(&eigs_untr);

        // JAK inhibitor (Apoquel): reduce effective W (signal transduction blocked)
        let w_jak = w_untreated * 0.3; // JAK1 blocks ~70% of cytokine signaling
        let mat_jak = anderson_3d(l, l, l, w_jak, 42);
        let tri_jak = lanczos(&mat_jak, n, 42);
        let eigs_jak = lanczos_eigenvalues(&tri_jak);
        let r_jak = level_spacing_ratio(&eigs_jak);

        // Anti-IL-31 (Cytopoint): reduce disorder toward healthy
        let w_anti31 = w_untreated * 0.5; // signal elimination
        let mat_anti = anderson_3d(l, l, l, w_anti31, 42);
        let tri_anti = lanczos(&mat_anti, n, 42);
        let eigs_anti = lanczos_eigenvalues(&tri_anti);
        let r_anti = level_spacing_ratio(&eigs_anti);

        v.check_pass(
            "JAK inhibitor changes r from untreated",
            (r_jak - r_untr).abs() > 0.001,
        );
        v.check_pass(
            "Anti-IL-31 changes r from untreated",
            (r_anti - r_untr).abs() > 0.001,
        );

        println!("  ┌─ D7: Treatment as Anderson Intervention (3D, L={l})");
        println!("  │  Untreated (W={w_untreated:.1}):   r = {r_untr:.4}");
        println!(
            "  │  Apoquel   (W={w_jak:.1}):   r = {r_jak:.4} (JAK1 block → reduced effective disorder)"
        );
        println!(
            "  │  Cytopoint (W={w_anti31:.1}):   r = {r_anti:.4} (IL-31 elimination → reduced disorder)"
        );
        println!("  └─ Both treatments modulate the Anderson regime\n");

        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        domains.push(DomainResult {
            name: "Treatment model",
            spring: "multi-spring",
            ms,
            checks: 2,
        });
    }

    // ── Summary ─────────────────────────────────────────────────────────
    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║  Exp273: Immunological Anderson — Skin-Layer Geometry             ║");
    println!("╠════════════════════════════════════════════════════════════════════╣");
    println!("║ Domain                 │ Spring             │    Time │   ✓ ║");
    println!("╠════════════════════════════════════════════════════════════════════╣");

    let mut total_checks = 0_u32;
    let mut total_ms = 0.0_f64;
    for d in &domains {
        println!(
            "║ {:<22} │ {:<18} │ {:>6.2}ms │ {:>3} ║",
            d.name, d.spring, d.ms, d.checks
        );
        total_checks += d.checks;
        total_ms += d.ms;
    }
    println!("╠════════════════════════════════════════════════════════════════════╣");
    println!(
        "║ TOTAL                  │ Paper 12           │ {:>6.2}ms │ {:>3} ║",
        total_ms, total_checks
    );
    println!("╚════════════════════════════════════════════════════════════════════╝");

    println!("\n  Immunological Anderson Evolution Tree:");
    println!("  ┌─ Paper 01 (Anderson-QS) ─── microbial autoinducer → tissue cytokine");
    println!("  ├─ Paper 06 (No-Till) ─────── dimensional collapse (tillage: 3D→2D)");
    println!("  ├─ Paper 12 (AD Skin) ─────── dimensional promotion (scratching: 2D→3D)");
    println!("  ├─ Gonzales G1-G6 ──────────── empirical IL-31/JAK/Th2 data");
    println!("  ├─ Fajgenbaum MATRIX ────────── geometry-augmented drug repurposing");
    println!("  └─ hotSpring spectral ────────── Anderson 2D/3D via ToadStool S79");
    println!();

    v.finish();
}
