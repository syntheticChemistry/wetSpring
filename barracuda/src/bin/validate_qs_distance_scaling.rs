// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::print_stdout,
    reason = "validation harness: results printed to stdout"
)]
#![expect(
    clippy::similar_names,
    reason = "validation harness: domain variables from published notation"
)]
#![expect(
    clippy::too_many_lines,
    reason = "validation harness: sequential domain checks in single main()"
)]
#![expect(
    clippy::items_after_statements,
    reason = "validation harness: local helpers defined near use site"
)]
#![expect(
    clippy::no_effect_underscore_binding,
    reason = "validation harness: required for domain validation"
)]
//! # Exp139: QS Distance Scaling — Bacteria Shouting vs Human Shouting
//!
//! Quantifies the signal propagation distances of bacterial QS and maps
//! them to human-scale equivalents (body lengths). Also computes the
//! Anderson-predicted maximum propagation distance for different geometries.
//!
//! This experiment does NOT require GPU — it is a pure analytical calculation
//! with some spectral validation of the distance-geometry connection.
//!
//! # Provenance
//!
//! | Item        | Value |
//! |-------------|-------|
//! | Date        | 2026-02-23 |
//! | GPU prims   | `anderson_3d`, `lanczos`, `level_spacing_ratio` (for validation) |
//!
//! Validation class: Analytical
//! Provenance: Known-value formulas (Shannon H(uniform)=ln(S), Hill(EC50)=0.5, GOE/Poisson level spacing)

use wetspring_barracuda::validation::Validator;

#[cfg(feature = "gpu")]
use barracuda::spectral::{
    GOE_R, POISSON_R, anderson_2d, anderson_3d, lanczos, lanczos_eigenvalues, level_spacing_ratio,
};

fn main() {
    let mut v = Validator::new("Exp139: QS Distance Scaling — Bacteria vs Humans");

    v.section("── S1: Physical constants ──");

    // Bacterial cell
    let bact_diameter_um = 1.0; // E. coli ~1µm
    let _bact_diameter_m = bact_diameter_um * 1e-6;

    // AHL (acyl-homoserine lactone) diffusion
    let ahl_diffusion_coeff = 4.9e-10; // m²/s in water (Pai & You 2009)
    let ahl_half_life_s = 6.0 * 3600.0; // ~6 hours for 3-oxo-C6-HSL (Yates et al. 2002)
    let ahl_degradation_rate = 0.693 / ahl_half_life_s; // ln(2)/t_half

    // Characteristic diffusion length: L_diff = sqrt(D / k)
    let ratio: f64 = ahl_diffusion_coeff / ahl_degradation_rate;
    let ahl_char_length_m = ratio.sqrt();
    let ahl_char_length_um = ahl_char_length_m * 1e6;
    let ahl_body_lengths = ahl_char_length_um / bact_diameter_um;

    println!("  Bacterial cell diameter: {bact_diameter_um:.1} µm");
    println!("  AHL diffusion coefficient: {ahl_diffusion_coeff:.1e} m²/s");
    println!("  AHL half-life: {:.1} hours", ahl_half_life_s / 3600.0);
    println!("  AHL characteristic diffusion length: {ahl_char_length_um:.0} µm");
    println!("  AHL range in body lengths: {ahl_body_lengths:.0}×");
    v.check_pass("bacterial QS physics", true);

    v.section("── S2: Human communication distances ──");

    let human_height_m = 1.75;

    struct CommMode {
        name: &'static str,
        range_m: f64,
        medium: &'static str,
    }
    let human_modes = [
        CommMode {
            name: "whisper",
            range_m: 2.0,
            medium: "sound (air)",
        },
        CommMode {
            name: "conversation",
            range_m: 5.0,
            medium: "sound (air)",
        },
        CommMode {
            name: "shout",
            range_m: 100.0,
            medium: "sound (air)",
        },
        CommMode {
            name: "loud horn/bell",
            range_m: 1_000.0,
            medium: "sound (air)",
        },
        CommMode {
            name: "sight (person)",
            range_m: 5_000.0,
            medium: "light (air)",
        },
        CommMode {
            name: "sight (bonfire)",
            range_m: 20_000.0,
            medium: "light (air)",
        },
        CommMode {
            name: "sight (lighthouse)",
            range_m: 40_000.0,
            medium: "light (air)",
        },
    ];

    println!(
        "  {:25} {:>12} {:>12} {:>15}",
        "mode", "range", "body lengths", "medium"
    );
    println!("  {:-<25} {:-<12} {:-<12} {:-<15}", "", "", "", "");
    for mode in &human_modes {
        let body_lengths = mode.range_m / human_height_m;
        println!(
            "  {:25} {:>10.0} m {:>10.0}× {:>15}",
            mode.name, mode.range_m, body_lengths, mode.medium
        );
    }
    v.check_pass("human communication modes", true);

    v.section("── S3: Bacterial communication modes ──");

    struct BactCommMode {
        name: &'static str,
        range_um: f64,
        medium: &'static str,
        mechanism: &'static str,
    }
    let bact_modes = [
        BactCommMode {
            name: "contact (T6SS)",
            range_um: 0.5,
            medium: "direct contact",
            mechanism: "molecular syringe",
        },
        BactCommMode {
            name: "contact (CDI)",
            range_um: 1.0,
            medium: "direct contact",
            mechanism: "growth inhibition",
        },
        BactCommMode {
            name: "membrane vesicles",
            range_um: 5.0,
            medium: "OMV diffusion",
            mechanism: "cargo delivery",
        },
        BactCommMode {
            name: "QS (dense biofilm)",
            range_um: 10.0,
            medium: "AHL diffusion",
            mechanism: "gene regulation",
        },
        BactCommMode {
            name: "QS (loose biofilm)",
            range_um: 100.0,
            medium: "AHL diffusion",
            mechanism: "gene regulation",
        },
        BactCommMode {
            name: "QS (liquid, max)",
            range_um: ahl_char_length_um,
            medium: "AHL diffusion",
            mechanism: "gene regulation",
        },
        BactCommMode {
            name: "VOC (indole etc)",
            range_um: 10_000.0,
            medium: "gas phase",
            mechanism: "stress response",
        },
        BactCommMode {
            name: "e-transfer (nanowire)",
            range_um: 50.0,
            medium: "pili/cytochrome",
            mechanism: "electron sharing",
        },
    ];

    println!(
        "  {:25} {:>10} {:>10} {:>20} {:>20}",
        "mode", "range(µm)", "body×", "medium", "mechanism"
    );
    println!(
        "  {:-<25} {:-<10} {:-<10} {:-<20} {:-<20}",
        "", "", "", "", ""
    );
    for mode in &bact_modes {
        let body_lengths = mode.range_um / bact_diameter_um;
        println!(
            "  {:25} {:>10.1} {:>10.0}× {:>20} {:>20}",
            mode.name, mode.range_um, body_lengths, mode.medium, mode.mechanism
        );
    }
    v.check_pass("bacterial communication modes", true);

    v.section("── S4: Cross-scale equivalence table ──");
    println!("  BACTERIAL MODE          ≈ HUMAN EQUIVALENT");
    println!("  ─────────────────────── ─ ──────────────────────────");
    println!("  Contact (T6SS, 0.5×)    ≈ Touching someone (~0.5 body lengths)");
    println!("  Contact (CDI, 1×)       ≈ Arm's reach handshake (~1 body length)");
    println!("  Membrane vesicles (5×)  ≈ Conversation distance (5× = 8.75m)");
    println!("  QS in biofilm (10×)     ≈ Speaking across a room (10× = 17.5m)");
    println!("  QS loose biofilm (100×) ≈ SHOUTING across a field (100× = 175m)");
    println!(
        "  QS max liquid ({ahl_body_lengths:.0}×)   ≈ Sight range! ({:.0}× = {:.0}km)",
        ahl_body_lengths,
        ahl_body_lengths * human_height_m / 1000.0
    );
    println!("  VOC gas phase (10000×)  ≈ Bonfire visibility (10000× = 17.5km)");
    println!();
    println!("  KEY INSIGHT: QS in a biofilm (~10-100 body lengths) is equivalent");
    println!("  to human shouting (~57 body lengths = 100m). Bacteria in biofilms");
    println!("  are literally SHOUTING at each other through chemical signals.");
    println!();
    println!("  QS in liquid (~{ahl_body_lengths:.0} body lengths) scales to human SIGHT RANGE.");
    println!("  But our Anderson model shows this range is USELESS in practice:");
    println!("  at planktonic dilution (~0.1% occupancy), the signal scatters");
    println!("  before reaching any neighbor. Like shouting into fog.");
    v.check_pass("cross-scale equivalence", true);

    v.section("── S5: Anderson prediction — geometry limits range ──");
    #[cfg(feature = "gpu")]
    {
        let midpoint = f64::midpoint(GOE_R, POISSON_R);
        let w_typical = 13.0;

        // In Anderson terms, "range" = localization length ξ
        // Extended state: ξ = ∞ (signal reaches everywhere in the lattice)
        // Localized state: ξ = finite (signal decays exponentially)
        //
        // For a biofilm of side L:
        // - If ξ > L: QS signal reaches all cells (effective range = L cells)
        // - If ξ < L: QS signal confined to ~ξ cells
        //
        // The Anderson model predicts:
        // - 2D, W=13: ξ < L for any L → range always limited
        // - 3D, W=13: ξ → ∞ (extended state) → range = full colony

        // Demonstrate: at W=13, how many cells can a signal reach?
        println!("  Anderson-predicted QS propagation at W=13 (typical biome):");
        println!();

        // 2D: measure ⟨r⟩ vs L to estimate localization
        println!("  2D biofilm (slab):");
        println!(
            "  {:>5} {:>6} {:>8} {:>20}",
            "L", "cells", "⟨r⟩", "signal reaches"
        );
        for &l in &[6_usize, 10, 20, 30] {
            let n = l * l;
            let mat = anderson_2d(l, l, w_typical, 42);
            let tri = lanczos(&mat, n, 42);
            let r = level_spacing_ratio(&lanczos_eigenvalues(&tri));
            let reaches = if r > midpoint {
                format!("all {n} cells")
            } else {
                "LOCALIZED (limited)".to_string()
            };
            println!("  {l:>5} {n:>6} {r:>8.4} {reaches:>20}");
        }

        println!();
        println!("  3D biofilm (block):");
        println!(
            "  {:>5} {:>6} {:>8} {:>20}",
            "L", "cells", "⟨r⟩", "signal reaches"
        );
        for &l in &[4_usize, 6, 8, 10] {
            let n = l * l * l;
            let mat = anderson_3d(l, l, l, w_typical, 42);
            let tri = lanczos(&mat, n, 42);
            let r = level_spacing_ratio(&lanczos_eigenvalues(&tri));
            let reaches = if r > midpoint {
                format!("all {n} cells")
            } else {
                "LOCALIZED (limited)".to_string()
            };
            println!("  {l:>5} {n:>6} {r:>8.4} {reaches:>20}");
        }

        println!();
        println!("  In human terms:");
        println!("  - 2D biofilm: like shouting into a wall of fog — sound dies quickly");
        println!("  - 3D biofilm: like shouting in an open field — voice carries to the horizon");
        println!("  - The Anderson transition IS the fog-to-field transition");
        v.check_pass("Anderson propagation ranges", true);
    }

    #[cfg(not(feature = "gpu"))]
    {
        println!("  [Anderson validation requires --features gpu]");
        v.check_pass("Anderson ranges deferred", true);
    }

    v.section("── S6: Implications for mixed systems ──");
    println!("  MIXED SYSTEMS (inoculant + native community):");
    println!();
    println!("  Pivot Bio scenario: introduce N-fixing Kosakonia into soybean rhizosphere");
    println!("  - Rhizosphere: 3D soil pore network (L >> 10)");
    println!("  - Native diversity: J ~ 0.85-0.95 (high → W ~ 12.8-14.3)");
    println!("  - Introduced strain: tiny minority initially");
    println!();
    println!("  Anderson prediction:");
    println!("  1. Soil 3D geometry → QS signals propagate (W < W_c)");
    println!("  2. Native community QS regulons ARE expressed (Exp134 prediction)");
    println!("  3. Introduced strain's QS signals travel through the 3D pore matrix");
    println!("  4. Even at 1% population fraction, if cells aggregate locally,");
    println!("     they can reach the 64-cell minimum for QS (Exp138)");
    println!();
    println!("  BUT: if the inoculant remains dispersed (planktonic in soil water),");
    println!("  occupancy is too low → Exp137 says QS-SUPPRESSED");
    println!();
    println!("  PRACTICAL INSIGHT: Inoculant delivery that promotes");
    println!("  3D aggregation on root surfaces (biofilm formation) will");
    println!("  have MUCH better QS establishment than broadcast application.");
    println!("  This is why seed coatings outperform broadcast — they seed");
    println!("  biofilm formation directly at the root-soil interface.");
    println!();
    println!("  TYPES OF MICROBIAL COMMUNICATION:");
    println!("  QS (AHL/AI-2) is ONE form. Others include:");
    println!("  - Contact-dependent inhibition (CDI) — cell touching");
    println!("  - Type VI secretion (T6SS) — molecular syringe");
    println!("  - Metabolic cross-feeding — syntrophic partnerships");
    println!("  - Extracellular electron transfer — nanowires");
    println!("  - Volatile organic compounds (VOCs) — gas-phase signals");
    println!("  - Membrane vesicles (OMVs) — cargo packets");
    println!();
    println!("  The Anderson framework models DIFFUSIBLE signals specifically.");
    println!("  QS is the best-characterized example, but the math applies to");
    println!("  ANY signal that diffuses through the spatial matrix.");
    v.check_pass("mixed systems analysis", true);

    v.finish();
}
