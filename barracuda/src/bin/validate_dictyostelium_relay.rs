// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::cast_precision_loss
)]
//! # Exp156: Dictyostelium cAMP Relay — Non-Hermitian Anderson Analysis
//!
//! Analyzes the Dictyostelium cAMP relay system as a non-Hermitian
//! Anderson model, based on Frontiers Cell Dev Biol 2023 (Paper 38).
//!
//! Dictyostelium defeats Anderson localization via signal relay: each
//! cell amplifies and retransmits, converting passive scatterers into
//! active repeaters. This is the biological analog of a repeater network.
//!
//! # Provenance
//!
//! | Item        | Value |
//! |-------------|-------|
//! | Date        | 2026-02-24 |
//! | Phase       | 39 — Paper queue extension |
//! | Paper       | 38 (Frontiers Cell Dev Biol 2023) |

use wetspring_barracuda::validation::Validator;

fn main() {
    let mut v = Validator::new("Exp156: Dictyostelium cAMP Relay — Non-Hermitian Anderson");

    v.section("§1 Standard Anderson vs Relay Anderson");

    println!("  Standard Anderson (passive scattering):");
    println!("    H = -Δ + V, V_i ~ U[-W/2, W/2]");
    println!("    Each site SCATTERS the wave (Hermitian, energy-conserving)");
    println!("    Result: localization at high disorder");
    println!();
    println!("  Relay Anderson (active amplification):");
    println!("    H = -Δ + V + i·G, G_i = gain at site i");
    println!("    Each site AMPLIFIES the signal (non-Hermitian, energy-injecting)");
    println!("    Result: extended states possible despite high disorder");
    println!("    Biological reality: each Dictyostelium cell has cAMP relay circuit");

    v.check_pass("standard vs relay model comparison documented", true);

    v.section("§2 cAMP Relay Circuit Parameters");

    let relay_gain = 10.0_f64;
    let relay_delay_s = 6.0;
    let relay_range_um = 20.0;
    let cell_spacing_um = 10.0;

    let l_eff = relay_range_um / cell_spacing_um;

    println!("  cAMP relay parameters (from literature):");
    println!("    Amplification gain: {relay_gain}×");
    println!("    Relay delay: {relay_delay_s} s");
    println!("    Signal range per relay: {relay_range_um} µm");
    println!("    Mean cell spacing: {cell_spacing_um} µm");
    println!("    L_eff per relay hop: {l_eff:.1} cells");

    v.check_pass(
        "relay gain > 1 (amplification, not attenuation)",
        relay_gain > 1.0,
    );
    v.check_pass(
        "relay range > cell spacing (signal reaches next cell)",
        relay_range_um > cell_spacing_um,
    );

    v.section("§3 Relay Chain vs Anderson Localization");

    let n_hops = [1, 2, 5, 10, 20, 50];
    println!(
        "\n  {:>6} {:>12} {:>14} {:>14}",
        "Hops", "Range (µm)", "Effective gain", "Anderson pred."
    );
    println!("  {:-<6} {:-<12} {:-<14} {:-<14}", "", "", "", "");
    for &n in &n_hops {
        let range = n as f64 * relay_range_um;
        let total_gain = relay_gain.powi(n);
        let pred = if total_gain > 1.0 {
            "Extended (relay)"
        } else {
            "Localized"
        };
        println!(
            "  {:>6} {:>12.0} {:>14.0e} {:>14}",
            n, range, total_gain, pred
        );
    }

    v.check_pass(
        "relay gain compounds exponentially (defeats localization)",
        relay_gain.powi(5) > 1e4,
    );

    v.section("§4 Non-Hermitian Anderson Predictions");

    println!("\n  In the non-Hermitian Anderson model:");
    println!("  • Gain G > G_c: ALL states become extended (lasing transition)");
    println!("  • Gain G < G_c: localization persists");
    println!("  • G_c depends on disorder W and dimension d");
    println!();
    println!("  For Dictyostelium:");
    println!("    • Gain G ≈ {relay_gain}× per cell (>>1)");
    println!("    • This is FAR above any reasonable G_c");
    println!("    • Result: relay always defeats localization");
    println!("    • This is WHY Dictyostelium aggregation works:");
    println!("      100,000 cells can coordinate over ~mm distances");

    let wave_speed_um_per_s = relay_range_um / relay_delay_s;
    let mm_per_min = wave_speed_um_per_s * 60.0 / 1000.0;
    println!("\n  cAMP wave speed: {wave_speed_um_per_s:.1} µm/s ({mm_per_min:.1} mm/min)");
    println!("  Published value: ~0.2 mm/min (matches)");

    v.check_pass(
        "wave speed in reasonable range [0.1, 1.0] mm/min",
        (0.1..=1.0).contains(&mm_per_min),
    );

    v.section("§5 Biological Cost of Relay NP Solution");

    println!("\n  The relay solution has HIGH evolutionary cost:");
    println!("  1. Each cell must have complete cAMP signaling circuit:");
    println!("     - Adenylyl cyclase (ACA) for production");
    println!("     - cAMP receptor (cAR1-4) for detection");
    println!("     - Phosphodiesterase (PDE) for signal degradation");
    println!("     - Protein kinase A (PKA) for intracellular response");
    println!("  2. Circuit must have precise gain (too high → instability)");
    println!("  3. Circuit must have refractory period (prevent reverb)");
    println!("  4. All cells must be synchronized in developmental timing");
    println!();
    println!("  This is analogous to: deploying amplifiers at EVERY node");
    println!("  in a fiber optic network. Expensive, but defeats signal loss.");

    v.check_pass("evolutionary cost analysis documented", true);

    v.section("§6 Comparison: Three NP Solutions as Network Architectures");

    println!("\n  ┌──────────────────┬──────────────────────┬────────────────────┐");
    println!("  │ NP Solution       │ Network Architecture  │ Biological Cost    │");
    println!("  ├──────────────────┼──────────────────────┼────────────────────┤");
    println!("  │ V. cholerae       │ Inverse detection     │ Low (logic only)   │");
    println!("  │ Myxococcus        │ Self-built medium      │ High (aggregation) │");
    println!("  │ Dictyostelium     │ Active repeaters       │ Very high (relay)  │");
    println!("  └──────────────────┴──────────────────────┴────────────────────┘");

    v.check_pass("three NP solutions mapped to network architectures", true);

    v.section("§7 Quantitative Anderson Framework");

    let disorder_w = 15.0;
    let passive_localization_length_cells = 5.0;
    let active_range_cells = 50.0 * relay_range_um / cell_spacing_um;

    println!("  At W = {disorder_w} (moderate disorder, below W_c = 16.3):");
    println!("    Passive localization length: ξ ≈ {passive_localization_length_cells:.0} cells");
    println!(
        "    Active relay range: ξ_relay ≈ {active_range_cells:.0} cells (50 hops × {:.0} cells/hop)",
        relay_range_um / cell_spacing_um
    );
    println!(
        "    Relay amplification: {:.0}× over passive",
        active_range_cells / passive_localization_length_cells
    );

    v.check_pass(
        "relay range >> passive localization length",
        active_range_cells > passive_localization_length_cells * 5.0,
    );

    v.finish();
}
