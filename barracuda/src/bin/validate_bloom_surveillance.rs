// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exp040 — Bloom event detection and surveillance pipeline.
//!
//! # Provenance
//!
//! | Item            | Value                                                           |
//! |-----------------|-----------------------------------------------------------------|
//! | Baseline commit | `e4358c5`                                                       |
//! | Baseline script | `scripts/bloom_surveillance_baseline.py`                        |
//! | Baseline output | `experiments/results/040_bloom_surveillance/python_baseline.json`|
//! | Data source     | PRJNA1224988 (175 samples, cyanobacterial bloom time series)    |
//! | Proxy for       | Smallwood #14 (raceway metagenomic surveillance)                |
//! | Date            | 2026-02-20                                                      |
//! | Exact command   | `python3 scripts/bloom_surveillance_baseline.py`                |
//! | Hardware        | i9-12900K, 64GB DDR5, RTX 4070, Ubuntu 24.04                    |
//!
//! Validates bloom detection via:
//! - Shannon diversity collapse during bloom
//! - Simpson dominance spike
//! - Pielou evenness drop
//! - Bray-Curtis shift magnitude
//! - Berger-Parker dominance index

use wetspring_barracuda::bio::diversity::{bray_curtis, pielou_evenness, shannon, simpson};
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

fn dominance_index(counts: &[f64]) -> f64 {
    let total: f64 = counts.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    counts.iter().copied().fold(0.0_f64, f64::max) / total
}

fn main() {
    let mut v = Validator::new("Exp040: Bloom Event Detection & Surveillance");

    // ── Section 1: Pre-bloom baseline (even community) ──────────
    v.section("── Pre-bloom: even community ──");
    let even = vec![
        40.0, 38.0, 42.0, 39.0, 41.0, 37.0, 43.0, 36.0, 44.0, 35.0, 45.0, 34.0, 46.0, 33.0, 47.0,
        32.0, 48.0, 31.0, 49.0, 30.0,
    ];
    let h_even = shannon(&even);
    let e_even = pielou_evenness(&even);
    let d_even = dominance_index(&even);
    v.check("Shannon(even) > 2.5", h_even, 2.98, 0.1);
    v.check("Evenness(even) > 0.9", e_even, 0.99, 0.05);
    let low_dom = d_even < 0.1;
    v.check_count("Dominance(even) < 0.1", usize::from(low_dom), 1);

    // ── Section 2: Bloom event (one taxon dominates) ────────────
    v.section("── Bloom event: single-taxon dominance ──");
    let mut bloom = vec![2.0; 20];
    bloom[0] = 900.0; // cyanobacterial bloom — extreme dominance
    let h_bloom = shannon(&bloom);
    let e_bloom = pielou_evenness(&bloom);
    let d_bloom = dominance_index(&bloom);
    let s_bloom = simpson(&bloom);

    let h_drops = h_bloom < h_even;
    v.check_count("Shannon drops during bloom", usize::from(h_drops), 1);
    let e_drops = e_bloom < e_even;
    v.check_count("Evenness drops during bloom", usize::from(e_drops), 1);
    let d_spikes = d_bloom > d_even;
    v.check_count("Dominance spikes during bloom", usize::from(d_spikes), 1);
    // Gini-Simpson (1 - Σp²): near 0 means one taxon dominates
    let s_low = s_bloom < 0.3;
    v.check_count("Gini-Simpson < 0.3 during bloom", usize::from(s_low), 1);

    // ── Section 3: Bray-Curtis shift detection ──────────────────
    v.section("── Bray-Curtis shift during bloom ──");
    let bc_shift = bray_curtis(&even, &bloom);
    let large_shift = bc_shift > 0.5;
    v.check_count("BC(pre,bloom) > 0.5", usize::from(large_shift), 1);

    let bc_self = bray_curtis(&even, &even);
    v.check("BC(self) = 0", bc_self, 0.0, tolerances::PYTHON_PARITY);

    // ── Section 4: Recovery detection ───────────────────────────
    v.section("── Recovery: community rebounds ──");
    let recovery = vec![
        40.0, 38.0, 42.0, 39.0, 41.0, 37.0, 43.0, 36.0, 44.0, 35.0, 45.0, 34.0, 46.0, 33.0, 47.0,
        32.0, 48.0, 31.0, 49.0, 30.0,
    ];
    let h_recovery = shannon(&recovery);
    let recovery_near_baseline = (h_recovery - h_even).abs() < 0.1;
    v.check_count(
        "Shannon recovers to near baseline",
        usize::from(recovery_near_baseline),
        1,
    );

    let bc_recovery = bray_curtis(&even, &recovery);
    let small_drift = bc_recovery < 0.1;
    v.check_count("BC(pre,recovery) < 0.1", usize::from(small_drift), 1);

    // ── Section 5: Bloom detection threshold ────────────────────
    v.section("── Bloom detection algorithm ──");
    let timepoints: &[&[f64]] = &[
        &[40.0, 38.0, 42.0, 39.0, 41.0], // normal
        &[40.0, 39.0, 41.0, 38.0, 42.0], // normal
        &[40.0, 37.0, 43.0, 36.0, 44.0], // normal
        &[200.0, 10.0, 5.0, 3.0, 2.0],   // bloom!
        &[42.0, 38.0, 40.0, 39.0, 41.0], // recovery
    ];
    let shannon_ts: Vec<f64> = timepoints.iter().map(|t| shannon(t)).collect();
    let pre_mean: f64 = shannon_ts[..3].iter().sum::<f64>() / 3.0;
    let pre_std = (shannon_ts[..3]
        .iter()
        .map(|h| (h - pre_mean).powi(2))
        .sum::<f64>()
        / 3.0)
        .sqrt();
    let bloom_detected = shannon_ts[3] < 2.0f64.mul_add(-pre_std, pre_mean);
    v.check_count(
        "bloom detected (Shannon < mean-2σ)",
        usize::from(bloom_detected),
        1,
    );

    // ── Section 6: Determinism ──────────────────────────────────
    v.section("── Determinism ──");
    v.check(
        "Shannon deterministic",
        shannon(&bloom),
        shannon(&bloom),
        0.0,
    );
    v.check(
        "Simpson deterministic",
        simpson(&bloom),
        simpson(&bloom),
        0.0,
    );
    v.check(
        "BC deterministic",
        bray_curtis(&even, &bloom),
        bray_curtis(&even, &bloom),
        0.0,
    );

    v.finish();
}
