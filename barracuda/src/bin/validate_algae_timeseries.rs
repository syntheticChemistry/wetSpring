// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exp039 — Algal pond time-series diversity surveillance.
//!
//! # Provenance
//!
//! | Item            | Value                                                               |
//! |-----------------|---------------------------------------------------------------------|
//! | Baseline commit | `e4358c5`                                                           |
//! | Baseline script | `scripts/algae_timeseries_baseline.py`                              |
//! | Baseline output | `experiments/results/039_algae_timeseries/python_baseline.json`      |
//! | Data source     | PRJNA382322 (128 samples, Nannochloropsis raceway, 4-month series)  |
//! | Proxy for       | Cahill #13 (phage biocontrol monitoring)                            |
//! | Date            | 2026-02-20                                                          |
//! | Exact command   | `python3 scripts/algae_timeseries_baseline.py`                      |
//! | Hardware        | i9-12900K, 64GB DDR5, RTX 4070, Ubuntu 24.04                       |
//!
//! Validates time-series diversity tracking: Shannon over time, Bray-Curtis
//! between consecutive timepoints, and Z-score anomaly detection.

use wetspring_barracuda::bio::diversity::{bray_curtis, shannon};
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

fn rolling_zscore(values: &[f64], window: usize) -> Vec<f64> {
    let mut zscores = vec![0.0; values.len()];
    for i in window..values.len() {
        let w = &values[i - window..i];
        let n = w.len();
        let mu: f64 = w.iter().sum::<f64>() / f64::from(u32::try_from(n).unwrap_or(u32::MAX));
        let var: f64 = w.iter().map(|x| (x - mu).powi(2)).sum::<f64>()
            / f64::from(u32::try_from(n).unwrap_or(u32::MAX));
        let std = var.sqrt();
        if std > 0.0 {
            zscores[i] = (values[i] - mu) / std;
        }
    }
    zscores
}

fn main() {
    let mut v = Validator::new("Exp039: Algal Pond Time-Series Surveillance");

    // ── Section 1: Diversity module on synthetic community ──────
    v.section("── Shannon diversity on known distributions ──");
    let uniform_4 = vec![25.0, 25.0, 25.0, 25.0];
    v.check(
        "Shannon(uniform,4)",
        shannon(&uniform_4),
        4.0_f64.ln(),
        tolerances::PYTHON_PARITY,
    );

    let dominant = vec![90.0, 5.0, 3.0, 2.0];
    let h_dom = shannon(&dominant);
    let h_uni = shannon(&uniform_4);
    v.check_count("dominant < uniform", usize::from(h_dom < h_uni), 1);

    // ── Section 2: Bray-Curtis on consecutive samples ──────────
    v.section("── Bray-Curtis between consecutive timepoints ──");
    let s1 = vec![50.0, 30.0, 20.0];
    let s2 = vec![50.0, 30.0, 20.0];
    v.check(
        "BC(identical)",
        bray_curtis(&s1, &s2),
        0.0,
        tolerances::PYTHON_PARITY,
    );

    let s3 = vec![100.0, 0.1, 0.1];
    let bc_shift = bray_curtis(&s1, &s3);
    let bc_positive = bc_shift > 0.0 && bc_shift <= 1.0;
    v.check_count("BC(shifted) in (0,1]", usize::from(bc_positive), 1);

    // ── Section 3: Anomaly detection via Z-score ───────────────
    v.section("── Z-score anomaly detection ──");
    // Use slight variation so std > 0, then inject a strong anomaly
    let mut anomaly_vals: Vec<f64> = (0..30)
        .map(|i| 0.01f64.mul_add(f64::from(i), 3.0))
        .collect();
    anomaly_vals[20] = 0.5;
    let zscores = rolling_zscore(&anomaly_vals, 5);
    let has_spike = zscores[20..].iter().any(|&z| z.abs() > 2.0);
    v.check_count("Z-score detects anomaly", usize::from(has_spike), 1);

    let stable: Vec<f64> = (0..30)
        .map(|i| 0.001f64.mul_add(f64::from(i), 3.0))
        .collect();
    let z_stable = rolling_zscore(&stable, 5);
    let no_spike = z_stable[5..].iter().all(|&z| z.abs() < 3.0);
    v.check_count("stable series: no large anomaly", usize::from(no_spike), 1);

    // ── Section 4: Time-series integration ─────────────────────
    v.section("── Shannon time series ──");
    let communities: Vec<Vec<f64>> = (0..20)
        .map(|t| {
            let base = 10.0f64.mul_add((f64::from(t) * 0.3).sin(), 50.0);
            vec![base, 100.0 - base, 30.0, 20.0]
        })
        .collect();
    let shannon_ts: Vec<f64> = communities.iter().map(|c| shannon(c)).collect();
    let all_positive = shannon_ts.iter().all(|&h| h > 0.0);
    v.check_count("all Shannon > 0", usize::from(all_positive), 1);
    let monotonic_test = shannon_ts.windows(2).any(|w| (w[0] - w[1]).abs() > 0.001);
    v.check_count("Shannon varies over time", usize::from(monotonic_test), 1);

    // ── Section 5: Crash detection scenario ────────────────────
    v.section("── Crash detection ──");
    let mut crash_ts: Vec<f64> = (0..50)
        .map(|i| 0.01f64.mul_add(f64::from(i), 3.0))
        .collect();
    crash_ts[35] = 0.5; // simulate pond crash
    let z_crash = rolling_zscore(&crash_ts, 5);
    let crash_detected = z_crash[35].abs() > 2.0;
    v.check_count("crash Z-score > 2", usize::from(crash_detected), 1);

    // ── Section 6: Determinism ─────────────────────────────────
    v.section("── Determinism ──");
    let h1 = shannon(&dominant);
    let h2 = shannon(&dominant);
    v.check("Shannon deterministic", h1, h2, 0.0);
    let bc1 = bray_curtis(&s1, &s3);
    let bc2 = bray_curtis(&s1, &s3);
    v.check("BC deterministic", bc1, bc2, 0.0);

    v.finish();
}
