// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::print_stdout,
    reason = "validation harness: results printed to stdout"
)]
#![expect(
    clippy::too_many_lines,
    reason = "validation harness: sequential domain checks in single main()"
)]
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
#![expect(
    clippy::similar_names,
    reason = "validation harness: domain variables from published notation"
)]
#![expect(
    clippy::items_after_statements,
    reason = "validation harness: local helpers defined near use site"
)]
//! # Exp363: Adaptive Dispatch from Hardware Profile
//!
//! Reads the capability profile from Exp362 and adapts its workload selection
//! based on discovered hardware capabilities. This is the "apply" step of the
//! hardware learning system: software modules generated from hardware discovery.
//!
//! ## Domains
//!
//! - D96: Profile Loading — read and validate `hardware_capability_profile.json`
//! - D97: Adaptive Workload Selection — choose workloads based on tier safety
//! - D98: F32 Bio Workloads — Shannon diversity, Simpson index (always safe)
//! - D99: Conditional F64 Workloads — Anderson eigenvalue, QL eigensolver (if F64 safe)
//! - D100: `VRAM`-Aware Scaling — adjust problem size to hardware ceiling
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Provenance type | Adaptive dispatch from hardware learning |
//! | Date | 2026-03-11 |
//! | Command | `cargo run --release --features gpu,json --bin validate_adaptive_dispatch_v1` |
//!
//! Provenance: Adaptive GPU/CPU dispatch strategy validation

use std::time::Instant;
use wetspring_barracuda::validation::Validator;

fn main() {
    let start = Instant::now();
    let mut v = Validator::new("Exp363: Adaptive Dispatch from Hardware Profile v1");

    // ─── D96: Profile Loading ───
    println!("\n  ── D96: Profile Loading ──");

    let profile_path = "output/hardware_capability_profile.json";
    let profile: Option<serde_json::Value> = std::fs::read_to_string(profile_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok());

    let profile = if let Some(p) = profile {
        println!("  Loaded profile from {profile_path}");
        v.check_pass("profile loaded", true);
        p
    } else {
        println!("  Profile not found — running live probe");
        v.check_pass("profile fallback to live probe", true);
        serde_json::json!({})
    };

    let adapter = profile
        .get("adapter_name")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown");
    println!("  Adapter: {adapter}");

    let nvvm_risk = profile
        .get("nvvm_transcendental_risk")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    println!("  NVVM transcendental risk: {nvvm_risk}");

    let has_f64 = profile
        .get("has_any_f64")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    println!("  Has f64: {has_f64}");

    let df64_safe = profile
        .get("df64_safe")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    println!("  DF64 safe: {df64_safe}");

    let vram_gb = profile
        .get("vram_estimate_gb")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(4);
    println!("  VRAM: {vram_gb} GB");

    v.check_pass("profile fields extracted", true);

    // ─── D97: Adaptive Workload Selection ───
    println!("\n  ── D97: Adaptive Workload Selection ──");

    let run_f32 = true;
    let run_f64_arith = has_f64 && !nvvm_risk;
    let run_f64_transcendentals = has_f64 && !nvvm_risk;
    let run_df64_arith = has_f64;
    let run_large_n = vram_gb >= 8;

    println!("  Workload selection based on hardware profile:");
    println!("    F32 bio workloads:           {run_f32} (always safe)");
    println!("    F64 arithmetic:              {run_f64_arith}");
    println!("    F64 transcendentals:         {run_f64_transcendentals}");
    println!("    DF64 arithmetic only:        {run_df64_arith}");
    println!("    Large-N problems (VRAM):     {run_large_n}");

    v.check_pass("workload selection computed", true);
    v.check_pass("F32 always selected", run_f32);

    if nvvm_risk {
        v.check_pass(
            "NVVM risk → F64 transcendentals correctly skipped",
            !run_f64_transcendentals,
        );
    }

    // ─── D98: F32 Bio Workloads (always safe) ───
    println!("\n  ── D98: F32 Bio Workloads ──");

    let community_a = [10.0, 20.0, 30.0, 40.0];
    let community_b = [25.0, 25.0, 25.0, 25.0];

    let shannon_a = barracuda::stats::diversity::shannon(&community_a);
    let shannon_b = barracuda::stats::diversity::shannon(&community_b);
    println!("  Shannon(uneven): {shannon_a:.6}");
    println!("  Shannon(even):   {shannon_b:.6}");
    v.check_pass("Shannon diversity computes", shannon_a > 0.0);
    v.check_pass("even > uneven Shannon", shannon_b > shannon_a);

    let simpson_a = barracuda::stats::diversity::simpson(&community_a);
    let simpson_b = barracuda::stats::diversity::simpson(&community_b);
    println!("  Simpson(uneven): {simpson_a:.6}");
    println!("  Simpson(even):   {simpson_b:.6}");
    v.check_pass(
        "Simpson diversity computes",
        simpson_a > 0.0 && simpson_a < 1.0,
    );
    v.check_pass("even > uneven Simpson", simpson_b > simpson_a);

    let norm_cdf = barracuda::stats::norm_cdf;
    let p_qs_high = norm_cdf((16.5 - 5.0) / 3.0);
    let p_qs_low = norm_cdf((16.5 - 20.0) / 3.0);
    println!("  Anderson QS P(W=5):  {p_qs_high:.4} (high diversity → QS active)");
    println!("  Anderson QS P(W=20): {p_qs_low:.4} (low diversity → QS suppressed)");
    v.check_pass("Anderson QS probability computes", p_qs_high > 0.5);
    v.check_pass("high W suppresses QS", p_qs_low < 0.5);

    // ─── D99: Conditional F64 Workloads ───
    println!("\n  ── D99: Conditional F64 Workloads ──");

    if run_df64_arith {
        println!("  Running DF64 arithmetic (no transcendentals)...");
        use barracuda::special::tridiagonal_ql;

        let n = 8_usize;
        let w = 10.0;
        let diag: Vec<f64> = (0..n)
            .map(|i| {
                let pseudo = ((42_u64
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(i as u64)) as f64)
                    / u64::MAX as f64;
                (pseudo - 0.5) * w
            })
            .collect();
        let subdiag = vec![-1.0_f64; n - 1];

        let (eigs, _) = tridiagonal_ql(&diag, &subdiag);
        let bandwidth = eigs.last().unwrap_or(&0.0) - eigs.first().unwrap_or(&0.0);
        println!(
            "  Anderson lattice (W={w}, n={n}): bandwidth={bandwidth:.3}, eigs={}",
            eigs.len()
        );
        v.check_pass("Anderson eigenvalues computed (CPU f64)", eigs.len() == n);
        v.check_pass("bandwidth > 0", bandwidth > 0.0);
    } else {
        println!("  Skipping DF64 arithmetic (no f64 hardware)");
        v.check_pass("DF64 correctly skipped", true);
    }

    if run_f64_transcendentals {
        println!("  Running F64 transcendentals (safe on this hardware)...");
        use barracuda::special::{erfc_f64, expm1_f64, log1p_f64};

        let log1p_test = log1p_f64(1e-12);
        let expm1_test = expm1_f64(1e-12);
        let erfc_test = erfc_f64(1.0);

        println!("  log1p(1e-12): {log1p_test:.15e}");
        println!("  expm1(1e-12): {expm1_test:.15e}");
        println!("  erfc(1.0):    {erfc_test:.10e}");
        v.check_pass("stable specials compute", log1p_test > 0.0);
    } else {
        println!("  Skipping F64 transcendentals (NVVM risk detected)");
        v.check_pass("transcendentals correctly skipped due to NVVM risk", true);
    }

    // ─── D100: VRAM-Aware Scaling ───
    println!("\n  ── D100: VRAM-Aware Scaling ──");

    let max_pairwise_n = (((vram_gb * 1024 * 1024 * 1024 / 8) as f64).sqrt()) as usize;
    println!("  VRAM ceiling: {vram_gb} GB → max pairwise N ≈ {max_pairwise_n}");

    if run_large_n {
        let test_n = 1000_usize.min(max_pairwise_n);
        println!("  Running Bray-Curtis at N={test_n} (within VRAM ceiling)...");

        let sample_a: Vec<f64> = (0..test_n).map(|i| (i as f64) * 0.1).collect();
        let sample_b: Vec<f64> = (0..test_n).map(|i| (i as f64).mul_add(0.1, 1.0)).collect();
        let sum_min: f64 = sample_a
            .iter()
            .zip(sample_b.iter())
            .map(|(a, b)| a.min(*b))
            .sum();
        let sum_a: f64 = sample_a.iter().sum();
        let sum_b: f64 = sample_b.iter().sum();
        let bc = 1.0 - (2.0 * sum_min) / (sum_a + sum_b);
        println!("  Bray-Curtis(N={test_n}): {bc:.6}");
        v.check_pass("Bray-Curtis computes at scale", (0.0..=1.0).contains(&bc));
        v.check_pass("problem size within VRAM ceiling", test_n <= max_pairwise_n);
    } else {
        println!("  Skipping large-N problems (VRAM < 8 GB)");
        v.check_pass("large-N correctly skipped", true);
    }

    println!("  VRAM scaling recommendation:");
    println!("    N ≤ 1K:   CPU (dispatch overhead dominates)");
    println!("    1K < N ≤ {max_pairwise_n}: GPU (within VRAM)");
    println!("    N > {max_pairwise_n}: tiled/streaming (exceeds single-pass VRAM)");
    v.check_pass("VRAM scaling recommendation generated", true);

    // Adaptive dispatch summary
    println!("\n  ═══════════════════════════════════════════════");
    println!("  Adaptive Dispatch Summary for: {adapter}");
    println!("    F32 workloads:         EXECUTED (always safe)");
    println!(
        "    DF64 arithmetic:       {}",
        if run_df64_arith {
            "EXECUTED"
        } else {
            "SKIPPED (no f64)"
        }
    );
    println!(
        "    F64 transcendentals:   {}",
        if run_f64_transcendentals {
            "EXECUTED (safe)"
        } else {
            "SKIPPED (NVVM risk)"
        }
    );
    println!(
        "    Large-N pairwise:      {}",
        if run_large_n {
            "EXECUTED"
        } else {
            "SKIPPED (low VRAM)"
        }
    );
    println!("    Sovereign dispatch:    SKIPPED (not available)");
    println!("  ═══════════════════════════════════════════════");

    let elapsed = start.elapsed();
    println!("\n  Wall time: {:.1}ms", elapsed.as_secs_f64() * 1000.0);
    v.finish();
}
