// SPDX-License-Identifier: AGPL-3.0-or-later
use super::*;
use crate::bio::variant_caller::CallerConfig;

#[test]
fn threshold_at_generation_zero_is_floor() {
    let t = LteeThresholds::at_generation(0);
    assert!((t.polymorphism_threshold() - 0.05).abs() < 1e-10);
    assert!((t.consensus_threshold() - 0.95).abs() < 1e-10);
}

#[test]
fn threshold_increases_with_generation() {
    let t500 = LteeThresholds::at_generation(500);
    let t5000 = LteeThresholds::at_generation(5_000);
    let t50000 = LteeThresholds::at_generation(50_000);

    let f500 = t500.polymorphism_threshold();
    let f5000 = t5000.polymorphism_threshold();
    let f50000 = t50000.polymorphism_threshold();

    assert!(f500 < f5000, "gen 500 < gen 5000: {f500} vs {f5000}");
    assert!(f5000 < f50000, "gen 5000 < gen 50000: {f5000} vs {f50000}");
}

#[test]
fn threshold_bounded_by_ceiling() {
    let t = LteeThresholds::at_generation(1_000_000);
    let f = t.polymorphism_threshold();
    // After many generations, should approach ceiling (0.20) but never exceed it
    assert!(
        f <= 0.20 + 1e-10,
        "threshold should not exceed ceiling: {f}"
    );
    assert!(f > 0.19, "should be near ceiling at 1M generations: {f}");
}

#[test]
fn threshold_at_tau_is_63_percent() {
    // At g = τ, the saturation is (1 - e^(-1)) ≈ 0.6321
    let t = LteeThresholds::at_generation(10_000); // τ = 10,000
    let f = t.polymorphism_threshold();
    // Expected: 0.05 + (0.20 - 0.05) * 0.6321 ≈ 0.05 + 0.0948 ≈ 0.1448
    assert!(
        (f - 0.1448).abs() < 0.001,
        "at τ, threshold should be ~0.145: {f}"
    );
}

#[test]
fn consensus_threshold_is_complement() {
    let t = LteeThresholds::at_generation(5_000);
    let poly = t.polymorphism_threshold();
    let cons = t.consensus_threshold();
    assert!(
        (poly + cons - 1.0).abs() < 1e-10,
        "poly + consensus should = 1: {poly} + {cons}"
    );
}

#[test]
fn quality_threshold_increases_with_generation() {
    let q0 = LteeThresholds::at_generation(0).quality_threshold();
    let q5000 = LteeThresholds::at_generation(5_000).quality_threshold();
    let q50000 = LteeThresholds::at_generation(50_000).quality_threshold();

    assert!((q0 - 6.0).abs() < 1e-10, "gen 0 quality = 6: {q0}");
    assert!(q5000 > q0, "gen 5000 > gen 0: {q5000} vs {q0}");
    assert!(q50000 > q5000, "gen 50000 > gen 5000: {q50000} vs {q5000}");
    assert!(q50000 < 20.0 + 1e-10, "bounded by ceiling: {q50000}");
}

#[test]
fn apply_to_overrides_config_correctly() {
    let t = LteeThresholds::at_generation(5_000);
    let default_cfg = CallerConfig::default();
    let applied = t.apply_to(&default_cfg);

    assert!((applied.min_alt_frequency - t.polymorphism_threshold()).abs() < 1e-10);
    assert!((applied.min_quality - t.quality_threshold()).abs() < 1e-10);
    // Other fields preserved
    assert_eq!(applied.min_depth, default_cfg.min_depth);
    assert_eq!(applied.min_alt_count, default_cfg.min_alt_count);
    assert!(applied.quality_weighted); // preserved
    assert!(applied.binomial_quality); // preserved
}

#[test]
fn custom_parameters() {
    let t = LteeThresholds::custom(0.02, 0.30, 5_000.0, 5_000);
    let f = t.polymorphism_threshold();
    // At τ=5000, gen=5000: saturation ≈ 0.6321
    // Expected: 0.02 + (0.30 - 0.02) * 0.6321 ≈ 0.02 + 0.177 ≈ 0.197
    assert!(
        (f - 0.197).abs() < 0.001,
        "custom params: expected ~0.197, got {f}"
    );
}

#[test]
fn monotonically_increasing() {
    let thresholds: Vec<f64> = (0..100)
        .map(|i| LteeThresholds::at_generation(i * 1000).polymorphism_threshold())
        .collect();
    for window in thresholds.windows(2) {
        assert!(
            window[1] >= window[0],
            "threshold must be monotonically increasing: {} -> {}",
            window[0],
            window[1]
        );
    }
}
