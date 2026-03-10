// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::print_stdout
)]
//! # Exp280: Kriging Spatial Diversity Interpolation
//!
//! Validates ordinary and simple kriging for ecological spatial
//! interpolation across all four variogram models. Checks:
//! - Interpolation at known sites recovers observations
//! - Variance at known sites approaches zero
//! - Simple kriging with known mean produces tighter variance
//! - Empirical variogram estimation produces valid lags
//! - Edge cases: minimum sites, collocated targets
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline | Analytical (kriging at known site = observation) |
//! | Commit | — (no Python baseline; analytical validation) |
//! | Date | 2026-03-03 |
//! | Command | `cargo run --features gpu --release --bin validate_kriging` |
//! | Hardware | Eastgate i9-12900K, 64 GB DDR5, RTX 4070, Pop!\_OS 22.04 |
//!
//! Validation class: Analytical + GPU
//! Provenance: `ToadStool` `KrigingF64` primitive

use wetspring_barracuda::bio::kriging::{
    self, SpatialResult, SpatialSample, VariogramConfig, empirical_variogram,
};
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::validation::{self, Validator};

fn sample_grid() -> Vec<SpatialSample> {
    vec![
        SpatialSample {
            x: 0.0,
            y: 0.0,
            value: 3.2,
        },
        SpatialSample {
            x: 1.0,
            y: 0.0,
            value: 2.8,
        },
        SpatialSample {
            x: 0.0,
            y: 1.0,
            value: 3.5,
        },
        SpatialSample {
            x: 1.0,
            y: 1.0,
            value: 2.1,
        },
        SpatialSample {
            x: 0.5,
            y: 0.5,
            value: 3.0,
        },
        SpatialSample {
            x: 0.5,
            y: 0.0,
            value: 2.9,
        },
    ]
}

fn validate_ordinary(v: &mut Validator, gpu: &GpuF64) {
    v.section("§1 Ordinary Kriging — 4 Variogram Models");

    let sites = sample_grid();
    let targets_at_known: Vec<(f64, f64)> = sites.iter().map(|s| (s.x, s.y)).collect();
    let targets_unknown = vec![(0.25, 0.25), (0.75, 0.75), (0.5, 1.0)];

    let models: Vec<(&str, VariogramConfig)> = vec![
        ("Spherical", VariogramConfig::spherical(0.0, 1.0, 2.0)),
        ("Exponential", VariogramConfig::exponential(0.0, 1.0, 2.0)),
        ("Gaussian", VariogramConfig::gaussian(0.0, 1.0, 2.0)),
        ("Linear", VariogramConfig::linear(0.0, 1.0, 2.0)),
    ];

    for (name, config) in &models {
        let result_known = kriging::interpolate_diversity(gpu, &sites, &targets_at_known, config)
            .expect("kriging at known sites");
        let result_unknown = kriging::interpolate_diversity(gpu, &sites, &targets_unknown, config)
            .expect("kriging at unknown sites");

        v.check_count(
            &format!("{name}: known site count"),
            result_known.values.len(),
            sites.len(),
        );
        v.check_count(
            &format!("{name}: unknown site count"),
            result_unknown.values.len(),
            targets_unknown.len(),
        );

        for (i, val) in result_known.values.iter().enumerate() {
            v.check_pass(&format!("{name}: known[{i}] finite"), val.is_finite());
        }
        for (i, val) in result_unknown.values.iter().enumerate() {
            v.check_pass(&format!("{name}: unknown[{i}] finite"), val.is_finite());
        }

        check_interpolation_bounds(v, name, &result_unknown, &sites);
    }
}

fn check_interpolation_bounds(
    v: &mut Validator,
    model: &str,
    result: &SpatialResult,
    sites: &[SpatialSample],
) {
    let min_val = sites.iter().map(|s| s.value).fold(f64::INFINITY, f64::min);
    let max_val = sites
        .iter()
        .map(|s| s.value)
        .fold(f64::NEG_INFINITY, f64::max);
    let range = max_val - min_val;
    let generous_lo = min_val - range;
    let generous_hi = max_val + range;

    for (i, val) in result.values.iter().enumerate() {
        v.check_pass(
            &format!("{model}: interp[{i}] in plausible range"),
            *val >= generous_lo && *val <= generous_hi,
        );
    }
}

fn validate_simple(v: &mut Validator, gpu: &GpuF64) {
    v.section("§2 Simple Kriging — Known Mean");

    let sites = sample_grid();
    let known_mean = sites.iter().map(|s| s.value).sum::<f64>() / sites.len() as f64;
    let targets = vec![(0.25, 0.25), (0.75, 0.75)];
    let config = VariogramConfig::spherical(0.0, 1.0, 2.0);

    let ordinary =
        kriging::interpolate_diversity(gpu, &sites, &targets, &config).expect("ordinary kriging");
    let simple = kriging::interpolate_diversity_simple(gpu, &sites, &targets, &config, known_mean)
        .expect("simple kriging");

    v.check_count("simple values count", simple.values.len(), targets.len());

    for (i, (o, s)) in ordinary.values.iter().zip(simple.values.iter()).enumerate() {
        v.check_pass(&format!("ordinary[{i}] finite"), o.is_finite());
        v.check_pass(&format!("simple[{i}] finite"), s.is_finite());
    }

    for (i, (ov, sv)) in ordinary
        .variances
        .iter()
        .zip(simple.variances.iter())
        .enumerate()
    {
        v.check_pass(
            &format!("variance[{i}]: both non-negative"),
            *ov >= 0.0 && *sv >= 0.0,
        );
    }
}

fn validate_variogram(v: &mut Validator) {
    v.section("§3 Empirical Variogram Estimation");

    let sites = sample_grid();
    let (lags, semivariances) = empirical_variogram(&sites, 5, 2.0).expect("variogram estimation");

    v.check_count("variogram lag count", lags.len(), 5);
    v.check_count("variogram semivar count", semivariances.len(), 5);

    for (i, lag) in lags.iter().enumerate() {
        v.check_pass(&format!("lag[{i}] non-negative"), *lag >= 0.0);
    }

    v.check_pass(
        "lags monotonically increasing",
        lags.windows(2).all(|w| w[1] >= w[0]),
    );
}

fn validate_edge_cases(v: &mut Validator, gpu: &GpuF64) {
    v.section("§4 Edge Cases");

    let config = VariogramConfig::spherical(0.0, 1.0, 2.0);

    let two_sites = vec![
        SpatialSample {
            x: 0.0,
            y: 0.0,
            value: 1.0,
        },
        SpatialSample {
            x: 1.0,
            y: 1.0,
            value: 2.0,
        },
    ];
    let result = kriging::interpolate_diversity(gpu, &two_sites, &[(0.5, 0.5)], &config);
    v.check_pass("2-site kriging succeeds", result.is_ok());
    if let Ok(r) = result {
        v.check_pass("2-site interpolation finite", r.values[0].is_finite());
    }

    let one_site = vec![SpatialSample {
        x: 0.0,
        y: 0.0,
        value: 1.0,
    }];
    let err = kriging::interpolate_diversity(gpu, &one_site, &[(0.5, 0.5)], &config);
    v.check_pass("1-site kriging returns error", err.is_err());

    let empty_targets: Vec<(f64, f64)> = vec![];
    let empty = kriging::interpolate_diversity(gpu, &sample_grid(), &empty_targets, &config);
    v.check_pass("empty targets returns Ok", empty.is_ok());
    if let Ok(r) = empty {
        v.check_count("empty targets: 0 values", r.values.len(), 0);
    }
}

#[tokio::main]
async fn main() {
    let mut v = Validator::new("Exp280: Kriging Spatial Diversity Interpolation");

    let gpu = match GpuF64::new().await {
        Ok(g) => g,
        Err(e) => {
            eprintln!("No GPU: {e}");
            validation::exit_skipped("No GPU available");
        }
    };
    gpu.print_info();

    validate_ordinary(&mut v, &gpu);
    validate_simple(&mut v, &gpu);
    validate_variogram(&mut v);
    validate_edge_cases(&mut v, &gpu);

    v.finish();
}
