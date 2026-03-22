// SPDX-License-Identifier: AGPL-3.0-or-later

#![expect(clippy::expect_used, clippy::unwrap_used)]

use super::*;

#[test]
fn check_exact_match() {
    assert!(check("exact", 42.0, 42.0, 0.0));
}

#[test]
fn check_within_tolerance() {
    assert!(check("close", 42.001, 42.0, 0.01));
}

#[test]
fn check_outside_tolerance() {
    assert!(!check("far", 50.0, 42.0, 1.0));
}

#[test]
fn check_count_exact() {
    assert!(check_count("exact", 42, 42));
}

#[test]
fn check_count_mismatch() {
    assert!(!check_count("diff", 42, 43));
}

#[test]
fn print_result_pass() {
    assert!(print_result("test", 3, 3));
}

#[test]
fn print_result_fail() {
    assert!(!print_result("test", 2, 3));
}

#[test]
fn validator_accumulates() {
    let mut v = Validator {
        name: String::from("test"),
        passed: 0,
        total: 0,
    };
    v.check("ok", 1.0, 1.0, 0.0);
    v.check("fail", 2.0, 1.0, 0.0);
    v.check_count("count_ok", 5, 5);
    v.check_count("count_fail", 4, 5);
    assert_eq!(v.counts(), (2, 4));
}

#[test]
fn validator_section_does_not_count() {
    let v = Validator {
        name: String::from("test"),
        passed: 0,
        total: 0,
    };
    v.section("── some section ──");
    assert_eq!(v.counts(), (0, 0));
}

#[test]
fn validator_new_prints_banner() {
    let v = Validator::new("My Test Suite");
    assert_eq!(v.counts(), (0, 0));
}

#[test]
fn validator_check_count_u64_pass_and_fail() {
    let mut v = Validator::new("u64 test");
    v.check_count_u64("exact", 42, 42);
    v.check_count_u64("off", 41, 42);
    assert_eq!(v.counts(), (1, 2));
}

#[test]
fn validator_check_pass() {
    let mut v = Validator::new("check_pass test");
    v.check_pass("pass", true);
    v.check_pass("fail", false);
    assert_eq!(v.counts(), (1, 2));
}

#[test]
fn validator_full_workflow() {
    let mut v = Validator::new("integration");
    v.section("── section A ──");
    v.check("float ok", 1.0, 1.0, 0.0);
    v.check("float fail", 2.0, 1.0, 0.0);
    v.section("── section B ──");
    v.check_count("count ok", 5, 5);
    v.check_count("count fail", 4, 5);
    v.check_count_u64("u64 ok", 100, 100);
    assert_eq!(v.counts(), (3, 5));
}

// ── resolve_data_dir: pure function tests (zero unsafe) ──────

#[test]
fn resolve_specific_override_wins() {
    let dir = resolve_data_dir(Some("/explicit/path"), Some("/root"), "data/default");
    assert_eq!(dir.to_string_lossy(), "/explicit/path");
}

#[test]
fn resolve_data_root_with_existing_subpath() {
    let tmp = tempfile::tempdir().unwrap();
    let subpath = "resolve_test/data";
    let full = tmp.path().join(subpath);
    std::fs::create_dir_all(&full).unwrap();

    let root = tmp.path().to_string_lossy().to_string();
    let dir = resolve_data_dir(None, Some(&root), subpath);
    assert_eq!(dir, full);
}

#[test]
fn resolve_data_root_nonexistent_falls_through() {
    let dir = resolve_data_dir(None, Some("/nonexistent_root"), "sub/path");
    let s = dir.to_string_lossy();
    assert!(s.contains("sub/path"));
}

#[test]
fn resolve_no_env_uses_cwd_fallback() {
    let subpath = "___nonexistent_resolve_test/data";
    let dir = resolve_data_dir(None, None, subpath);
    assert_eq!(dir.to_string_lossy(), subpath);
}

#[test]
fn resolve_nested_subpath() {
    let dir = resolve_data_dir(None, None, "a/b/c/d/e");
    let s = dir.to_string_lossy();
    assert!(s.contains("a/b/c/d/e"));
}

// ── data_dir: integration (reads real env, no mutation) ──────

#[test]
fn data_dir_fallback_uses_manifest() {
    let dir = data_dir("WETSPRING_NONEXISTENT_12345", "data/test");
    let s = dir.to_string_lossy();
    assert!(s.contains("data/test"), "path should contain subpath");
}

#[test]
fn data_dir_env_override() {
    let key = "WETSPRING_TEST_DATA_DIR_UNIT";
    let dir = data_dir(key, "data/default");
    let s = dir.to_string_lossy();
    assert!(
        s.contains("data/default"),
        "fallback path should contain subpath"
    );
}

// ── check() edge cases ──────────────────────────────────────

#[test]
fn check_nan_always_fails() {
    assert!(!check("NaN test", f64::NAN, 0.0, 1.0));
    assert!(!check("NaN expected", 0.0, f64::NAN, 1.0));
}

#[test]
fn check_infinity_values() {
    assert!(!check("inf-inf is NaN", f64::INFINITY, f64::INFINITY, 0.0));
    assert!(!check("inf vs finite", f64::INFINITY, 0.0, 1e100));
}

#[test]
fn check_negative_zero() {
    assert!(check("neg zero", -0.0, 0.0, 0.0));
}

#[test]
fn check_boundary_tolerance() {
    assert!(!check("at exact boundary (fp rounding)", 1.01, 1.0, 0.01));
    assert!(check("within tolerance", 1.009, 1.0, 0.01));
    assert!(!check("past boundary", 1.02, 1.0, 0.01));
}

#[test]
fn check_count_zero() {
    assert!(check_count("both zero", 0, 0));
}

#[test]
fn check_count_large() {
    assert!(check_count("large", usize::MAX, usize::MAX));
    assert!(!check_count("large diff", usize::MAX, usize::MAX - 1));
}

#[test]
fn print_result_zero_total_is_failure() {
    assert!(!print_result("empty", 0, 0));
}

#[test]
fn validator_all_pass() {
    let mut v = Validator {
        name: String::from("all-pass"),
        passed: 0,
        total: 0,
    };
    for i in 0..10 {
        v.check(&format!("check {i}"), 1.0, 1.0, 0.0);
    }
    assert_eq!(v.counts(), (10, 10));
}

#[test]
fn validator_all_fail() {
    let mut v = Validator {
        name: String::from("all-fail"),
        passed: 0,
        total: 0,
    };
    for i in 0..5 {
        v.check(&format!("fail {i}"), 999.0, 0.0, 0.0);
    }
    assert_eq!(v.counts(), (0, 5));
}

// ── resolve_data_dir: explicit branch coverage ──────────────────

#[test]
fn resolve_data_dir_specific_override() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_str().unwrap();
    let result = resolve_data_dir(Some(path), None, "fallback");
    assert_eq!(result, std::path::PathBuf::from(path));
}

#[test]
fn resolve_data_dir_data_root_exists() {
    let dir = tempfile::tempdir().unwrap();
    let subdir = dir.path().join("subpath");
    std::fs::create_dir_all(&subdir).unwrap();
    let result = resolve_data_dir(None, Some(dir.path().to_str().unwrap()), "subpath");
    assert_eq!(result, subdir);
}

#[test]
fn resolve_data_dir_data_root_subpath_missing() {
    let dir = tempfile::tempdir().unwrap();
    let result = resolve_data_dir(None, Some(dir.path().to_str().unwrap()), "nonexistent");
    // Falls through to CARGO_MANIFEST_DIR or deployment fallback
    assert!(result.to_str().unwrap().contains("nonexistent"));
}

#[test]
fn resolve_data_dir_deployment_fallback() {
    let result = resolve_data_dir(None, None, "custom/path");
    // Should either be the manifest-dir version or the raw subpath
    assert!(result.to_str().unwrap().contains("custom/path"));
}

#[test]
fn resolve_data_dir_cargo_manifest_fallback() {
    let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap();
    let subpath = "__validation_test_manifest_fallback";
    let full = base.join(subpath);
    std::fs::create_dir_all(&full).unwrap();
    let result = resolve_data_dir(None, None, subpath);
    assert_eq!(
        result.canonicalize().unwrap(),
        full.canonicalize().unwrap(),
        "CARGO_MANIFEST_DIR/../subpath should resolve to workspace_root/subpath"
    );
    std::fs::remove_dir_all(&full).unwrap();
}

// ── bench, timed_us, print_timing_table ────────────────────────

#[test]
fn bench_returns_result_and_elapsed_ms() {
    let (result, ms) = bench(|| 42);
    assert_eq!(result, 42);
    assert!(ms >= 0.0, "elapsed ms should be non-negative");
}

#[test]
fn timed_us_returns_result_and_elapsed_us() {
    let (result, us) = timed_us(|| 42u64);
    assert_eq!(result, 42);
    assert!(us >= 0.0, "elapsed µs should be non-negative");
}

#[test]
fn print_timing_table_formats_rows() {
    let rows = [
        ("Domain A", 100.0, 50.0, "OK"),
        ("Domain B", 200.0, 150.0, "FAIL"),
    ];
    print_timing_table(&rows);
    // No panic = success
}

// ── Validator edge cases ────────────────────────────────────────

#[test]
fn validator_empty_name() {
    let v = Validator::new("");
    assert_eq!(v.counts(), (0, 0));
}

#[test]
fn validator_zero_checks() {
    let v = Validator::new("zero-checks");
    // No check calls
    assert_eq!(v.counts(), (0, 0));
}

// ── Determinism (rerun-identical) tests ────────────────────────

#[test]
#[expect(clippy::float_cmp)]
fn determinism_diversity() {
    use crate::bio::diversity::{bray_curtis, chao1, shannon, simpson};

    let counts = [10.0, 20.0, 30.0, 40.0];
    let a = [1.0, 2.0, 3.0];
    let b = [2.0, 3.0, 4.0];

    let sh1 = shannon(&counts);
    let sh2 = shannon(&counts);
    assert_eq!(sh1, sh2, "shannon must be bitwise identical");

    let si1 = simpson(&counts);
    let si2 = simpson(&counts);
    assert_eq!(si1, si2, "simpson must be bitwise identical");

    let c1 = chao1(&counts);
    let c2 = chao1(&counts);
    assert_eq!(c1, c2, "chao1 must be bitwise identical");

    let bc1 = bray_curtis(&a, &b);
    let bc2 = bray_curtis(&a, &b);
    assert_eq!(bc1, bc2, "bray_curtis must be bitwise identical");
}

#[test]
#[expect(clippy::float_cmp)]
fn determinism_ode() {
    use crate::bio::qs_biofilm::{QsBiofilmParams, run_scenario};

    let y0 = [0.01, 0.0, 0.0, 2.0, 0.5];
    let t_end = 10.0;
    let dt = 0.1;
    let params = QsBiofilmParams::default();

    let r1 = run_scenario(&y0, t_end, dt, &params);
    let r2 = run_scenario(&y0, t_end, dt, &params);

    assert_eq!(r1.t.len(), r2.t.len(), "ODE trajectory length");
    for (i, (&t1, &t2)) in r1.t.iter().zip(r2.t.iter()).enumerate() {
        assert_eq!(t1, t2, "t[{i}] must be bitwise identical");
    }
    for (i, (&y1, &y2)) in r1.y.iter().zip(r2.y.iter()).enumerate() {
        assert_eq!(y1, y2, "y[{i}] must be bitwise identical");
    }
}

#[test]
#[expect(clippy::float_cmp)]
fn determinism_special_functions() {
    let x = 1.5;
    let erf1 = barracuda::special::erf(x);
    let erf2 = barracuda::special::erf(x);
    assert_eq!(erf1, erf2, "erf must be bitwise identical");

    let ncdf1 = barracuda::stats::norm_cdf(x);
    let ncdf2 = barracuda::stats::norm_cdf(x);
    assert_eq!(ncdf1, ncdf2, "norm_cdf must be bitwise identical");

    let lg1 = barracuda::special::ln_gamma(x).unwrap_or(f64::INFINITY);
    let lg2 = barracuda::special::ln_gamma(x).unwrap_or(f64::INFINITY);
    assert_eq!(lg1, lg2, "ln_gamma must be bitwise identical");
}

#[cfg(feature = "gpu")]
#[test]
#[expect(clippy::float_cmp)]
fn determinism_anderson_spectral() {
    use barracuda::spectral::{anderson_2d, lanczos, lanczos_eigenvalues, level_spacing_ratio};

    const SEED: u64 = 42;
    let l = 8_usize;
    let w = 2.0_f64;
    let n_iter = 30_usize;

    let mat1 = anderson_2d(l, l, w, SEED);
    let mat2 = anderson_2d(l, l, w, SEED);
    let tri1 = lanczos(&mat1, n_iter, SEED);
    let tri2 = lanczos(&mat2, n_iter, SEED);
    let eigs1 = lanczos_eigenvalues(&tri1);
    let eigs2 = lanczos_eigenvalues(&tri2);
    let r1 = level_spacing_ratio(&eigs1);
    let r2 = level_spacing_ratio(&eigs2);

    assert_eq!(eigs1.len(), eigs2.len(), "eigenvalue count");
    for (i, (&e1, &e2)) in eigs1.iter().zip(eigs2.iter()).enumerate() {
        assert_eq!(e1, e2, "eigenvalue[{i}] must be bitwise identical");
    }
    assert_eq!(r1, r2, "level_spacing_ratio must be bitwise identical");
}

#[test]
fn determinism_encoding_roundtrip() {
    use crate::encoding::{base64_decode, base64_encode};

    let data = b"Hello, deterministic world!";
    let enc1 = base64_encode(data);
    let enc2 = base64_encode(data);
    assert_eq!(enc1, enc2, "base64 encode must be identical");

    let dec1 = base64_decode(&enc1).expect("decode");
    let dec2 = base64_decode(&enc2).expect("decode");
    assert_eq!(dec1, dec2, "base64 decode must be identical");
}

#[test]
#[expect(clippy::float_cmp)]
fn determinism_fastq_parsing() {
    use crate::io::fastq::stats_from_file;
    use std::fs::File;
    use std::io::Write;

    let synthetic = "@r1\nACGTACGT\n+\nIIIIIIII\n@r2\nGGCCAATT\n+\n!!!!!!!!\n";
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("determinism.fastq");
    let mut f = File::create(&path).expect("create");
    f.write_all(synthetic.as_bytes()).expect("write");
    drop(f);

    let stats1 = stats_from_file(&path).expect("stats 1");
    let stats2 = stats_from_file(&path).expect("stats 2");

    assert_eq!(stats1.num_sequences, stats2.num_sequences);
    assert_eq!(stats1.total_bases, stats2.total_bases);
    assert_eq!(stats1.min_length, stats2.min_length);
    assert_eq!(stats1.max_length, stats2.max_length);
    assert_eq!(
        stats1.mean_length, stats2.mean_length,
        "mean_length bitwise"
    );
    assert_eq!(
        stats1.mean_quality, stats2.mean_quality,
        "mean_quality bitwise"
    );
    assert_eq!(stats1.gc_content, stats2.gc_content, "gc_content bitwise");
    assert_eq!(stats1.q30_count, stats2.q30_count);
    assert_eq!(stats1.length_distribution, stats2.length_distribution);
}

// ── DomainResult + print_domain_summary ─────────────────────

#[test]
fn domain_result_without_spring() {
    let domains = vec![
        DomainResult {
            name: "diversity",
            spring: None,
            ms: 10.5,
            checks: 4,
        },
        DomainResult {
            name: "phylogeny",
            spring: None,
            ms: 20.3,
            checks: 6,
        },
    ];
    print_domain_summary("test: no spring column", &domains);
}

#[test]
fn domain_result_with_spring() {
    let domains = vec![
        DomainResult {
            name: "diversity",
            spring: Some("wetSpring"),
            ms: 10.5,
            checks: 4,
        },
        DomainResult {
            name: "anderson",
            spring: Some("hotSpring"),
            ms: 5.0,
            checks: 3,
        },
    ];
    print_domain_summary("test: spring column", &domains);
}

#[test]
fn domain_result_empty() {
    print_domain_summary("test: empty", &[]);
}

#[test]
fn domain_result_mixed_spring() {
    let domains = vec![
        DomainResult {
            name: "local",
            spring: None,
            ms: 1.0,
            checks: 1,
        },
        DomainResult {
            name: "cross",
            spring: Some("groundSpring"),
            ms: 2.0,
            checks: 2,
        },
    ];
    print_domain_summary("test: mixed", &domains);
}
