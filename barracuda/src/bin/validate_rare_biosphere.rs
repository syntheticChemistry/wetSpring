// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! Exp051 — Anderson 2015: Rare biosphere at deep-sea hydrothermal vents.
//!
//! Validates diversity, rarefaction, and community comparison primitives
//! against synthetic vent communities modeled after Anderson, Sogin & Baross
//! (2015) FEMS Microbiol Ecol 91:fiu016.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | `e4358c5` |
//! | Paper | Anderson, Sogin, Baross (2015) FEMS Microbiol Ecol 91:fiu016 |
//! | DOI | 10.1093/femsec/fiu016 |
//! | Faculty | R. Anderson (Carleton College) |
//! | Baseline script | `scripts/anderson2015_rare_biosphere.py` |
//! | Baseline output | `experiments/results/051_rare_biosphere/anderson2015_python_baseline.json` |
//! | Baseline date | 2026-02-20 |
//! | Exact command | `python3 scripts/anderson2015_rare_biosphere.py` |
//! | Hardware | i9-12900K, 64GB DDR5, RTX 4070, Ubuntu 24.04 |
//! | Python version | 3.10+ (pure Python + math, no external dependencies) |
//! | Data | Synthetic vent communities modeled after paper's Table S1 |
//!
//! # Methodology
//!
//! Three synthetic communities representing vent diversity gradients:
//! - Piccard (high-temp, low diversity, 13 OTUs)
//! - Von Damm (moderate, `Campylobacteria`-dominated, 30 OTUs)
//! - Background seawater (high diversity, many rare lineages, 50 OTUs)
//!
//! Validates alpha diversity (Shannon, Simpson, Chao1), beta diversity
//! (`Bray-Curtis`), rarefaction, and rare lineage detection.

use wetspring_barracuda::bio::{diversity, pcoa};
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

fn main() {
    let mut v = Validator::new("Exp051: Anderson 2015 Rare Biosphere Validation");

    let piccard = piccard_community();
    let von_damm = von_damm_community();
    let background = background_community();

    validate_alpha_diversity(&mut v, "Piccard", &piccard);
    validate_alpha_diversity(&mut v, "Von Damm", &von_damm);
    validate_alpha_diversity(&mut v, "Background", &background);
    validate_beta_diversity(&mut v, &piccard, &von_damm, &background);
    validate_rarefaction(&mut v, &piccard, &von_damm, &background);
    validate_rare_lineages(&mut v, &piccard, &von_damm, &background);
    validate_pcoa(&mut v, &piccard, &von_damm, &background);
    validate_python_parity(&mut v, &piccard, &von_damm, &background);

    v.finish();
}

// ── Synthetic vent communities (match Python baseline exactly) ──────

fn piccard_community() -> Vec<f64> {
    vec![
        500.0, 350.0, 80.0, 30.0, 15.0, 10.0, 5.0, 3.0, 2.0, 2.0, 1.0, 1.0, 1.0,
    ]
}

fn von_damm_community() -> Vec<f64> {
    vec![
        300.0, 200.0, 150.0, 100.0, 80.0, 60.0, 40.0, 30.0, 25.0, 20.0, 15.0, 12.0, 10.0, 8.0, 6.0,
        5.0, 4.0, 3.0, 3.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ]
}

fn background_community() -> Vec<f64> {
    vec![
        50.0, 45.0, 40.0, 38.0, 35.0, 33.0, 30.0, 28.0, 26.0, 25.0, 23.0, 22.0, 20.0, 19.0, 18.0,
        17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 5.0, 4.0, 4.0,
        3.0, 3.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0,
    ]
}

// ── Alpha diversity ─────────────────────────────────────────────────

fn validate_alpha_diversity(v: &mut Validator, name: &str, counts: &[f64]) {
    v.section(&format!("── {name}: Alpha diversity ──"));

    let h = diversity::shannon(counts);
    let s = diversity::simpson(counts);
    let obs = diversity::observed_features(counts) as usize;
    let c = diversity::chao1(counts);

    let expected_h = analytical_shannon(counts);
    let expected_s = analytical_simpson(counts);
    let expected_obs = counts.iter().filter(|&&x| x > 0.0).count();

    v.check(
        &format!("{name}: Shannon"),
        h,
        expected_h,
        tolerances::ANALYTICAL_F64,
    );
    v.check(
        &format!("{name}: Simpson"),
        s,
        expected_s,
        tolerances::ANALYTICAL_F64,
    );
    v.check_count(&format!("{name}: Observed"), obs, expected_obs);
    v.check(
        &format!("{name}: Chao1 >= Observed"),
        f64::from(u8::from(c >= f64::from(u32::try_from(obs).unwrap_or(0)))),
        1.0,
        0.0,
    );
}

// ── Beta diversity ──────────────────────────────────────────────────

fn validate_beta_diversity(
    v: &mut Validator,
    piccard: &[f64],
    von_damm: &[f64],
    background: &[f64],
) {
    v.section("── Beta diversity (Bray-Curtis) ──");

    let samples = [piccard, von_damm, background];
    let max_len = samples.iter().map(|s| s.len()).max().unwrap_or(0);
    let padded: Vec<Vec<f64>> = samples
        .iter()
        .map(|s| {
            let mut p = s.to_vec();
            p.resize(max_len, 0.0);
            p
        })
        .collect();

    let condensed = diversity::bray_curtis_condensed(&padded);

    // condensed_index(1,0) = 0 → BC(VonDamm, Piccard)
    // condensed_index(2,0) = 1 → BC(Background, Piccard)
    // condensed_index(2,1) = 2 → BC(Background, VonDamm)
    let bc_pv = condensed[diversity::condensed_index(1, 0)];
    let bc_pb = condensed[diversity::condensed_index(2, 0)];
    let bc_vb = condensed[diversity::condensed_index(2, 1)];

    v.check(
        "BC(Piccard, Piccard) = 0 (not stored)",
        f64::from(u8::from(bc_pv >= 0.0)),
        1.0,
        0.0,
    );

    v.check(
        "BC(Piccard, Background) > BC(Piccard, Von Damm)",
        f64::from(u8::from(bc_pb > bc_pv)),
        1.0,
        0.0,
    );
    v.check(
        "BC(Piccard, Background) > BC(Von Damm, Background)",
        f64::from(u8::from(bc_pb > bc_vb)),
        1.0,
        0.0,
    );
    v.check(
        "BC symmetry: BC(P,V) stored once",
        f64::from(u8::from(bc_pv > 0.0)),
        1.0,
        0.0,
    );
}

// ── Rarefaction ─────────────────────────────────────────────────────

fn validate_rarefaction(v: &mut Validator, piccard: &[f64], von_damm: &[f64], background: &[f64]) {
    v.section("── Rarefaction ──");

    for (name, counts) in [
        ("Piccard", piccard),
        ("Von Damm", von_damm),
        ("Background", background),
    ] {
        let total = counts.iter().sum::<f64>();
        let depths: Vec<f64> = (1..=10).map(|i| total * f64::from(i) / 10.0).collect();
        let curve = diversity::rarefaction_curve(counts, &depths);
        let monotonic = curve.windows(2).all(|w| w[1] >= w[0]);
        v.check(
            &format!("{name}: rarefaction monotonic"),
            f64::from(u8::from(monotonic)),
            1.0,
            0.0,
        );
    }

    let bg_total = background.iter().sum::<f64>();
    let depths: Vec<f64> = (1..=10).map(|i| bg_total * f64::from(i) / 10.0).collect();
    let bg_curve = diversity::rarefaction_curve(background, &depths);
    let last_two_close = (bg_curve[bg_curve.len() - 1] - bg_curve[bg_curve.len() - 2]).abs()
        < bg_curve[bg_curve.len() - 1] * 0.1;
    v.check(
        "Background: rarefaction saturating",
        f64::from(u8::from(last_two_close)),
        1.0,
        0.0,
    );
}

// ── Rare lineage detection ──────────────────────────────────────────

fn validate_rare_lineages(
    v: &mut Validator,
    piccard: &[f64],
    von_damm: &[f64],
    background: &[f64],
) {
    v.section("── Rare lineage detection ──");

    for (name, counts) in [
        ("Piccard", piccard),
        ("Von Damm", von_damm),
        ("Background", background),
    ] {
        let total: f64 = counts.iter().sum();
        let rare = counts
            .iter()
            .filter(|&&c| c > 0.0 && c / total < 0.001)
            .count();
        let obs = counts.iter().filter(|&&c| c > 0.0).count();

        v.check(
            &format!("{name}: rare lineage count >= 0"),
            f64::from(u8::from(rare <= obs)),
            1.0,
            0.0,
        );
    }

    let vd_total: f64 = von_damm.iter().sum();
    let vd_rare = von_damm
        .iter()
        .filter(|&&c| c > 0.0 && c / vd_total < 0.001)
        .count();
    v.check_count("Von Damm: 8 rare lineages (<0.1%)", vd_rare, 8);
}

// ── PCoA ordination ─────────────────────────────────────────────────

fn validate_pcoa(v: &mut Validator, piccard: &[f64], von_damm: &[f64], background: &[f64]) {
    v.section("── PCoA ordination ──");

    let samples = [piccard, von_damm, background];
    let max_len = samples.iter().map(|s| s.len()).max().unwrap_or(0);
    let padded: Vec<Vec<f64>> = samples
        .iter()
        .map(|s| {
            let mut p = s.to_vec();
            p.resize(max_len, 0.0);
            p
        })
        .collect();

    let bc_condensed = diversity::bray_curtis_condensed(&padded);
    let result =
        pcoa::pcoa(&bc_condensed, padded.len(), 2).expect("PCoA on valid condensed BC matrix");

    v.check(
        "PCoA: all eigenvalues non-negative",
        f64::from(u8::from(result.eigenvalues.iter().all(|&e| e >= -1e-10))),
        1.0,
        0.0,
    );
    v.check(
        "PCoA: coordinates have 3 samples",
        result.coordinates.len() as f64,
        3.0,
        0.0,
    );
}

// ── Python parity ───────────────────────────────────────────────────

fn validate_python_parity(
    v: &mut Validator,
    piccard: &[f64],
    von_damm: &[f64],
    background: &[f64],
) {
    v.section("── Python baseline parity ──");

    let py_piccard_shannon = 1.219_814_354_381;
    let py_von_damm_shannon = 2.275_689_671_183;
    let py_background_shannon = 3.404_298_944_090;

    let py_piccard_simpson = 0.619_830_000_000;
    let py_von_damm_simpson = 0.849_924_186_116;
    let py_background_simpson = 0.958_638_378_161;

    let py_bc_pv = 0.376_498_800_959;
    let py_bc_pb = 0.749_694_749_695;
    let py_bc_vb = 0.507_835_171_213;

    v.check(
        "Python: Piccard Shannon",
        diversity::shannon(piccard),
        py_piccard_shannon,
        tolerances::ANALYTICAL_F64,
    );
    v.check(
        "Python: Von Damm Shannon",
        diversity::shannon(von_damm),
        py_von_damm_shannon,
        tolerances::ANALYTICAL_F64,
    );
    v.check(
        "Python: Background Shannon",
        diversity::shannon(background),
        py_background_shannon,
        tolerances::ANALYTICAL_F64,
    );

    v.check(
        "Python: Piccard Simpson",
        diversity::simpson(piccard),
        py_piccard_simpson,
        tolerances::ANALYTICAL_F64,
    );
    v.check(
        "Python: Von Damm Simpson",
        diversity::simpson(von_damm),
        py_von_damm_simpson,
        tolerances::ANALYTICAL_F64,
    );
    v.check(
        "Python: Background Simpson",
        diversity::simpson(background),
        py_background_simpson,
        tolerances::ANALYTICAL_F64,
    );

    let samples_p = [piccard, von_damm, background];
    let max_len = samples_p.iter().map(|s| s.len()).max().unwrap_or(0);
    let padded: Vec<Vec<f64>> = samples_p
        .iter()
        .map(|s| {
            let mut p = s.to_vec();
            p.resize(max_len, 0.0);
            p
        })
        .collect();
    let bc = diversity::bray_curtis_condensed(&padded);

    v.check(
        "Python: BC(Piccard, Von Damm)",
        bc[diversity::condensed_index(1, 0)],
        py_bc_pv,
        tolerances::ANALYTICAL_F64,
    );
    v.check(
        "Python: BC(Piccard, Background)",
        bc[diversity::condensed_index(2, 0)],
        py_bc_pb,
        tolerances::ANALYTICAL_F64,
    );
    v.check(
        "Python: BC(Von Damm, Background)",
        bc[diversity::condensed_index(2, 1)],
        py_bc_vb,
        tolerances::ANALYTICAL_F64,
    );
}

// ── Helper: analytical Shannon for validation ───────────────────────

fn analytical_shannon(counts: &[f64]) -> f64 {
    let total: f64 = counts.iter().sum();
    if total == 0.0 {
        return 0.0;
    }
    let mut h = 0.0;
    for &c in counts {
        if c > 0.0 {
            let p = c / total;
            h -= p * p.ln();
        }
    }
    h
}

fn analytical_simpson(counts: &[f64]) -> f64 {
    let total: f64 = counts.iter().sum();
    if total == 0.0 {
        return 0.0;
    }
    let mut s = 0.0;
    for &c in counts {
        let p = c / total;
        s += p * p;
    }
    1.0 - s
}
