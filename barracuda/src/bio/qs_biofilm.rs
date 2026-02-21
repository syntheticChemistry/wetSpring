// SPDX-License-Identifier: AGPL-3.0-or-later
//! Quorum sensing / c-di-GMP biofilm ODE model.
//!
//! Implements the Waters 2008 model for QS-controlled biofilm formation in
//! *Vibrio cholerae* through modulation of cyclic di-GMP levels.
//!
//! # References
//!
//! - Waters et al. 2008, *J Bacteriol* 190:2527-36
//! - Hammer & Bassler 2009, *J Bacteriol* 191:169-177
//! - Massie et al. 2012, *PNAS* 109:12746-51
//! - Bridges et al. 2022, *`PLoS` Biol* 20:e3001585
//!   (code: <https://zenodo.org/record/5519935>, CC-BY 4.0)
//!
//! # State variables
//!
//! | Index | Variable | Description |
//! |-------|----------|-------------|
//! | 0 | N | Cell density (OD equivalent, 0–1) |
//! | 1 | A | Autoinducer concentration (CAI-1/AI-2, µM) |
//! | 2 | H | HapR protein level (normalized, 0–1+) |
//! | 3 | C | c-di-GMP concentration (µM) |
//! | 4 | B | Biofilm state (VPS expression, 0–1) |

use super::ode::{rk4_integrate, OdeResult};

/// Model parameters for the QS / c-di-GMP / biofilm system.
///
/// Default values are from the literature (see module-level references).
#[derive(Debug, Clone)]
pub struct QsBiofilmParams {
    /// Maximum growth rate (h⁻¹). Logistic term uses `1 - N/K`.
    pub mu_max: f64,
    /// Carrying capacity (OD-equivalent). Cell density caps at this value.
    pub k_cap: f64,
    /// Cell death rate (h⁻¹).
    pub death_rate: f64,
    /// Autoinducer (CAI-1/AI-2) production rate per cell.
    pub k_ai_prod: f64,
    /// Autoinducer degradation/dilution rate.
    pub d_ai: f64,
    /// Max `HapR` production rate. Activated by AI via Hill function.
    pub k_hapr_max: f64,
    /// Half-saturation for AI activation of `HapR`.
    pub k_hapr_ai: f64,
    /// Hill coefficient for `HapR` activation by AI.
    pub n_hapr: f64,
    /// `HapR` degradation rate. Represses DGC, activates PDE.
    pub d_hapr: f64,
    /// Basal diguanylate cyclase activity. Produces c-di-GMP.
    pub k_dgc_basal: f64,
    /// DGC repression factor by `HapR`.
    pub k_dgc_rep: f64,
    /// Basal phosphodiesterase activity. Degrades c-di-GMP.
    pub k_pde_basal: f64,
    /// PDE activation by `HapR`. High QS → dispersal.
    pub k_pde_act: f64,
    /// c-di-GMP turnover/dilution rate.
    pub d_cdg: f64,
    /// Maximum biofilm promotion rate. Hill-saturated by c-di-GMP.
    pub k_bio_max: f64,
    /// Half-saturation for c-di-GMP activation of biofilm.
    pub k_bio_cdg: f64,
    /// Hill coefficient for biofilm promotion.
    pub n_bio: f64,
    /// Biofilm loss (dispersal) rate.
    pub d_bio: f64,
}

impl Default for QsBiofilmParams {
    fn default() -> Self {
        Self {
            mu_max: 0.8,
            k_cap: 1.0,
            death_rate: 0.02,
            k_ai_prod: 5.0,
            d_ai: 1.0,
            k_hapr_max: 1.0,
            k_hapr_ai: 0.5,
            n_hapr: 2.0,
            d_hapr: 0.5,
            k_dgc_basal: 2.0,
            k_dgc_rep: 0.8,
            k_pde_basal: 0.5,
            k_pde_act: 2.0,
            d_cdg: 0.3,
            k_bio_max: 1.0,
            k_bio_cdg: 1.5,
            n_bio: 2.0,
            d_bio: 0.2,
        }
    }
}

/// Hill activation: x^n / (k^n + x^n).
#[inline]
fn hill(x: f64, k: f64, n: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let xn = x.powf(n);
    xn / (k.powf(n) + xn)
}

/// Right-hand side of the wild-type QS/biofilm ODE system.
#[allow(clippy::many_single_char_names)] // standard ODE state vector notation
fn qs_rhs(state: &[f64], _t: f64, p: &QsBiofilmParams) -> Vec<f64> {
    let cell = state[0].max(0.0);
    let ai = state[1].max(0.0);
    let hapr = state[2].max(0.0);
    let cdg = state[3].max(0.0);
    let bio = state[4].max(0.0);

    let d_cell = (p.mu_max * cell).mul_add(1.0 - cell / p.k_cap, -(p.death_rate * cell));
    let d_ai = p.k_ai_prod.mul_add(cell, -p.d_ai * ai);
    let d_hapr = p
        .k_hapr_max
        .mul_add(hill(ai, p.k_hapr_ai, p.n_hapr), -p.d_hapr * hapr);

    let dgc_rate = p.k_dgc_basal * p.k_dgc_rep.mul_add(-hapr, 1.0).max(0.0);
    let pde_rate = p.k_pde_act.mul_add(hapr, p.k_pde_basal);
    let mut d_cdg = p.d_cdg.mul_add(-cdg, dgc_rate - pde_rate * cdg);
    if cdg < 1e-12 && d_cdg < 0.0 {
        d_cdg = 0.0;
    }

    let bio_promote = p.k_bio_max * hill(cdg, p.k_bio_cdg, p.n_bio);
    let d_bio = bio_promote.mul_add(1.0 - bio, -(p.d_bio * bio));

    vec![d_cell, d_ai, d_hapr, d_cdg, d_bio]
}

/// Biological bounds for the 5-variable system: all non-negative.
const CLAMP: [(f64, f64); 5] = [
    (0.0, f64::INFINITY), // N
    (0.0, f64::INFINITY), // A
    (0.0, f64::INFINITY), // H
    (0.0, f64::INFINITY), // C ≥ 0 (enzymatic floor)
    (0.0, 1.0),           // B ∈ [0, 1]
];

/// Run a scenario with given initial conditions and integration time.
#[must_use]
pub fn run_scenario(y0: &[f64; 5], t_end: f64, dt: f64, params: &QsBiofilmParams) -> OdeResult {
    let p = params.clone();
    rk4_integrate(
        move |y, t| qs_rhs(y, t, &p),
        y0,
        0.0,
        t_end,
        dt,
        Some(&CLAMP),
    )
}

/// Standard growth scenario: low inoculum → QS activation → biofilm dispersal.
///
/// At low density, c-di-GMP is high → biofilm ON.
/// At high density, `HapR` activates → c-di-GMP drops → biofilm OFF.
#[must_use]
pub fn scenario_standard_growth(params: &QsBiofilmParams, dt: f64) -> OdeResult {
    run_scenario(&[0.01, 0.0, 0.0, 2.0, 0.5], 24.0, dt, params)
}

/// High-density inoculum: QS immediately represses biofilm (dispersal).
#[must_use]
pub fn scenario_high_density(params: &QsBiofilmParams, dt: f64) -> OdeResult {
    run_scenario(&[0.8, 0.0, 0.0, 3.0, 0.8], 12.0, dt, params)
}

/// Δ`hapR` mutant: constitutive biofilm (`HapR` knocked out).
#[must_use]
pub fn scenario_hapr_mutant(params: &QsBiofilmParams, dt: f64) -> OdeResult {
    let mut p = params.clone();
    p.k_hapr_max = 0.0; // HapR production disabled
    p.d_hapr = 0.0; // no degradation needed either (starts at 0)
    run_scenario(&[0.01, 0.0, 0.0, 2.0, 0.5], 24.0, dt, &p)
}

/// DGC overexpression: elevated c-di-GMP despite QS activation.
#[must_use]
pub fn scenario_dgc_overexpression(params: &QsBiofilmParams, dt: f64) -> OdeResult {
    let p = QsBiofilmParams {
        k_dgc_basal: params.k_dgc_basal * 3.0,
        k_dgc_rep: params.k_dgc_rep * 0.3,
        ..params.clone()
    };
    run_scenario(&[0.01, 0.0, 0.0, 2.0, 0.5], 24.0, dt, &p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bio::ode::steady_state_mean;

    const DT: f64 = 0.001;
    const SS_FRAC: f64 = 0.1;

    #[test]
    fn standard_growth_biofilm_disperses() {
        let p = QsBiofilmParams::default();
        let r = scenario_standard_growth(&p, DT);
        let b_ss = steady_state_mean(&r, 4, SS_FRAC);
        assert!(
            b_ss < 0.05,
            "biofilm should disperse at high density, got {b_ss}"
        );
    }

    #[test]
    fn standard_growth_hapr_active() {
        let p = QsBiofilmParams::default();
        let r = scenario_standard_growth(&p, DT);
        let h_ss = steady_state_mean(&r, 2, SS_FRAC);
        assert!(
            h_ss > 0.5,
            "HapR should be active at high density, got {h_ss}"
        );
    }

    #[test]
    fn standard_growth_reaches_carrying_capacity() {
        let p = QsBiofilmParams::default();
        let r = scenario_standard_growth(&p, DT);
        let n_ss = steady_state_mean(&r, 0, SS_FRAC);
        assert!(
            (n_ss - p.k_cap).abs() < 0.05,
            "cell density should reach K={}, got {n_ss}",
            p.k_cap
        );
    }

    #[test]
    fn high_density_rapid_dispersal() {
        let p = QsBiofilmParams::default();
        let r = scenario_high_density(&p, DT);
        let b_ss = steady_state_mean(&r, 4, SS_FRAC);
        assert!(b_ss < 0.3, "high-density start should disperse, got {b_ss}");
    }

    #[test]
    fn hapr_mutant_constitutive_biofilm() {
        let p = QsBiofilmParams::default();
        let r = scenario_hapr_mutant(&p, DT);
        let b_ss = steady_state_mean(&r, 4, SS_FRAC);
        let c_ss = steady_state_mean(&r, 3, SS_FRAC);
        assert!(b_ss > 0.7, "ΔhapR should have high biofilm, got {b_ss}");
        assert!(c_ss > 1.5, "ΔhapR should have high c-di-GMP, got {c_ss}");
    }

    #[test]
    fn dgc_overexpression_elevated_cdg() {
        let p = QsBiofilmParams::default();
        let r = scenario_dgc_overexpression(&p, DT);
        let c_ss = steady_state_mean(&r, 3, SS_FRAC);
        assert!(c_ss > 0.5, "DGC OE should elevate c-di-GMP, got {c_ss}");
    }

    #[test]
    fn all_variables_non_negative() {
        let p = QsBiofilmParams::default();
        for result in [
            scenario_standard_growth(&p, DT),
            scenario_high_density(&p, DT),
            scenario_hapr_mutant(&p, DT),
            scenario_dgc_overexpression(&p, DT),
        ] {
            for (step, row) in result.y.iter().enumerate() {
                for (var, &val) in row.iter().enumerate() {
                    assert!(
                        val >= 0.0,
                        "variable {var} went negative ({val}) at step {step}"
                    );
                }
            }
        }
    }

    #[test]
    fn matches_python_steady_state() {
        let p = QsBiofilmParams::default();
        let tol = 1e-3;

        // Standard growth
        let r = scenario_standard_growth(&p, DT);
        let n_ss = steady_state_mean(&r, 0, SS_FRAC);
        let b_ss = steady_state_mean(&r, 4, SS_FRAC);
        assert!((n_ss - 0.975).abs() < tol, "N_ss={n_ss}");
        assert!(b_ss < 0.05, "B_ss={b_ss}");

        // ΔhapR
        let r = scenario_hapr_mutant(&p, DT);
        let c_ss = steady_state_mean(&r, 3, SS_FRAC);
        let b_ss = steady_state_mean(&r, 4, SS_FRAC);
        assert!((c_ss - 2.5).abs() < tol, "C_ss={c_ss}");
        assert!((b_ss - 0.786).abs() < 0.01, "B_ss={b_ss}");

        // DGC OE
        let r = scenario_dgc_overexpression(&p, DT);
        let c_ss = steady_state_mean(&r, 3, SS_FRAC);
        assert!((c_ss - 0.662).abs() < tol, "C_ss={c_ss}");
    }

    #[test]
    fn deterministic_across_runs() {
        let p = QsBiofilmParams::default();
        let r1 = scenario_standard_growth(&p, DT);
        let r2 = scenario_standard_growth(&p, DT);
        for (a, b) in r1.y_final.iter().zip(&r2.y_final) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "ODE should be bitwise deterministic"
            );
        }
    }
}
