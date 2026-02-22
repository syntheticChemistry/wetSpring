// SPDX-License-Identifier: AGPL-3.0-or-later
//! Multi-input quorum sensing regulatory network — Srivastava et al. 2011.
//!
//! Extends the Waters 2008 model with dual autoinducer signaling (CAI-1 and
//! AI-2) feeding through separate receptor pathways that converge on `HapR`.
//! Models how *V. cholerae* integrates multiple signal inputs to make a
//! binary virulence/biofilm decision.
//!
//! # References
//!
//! - Srivastava et al. 2011, *J Bacteriology* 193:6331-41
//!   "Integration of Cyclic di-GMP and Quorum Sensing in V. cholerae"
//! - Waters et al. 2008, *J Bacteriol* 190:2527-36
//!
//! # State variables
//!
//! | Index | Variable | Description |
//! |-------|----------|-------------|
//! | 0 | N | Cell density |
//! | 1 | A1 | CAI-1 (intraspecies signal, µM) |
//! | 2 | A2 | AI-2 (interspecies signal, µM) |
//! | 3 | P | `LuxO`~P (phosphorylated regulator, normalized) |
//! | 4 | H | `HapR` level |
//! | 5 | C | c-di-GMP (µM) |
//! | 6 | B | Biofilm state (0–1) |

use super::ode::{rk4_integrate, OdeResult};

/// Parameters for the dual-signal QS network.
#[derive(Debug, Clone)]
pub struct MultiSignalParams {
    /// Maximum growth rate (h⁻¹). Logistic term uses `1 - N/K`.
    pub mu_max: f64,
    /// Carrying capacity (OD-equivalent). Cell density caps at this value.
    pub k_cap: f64,
    /// Cell death rate (h⁻¹). Subtracted from growth.
    pub death_rate: f64,
    /// CAI-1 production rate per cell. Intraspecies signal in *V. cholerae*.
    pub k_cai1_prod: f64,
    /// CAI-1 degradation/dilution rate.
    pub d_cai1: f64,
    /// Half-sat for CAI-1 dephosphorylation of `CqsS`.
    pub k_cqs: f64,
    /// AI-2 production rate per cell. Interspecies signal.
    pub k_ai2_prod: f64,
    /// AI-2 degradation/dilution rate.
    pub d_ai2: f64,
    /// Half-sat for AI-2 dephosphorylation of `LuxPQ`.
    pub k_luxpq: f64,
    /// `LuxO` kinase rate. Phosphorylated `LuxO` represses `HapR`.
    pub k_luxo_phos: f64,
    /// LuxO~P dephosphorylation rate. Signal-bound CqsS/LuxPQ enhance this.
    pub d_luxo_p: f64,
    /// Max `HapR` production rate. Actual rate scaled by LuxO~P repression.
    pub k_hapr_max: f64,
    /// Hill coefficient for `LuxO`~P repression of `HapR`.
    pub n_repress: f64,
    /// Half-sat for `LuxO`~P repression.
    pub k_repress: f64,
    /// `HapR` degradation rate. Master regulator of virulence and biofilm.
    pub d_hapr: f64,
    /// Basal diguanylate cyclase (DGC) activity. Produces c-di-GMP.
    pub k_dgc_basal: f64,
    /// DGC repression factor by `HapR`. Higher `HapR` → less c-di-GMP.
    pub k_dgc_rep: f64,
    /// Basal phosphodiesterase (PDE) activity. Degrades c-di-GMP.
    pub k_pde_basal: f64,
    /// PDE activation by `HapR`. High QS → high PDE → low c-di-GMP → dispersal.
    pub k_pde_act: f64,
    /// c-di-GMP turnover/dilution rate.
    pub d_cdg: f64,
    /// Maximum biofilm promotion rate. Hill-saturated by c-di-GMP.
    pub k_bio_max: f64,
    /// Half-saturation for c-di-GMP activation of biofilm.
    pub k_bio_cdg: f64,
    /// Hill coefficient for biofilm promotion by c-di-GMP.
    pub n_bio: f64,
    /// Biofilm loss (dispersal) rate.
    pub d_bio: f64,
}

impl Default for MultiSignalParams {
    fn default() -> Self {
        Self {
            mu_max: 0.8,
            k_cap: 1.0,
            death_rate: 0.02,
            k_cai1_prod: 3.0,
            d_cai1: 1.0,
            k_cqs: 0.5,
            k_ai2_prod: 3.0,
            d_ai2: 1.0,
            k_luxpq: 0.5,
            k_luxo_phos: 2.0,
            d_luxo_p: 0.5,
            k_hapr_max: 1.0,
            n_repress: 2.0,
            k_repress: 0.5,
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

/// Number of state variables in the multi-signal ODE system.
pub const N_VARS: usize = 7;
/// Number of f64 parameters when flattened for GPU dispatch.
pub const N_PARAMS: usize = 24;

impl MultiSignalParams {
    /// Flatten parameters into a contiguous `f64` slice for GPU dispatch.
    ///
    /// Layout matches the field order of [`MultiSignalParams`]. Used by the
    /// batched ODE RK4 GPU shader (parameter buffer binding).
    #[must_use]
    pub const fn to_flat(&self) -> [f64; N_PARAMS] {
        [
            self.mu_max,
            self.k_cap,
            self.death_rate,
            self.k_cai1_prod,
            self.d_cai1,
            self.k_cqs,
            self.k_ai2_prod,
            self.d_ai2,
            self.k_luxpq,
            self.k_luxo_phos,
            self.d_luxo_p,
            self.k_hapr_max,
            self.n_repress,
            self.k_repress,
            self.d_hapr,
            self.k_dgc_basal,
            self.k_dgc_rep,
            self.k_pde_basal,
            self.k_pde_act,
            self.d_cdg,
            self.k_bio_max,
            self.k_bio_cdg,
            self.n_bio,
            self.d_bio,
        ]
    }

    /// Reconstruct from a flat `f64` slice (inverse of [`to_flat`](Self::to_flat)).
    ///
    /// # Panics
    ///
    /// Panics if `flat.len() < N_PARAMS`.
    #[must_use]
    pub fn from_flat(flat: &[f64]) -> Self {
        assert!(flat.len() >= N_PARAMS, "need {N_PARAMS} values");
        Self {
            mu_max: flat[0],
            k_cap: flat[1],
            death_rate: flat[2],
            k_cai1_prod: flat[3],
            d_cai1: flat[4],
            k_cqs: flat[5],
            k_ai2_prod: flat[6],
            d_ai2: flat[7],
            k_luxpq: flat[8],
            k_luxo_phos: flat[9],
            d_luxo_p: flat[10],
            k_hapr_max: flat[11],
            n_repress: flat[12],
            k_repress: flat[13],
            d_hapr: flat[14],
            k_dgc_basal: flat[15],
            k_dgc_rep: flat[16],
            k_pde_basal: flat[17],
            k_pde_act: flat[18],
            d_cdg: flat[19],
            k_bio_max: flat[20],
            k_bio_cdg: flat[21],
            n_bio: flat[22],
            d_bio: flat[23],
        }
    }
}

#[inline]
fn hill(x: f64, k: f64, n: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let xn = x.powf(n);
    xn / (k.powf(n) + xn)
}

/// Hill repression: k^n / (k^n + x^n).
#[inline]
fn hill_repress(x: f64, k: f64, n: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    let kn = k.powf(n);
    kn / (kn + x.powf(n))
}

/// Right-hand side for the dual-signal QS ODE system.
#[allow(clippy::many_single_char_names)]
fn multi_rhs(state: &[f64], _t: f64, p: &MultiSignalParams) -> Vec<f64> {
    let cell = state[0].max(0.0);
    let cai1 = state[1].max(0.0);
    let ai2 = state[2].max(0.0);
    let luxo_p = state[3].max(0.0);
    let hapr = state[4].max(0.0);
    let cdg = state[5].max(0.0);
    let bio = state[6].max(0.0);

    // Growth
    let d_cell = (p.mu_max * cell).mul_add(1.0 - cell / p.k_cap, -(p.death_rate * cell));

    // Dual autoinducer production
    let d_cai1 = p.k_cai1_prod.mul_add(cell, -p.d_cai1 * cai1);
    let d_ai2 = p.k_ai2_prod.mul_add(cell, -p.d_ai2 * ai2);

    // LuxO phosphorylation: dephosphorylated by BOTH signals
    // Each signal independently contributes to dephosphorylation
    let dephos_cai1 = hill(cai1, p.k_cqs, 2.0);
    let dephos_ai2 = hill(ai2, p.k_luxpq, 2.0);
    let total_dephos = dephos_cai1 + dephos_ai2;
    let d_luxo_p = (p.d_luxo_p + total_dephos).mul_add(-luxo_p, p.k_luxo_phos);

    // HapR: repressed by LuxO~P
    let d_hapr = p.k_hapr_max.mul_add(
        hill_repress(luxo_p, p.k_repress, p.n_repress),
        -p.d_hapr * hapr,
    );

    // c-di-GMP (same as Waters 2008)
    let dgc_rate = p.k_dgc_basal * p.k_dgc_rep.mul_add(-hapr, 1.0).max(0.0);
    let pde_rate = p.k_pde_act.mul_add(hapr, p.k_pde_basal);
    let mut d_cdg = p.d_cdg.mul_add(-cdg, dgc_rate - pde_rate * cdg);
    if cdg < crate::tolerances::ODE_CDG_CONVERGENCE && d_cdg < 0.0 {
        d_cdg = 0.0;
    }

    // Biofilm
    let bio_promote = p.k_bio_max * hill(cdg, p.k_bio_cdg, p.n_bio);
    let d_bio = bio_promote.mul_add(1.0 - bio, -(p.d_bio * bio));

    vec![d_cell, d_cai1, d_ai2, d_luxo_p, d_hapr, d_cdg, d_bio]
}

const CLAMP: [(f64, f64); 7] = [
    (0.0, f64::INFINITY),
    (0.0, f64::INFINITY),
    (0.0, f64::INFINITY),
    (0.0, f64::INFINITY),
    (0.0, f64::INFINITY),
    (0.0, f64::INFINITY),
    (0.0, 1.0),
];

/// Run the multi-signal model.
#[must_use]
pub fn run_multi_signal(
    y0: &[f64; 7],
    t_end: f64,
    dt: f64,
    params: &MultiSignalParams,
) -> OdeResult {
    rk4_integrate(
        |y, t| multi_rhs(y, t, params),
        y0,
        0.0,
        t_end,
        dt,
        Some(&CLAMP),
    )
}

/// Wild-type: both signals present at high density.
#[must_use]
pub fn scenario_wild_type(params: &MultiSignalParams, dt: f64) -> OdeResult {
    run_multi_signal(&[0.01, 0.0, 0.0, 2.0, 0.0, 2.0, 0.5], 24.0, dt, params)
}

/// CAI-1 only: AI-2 production knocked out (`ΔluxS`).
#[must_use]
pub fn scenario_cai1_only(params: &MultiSignalParams, dt: f64) -> OdeResult {
    let p = MultiSignalParams {
        k_ai2_prod: 0.0,
        ..params.clone()
    };
    run_multi_signal(&[0.01, 0.0, 0.0, 2.0, 0.0, 2.0, 0.5], 24.0, dt, &p)
}

/// AI-2 only: CAI-1 production knocked out (`ΔcqsA`).
#[must_use]
pub fn scenario_ai2_only(params: &MultiSignalParams, dt: f64) -> OdeResult {
    let p = MultiSignalParams {
        k_cai1_prod: 0.0,
        ..params.clone()
    };
    run_multi_signal(&[0.01, 0.0, 0.0, 2.0, 0.0, 2.0, 0.5], 24.0, dt, &p)
}

/// No QS: both signals knocked out (`ΔluxS ΔcqsA`).
#[must_use]
pub fn scenario_no_qs(params: &MultiSignalParams, dt: f64) -> OdeResult {
    let p = MultiSignalParams {
        k_cai1_prod: 0.0,
        k_ai2_prod: 0.0,
        ..params.clone()
    };
    run_multi_signal(&[0.01, 0.0, 0.0, 2.0, 0.0, 2.0, 0.5], 24.0, dt, &p)
}

/// Exogenous signal: add external CAI-1 to low-density culture.
#[must_use]
pub fn scenario_exogenous_cai1(params: &MultiSignalParams, dt: f64) -> OdeResult {
    run_multi_signal(&[0.01, 5.0, 0.0, 2.0, 0.0, 2.0, 0.5], 24.0, dt, params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bio::ode::steady_state_mean;

    const DT: f64 = 0.001;
    const SS_FRAC: f64 = 0.1;

    #[test]
    fn wild_type_reaches_high_density() {
        let p = MultiSignalParams::default();
        let r = scenario_wild_type(&p, DT);
        let n_ss = steady_state_mean(&r, 0, SS_FRAC);
        assert!(
            (n_ss - p.k_cap).abs() < 0.05,
            "wild type should reach K: N_ss={n_ss}"
        );
    }

    #[test]
    fn wild_type_activates_hapr() {
        let p = MultiSignalParams::default();
        let r = scenario_wild_type(&p, DT);
        let h_ss = steady_state_mean(&r, 4, SS_FRAC);
        assert!(
            h_ss > 0.3,
            "wild type at high density should activate HapR: H={h_ss}"
        );
    }

    #[test]
    fn wild_type_moderate_biofilm() {
        let p = MultiSignalParams::default();
        let r = scenario_wild_type(&p, DT);
        let b_ss = steady_state_mean(&r, 6, SS_FRAC);
        assert!(
            (b_ss - 0.413).abs() < 0.01,
            "wild type should match Python B_ss: {b_ss}"
        );
    }

    #[test]
    fn no_qs_maintains_biofilm() {
        let p = MultiSignalParams::default();
        let r = scenario_no_qs(&p, DT);
        let b_ss = steady_state_mean(&r, 6, SS_FRAC);
        let h_ss = steady_state_mean(&r, 4, SS_FRAC);
        assert!(b_ss > 0.3, "no QS should maintain biofilm: B={b_ss}");
        assert!(h_ss < 0.3, "no QS should have low HapR: H={h_ss}");
    }

    #[test]
    fn single_signal_partial_response() {
        let p = MultiSignalParams::default();
        let r_wt = scenario_wild_type(&p, DT);
        let r_cai1 = scenario_cai1_only(&p, DT);
        let r_ai2 = scenario_ai2_only(&p, DT);

        let h_wt = steady_state_mean(&r_wt, 4, SS_FRAC);
        let h_cai1 = steady_state_mean(&r_cai1, 4, SS_FRAC);
        let h_ai2 = steady_state_mean(&r_ai2, 4, SS_FRAC);

        assert!(
            h_cai1 < h_wt,
            "single signal should give less HapR than dual: cai1={h_cai1} vs wt={h_wt}"
        );
        assert!(
            h_ai2 < h_wt,
            "single signal should give less HapR than dual: ai2={h_ai2} vs wt={h_wt}"
        );
    }

    #[test]
    fn exogenous_cai1_accelerates_hapr() {
        let p = MultiSignalParams::default();
        let r_exo = scenario_exogenous_cai1(&p, DT);
        let h_ss = steady_state_mean(&r_exo, 4, SS_FRAC);
        assert!(
            h_ss > 0.3,
            "exogenous CAI-1 should activate HapR even at low density: H={h_ss}"
        );
    }

    #[test]
    fn all_variables_non_negative() {
        let p = MultiSignalParams::default();
        for result in [
            scenario_wild_type(&p, DT),
            scenario_cai1_only(&p, DT),
            scenario_ai2_only(&p, DT),
            scenario_no_qs(&p, DT),
            scenario_exogenous_cai1(&p, DT),
        ] {
            for (step, row) in result.states().enumerate() {
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
    fn deterministic_across_runs() {
        let p = MultiSignalParams::default();
        let r1 = scenario_wild_type(&p, DT);
        let r2 = scenario_wild_type(&p, DT);
        for (a, b) in r1.y_final.iter().zip(&r2.y_final) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "ODE should be bitwise deterministic"
            );
        }
    }

    #[test]
    fn flat_params_round_trip() {
        let p = MultiSignalParams::default();
        let flat = p.to_flat();
        assert_eq!(flat.len(), N_PARAMS);
        let p2 = MultiSignalParams::from_flat(&flat);
        let flat2 = p2.to_flat();
        for (a, b) in flat.iter().zip(&flat2) {
            assert_eq!(a.to_bits(), b.to_bits(), "round-trip must be bitwise exact");
        }
    }

    #[test]
    fn flat_params_gpu_parity() {
        let p = MultiSignalParams::default();
        let flat = p.to_flat();
        let p2 = MultiSignalParams::from_flat(&flat);
        let r1 = scenario_wild_type(&p, DT);
        let r2 = scenario_wild_type(&p2, DT);
        for (a, b) in r1.y_final.iter().zip(&r2.y_final) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "flat round-trip must produce identical ODE results"
            );
        }
    }
}
