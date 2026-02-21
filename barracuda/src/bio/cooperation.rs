// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cooperative quorum sensing game theory — Bruger & Waters 2018.
//!
//! Models the evolutionary dynamics of cooperators (QS signal producers) vs
//! cheaters (non-producers who exploit the public good) in *V. cholerae*.
//! Uses a two-population ODE model with frequency-dependent fitness.
//!
//! # References
//!
//! - Bruger & Waters 2018, *AEM* 84:e00402-18
//!   "Maximizing Growth Yield and Dispersal via QS Promotes Cooperation in V. cholerae"
//!
//! # State variables
//!
//! | Index | Variable | Description |
//! |-------|----------|-------------|
//! | 0 | `Nc` | Cooperator density |
//! | 1 | `Nd` | Cheater (defector) density |
//! | 2 | A | Autoinducer (public good, produced by cooperators) |
//! | 3 | B | Biofilm state (shared, depends on total QS) |

use super::ode::{rk4_integrate, OdeResult};

/// Parameters for the cooperation game.
#[derive(Debug, Clone)]
pub struct CooperationParams {
    /// Max growth rate for cooperators.
    pub mu_coop: f64,
    /// Max growth rate for cheaters (slightly higher — no cost of production).
    pub mu_cheat: f64,
    /// Carrying capacity (total `N_c + N_d`).
    pub k_cap: f64,
    /// Per-capita death rate.
    pub death_rate: f64,
    /// AI production rate per cooperator cell.
    pub k_ai_prod: f64,
    /// AI degradation rate.
    pub d_ai: f64,
    /// Signal benefit: growth enhancement from AI sensing.
    pub benefit: f64,
    /// Half-sat for benefit from AI.
    pub k_benefit: f64,
    /// Cost of signal production (reduces cooperator growth).
    pub cost: f64,
    /// Biofilm formation rate (from signal).
    pub k_bio: f64,
    /// `Half-sat` for biofilm from AI.
    pub k_bio_ai: f64,
    /// Biofilm dispersal advantage: cells leaving biofilm get a growth burst.
    pub dispersal_bonus: f64,
    /// Biofilm degradation rate.
    pub d_bio: f64,
}

impl Default for CooperationParams {
    fn default() -> Self {
        Self {
            mu_coop: 0.7,
            mu_cheat: 0.75,
            k_cap: 1.0,
            death_rate: 0.02,
            k_ai_prod: 5.0,
            d_ai: 1.0,
            benefit: 0.3,
            k_benefit: 0.5,
            cost: 0.05,
            k_bio: 1.0,
            k_bio_ai: 0.5,
            dispersal_bonus: 0.2,
            d_bio: 0.3,
        }
    }
}

#[inline]
fn hill(x: f64, k: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let x2 = x * x;
    x2 / k.mul_add(k, x2)
}

/// Right-hand side of the cooperation game ODE.
#[allow(clippy::many_single_char_names)]
fn coop_rhs(state: &[f64], _t: f64, p: &CooperationParams) -> Vec<f64> {
    let nc = state[0].max(0.0);
    let nd = state[1].max(0.0);
    let ai = state[2].max(0.0);
    let bio = state[3].max(0.0);

    let n_total = nc + nd;
    let crowding = (1.0 - n_total / p.k_cap).max(0.0);

    let signal_benefit = p.benefit * hill(ai, p.k_benefit);
    let dispersal = p.dispersal_bonus * (1.0 - bio);

    let fitness_coop = (p.mu_coop - p.cost + signal_benefit + dispersal) * crowding;
    let fitness_cheat = (p.mu_cheat + signal_benefit + dispersal) * crowding;

    let growth_coop = fitness_coop.mul_add(nc, -(p.death_rate * nc));
    let growth_cheat = fitness_cheat.mul_add(nd, -(p.death_rate * nd));
    let d_ai = p.k_ai_prod.mul_add(nc, -p.d_ai * ai);
    let d_bio = (p.k_bio * hill(ai, p.k_bio_ai)).mul_add(1.0 - bio, -(p.d_bio * bio));

    vec![growth_coop, growth_cheat, d_ai, d_bio]
}

const CLAMP: [(f64, f64); 4] = [
    (0.0, f64::INFINITY),
    (0.0, f64::INFINITY),
    (0.0, f64::INFINITY),
    (0.0, 1.0),
];

/// Run the cooperation game model.
#[must_use]
pub fn run_cooperation(
    y0: &[f64; 4],
    t_end: f64,
    dt: f64,
    params: &CooperationParams,
) -> OdeResult {
    let p = params.clone();
    rk4_integrate(
        move |y, t| coop_rhs(y, t, &p),
        y0,
        0.0,
        t_end,
        dt,
        Some(&CLAMP),
    )
}

/// Equal start: 50/50 cooperators and cheaters.
#[must_use]
pub fn scenario_equal_start(params: &CooperationParams, dt: f64) -> OdeResult {
    run_cooperation(&[0.01, 0.01, 0.0, 0.0], 48.0, dt, params)
}

/// Cooperator-dominated start (90/10).
#[must_use]
pub fn scenario_coop_dominated(params: &CooperationParams, dt: f64) -> OdeResult {
    run_cooperation(&[0.09, 0.01, 0.0, 0.0], 48.0, dt, params)
}

/// Cheater-dominated start (10/90).
#[must_use]
pub fn scenario_cheat_dominated(params: &CooperationParams, dt: f64) -> OdeResult {
    run_cooperation(&[0.01, 0.09, 0.0, 0.0], 48.0, dt, params)
}

/// Pure cooperators (no cheaters).
#[must_use]
pub fn scenario_pure_coop(params: &CooperationParams, dt: f64) -> OdeResult {
    run_cooperation(&[0.01, 0.0, 0.0, 0.0], 48.0, dt, params)
}

/// Pure cheaters (no cooperators — no signal produced).
#[must_use]
pub fn scenario_pure_cheat(params: &CooperationParams, dt: f64) -> OdeResult {
    run_cooperation(&[0.0, 0.01, 0.0, 0.0], 48.0, dt, params)
}

/// Compute cooperator frequency at each time step.
#[must_use]
pub fn cooperator_frequency(result: &OdeResult) -> Vec<f64> {
    result
        .y
        .iter()
        .map(|row| {
            let nc = row[0].max(0.0);
            let nd = row[1].max(0.0);
            let total = nc + nd;
            if total < 1e-15 {
                0.5
            } else {
                nc / total
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bio::ode::steady_state_mean;

    const DT: f64 = 0.001;
    const SS_FRAC: f64 = 0.1;

    #[test]
    #[allow(clippy::similar_names)]
    fn equal_start_coexistence() {
        let p = CooperationParams::default();
        let r = scenario_equal_start(&p, DT);
        let nc_ss = steady_state_mean(&r, 0, SS_FRAC);
        let nd_ss = steady_state_mean(&r, 1, SS_FRAC);
        assert!(nc_ss > 0.01, "cooperators should survive: Nc={nc_ss}");
        assert!(nd_ss > 0.01, "cheaters should survive: Nd={nd_ss}");
    }

    #[test]
    fn pure_coop_produces_signal() {
        let p = CooperationParams::default();
        let r = scenario_pure_coop(&p, DT);
        let ai_ss = steady_state_mean(&r, 2, SS_FRAC);
        assert!(
            ai_ss > 1.0,
            "pure cooperators should produce signal: AI={ai_ss}"
        );
    }

    #[test]
    fn pure_cheat_no_signal() {
        let p = CooperationParams::default();
        let r = scenario_pure_cheat(&p, DT);
        let ai_ss = steady_state_mean(&r, 2, SS_FRAC);
        assert!(
            ai_ss < 0.01,
            "pure cheaters should have no signal: AI={ai_ss}"
        );
    }

    #[test]
    fn cheaters_have_frequency_advantage() {
        let p = CooperationParams::default();
        let r = scenario_equal_start(&p, DT);
        let freq = cooperator_frequency(&r);
        let final_freq = freq.last().copied().unwrap_or(0.5);
        assert!(
            final_freq < 0.5,
            "cheaters should have frequency advantage from cost: f_coop={final_freq}"
        );
    }

    #[test]
    #[allow(clippy::similar_names)]
    fn total_density_reaches_capacity() {
        let p = CooperationParams::default();
        let r = scenario_equal_start(&p, DT);
        let nc_ss = steady_state_mean(&r, 0, SS_FRAC);
        let nd_ss = steady_state_mean(&r, 1, SS_FRAC);
        let total = nc_ss + nd_ss;
        assert!(
            (total - p.k_cap).abs() < 0.1,
            "total density should approach K: {total}"
        );
    }

    #[test]
    fn all_variables_non_negative() {
        let p = CooperationParams::default();
        for result in [
            scenario_equal_start(&p, DT),
            scenario_coop_dominated(&p, DT),
            scenario_cheat_dominated(&p, DT),
            scenario_pure_coop(&p, DT),
            scenario_pure_cheat(&p, DT),
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
    fn deterministic_across_runs() {
        let p = CooperationParams::default();
        let r1 = scenario_equal_start(&p, DT);
        let r2 = scenario_equal_start(&p, DT);
        for (a, b) in r1.y_final.iter().zip(&r2.y_final) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "ODE should be bitwise deterministic"
            );
        }
    }
}
