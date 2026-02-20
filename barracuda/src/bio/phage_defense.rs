// SPDX-License-Identifier: AGPL-3.0-or-later
//! Phage defense deaminase model — Hsueh, Severin et al. 2022.
//!
//! Models the evolutionary arms race between bacteria with deoxycytidine
//! deaminase (`DCD`) defense and lytic phage. The defense imposes a fitness
//! cost but provides protection against phage infection.
//!
//! # References
//!
//! - Hsueh, Severin et al. 2022, *Nature Microbiology* 7:1210-1220
//!
//! # State variables
//!
//! | Index | Variable | Description |
//! |-------|----------|-------------|
//! | 0 | `Bd` | Defended bacteria (with `DCD`) |
//! | 1 | `Bu` | Undefended bacteria (no `DCD`) |
//! | 2 | P | Free phage |
//! | 3 | R | Resources (nutrients) |
//!
//! The defense (`DCD`) deaminates cytosine in phage DNA, reducing phage
//! burst size. Defended bacteria pay a growth cost but survive infection.

use super::ode::{rk4_integrate, OdeResult};

/// Parameters for the phage-bacteria defense model.
#[derive(Debug, Clone)]
pub struct PhageDefenseParams {
    /// Max growth rate of undefended bacteria.
    pub mu_max: f64,
    /// Growth cost of `DCD` defense (fraction of `mu_max`).
    pub defense_cost: f64,
    /// Half-saturation for resource-limited growth.
    pub k_resource: f64,
    /// Resource consumption per unit growth.
    pub yield_coeff: f64,
    /// Phage adsorption rate (encounters per phage per bacterium per hour).
    pub adsorption_rate: f64,
    /// Phage burst size from undefended bacteria.
    pub burst_size: f64,
    /// `DCD` defense efficiency (fraction of burst size reduction).
    pub defense_efficiency: f64,
    /// Phage decay rate.
    pub phage_decay: f64,
    /// Resource inflow rate.
    pub resource_inflow: f64,
    /// Resource dilution rate.
    pub resource_dilution: f64,
    /// Bacterial death/dilution rate.
    pub death_rate: f64,
}

impl Default for PhageDefenseParams {
    fn default() -> Self {
        Self {
            mu_max: 1.0,
            defense_cost: 0.15,
            k_resource: 0.5,
            yield_coeff: 0.5,
            adsorption_rate: 1e-7,
            burst_size: 50.0,
            defense_efficiency: 0.9,
            phage_decay: 0.1,
            resource_inflow: 10.0,
            resource_dilution: 0.1,
            death_rate: 0.05,
        }
    }
}

#[inline]
fn monod(r: f64, k: f64) -> f64 {
    r / (k + r)
}

/// Right-hand side of the phage-bacteria defense ODE.
#[allow(clippy::many_single_char_names)]
fn defense_rhs(state: &[f64], _t: f64, p: &PhageDefenseParams) -> Vec<f64> {
    let bd = state[0].max(0.0);
    let bu = state[1].max(0.0);
    let phage = state[2].max(0.0);
    let r = state[3].max(0.0);

    let growth_limit = monod(r, p.k_resource);

    // Defended bacteria: reduced growth, reduced phage kill
    let mu_d = p.mu_max * (1.0 - p.defense_cost) * growth_limit;
    let infection_d = p.adsorption_rate * bd * phage;
    let kill_d = infection_d * (1.0 - p.defense_efficiency);
    let growth_defended = mu_d * bd - kill_d - p.death_rate * bd;

    // Undefended bacteria: full growth, full phage kill
    let mu_u = p.mu_max * growth_limit;
    let infection_u = p.adsorption_rate * bu * phage;
    let growth_undefended = mu_u * bu - infection_u - p.death_rate * bu;

    // Phage: bursts from killed bacteria, decay
    let burst_from_u = p.burst_size * infection_u;
    let burst_from_d = p.burst_size * (1.0 - p.defense_efficiency) * infection_d;
    let d_phage =
        burst_from_u + burst_from_d - p.phage_decay * phage - p.adsorption_rate * (bd + bu) * phage;

    // Resources: inflow - consumption - dilution
    let consumption = p.yield_coeff * (mu_d * bd + mu_u * bu);
    let d_r = p.resource_inflow - consumption - p.resource_dilution * r;

    vec![growth_defended, growth_undefended, d_phage, d_r]
}

const CLAMP: [(f64, f64); 4] = [
    (0.0, f64::INFINITY),
    (0.0, f64::INFINITY),
    (0.0, f64::INFINITY),
    (0.0, f64::INFINITY),
];

/// Run the phage defense model.
#[must_use]
pub fn run_defense(y0: &[f64; 4], t_end: f64, dt: f64, params: &PhageDefenseParams) -> OdeResult {
    let p = params.clone();
    rk4_integrate(
        move |y, t| defense_rhs(y, t, &p),
        y0,
        0.0,
        t_end,
        dt,
        Some(&CLAMP),
    )
}

/// No phage: both bacterial populations grow, defended slower.
#[must_use]
pub fn scenario_no_phage(params: &PhageDefenseParams, dt: f64) -> OdeResult {
    run_defense(&[1e6, 1e6, 0.0, 10.0], 48.0, dt, params)
}

/// Phage attack: undefended bacteria crash, defended survive.
#[must_use]
pub fn scenario_phage_attack(params: &PhageDefenseParams, dt: f64) -> OdeResult {
    run_defense(&[1e6, 1e6, 1e4, 10.0], 48.0, dt, params)
}

/// Defense-only: no undefended competitors.
#[must_use]
pub fn scenario_pure_defended(params: &PhageDefenseParams, dt: f64) -> OdeResult {
    run_defense(&[1e6, 0.0, 1e4, 10.0], 48.0, dt, params)
}

/// No defense: all bacteria undefended.
#[must_use]
pub fn scenario_pure_undefended(params: &PhageDefenseParams, dt: f64) -> OdeResult {
    run_defense(&[0.0, 1e6, 1e4, 10.0], 48.0, dt, params)
}

/// High cost defense: `defense_cost` = 0.5.
#[must_use]
pub fn scenario_high_cost(params: &PhageDefenseParams, dt: f64) -> OdeResult {
    let p = PhageDefenseParams {
        defense_cost: 0.5,
        ..params.clone()
    };
    run_defense(&[1e6, 1e6, 1e4, 10.0], 48.0, dt, &p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bio::ode::steady_state_mean;

    const DT: f64 = 0.001;
    const SS_FRAC: f64 = 0.1;

    #[test]
    fn no_phage_undefended_wins() {
        let p = PhageDefenseParams::default();
        let r = scenario_no_phage(&p, DT);
        let bd = steady_state_mean(&r, 0, SS_FRAC);
        let bu = steady_state_mean(&r, 1, SS_FRAC);
        assert!(
            bu > bd,
            "without phage, undefended (no cost) should outcompete: Bd={bd:.0} Bu={bu:.0}"
        );
    }

    #[test]
    fn phage_attack_defended_survives() {
        let p = PhageDefenseParams::default();
        let r = scenario_phage_attack(&p, DT);
        let bd = steady_state_mean(&r, 0, SS_FRAC);
        let bu = steady_state_mean(&r, 1, SS_FRAC);
        assert!(
            bd > bu,
            "with phage, defended bacteria should dominate: Bd={bd:.0} Bu={bu:.0}"
        );
    }

    #[test]
    fn pure_undefended_crashes() {
        let p = PhageDefenseParams::default();
        let r = scenario_pure_undefended(&p, DT);
        let bu = steady_state_mean(&r, 1, SS_FRAC);
        let phage = steady_state_mean(&r, 2, SS_FRAC);
        assert!(bu < 1e5, "undefended should crash: Bu={bu:.0}");
        assert!(phage > 0.0, "phage should persist");
    }

    #[test]
    fn defense_efficiency_matters() {
        let p_strong = PhageDefenseParams::default();
        let p_weak = PhageDefenseParams {
            defense_efficiency: 0.3,
            ..PhageDefenseParams::default()
        };
        let r_strong = scenario_phage_attack(&p_strong, DT);
        let r_weak = scenario_phage_attack(&p_weak, DT);
        let bd_strong = steady_state_mean(&r_strong, 0, SS_FRAC);
        let bd_weak = steady_state_mean(&r_weak, 0, SS_FRAC);
        assert!(
            bd_strong > bd_weak,
            "stronger defense → more survivors: {bd_strong:.0} vs {bd_weak:.0}"
        );
    }

    #[test]
    fn all_non_negative() {
        let p = PhageDefenseParams::default();
        for result in [
            scenario_no_phage(&p, DT),
            scenario_phage_attack(&p, DT),
            scenario_pure_defended(&p, DT),
            scenario_pure_undefended(&p, DT),
            scenario_high_cost(&p, DT),
        ] {
            for (step, row) in result.y.iter().enumerate() {
                for (var, &val) in row.iter().enumerate() {
                    assert!(val >= 0.0, "var {var} negative ({val}) at step {step}");
                }
            }
        }
    }

    #[test]
    fn deterministic() {
        let p = PhageDefenseParams::default();
        let r1 = scenario_phage_attack(&p, DT);
        let r2 = scenario_phage_attack(&p, DT);
        for (a, b) in r1.y_final.iter().zip(&r2.y_final) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }
}
