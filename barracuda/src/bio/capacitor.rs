// SPDX-License-Identifier: AGPL-3.0-or-later
//! Phenotypic capacitor model — Mhatre et al. 2020.
//!
//! Models how a single regulatory node (`VpsR` in *V. cholerae*) acts as a
//! capacitor that stores c-di-GMP signal and releases it as different phenotypes
//! depending on environmental context (nutrient, stress).
//!
//! # References
//!
//! - Mhatre et al. 2020, *PNAS* 117:21647-21657
//!
//! # State variables
//!
//! | Index | Variable | Description |
//! |-------|----------|-------------|
//! | 0 | N | Cell density |
//! | 1 | C | c-di-GMP (µM) |
//! | 2 | V | `VpsR` activity (capacitor, 0–1) |
//! | 3 | B | Biofilm phenotype (0–1) |
//! | 4 | M | Motility phenotype (0–1) |
//! | 5 | R | Rugose colony phenotype (0–1) |
//!
//! The capacitor `VpsR` integrates c-di-GMP over time and distributes
//! its output to three phenotypic channels depending on parameter context.

use super::ode::{OdeResult, rk4_integrate};

/// Parameters for the phenotypic capacitor model.
#[derive(Debug, Clone)]
pub struct CapacitorParams {
    /// Maximum growth rate.
    pub mu_max: f64,
    /// Carrying capacity (cell density).
    pub k_cap: f64,
    /// Per-capita death rate.
    pub death_rate: f64,
    /// c-di-GMP production rate.
    pub k_cdg_prod: f64,
    /// c-di-GMP degradation rate.
    pub d_cdg: f64,
    /// `VpsR` activation by c-di-GMP (capacitor charging).
    pub k_vpsr_charge: f64,
    /// `VpsR` deactivation (capacitor discharge).
    pub k_vpsr_discharge: f64,
    /// Hill coefficient for `VpsR` activation.
    pub n_vpsr: f64,
    /// `Half-sat` for `VpsR` activation by c-di-GMP.
    pub k_vpsr_cdg: f64,
    /// Biofilm output weight from `VpsR`.
    pub w_biofilm: f64,
    /// Motility output weight from `VpsR` (inversely related).
    pub w_motility: f64,
    /// Rugose colony output weight from `VpsR`.
    pub w_rugose: f64,
    /// Biofilm phenotype decay rate.
    pub d_bio: f64,
    /// Motility phenotype decay rate.
    pub d_mot: f64,
    /// Rugose phenotype decay rate.
    pub d_rug: f64,
    /// Environmental modifier: nutrient stress increases c-di-GMP.
    pub stress_factor: f64,
}

/// Number of state variables in the capacitor ODE system.
pub const N_VARS: usize = 6;
/// Number of f64 parameters when flattened for GPU dispatch.
pub const N_PARAMS: usize = 16;

impl CapacitorParams {
    /// Flatten parameters into a contiguous `f64` slice for GPU dispatch.
    #[must_use]
    pub const fn to_flat(&self) -> [f64; N_PARAMS] {
        [
            self.mu_max,
            self.k_cap,
            self.death_rate,
            self.k_cdg_prod,
            self.d_cdg,
            self.k_vpsr_charge,
            self.k_vpsr_discharge,
            self.n_vpsr,
            self.k_vpsr_cdg,
            self.w_biofilm,
            self.w_motility,
            self.w_rugose,
            self.d_bio,
            self.d_mot,
            self.d_rug,
            self.stress_factor,
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
            k_cdg_prod: flat[3],
            d_cdg: flat[4],
            k_vpsr_charge: flat[5],
            k_vpsr_discharge: flat[6],
            n_vpsr: flat[7],
            k_vpsr_cdg: flat[8],
            w_biofilm: flat[9],
            w_motility: flat[10],
            w_rugose: flat[11],
            d_bio: flat[12],
            d_mot: flat[13],
            d_rug: flat[14],
            stress_factor: flat[15],
        }
    }
}

impl Default for CapacitorParams {
    fn default() -> Self {
        Self {
            mu_max: 0.8,
            k_cap: 1.0,
            death_rate: 0.02,
            k_cdg_prod: 2.0,
            d_cdg: 0.5,
            k_vpsr_charge: 1.0,
            k_vpsr_discharge: 0.3,
            n_vpsr: 3.0,
            k_vpsr_cdg: 1.0,
            w_biofilm: 0.8,
            w_motility: 0.6,
            w_rugose: 0.4,
            d_bio: 0.3,
            d_mot: 0.3,
            d_rug: 0.3,
            stress_factor: 1.0,
        }
    }
}

use barracuda::numerical::{CapacitorOde, OdeSystem as _};

const CLAMP: [(f64, f64); 6] = [
    (0.0, f64::INFINITY),
    (0.0, f64::INFINITY),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
];

/// Run the capacitor model.
#[must_use]
pub fn run_capacitor(y0: &[f64; 6], t_end: f64, dt: f64, params: &CapacitorParams) -> OdeResult {
    let flat = params.to_flat();
    rk4_integrate(
        |y, t| CapacitorOde::cpu_derivative(t, y, &flat),
        y0,
        0.0,
        t_end,
        dt,
        Some(&CLAMP),
    )
}

/// Normal growth: moderate c-di-GMP → balanced phenotype mix.
#[must_use]
pub fn scenario_normal(params: &CapacitorParams, dt: f64) -> OdeResult {
    run_capacitor(&[0.01, 1.0, 0.0, 0.0, 0.5, 0.0], 48.0, dt, params)
}

/// Nutrient stress: elevated c-di-GMP → `VpsR` saturated → biofilm + rugose.
#[must_use]
pub fn scenario_stress(params: &CapacitorParams, dt: f64) -> OdeResult {
    let p = CapacitorParams {
        stress_factor: 3.0,
        ..params.clone()
    };
    run_capacitor(&[0.01, 1.0, 0.0, 0.0, 0.5, 0.0], 48.0, dt, &p)
}

/// Low c-di-GMP: `VpsR` off → motility favored.
#[must_use]
pub fn scenario_low_cdg(params: &CapacitorParams, dt: f64) -> OdeResult {
    let p = CapacitorParams {
        k_cdg_prod: 0.3,
        ..params.clone()
    };
    run_capacitor(&[0.01, 0.1, 0.0, 0.0, 0.5, 0.0], 48.0, dt, &p)
}

/// `VpsR` knockout (`ΔvpsR`): no phenotypic output from capacitor.
#[must_use]
pub fn scenario_vpsr_knockout(params: &CapacitorParams, dt: f64) -> OdeResult {
    let p = CapacitorParams {
        k_vpsr_charge: 0.0,
        ..params.clone()
    };
    run_capacitor(&[0.01, 1.0, 0.0, 0.0, 0.5, 0.0], 48.0, dt, &p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bio::ode::steady_state_mean;

    const DT: f64 = crate::tolerances::ODE_DEFAULT_DT;
    const SS_FRAC: f64 = 0.1;

    #[test]
    fn normal_growth_mixed_phenotype() {
        let p = CapacitorParams::default();
        let r = scenario_normal(&p, DT);
        let bio = steady_state_mean(&r, 3, SS_FRAC);
        let mot = steady_state_mean(&r, 4, SS_FRAC);
        assert!(bio > 0.1, "normal: should have some biofilm: B={bio}");
        assert!(mot > 0.1, "normal: should have some motility: M={mot}");
    }

    #[test]
    fn stress_favors_biofilm_and_rugose() {
        let p = CapacitorParams::default();
        let r_stress = scenario_stress(&p, DT);
        let r_normal = scenario_normal(&p, DT);
        let bio_stress = steady_state_mean(&r_stress, 3, SS_FRAC);
        let bio_normal = steady_state_mean(&r_normal, 3, SS_FRAC);
        let rug_stress = steady_state_mean(&r_stress, 5, SS_FRAC);
        assert!(
            bio_stress > bio_normal,
            "stress should increase biofilm: {bio_stress} vs {bio_normal}"
        );
        assert!(
            rug_stress > 0.1,
            "stress should activate rugose: R={rug_stress}"
        );
    }

    #[test]
    fn low_cdg_favors_motility() {
        let p = CapacitorParams::default();
        let r = scenario_low_cdg(&p, DT);
        let mot = steady_state_mean(&r, 4, SS_FRAC);
        let bio = steady_state_mean(&r, 3, SS_FRAC);
        assert!(
            mot > bio,
            "low c-di-GMP should favor motility over biofilm: M={mot} vs B={bio}"
        );
    }

    #[test]
    fn vpsr_knockout_no_biofilm() {
        let p = CapacitorParams::default();
        let r = scenario_vpsr_knockout(&p, DT);
        let bio = steady_state_mean(&r, 3, SS_FRAC);
        let rug = steady_state_mean(&r, 5, SS_FRAC);
        assert!(bio < 0.05, "ΔvpsR should have very low biofilm: B={bio}");
        assert!(rug < 0.05, "ΔvpsR should have very low rugose: R={rug}");
    }

    #[test]
    fn all_variables_non_negative() {
        let p = CapacitorParams::default();
        for result in [
            scenario_normal(&p, DT),
            scenario_stress(&p, DT),
            scenario_low_cdg(&p, DT),
            scenario_vpsr_knockout(&p, DT),
        ] {
            for (step, row) in result.states().enumerate() {
                for (var, &val) in row.iter().enumerate() {
                    assert!(val >= 0.0, "var {var} negative ({val}) at step {step}");
                }
            }
        }
    }

    #[test]
    fn deterministic() {
        let p = CapacitorParams::default();
        let r1 = scenario_normal(&p, DT);
        let r2 = scenario_normal(&p, DT);
        for (a, b) in r1.y_final.iter().zip(&r2.y_final) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn flat_params_round_trip() {
        let p = CapacitorParams::default();
        let flat = p.to_flat();
        assert_eq!(flat.len(), N_PARAMS);
        let p2 = CapacitorParams::from_flat(&flat);
        let flat2 = p2.to_flat();
        for (a, b) in flat.iter().zip(&flat2) {
            assert_eq!(a.to_bits(), b.to_bits(), "round-trip must be bitwise exact");
        }
    }

    #[test]
    fn flat_params_gpu_parity() {
        let p = CapacitorParams::default();
        let flat = p.to_flat();
        let p2 = CapacitorParams::from_flat(&flat);
        let r1 = scenario_normal(&p, DT);
        let r2 = scenario_normal(&p2, DT);
        for (a, b) in r1.y_final.iter().zip(&r2.y_final) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "flat round-trip must produce identical ODE results"
            );
        }
    }
}
