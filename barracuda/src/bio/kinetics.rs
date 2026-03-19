// SPDX-License-Identifier: AGPL-3.0-or-later
//! Microbial growth kinetics: Monod, Haldane, and related rate laws.
//!
//! Centralizes the substrate-dependent growth rate formulas used across
//! 12+ validation binaries. These are candidates for upstream absorption
//! into `barracuda::stats::monod` / `barracuda::stats::haldane` once
//! barraCuda exposes them.
//!
//! # Formulas
//!
//! **Monod** (Monod 1949): `μ = μ_max × S / (K_s + S)`
//! **Haldane** (Andrews 1968): `μ = μ_max × S / (K_s + S + S²/K_i)`

/// Monod growth kinetics (Monod 1949).
///
/// Models substrate-limited microbial growth with `Michaelis-Menten`
/// saturation.
///
/// - `s`: substrate concentration
/// - `mu_max`: maximum specific growth rate
/// - `ks`: half-saturation constant (substrate concentration at `μ = μ_max/2`)
///
/// Returns the specific growth rate μ.
#[must_use]
pub fn monod(s: f64, mu_max: f64, ks: f64) -> f64 {
    mu_max * s / (ks + s)
}

/// Haldane substrate inhibition kinetics (Andrews 1968).
///
/// Extends Monod with substrate inhibition at high concentrations.
///
/// - `s`: substrate concentration
/// - `mu_max`: maximum specific growth rate
/// - `ks`: half-saturation constant
/// - `ki`: substrate inhibition constant
///
/// Returns the specific growth rate μ, which decreases at high `s`
/// due to the `S²/K_i` inhibition term.
#[must_use]
pub fn haldane(s: f64, mu_max: f64, ks: f64, ki: f64) -> f64 {
    mu_max * s / (ks + s + s * s / ki)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tolerances;

    #[test]
    fn monod_zero_substrate() {
        assert!(monod(0.0, 0.5, 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn monod_half_saturation() {
        let mu_max = 0.5;
        let ks = 10.0;
        let result = monod(ks, mu_max, ks);
        assert!((result - mu_max / 2.0).abs() < tolerances::ANALYTICAL_F64);
    }

    #[test]
    fn monod_high_substrate_approaches_mu_max() {
        let mu_max = 0.5;
        let ks = 10.0;
        let result = monod(1e6, mu_max, ks);
        assert!((result - mu_max).abs() < tolerances::ASYMPTOTIC_LIMIT);
    }

    #[test]
    fn haldane_zero_substrate() {
        assert!(haldane(0.0, 0.5, 10.0, 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn haldane_inhibition_decreases_rate() {
        let mu_max = 0.5;
        let ks = 10.0;
        let ki = 100.0;
        let low_s = haldane(50.0, mu_max, ks, ki);
        let high_s = haldane(500.0, mu_max, ks, ki);
        assert!(
            low_s > high_s,
            "Haldane rate should decrease at high substrate: {low_s} > {high_s}"
        );
    }

    #[test]
    fn haldane_reduces_to_monod_with_large_ki() {
        let s = 20.0;
        let mu_max = 0.5;
        let ks = 10.0;
        let ki = 1e15;
        let haldane_rate = haldane(s, mu_max, ks, ki);
        let monod_rate = monod(s, mu_max, ks);
        assert!(
            (haldane_rate - monod_rate).abs() < tolerances::ANALYTICAL_F64,
            "Haldane with large K_i should equal Monod"
        );
    }
}
