// SPDX-License-Identifier: AGPL-3.0-or-later
//! Biphasic dose-response (hormesis) via the Anderson localization framework.
//!
//! Models the universal observation that low-dose stress stimulates while
//! high-dose stress inhibits. The Anderson connection: biological systems
//! operate near the critical disorder threshold `W_c`. Perturbations that
//! push `W` toward `W_c` enhance coordination (hormetic benefit);
//! perturbations past `W_c` fragment it (toxicity).
//!
//! # Model
//!
//! The biphasic response combines stimulation and inhibition Hill terms:
//!
//! `R(d) = (1 + A × hill(d, K_stim, n_s)) × (1 − hill(d, K_inh, n_i))`
//!
//! where `K_inh >> K_stim` ensures inhibition dominates only at high doses.
//!
//! # References
//!
//! - Calabrese EJ (2008) Hormesis: why it is important to toxicology
//!   and toxicologists. *Environ Toxicol Chem* 27:1451-1474
//! - Anderson PW (1958) Absence of diffusion in certain random lattices.
//!   *Phys Rev* 109:1492-1505

use barracuda::stats;

/// Parameters for a biphasic (hormetic) dose-response curve.
#[derive(Debug, Clone)]
pub struct HormesisParams {
    /// Maximum stimulation amplitude (dimensionless, typically 0.1–0.5).
    /// A value of 0.3 means the peak response is 30% above baseline.
    pub amplitude: f64,

    /// Dose for half-maximal stimulation (same units as dose).
    pub k_stim: f64,

    /// Hill coefficient for the stimulation component.
    /// Higher values produce a sharper stimulation onset.
    pub n_stim: f64,

    /// Dose for half-maximal inhibition (same units as dose).
    /// Must be significantly larger than `k_stim` for clear hormesis.
    pub k_inh: f64,

    /// Hill coefficient for the inhibition component.
    /// Higher values produce a sharper toxicity threshold.
    pub n_inh: f64,
}

impl HormesisParams {
    /// Construct with validation. Returns `None` if parameters are
    /// non-physical (negative values, `k_inh` not sufficiently greater
    /// than `k_stim`).
    #[must_use]
    pub fn new(amplitude: f64, k_stim: f64, n_stim: f64, k_inh: f64, n_inh: f64) -> Option<Self> {
        if amplitude <= 0.0
            || k_stim <= 0.0
            || n_stim <= 0.0
            || k_inh <= 0.0
            || n_inh <= 0.0
            || k_inh <= k_stim
        {
            return None;
        }
        Some(Self {
            amplitude,
            k_stim,
            n_stim,
            k_inh,
            n_inh,
        })
    }
}

/// Result of a single hormesis dose-response evaluation.
#[derive(Debug, Clone)]
pub struct HormesisPoint {
    /// The input dose.
    pub dose: f64,
    /// The stimulation component (≥ 1.0 in the hormetic zone).
    pub stimulation: f64,
    /// The survival component (1.0 at zero dose, 0.0 at lethal dose).
    pub survival: f64,
    /// The composite response (stimulation × survival).
    pub response: f64,
    /// Classification of the dose regime.
    pub regime: DoseRegime,
}

/// Classification of where a dose falls relative to the hormetic zone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DoseRegime {
    /// Below detectable effect (response ≈ baseline).
    Subthreshold,
    /// In the hormetic zone (response > baseline).
    Hormetic,
    /// At or near the peak stimulation.
    Peak,
    /// Between hormetic benefit and toxicity (response declining but > baseline).
    Transition,
    /// Toxic regime (response < baseline).
    Toxic,
}

/// Evaluate the biphasic dose-response at a single dose.
///
/// Returns a response value where 1.0 is baseline. Values > 1.0 indicate
/// hormetic benefit; values < 1.0 indicate net harm.
#[must_use]
pub fn response(dose: f64, params: &HormesisParams) -> f64 {
    if dose <= 0.0 {
        return 1.0;
    }
    let stim = params
        .amplitude
        .mul_add(stats::hill(dose, params.k_stim, params.n_stim), 1.0);
    let surv = 1.0 - stats::hill(dose, params.k_inh, params.n_inh);
    stim * surv
}

/// Evaluate the biphasic dose-response with full diagnostic output.
#[must_use]
pub fn evaluate(dose: f64, params: &HormesisParams) -> HormesisPoint {
    let stim = if dose <= 0.0 {
        1.0
    } else {
        params
            .amplitude
            .mul_add(stats::hill(dose, params.k_stim, params.n_stim), 1.0)
    };
    let surv = if dose <= 0.0 {
        1.0
    } else {
        1.0 - stats::hill(dose, params.k_inh, params.n_inh)
    };
    let resp = stim * surv;

    let regime = classify_regime(resp, stim, dose, params);
    HormesisPoint {
        dose,
        stimulation: stim,
        survival: surv,
        response: resp,
        regime,
    }
}

/// Sweep a range of doses, returning one `HormesisPoint` per value.
#[must_use]
pub fn sweep(doses: &[f64], params: &HormesisParams) -> Vec<HormesisPoint> {
    doses.iter().map(|&d| evaluate(d, params)).collect()
}

/// Find the dose that produces peak hormetic stimulation.
///
/// Searches `doses` for the maximum composite response. Returns the dose
/// and response at the peak. For a well-formed hormesis curve, the peak
/// will be between `k_stim` and `k_inh`.
#[must_use]
pub fn find_peak(doses: &[f64], params: &HormesisParams) -> Option<(f64, f64)> {
    doses
        .iter()
        .map(|&d| (d, response(d, params)))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal))
        .filter(|&(_, r)| r > 1.0)
}

/// Identify the boundaries of the hormetic zone.
///
/// Returns `(low_boundary, peak_dose, high_boundary)` where response > 1.0.
/// The hormetic zone is the dose range where the system benefits from stress.
/// Returns `None` if no hormetic zone exists in the swept range.
#[must_use]
pub fn hormetic_zone(doses: &[f64], params: &HormesisParams) -> Option<(f64, f64, f64)> {
    let responses: Vec<(f64, f64)> = doses.iter().map(|&d| (d, response(d, params))).collect();

    let first_above = responses.iter().position(|&(_, r)| r > 1.0)?;
    let last_above = responses.iter().rposition(|&(_, r)| r > 1.0)?;

    let (peak_dose, _) = responses[first_above..=last_above]
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal))?;

    Some((
        responses[first_above].0,
        *peak_dose,
        responses[last_above].0,
    ))
}

/// Map a dose to Anderson disorder strength.
///
/// `W(dose) = w_baseline + sensitivity × dose^gamma`
///
/// - `w_baseline`: the system's undisturbed disorder level
/// - `sensitivity`: how strongly the dose perturbs disorder
/// - `gamma`: dose-response shape (1.0 = linear, <1.0 = saturating, >1.0 = cooperative)
#[must_use]
pub fn dose_to_disorder(dose: f64, w_baseline: f64, sensitivity: f64, gamma: f64) -> f64 {
    if dose <= 0.0 {
        return w_baseline;
    }
    sensitivity.mul_add(dose.powf(gamma), w_baseline)
}

/// Combined hormesis × Anderson analysis.
///
/// For each dose, computes both the biphasic response AND the Anderson
/// regime at the corresponding disorder level. This connects the
/// phenomenological dose-response to the underlying physics.
#[derive(Debug, Clone)]
pub struct HormesisAndersonPoint {
    /// The biphasic dose-response evaluation.
    pub hormesis: HormesisPoint,
    /// The Anderson disorder W at this dose.
    pub w: f64,
    /// Whether the Anderson regime is extended (coordinated) at this W.
    pub is_extended: bool,
    /// The level spacing ratio at this W (if computed).
    pub r: Option<f64>,
}

/// Sweep doses with both hormesis and disorder mapping.
///
/// Does not run the full eigensolver — returns W values for downstream
/// Anderson analysis. Use `anderson_spectral::sweep` on the W values
/// to get the full spectral diagnostic.
#[must_use]
pub fn sweep_with_disorder(
    doses: &[f64],
    hormesis_params: &HormesisParams,
    w_baseline: f64,
    sensitivity: f64,
    gamma: f64,
) -> Vec<(HormesisPoint, f64)> {
    doses
        .iter()
        .map(|&d| {
            let hp = evaluate(d, hormesis_params);
            let w = dose_to_disorder(d, w_baseline, sensitivity, gamma);
            (hp, w)
        })
        .collect()
}

/// Predict the hormetic zone from Anderson `W_c`.
///
/// Given a system with baseline disorder `w_baseline` and known `w_c`,
/// computes the dose range where `W(dose)` is in the near-critical zone:
/// `|W - W_c| < margin × W_c`.
///
/// Returns `(dose_low, dose_high)` or `None` if the system can't reach `W_c`.
#[must_use]
pub fn predict_hormetic_zone_from_wc(
    w_baseline: f64,
    w_c: f64,
    sensitivity: f64,
    gamma: f64,
    margin: f64,
) -> Option<(f64, f64)> {
    if sensitivity <= 0.0 || gamma <= 0.0 || margin <= 0.0 || margin >= 1.0 {
        return None;
    }

    let w_low = w_c * (1.0 - margin);
    let w_high = w_c * (1.0 + margin);

    let dose_for_w = |w_target: f64| -> Option<f64> {
        let delta_w = w_target - w_baseline;
        if delta_w <= 0.0 {
            return Some(0.0);
        }
        let base = delta_w / sensitivity;
        if base < 0.0 {
            return None;
        }
        Some(base.powf(1.0 / gamma))
    };

    let d_low = dose_for_w(w_low)?;
    let d_high = dose_for_w(w_high)?;

    Some((d_low, d_high))
}

fn classify_regime(resp: f64, stim: f64, dose: f64, params: &HormesisParams) -> DoseRegime {
    let stim_fraction = stats::hill(dose, params.k_stim, params.n_stim);
    let inh_fraction = stats::hill(dose, params.k_inh, params.n_inh);

    if resp < 1.0 {
        DoseRegime::Toxic
    } else if stim_fraction < 0.05 {
        DoseRegime::Subthreshold
    } else if stim_fraction > 0.9 && inh_fraction < 0.1 {
        DoseRegime::Peak
    } else if stim > 1.0 && resp > 1.0 {
        DoseRegime::Hormetic
    } else {
        DoseRegime::Transition
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "test module: assertions use unwrap for clarity"
)]
mod tests {
    use super::*;
    use crate::tolerances;

    fn standard_params() -> HormesisParams {
        HormesisParams::new(0.3, 1.0, 2.0, 100.0, 3.0).unwrap()
    }

    #[test]
    fn zero_dose_returns_baseline() {
        let params = standard_params();
        let r = response(0.0, &params);
        assert!(
            (r - 1.0).abs() < f64::EPSILON,
            "zero dose should give baseline 1.0: {r}"
        );
    }

    #[test]
    fn negative_dose_returns_baseline() {
        let params = standard_params();
        assert!((response(-5.0, &params) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn hormetic_peak_exceeds_baseline() {
        let params = standard_params();
        let doses: Vec<f64> = (0..100).map(|i| f64::from(i) * 0.1).collect();
        let peak = find_peak(&doses, &params);
        assert!(peak.is_some(), "should find a hormetic peak");
        let (_, peak_response) = peak.unwrap();
        assert!(
            peak_response > 1.0,
            "peak response should exceed baseline: {peak_response}"
        );
    }

    #[test]
    fn high_dose_is_toxic() {
        let params = standard_params();
        let r = response(1000.0, &params);
        assert!(
            r < tolerances::ASYMPTOTIC_LIMIT,
            "very high dose should be near-zero: {r}"
        );
    }

    #[test]
    fn biphasic_shape() {
        let params = standard_params();
        let baseline = response(0.0, &params);
        let low_dose = response(1.0, &params);
        let high_dose = response(500.0, &params);

        assert!(
            low_dose > baseline,
            "low dose should stimulate: {low_dose} > {baseline}"
        );
        assert!(
            high_dose < baseline,
            "high dose should inhibit: {high_dose} < {baseline}"
        );
    }

    #[test]
    fn hormetic_zone_boundaries() {
        let params = standard_params();
        let doses: Vec<f64> = (0..1000).map(|i| f64::from(i) * 0.5).collect();
        let zone = hormetic_zone(&doses, &params);
        assert!(zone.is_some(), "should find hormetic zone");
        let (low, peak, high) = zone.unwrap();
        assert!(low < peak, "low boundary < peak: {low} < {peak}");
        assert!(peak < high, "peak < high boundary: {peak} < {high}");
        assert!(low >= 0.0, "low boundary should be non-negative: {low}");
    }

    #[test]
    fn dose_to_disorder_baseline_at_zero() {
        let w = dose_to_disorder(0.0, 10.0, 0.5, 1.0);
        assert!(
            (w - 10.0).abs() < f64::EPSILON,
            "zero dose should give baseline W: {w}"
        );
    }

    #[test]
    fn dose_to_disorder_linear() {
        let w = dose_to_disorder(4.0, 10.0, 0.5, 1.0);
        assert!(
            (w - 12.0).abs() < tolerances::ANALYTICAL_F64,
            "linear dose-disorder: {w}"
        );
    }

    #[test]
    fn dose_to_disorder_saturating() {
        let w_half = dose_to_disorder(1.0, 0.0, 1.0, 0.5);
        let w_full = dose_to_disorder(4.0, 0.0, 1.0, 0.5);
        assert!(
            w_full < 2.0f64.mul_add(w_half, tolerances::ANALYTICAL_F64),
            "saturating response should be sublinear: {w_full} < 2×{w_half}"
        );
    }

    #[test]
    fn predict_zone_from_wc() {
        let zone = predict_hormetic_zone_from_wc(10.0, 16.5, 1.0, 1.0, 0.1);
        assert!(zone.is_some(), "should predict a zone");
        let (d_low, d_high) = zone.unwrap();
        assert!(d_low < d_high, "low < high: {d_low} < {d_high}");
        let w_at_low = dose_to_disorder(d_low, 10.0, 1.0, 1.0);
        let w_at_high = dose_to_disorder(d_high, 10.0, 1.0, 1.0);
        assert!(
            16.5f64.mul_add(-0.9, w_at_low).abs() < tolerances::ANALYTICAL_F64,
            "W at low boundary: {w_at_low}"
        );
        assert!(
            16.5f64.mul_add(-1.1, w_at_high).abs() < tolerances::ANALYTICAL_F64,
            "W at high boundary: {w_at_high}"
        );
    }

    #[test]
    fn invalid_params_rejected() {
        assert!(HormesisParams::new(-0.3, 1.0, 2.0, 100.0, 3.0).is_none());
        assert!(HormesisParams::new(0.3, 1.0, 2.0, 0.5, 3.0).is_none());
        assert!(HormesisParams::new(0.3, 0.0, 2.0, 100.0, 3.0).is_none());
    }

    #[test]
    fn sweep_returns_correct_count() {
        let params = standard_params();
        let doses: Vec<f64> = (0..20).map(f64::from).collect();
        let points = sweep(&doses, &params);
        assert_eq!(points.len(), 20);
    }

    #[test]
    fn regime_classification() {
        let params = standard_params();
        let sub = evaluate(0.001, &params);
        assert_eq!(sub.regime, DoseRegime::Subthreshold);

        let toxic = evaluate(1000.0, &params);
        assert_eq!(toxic.regime, DoseRegime::Toxic);
    }

    #[test]
    fn sweep_with_disorder_maps_correctly() {
        let params = standard_params();
        let doses = vec![0.0, 1.0, 10.0, 100.0];
        let results = sweep_with_disorder(&doses, &params, 10.0, 0.5, 1.0);
        assert_eq!(results.len(), 4);
        assert!(
            (results[0].1 - 10.0).abs() < f64::EPSILON,
            "zero dose → baseline W"
        );
        assert!(results[3].1 > results[0].1, "higher dose → higher W");
    }
}
