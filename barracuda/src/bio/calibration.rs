// SPDX-License-Identifier: AGPL-3.0-or-later
//! Calibration and quantitation for analytical chemistry.
//!
//! Builds standard curves from known concentration/response pairs and
//! predicts unknown concentrations from observed responses (peak areas
//! or heights). Delegates regression to `barracuda::stats::fit_linear`.
//!
//! # Pipeline
//!
//! ```text
//! Standards (concentration, peak_area) → fit_linear → CalibrationCurve
//!     → predict(unknown_area) → concentration ± uncertainty
//! ```
//!
//! # References
//!
//! - ICH Q2(R1): Validation of Analytical Procedures (linearity, range)
//! - EPA Method 533 / 537.1: PFAS quantitation via isotope dilution

/// A fitted calibration curve: response = slope × concentration + intercept.
#[derive(Debug, Clone)]
pub struct CalibrationCurve {
    /// Slope of the fitted line (response per concentration unit).
    pub slope: f64,
    /// Y-intercept of the fitted line.
    pub intercept: f64,
    /// Coefficient of determination (R²).
    pub r_squared: f64,
    /// Root mean square error of the fit.
    pub rmse: f64,
    /// Number of calibration points used.
    pub n_points: usize,
    /// Concentration unit label (e.g. "ng/L", "µM").
    pub conc_unit: String,
    /// Response unit label (e.g. "AU·min", "counts").
    pub response_unit: String,
}

/// Result of quantitation: predicted concentration with quality metrics.
#[derive(Debug, Clone)]
pub struct QuantResult {
    /// Predicted concentration from the calibration curve.
    pub concentration: f64,
    /// Observed response (peak area or height) that was quantified.
    pub response: f64,
    /// Whether the response falls within the calibration range.
    pub within_range: bool,
}

/// Fit a calibration curve from known concentration/response pairs.
///
/// Delegates to `barracuda::stats::fit_linear(concentrations, responses)`
/// to produce `response = slope × concentration + intercept`.
///
/// # Arguments
///
/// * `concentrations` — Known standard concentrations (x-axis).
/// * `responses` — Measured responses at each concentration (y-axis).
/// * `conc_unit` — Label for concentration units.
/// * `response_unit` — Label for response units.
///
/// # Returns
///
/// `Some(CalibrationCurve)` if the fit succeeds (≥ 2 points, non-singular),
/// `None` otherwise.
#[must_use]
pub fn fit_calibration(
    concentrations: &[f64],
    responses: &[f64],
    conc_unit: &str,
    response_unit: &str,
) -> Option<CalibrationCurve> {
    let result = barracuda::stats::fit_linear(concentrations, responses)?;
    let slope = result.params[0];
    let intercept = result.params[1];
    Some(CalibrationCurve {
        slope,
        intercept,
        r_squared: result.r_squared,
        rmse: result.rmse,
        n_points: concentrations.len(),
        conc_unit: conc_unit.into(),
        response_unit: response_unit.into(),
    })
}

impl CalibrationCurve {
    /// Predict concentration from an observed response.
    ///
    /// Inverts `response = slope × conc + intercept` to give
    /// `conc = (response − intercept) / slope`.
    ///
    /// Returns `None` if the slope is zero (degenerate curve).
    #[must_use]
    pub fn predict(&self, response: f64) -> Option<QuantResult> {
        if self.slope.abs() < f64::EPSILON {
            return None;
        }
        let conc = (response - self.intercept) / self.slope;
        Some(QuantResult {
            concentration: conc,
            response,
            within_range: true,
        })
    }

    /// Predict concentration and check whether it falls within the
    /// calibration range `[min_conc, max_conc]`.
    #[must_use]
    pub fn predict_with_range(
        &self,
        response: f64,
        min_conc: f64,
        max_conc: f64,
    ) -> Option<QuantResult> {
        let mut qr = self.predict(response)?;
        qr.within_range = qr.concentration >= min_conc && qr.concentration <= max_conc;
        Some(qr)
    }

    /// Compute the expected response for a given concentration.
    ///
    /// `response = slope × concentration + intercept`
    #[must_use]
    pub fn expected_response(&self, concentration: f64) -> f64 {
        self.slope.mul_add(concentration, self.intercept)
    }

    /// Compute residuals for the calibration standards.
    ///
    /// Each residual = observed − expected.
    #[must_use]
    pub fn residuals(&self, concentrations: &[f64], responses: &[f64]) -> Vec<f64> {
        concentrations
            .iter()
            .zip(responses.iter())
            .map(|(&c, &r)| r - self.expected_response(c))
            .collect()
    }
}

/// Batch quantitation: predict concentrations for multiple responses.
///
/// # Arguments
///
/// * `curve` — Fitted calibration curve.
/// * `responses` — Observed response values to quantify.
///
/// # Returns
///
/// Vector of [`QuantResult`], one per response. Entries are `None`-filtered
/// (only possible for degenerate curves with zero slope).
#[must_use]
pub fn quantify_batch(curve: &CalibrationCurve, responses: &[f64]) -> Vec<QuantResult> {
    responses
        .iter()
        .filter_map(|&r| curve.predict(r))
        .collect()
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::tolerances;

    #[test]
    fn fit_perfect_linear() {
        let conc = [0.0, 1.0, 2.0, 3.0, 4.0];
        let resp = [0.0, 100.0, 200.0, 300.0, 400.0];
        let curve = fit_calibration(&conc, &resp, "ng/L", "AU·min");
        assert!(curve.is_some());
        let c = curve.unwrap();
        assert!((c.slope - 100.0).abs() < tolerances::ANALYTICAL_LOOSE);
        assert!(c.intercept.abs() < tolerances::ANALYTICAL_LOOSE);
        assert!((c.r_squared - 1.0).abs() < tolerances::ANALYTICAL_LOOSE);
        assert_eq!(c.n_points, 5);
    }

    #[test]
    fn fit_with_intercept() {
        let conc = [1.0, 2.0, 3.0, 4.0, 5.0];
        let resp = [150.0, 250.0, 350.0, 450.0, 550.0];
        let curve = fit_calibration(&conc, &resp, "µM", "counts").unwrap();
        assert!((curve.slope - 100.0).abs() < tolerances::ANALYTICAL_LOOSE);
        assert!((curve.intercept - 50.0).abs() < tolerances::ANALYTICAL_LOOSE);
    }

    #[test]
    fn predict_inverse() {
        let conc = [0.0, 1.0, 2.0, 3.0];
        let resp = [10.0, 30.0, 50.0, 70.0];
        let curve = fit_calibration(&conc, &resp, "ng/L", "AU").unwrap();
        let qr = curve.predict(50.0).unwrap();
        assert!((qr.concentration - 2.0).abs() < tolerances::ANALYTICAL_LOOSE);
    }

    #[test]
    fn predict_with_range_checks() {
        let conc = [0.0, 10.0, 20.0, 30.0];
        let resp = [0.0, 100.0, 200.0, 300.0];
        let curve = fit_calibration(&conc, &resp, "ng/L", "AU").unwrap();

        let in_range = curve.predict_with_range(150.0, 0.0, 30.0).unwrap();
        assert!(in_range.within_range);

        let out_range = curve.predict_with_range(500.0, 0.0, 30.0).unwrap();
        assert!(!out_range.within_range);
    }

    #[test]
    fn residuals_perfect_fit() {
        let conc = [0.0, 1.0, 2.0];
        let resp = [0.0, 10.0, 20.0];
        let curve = fit_calibration(&conc, &resp, "u", "u").unwrap();
        let resid = curve.residuals(&conc, &resp);
        for r in &resid {
            assert!(r.abs() < tolerances::ANALYTICAL_LOOSE, "residual {r}");
        }
    }

    #[test]
    fn quantify_batch_multiple() {
        let conc = [0.0, 5.0, 10.0, 20.0];
        let resp = [0.0, 500.0, 1000.0, 2000.0];
        let curve = fit_calibration(&conc, &resp, "ppb", "AU").unwrap();
        let results = quantify_batch(&curve, &[250.0, 750.0, 1500.0]);
        assert_eq!(results.len(), 3);
        assert!((results[0].concentration - 2.5).abs() < tolerances::ANALYTICAL_LOOSE);
        assert!((results[1].concentration - 7.5).abs() < tolerances::ANALYTICAL_LOOSE);
        assert!((results[2].concentration - 15.0).abs() < tolerances::ANALYTICAL_LOOSE);
    }

    #[test]
    fn fit_insufficient_points_returns_none() {
        let conc = [1.0];
        let resp = [100.0];
        assert!(fit_calibration(&conc, &resp, "u", "u").is_none());
    }

    #[test]
    fn expected_response_round_trip() {
        let conc = [0.0, 1.0, 2.0, 3.0];
        let resp = [5.0, 15.0, 25.0, 35.0];
        let curve = fit_calibration(&conc, &resp, "u", "u").unwrap();
        for (&c, &r) in conc.iter().zip(resp.iter()) {
            let er = curve.expected_response(c);
            assert!((er - r).abs() < tolerances::ANALYTICAL_LOOSE);
        }
    }
}
