// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bistable phenotypic switching model — Fernandez et al. 2020.
//!
//! Extends the Waters 2008 QS/c-di-GMP model with a positive feedback loop
//! on cell shape / VPS production that creates bistability: cells can occupy
//! either a motile or sessile state depending on history.
//!
//! # References
//!
//! - Fernandez et al. 2020, *PNAS* 117:29046-29054
//!   "V. cholerae adapts to sessile and motile lifestyles by c-di-GMP regulation of cell shape"
//! - Waters et al. 2008, *J Bacteriol* 190:2527-36
//!
//! # State variables
//!
//! | Index | Variable | Description |
//! |-------|----------|-------------|
//! | 0 | N | Cell density (OD, 0–K) |
//! | 1 | A | Autoinducer (µM) |
//! | 2 | H | `HapR` level (normalized) |
//! | 3 | C | c-di-GMP (µM) |
//! | 4 | B | Biofilm / sessile state (0–1) |
//!
//! The bistability arises from positive feedback: high B reinforces c-di-GMP
//! production (sessile cells make more DGC), creating hysteresis. A cell that
//! enters the sessile state requires a stronger dispersal signal to leave it
//! than was needed to enter it.

use super::ode::{rk4_integrate, steady_state_mean, OdeResult};
use super::qs_biofilm::QsBiofilmParams;

/// Extended parameters for the bistable model (Fernandez 2020).
///
/// Adds a positive-feedback gain `alpha_fb` on DGC production from biofilm
/// state and a cooperative feedback Hill coefficient `n_fb`.
#[derive(Debug, Clone)]
pub struct BistableParams {
    /// Base QS/biofilm parameters (Waters 2008).
    pub base: QsBiofilmParams,
    /// Positive feedback gain: sessile cells boost DGC production.
    /// `alpha_fb` = 0 recovers the monostable Waters 2008 model.
    pub alpha_fb: f64,
    /// Hill coefficient for the positive feedback loop.
    pub n_fb: f64,
    /// Half-saturation for the feedback Hill function.
    pub k_fb: f64,
}

impl Default for BistableParams {
    /// Bistable regime parameters.
    ///
    /// Uses reduced `HapR` repression of DGC (`k_dgc_rep = 0.3`), reduced PDE
    /// activation (`k_pde_act = 0.5`), steep Hill functions (`n_bio = 4`,
    /// `n_fb = 4`), and moderate feedback gain (`alpha_fb = 3.0`).
    /// `k_bio_cdg = 1.5` ensures a distinct low-B attractor exists.
    /// This regime produces a fold bifurcation with hysteresis in `alpha_fb`.
    fn default() -> Self {
        let base = QsBiofilmParams {
            k_dgc_rep: 0.3,
            k_pde_act: 0.5,
            k_bio_cdg: 1.5,
            n_bio: 4.0,
            ..QsBiofilmParams::default()
        };
        Self {
            base,
            alpha_fb: 3.0,
            n_fb: 4.0,
            k_fb: 0.6,
        }
    }
}

/// Number of state variables (same as `qs_biofilm`).
pub const N_VARS: usize = 5;
/// Number of f64 parameters when flattened for GPU dispatch.
///
/// 18 from [`QsBiofilmParams`] + 3 feedback params = 21.
pub const N_PARAMS: usize = 21;

impl BistableParams {
    /// Flatten parameters into a contiguous `f64` slice for GPU dispatch.
    ///
    /// The first 18 values are the base [`QsBiofilmParams`] layout, followed
    /// by `[alpha_fb, n_fb, k_fb]`.
    #[must_use]
    pub fn to_flat(&self) -> [f64; N_PARAMS] {
        let base = self.base.to_flat();
        let mut out = [0.0; N_PARAMS];
        out[..super::qs_biofilm::N_PARAMS].copy_from_slice(&base);
        out[18] = self.alpha_fb;
        out[19] = self.n_fb;
        out[20] = self.k_fb;
        out
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
            base: QsBiofilmParams::from_flat(flat),
            alpha_fb: flat[18],
            n_fb: flat[19],
            k_fb: flat[20],
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

/// Right-hand side of the bistable QS / c-di-GMP / biofilm ODE system.
///
/// Compared to `qs_biofilm::qs_rhs`, adds:
///   `DGC_rate += alpha_fb * Hill(B, k_fb, n_fb)`
#[allow(clippy::many_single_char_names)]
fn bistable_rhs(state: &[f64], _t: f64, p: &BistableParams) -> Vec<f64> {
    let cell = state[0].max(0.0);
    let ai = state[1].max(0.0);
    let hapr = state[2].max(0.0);
    let cdg = state[3].max(0.0);
    let bio = state[4].max(0.0);

    let b = &p.base;

    let d_cell = (b.mu_max * cell).mul_add(1.0 - cell / b.k_cap, -(b.death_rate * cell));
    let d_ai = b.k_ai_prod.mul_add(cell, -b.d_ai * ai);
    let d_hapr = b
        .k_hapr_max
        .mul_add(hill(ai, b.k_hapr_ai, b.n_hapr), -b.d_hapr * hapr);

    let basal_dgc = b.k_dgc_basal * b.k_dgc_rep.mul_add(-hapr, 1.0).max(0.0);
    let feedback_dgc = p.alpha_fb * hill(bio, p.k_fb, p.n_fb);
    let dgc_rate = basal_dgc + feedback_dgc;

    let pde_rate = b.k_pde_act.mul_add(hapr, b.k_pde_basal);
    let mut d_cdg = b.d_cdg.mul_add(-cdg, dgc_rate - pde_rate * cdg);
    if cdg < crate::tolerances::ODE_CDG_CONVERGENCE && d_cdg < 0.0 {
        d_cdg = 0.0;
    }

    let bio_promote = b.k_bio_max * hill(cdg, b.k_bio_cdg, b.n_bio);
    let d_bio = bio_promote.mul_add(1.0 - bio, -(b.d_bio * bio));

    vec![d_cell, d_ai, d_hapr, d_cdg, d_bio]
}

/// Biological bounds for the 5-variable system.
const CLAMP: [(f64, f64); 5] = [
    (0.0, f64::INFINITY),
    (0.0, f64::INFINITY),
    (0.0, f64::INFINITY),
    (0.0, f64::INFINITY),
    (0.0, 1.0),
];

/// Run the bistable model from given initial conditions.
#[must_use]
pub fn run_bistable(y0: &[f64; 5], t_end: f64, dt: f64, params: &BistableParams) -> OdeResult {
    rk4_integrate(
        |y, t| bistable_rhs(y, t, params),
        y0,
        0.0,
        t_end,
        dt,
        Some(&CLAMP),
    )
}

/// Result of a bifurcation scan: steady-state values at each parameter value.
#[derive(Debug, Clone)]
pub struct BifurcationResult {
    /// Parameter values scanned (e.g., `alpha_fb`).
    pub param_values: Vec<f64>,
    /// Steady-state biofilm (B) at each parameter value, scanning forward.
    pub b_forward: Vec<f64>,
    /// Steady-state biofilm (B) at each parameter value, scanning backward.
    pub b_backward: Vec<f64>,
    /// Hysteresis width: difference between critical thresholds.
    pub hysteresis_width: f64,
}

/// Scan `alpha_fb` from `lo` to `hi` in `n_steps`, running to steady state
/// at each point. Uses continuation (previous steady state as next IC) to
/// detect hysteresis.
///
/// Returns forward and backward branches of the bifurcation diagram.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn bifurcation_scan(
    base_params: &BistableParams,
    lo: f64,
    hi: f64,
    n_steps: usize,
    dt: f64,
    t_settle: f64,
) -> BifurcationResult {
    let step = (hi - lo) / n_steps as f64;
    let ss_frac = 0.1;

    let mut param_values = Vec::with_capacity(n_steps + 1);
    let mut b_forward = Vec::with_capacity(n_steps + 1);
    let mut b_backward = Vec::with_capacity(n_steps + 1);

    // Forward sweep: start from low-biofilm state
    let mut y = [0.9, 4.0, 1.8, 0.1, 0.02]; // motile steady state
    for i in 0..=n_steps {
        let alpha = lo.mul_add(1.0, step * i as f64);
        param_values.push(alpha);

        let mut p = base_params.clone();
        p.alpha_fb = alpha;
        let result = run_bistable(&[y[0], y[1], y[2], y[3], y[4]], t_settle, dt, &p);
        let b_ss = steady_state_mean(&result, 4, ss_frac);
        b_forward.push(b_ss);

        y = [
            result.y_final[0],
            result.y_final[1],
            result.y_final[2],
            result.y_final[3],
            result.y_final[4],
        ];
    }

    // Backward sweep: start from high-biofilm state
    y = [0.9, 4.0, 1.8, 3.0, 0.9]; // sessile steady state
    for i in (0..=n_steps).rev() {
        let alpha = lo.mul_add(1.0, step * i as f64);

        let mut p = base_params.clone();
        p.alpha_fb = alpha;
        let result = run_bistable(&[y[0], y[1], y[2], y[3], y[4]], t_settle, dt, &p);
        let b_ss = steady_state_mean(&result, 4, ss_frac);
        b_backward.push(b_ss);

        y = [
            result.y_final[0],
            result.y_final[1],
            result.y_final[2],
            result.y_final[3],
            result.y_final[4],
        ];
    }

    b_backward.reverse();

    // Compute hysteresis width: find where forward and backward branches
    // diverge by more than 0.1 (threshold for "different attractors")
    let diverge_threshold = 0.1;
    let mut first_diverge = None;
    let mut last_diverge = None;
    for (i, (&fwd, &bwd)) in b_forward.iter().zip(&b_backward).enumerate() {
        if (fwd - bwd).abs() > diverge_threshold {
            if first_diverge.is_none() {
                first_diverge = Some(i);
            }
            last_diverge = Some(i);
        }
    }

    let hysteresis_width = match (first_diverge, last_diverge) {
        (Some(f), Some(l)) => param_values[l] - param_values[f],
        _ => 0.0,
    };

    BifurcationResult {
        param_values,
        b_forward,
        b_backward,
        hysteresis_width,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f64 = 0.001;
    const SS_FRAC: f64 = 0.1;

    #[test]
    fn zero_feedback_matches_python() {
        let p = BistableParams {
            alpha_fb: 0.0,
            ..Default::default()
        };
        let r = run_bistable(&[0.01, 0.0, 0.0, 2.0, 0.5], 48.0, DT, &p);
        let n_ss = steady_state_mean(&r, 0, SS_FRAC);
        let b_ss = steady_state_mean(&r, 4, SS_FRAC);
        assert!(
            (n_ss - 0.975).abs() < 1e-3,
            "zero feedback N_ss should match Python: {n_ss}"
        );
        assert!(
            (b_ss - 0.040).abs() < 0.005,
            "zero feedback B_ss should match Python: {b_ss}"
        );
    }

    #[test]
    fn strong_feedback_locks_sessile() {
        let p = BistableParams {
            alpha_fb: 8.0,
            ..Default::default()
        };
        let r = run_bistable(&[0.01, 0.0, 0.0, 2.0, 0.8], 48.0, DT, &p);
        let b_ss = steady_state_mean(&r, 4, SS_FRAC);
        assert!(
            (b_ss - 0.831).abs() < 0.01,
            "strong feedback should match Python sessile: B_ss={b_ss}"
        );
    }

    #[test]
    fn hysteresis_detected() {
        let p = BistableParams::default();
        let bif = bifurcation_scan(&p, 0.0, 10.0, 20, 0.01, 48.0);
        assert!(
            bif.hysteresis_width > 0.5,
            "should detect hysteresis with default params, got width={}",
            bif.hysteresis_width
        );
    }

    #[test]
    fn minimal_hysteresis_with_original_waters_params() {
        let p = BistableParams {
            base: QsBiofilmParams::default(), // n_bio=2 → near-monostable
            alpha_fb: 0.0,
            n_fb: 4.0,
            k_fb: 0.6,
        };
        let bif = bifurcation_scan(&p, 0.0, 10.0, 20, 0.01, 48.0);
        let bistable_p = BistableParams::default();
        let bif_bistable = bifurcation_scan(&bistable_p, 0.0, 10.0, 20, 0.01, 48.0);
        assert!(
            bif.hysteresis_width < bif_bistable.hysteresis_width * 0.5,
            "original Waters params should have much less hysteresis than bistable: {} vs {}",
            bif.hysteresis_width,
            bif_bistable.hysteresis_width
        );
    }

    #[test]
    fn all_variables_non_negative() {
        let p = BistableParams::default();
        let result = run_bistable(&[0.01, 0.0, 0.0, 2.0, 0.5], 24.0, DT, &p);
        for (step, row) in result.y.iter().enumerate() {
            for (var, &val) in row.iter().enumerate() {
                assert!(
                    val >= 0.0,
                    "variable {var} went negative ({val}) at step {step}"
                );
            }
        }
    }

    #[test]
    fn deterministic_across_runs() {
        let p = BistableParams::default();
        let r1 = run_bistable(&[0.01, 0.0, 0.0, 2.0, 0.5], 24.0, DT, &p);
        let r2 = run_bistable(&[0.01, 0.0, 0.0, 2.0, 0.5], 24.0, DT, &p);
        for (a, b) in r1.y_final.iter().zip(&r2.y_final) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "bistable ODE should be bitwise deterministic"
            );
        }
    }

    #[test]
    fn bifurcation_scan_length() {
        let p = BistableParams::default();
        let bif = bifurcation_scan(&p, 0.0, 6.0, 10, 0.01, 24.0);
        assert_eq!(bif.param_values.len(), 11);
        assert_eq!(bif.b_forward.len(), 11);
        assert_eq!(bif.b_backward.len(), 11);
    }

    #[test]
    fn flat_params_round_trip() {
        let p = BistableParams::default();
        let flat = p.to_flat();
        assert_eq!(flat.len(), N_PARAMS);
        let p2 = BistableParams::from_flat(&flat);
        let flat2 = p2.to_flat();
        for (a, b) in flat.iter().zip(&flat2) {
            assert_eq!(a.to_bits(), b.to_bits(), "round-trip must be bitwise exact");
        }
    }

    #[test]
    fn flat_params_gpu_parity() {
        let p = BistableParams::default();
        let flat = p.to_flat();
        let p2 = BistableParams::from_flat(&flat);
        let r1 = run_bistable(&[0.01, 0.0, 0.0, 2.0, 0.5], 24.0, DT, &p);
        let r2 = run_bistable(&[0.01, 0.0, 0.0, 2.0, 0.5], 24.0, DT, &p2);
        for (a, b) in r1.y_final.iter().zip(&r2.y_final) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "flat round-trip must produce identical ODE results"
            );
        }
    }
}
