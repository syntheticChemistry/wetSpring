// SPDX-License-Identifier: AGPL-3.0-or-later
//! Generic Runge–Kutta 4th-order (RK4) ODE integrator.
//!
//! Fixed-step classical RK4 for systems of ordinary differential equations.
//! Designed for biological models where state variables have physical bounds
//! (non-negative concentrations, bounded fractions).
//!
//! # Example
//!
//! ```
//! use wetspring_barracuda::bio::ode::{rk4_integrate, OdeResult};
//!
//! // Exponential decay: dy/dt = -0.5 * y
//! let result = rk4_integrate(
//!     |y, _t| vec![-0.5 * y[0]],
//!     &[1.0],
//!     0.0,
//!     10.0,
//!     0.01,
//!     None,
//! );
//! assert!((result.y_final[0] - (-0.5_f64 * 10.0).exp()).abs() < 1e-6);
//! ```

/// Result of an ODE integration.
///
/// State trajectory is stored as a flat row-major `Vec<f64>` of length
/// `n_points * n_vars` for contiguous memory access. Use [`state_at`](Self::state_at)
/// for ergonomic indexing.
#[derive(Debug, Clone)]
pub struct OdeResult {
    /// Time points sampled.
    pub t: Vec<f64>,
    /// Number of state variables per time point.
    pub n_vars: usize,
    /// Flat row-major trajectory: `[y0_t0, y1_t0, ..., y0_t1, y1_t1, ...]`.
    pub y: Vec<f64>,
    /// Final state vector.
    pub y_final: Vec<f64>,
    /// Number of RK4 steps taken.
    pub steps: usize,
}

impl OdeResult {
    /// Number of time points in the trajectory.
    #[inline]
    #[must_use]
    pub fn n_points(&self) -> usize {
        self.t.len()
    }

    /// Slice of all state variables at time point `i`.
    #[inline]
    #[must_use]
    pub fn state_at(&self, i: usize) -> &[f64] {
        let start = i * self.n_vars;
        &self.y[start..start + self.n_vars]
    }

    /// Single variable value at `(time_point, var_index)`.
    #[inline]
    #[must_use]
    pub fn var_at(&self, time_point: usize, var_idx: usize) -> f64 {
        self.y[time_point * self.n_vars + var_idx]
    }

    /// Iterator over all state snapshots (each is a `&[f64]` slice of `n_vars` elements).
    #[inline]
    pub fn states(&self) -> impl Iterator<Item = &[f64]> {
        self.y.chunks_exact(self.n_vars)
    }
}

/// Perform a single RK4 step.
///
/// Given `dy/dt = f(y, t)`, advance from `y` at time `t` by step `dt`.
#[must_use]
pub fn rk4_step<F>(f: &F, y: &[f64], t: f64, dt: f64) -> Vec<f64>
where
    F: Fn(&[f64], f64) -> Vec<f64>,
{
    let half_dt = 0.5 * dt;
    let k1 = f(y, t);

    let y2: Vec<f64> = y
        .iter()
        .zip(&k1)
        .map(|(&yi, &ki)| half_dt.mul_add(ki, yi))
        .collect();
    let k2 = f(&y2, t + half_dt);

    let y3: Vec<f64> = y
        .iter()
        .zip(&k2)
        .map(|(&yi, &ki)| half_dt.mul_add(ki, yi))
        .collect();
    let k3 = f(&y3, t + half_dt);

    let y4: Vec<f64> = y
        .iter()
        .zip(&k3)
        .map(|(&yi, &ki)| dt.mul_add(ki, yi))
        .collect();
    let k4 = f(&y4, t + dt);

    let sixth_dt = dt / 6.0;
    y.iter()
        .enumerate()
        .map(|(i, &yi)| {
            let slope = 2.0f64.mul_add(k2[i] + k3[i], k1[i] + k4[i]);
            sixth_dt.mul_add(slope, yi)
        })
        .collect()
}

/// Integrate an ODE system using classical RK4 with fixed step size.
///
/// - `f`: right-hand side `dy/dt = f(y, t)`
/// - `y0`: initial state
/// - `t_start`, `t_end`: integration interval
/// - `dt`: step size (smaller → more accurate, typical 0.001 for biological ODE)
/// - `clamp`: optional per-variable `(min, max)` bounds applied after each step
#[must_use]
pub fn rk4_integrate<F>(
    f: F,
    y0: &[f64],
    t_start: f64,
    t_end: f64,
    dt: f64,
    clamp: Option<&[(f64, f64)]>,
) -> OdeResult
where
    F: Fn(&[f64], f64) -> Vec<f64>,
{
    let n_vars = y0.len();
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let n_steps = ((t_end - t_start) / dt).ceil() as usize;

    let mut t_vec = Vec::with_capacity(n_steps + 1);
    let mut y_flat = Vec::with_capacity((n_steps + 1) * n_vars);

    let mut t = t_start;
    let mut y = y0.to_vec();

    t_vec.push(t);
    y_flat.extend_from_slice(&y);

    for _ in 0..n_steps {
        let actual_dt = dt.min(t_end - t);
        if actual_dt <= 0.0 {
            break;
        }
        y = rk4_step(&f, &y, t, actual_dt);

        if let Some(bounds) = clamp {
            for (yi, &(lo, hi)) in y.iter_mut().zip(bounds) {
                *yi = yi.clamp(lo, hi);
            }
        }

        t += actual_dt;
        t_vec.push(t);
        y_flat.extend_from_slice(&y);
    }

    let y_final = y;
    OdeResult {
        t: t_vec,
        n_vars,
        y: y_flat,
        y_final,
        steps: n_steps,
    }
}

/// Compute the mean of the last `frac` fraction of a trajectory column.
///
/// Used for steady-state analysis: `steady_state_mean(&result, 0, 0.1)` gives
/// the mean of variable 0 over the last 10% of time points.
#[must_use]
pub fn steady_state_mean(result: &OdeResult, var_idx: usize, frac: f64) -> f64 {
    let n = result.n_points();
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    let tail_len = (n as f64 * frac).ceil() as usize;
    let start = n.saturating_sub(tail_len);
    let count = n - start;
    if count == 0 {
        return 0.0;
    }
    let sum: f64 = (start..n).map(|i| result.var_at(i, var_idx)).sum();
    #[allow(clippy::cast_precision_loss)]
    let mean = sum / count as f64;
    mean
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exponential_decay() {
        let result = rk4_integrate(|y, _t| vec![-0.5 * y[0]], &[1.0], 0.0, 10.0, 0.01, None);
        let expected = (-0.5_f64 * 10.0).exp();
        assert!(
            (result.y_final[0] - expected).abs() < 1e-8,
            "RK4 exponential decay: got {}, expected {expected}",
            result.y_final[0]
        );
    }

    #[test]
    fn logistic_growth() {
        let k = 1.0;
        let r = 0.8;
        let y0 = 0.01;
        let t_end = 30.0;
        let result = rk4_integrate(
            move |y, _t| vec![r * y[0] * (1.0 - y[0] / k)],
            &[y0],
            0.0,
            t_end,
            0.01,
            None,
        );
        assert!(
            (result.y_final[0] - k).abs() < 1e-4,
            "logistic should reach carrying capacity: got {}",
            result.y_final[0]
        );
    }

    #[test]
    fn two_variable_system() {
        // dx/dt = -y, dy/dt = x  →  circular orbit, ||(x,y)|| should stay ≈ 1
        let result = rk4_integrate(
            |y, _t| vec![-y[1], y[0]],
            &[1.0, 0.0],
            0.0,
            std::f64::consts::TAU,
            0.001,
            None,
        );
        let r = result.y_final[0].hypot(result.y_final[1]);
        assert!(
            (r - 1.0).abs() < 1e-6,
            "circular orbit radius should be 1.0, got {r}"
        );
    }

    #[test]
    fn clamping_prevents_negative() {
        let result = rk4_integrate(
            |_y, _t| vec![-10.0],
            &[1.0],
            0.0,
            5.0,
            0.01,
            Some(&[(0.0, f64::INFINITY)]),
        );
        assert!(
            result.y_final[0] >= 0.0,
            "clamped variable must stay non-negative"
        );
    }

    #[test]
    fn steady_state_mean_computation() {
        let result = rk4_integrate(
            move |y, _t| vec![0.8 * y[0] * (1.0 - y[0])],
            &[0.01],
            0.0,
            50.0,
            0.01,
            None,
        );
        let ss = steady_state_mean(&result, 0, 0.1);
        assert!(
            (ss - 1.0).abs() < 0.01,
            "steady-state mean should be ~1.0, got {ss}"
        );
    }

    #[test]
    fn step_count_matches_expectation() {
        let result = rk4_integrate(|y, _t| vec![-y[0]], &[1.0], 0.0, 1.0, 0.1, None);
        assert_eq!(result.steps, 10);
        assert_eq!(result.t.len(), 11);
        assert_eq!(result.n_points(), 11);
    }

    #[test]
    fn zero_dt_does_not_panic() {
        let result = rk4_integrate(|y, _t| vec![-y[0]], &[1.0], 0.0, 0.0, 0.01, None);
        assert_eq!(result.y_final, vec![1.0]);
    }
}
