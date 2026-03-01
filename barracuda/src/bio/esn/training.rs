// SPDX-License-Identifier: AGPL-3.0-or-later
//! Ridge regression training for ESN readout.

/// Solve `(SᵀS + λI)·w = Sᵀy` via `ToadStool`'s Cholesky-based ridge regression.
///
/// Falls back to zero weights if the solve fails.
pub(super) fn solve_ridge(
    w_out: &mut [f64],
    flat_states: &[f64],
    flat_targets: &[f64],
    n_samples: usize,
    n_res: usize,
    n_out: usize,
    regularization: f64,
) {
    match barracuda::linalg::ridge_regression(
        flat_states,
        flat_targets,
        n_samples,
        n_res,
        n_out,
        regularization,
    ) {
        Ok(result) => {
            w_out[..result.weights.len()].copy_from_slice(&result.weights);
        }
        Err(_) => {
            w_out.fill(0.0);
        }
    }
}
