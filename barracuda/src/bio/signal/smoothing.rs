// SPDX-License-Identifier: AGPL-3.0-or-later
//! Savitzky-Golay smoothing for chromatographic signals (EIC pre-processing).
//!
//! Fits a local polynomial by least squares over a sliding window, evaluates at
//! center. `(VᵀV)c = e₀` via Vandermonde V, `conv_coeffs = V·c`. Edge: shrink window.

use crate::error::{Error, Result};

fn coefficients_for_x(x_values: &[f64], poly_order: usize) -> Result<Vec<f64>> {
    let win_len = x_values.len();
    if poly_order >= win_len {
        return Err(Error::InvalidInput(
            "poly_order must be less than window length".into(),
        ));
    }
    let n_cols = poly_order + 1;
    let mut v = vec![0.0; win_len * n_cols];
    for (i, row) in v.chunks_exact_mut(n_cols).enumerate() {
        let mut xp = 1.0;
        for r in row.iter_mut() {
            *r = xp;
            xp *= x_values[i];
        }
    }
    let mut vtv = vec![0.0; n_cols * n_cols];
    for i in 0..n_cols {
        for j in 0..n_cols {
            vtv[i * n_cols + j] = (0..win_len)
                .map(|k| v[k * n_cols + i] * v[k * n_cols + j])
                .sum();
        }
    }
    let mut b = vec![0.0; n_cols];
    b[0] = 1.0;
    let x = solve_symmetric(&vtv, &b, n_cols)?;
    let mut coeffs = vec![0.0; win_len];
    for k in 0..win_len {
        coeffs[k] = (0..n_cols).map(|j| v[k * n_cols + j] * x[j]).sum();
    }
    Ok(coeffs)
}

/// Compute Savitzky-Golay convolution coefficients for a symmetric window.
///
/// `window_size` must be odd and positive; `poly_order` must be less than `window_size`.
/// Builds Vandermonde matrix V with x ∈ `[-half, half]`, solves `(VᵀV)c = e₀`.
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] if `window_size` is even, zero, or ≤ `poly_order`.
pub fn savitzky_golay_coefficients(window_size: usize, poly_order: usize) -> Result<Vec<f64>> {
    if window_size == 0 || window_size.is_multiple_of(2) {
        return Err(Error::InvalidInput(
            "window_size must be odd and positive".into(),
        ));
    }
    if poly_order >= window_size {
        return Err(Error::InvalidInput(
            "poly_order must be less than window_size".into(),
        ));
    }
    let half = (window_size - 1) / 2;
    let half_i32 = i32::try_from(half)
        .map_err(|_| Error::InvalidInput("window_size too large for Savitzky-Golay".into()))?;
    let x_values: Vec<f64> = (0..window_size)
        .map(|i| {
            #[expect(
                clippy::cast_possible_wrap,
                reason = "i < window_size ≤ i32::MAX checked above"
            )]
            let idx = i as i32;
            f64::from(idx - half_i32)
        })
        .collect();
    coefficients_for_x(&x_values, poly_order)
}

fn solve_symmetric(a: &[f64], b: &[f64], n: usize) -> Result<Vec<f64>> {
    let mut aug = Vec::with_capacity(n * (n + 1));
    for i in 0..n {
        aug.extend_from_slice(&a[i * n..(i + 1) * n]);
        aug.push(b[i]);
    }
    for k in 0..n {
        let mut max_row = k;
        let mut max_val = aug[k * (n + 1) + k].abs();
        for r in (k + 1)..n {
            let v = aug[r * (n + 1) + k].abs();
            if v > max_val {
                max_val = v;
                max_row = r;
            }
        }
        if max_val < 1e-15 {
            return Err(Error::InvalidInput(
                "singular matrix in Savitzky-Golay coefficient computation".into(),
            ));
        }
        if max_row != k {
            for j in 0..=n {
                aug.swap(k * (n + 1) + j, max_row * (n + 1) + j);
            }
        }
        let pivot = aug[k * (n + 1) + k];
        for j in 0..=n {
            aug[k * (n + 1) + j] /= pivot;
        }
        for r in 0..n {
            if r != k {
                let f = aug[r * (n + 1) + k];
                for j in 0..=n {
                    aug[r * (n + 1) + j] -= f * aug[k * (n + 1) + j];
                }
            }
        }
    }
    Ok((0..n).map(|i| aug[i * (n + 1) + n]).collect())
}

/// Apply Savitzky-Golay smoothing with boundary-adaptive windows.
///
/// Smooths `data` using a local polynomial of degree `poly_order` over a
/// sliding window of `window_size` points. Near boundaries the window shrinks
/// to avoid zero-padding artifacts.
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] if `window_size` is even/zero,
/// `poly_order >= window_size`, or `window_size > data.len()`.
pub fn savitzky_golay(data: &[f64], window_size: usize, poly_order: usize) -> Result<Vec<f64>> {
    if data.is_empty() {
        return Err(Error::InvalidInput("data must be non-empty".into()));
    }
    if window_size == 0 || window_size.is_multiple_of(2) {
        return Err(Error::InvalidInput(
            "window_size must be odd and positive".into(),
        ));
    }
    if poly_order >= window_size {
        return Err(Error::InvalidInput(
            "poly_order must be less than window_size".into(),
        ));
    }
    if window_size > data.len() {
        return Err(Error::InvalidInput(
            "window_size must not exceed data length".into(),
        ));
    }
    let half = (window_size - 1) / 2;
    let n = data.len();
    let mut out = vec![0.0; n];

    for i in 0..n {
        let left = i.saturating_sub(half);
        let right = (i + half).min(n - 1);
        let win_len = right - left + 1;
        let order = poly_order.min(win_len - 1);
        if order == 0 {
            out[i] = data[i];
            continue;
        }
        let x_values: Vec<f64> = (left..=right)
            .map(|j| {
                #[expect(
                    clippy::cast_possible_wrap,
                    reason = "j and i bounded by data.len() ≪ i32::MAX for chromatographic data"
                )]
                let offset = j as i32 - i as i32;
                f64::from(offset)
            })
            .collect();
        let coeffs = coefficients_for_x(&x_values, order)?;
        let mut sum = 0.0;
        for (k, &c) in coeffs.iter().enumerate() {
            sum = c.mul_add(data[left + k], sum);
        }
        out[i] = sum;
    }
    Ok(out)
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "test module: assertions use unwrap for clarity"
)]
#[expect(
    clippy::cast_precision_loss,
    reason = "test module: small array lengths fit f64"
)]
mod tests {
    use super::*;
    use crate::tolerances;

    #[test]
    fn identity_and_polynomial() {
        let c = vec![std::f64::consts::PI; 21];
        let out = savitzky_golay(&c, 5, 2).unwrap();
        assert!(
            c.iter()
                .zip(out.iter())
                .all(|(a, b)| (a - b).abs() < tolerances::ANALYTICAL_LOOSE)
        );
        let poly: Vec<f64> = (0..15_i32)
            .map(|i| f64::from(i).mul_add(2.0, 1.0) + f64::from(i * i))
            .collect();
        let out2 = savitzky_golay(&poly, 5, 2).unwrap();
        assert!(
            poly.iter()
                .zip(out2.iter())
                .all(|(a, b)| (a - b).abs() < tolerances::ANALYTICAL_LOOSE)
        );
    }

    #[test]
    fn noise_reduction() {
        let noisy: Vec<f64> = (0..50_i32)
            .map(|i| f64::from(i).mul_add(0.1, 0.5 * (f64::from(i) * 0.7).sin()))
            .collect();
        let smoothed = savitzky_golay(&noisy, 11, 3).unwrap();
        let var = |v: &[f64]| {
            let n = v.len() as f64;
            let m = v.iter().sum::<f64>() / n;
            v.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / n
        };
        assert!(var(&smoothed) < var(&noisy));
    }

    #[test]
    fn edge_cases_and_scipy() {
        assert_eq!(
            savitzky_golay(&[1.0, 2.0, 3.0], 1, 0).unwrap(),
            [1.0, 2.0, 3.0]
        );
        assert!(savitzky_golay(&[1.0, 2.0, 3.0], 4, 2).is_err());
        assert!(savitzky_golay(&[1.0, 2.0, 3.0], 7, 2).is_err());
        let out = savitzky_golay(&[2.0, 2.0, 5.0, 2.0, 1.0, 0.0, 1.0, 4.0, 9.0], 5, 2).unwrap();
        for (i, &e) in [3.54, 2.86, 0.66, 0.17].iter().enumerate() {
            assert!((out[i + 2] - e).abs() < 0.02);
        }
    }
}
