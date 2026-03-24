// SPDX-License-Identifier: AGPL-3.0-or-later
//! ESN (Echo State Network) tolerances.

/// ESN ridge regression regularisation (Tikhonov λ).
///
/// Jaeger (2001) "The echo state approach" recommends λ ∈ [1e-8, 1e-2].
/// Default 1e-6 balances numerical stability with fitting accuracy,
/// matching the common practice in Lukoševičius (2012) "A Practical
/// Guide to Applying Echo State Networks."
pub const ESN_REGULARIZATION: f64 = 1e-6;

/// ESN ridge regularization for higher-capacity reservoirs.
///
/// Bloom Sentinel and Disorder classifiers use 1e-5 for slightly
/// tighter fitting when reservoir size and connectivity allow.
/// Validated: Exp194 (NPU Live — Exp118, Exp119).
pub const ESN_REGULARIZATION_TIGHT: f64 = 1e-5;

/// ESN Gaussian output mean tolerance (Box-Muller / LCG tests).
///
/// Sample mean of 10,000 Gaussian draws from the LCG-based generator
/// should be near zero. 0.1 covers finite-sample sampling error
/// (σ/√n ≈ 0.01 for n=10k) with margin for LCG bias.
/// Validated: `esn/tests` `lcg_gaussian_mean_near_zero`.
pub const ESN_GAUSSIAN_MEAN: f64 = 0.1;

/// ESN spectral radius scaling tolerance.
///
/// Max `|w_res|` after scaling should equal the configured spectral radius.
/// 0.01 covers rounding in the scaling chain (eigendecomposition,
/// rescaling by `ρ/λ_max`).
/// Validated: `esn/tests` `spectral_radius_scaling`.
pub const ESN_SPECTRAL_RADIUS: f64 = 0.01;

/// ESN prediction reasonableness ceiling for simple identity-like tasks.
///
/// A reservoir computing a near-identity function (sum of 2 inputs,
/// 50 training steps) produces predictions within ±10 of the target.
/// This is a smoke test ceiling, not a precision benchmark — it confirms
/// the train/predict pipeline produces bounded, non-degenerate output.
/// Validated: `esn/tests` `bio_esn_train_and_predict`.
pub const ESN_PREDICTION_REASONABLE: f64 = 10.0;
