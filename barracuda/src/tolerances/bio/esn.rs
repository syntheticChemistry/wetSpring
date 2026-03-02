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
