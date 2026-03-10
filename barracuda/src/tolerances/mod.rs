// SPDX-License-Identifier: AGPL-3.0-or-later
//! Centralized validation tolerances with scientific justification.
//!
//! Every tolerance threshold used in validation binaries is defined here
//! with documentation of its origin. No ad-hoc magic numbers.
//!
//! # Tolerance categories
//!
//! | Category | Basis | Example |
//! |----------|-------|---------|
//! | Exact | IEEE 754 f64 | 0.0 for integer counts |
//! | Machine | f64 arithmetic | 1e-12 for analytical formulas |
//! | Instrument | Measurement error | 0.005 for GC content |
//! | GPU vs CPU | Instruction reorder | 1e-6 for GPU f64 vs CPU f64 |
//!
//! ```
//! use wetspring_barracuda::tolerances;
//!
//! // Analytical formulas should be within machine precision
//! let h = 4.0_f64.ln();
//! assert!((h - 4.0_f64.ln()).abs() <= tolerances::ANALYTICAL_F64);
//! ```

mod bio;
pub use bio::*;

mod instrument;
pub use instrument::*;

mod gpu;
pub use gpu::*;

mod spectral;
pub use spectral::*;

// ═══════════════════════════════════════════════════════════════════
// Machine-precision tolerances (IEEE 754 f64)
// ═══════════════════════════════════════════════════════════════════

/// Operations that must be exact (integer counts, file counts).
pub const EXACT: f64 = 0.0;

/// Analytical formulas with minimal f64 rounding (Shannon, Simpson).
///
/// f64 has ~15.9 significant digits; 1e-12 allows 3 digits of
/// accumulated rounding in simple arithmetic chains.
pub const ANALYTICAL_F64: f64 = 1e-12;

/// Loose analytical tolerance for test assertions.
///
/// Used when comparing computed values to known analytical results
/// where intermediate arithmetic (division, sqrt, ln) accumulates
/// more rounding than pure add/multiply chains. Covers ~5 digits
/// of accumulated error. Tighter than instrument tolerances, looser
/// than [`ANALYTICAL_F64`].
pub const ANALYTICAL_LOOSE: f64 = 1e-10;

/// Limit convergence tolerance for bounded values approaching
/// theoretical limits (0 or 1) with residual from polynomial
/// approximation or finite-sample effects.
///
/// Covers: `norm_cdf(large_x) → 1`, neutral drift `P_fix → 1/N`,
/// eigenvalue positive-semidefiniteness checks. Tighter than
/// [`ERF_PARITY`] (5e-7), looser than [`ANALYTICAL_LOOSE`] (1e-10).
pub const LIMIT_CONVERGENCE: f64 = 1e-8;

/// Variance of exactly constant data.
///
/// Theoretically zero, but generalized jackknife delete-1 can
/// accumulate O(N) cancellation terms. Stricter than [`EXACT_F64`]
/// (1e-15) to catch genuine non-zero variance while allowing
/// extended-precision rounding.
pub const VARIANCE_EXACT: f64 = 1e-20;

/// NMF matrix sparsity threshold for counting near-zero entries.
///
/// Elements below this are considered effectively zero when computing
/// W/H sparsity ratios. Between [`LIMIT_CONVERGENCE`] (1e-8) and
/// [`ANALYTICAL_LOOSE`] (1e-10) to avoid counting numerical noise
/// as non-zero structure.
pub const NMF_SPARSITY_THRESHOLD: f64 = 1e-8;

/// NMF iteration convergence tolerance.
///
/// Iteration stops when the relative change in reconstruction error
/// drops below this threshold. 1e-4 provides ~4 significant digits
/// of accuracy while avoiding excessive iterations.
pub const NMF_CONVERGENCE: f64 = 1e-4;

// ═══════════════════════════════════════════════════════════════════
// Special-function evaluation tolerances
// ═══════════════════════════════════════════════════════════════════

/// Error function (erf) evaluation tolerance vs reference values.
///
/// `erf(x)` computed via Horner polynomial agrees with Abramowitz & Stegun
/// table values to ~5e-7 absolute. Covers both CPU and GPU `erf` polyfills.
/// Validated: Exp179 (Track 4 QS pore geometry), Exp183 (no-till meta-analysis).
pub const ERF_PARITY: f64 = 5e-7;

/// Normal CDF Φ(x) evaluation tolerance for non-extreme arguments.
///
/// `norm_cdf(x)` for |x| < 4 agrees with Python `scipy.stats.norm.cdf`
/// to ~1e-3 absolute due to erf polynomial approximation differences.
/// For extreme arguments (|x| > 6), use [`PYTHON_PARITY`] (1e-10).
/// Validated: Exp179/183 (Anderson disorder mapping, soil QS).
pub const NORM_CDF_PARITY: f64 = 1e-3;

/// Normal CDF Φ(x) evaluation tolerance for CDF tail values.
///
/// `norm_cdf(x)` for |x| ~ 3 (tail region, CDF ~ 0.9987) retains ~4
/// significant digits. 1e-4 covers polynomial approximation drift.
/// Validated: Exp185 (soil biofilm aggregate).
pub const NORM_CDF_TAIL: f64 = 1e-4;

/// Normal PPF (percent point function) tolerance for known quantile values.
///
/// `norm_ppf(0.975)` ≈ 1.96 and `norm_ppf(0.025)` ≈ −1.96. Newton-Raphson
/// inversion of the CDF polynomial yields ~0.01 absolute error for these
/// standard quantiles.
/// Validated: `special` `norm_ppf_known_quantiles`.
pub const NORM_PPF_KNOWN: f64 = 0.01;

// ═══════════════════════════════════════════════════════════════════
// Jacobi eigendecomposition (PCoA)
// ═══════════════════════════════════════════════════════════════════

/// Jacobi convergence: sum of squared off-diagonal elements.
///
/// Iteration stops when the off-diagonal Frobenius norm drops below this
/// threshold. 1e-24 is ~(1e-12)² — squaring machine epsilon ensures
/// eigenvalues are accurate to full f64 precision even for ill-conditioned
/// centering matrices. Reference: Golub & Van Loan (2013), Matrix
/// Computations, 4th ed. §8.5.2 — "classical Jacobi method."
pub const JACOBI_CONVERGENCE: f64 = 1e-24;

/// Jacobi element skip: off-diagonal elements below this are treated as zero.
///
/// Avoids unnecessary Givens rotations for near-zero elements.
/// `1e-15` ≈ `f64::EPSILON` × 1000, allowing for accumulated rounding in
/// the centering step. Standard practice per Golub & Van Loan §8.5.
pub const JACOBI_ELEMENT_SKIP: f64 = 1e-15;

/// Jacobi tau overflow guard for Givens rotation parameter.
///
/// When `|τ| = |(a_qq - a_pp) / (2·a_pq)|` exceeds this threshold, the
/// tangent approximation `t ≈ 1/(2τ)` is used instead of the full formula
/// to avoid f64 overflow in `τ² + 1`. `1e15` is conservative — overflow
/// occurs at ~`1.34e154` for f64. Reference: Golub & Van Loan §8.5.2.
pub const JACOBI_TAU_OVERFLOW: f64 = 1e15;

/// Jacobi maximum sweeps multiplier.
///
/// Total sweeps = this × N for an N×N matrix. Classical Jacobi converges
/// quadratically after O(N²) rotations; 100×N is generous for the matrix
/// sizes in `PCoA` (typically N < 10,000). Reference: Golub & Van Loan §8.5.
pub const JACOBI_SWEEP_MULTIPLIER: usize = 100;

// ═══════════════════════════════════════════════════════════════════
// Numerical stability guards
// ═══════════════════════════════════════════════════════════════════

/// Division-by-zero guard for matrix operations (NMF, cosine, norms).
///
/// Applied as a denominator floor to prevent NaN/Inf. Smaller than
/// `f64::EPSILON` (2.2e-16) so it never distorts non-degenerate results.
/// Also used as the denominator guard in ESN Cholesky fallback and
/// pangenome linear regression.
pub const MATRIX_EPS: f64 = 1e-15;

/// Box-Muller u1 floor: avoids `ln(0)` in Gaussian generation.
///
/// The Box-Muller transform computes `sqrt(-2 ln(u1))`, which diverges
/// at u1 = 0. Clamping to this floor bounds the output to ~8.3σ.
pub const BOX_MULLER_U1_FLOOR: f64 = 1e-15;

/// Regularized gamma right-tail early exit threshold.
///
/// When `x > a + GAMMA_RIGHT_TAIL_OFFSET`, P(a,x) ≈ 1.0 to f64
/// precision. Avoids expensive series summation in the deep right tail.
pub const GAMMA_RIGHT_TAIL_OFFSET: f64 = 200.0;

/// ODE division-by-zero guard for WGSL and CPU derivatives.
///
/// Applied in Hill function denominators and Monod kinetics to prevent
/// NaN when concentrations are near zero. Small enough to not affect
/// biologically meaningful concentration ranges (typically > 1e-9).
pub const ODE_DIVISION_GUARD: f64 = 1e-30;

/// Error message truncation length for API responses.
pub const ERROR_BODY_PREVIEW_LEN: usize = 200;

/// Regularized incomplete gamma series convergence epsilon.
///
/// Used by `special::regularized_gamma_p` for series termination.
/// Matches `scipy`'s `gammainc` convergence behavior at 1e-15.
pub const GAMMA_SERIES_CONVERGENCE: f64 = 1e-15;

/// Maximum iterations for regularized gamma series expansion.
pub const GAMMA_SERIES_MAX_ITER: usize = 1000;

// ═══════════════════════════════════════════════════════════════════
// ODE integration parameters
// ═══════════════════════════════════════════════════════════════════

/// Default RK4 time step for ODE integration.
///
/// dt = 0.001 provides sub-percent accuracy for the stiff multi-species
/// systems (QS biofilm, capacitor, phage defense) while keeping step
/// counts reasonable. Validated against `scipy.integrate.odeint` (LSODA)
/// across all 6 ODE models (Exp020/023/024/025/027/030).
pub const ODE_DEFAULT_DT: f64 = 0.001;

// ═══════════════════════════════════════════════════════════════════
// DF64 / streaming / performance
// ═══════════════════════════════════════════════════════════════════

/// DF64 host pack/unpack roundtrip tolerance (double-double precision).
///
/// DF64 wire format preserves ~26 significant digits; 1e-13 covers
/// typical GEMM output roundtrip error on streaming data.
/// Validated: Exp227 (Pure GPU Streaming v4), Exp228 (metalForge v8).
pub const DF64_ROUNDTRIP: f64 = 1e-13;

/// GEMM pipeline compilation timeout (milliseconds).
///
/// The GPU GEMM shader (via hotSpring f64 polyfills) must compile in
/// under 30 seconds on supported hardware. Exceeding this indicates
/// a driver or shader complexity regression.
pub const GEMM_COMPILE_TIMEOUT_MS: f64 = 30_000.0;

#[cfg(test)]
mod tests;
