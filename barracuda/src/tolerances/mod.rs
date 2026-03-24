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

/// NMF iteration convergence tolerance (IPC handler).
///
/// Iteration stops when the absolute change in reconstruction error
/// drops below this threshold. 1e-6 provides ~6 significant digits
/// of accuracy for the multiplicative-update algorithm (Lee & Seung 2001).
/// Tighter than [`NMF_CONVERGENCE_LOOSE`] for science-grade results.
pub const NMF_CONVERGENCE: f64 = 1e-6;

/// NMF convergence tolerance (loose).
///
/// Used in contexts where fast convergence is preferred over
/// precision. 1e-4 provides ~4 significant digits.
pub const NMF_CONVERGENCE_LOOSE: f64 = 1e-4;

// ═══════════════════════════════════════════════════════════════════
// Special-function evaluation tolerances
// ═══════════════════════════════════════════════════════════════════

/// Error function (erf) evaluation tolerance vs reference values.
///
/// `erf(x)` computed via Horner polynomial agrees with Abramowitz & Stegun
/// table values to ~5e-7 absolute. Covers both CPU and GPU `erf` polyfills.
/// Validated: Exp179 (Track 4 QS pore geometry), Exp183 (no-till meta-analysis).
pub const ERF_PARITY: f64 = 5e-7;

/// Normal CDF symmetry check: `Φ(x) + Φ(−x) = 1`.
///
/// Polynomial approximation preserves odd symmetry of erf(x) to ~1e-7.
/// Validated: `special` `norm_cdf_symmetry`.
pub const NORM_CDF_SYMMETRY: f64 = 1e-7;

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

/// Stable special functions (log1p, expm1) near-machine precision for tiny x.
///
/// log1p(1e-15) ≈ 1e-15 and expm1(1e-15) ≈ 1e-15 to within 1e-28 absolute.
/// Used to verify stable implementations avoid catastrophic cancellation
/// at sub-epsilon arguments. 1e-28 is ~(1e-14)² — squared machine epsilon.
/// Validated: `validate_stable_specials_v1` (D80).
pub const STABLE_SPECIAL_TINY: f64 = 1e-28;

/// Stable `ln1p(x) ≈ x` and `expm1(x) ≈ x` test tolerance for |x| ~ 1e-15.
///
/// IEEE 754 `ln_1p` and `exp_m1` recover the identity `f(x) ≈ x` for tiny x
/// with error bounded by x² / 2 ≈ 5e-31. We use 1e-25 (four orders above
/// the theoretical bound) as a practical ceiling that accommodates any
/// conforming implementation. Covers `numerics::stable_ln1p` and
/// `stable_expm1` test assertions.
pub const STABLE_IDENTITY_TINY: f64 = 1e-25;

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

/// Strict DF64 roundtrip tolerance for bit-exact protocol validation.
///
/// Tighter than [`DF64_ROUNDTRIP`] for testing the DF64 host wire format
/// itself (encode→decode) rather than end-to-end GEMM pipelines.
pub const DF64_ROUNDTRIP_STRICT: f64 = 1e-14;

/// DF64 small-value pack/unpack roundtrip precision.
///
/// For values near 1e-10 the DF64 hi/lo split preserves ~30 significant
/// digits. 1e-25 is the empirical ceiling from `df64_host` tests at that
/// magnitude. Tighter than [`STABLE_SPECIAL_TINY`] (1e-28) because the
/// round-trip includes an f32 truncation step that widens error for
/// non-zero values vs the sub-epsilon regime of `log1p`/`expm1`.
pub const DF64_SMALL_VALUE_ROUNDTRIP: f64 = 1e-25;

/// GEMM pipeline compilation timeout (milliseconds).
///
/// The GPU GEMM shader (via hotSpring f64 polyfills) must compile in
/// under 30 seconds on supported hardware. Exceeding this indicates
/// a driver or shader complexity regression.
pub const GEMM_COMPILE_TIMEOUT_MS: f64 = 30_000.0;

// ═══════════════════════════════════════════════════════════════════
// Pharmacological / dose-response
// ═══════════════════════════════════════════════════════════════════

/// Pharmacokinetic and IC50 dose-response parity tolerance.
///
/// Published IC50 values (e.g., Gonzales 2014: JAK1 IC50 = 10 nM),
/// onset times, and selectivity ratios are reproduced within 0.1
/// absolute units. Covers rounding in Hill equation fitting and
/// log-space IC50 conversion.
/// Validated: Exp280 (Gonzales IC50), Exp281 (Gonzales PK).
pub const PHARMACOKINETIC_PARITY: f64 = 0.1;

/// IC50 response-at-midpoint tolerance.
///
/// At the IC50 concentration, the Hill equation predicts exactly 50%
/// inhibition. 0.01 covers numerical drift in the Hill sigmoid
/// evaluation at the inflection point.
/// Validated: Exp280 (Gonzales IC50 dose-response).
pub const IC50_RESPONSE_TOL: f64 = 0.01;

/// Regression parameter fit parity tolerance.
///
/// Nonlinear regression (Levenberg-Marquardt, Newton) parameter
/// estimates should match published values within 0.01 when fitting
/// to the same data. Covers optimizer convergence differences.
/// Validated: Exp283 (Gonzales CPU parity).
pub const REGRESSION_FIT_PARITY: f64 = 0.01;

// ═══════════════════════════════════════════════════════════════════
// Hormesis / binding landscape
// ═══════════════════════════════════════════════════════════════════

/// Hormetic peak detection threshold.
///
/// A dose-response point is in the hormetic zone when the composite
/// response exceeds baseline by at least this margin. Set at 1% to
/// distinguish genuine stimulation from numerical noise while capturing
/// subtle hormetic effects reported in Calabrese & Mattson (2017).
pub const HORMESIS_PEAK_MARGIN: f64 = 0.01;

/// Colonization resistance threshold.
///
/// Fraction of epithelial binding sites that must be occupied for the
/// lattice to be classified as "resistant" to pathogen invasion. The
/// 90% threshold follows the competitive exclusion principle: pathogens
/// need unoccupied niches to establish, and 90% occupancy leaves
/// insufficient niche space.
/// Reference: healthSpring exp098 (colonization resistance surface).
pub const COLONIZATION_RESISTANCE_THRESHOLD: f64 = 0.9;

/// Binding delocalization IPR threshold.
///
/// Inverse participation ratio below this value indicates delocalized
/// binding (load spread across tissues). From healthSpring TOXICITY.md:
/// IPR < 0.15 corresponds to ξ > 6.7 tissues sharing the burden,
/// which keeps per-tissue occupancy below repair capacity thresholds.
pub const BINDING_IPR_DELOCALIZED: f64 = 0.15;

/// Composite binding activation floor.
///
/// Minimum composite binding score (product of fractional occupancies)
/// required to register as a biological signal. Below this, the
/// coincidence detection model treats the composite as noise.
pub const COMPOSITE_BINDING_FLOOR: f64 = 1e-6;

// ═══════════════════════════════════════════════════════════════════
// Bootstrap / rarefaction
// ═══════════════════════════════════════════════════════════════════

/// Asymptotic convergence tolerance for limit-approaching checks.
///
/// Covers tests of the form `f(large_x) ≈ f(∞)`, where the residual
/// comes from finite evaluation point rather than numerical error.
/// Examples: `first_order(200, B_max, k) ≈ B_max`, `monod(50000, mu_max, Ks) ≈ mu_max`.
/// Validated: Exp020 (QS ODE), Exp039 (fungal kinetics).
pub const ASYMPTOTIC_LIMIT: f64 = 0.01;

/// Wide tolerance for visualization scenario smoke tests.
///
/// Visualization validators confirm that real math feeds into petalTongue
/// scenarios without requiring baseline-parity precision. 0.5 accommodates
/// intentionally approximate "visual plausibility" checks.
/// Validated: Exp233 (petalTongue live v1), Exp241 (cross-primal viz).
pub const VISUALIZATION_SCENARIO: f64 = 0.5;

/// Tighter visualization tolerance for structural comparisons.
///
/// Used when visualization smoke tests compare values that should be
/// close (e.g., soil Shannon > algae Shannon, expected ~2.0 vs ~1.55)
/// but do not need baseline parity.
pub const VISUALIZATION_SCENARIO_TIGHT: f64 = 0.2;

/// End-to-end pipeline IPC roundtrip tolerance.
///
/// Wider than [`IPC_JSON_ROUNDTRIP`] (1e-6) because the full pipeline
/// includes dispatch, handler computation, and JSON serialization. The
/// 0.01 margin covers cumulative numerical drift through the pipeline.
/// Validated: Exp298 (cross-primal pipeline v98).
pub const PIPELINE_E2E: f64 = 0.01;

/// Bootstrap Shannon mean tolerance (GPU extended rarefaction).
///
/// Bootstrap resampling (1,000 replicates) of Shannon diversity produces
/// a mean that should be within 0.5 of the CPU point estimate. The margin
/// covers stochastic variation from multinomial resampling at small N.
/// Validated: Exp101 (GPU extended validation).
pub const RAREFACTION_BOOTSTRAP_SHANNON: f64 = 0.5;

#[cfg(test)]
mod tests;
