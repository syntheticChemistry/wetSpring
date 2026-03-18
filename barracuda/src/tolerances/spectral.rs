// SPDX-License-Identifier: AGPL-3.0-or-later
//! Anderson localization, spectral theory, and dynamic disorder tolerances.
//!
//! Track 4 (soil QS ↔ Anderson localization analogy) validation constants
//! for level spacing ratios, Lyapunov exponents, spectrum bounds, and
//! dynamic W(t) function checks.

/// Dynamic W(t) function evaluation tolerance for exact-value checks.
///
/// W(t) at known points (t=0, t=-1) should match the model formula exactly
/// up to f64 rounding. 0.01 covers accumulated rounding in `exp()`/`sin()`
/// evaluations for these analytically exact evaluation points.
/// Validated: Exp186 (Dynamic Anderson W(t)), Exp190 (CPU v10).
pub const DYNAMIC_WT_EXACT: f64 = 0.01;

/// Dynamic W(t) asymptotic evaluation tolerance.
///
/// W(t) at large t (exponential decay, seasonal average) approaches the
/// asymptotic value with residual transient. 0.1 covers the transient
/// tail for `exp(-kt)` decay models at t ≥ 100.
/// Validated: Exp186 (Dynamic Anderson W(t)), Exp190 (CPU v10).
pub const DYNAMIC_WT_ASYMPTOTIC: f64 = 0.1;

/// Periodicity tolerance for oscillating W(t) diagnostics.
///
/// For periodic W(t), the mean over the first half-period should equal the
/// mean over the second half-period within this tolerance. 0.05 covers
/// finite-sample averaging error for sinusoidal W(t) on discrete grids.
/// Validated: Exp186 (Dynamic Anderson W(t)).
pub const DYNAMIC_WT_PERIODICITY: f64 = 0.05;

/// Seasonal oscillation amplitude tolerance for periodic W(t) functions.
///
/// Seasonal W(t) = `W_0` + A·sin(2πt/365) evaluated at quarter-periods
/// produces ±A offsets. With A=4 and `W_0`=16, a tolerance of 0.5 covers
/// numerical imprecision in `sin()` at non-exact multiples of π/2.
/// Validated: Exp186 (Dynamic Anderson W(t)), Exp190 (CPU v10).
pub const SEASONAL_OSCILLATION: f64 = 0.5;

/// Poisson level spacing ratio tolerance for spectral cross-spring validation.
///
/// At strong disorder (W >> `W_c`), ⟨r⟩ should approach Poisson (0.3863).
/// Tolerance of 0.06 covers finite-size fluctuations and disorder averaging
/// with 8 realizations on lattices L = 6–14.
/// Validated: Exp107 (cross-spring spectral theory), commit `756df26`.
pub const SPECTRAL_POISSON_PARITY: f64 = 0.06;

/// Lyapunov exponent tolerance for localization diagnostics.
///
/// Normalized Lyapunov exponent γL should be > 0 in the localized phase.
/// Tolerance of 0.03 covers transfer-matrix convergence for chain lengths
/// ≥ 10,000 sites at moderate disorder.
/// Validated: Exp107 (cross-spring spectral theory), commit `756df26`.
pub const SPECTRAL_LYAPUNOV_PARITY: f64 = 0.03;

/// Level spacing ratio ⟨r⟩ margin for phase identification.
///
/// When checking ⟨r⟩ against Poisson or GOE reference values, this margin
/// accounts for finite-size fluctuations and disorder averaging. Used as
/// a symmetric offset: `r > POISSON_R - margin` or `r > GOE_R - margin`.
/// Validated: Exp107 (cross-spring spectral theory), commit `756df26`.
pub const SPECTRAL_R_MARGIN: f64 = 0.02;

/// Gershgorin spectrum bound tolerance for eigenvalue containment checks.
///
/// Eigenvalues of the Anderson Hamiltonian should lie within the Gershgorin
/// circle `[-W/2 - 2d, W/2 + 2d]`. Tolerance of 0.5 covers rounding in
/// tridiagonal eigenvalue computation and edge effects at the spectrum boundary.
/// Validated: Exp107 (cross-spring spectral theory), commit `756df26`.
pub const SPECTRAL_GERSHGORIN_MARGIN: f64 = 0.5;

/// Almost-Mathieu spectrum bound tolerance.
///
/// For the Aubry–André (almost-Mathieu) model, eigenvalues should lie
/// within `[-2-λ, 2+λ]`. Tolerance of 0.01 covers finite-chain boundary
/// effects for chains ≥ 100 sites.
/// Validated: Exp107 (cross-spring spectral theory), commit `756df26`.
pub const SPECTRAL_ALMOST_MATHIEU_MARGIN: f64 = 0.01;

/// Aubry–André extended-phase Lyapunov exponent tolerance.
///
/// In the extended phase (λ < 2), the normalized Lyapunov exponent γL
/// should be near zero (extended wavefunctions). 0.05 covers transfer-matrix
/// convergence at chain lengths ≥ 10,000 and λ values away from the
/// transition (λ ≈ 0.5).
/// Validated: Exp107 (cross-spring spectral theory), commit `756df26`.
pub const SPECTRAL_EXTENDED_LYAPUNOV: f64 = 0.05;

/// Herman–Avila–Bochi cocycle tolerance for spectral diagnostics.
///
/// The Herman bound on Lyapunov exponents provides a lower bound on the
/// localization rate. Tolerance of 0.02 covers polynomial expansion drift
/// in the cocycle integral for moderate disorder.
/// Validated: Exp107 (cross-spring spectral theory), commit `756df26`.
pub const SPECTRAL_HERMAN_PARITY: f64 = 0.02;

/// Trapezoidal integration tolerance for coarse grids (N = 100).
///
/// `barracuda::numerical::trapz` on N = 100 points for smooth functions
/// (e.g., x² on \[0,1\]) introduces discretization error ∝ h² ≈ 1e-4.
/// 1e-6 covers typical analytical targets with margin.
/// Validated: Exp169/183 (cross-spring modern/S65), commit `756df26`.
pub const TRAPZ_COARSE: f64 = 1e-6;

/// Trapezoidal integration tolerance for N = 101 points (x² on \[0,1\]).
///
/// Discretization error ∝ h² ≈ 1e-4 for ∫₀¹ x² dx with 101 points.
/// Validated: Exp169 (`benchmark_cross_spring_modern`).
pub const TRAPZ_101: f64 = 1e-4;

/// Lanczos vs Sturm eigenvalue comparison tolerance.
///
/// Lanczos tridiagonalization with k=60 vectors and Sturm bisection
/// produce eigenvalue estimates that differ by up to ~0.3 for Anderson
/// lattices at L=10 (N=1000). 0.5 covers the worst-case algorithmic
/// discrepancy with margin.
/// Validated: Exp107 (cross-spring spectral theory).
pub const LANCZOS_VS_STURM: f64 = 0.5;

/// Cross-spring numerical function parity (trapz, pearson, ridge).
///
/// Verifying shared barracuda primitives (trapezoidal integration,
/// Pearson correlation, `norm_cdf`) against known analytical values.
/// 1e-3 covers polynomial approximation drift in `norm_cdf` and
/// discretization error in trapz for coarse grids (N=100).
/// Validated: Exp183 (cross-spring S65), Exp169 (cross-spring modern).
pub const CROSS_SPRING_NUMERICAL: f64 = 1e-3;

/// Soil distance colonization minimum difference threshold.
///
/// Used to confirm that biofilm formation varies meaningfully with
/// distance from the QS source. 0.001 is the minimum absolute
/// difference in biofilm concentration between near-source and
/// far-source positions that constitutes "spatial variation."
/// Validated: Exp181 (soil distance colonization), commit `756df26`.
pub const SOIL_DISTANCE_MIN_DIFF: f64 = 0.001;

/// Cooperation frequency affected-group threshold.
///
/// In QS-distance colonization models, a cooperation frequency shift
/// greater than 0.005 from baseline indicates the colony is "affected"
/// by the QS gradient. Below this, the group is considered unaffected.
/// Validated: Exp181 (soil distance colonization), commit `756df26`.
pub const SOIL_COOP_FREQ_AFFECTED: f64 = 0.005;

/// Soil physics model approximation tolerance.
///
/// Used for soil-specific model comparisons: chemotaxis reduction of
/// effective `W_c` (~15%), carbon enrichment ratios (Brandt farm 2.15×),
/// and critical colonization distance estimates. These are inherently
/// approximate models with ~5% uncertainty from parameter fitting.
/// Validated: Exp170–182 (Track 4 soil QS).
pub const SOIL_MODEL_APPROX: f64 = 0.1;

/// Tillage effect tolerance on `P(QS)` and model ratios.
///
/// No-till vs conventional tillage comparisons (P(QS) difference,
/// receiver:inhibitor ratio) are inherently approximate due to
/// community-level variability. 0.2 covers inter-sample spread in
/// meta-analysis data (Zuber & Villamil 2016, Islam 2014).
/// Validated: Exp175 (KBS LTER), Exp291 (paper math control v4).
pub const SOIL_QS_TILLAGE: f64 = 0.2;

/// Anderson localization exponent ν fitting tolerance.
///
/// Critical exponent ν from finite-size scaling fits to Anderson
/// lattice data. Literature ν ≈ 1.57 (3D) with numerical fitting
/// uncertainty ~0.3 from lattice-size and disorder-averaging limitations.
/// Validated: Exp142 (DF64 Anderson).
pub const ANDERSON_NU_PARITY: f64 = 0.4;

/// 3D vs 2D geometry ratio tolerance for level-spacing and
/// cross-species correlation comparisons.
///
/// Dimensional comparisons (e.g., 3D vent vs 2D surface, canine vs
/// human epidermis) carry inherent model uncertainty from the mapping
/// between biological structure and effective lattice dimension.
/// Validated: Exp144 (vent chimney QS), Exp275 (heterogeneity sweep).
pub const GEOMETRY_DIMENSIONAL_PARITY: f64 = 0.15;

/// FAO-56 reference evapotranspiration cross-spring parity.
///
/// Cross-spring ET₀ computation (airSpring → wetSpring bridge) matches
/// to within 0.15 due to different numerical integration paths for the
/// Penman-Monteith equation (airSpring uses Brent optimization).
/// Validated: Exp157 (cross-spring S86).
pub const FAO56_ET0_PARITY: f64 = 0.15;

/// Regression intercept near-zero tolerance.
///
/// Linear regression fits where the true intercept is zero (e.g.,
/// proportional relationships) produce residual intercepts from
/// finite-sample effects. 0.5 covers typical scatter in biological
/// datasets with N < 100 samples.
/// Validated: Exp157 (cross-spring S86).
pub const INTERCEPT_NEAR_ZERO: f64 = 0.5;

/// Minimum spectral change for Anderson treatment effect detection.
///
/// When comparing level-spacing ratio r before/after pharmacological
/// intervention (e.g., JAK inhibitor, anti-IL-31), a change greater
/// than this threshold indicates the treatment had a measurable
/// effect on the Anderson regime. 0.001 distinguishes genuine
/// spectral shifts from finite-size fluctuations.
/// Validated: Exp273 (skin Anderson s79), Paper 12 immunological model.
pub const SPECTRAL_TREATMENT_EFFECT_MIN: f64 = 0.001;

/// Level-spacing ratio valid-range margin for pipeline assertions.
///
/// When checking that a computed ⟨r⟩ falls within `[POISSON_R − margin,
/// GOE_R + margin]`, this symmetric margin accommodates finite-size
/// lattice fluctuations and disorder-averaging noise in real-data
/// pipelines (NCBI, cold seep). Wider than [`SPECTRAL_R_MARGIN`] (0.02)
/// which targets clean phase identification in controlled sweeps.
/// Validated: Exp185 (NCBI real pipeline), commit `756df26`.
pub const SPECTRAL_R_PIPELINE_MARGIN: f64 = 0.05;
