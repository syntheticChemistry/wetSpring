// SPDX-License-Identifier: AGPL-3.0-or-later
//! Anderson / spectral theory (biological context) tolerances.

/// GPU bifurcation eigenvalue relative parity (Jacobian eigenvalues).
///
/// Jacobi eigendecomposition on GPU vs CPU yields eigenvalues that
/// differ by up to 3% in relative terms for near-zero eigenvalues
/// of the ODE Jacobian. 5% covers the worst observed case.
/// Validated: Exp050 (bifurcation eigenvalue), commit `e4358c5`.
pub const GPU_EIGENVALUE_REL: f64 = 0.05;

/// GPU Lanczos eigenvalue absolute parity (individual eigenvalues).
///
/// GPU Lanczos tridiagonalization followed by QR eigendecomposition
/// on large Anderson lattices (L ≥ 14, N = L³ ≥ 2744) produces
/// eigenvalues that differ from CPU reference by up to 0.02 absolute
/// due to `SpMV` summation order differences on GPU. 0.03 covers the
/// worst observed case with 50% margin.
/// Validated: Exp184b (GPU Anderson finite-size scaling), planned.
pub const GPU_LANCZOS_EIGENVALUE_ABS: f64 = 0.03;

/// Finite-size scaling `W_c` relative tolerance across lattice sizes.
///
/// The critical disorder `W_c` estimated from level spacing ratio
/// crossing points varies by up to 5% across lattice sizes L = 6–20
/// due to finite-size corrections. 8% covers the largest expected
/// deviation at L = 6 with margin.
/// Validated: Exp150 (L=6–12), Exp184b (L=14–20).
pub const FINITE_SIZE_SCALING_REL: f64 = 0.08;

/// Level spacing ratio ⟨r⟩ standard error tolerance for disorder averaging.
///
/// With 8 disorder realizations per (L, W) point, the standard error of
/// ⟨r⟩ should be below 0.015 (well-sampled regime). Values above this
/// indicate insufficient averaging or a phase boundary artifact.
/// Validated: Exp150 (8 realizations), Exp184b (16 realizations).
pub const LEVEL_SPACING_STDERR_MAX: f64 = 0.015;

/// 1D Anderson localization: weak-disorder ⟨r⟩ floor.
///
/// At W = 0.5 (weak disorder) on a 400-site 1D lattice, ⟨r⟩ should
/// exceed this threshold, staying in the GOE-like regime. Below it,
/// the system is already localized which contradicts 1D random-matrix
/// expectations at weak disorder.
/// Validated: Exp122 (`validate_anderson_2d_qs`, commit `756df26`, 2026-02-26).
/// Physics: Atas et al. PRL 2013, GOE ⟨r⟩ ≈ 0.5307.
pub const ANDERSON_1D_WEAK_DISORDER_FLOOR: f64 = 0.4;

/// 2D Anderson localization: strong-disorder ⟨r⟩ ceiling.
///
/// At W = 15.0 (strong disorder) on a 20×20 2D lattice, ⟨r⟩ should
/// stay below this threshold, confirming localization. Values above it
/// at strong disorder would indicate a numerical or lattice artifact.
/// Validated: Exp122 (`validate_anderson_2d_qs`, commit `756df26`, 2026-02-26).
/// Physics: Atas et al. PRL 2013, Poisson ⟨r⟩ ≈ 0.3863.
pub const ANDERSON_STRONG_DISORDER_CEILING: f64 = 0.45;

/// 2D Anderson localization: weak-disorder ⟨r⟩ floor (stricter than 1D).
///
/// At W = 0.5 on a 20×20 2D lattice, extended states should yield
/// ⟨r⟩ > 0.45 (closer to GOE). The 2D extended regime persists to
/// higher disorder than 1D.
/// Validated: Exp122 (`validate_anderson_2d_qs`, commit `756df26`, 2026-02-26).
pub const ANDERSON_2D_WEAK_DISORDER_FLOOR: f64 = 0.45;

/// Anderson QS model correlation comparison margin.
///
/// When comparing model correlations (e.g. H3 vs H2), allow H3 to be
/// up to 0.1 lower than H2 and still consider them "comparable."
/// Validated: `validate_anderson_qs_environments_v1` S3.
pub const MODEL_CORRELATION_MARGIN: f64 = 0.1;

/// Anderson QS model MAE comparison margin.
///
/// When comparing model MAE (e.g. H3 vs H2), allow H3 to exceed H2 by
/// up to 0.05 and still consider H3 "close" to H2.
/// Validated: `validate_anderson_qs_environments_v1` S6.
pub const MODEL_MAE_MARGIN: f64 = 0.05;
