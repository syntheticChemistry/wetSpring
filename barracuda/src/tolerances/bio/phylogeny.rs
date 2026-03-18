// SPDX-License-Identifier: AGPL-3.0-or-later
//! Phylogenetic and tree-related tolerances.

/// Minimum dS value for omega ratio computation.
///
/// When dS is below this threshold, omega (dN/dS) is undefined
/// because dividing by near-zero dS would produce an unstable ratio.
pub const DNDS_OMEGA_GUARD: f64 = 1e-10;

/// Phylogenetic log-likelihood (Felsenstein pruning): Python vs Rust.
///
/// Matrix exponentiation and per-site LL accumulation differ at ~1e-8
/// between `NumPy` and Rust due to loop ordering and FMA behavior.
pub const PHYLO_LIKELIHOOD: f64 = 1e-8;

/// JC69 transition probability comparisons.
///
/// Analytic formula P(t) = 0.25 + 0.75·exp(-4μt/3) rounding at 1e-6.
pub const JC69_PROBABILITY: f64 = 1e-6;

/// Stochastic ensemble mean tolerance (10% relative for Gillespie SSA).
///
/// With 10,000 replicates, the sample mean of a birth-death process
/// should be within ~10% of the analytical mean by CLT.
pub const GILLESPIE_MEAN_REL: f64 = 0.1;

/// Fano factor tolerance for birth-death process.
///
/// Analytical Fano = 1.0 for Poisson; stochastic ensemble variance
/// yields Fano within ±0.5 of theoretical at 10,000 replicates.
pub const GILLESPIE_FANO: f64 = 0.5;

/// Gillespie Python-range relative tolerance (15%).
///
/// Since Rust and Python use different PRNGs, we cannot match bitwise.
/// With N=1000 replicates, the Rust ensemble mean should be within 15%
/// of the analytical mean (same regime as the Python baseline).
/// Validated: `scripts/gillespie_baseline.py`, commit `e4358c5`.
pub const GILLESPIE_PYTHON_RANGE_REL: f64 = 0.15;

/// Fano factor physical range for birth-death validation.
///
/// The Fano factor must lie in \[0, 2\] for a near-Poisson process.
/// ±1.0 around the theoretical value of 1.0 covers the full physical
/// range. Values outside indicate a bug in variance accumulation.
pub const GILLESPIE_FANO_PHYSICAL: f64 = 1.0;

/// Bootstrap log-likelihood ensemble tolerance.
///
/// 100 bootstrap replicates produce mean LL near the original tree's LL.
/// Sampling variance across resampled alignments yields a spread of
/// ±5.0 log-likelihood units for typical 400-site alignments.
/// Validated: Exp031 (Wang 2021 RAWR), `scripts/wang2021_rawr_bootstrap.py`.
pub const BOOTSTRAP_LL_ENSEMBLE: f64 = 5.0;

/// Single-run SSA mean tolerance (absolute) for birth-death steady state.
///
/// A single Gillespie SSA trajectory of a birth-death process with
/// `k_b/k_d = 100` has large shot-noise variance. The steady-state mean
/// can deviate by ±30 from the analytical expectation. This is wider
/// than [`GILLESPIE_MEAN_REL`] (10% relative for 10 000-replicate
/// ensembles) because single runs lack CLT averaging.
/// Validated: Exp292 (`BarraCuda` CPU v22, D34 stochastic domain).
pub const SSA_SINGLE_RUN_ABSOLUTE: f64 = 30.0;

/// Bootstrap estimate tolerance for small-sample CI.
///
/// For `N = 10` data points with `mean ≈ 5.5`, 1 000 bootstrap replicates
/// produce a point estimate within ±0.5 of the sample mean. Wider than
/// jackknife-vs-bootstrap tolerance because bootstrap resampling introduces
/// additional sampling noise beyond jackknife leave-one-out.
/// Validated: Exp292 (`BarraCuda` CPU v22, D40 statistics domain).
pub const BOOTSTRAP_ESTIMATE_SMALL: f64 = 0.5;

/// GPU phylogenetic pruning relative tolerance.
///
/// Felsenstein pruning on GPU vs CPU: polynomial exp/log transcendental
/// fallback compounds errors across multi-step recursive pruning.
/// For balanced trees with 4+ tips, the relative error
/// `|CPU_LL − GPU_LL| / |CPU_LL|` stays below 10%.
/// Validated: Exp250 (streaming ODE + phylo), commit `5e6a00b`.
pub const PHYLO_GPU_RELATIVE: f64 = 0.10;
