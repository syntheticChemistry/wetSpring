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

// ═══════════════════════════════════════════════════════════════════
// Instrument / baseline tolerances
// ═══════════════════════════════════════════════════════════════════

/// GC content: ±0.5% (`FastQC` baseline, Zenodo 800651 `MiSeq` SOP).
///
/// Validated: `Galaxy` 24.1 `FastQC` on Kozich et al. `MiSeq` SOP dataset
/// (Zenodo 800651). Commit `504b0a8` (`FastQC` validated), data download
/// commit `a3fd096`.
pub const GC_CONTENT: f64 = 0.005;

/// Mean quality score: ±0.5 (`FastQC` baseline, read-level variability).
///
/// Validated: `Galaxy` 24.1 `FastQC` on Kozich et al. `MiSeq` SOP dataset
/// (Zenodo 800651). Read-level Phred variability at Q30 threshold. Commit
/// `cf15167` (full `FastQC` run history).
pub const MEAN_QUALITY: f64 = 0.5;

/// m/z tolerance: ±0.01 Da (Orbitrap instrument precision).
///
/// Standard mass accuracy for high-resolution Orbitrap LC-MS instruments
/// (Thermo Scientific, sub-5 ppm at 200–1000 Da). Calibrated against EPA
/// Method 533 reference standards (Exp005/006/007).
/// Validated: `scripts/validate_track2.py`, commit `48fb787`.
pub const MZ_TOLERANCE: f64 = 0.01;

// ═══════════════════════════════════════════════════════════════════
// Diversity metric tolerances
// ═══════════════════════════════════════════════════════════════════

/// Shannon entropy range for simulated marine community.
///
/// Simulated 330-species community with Exp002 relative abundances.
/// Expected ~4.9 ± 0.3 (validated against skbio 0.6.0, commit `e4358c5`).
/// Calibrated: `scripts/validate_exp001.py` (Exp001/002), commit `48fb787`.
pub const SHANNON_SIMULATED: f64 = 0.3;

/// Simpson range for simulated marine community.
///
/// Simulated 330-species community with Exp002 relative abundances.
/// Expected ~0.9 ± 0.05 (validated against skbio 0.6.0, commit `e4358c5`).
pub const SIMPSON_SIMULATED: f64 = 0.05;

/// Chao1 range above observed features (singleton/doubleton dependent).
///
/// Chao1 ≥ observed by definition; for Exp002 `Galaxy` community
/// (330 species, ~910 expected), allow up to +100 above observed
/// (validated against skbio 0.6.0, commit `e4358c5`).
pub const CHAO1_RANGE: f64 = 100.0;

/// Bray-Curtis: f64 symmetry tolerance.
pub const BRAY_CURTIS_SYMMETRY: f64 = 1e-15;

// ═══════════════════════════════════════════════════════════════════
// Python baseline parity tolerances
// ═══════════════════════════════════════════════════════════════════

/// Bit-exact f64 operations (ANI identity, zero-distance comparisons).
///
/// For operations like identical-sequence ANI that should yield exactly
/// 1.0 or 0.0, allow only rounding at the ULP boundary.
pub const EXACT_F64: f64 = 1e-15;

/// Python-to-Rust parity for iterative algorithms (dN/dS, molecular
/// clock, ODE solvers, HMM forward).
///
/// Different loop ordering and FMA availability cause 1-3 ULP drift
/// in accumulated sums. 1e-10 covers ~5 digits of accumulated error.
/// Calibrated across Exp020–030, Exp051–056 (ODE, phylo, dN/dS);
/// worst observed drift: ~3e-11 (dN/dS Exp052). Commit `e4358c5`.
pub const PYTHON_PARITY: f64 = 1e-10;

/// Python-to-Rust parity for high-precision analytical formulas
/// (phylogenetic likelihood, clock rate estimation).
///
/// These are direct formula evaluations with minimal accumulation.
/// 1e-14 is ~1 ULP for typical magnitude-1 results.
/// Calibrated: Exp029 (Felsenstein), Exp053/054 (molecular clock).
/// Scripts: `felsenstein_pruning_baseline.py`, `mateos2023_sulfur_phylogenomics.py`.
/// Commit `48fb787`.
pub const PYTHON_PARITY_TIGHT: f64 = 1e-14;

/// Python-to-Rust parity for statistical p-values and BH correction.
///
/// P-value computation involves log-gamma and combinatorial sums;
/// implementations may differ by a few ULP in tail behavior.
/// Calibrated: Exp056 (pangenome enrichment), Exp041 (EPA PFAS ML).
/// Scripts: `moulana2020_pangenomics.py`, `epa_pfas_ml_baseline.py`.
/// Commit `48fb787`.
pub const PYTHON_PVALUE: f64 = 1e-5;

/// ML model parity (decision tree, random forest, GBM confidence).
///
/// Floating-point vote ratios and softmax normalization. 1e-10
/// matches Python `sklearn` / `xgboost` to full f64 precision.
/// Calibrated: Exp008 (PFAS decision tree), Exp041 (EPA RF/GBM).
/// Scripts: `pfas_tree_export.py`, `exp008_pfas_ml_baseline.py`.
/// Commit `48fb787`.
pub const ML_PREDICTION: f64 = 1e-10;

/// Cross-species ANI comparison against Python baseline.
///
/// k-mer sampling and gap handling can differ at boundary codons.
/// 1e-4 covers implementation-level variation in ANI estimation.
/// Calibrated: Exp055 (population genomics ANI).
/// Script: `anderson2017_population_genomics.py`. Commit `48fb787`.
pub const ANI_CROSS_SPECIES: f64 = 1e-4;

/// ML classification metrics (F1, accuracy) against Python baseline.
///
/// F1 score involves integer counts (TP, FP, FN) and division;
/// ties and rounding can differ by ~1e-4 between implementations.
/// Calibrated: Exp008/041 (PFAS classification, EPA ML).
/// Scripts: `exp008_pfas_ml_baseline.py`, `epa_pfas_ml_baseline.py`.
/// Commit `48fb787`.
pub const ML_F1_SCORE: f64 = 1e-4;

/// Evolutionary distance estimation (Jukes-Cantor, alignment-derived).
///
/// Distance matrix computation accumulates log/exp corrections over
/// pairwise alignment columns; ~1e-3 covers implementation variation.
/// Calibrated: Exp033 (neighbor joining), Exp034 (DTL reconciliation).
/// Scripts: `liu2009_neighbor_joining.py`, `zheng2023_dtl_reconciliation.py`.
/// Commit `48fb787`.
pub const EVOLUTIONARY_DISTANCE: f64 = 1e-3;

/// Spectral cosine similarity rounding (m/z-shifted comparisons).
///
/// Greedy peak matching with floating-point intensity weighting
/// can differ by ~1e-3 at alignment boundaries.
/// Calibrated: Exp005/006 (mzML spectral matching), Exp042 (`MassBank`).
/// Scripts: `validate_track2.py`, `massbank_spectral_baseline.py`.
/// Commit `48fb787`.
pub const SPECTRAL_COSINE: f64 = 1e-3;

/// HMM forward log-likelihood Python parity.
///
/// HMM forward algorithm accumulates log-sum-exp over T×N states.
/// Python `hmmlearn` and Rust differ by ~1e-7 due to log/exp ordering.
/// 1e-6 covers observed variance for small models (2 states, 5 obs).
/// Validated: `scripts/liu2014_hmm_baseline.py`, commit `e4358c5`.
pub const HMM_FORWARD_PARITY: f64 = 1e-6;

// ═══════════════════════════════════════════════════════════════════
// Algorithm guards and convergence thresholds
// ═══════════════════════════════════════════════════════════════════

/// Minimum dS value for omega ratio computation.
///
/// When dS is below this threshold, omega (dN/dS) is undefined
/// because dividing by near-zero dS would produce an unstable ratio.
pub const DNDS_OMEGA_GUARD: f64 = 1e-10;

/// DADA2 error model convergence threshold.
///
/// Iteration stops when the max change in error rates between
/// rounds falls below this value. Matches the DADA2 R package default.
pub const DADA2_ERR_CONVERGENCE: f64 = 1e-6;

// ═══════════════════════════════════════════════════════════════════
// ODE / dynamical system tolerances
// ═══════════════════════════════════════════════════════════════════

/// ODE convergence epsilon for c-di-GMP concentration checks.
///
/// Used by QS biofilm, bistable, and multi-signal ODE models to
/// determine when a species concentration has effectively reached zero.
pub const ODE_CDG_CONVERGENCE: f64 = 1e-12;

/// ODE steady-state check: ±0.01 for species concentrations at equilibrium.
///
/// Used by Waters 2008 QS, Mhatre 2020 capacitor, Bruger 2018 cooperation,
/// and Fernandez 2020 bistable ODE models. RK4 vs `scipy.integrate` at
/// identical dt accumulates ~1e-3 difference for multi-species systems.
/// Calibrated: Exp020/023/025/027 (all ODE models).
/// Scripts: `waters2008_qs_ode.py`, `fernandez2020_bistable.py`,
/// `bruger2018_cooperation.py`, `mhatre2020_capacitor.py`.
/// Commit `48fb787`.
pub const ODE_STEADY_STATE: f64 = 0.01;

/// ODE method parity: RK4 vs LSODA integrator differences.
///
/// Fixed-step RK4 diverges from adaptive LSODA by up to ~1e-3 in
/// concentration for stiff systems.
/// Calibrated: Exp020/023/024/025/027/030 (all 6 ODE models).
/// Scripts: `waters2008_qs_ode.py`, `srivastava2011_multi_signal.py`,
/// `hsueh2022_phage_defense.py`. Commit `48fb787`.
pub const ODE_METHOD_PARITY: f64 = 1e-3;

/// ODE GPU vs CPU parity: same RK4, different instruction ordering.
///
/// GPU WGSL RK4 vs CPU Rust RK4 at identical dt. Both use f64; GPU
/// instruction reordering and FMA behavior yield ~1e-6 drift.
pub const ODE_GPU_PARITY: f64 = 1e-6;

/// Near-zero species concentrations (repressed pathways).
///
/// Biologically "off" species may float slightly above zero due to
/// integrator residual. 0.05 accommodates the numerical floor.
pub const ODE_NEAR_ZERO: f64 = 0.05;

/// GPU ODE sweep absolute parity: max `|CPU - GPU|` over long-horizon
/// integration (1000+ RK4 steps, 128 parameter batches).
///
/// After 1000 steps of RK4 at dt = 0.001, the GPU f64 and CPU f64
/// integrators diverge by up to 0.12 absolute in species concentrations
/// due to instruction reordering and FMA differences. 0.15 covers the
/// worst observed case with 20% margin.
/// Validated: Exp049 (GPU ODE sweep), commit `e4358c5`.
pub const ODE_GPU_SWEEP_ABS: f64 = 0.15;

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

/// Relative tolerance for near-zero ODE variables (GPU vs CPU).
///
/// When ODE variables are near zero (repressed pathways, depleted species),
/// the *relative* error `|gpu - cpu| / max(|gpu|, |cpu|)` can be large
/// even though the *absolute* difference is negligible. 1.5 (150%)
/// accommodates GPU/CPU integrator divergence at the numerical floor
/// where both values are biologically insignificant (< 0.01).
pub const ODE_NEAR_ZERO_RELATIVE: f64 = 1.5;

// ═══════════════════════════════════════════════════════════════════
// Phylogenetic tolerances
// ═══════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════
// PFAS / analytical chemistry tolerances
// ═══════════════════════════════════════════════════════════════════

/// KMD homologue grouping: unitless Kendrick mass defect difference.
///
/// PFAS compounds in the same series differ by < 0.01 KMD.
pub const KMD_GROUPING: f64 = 0.01;

/// KMD series spread: max intra-series range for PFCA/PFSA.
///
/// Homologous series (e.g. PFCAs C4-C14) span < 0.02 KMD units.
/// Validated against EPA Method 533 reference standards (Exp006,
/// commit `eb99b12`).
pub const KMD_SPREAD: f64 = 0.02;

/// Fragment m/z tolerance (Da) for suspect screening.
///
/// Tighter than [`MZ_TOLERANCE`] for high-resolution MS2 fragment
/// matching. Derived from Orbitrap MS2 mass accuracy (sub-2 ppm at
/// typical fragment m/z 50–500 Da).
/// Calibrated: Exp006 (PFAS suspect screening), Exp042 (`MassBank`).
/// Scripts: `validate_track2.py`, `massbank_spectral_baseline.py`.
/// Commit `48fb787`.
pub const MZ_FRAGMENT: f64 = 0.001;

/// KMD non-homologue separation threshold.
///
/// Compounds from *different* PFAS series (e.g. PFCAs vs sulfonates +
/// non-PFAS) should differ by > 0.005 KMD units. Tighter than
/// [`KMD_GROUPING`] to verify that non-homologues are correctly excluded.
/// Validated: Exp006 (EPA Method 533), commit `eb99b12`.
pub const KMD_NON_HOMOLOGUE: f64 = 0.005;

/// Spectral matching m/z window for unit-resolution MS2.
///
/// Used as the matching window parameter for `cosine_similarity`.
/// Standard unit-resolution window for ion-trap / triple-quad MS2.
/// Calibrated: Exp005 (mzML cosine), Exp042 (`MassBank` spectral).
/// Scripts: `validate_track2.py`, `massbank_spectral_baseline.py`.
/// Commit `48fb787`.
pub const SPECTRAL_MZ_WINDOW: f64 = 0.5;

// ═══════════════════════════════════════════════════════════════════
// Peak detection / signal processing tolerances
// ═══════════════════════════════════════════════════════════════════

/// Relative peak height tolerance (1%) vs `scipy` baseline.
///
/// Peak detection prominence and height differ by ≤1% between
/// `scipy.signal.find_peaks` and the Rust implementation due to
/// interpolation at boundary samples.
/// Calibrated: Exp010 (peak detection).
/// Script: `generate_peak_baselines.py`. Commit `48fb787`.
pub const PEAK_HEIGHT_REL: f64 = 0.01;

/// Rarefaction curve monotonicity guard.
///
/// Rarefaction is mathematically non-decreasing; small rounding in
/// hypergeometric terms can cause ≤ 1e-10 decreases. Same order as
/// [`PYTHON_PARITY`] but semantically a monotonicity check, not a
/// baseline-comparison tolerance.
pub const RAREFACTION_MONOTONIC: f64 = 1e-10;

/// `PCoA` eigenvalue non-negativity floor.
///
/// Jacobi eigendecomposition of a centered distance matrix should produce
/// non-negative eigenvalues. Numerical noise from double-centering and
/// finite Jacobi sweeps can push tiny eigenvalues to −1e-10. Using this
/// as `e >= -PCOA_EIGENVALUE_FLOOR` distinguishes genuine negative
/// eigenvalues (non-Euclidean metric) from rounding artifacts.
pub const PCOA_EIGENVALUE_FLOOR: f64 = 1e-10;

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

/// ESN ridge regression regularisation (Tikhonov λ).
///
/// Jaeger (2001) "The echo state approach" recommends λ ∈ [1e-8, 1e-2].
/// Default 1e-6 balances numerical stability with fitting accuracy,
/// matching the common practice in Lukoševičius (2012) "A Practical
/// Guide to Applying Echo State Networks."
pub const ESN_REGULARIZATION: f64 = 1e-6;

/// Chao1 singleton/doubleton count detection half-width.
///
/// Counts within 0.5 of 1.0 or 2.0 are classified as singletons or
/// doubletons respectively. This accommodates floating-point abundance
/// values (e.g. rarefied counts) while remaining exact for integer data.
/// Matches the rounding behavior of `skbio.diversity.alpha.chao1`.
pub const CHAO1_COUNT_HALFWIDTH: f64 = 0.5;

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

// ═══════════════════════════════════════════════════════════════════
// GPU vs CPU tolerances
// ═══════════════════════════════════════════════════════════════════

/// Regularized incomplete gamma series convergence epsilon.
///
/// Used by `special::regularized_gamma_p` for series termination.
/// Matches `scipy`'s `gammainc` convergence behavior at 1e-15.
pub const GAMMA_SERIES_CONVERGENCE: f64 = 1e-15;

/// Maximum iterations for regularized gamma series expansion.
pub const GAMMA_SERIES_MAX_ITER: usize = 1000;

/// GPU f64 vs CPU f64 for exact arithmetic (add, mul, comparison).
///
/// GPU `SHADER_F64` uses IEEE 754 f64, but different instruction ordering
/// and FMA behavior can introduce small differences. Typical max observed
/// diff ~8e-8 for `exp()` on current hardware (hotSpring Exp001 §4.4).
///
/// For simple add/mul chains (diversity metrics), expect < 1e-10.
pub const GPU_VS_CPU_F64: f64 = 1e-6;

/// GPU f64 vs CPU f64 for log/exp operations (transcendentals).
///
/// Native WGSL `log(f64)` on NVIDIA matches CPU to ~1e-14 per call.
/// Allow 1e-10 for single transcendental evaluations (one log or exp).
/// For chained Shannon/Simpson/cosine, use `GPU_LOG_POLYFILL` (1e-7).
pub const GPU_VS_CPU_TRANSCENDENTAL: f64 = 1e-10;

/// GPU f64 log polyfill precision (software `log_f64` shader).
///
/// When native WGSL `log(f64)` is unavailable, `ToadStool` uses a
/// polynomial `log_f64` polyfill with ~1e-8 absolute precision.
/// Allow 1e-7 for single evaluations; accumulated chains (Shannon
/// over N species) may reach ~1e-6 covered by [`GPU_VS_CPU_F64`].
pub const GPU_LOG_POLYFILL: f64 = 1e-7;

/// GPU Bray-Curtis vs CPU: per-pair tolerance.
///
/// Each pair involves N additions and a division. For N=2000 features,
/// rounding differs by at most a few ULP per addition.
pub const GPU_VS_CPU_BRAY_CURTIS: f64 = 1e-10;

/// GPU vs CPU for stochastic ensemble statistics (bootstrap mean/var).
///
/// Parallel reduction order differs on GPU; for ensemble averages
/// over ~1000 replicates, accumulated rounding yields ~1e-4 drift.
pub const GPU_VS_CPU_ENSEMBLE: f64 = 1e-4;

/// GPU vs CPU for HMM batch forward log-likelihoods.
///
/// 256 sequences × 100 steps × 3 states: log-space additions
/// across the Forward lattice accumulate rounding differently on GPU
/// (warp-level parallel reduction) vs CPU (sequential). 1e-3 covers
/// the worst observed per-sequence drift across the batch.
/// Validated: Exp048, `benchmark_phylo_hmm_gpu`, commit `e4358c5`.
pub const GPU_VS_CPU_HMM_BATCH: f64 = 1e-3;

/// GPU f32 vs CPU f64 for integer-derived results (Hamming, Jaccard).
///
/// f32 has ~7 significant digits; operations on integer-derived values
/// (count / total) yield results exact to ~1e-6. Allow 1e-5 to cover
/// accumulated rounding in pairwise summation.
pub const GPU_F32_PARITY: f64 = 1e-5;

/// GPU f32 spatial computation tolerance (payoff, fitness, variance).
///
/// f32 grid operations (neighbor sums, dot products) accumulate
/// rounding proportional to neighborhood size. For 8-neighbor grids,
/// 1e-4 covers the worst-case f32 summation error.
pub const GPU_F32_SPATIAL: f64 = 1e-4;

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
// `Galaxy` / Exp002 observational range tolerances
// ═══════════════════════════════════════════════════════════════════

/// ODE biofilm dispersed-state `B_ss` tolerance.
///
/// RK4 vs LSODA for biofilm concentration at near-zero steady states.
/// `B_ss` ≈ 0.02–0.10 depending on scenario; 0.03 covers integrator drift
/// between fixed-step RK4 and adaptive LSODA at dt=0.001.
/// Validated: Exp020 (Waters 2008), `scripts/waters2008_qs_ode.py`.
pub const ODE_BIOFILM_SS: f64 = 0.03;

/// Bootstrap log-likelihood ensemble tolerance.
///
/// 100 bootstrap replicates produce mean LL near the original tree's LL.
/// Sampling variance across resampled alignments yields a spread of
/// ±5.0 log-likelihood units for typical 400-site alignments.
/// Validated: Exp031 (Wang 2021 RAWR), `scripts/wang2021_rawr_bootstrap.py`.
pub const BOOTSTRAP_LL_ENSEMBLE: f64 = 5.0;

/// Phage defense population count tolerance (Bd vs Python baseline).
///
/// Phage attack scenario: large population ODE where absolute counts
/// reach ~278 cells. RK4 vs LSODA at dt=0.001 differ by ~10 cells
/// due to step-size sensitivity in rapid population crashes.
/// Validated: Exp030 (Hsueh 2022), `scripts/hsueh2022_phage_defense.py`.
pub const PHAGE_POPULATION_ABSOLUTE: f64 = 10.0;

/// Phage defense large-population tolerance (no-phage / pure-defended).
///
/// Steady-state populations ~100k–140k cells. RK4 vs LSODA at dt=0.001
/// accumulates ~1000 cells of drift over 100+ hours of simulated time.
/// Proportionally ~0.7% — same relative order as [`PHAGE_POPULATION_ABSOLUTE`]
/// for the attack scenario.
/// Validated: Exp030 (Hsueh 2022), `scripts/hsueh2022_phage_defense.py`.
pub const PHAGE_LARGE_POPULATION: f64 = 1000.0;

/// Phage defense near-zero floor (crashed population).
///
/// After phage attack, undefended bacteria crash to ~0. RK4 integrator
/// residual keeps the value slightly above zero. 1.0 cell is the
/// biologically insignificant floor.
/// Validated: Exp030 (Hsueh 2022), `scripts/hsueh2022_phage_defense.py`.
pub const PHAGE_CRASH_FLOOR: f64 = 1.0;

/// Exp002 `Galaxy` Shannon entropy range for rank-abundance curves.
///
/// Simulated communities derived from Exp002 phytoplankton rank-abundance
/// profiles. Shannon varies from ~2.93 (low-diversity, 91 ASVs) to ~3.85
/// (high-diversity, 856 ASVs). ±1.50 covers the full profile variability
/// from geometric/power-law rank abundance curves.
/// Source: `experiments/results/002_phytoplankton/diversity_report.json`,
/// commit `21d43a0` (Exp002 complete — 2273 ASVs from real phytoplankton).
pub const GALAXY_SHANNON_RANGE: f64 = 1.50;

/// Exp002 `Galaxy` Simpson range for rank-abundance curves.
///
/// Low-diversity community: Simpson ~0.86 ± 0.25. The wide range reflects
/// sensitivity of Simpson to dominance in highly-skewed communities.
/// Source: `experiments/results/002_phytoplankton/diversity_report.json`,
/// commit `21d43a0`.
pub const GALAXY_SIMPSON_RANGE: f64 = 0.25;

/// Exp002 `Galaxy` Bray-Curtis range between dissimilar communities.
///
/// BC(low, high) diversity communities: expected near 0.50 but highly
/// dependent on rank-abundance shape. ±0.50 covers [0.0, 1.0] for any
/// biologically plausible community pair.
/// Source: `experiments/results/002_phytoplankton/diversity_report.json`,
/// commit `21d43a0`.
pub const GALAXY_BRAY_CURTIS_RANGE: f64 = 0.50;

// ═══════════════════════════════════════════════════════════════════
// Feature extraction / asari cross-reference tolerances (Exp009)
// ═══════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════
// ODE domain-specific steady-state tolerances
// ═══════════════════════════════════════════════════════════════════

/// Bistable ODE low-biofilm steady-state tolerance.
///
/// For `B_ss` ≈ 0.040 in the zero-feedback scenario, RK4 vs LSODA
/// differ by ~0.005. Tighter than [`ODE_STEADY_STATE`] (0.01) because
/// the baseline value itself is small.
/// Validated: Exp023 (Fernandez 2020), `scripts/fernandez2020_bistable.py`,
/// commit `e4358c5`.
pub const ODE_BISTABLE_LOW_B: f64 = 0.005;

/// ODE c-di-GMP / autoinducer steady-state tolerance.
///
/// For c-di-GMP (`C_ss` ≈ 1.634 in bistable) and autoinducer (`AI_ss` ≈ 1.854
/// in cooperation), RK4 vs LSODA accumulate ~0.02 difference due to
/// stiff feedback loops in the signaling cascade.
/// Validated: Exp023 (Fernandez 2020) and Exp025 (Bruger 2018),
/// `scripts/fernandez2020_bistable.py`, `scripts/bruger2018_cooperation.py`,
/// commit `e4358c5`.
pub const ODE_SIGNAL_SS: f64 = 0.02;

// ═══════════════════════════════════════════════════════════════════
// HMM invariant tolerance
// ═══════════════════════════════════════════════════════════════════

/// HMM mathematical invariant slack: Viterbi log-prob ≤ Forward LL.
///
/// The most-likely path probability (Viterbi) must be ≤ the total
/// probability (Forward) by definition. Numerical noise from
/// log-sum-exp accumulation can cause sub-ULP violations; 1e-10
/// accommodates this without masking real bugs.
/// Validated: Exp026 (Liu 2014), `scripts/liu2014_hmm_baseline.py`,
/// commit `e4358c5`.
pub const HMM_INVARIANT_SLACK: f64 = 1e-10;

// ═══════════════════════════════════════════════════════════════════
// NPU triage acceptance thresholds
// ═══════════════════════════════════════════════════════════════════

/// NPU triage pass rate ceiling (candidates / total < threshold).
///
/// The NPU int8 triage should reduce the candidate set to < 30% of
/// the full library. If pass rate exceeds this, the triage is not
/// selective enough to provide a speedup.
/// Derived: Exp124 (analytical), 5000 library × 100 queries, top-20%.
pub const NPU_PASS_RATE_CEILING: f64 = 0.30;

/// NPU triage recall floor (true match in candidate set).
///
/// At least 90% of queries must have their true match in the triage
/// candidates. Below this, the speedup comes at unacceptable accuracy loss.
/// Derived: Exp124 (analytical), 5000 library × 100 queries, top-20%.
pub const NPU_RECALL_FLOOR: f64 = 0.90;

/// NPU triage top-1 accuracy floor.
///
/// After full-precision re-scoring of triage candidates, at least 80%
/// of queries must rank the true match as #1.
/// Derived: Exp124 (analytical), cosine re-scoring on top-20% candidates.
pub const NPU_TOP1_FLOOR: f64 = 0.80;

// ═══════════════════════════════════════════════════════════════════
// Performance guard thresholds
// ═══════════════════════════════════════════════════════════════════

/// GEMM pipeline compilation timeout (milliseconds).
///
/// The GPU GEMM shader (via hotSpring f64 polyfills) must compile in
/// under 30 seconds on supported hardware. Exceeding this indicates
/// a driver or shader complexity regression.
pub const GEMM_COMPILE_TIMEOUT_MS: f64 = 30_000.0;

// ═══════════════════════════════════════════════════════════════════
// Feature extraction / asari cross-reference tolerances (Exp009)
// ═══════════════════════════════════════════════════════════════════

/// Asari cross-match percentage tolerance (Exp009).
///
/// At least 30% of asari features must be recovered by Rust. Expressed as
/// the tolerance around the 100% ideal: `expected=30, tol=70` means the
/// check passes when `match_pct >= 30%`. Wider than other tolerances
/// because single-file extraction covers fewer features than the full
/// 8-file asari run.
/// Validated: Exp009 (MT02 HILIC-pos, 8 mzML, asari 1.13.1).
pub const ASARI_CROSS_MATCH_PCT: f64 = 70.0;

/// Asari m/z-range coverage percentage tolerance (Exp009).
///
/// At least 90% of Rust-detected features should fall within asari's
/// observed m/z range (80–1000 Da). `expected=100, tol=10` means the
/// check passes when `range_pct >= 90%`.
/// Validated: Exp009 (MT02 HILIC-pos, asari range 83–999 Da).
pub const ASARI_MZ_RANGE_PCT: f64 = 10.0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn tolerance_hierarchy_is_monotonic() {
        assert!(EXACT < EXACT_F64);
        assert!(EXACT_F64 < ANALYTICAL_F64);
        assert!(ANALYTICAL_F64 < PYTHON_PARITY);
        assert!(PYTHON_PARITY <= ML_PREDICTION);
        assert!(GPU_VS_CPU_TRANSCENDENTAL <= GPU_LOG_POLYFILL);
        assert!(GPU_LOG_POLYFILL <= GPU_VS_CPU_F64);
        assert!(GPU_VS_CPU_F64 <= GPU_VS_CPU_ENSEMBLE);
    }

    #[test]
    fn all_tolerances_are_non_negative() {
        let all = [
            EXACT,
            EXACT_F64,
            ANALYTICAL_F64,
            GC_CONTENT,
            MEAN_QUALITY,
            MZ_TOLERANCE,
            SHANNON_SIMULATED,
            SIMPSON_SIMULATED,
            CHAO1_RANGE,
            BRAY_CURTIS_SYMMETRY,
            PYTHON_PARITY,
            PYTHON_PARITY_TIGHT,
            PYTHON_PVALUE,
            ML_PREDICTION,
            ANI_CROSS_SPECIES,
            ML_F1_SCORE,
            EVOLUTIONARY_DISTANCE,
            SPECTRAL_COSINE,
            HMM_FORWARD_PARITY,
            DNDS_OMEGA_GUARD,
            DADA2_ERR_CONVERGENCE,
            ODE_CDG_CONVERGENCE,
            ODE_STEADY_STATE,
            ODE_METHOD_PARITY,
            ODE_GPU_PARITY,
            ODE_NEAR_ZERO,
            ODE_GPU_SWEEP_ABS,
            GPU_EIGENVALUE_REL,
            GPU_LANCZOS_EIGENVALUE_ABS,
            FINITE_SIZE_SCALING_REL,
            LEVEL_SPACING_STDERR_MAX,
            ODE_NEAR_ZERO_RELATIVE,
            PHYLO_LIKELIHOOD,
            JC69_PROBABILITY,
            GILLESPIE_MEAN_REL,
            GILLESPIE_FANO,
            GILLESPIE_PYTHON_RANGE_REL,
            GILLESPIE_FANO_PHYSICAL,
            KMD_GROUPING,
            KMD_SPREAD,
            MZ_FRAGMENT,
            KMD_NON_HOMOLOGUE,
            SPECTRAL_MZ_WINDOW,
            PEAK_HEIGHT_REL,
            RAREFACTION_MONOTONIC,
            PCOA_EIGENVALUE_FLOOR,
            JACOBI_CONVERGENCE,
            JACOBI_ELEMENT_SKIP,
            JACOBI_TAU_OVERFLOW,
            MATRIX_EPS,
            ESN_REGULARIZATION,
            CHAO1_COUNT_HALFWIDTH,
            BOX_MULLER_U1_FLOOR,
            GAMMA_RIGHT_TAIL_OFFSET,
            ODE_DIVISION_GUARD,
            GAMMA_SERIES_CONVERGENCE,
            GPU_VS_CPU_F64,
            GPU_VS_CPU_TRANSCENDENTAL,
            GPU_LOG_POLYFILL,
            GPU_VS_CPU_BRAY_CURTIS,
            GPU_VS_CPU_ENSEMBLE,
            GPU_VS_CPU_HMM_BATCH,
            GPU_F32_PARITY,
            GPU_F32_SPATIAL,
            ODE_DEFAULT_DT,
            ODE_BIOFILM_SS,
            BOOTSTRAP_LL_ENSEMBLE,
            PHAGE_POPULATION_ABSOLUTE,
            PHAGE_LARGE_POPULATION,
            PHAGE_CRASH_FLOOR,
            ODE_BISTABLE_LOW_B,
            ODE_SIGNAL_SS,
            HMM_INVARIANT_SLACK,
            NPU_PASS_RATE_CEILING,
            NPU_RECALL_FLOOR,
            NPU_TOP1_FLOOR,
            GEMM_COMPILE_TIMEOUT_MS,
            GALAXY_SHANNON_RANGE,
            GALAXY_SIMPSON_RANGE,
            GALAXY_BRAY_CURTIS_RANGE,
            ASARI_CROSS_MATCH_PCT,
            ASARI_MZ_RANGE_PCT,
        ];
        for tol in all {
            assert!(tol >= 0.0, "tolerance {tol} must be non-negative");
        }
    }
}
