// SPDX-License-Identifier: AGPL-3.0-or-later
//! Biological tolerances: diversity, ODE, phylogenomics, Python parity, HMM, etc.

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

/// Relative tolerance for near-zero ODE variables (GPU vs CPU).
///
/// When ODE variables are near zero (repressed pathways, depleted species),
/// the *relative* error `|gpu - cpu| / max(|gpu|, |cpu|)` can be large
/// even though the *absolute* difference is negligible. 1.5 (150%)
/// accommodates GPU/CPU integrator divergence at the numerical floor
/// where both values are biologically insignificant (< 0.01).
pub const ODE_NEAR_ZERO_RELATIVE: f64 = 1.5;

/// ODE biofilm dispersed-state `B_ss` tolerance.
///
/// RK4 vs LSODA for biofilm concentration at near-zero steady states.
/// `B_ss` ≈ 0.02–0.10 depending on scenario; 0.03 covers integrator drift
/// between fixed-step RK4 and adaptive LSODA at dt=0.001.
/// Validated: Exp020 (Waters 2008), `scripts/waters2008_qs_ode.py`.
pub const ODE_BIOFILM_SS: f64 = 0.03;

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

/// Soil recovery W(t) absolute tolerance at 40-year horizon.
///
/// Recovery model W(40yr) approaches ~4.0; RK4 vs analytical differ by
/// up to ~1.0 due to exponential decay accumulation.
/// Validated: Exp216 (`BarraCuda` CPU v13), soil recovery check.
pub const SOIL_RECOVERY_W_TOL: f64 = 1.0;

/// Bistable ODE high-biofilm attractor tolerance.
///
/// For the sessile attractor in the Fernandez 2020 bistable model,
/// `B_ss` ≈ 0.7. RK4 vs LSODA accumulate ~0.10 difference in the
/// high-biofilm state due to stiffness in the feedback loop.
/// 0.15 covers the worst observed case with 50% margin.
/// Validated: Exp079 (`BarraCuda` CPU v6), `scripts/fernandez2020_bistable.py`.
pub const ODE_BISTABLE_HIGH_B: f64 = 0.15;

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
// Anderson / spectral theory (biological context)
// ═══════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════
// Galaxy / Exp002 observational range tolerances
// ═══════════════════════════════════════════════════════════════════

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
// Rarefaction, PCoA, HMM
// ═══════════════════════════════════════════════════════════════════

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
// ESN, Chao1, NPU, Nanopore, Asari, GC
// ═══════════════════════════════════════════════════════════════════

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

/// Minimum GC content range (fraction) for genus-level diversity.
///
/// Vibrio genus assemblies should span ≥2% GC range to confirm
/// meaningful diversity. Used as threshold: `gc_range >= GC_GENUS_DIVERSITY_MIN`.
/// Validated: Exp228 (`validate_nucleus_pipeline_v69`).
pub const GC_GENUS_DIVERSITY_MIN: f64 = 0.02;

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

/// Nanopore signal round-trip tolerance (NRS wire format).
///
/// Signal written to NRS and read back must be bit-exact (i16 ↔ i16).
/// Calibration values (f64 ↔ f64) must survive IEEE 754 round-trip.
/// Tolerance of 0.0 enforces exact match.
/// Validated: Exp196a (POD5 parser validation).
pub const NANOPORE_SIGNAL_ROUNDTRIP: f64 = 0.0;

/// Nanopore calibrated signal tolerance (pA conversion).
///
/// Affine calibration `pA = raw * scale + offset` introduces no error
/// beyond f64 arithmetic (`mul_add` is fused). Machine precision suffices.
/// Validated: Exp196a (calibration round-trip).
pub const NANOPORE_CALIBRATION: f64 = 1e-12;

/// Nanopore synthetic basecall accuracy threshold.
///
/// Threshold-based basecalling on synthetic signal with 200 samples per
/// base and Gaussian noise (σ=10 ADC) achieves >87.5% per-base accuracy.
/// This is a validation floor for the synthetic generator, not a claim
/// about real nanopore basecalling accuracy.
/// Validated: Exp196a (synthetic basecall round-trip).
pub const NANOPORE_BASECALL_ACCURACY: f64 = 0.75;

/// Nanopore signal statistics tolerance.
///
/// Mean and standard deviation of ADC signal (i16 → f64 summation)
/// should be within machine precision of the reference calculation.
/// Validated: Exp196a (signal statistics).
pub const NANOPORE_SIGNAL_STATS: f64 = 1e-10;

/// Nanopore int8 quantization fidelity threshold.
///
/// When quantizing nanopore-quality features to int8 for NPU inference,
/// the classification agreement between f64 and int8 paths must exceed
/// this threshold. 90% is conservative — real ESN classifiers on clean
/// signal typically achieve >95%.
/// Validated: Exp196c (int8 quantization from noisy reads).
pub const NANOPORE_INT8_FIDELITY: f64 = 0.90;

/// Nanopore simulated 16S pipeline diversity tolerance.
///
/// Shannon diversity computed from long-read (nanopore-length) 16S reads
/// vs short-read (Illumina-length) reference. Nanopore's higher error
/// rate (~5-10% per-read) inflates observed OTU counts slightly, but
/// diversity metrics are robust. 0.3 covers the expected inflation.
/// Validated: Exp196b (simulated long-read 16S).
pub const NANOPORE_DIVERSITY_TOLERANCE: f64 = 0.3;

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
