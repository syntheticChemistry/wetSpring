// SPDX-License-Identifier: AGPL-3.0-or-later
//! Miscellaneous biological tolerances: DADA2, `PCoA`, NPU, Nanopore, validation.

/// DADA2 error model convergence threshold.
///
/// Iteration stops when the max change in error rates between
/// rounds falls below this value. Matches the DADA2 R package default.
pub const DADA2_ERR_CONVERGENCE: f64 = 1e-6;

/// `PCoA` eigenvalue non-negativity floor.
///
/// Jacobi eigendecomposition of a centered distance matrix should produce
/// non-negative eigenvalues. Numerical noise from double-centering and
/// finite Jacobi sweeps can push tiny eigenvalues to −1e-10. Using this
/// as `e >= -PCOA_EIGENVALUE_FLOOR` distinguishes genuine negative
/// eigenvalues (non-Euclidean metric) from rounding artifacts.
pub const PCOA_EIGENVALUE_FLOOR: f64 = 1e-10;

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

/// Observed feature count tolerance (integer-valued diversity metric).
///
/// `observed_features()` returns an integer-valued count cast to f64.
/// Exact match is expected, but ±1.0 accommodates rounding in
/// rarefied / subsampled communities where floating-point abundances
/// are rounded to presence/absence.
/// Validated: Exp002 (diversity), commit `e4358c5`, 2026-02-19.
pub const OBSERVED_FEATURES_TOL: f64 = 1.0;

/// Chimera detection parent-count tolerance.
///
/// After chimera removal the number of surviving parents is an
/// integer; ±1 covers borderline score ties.
/// Validated: Exp001 (16S pipeline), commit `e4358c5`, 2026-02-19.
pub const CHIMERA_PARENT_TOL: f64 = 1.0;

/// Batch merge-pairs mean-overlap tolerance (bp).
///
/// Mean overlap across N pairs may drift ±1 bp due to integer/float
/// rounding of overlap positions.
/// Validated: Exp001 (FASTQ merge), commit `e4358c5`, 2026-02-19.
pub const MERGE_OVERLAP_TOL: f64 = 1.0;

/// Numerical Hessian diagonal element tolerance.
///
/// `numerical_hessian` at ε = 1e-5 uses O(ε²) central differences.
/// For the Rosenbrock function at (1,1), H\[0,0\] ≈ 802 and H\[1,1\] ≈ 200.
/// The finite-difference truncation error is O(ε²) ≈ 1e-10, but the
/// conditioning of the second-order quotient amplifies rounding to ~2.0
/// for H\[0,0\] and ~1.0 for H\[1,1\].
/// Validated: Exp224 (`BarraCuda` CPU v14), commit `e4358c5`, 2026-02-19.
/// Source: neuralSpring Rosenbrock baseline.
pub const HESSIAN_H00_TOL: f64 = 2.0;

/// Numerical Hessian off-diagonal / second diagonal tolerance.
///
/// See [`HESSIAN_H00_TOL`] for derivation.
pub const HESSIAN_H11_TOL: f64 = 1.0;

/// Simulated nanopore mean read length tolerance (bp).
///
/// Synthetic 16S reads are generated at a target length of 1450 bp.
/// The tolerance covers the ±10 bp variation from the LCG-based
/// read generator. All reads are the exact target length in the
/// current deterministic generator; 10.0 provides margin for future
/// stochastic length models.
/// Validated: Exp196b (simulated long-read 16S), Phase 61, 2026-02-26.
pub const NANOPORE_MEAN_READ_LENGTH_TOL: f64 = 10.0;

/// PFAS fragment screening minimum intensity threshold (%).
///
/// Fragments below 5% relative intensity are noise; above 5% they
/// contribute to the CF2/C2F4/HF difference screening.
/// Source: JonZwe/FindPFAS default parameterisation.
/// Validated: Exp006 (PFAS screening), commit `eb99b12`, 2026-02-16.
pub const PFAS_MIN_INTENSITY_PCT: f64 = 5.0;

/// PFAS ML classification acceptance floor (accuracy and F1).
///
/// Decision tree and random forest models trained on Exp008 PFAS data
/// must exceed 80% accuracy and F1 to be considered viable.
/// Calibrated: Exp008 (PFAS decision tree), sklearn baseline achieves
/// ~0.95 accuracy; 0.80 is a conservative floor.
/// Script: `pfas_tree_export.py`, commit `e4358c5`, 2026-02-19.
pub const PFAS_ML_ACCEPTANCE_FLOOR: f64 = 0.80;

/// Islam 2014 soil disorder W analytical tolerance.
///
/// W = 25 × (1 − connectivity). For connectivity = 0.793 (no-till)
/// and 0.385 (tilled), W = 5.175 and 15.375 respectively.
/// These are exact analytical values; tolerance covers only f64
/// rounding in the multiplication chain.
/// Source: Islam (2014), Brandt Farm no-till study.
/// Validated: Exp183 (soil QS), commit `48fb787`.
pub const SOIL_DISORDER_ANALYTICAL: f64 = 0.01;

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

// ═══════════════════════════════════════════════════════════════════
// NMF / drug repurposing
// ═══════════════════════════════════════════════════════════════════

/// NMF convergence tolerance for KL-divergence objective.
///
/// Iteration stops when relative change in objective falls below this.
/// KL-divergence NMF typically converges more slowly than Euclidean;
/// 1e-4 balances accuracy with iteration count for drug-disease matrices.
/// Validated: Exp267 (`ToadStool` Dispatch v3), Exp224 (Paper Math Control).
pub const NMF_CONVERGENCE_KL: f64 = 1e-4;

/// NMF convergence tolerance for Euclidean objective.
///
/// Tighter than [`NMF_CONVERGENCE_KL`] for Euclidean NMF (W·H ≈ V).
/// Drug repurposing GEMM reconstruction and top-k ranking use this.
/// Validated: Exp164 (GPU Drug Repurposing).
pub const NMF_CONVERGENCE_EUCLIDEAN: f64 = 1e-6;

/// NMF convergence tolerance for rank-sensitivity sweeps.
///
/// Slightly tighter than [`NMF_CONVERGENCE_KL`] (1e-4) to capture
/// rank-dependent convergence differences without over-iterating.
/// Used in rank-search validation (Exp267, `RepoDB` NMF rank sweep).
pub const NMF_CONVERGENCE_RANK_SEARCH: f64 = 1e-5;

/// Ridge regression default regularization (Tikhonov λ).
///
/// Common default for small-to-medium regression problems. Prevents
/// overfitting when `n_features` ≈ `n_samples`. Used in `ToadStool` Dispatch
/// and Paper Math Control ridge validation.
pub const RIDGE_REGULARIZATION_DEFAULT: f64 = 0.01;

/// Ridge regression small regularization for well-conditioned problems.
///
/// When design matrix is well-conditioned, smaller λ preserves
/// coefficient accuracy. Used in Fajgenbaum NMF pathway scoring.
pub const RIDGE_REGULARIZATION_SMALL: f64 = 1e-6;

/// Ridge regression output tolerance vs analytical expectation.
///
/// Ridge weights and predictions vs known linear targets. Covers
/// Cholesky solve rounding and matrix conditioning effects.
/// Validated: Exp267, Exp224.
pub const RIDGE_TEST_TOL: f64 = 1e-4;

// ═══════════════════════════════════════════════════════════════════
// Numerical differentiation / integration
// ═══════════════════════════════════════════════════════════════════

/// Numerical Hessian finite-difference step (central differences).
///
/// ε for O(ε²) central-difference quotient. Too small amplifies
/// rounding; too large increases truncation error. 1e-5 is standard
/// for magnitude-1 functions (e.g. x²+y² at (1,1)).
/// Validated: Exp267 (Rosenbrock-style Hessian).
pub const NUMERICAL_HESSIAN_EPSILON: f64 = 1e-5;

/// Numerical Hessian output tolerance vs analytical second derivatives.
///
/// For f(x,y)=x²+y², H\[0,0\]=H\[1,1\]=2, H\[0,1\]=0. Finite-difference
/// truncation and conditioning yield ~1e-4 absolute error.
/// Validated: Exp267.
pub const HESSIAN_TEST_TOL: f64 = 1e-4;

// ═══════════════════════════════════════════════════════════════════
// Knowledge graph / embedding
// ═══════════════════════════════════════════════════════════════════

/// Embedding norm floor for L2 normalization.
///
/// When ‖v‖ < this, skip normalization to avoid division by near-zero.
/// `TransE` and KG embedding init use this to handle zero/negligible vectors.
/// 1e-12 allows ~1e-6 relative error in subsequent dot products.
pub const EMBEDDING_NORM_FLOOR: f64 = 1e-12;

/// Knowledge graph Hits@10 minimum threshold.
///
/// Link prediction: Hits@10 must exceed 5% to beat random baseline.
/// `TransE` on synthetic drug-disease KG typically achieves 20–40%.
/// Validated: Exp161 (ROBOKOP KG embedding).
pub const KNOWLEDGE_GRAPH_HITS10_FLOOR: f64 = 0.05;

// ═══════════════════════════════════════════════════════════════════
// ODE biological thresholds
// ═══════════════════════════════════════════════════════════════════

/// Cooperators persist threshold (Bruger & Waters 2018).
///
/// Steady-state cooperator count must exceed this to confirm coexistence.
/// Below 0.001, cheaters have effectively dominated.
/// Validated: Exp102 (`BarraCuda` CPU v8).
pub const ODE_COOPERATOR_PERSIST_THRESHOLD: f64 = 0.001;

/// Capacitor cell growth minimum (Mhatre 2020).
///
/// Steady-state cell count must exceed 0.01 to confirm phenotypic
/// capacitor effect. Near-zero indicates failed growth.
/// Validated: Exp102 (`BarraCuda` CPU v8).
pub const ODE_CELL_GROWTH_THRESHOLD: f64 = 0.01;

// ═══════════════════════════════════════════════════════════════════
// Nautilus / ESN / taxonomy
// ═══════════════════════════════════════════════════════════════════

/// Default ridge regularization for Nautilus bio-ESN shell readout.
///
/// Prevents overfitting in the shell's linear readout layer.
/// Larger than [`RIDGE_REGULARIZATION_SMALL`] (1e-6) because the
/// ESN reservoir state matrix is typically ill-conditioned.
pub const RIDGE_NAUTILUS_DEFAULT: f64 = 1e-4;

/// Log-probability floor for Bayesian classifiers.
///
/// Prevents `ln(0)` in naive Bayes / taxonomy classification when
/// a k-mer has zero observed frequency. 1e-300 is safely above
/// `f64::MIN_POSITIVE` (2.2e-308) while being negligible in probability space.
pub const LOG_PROB_FLOOR: f64 = 1e-300;

// ═══════════════════════════════════════════════════════════════════
// Synthetic signal generation
// ═══════════════════════════════════════════════════════════════════

/// Box-Muller u1 floor for synthetic nanopore signal generation.
///
/// More conservative than [`BOX_MULLER_U1_FLOOR`](crate::tolerances::BOX_MULLER_U1_FLOOR)
/// (1e-15) because synthetic signals use a low-quality LCG whose
/// output can cluster near zero. 1e-30 bounds the noise to ~11.7σ.
pub const BOX_MULLER_U1_FLOOR_SYNTHETIC: f64 = 1e-30;
