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
pub const GC_CONTENT: f64 = 0.005;

/// Mean quality score: ±0.5 (`FastQC` baseline, read-level variability).
pub const MEAN_QUALITY: f64 = 0.5;

/// m/z tolerance: ±0.01 Da (Orbitrap instrument precision).
pub const MZ_TOLERANCE: f64 = 0.01;

// ═══════════════════════════════════════════════════════════════════
// Diversity metric tolerances
// ═══════════════════════════════════════════════════════════════════

/// Shannon entropy range for simulated marine community.
///
/// Simulated 330-species community with Exp002 relative abundances.
/// Expected ~4.9 ± 0.3 (validated against skbio 0.6.0).
pub const SHANNON_SIMULATED: f64 = 0.3;

/// Simpson range for simulated marine community.
///
/// Simulated 330-species community with Exp002 relative abundances.
/// Expected ~0.9 ± 0.05 (validated against skbio 0.6.0, commit `e4358c5`).
pub const SIMPSON_SIMULATED: f64 = 0.05;

/// Chao1 range above observed features (singleton/doubleton dependent).
///
/// Chao1 ≥ observed by definition; for Exp002 Galaxy community
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
pub const PYTHON_PARITY: f64 = 1e-10;

/// Python-to-Rust parity for high-precision analytical formulas
/// (phylogenetic likelihood, clock rate estimation).
///
/// These are direct formula evaluations with minimal accumulation.
/// 1e-14 is ~1 ULP for typical magnitude-1 results.
pub const PYTHON_PARITY_TIGHT: f64 = 1e-14;

/// Python-to-Rust parity for statistical p-values and BH correction.
///
/// P-value computation involves log-gamma and combinatorial sums;
/// implementations may differ by a few ULP in tail behavior.
pub const PYTHON_PVALUE: f64 = 1e-5;

/// ML model parity (decision tree, random forest, GBM confidence).
///
/// Floating-point vote ratios and softmax normalization. 1e-10
/// matches Python `sklearn` / `xgboost` to full f64 precision.
pub const ML_PREDICTION: f64 = 1e-10;

/// Cross-species ANI comparison against Python baseline.
///
/// k-mer sampling and gap handling can differ at boundary codons.
/// 1e-4 covers implementation-level variation in ANI estimation.
pub const ANI_CROSS_SPECIES: f64 = 1e-4;

/// ML classification metrics (F1, accuracy) against Python baseline.
///
/// F1 score involves integer counts (TP, FP, FN) and division;
/// ties and rounding can differ by ~1e-4 between implementations.
pub const ML_F1_SCORE: f64 = 1e-4;

/// Evolutionary distance estimation (Jukes-Cantor, alignment-derived).
///
/// Distance matrix computation accumulates log/exp corrections over
/// pairwise alignment columns; ~1e-3 covers implementation variation.
pub const EVOLUTIONARY_DISTANCE: f64 = 1e-3;

/// Spectral cosine similarity rounding (m/z-shifted comparisons).
///
/// Greedy peak matching with floating-point intensity weighting
/// can differ by ~1e-3 at alignment boundaries.
pub const SPECTRAL_COSINE: f64 = 1e-3;

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
pub const ODE_STEADY_STATE: f64 = 0.01;

/// ODE method parity: RK4 vs LSODA integrator differences.
///
/// Fixed-step RK4 diverges from adaptive LSODA by up to ~1e-3 in
/// concentration for stiff systems. Validated against Python baselines.
pub const ODE_METHOD_PARITY: f64 = 1e-3;

/// Near-zero species concentrations (repressed pathways).
///
/// Biologically "off" species may float slightly above zero due to
/// integrator residual. 0.05 accommodates the numerical floor.
pub const ODE_NEAR_ZERO: f64 = 0.05;

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
/// Tighter than `MZ_TOLERANCE` for high-resolution fragment matching.
pub const MZ_FRAGMENT: f64 = 0.001;

/// Spectral matching m/z window for unit-resolution MS2.
///
/// Used as the matching window parameter for `cosine_similarity`.
pub const SPECTRAL_MZ_WINDOW: f64 = 0.5;

// ═══════════════════════════════════════════════════════════════════
// Peak detection / signal processing tolerances
// ═══════════════════════════════════════════════════════════════════

/// Relative peak height tolerance (1%) vs scipy baseline.
pub const PEAK_HEIGHT_REL: f64 = 0.01;

// ═══════════════════════════════════════════════════════════════════
// GPU vs CPU tolerances
// ═══════════════════════════════════════════════════════════════════

/// Regularized incomplete gamma series convergence epsilon.
///
/// Used by `special::regularized_gamma_p` for series termination.
/// Matches scipy's `gammainc` convergence behavior at 1e-15.
pub const GAMMA_SERIES_CONVERGENCE: f64 = 1e-15;

/// Maximum iterations for regularized gamma series expansion.
pub const GAMMA_SERIES_MAX_ITER: usize = 1000;

/// GPU f64 vs CPU f64 for exact arithmetic (add, mul, comparison).
///
/// GPU `SHADER_F64` uses IEEE 754 f64, but different instruction ordering
/// and FMA behavior can introduce small differences. On RTX 4070:
/// max observed diff 8e-8 for `exp()` (hotSpring Exp001 §4.4).
///
/// For simple add/mul chains (diversity metrics), expect < 1e-10.
pub const GPU_VS_CPU_F64: f64 = 1e-6;

/// GPU f64 vs CPU f64 for log/exp operations (transcendentals).
///
/// Native WGSL `log(f64)` on NVIDIA matches CPU to ~1e-14.
/// Allow 1e-10 for accumulated error in Shannon/Simpson computation.
pub const GPU_VS_CPU_TRANSCENDENTAL: f64 = 1e-10;

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
        assert!(GPU_VS_CPU_TRANSCENDENTAL <= GPU_VS_CPU_F64);
        assert!(GPU_VS_CPU_F64 <= GPU_VS_CPU_ENSEMBLE);
    }

    #[test]
    fn all_tolerances_are_non_negative() {
        let all = [
            EXACT,
            EXACT_F64,
            ANALYTICAL_F64,
            DNDS_OMEGA_GUARD,
            DADA2_ERR_CONVERGENCE,
            ODE_CDG_CONVERGENCE,
            PYTHON_PARITY,
            PYTHON_PARITY_TIGHT,
            PYTHON_PVALUE,
            ML_PREDICTION,
            ML_F1_SCORE,
            ANI_CROSS_SPECIES,
            EVOLUTIONARY_DISTANCE,
            SPECTRAL_COSINE,
            GC_CONTENT,
            MEAN_QUALITY,
            MZ_TOLERANCE,
            SHANNON_SIMULATED,
            SIMPSON_SIMULATED,
            CHAO1_RANGE,
            BRAY_CURTIS_SYMMETRY,
            ODE_STEADY_STATE,
            ODE_METHOD_PARITY,
            ODE_NEAR_ZERO,
            PHYLO_LIKELIHOOD,
            JC69_PROBABILITY,
            GILLESPIE_MEAN_REL,
            GILLESPIE_FANO,
            KMD_GROUPING,
            KMD_SPREAD,
            MZ_FRAGMENT,
            SPECTRAL_MZ_WINDOW,
            PEAK_HEIGHT_REL,
            GPU_VS_CPU_F64,
            GPU_VS_CPU_TRANSCENDENTAL,
            GPU_VS_CPU_BRAY_CURTIS,
            GPU_VS_CPU_ENSEMBLE,
        ];
        for tol in all {
            assert!(tol >= 0.0, "tolerance {tol} must be non-negative");
        }
    }
}
