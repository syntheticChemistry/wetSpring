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
pub const SIMPSON_SIMULATED: f64 = 0.05;

/// Chao1 range above observed features (singleton/doubleton dependent).
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
// GPU vs CPU tolerances
// ═══════════════════════════════════════════════════════════════════

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
