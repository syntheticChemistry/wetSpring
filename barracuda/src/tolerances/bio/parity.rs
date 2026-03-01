// SPDX-License-Identifier: AGPL-3.0-or-later
//! Python baseline parity and ML tolerances.

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

/// ML classification metrics (F1, accuracy) against Python baseline.
///
/// F1 score involves integer counts (TP, FP, FN) and division;
/// ties and rounding can differ by ~1e-4 between implementations.
/// Calibrated: Exp008/041 (PFAS classification, EPA ML).
/// Scripts: `exp008_pfas_ml_baseline.py`, `epa_pfas_ml_baseline.py`.
/// Commit `48fb787`.
pub const ML_F1_SCORE: f64 = 1e-4;
