// SPDX-License-Identifier: AGPL-3.0-or-later
//! Instrument and measurement tolerances: m/z, retention time, GC, PFAS, etc.

// ═══════════════════════════════════════════════════════════════════
// Sequencing / FastQC baselines
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

// ═══════════════════════════════════════════════════════════════════
// Mass spectrometry (m/z, ppm)
// ═══════════════════════════════════════════════════════════════════

/// m/z tolerance: ±0.01 Da (Orbitrap instrument precision).
///
/// Standard mass accuracy for high-resolution Orbitrap LC-MS instruments
/// (Thermo Scientific, sub-5 ppm at 200–1000 Da). Calibrated against EPA
/// Method 533 reference standards (Exp005/006/007).
/// Validated: `scripts/validate_track2.py`, commit `48fb787`.
pub const MZ_TOLERANCE: f64 = 0.01;

/// Fragment m/z tolerance (Da) for suspect screening.
///
/// Tighter than [`MZ_TOLERANCE`] for high-resolution MS2 fragment
/// matching. Derived from Orbitrap MS2 mass accuracy (sub-2 ppm at
/// typical fragment m/z 50–500 Da).
/// Calibrated: Exp006 (PFAS suspect screening), Exp042 (`MassBank`).
/// Scripts: `validate_track2.py`, `massbank_spectral_baseline.py`.
/// Commit `48fb787`.
pub const MZ_FRAGMENT: f64 = 0.001;

/// Spectral matching m/z window for unit-resolution MS2.
///
/// Used as the matching window parameter for `cosine_similarity`.
/// Standard unit-resolution window for ion-trap / triple-quad MS2.
/// Calibrated: Exp005 (mzML cosine), Exp042 (`MassBank` spectral).
/// Scripts: `validate_track2.py`, `massbank_spectral_baseline.py`.
/// Commit `48fb787`.
pub const SPECTRAL_MZ_WINDOW: f64 = 0.5;

/// Parts-per-million conversion factor (1 ppm = 1e-6).
///
/// Used by EIC extraction and tolerance-based m/z matching to convert
/// ppm parameters to fractional Da tolerances: `tol = mz * ppm * PPM_FACTOR`.
pub const PPM_FACTOR: f64 = 1e-6;

// ═══════════════════════════════════════════════════════════════════
// PFAS / analytical chemistry (KMD)
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

/// KMD non-homologue separation threshold.
///
/// Compounds from *different* PFAS series (e.g. PFCAs vs sulfonates +
/// non-PFAS) should differ by > 0.005 KMD units. Tighter than
/// [`KMD_GROUPING`] to verify that non-homologues are correctly excluded.
/// Validated: Exp006 (EPA Method 533), commit `eb99b12`.
pub const KMD_NON_HOMOLOGUE: f64 = 0.005;

// ═══════════════════════════════════════════════════════════════════
// EIC / Extracted Ion Chromatogram
// ═══════════════════════════════════════════════════════════════════

/// EIC trapezoid area integration tolerance.
///
/// Trapezoid rule integration of extracted ion chromatogram peaks
/// matches analytical expectation within 1% (same order as [`KMD_GROUPING`]).
/// Validated: Exp252 (`BarraCuda` CPU v19), D24 EIC domain.
pub const EIC_TRAPEZOID: f64 = 0.01;

// ═══════════════════════════════════════════════════════════════════
// Peak detection / signal processing
// ═══════════════════════════════════════════════════════════════════

/// Relative peak height tolerance (1%) vs `scipy` baseline.
///
/// Peak detection prominence and height differ by ≤1% between
/// `scipy.signal.find_peaks` and the Rust implementation due to
/// interpolation at boundary samples.
/// Calibrated: Exp010 (peak detection).
/// Script: `generate_peak_baselines.py`. Commit `48fb787`.
pub const PEAK_HEIGHT_REL: f64 = 0.01;

/// Minimum prominence for peak detection (Gaussian/Lorentzian).
///
/// Peaks with prominence below this are filtered. 0.05 (5% of typical
/// signal range) distinguishes true peaks from noise in synthetic
/// validation signals.
/// Validated: Exp102 (`BarraCuda` CPU v8 signal module).
pub const PEAK_MIN_PROMINENCE: f64 = 0.05;

// ═══════════════════════════════════════════════════════════════════
// PFAS / m/z search
// ═══════════════════════════════════════════════════════════════════

/// m/z exact match tolerance after tolerance search.
///
/// When `find_within_ppm` or `find_within_da` returns indices, the
/// matched mass should equal the query within this tolerance (exact
/// mass comparison). 1e-6 Da covers f64 rounding at typical PFAS m/z.
/// Validated: Exp018 (PFAS library validation).
pub const MZ_SEARCH_EXACT: f64 = 1e-6;

/// m/z relaxed match for edge-case validation.
///
/// For ppm-window boundary tests (e.g. 4.9 ppm shift at 5 ppm window),
/// slightly relaxed tolerance accounts for binning at search boundaries.
/// Validated: Exp018 (tolerance edge cases).
pub const MZ_SEARCH_RELAXED: f64 = 1e-4;

/// CF₂ homologue spacing validation tolerance (Da).
///
/// PFCA/PFSA adjacent members differ by ~49.997 Da (CF₂). Mean spacing
/// from library should match within 0.01 Da for correct series detection.
/// Validated: Exp018 (cross-series discrimination).
pub const CF2_SPACING_TOL: f64 = 0.01;
