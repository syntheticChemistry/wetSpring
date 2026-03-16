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

/// PFSA homologue separation window (Da) for library filtering.
///
/// PFSA homologous series members (e.g., PFBS → `PFHxS` → PFOS) are
/// spaced ~50 Da apart. The filtering window of 60 Da allows matching
/// with margin for adduct variation and mass calibration drift.
/// Validated: Exp018 (PFAS library validation).
pub const PFSA_HOMOLOGUE_WINDOW: f64 = 60.0;

/// Retention time parse-back parity (minutes).
///
/// Retention time strings parsed from mzXML/mzML (e.g. "PT1.5S" → 0.025 min)
/// must round-trip within 1e-6 minutes of the reference value.
/// Validated: `io::mzxml` and `io::mzml` unit tests.
pub const RT_PARSE_PARITY: f64 = 1e-6;

/// JCAMP-DX Y-value parse precision.
///
/// Y-axis data in JCAMP-DX files (absorbance, transmittance) is parsed
/// from ASCII float strings. 1e-9 covers rounding in the ASCII→f64 path
/// for the typical 6–8 significant digit JCAMP encoding.
/// Validated: `io::jcamp` unit tests.
pub const JCAMP_Y_PARSE: f64 = 1e-9;

/// Feature m/z match tolerance (Da) for feature-table pipeline.
///
/// After EIC extraction and peak detection, features are matched to
/// expected targets within ±1.0 Da. Looser than instrument-grade
/// [`MZ_TOLERANCE`] because the feature table aggregates across scans.
/// Validated: Exp009 (asari pipeline), `bio::feature_table` tests.
pub const FEATURE_MZ_MATCH: f64 = 1.0;

/// Feature RT apex tolerance (minutes) for feature-table pipeline.
///
/// Extracted chromatographic peak apex should match expected retention
/// time within ±0.2 minutes. Covers scan-to-scan RT jitter in LC-MS.
/// Validated: Exp009 (asari pipeline), `bio::feature_table` tests.
pub const FEATURE_RT_APEX: f64 = 0.2;

/// Retention index paper deviation (%) for VOC identification.
///
/// Published Kovats RI values (Reese 2019 Table 1) are matched within
/// ±5% relative tolerance. Covers column-phase and temperature-program
/// variation between laboratories.
/// Validated: Exp013 (VOC peaks, Reese 2019).
pub const RI_PAPER_DEVIATION: f64 = 5.0;

/// Retention index search relative fraction (5% = 0.05).
///
/// The RI search window is `theoretical_ri × RI_SEARCH_RELATIVE`.
/// Validated: Exp013 (VOC peak validation).
pub const RI_SEARCH_RELATIVE: f64 = 0.05;

/// Retention index matching tolerance (unitless RI units).
///
/// Kovats or linear retention indices from GC-MS should match
/// literature/theoretical values within 0.1 RI units for confirmed
/// VOC identifications. Covers column-temperature variation and
/// calibrant drift.
/// Validated: Exp013 (VOC peak validation vs Reese 2019 Table 1).
pub const RETENTION_INDEX_MATCH: f64 = 0.1;
