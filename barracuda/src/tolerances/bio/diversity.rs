// SPDX-License-Identifier: AGPL-3.0-or-later
//! Diversity metric tolerances.

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

/// Pielou evenness J ≈ 1.0 tolerance for perfectly even communities.
///
/// J = H / ln(S) equals 1.0 for uniform abundance; 1% accommodates
/// f64 rounding in Shannon and Simpson chains.
pub const DIVERSITY_EVENNESS_TOL: f64 = 0.01;

/// Shannon timeseries monotonicity threshold.
///
/// When checking "Shannon varies over time," a decrease > 0.001 between
/// consecutive time points indicates meaningful temporal variation.
pub const DIVERSITY_TS_MONOTONIC: f64 = 0.001;

/// Shannon recovery near baseline after bloom.
///
/// Post-bloom community rebounds to within this tolerance of pre-bloom
/// Shannon entropy. 0.1 accommodates residual dominance from bloom species.
pub const SHANNON_RECOVERY_TOL: f64 = 0.1;

/// Chao1 range above observed features (singleton/doubleton dependent).
///
/// Chao1 ≥ observed by definition; for Exp002 `Galaxy` community
/// (330 species, ~910 expected), allow up to +100 above observed
/// (validated against skbio 0.6.0, commit `e4358c5`).
pub const CHAO1_RANGE: f64 = 100.0;

/// Bray-Curtis: f64 symmetry tolerance.
///
/// `BC(a,b)` and `BC(b,a)` involve identical f64 arithmetic but with
/// operands in different order. The sum `|a_i - b_i|` vs `|b_i - a_i|`
/// is exact (absolute value), and the denominator `Σ(a_i + b_i)` is
/// commutative. 1e-15 (≈4.5 × `f64::EPSILON`) covers any compiler
/// reordering of the summation accumulator.
/// Validated: Exp002 (phytoplankton diversity), `validate_diversity`,
/// commit `e4358c5`, 2026-02-19.
pub const BRAY_CURTIS_SYMMETRY: f64 = 1e-15;

/// Chao1 singleton/doubleton count detection half-width.
///
/// Counts within 0.5 of 1.0 or 2.0 are classified as singletons or
/// doubletons respectively. This accommodates floating-point abundance
/// values (e.g. rarefied counts) while remaining exact for integer data.
/// Matches the rounding behavior of `skbio.diversity.alpha.chao1`.
pub const CHAO1_COUNT_HALFWIDTH: f64 = 0.5;

/// Chao1 richness estimator vs R/vegan estimateR variance.
///
/// R/vegan estimateR uses bias correction; singleton/doubleton handling
/// differs slightly between implementations. 0.5 covers small-sample
/// variance and implementation differences for typical communities.
/// Validated: `validate_r_industry_parity` §4 (Chao1 `S.chao1`=20).
pub const CHAO1_ESTIMATE_VARIANCE: f64 = 0.5;

/// Shannon H' near-zero threshold for monoculture (single species).
///
/// Monoculture has H' = 0 exactly. Floating-point residual from
/// `diversity::shannon` on `[1000.0]` can be non-zero; 0.01 captures
/// any such residual for "effectively zero" checks.
/// Validated: `validate_anderson_qs_environments_v1` (E. coli monoculture).
pub const DIVERSITY_MONOCULTURE_NEAR_ZERO: f64 = 0.01;

/// Minimum GC content range (fraction) for genus-level diversity.
///
/// Vibrio genus assemblies should span ≥2% GC range to confirm
/// meaningful diversity. Used as threshold: `gc_range >= GC_GENUS_DIVERSITY_MIN`.
/// Validated: Exp228 (`validate_nucleus_pipeline_v69`).
pub const GC_GENUS_DIVERSITY_MIN: f64 = 0.02;

/// Rarefaction curve monotonicity guard.
///
/// Rarefaction is mathematically non-decreasing; small rounding in
/// hypergeometric terms can cause ≤ 1e-10 decreases. Same order as
/// Python parity (1e-10) but semantically a monotonicity check, not a
/// baseline-comparison tolerance.
pub const RAREFACTION_MONOTONIC: f64 = 1e-10;

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

/// Nanopore simulated 16S pipeline diversity tolerance.
///
/// Shannon diversity computed from long-read (nanopore-length) 16S reads
/// vs short-read (Illumina-length) reference. Nanopore's higher error
/// rate (~5-10% per-read) inflates observed OTU counts slightly, but
/// diversity metrics are robust. 0.3 covers the expected inflation.
/// Validated: Exp196b (simulated long-read 16S).
pub const NANOPORE_DIVERSITY_TOLERANCE: f64 = 0.3;

/// Cold seep pipeline Shannon entropy expectation range.
///
/// Ruff et al. (2019) cold seep communities: Shannon H' = 2.0–6.0.
/// Synthetic communities calibrated to published diversity ranges.
/// Source: Ruff et al., Nat. Microbiol. 2019, PRJNA315684.
/// Validated: Exp185 (cold seep metagenomes), commit `756df26`, 2026-02-26.
pub const COLD_SEEP_SHANNON_MIN: f64 = 2.0;

/// Upper bound for cold seep Shannon expectation range.
///
/// See [`COLD_SEEP_SHANNON_MIN`] for derivation.
pub const COLD_SEEP_SHANNON_MAX: f64 = 6.0;

/// Cold seep Simpson D minimum floor.
///
/// Published cold seep community Simpson D ≥ 0.7 across all samples
/// (high-diversity, multi-species sediment). Source: Ruff et al. 2019.
/// Validated: Exp185, commit `756df26`, 2026-02-26.
pub const COLD_SEEP_SIMPSON_MIN: f64 = 0.7;

/// Cold seep observed features minimum.
///
/// At least 50 OTUs expected per sample in cold seep 16S amplicons.
/// Source: Ruff et al. 2019 (170 samples, median `S_obs` > 100).
/// Validated: Exp185, commit `756df26`, 2026-02-26.
pub const COLD_SEEP_OBS_FEATURES_MIN: f64 = 50.0;

/// Rare biosphere lineage threshold (fraction of total abundance).
///
/// OTUs with relative abundance below this are classified as "rare"
/// lineages. 0.001 (0.1%) matches Anderson, Sogin & Baross (2015)
/// FEMS Microbiol Ecol 91:fiu016 methodology for vent community analysis.
/// Validated: Exp051 (rare biosphere), commit `e4358c5`.
pub const RARE_BIOSPHERE_THRESHOLD: f64 = 0.001;

/// Cold seep pipeline: minimum fraction classified as extended (QS viable).
///
/// Synthetic cold seep communities calibrated to Ruff et al. (2019)
/// parameters should produce > 60% samples in the extended (GOE-like)
/// spectral regime, confirming high-diversity deep-sea sediment supports
/// QS signaling.
/// Validated: Exp185 (cold seep pipeline), commit `756df26`.
pub const COLD_SEEP_EXTENDED_MIN_FRACTION: f64 = 0.60;

/// Minimum |Spearman ρ| for diversity–spectral-regime correlation.
///
/// Shannon H' and level-spacing ratio ⟨r⟩ should be positively correlated
/// in ecology datasets (higher diversity → more extended → higher r).
/// 0.1 is a weak-correlation floor that distinguishes genuine signal
/// from noise in small-sample (N ≈ 100) synthetic pipelines.
/// Validated: Exp185 (cold seep pipeline), commit `756df26`.
pub const DIVERSITY_SPECTRAL_SPEARMAN_MIN: f64 = 0.1;
