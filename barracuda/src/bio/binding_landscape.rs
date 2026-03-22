// SPDX-License-Identifier: AGPL-3.0-or-later
//! Binding landscape and colonization resistance modeling.
//!
//! wetSpring's contribution to the joint healthSpring×wetSpring experiment.
//! Models how binding affinity, species diversity, and tissue disorder
//! interact to produce colonization resistance (gut) or selective targeting
//! (cancer/infection).
//!
//! # Key Concepts
//!
//! - **Composite binding**: Multiple weak interactions combine through
//!   coincidence detection — selectivity emerges from the product of
//!   individually insufficient binding events (NK cell activation model).
//! - **Colonization occupancy**: Fraction of lattice sites occupied above
//!   a minimal binding threshold. High occupancy = resistance to pathogen
//!   invasion.
//! - **Delocalized binding**: Weak binders spread load across many tissues,
//!   staying in the linear clearance regime and below repair capacity
//!   thresholds at every site. This is the Anderson extended state applied
//!   to drug safety.
//!
//! # Anderson Connection
//!
//! Strong binding → Anderson localization (concentrated at one site).
//! Weak binding → Anderson delocalization (spread across the lattice).
//! Colonization resistance is a delocalized phenomenon: many species, each
//! weakly bound, creating cumulative occupancy that excludes pathogens.
//! See healthSpring `TOXICITY.md` for the safety implications.

use barracuda::stats;

use crate::cast;

/// Parameters for a single binding interaction.
#[derive(Debug, Clone)]
pub struct BindingParams {
    /// Half-maximal binding concentration (IC50/Kd analog, in µM).
    pub kd: f64,
    /// Hill coefficient governing binding cooperativity.
    pub hill_n: f64,
}

/// Fractional occupancy at a single binding site.
///
/// Uses the Hill equation: `θ = [L]^n / (Kd^n + [L]^n)` where `[L]` is
/// ligand concentration.
#[must_use]
pub fn fractional_occupancy(concentration: f64, params: &BindingParams) -> f64 {
    if concentration <= 0.0 {
        return 0.0;
    }
    stats::hill(concentration, params.kd, params.hill_n)
}

/// Composite binding score from multiple weak interactions.
///
/// Models coincidence detection: the composite signal is the product of
/// fractional occupancies at each binding site. Each individual interaction
/// may be weak (`θ_i` < 0.5), but the composite exceeds the activation
/// threshold only when MULTIPLE interactions occur simultaneously.
///
/// This mirrors NK cell activation: no single activating receptor is
/// sufficient, but the integration of several weak signals triggers killing.
#[must_use]
pub fn composite_binding(occupancies: &[f64]) -> f64 {
    if occupancies.is_empty() {
        return 0.0;
    }
    occupancies.iter().product()
}

/// Selectivity index: ratio of on-target to off-target composite binding.
///
/// `SI = composite(target_occupancies) / composite(off_target_occupancies)`
///
/// High SI means the therapeutic has good selectivity through coincidence
/// detection even with individually weak binders. Returns `f64::INFINITY`
/// if off-target composite is zero.
#[must_use]
pub fn selectivity_index(target_occupancies: &[f64], off_target_occupancies: &[f64]) -> f64 {
    let on = composite_binding(target_occupancies);
    let off = composite_binding(off_target_occupancies);
    if off < f64::EPSILON {
        return if on > 0.0 { f64::INFINITY } else { 1.0 };
    }
    on / off
}

/// Result of a colonization resistance computation at a single point
/// in the (affinity, diversity, disorder) parameter space.
#[derive(Debug, Clone)]
pub struct ResistancePoint {
    /// Adhesion strength (Kd in µM).
    pub kd: f64,
    /// Number of colonizing species.
    pub n_species: usize,
    /// Epithelial disorder W (Anderson parameter).
    pub disorder_w: f64,
    /// Fraction of sites with occupancy above threshold.
    pub resistance_fraction: f64,
    /// Mean occupancy across all sites.
    pub mean_occupancy: f64,
    /// Whether the resistance exceeds the 90% threshold.
    pub is_resistant: bool,
}

/// Model colonization occupancy on a 1D epithelial lattice.
///
/// Each lattice site has a random on-site energy drawn from
/// `[-disorder_w/2, +disorder_w/2]`, representing variation in mucin
/// glycosylation or receptor density. Species bind with strength
/// modulated by this disorder.
///
/// - `n_sites`: number of epithelial binding sites
/// - `n_species`: number of colonizing species (each with slightly
///   different adhesion profile)
/// - `kd_base`: base adhesion Kd (µM)
/// - `disorder_w`: Anderson disorder parameter for epithelial heterogeneity
/// - `concentration`: effective bacterial concentration per species
/// - `seed`: RNG seed for deterministic disorder realization
///
/// Returns the fraction of sites with cumulative occupancy above `threshold`.
#[must_use]
pub fn colonization_resistance(
    n_sites: usize,
    n_species: usize,
    kd_base: f64,
    disorder_w: f64,
    concentration: f64,
    threshold: f64,
    seed: u64,
) -> ResistancePoint {
    let site_occupancies =
        site_occupancy_profile(n_sites, n_species, kd_base, disorder_w, concentration, seed);

    let occupied_count = site_occupancies.iter().filter(|&&o| o > threshold).count();

    let resistance_fraction = if n_sites == 0 {
        0.0
    } else {
        cast::usize_f64(occupied_count) / cast::usize_f64(n_sites)
    };

    let mean_occupancy = if site_occupancies.is_empty() {
        0.0
    } else {
        site_occupancies.iter().sum::<f64>() / cast::usize_f64(site_occupancies.len())
    };

    ResistancePoint {
        kd: kd_base,
        n_species,
        disorder_w,
        resistance_fraction,
        mean_occupancy,
        is_resistant: resistance_fraction > 0.9,
    }
}

/// Compute per-site occupancy profile on a disordered 1D lattice.
///
/// Each site gets a disorder-shifted effective Kd. Each species contributes
/// independently (no competitive exclusion in this simplified model).
/// Cumulative occupancy at each site is capped at 1.0.
#[must_use]
pub fn site_occupancy_profile(
    n_sites: usize,
    n_species: usize,
    kd_base: f64,
    disorder_w: f64,
    concentration: f64,
    seed: u64,
) -> Vec<f64> {
    (0..n_sites)
        .map(|site_idx| {
            let site_disorder = deterministic_disorder(site_idx, seed, disorder_w);
            let mut site_occ = 0.0;

            for species_idx in 0..n_species {
                let species_kd_shift = deterministic_species_shift(species_idx, seed);
                let effective_kd =
                    (kd_base * (1.0 + site_disorder) * (1.0 + species_kd_shift)).max(f64::EPSILON);

                site_occ = (site_occ + stats::hill(concentration, effective_kd, 1.0)).min(1.0);
            }

            site_occ
        })
        .collect()
}

/// Sweep the 3D parameter space for colonization resistance.
///
/// Returns one `ResistancePoint` per combination of (`kd`, `n_species`, `w`).
/// The resulting surface reveals where the resistance phase boundary lies.
#[must_use]
pub fn resistance_surface_sweep(
    kd_values: &[f64],
    n_species_values: &[usize],
    w_values: &[f64],
    concentration: f64,
    threshold: f64,
    seed: u64,
    n_sites: usize,
) -> Vec<ResistancePoint> {
    let mut results = Vec::with_capacity(kd_values.len() * n_species_values.len() * w_values.len());

    for (ki, &kd) in kd_values.iter().enumerate() {
        for (ni, &n_species) in n_species_values.iter().enumerate() {
            for (wi, &w) in w_values.iter().enumerate() {
                let point_seed = seed
                    .wrapping_add(cast::usize_u64(ki).wrapping_mul(10_000))
                    .wrapping_add(cast::usize_u64(ni).wrapping_mul(100))
                    .wrapping_add(cast::usize_u64(wi));
                results.push(colonization_resistance(
                    n_sites,
                    n_species,
                    kd,
                    w,
                    concentration,
                    threshold,
                    point_seed,
                ));
            }
        }
    }

    results
}

/// Inverse participation ratio (IPR) of a binding profile across tissues.
///
/// `IPR = Σ(θ_i⁴) / (Σ(θ_i²))²` where `θ_i` is fractional occupancy at tissue i.
///
/// - IPR → 1/N: delocalized (binding spread evenly across N tissues)
/// - IPR → 1: localized (binding concentrated at one tissue)
///
/// From healthSpring TOXICITY.md: delocalized binding is safer because no
/// single tissue bears disproportionate burden.
#[must_use]
pub fn binding_ipr(tissue_occupancies: &[f64]) -> f64 {
    if tissue_occupancies.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = tissue_occupancies.iter().map(|&o| o * o).sum();
    if sum_sq < f64::EPSILON {
        return 0.0;
    }
    let sum_fourth: f64 = tissue_occupancies.iter().map(|&o| o * o * o * o).sum();
    sum_fourth / (sum_sq * sum_sq)
}

/// Localization length from a binding profile.
///
/// ξ = 1 / IPR — the effective number of tissues sharing the binding load.
/// Higher ξ = more delocalized = safer toxicity profile.
#[must_use]
pub fn localization_length(tissue_occupancies: &[f64]) -> f64 {
    let ipr = binding_ipr(tissue_occupancies);
    if ipr < f64::EPSILON {
        return 0.0;
    }
    1.0 / ipr
}

/// Deterministic disorder for a lattice site (reproducible from seed).
///
/// Returns a value in `[-w/2, +w/2]` using a simple hash-based PRNG.
fn deterministic_disorder(site_idx: usize, seed: u64, w: f64) -> f64 {
    let hash = splitmix64(seed.wrapping_add(cast::usize_u64(site_idx)));
    let uniform = cast::u64_f64(hash) / cast::u64_f64(u64::MAX);
    w * (uniform - 0.5)
}

/// Deterministic species shift — gives each species a slightly different Kd.
///
/// Returns a value in `[-0.3, +0.3]` to model adhesin variation.
fn deterministic_species_shift(species_idx: usize, seed: u64) -> f64 {
    let hash = splitmix64(
        seed.wrapping_add(1_000_000)
            .wrapping_add(cast::usize_u64(species_idx)),
    );
    let uniform = cast::u64_f64(hash) / cast::u64_f64(u64::MAX);
    0.6 * (uniform - 0.5)
}

/// `SplitMix64` hash for deterministic pseudo-random generation.
const fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tolerances;

    #[test]
    fn fractional_occupancy_zero_concentration() {
        let params = BindingParams {
            kd: 10.0,
            hill_n: 1.0,
        };
        assert!(fractional_occupancy(0.0, &params).abs() < f64::EPSILON);
    }

    #[test]
    fn fractional_occupancy_at_kd_is_half() {
        let params = BindingParams {
            kd: 10.0,
            hill_n: 1.0,
        };
        let occ = fractional_occupancy(10.0, &params);
        assert!(
            (occ - 0.5).abs() < tolerances::IC50_RESPONSE_TOL,
            "occupancy at Kd should be 0.5: {occ}"
        );
    }

    #[test]
    fn fractional_occupancy_high_concentration() {
        let params = BindingParams {
            kd: 10.0,
            hill_n: 1.0,
        };
        let occ = fractional_occupancy(1e6, &params);
        assert!(
            (occ - 1.0).abs() < tolerances::ASYMPTOTIC_LIMIT,
            "high concentration should saturate: {occ}"
        );
    }

    #[test]
    fn composite_binding_empty() {
        assert!(composite_binding(&[]).abs() < f64::EPSILON);
    }

    #[test]
    fn composite_binding_single_strong() {
        assert!((composite_binding(&[0.9]) - 0.9).abs() < tolerances::ANALYTICAL_F64);
    }

    #[test]
    fn composite_binding_multiple_weak() {
        let weak = [0.3, 0.3, 0.3, 0.3];
        let composite = composite_binding(&weak);
        assert!(
            (composite - 0.3_f64.powi(4)).abs() < tolerances::ANALYTICAL_F64,
            "composite of 4 weak binders: {composite}"
        );
        assert!(
            composite < 0.01,
            "4 weak binders should give very low composite: {composite}"
        );
    }

    #[test]
    fn selectivity_from_coincidence() {
        let target = [0.8, 0.7, 0.6, 0.5];
        let off_target = [0.1, 0.1, 0.1, 0.1];
        let si = selectivity_index(&target, &off_target);
        assert!(
            si > 100.0,
            "coincidence detection should give high selectivity: {si}"
        );
    }

    #[test]
    fn colonization_single_species_vs_many() {
        let single = colonization_resistance(100, 1, 50.0, 2.0, 5.0, 0.5, 42);
        let many = colonization_resistance(100, 15, 50.0, 2.0, 5.0, 0.5, 42);
        assert!(
            many.resistance_fraction > single.resistance_fraction,
            "more species should increase resistance: many={:.3} > single={:.3}",
            many.resistance_fraction,
            single.resistance_fraction
        );
    }

    #[test]
    fn colonization_many_species_high_resistance() {
        let point = colonization_resistance(100, 15, 10.0, 0.5, 5.0, 0.1, 42);
        assert!(
            point.resistance_fraction > point.mean_occupancy * 0.5,
            "many species should increase resistance: frac={:.3}, mean={:.3}",
            point.resistance_fraction,
            point.mean_occupancy
        );
    }

    #[test]
    fn resistance_surface_produces_correct_count() {
        let kds = [1.0, 10.0, 100.0];
        let species = [1, 5, 15];
        let ws = [0.1, 1.0, 3.0];
        let surface = resistance_surface_sweep(&kds, &species, &ws, 5.0, 0.1, 42, 50);
        assert_eq!(surface.len(), 27, "3×3×3 = 27 points");
    }

    #[test]
    fn ipr_uniform_is_delocalized() {
        let uniform = [0.5; 8];
        let ipr = binding_ipr(&uniform);
        assert!(
            (ipr - 1.0 / 8.0).abs() < tolerances::ANALYTICAL_F64,
            "uniform binding should give IPR = 1/N: {ipr}"
        );
    }

    #[test]
    fn ipr_concentrated_is_localized() {
        let concentrated = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let ipr = binding_ipr(&concentrated);
        assert!(
            (ipr - 1.0).abs() < tolerances::ANALYTICAL_F64,
            "concentrated binding should give IPR = 1: {ipr}"
        );
    }

    #[test]
    fn localization_length_uniform() {
        let uniform = [0.5; 8];
        let xi = localization_length(&uniform);
        assert!(
            (xi - 8.0).abs() < tolerances::ANALYTICAL_F64,
            "uniform gives ξ = N: {xi}"
        );
    }

    #[test]
    fn localization_length_concentrated() {
        let concentrated = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let xi = localization_length(&concentrated);
        assert!(
            (xi - 1.0).abs() < tolerances::ANALYTICAL_F64,
            "concentrated gives ξ = 1: {xi}"
        );
    }

    #[test]
    fn deterministic_disorder_reproducible() {
        let d1 = deterministic_disorder(5, 42, 2.0);
        let d2 = deterministic_disorder(5, 42, 2.0);
        assert!(
            (d1 - d2).abs() < f64::EPSILON,
            "same seed+site should give same disorder"
        );
    }

    #[test]
    fn deterministic_disorder_in_range() {
        for site in 0..100 {
            let d = deterministic_disorder(site, 42, 4.0);
            assert!(
                (-2.0..=2.0).contains(&d),
                "disorder should be in [-W/2, W/2]: site={site}, d={d}"
            );
        }
    }

    #[test]
    fn splitmix64_varies() {
        let a = splitmix64(0);
        let b = splitmix64(1);
        assert_ne!(a, b, "different seeds should give different hashes");
    }
}
