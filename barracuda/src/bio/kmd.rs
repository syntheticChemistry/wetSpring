// SPDX-License-Identifier: AGPL-3.0-or-later
//! Kendrick Mass Defect (KMD) analysis for PFAS homologue detection.
//!
//! Kendrick mass normalization converts exact masses so that homologous
//! series (differing by a repeating unit) share the same mass defect.
//! For PFAS, the primary repeating unit is CF₂ (49.99681 Da, nominal 50).
//!
//! # Algorithm
//!
//! 1. Kendrick mass: KM = exact\_mass × (nominal\_unit / exact\_unit)
//! 2. Kendrick mass defect: KMD = nominal\_KM − KM (or KM − floor(KM))
//! 3. Group features by KMD within tolerance → homologous series
//!
//! # References
//!
//! - Kendrick, E. (1963). Anal. Chem. 35(13): 2146–2154.
//! - Zweigle, J. et al. "`PFΔScreen`." Anal. Bioanal. Chem. (2023).
//! - Luo, Y.-R. et al. "KMD plot for PFAS screening." JASMS (2020).

/// Common repeating units for Kendrick analysis.
pub mod units {
    /// CF₂ (perfluoromethylene): exact mass 49.99681 Da, nominal 50.
    pub const CF2_EXACT: f64 = 49.996_806_03;
    /// CF₂ nominal mass.
    pub const CF2_NOMINAL: f64 = 50.0;
    /// CH₂ (methylene): exact mass 14.01565 Da, nominal 14.
    pub const CH2_EXACT: f64 = 14.015_650_64;
    /// CH₂ nominal mass.
    pub const CH2_NOMINAL: f64 = 14.0;
    /// C₂F₄ (two CF₂ units): exact mass 99.99361 Da, nominal 100.
    pub const C2F4_EXACT: f64 = 99.993_612_06;
    /// C₂F₄ nominal mass.
    pub const C2F4_NOMINAL: f64 = 100.0;
}

/// Kendrick mass defect result for a single feature.
#[derive(Debug, Clone, PartialEq)]
pub struct KmdResult {
    /// Original exact mass.
    pub exact_mass: f64,
    /// Kendrick mass (rescaled by repeating unit).
    pub kendrick_mass: f64,
    /// Kendrick mass defect (fractional part of Kendrick mass).
    pub kmd: f64,
    /// Nominal Kendrick mass (floor).
    pub nominal_km: f64,
}

/// Compute Kendrick mass defect for an array of exact masses.
///
/// # Arguments
///
/// * `exact_masses` — Measured exact masses (Da).
/// * `exact_unit` — Exact mass of the repeating unit (e.g., [`units::CF2_EXACT`]).
/// * `nominal_unit` — Nominal mass of the repeating unit (e.g., [`units::CF2_NOMINAL`]).
///
/// # Returns
///
/// Vector of [`KmdResult`] in the same order as input masses.
#[must_use]
pub fn kendrick_mass_defect(
    exact_masses: &[f64],
    exact_unit: f64,
    nominal_unit: f64,
) -> Vec<KmdResult> {
    let scale = nominal_unit / exact_unit;

    exact_masses
        .iter()
        .map(|&mass| {
            let km = mass * scale;
            let nominal = km.floor();
            let kmd = nominal - km; // Convention: KMD can be negative

            KmdResult {
                exact_mass: mass,
                kendrick_mass: km,
                kmd,
                nominal_km: nominal,
            }
        })
        .collect()
}

/// Group features into homologous series by KMD similarity.
///
/// Features with KMD values within `kmd_tolerance` are grouped together.
/// Each group represents a potential homologous series.
///
/// # Arguments
///
/// * `kmd_results` — KMD results from [`kendrick_mass_defect`].
/// * `kmd_tolerance` — Maximum KMD difference to consider members of the
///   same series (typically 0.001–0.005 for PFAS).
///
/// # Returns
///
/// Vector of groups, where each group is a vector of indices into `kmd_results`.
#[must_use]
pub fn group_homologues(kmd_results: &[KmdResult], kmd_tolerance: f64) -> Vec<Vec<usize>> {
    if kmd_results.is_empty() {
        return vec![];
    }

    // Sort by KMD for efficient grouping
    let mut sorted_indices: Vec<usize> = (0..kmd_results.len()).collect();
    sorted_indices.sort_by(|&a, &b| {
        kmd_results[a]
            .kmd
            .partial_cmp(&kmd_results[b].kmd)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut current_group = vec![sorted_indices[0]];

    for &idx in &sorted_indices[1..] {
        // Safety: current_group always has at least one element (initialized above,
        // and re-pushed after each take)
        let last_idx = current_group[current_group.len() - 1];
        let last_kmd = kmd_results[last_idx].kmd;
        let this_kmd = kmd_results[idx].kmd;

        if (this_kmd - last_kmd).abs() <= kmd_tolerance {
            current_group.push(idx);
        } else {
            groups.push(std::mem::take(&mut current_group));
            current_group.push(idx);
        }
    }
    if !current_group.is_empty() {
        groups.push(current_group);
    }

    groups
}

/// Convenience: CF₂-based KMD analysis for PFAS screening.
///
/// Computes KMD using the CF₂ repeating unit and groups into homologous series.
///
/// # Returns
///
/// `(kmd_results, groups)` — KMD values and grouped indices.
#[must_use]
pub fn pfas_kmd_screen(
    exact_masses: &[f64],
    kmd_tolerance: f64,
) -> (Vec<KmdResult>, Vec<Vec<usize>>) {
    let results = kendrick_mass_defect(exact_masses, units::CF2_EXACT, units::CF2_NOMINAL);
    let groups = group_homologues(&results, kmd_tolerance);
    (results, groups)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cf2_kmd_homologues() {
        // PFOS (C8F17SO3H) ≈ 498.930 Da
        // PFHxS (C6F13SO3H) ≈ 398.936 Da — differs by 1 CF2 unit
        // PFBS (C4F9SO3H) ≈ 298.943 Da — differs by 2 CF2 units
        let masses = vec![498.930, 398.936, 298.943];
        let results = kendrick_mass_defect(&masses, units::CF2_EXACT, units::CF2_NOMINAL);

        assert_eq!(results.len(), 3);

        // KMDs should be very similar for homologues
        let kmd_spread = (results[0].kmd - results[1].kmd).abs();
        assert!(kmd_spread < 0.01, "KMD spread {kmd_spread} too large");
    }

    #[test]
    fn grouping_separates_series() {
        // Two homologous series with different KMDs
        let masses = vec![
            100.0, 150.0, 200.0, // series A
            113.5, 163.5, 213.5, // series B — different KMD
        ];
        let results = kendrick_mass_defect(&masses, units::CF2_EXACT, units::CF2_NOMINAL);
        let groups = group_homologues(&results, 0.01);

        // Should have at least 2 distinct groups
        assert!(
            groups.len() >= 2,
            "expected ≥2 groups, got {}",
            groups.len()
        );
    }

    #[test]
    fn pfas_screen_convenience() {
        let masses = vec![498.930, 398.936, 298.943, 600.0];
        let (results, groups) = pfas_kmd_screen(&masses, 0.005);
        assert_eq!(results.len(), 4);
        assert!(!groups.is_empty());
    }

    #[test]
    fn ch2_kmd() {
        // CH2 series: alkanes differ by CH2 (14.01565 Da)
        let masses = vec![100.0, 114.016, 128.031];
        let results = kendrick_mass_defect(&masses, units::CH2_EXACT, units::CH2_NOMINAL);
        let kmd_spread = (results[0].kmd - results[1].kmd).abs();
        assert!(kmd_spread < 0.01, "CH2 KMD spread {kmd_spread}");
    }

    #[test]
    fn empty_input() {
        let results = kendrick_mass_defect(&[], units::CF2_EXACT, units::CF2_NOMINAL);
        assert!(results.is_empty());

        let groups = group_homologues(&[], 0.005);
        assert!(groups.is_empty());
    }

    #[test]
    fn single_mass() {
        let results = kendrick_mass_defect(&[500.0], units::CF2_EXACT, units::CF2_NOMINAL);
        assert_eq!(results.len(), 1);

        let groups = group_homologues(&results, 0.005);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 1);
    }
}
