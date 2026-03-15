// SPDX-License-Identifier: AGPL-3.0-or-later
//! Campylobacterota assembly records (Exp125).

/// Campylobacterota assembly record (Exp125).
#[derive(Debug, Clone)]
pub struct CampyAssembly {
    /// NCBI accession.
    pub accession: String,
    /// Full organism name.
    pub organism: String,
    /// Genus grouping (Campylobacter, Helicobacter, Sulfurimonas, etc.).
    pub genus: String,
    /// Genome size in base pairs.
    pub genome_size_bp: u64,
    /// Annotated gene count.
    pub gene_count: u32,
    /// Isolation source metadata.
    pub isolation_source: String,
}

impl CampyAssembly {
    /// Parse a `CampyAssembly` from a JSON object string.
    #[must_use]
    pub fn from_json_obj(obj: &str) -> Self {
        Self {
            accession: super::json_str_value(obj, "accession"),
            organism: super::json_str_value(obj, "organism"),
            genus: super::json_str_value(obj, "genus"),
            genome_size_bp: super::json_int_value(obj, "genome_size_bp"),
            gene_count: super::json_int_value(obj, "gene_count") as u32,
            isolation_source: super::json_str_value(obj, "isolation_source"),
        }
    }
}

/// Load Campylobacterota assemblies from NCBI JSON data.
///
/// # Errors
///
/// Returns `Err` if the JSON file is missing or contains no valid records.
pub fn try_load_campylobacterota() -> crate::error::Result<Vec<CampyAssembly>> {
    let path = super::data_dir().join("campylobacterota_assemblies.json");
    super::try_load_json_array(&path, "assemblies", CampyAssembly::from_json_obj, |a| {
        !a.accession.is_empty()
    })
}

/// Load Campylobacterota assemblies, falling back to synthetic data when offline.
///
/// Returns `(records, is_real_data)`. See [`try_load_campylobacterota`] for
/// explicit error handling.
#[must_use]
pub fn load_campylobacterota() -> (Vec<CampyAssembly>, bool) {
    try_load_campylobacterota().map_or_else(|_| (gen_synthetic_campy(), false), |a| (a, true))
}

/// Synthetic data: 80 assemblies across genera and ecosystems.
///
/// Deterministic accessions (`GCF_CAM_*`). Used only as offline/CI fallback.
#[expect(clippy::cast_precision_loss)] // Precision: genome size and counts fit f64
pub fn gen_synthetic_campy() -> Vec<CampyAssembly> {
    let genera = [
        (
            "Campylobacter",
            "Campylobacter jejuni",
            1_700_000_u64,
            1700_u32,
            "gut",
        ),
        (
            "Campylobacter",
            "Campylobacter coli",
            1_800_000,
            1750,
            "food",
        ),
        (
            "Helicobacter",
            "Helicobacter pylori",
            1_600_000,
            1550,
            "gut",
        ),
        (
            "Helicobacter",
            "Helicobacter hepaticus",
            1_800_000,
            1700,
            "gut",
        ),
        (
            "Sulfurimonas",
            "Sulfurimonas denitrificans",
            2_200_000,
            2100,
            "vent",
        ),
        (
            "Sulfurimonas",
            "Sulfurimonas autotrophica",
            2_100_000,
            2000,
            "vent",
        ),
        (
            "Sulfurospirillum",
            "Sulfurospirillum multivorans",
            3_200_000,
            3000,
            "sediment",
        ),
        (
            "Arcobacter",
            "Arcobacter butzleri",
            2_300_000,
            2200,
            "water",
        ),
        ("Nautilia", "Nautilia profundicola", 1_700_000, 1600, "vent"),
    ];

    let mut rng = 99_u64;
    let mut assemblies = Vec::with_capacity(80);
    for i in 0..80_u32 {
        let g = &genera[(i as usize) % genera.len()];
        rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let var = (f64::from((rng >> 33) as u32) / f64::from(u32::MAX)).mul_add(0.1, -0.05);
        assemblies.push(CampyAssembly {
            accession: format!("GCF_CAM_{i:04}"),
            organism: format!("{} strain SYN{i:03}", g.1),
            genus: g.0.to_string(),
            genome_size_bp: ((g.2 as f64) * (1.0 + var)) as u64,
            gene_count: (f64::from(g.3) * (1.0 + var)) as u32,
            isolation_source: g.4.to_string(),
        });
    }
    assemblies
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn load_from_dir(dir: &std::path::Path) -> (Vec<CampyAssembly>, bool) {
        super::super::try_load_json_array(
            &dir.join("campylobacterota_assemblies.json"),
            "assemblies",
            CampyAssembly::from_json_obj,
            |a| !a.accession.is_empty(),
        )
        .map_or_else(|_| (gen_synthetic_campy(), false), |a| (a, true))
    }

    #[test]
    fn from_json_obj_parses_fields() {
        let obj = r#"{"accession": "GCF_001234", "organism": "Campylobacter jejuni", "genus": "Campylobacter", "genome_size_bp": 1700000, "gene_count": 1700, "isolation_source": "gut"}"#;
        let a = CampyAssembly::from_json_obj(obj);
        assert_eq!(a.accession, "GCF_001234");
        assert_eq!(a.genus, "Campylobacter");
        assert_eq!(a.genome_size_bp, 1_700_000);
    }

    #[test]
    fn forced_synthetic() {
        let temp = TempDir::new().unwrap();
        let (assemblies, is_real) = load_from_dir(temp.path());
        assert!(!is_real);
        assert_eq!(assemblies.len(), 80);
        assert!(assemblies[0].accession.starts_with("GCF_CAM_"));
    }

    #[test]
    fn genera_are_diverse() {
        let (assemblies, _) = load_campylobacterota();
        let genera: std::collections::HashSet<&str> =
            assemblies.iter().map(|a| a.genus.as_str()).collect();
        assert!(genera.len() >= 5);
    }

    #[test]
    fn synthetic_fallback_via_public_api() {
        let (assemblies, is_real) = load_campylobacterota();
        assert!(!assemblies.is_empty());
        if !is_real {
            assert_eq!(assemblies.len(), 80);
            assert!(!assemblies[0].genus.is_empty());
        }
    }
}
