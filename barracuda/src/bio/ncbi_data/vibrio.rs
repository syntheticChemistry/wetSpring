// SPDX-License-Identifier: AGPL-3.0-or-later
//! Vibrio genome assembly records (Exp121).

/// Vibrio genome assembly record (Exp121).
#[derive(Debug, Clone)]
pub struct VibrioAssembly {
    /// NCBI accession (e.g., "`GCF_000006745.1`").
    pub accession: String,
    /// Full organism name (e.g., "Vibrio cholerae O1 biovar El Tor str. N16961").
    pub organism: String,
    /// Total genome size in base pairs.
    pub genome_size_bp: u64,
    /// Number of annotated protein-coding genes.
    pub gene_count: u32,
    /// Number of scaffolds / contigs.
    pub scaffold_count: u32,
    /// Isolation source or infraspecific name (clinical, environmental, etc.).
    pub isolation_source: String,
}

impl VibrioAssembly {
    /// Parse a `VibrioAssembly` from a JSON object string.
    ///
    /// NCBI `gene_count` and `scaffold_count` fit in u32 for all known assemblies.
    #[must_use]
    #[expect(clippy::cast_possible_truncation)]
    pub fn from_json_obj(obj: &str) -> Self {
        Self {
            accession: super::json_str_value(obj, "accession"),
            organism: super::json_str_value(obj, "organism"),
            genome_size_bp: super::json_int_value(obj, "genome_size_bp"),
            gene_count: super::json_int_value(obj, "gene_count") as u32,
            scaffold_count: super::json_int_value(obj, "scaffold_count") as u32,
            isolation_source: super::json_str_value(obj, "isolation_source"),
        }
    }
}

/// Load Vibrio assemblies from NCBI JSON data.
///
/// Returns an error when the data file is absent or unparseable.
/// Callers that need offline/CI fallback should use
/// [`load_vibrio_assemblies`] instead.
///
/// # Errors
///
/// Returns `Err` if the JSON file is missing or contains no valid records.
pub fn try_load_vibrio_assemblies() -> crate::error::Result<Vec<VibrioAssembly>> {
    let path = super::data_dir().join("vibrio_assemblies.json");
    super::try_load_json_array(&path, "assemblies", VibrioAssembly::from_json_obj, |a| {
        !a.accession.is_empty()
    })
}

/// Load Vibrio assemblies, falling back to synthetic data when offline.
///
/// Returns `(records, is_real_data)`. Validation binaries use this for
/// CI/offline resilience. Library consumers should prefer
/// [`try_load_vibrio_assemblies`] for explicit error handling.
#[must_use]
pub fn load_vibrio_assemblies() -> (Vec<VibrioAssembly>, bool) {
    try_load_vibrio_assemblies().map_or_else(|_| (gen_synthetic_vibrio(), false), |a| (a, true))
}

/// Synthetic data: 150 assemblies mirroring real Vibrio diversity.
///
/// Deterministic accessions (`GCF_SYN_*`) so callers can distinguish
/// synthetic from real data. Used only as offline/CI fallback.
///
/// `rng>>33` yields ≤2^31; `genome_size`/`gene_count`/`scaffold_count` fit target types.
#[expect(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
pub fn gen_synthetic_vibrio() -> Vec<VibrioAssembly> {
    let mut rng = 42_u64;
    let species = [
        ("Vibrio cholerae", 4_000_000_u64, 3800_u32),
        ("Vibrio parahaemolyticus", 5_100_000, 4700),
        ("Vibrio vulnificus", 5_000_000, 4500),
        ("Vibrio harveyi", 6_000_000, 5500),
        ("Vibrio fischeri", 4_300_000, 3900),
        ("Vibrio alginolyticus", 5_500_000, 5000),
        ("Vibrio campbellii", 5_800_000, 5200),
        ("Vibrio natriegens", 4_600_000, 4100),
        ("Vibrio anguillarum", 4_100_000, 3700),
        ("Vibrio splendidus", 5_200_000, 4800),
    ];
    let sources = [
        "clinical",
        "environmental",
        "aquaculture",
        "marine",
        "estuarine",
    ];

    let mut assemblies = Vec::with_capacity(150);
    for i in 0..150_u32 {
        rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let sp = &species[(i as usize) % species.len()];
        let size_var = (f64::from((rng >> 33) as u32) / f64::from(u32::MAX)).mul_add(0.2, -0.1);
        let genome_size = ((sp.1 as f64) * (1.0 + size_var)) as u64;
        rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let gene_var = (f64::from((rng >> 33) as u32) / f64::from(u32::MAX)).mul_add(0.15, -0.075);
        let gene_count = (f64::from(sp.2) * (1.0 + gene_var)) as u32;
        rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let src_idx = ((rng >> 33) as usize) % sources.len();

        assemblies.push(VibrioAssembly {
            accession: format!("GCF_SYN_{i:04}"),
            organism: format!("{} strain SYN{i:03}", sp.0),
            genome_size_bp: genome_size,
            gene_count,
            scaffold_count: if i % 5 == 0 { 1 } else { 2 + (i % 8) },
            isolation_source: sources[src_idx].to_string(),
        });
    }
    assemblies
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn load_from_dir(dir: &std::path::Path) -> (Vec<VibrioAssembly>, bool) {
        super::super::try_load_json_array(
            &dir.join("vibrio_assemblies.json"),
            "assemblies",
            VibrioAssembly::from_json_obj,
            |a| !a.accession.is_empty(),
        )
        .map_or_else(|_| (gen_synthetic_vibrio(), false), |a| (a, true))
    }

    #[test]
    fn from_json_obj_parses_fields() {
        let obj = r#"{"accession": "GCF_000006745.1", "organism": "Vibrio cholerae", "genome_size_bp": 4033464, "gene_count": 3835, "scaffold_count": 2, "isolation_source": "clinical"}"#;
        let a = VibrioAssembly::from_json_obj(obj);
        assert_eq!(a.accession, "GCF_000006745.1");
        assert_eq!(a.organism, "Vibrio cholerae");
        assert_eq!(a.genome_size_bp, 4_033_464);
        assert_eq!(a.gene_count, 3835);
        assert_eq!(a.scaffold_count, 2);
        assert_eq!(a.isolation_source, "clinical");
    }

    #[test]
    fn forced_synthetic() {
        let temp = TempDir::new().unwrap();
        let (assemblies, is_real) = load_from_dir(temp.path());
        assert!(!is_real, "should use synthetic when data dir is empty");
        assert_eq!(assemblies.len(), 150);
        assert!(assemblies[0].accession.starts_with("GCF_SYN_"));
        assert!(assemblies[0].genome_size_bp > 0);
        assert!(assemblies[0].gene_count > 0);
    }

    #[test]
    fn json_path_loads() {
        let temp = TempDir::new().unwrap();
        let json_path = temp.path().join("vibrio_assemblies.json");
        let json = r#"{"assemblies": [{"accession": "GCF_TEST_001", "organism": "Vibrio test", "genome_size_bp": 4000000, "gene_count": 3800, "scaffold_count": 2, "isolation_source": "marine"}]}"#;
        std::fs::write(&json_path, json).unwrap();
        let (assemblies, is_real) = load_from_dir(temp.path());
        assert!(is_real, "should use real JSON when file exists");
        assert_eq!(assemblies.len(), 1);
        assert_eq!(assemblies[0].accession, "GCF_TEST_001");
    }

    #[test]
    fn try_load_missing_file_returns_error() {
        let temp = TempDir::new().unwrap();
        let result = super::super::try_load_json_array::<VibrioAssembly>(
            &temp.path().join("nonexistent.json"),
            "assemblies",
            VibrioAssembly::from_json_obj,
            |a| !a.accession.is_empty(),
        );
        assert!(result.is_err(), "missing file should return Err");
    }

    #[test]
    fn try_load_invalid_json_returns_error() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("vibrio_assemblies.json");
        std::fs::write(&path, "not valid json at all").unwrap();
        let result = super::super::try_load_json_array::<VibrioAssembly>(
            &path,
            "assemblies",
            VibrioAssembly::from_json_obj,
            |a| !a.accession.is_empty(),
        );
        assert!(result.is_err(), "invalid JSON should return Err");
    }

    #[test]
    fn try_load_empty_array_returns_error() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("vibrio_assemblies.json");
        std::fs::write(&path, r#"{"assemblies": []}"#).unwrap();
        let result = super::super::try_load_json_array::<VibrioAssembly>(
            &path,
            "assemblies",
            VibrioAssembly::from_json_obj,
            |a| !a.accession.is_empty(),
        );
        assert!(result.is_err(), "empty array should return Err");
    }

    #[test]
    fn try_load_valid_json_returns_ok() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("vibrio_assemblies.json");
        let json = r#"{"assemblies": [{"accession": "GCF_TEST", "organism": "Vibrio", "genome_size_bp": 4000000, "gene_count": 3800, "scaffold_count": 2, "isolation_source": "marine"}]}"#;
        std::fs::write(&path, json).unwrap();
        let result = super::super::try_load_json_array(
            &path,
            "assemblies",
            VibrioAssembly::from_json_obj,
            |a| !a.accession.is_empty(),
        );
        let assemblies = result.unwrap();
        assert_eq!(assemblies.len(), 1);
        assert_eq!(assemblies[0].accession, "GCF_TEST");
    }

    #[test]
    fn synthetic_fallback_is_deterministic() {
        let (a1, _) = load_vibrio_assemblies();
        let (a2, _) = load_vibrio_assemblies();
        assert_eq!(a1.len(), a2.len());
        for (x, y) in a1.iter().zip(a2.iter()) {
            assert_eq!(x.accession, y.accession);
            assert_eq!(x.genome_size_bp, y.genome_size_bp);
        }
    }

    #[test]
    fn synthetic_fallback_via_public_api() {
        let (assemblies, is_real) = load_vibrio_assemblies();
        assert!(!assemblies.is_empty());
        if !is_real {
            assert_eq!(assemblies.len(), 150);
            assert!(assemblies[0].accession.starts_with("GCF_SYN_"));
        }
    }
}
