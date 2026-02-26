// SPDX-License-Identifier: AGPL-3.0-or-later
//! 16S `BioProject` records with biome classification (Exp126).

/// 16S `BioProject` record with biome classification (Exp126).
#[derive(Debug, Clone)]
pub struct BiomeProject {
    /// `BioProject` accession (e.g., "PRJNA000000").
    pub accession: String,
    /// Project title.
    pub title: String,
    /// Biome classification (gut, soil, marine, vent, etc.).
    pub biome: String,
    /// Primary organism or target.
    pub organism: String,
}

impl BiomeProject {
    /// Parse a `BiomeProject` from a JSON object string.
    #[must_use]
    pub fn from_json_obj(obj: &str) -> Self {
        Self {
            accession: super::json_str_value(obj, "accession"),
            title: super::json_str_value(obj, "title"),
            biome: super::json_str_value(obj, "biome"),
            organism: super::json_str_value(obj, "organism"),
        }
    }
}

/// Load 16S `BioProject` records or generate synthetic fallback.
#[must_use]
pub fn load_biome_projects() -> (Vec<BiomeProject>, bool) {
    let path = super::data_dir().join("biome_16s_projects.json");
    super::load_json_array_or_fallback(
        &path,
        "projects",
        BiomeProject::from_json_obj,
        |p| !p.accession.is_empty(),
        gen_synthetic_biome_projects,
    )
}

/// Synthetic fallback: 28 `BioProject` records across 14 biomes.
///
/// Used when `biome_16s_projects.json` is absent (offline / CI).
fn gen_synthetic_biome_projects() -> Vec<BiomeProject> {
    let biomes = [
        ("gut", "Human gut microbiome 16S survey"),
        ("gut", "Infant gut longitudinal 16S"),
        ("oral", "Human oral microbiome 16S"),
        ("oral", "Periodontal disease oral 16S"),
        ("soil", "Temperate forest soil 16S survey"),
        ("soil", "Agricultural soil microbiome"),
        ("marine_sediment", "Continental shelf sediment 16S"),
        ("marine_sediment", "Abyssal plain sediment 16S"),
        ("vent", "Mid-Atlantic Ridge vent 16S"),
        ("vent", "East Pacific Rise vent chimney 16S"),
        ("rhizosphere", "Rice paddy rhizosphere 16S"),
        ("rhizosphere", "Arabidopsis root microbiome 16S"),
        ("freshwater", "Lake Erie freshwater 16S"),
        ("freshwater", "Amazon river microbiome 16S"),
        ("wastewater", "Activated sludge 16S survey"),
        ("wastewater", "Anaerobic digester 16S"),
        ("coral", "Great Barrier Reef coral 16S"),
        ("coral", "Caribbean coral bleaching 16S"),
        ("deep_sea", "Mariana Trench sediment 16S"),
        ("deep_sea", "Hadal zone water column 16S"),
        ("permafrost", "Siberian permafrost thaw 16S"),
        ("permafrost", "Arctic permafrost core 16S"),
        ("biofilm", "Hospital drain biofilm 16S"),
        ("biofilm", "Ship hull biofilm 16S"),
        ("hot_spring", "Yellowstone hot spring 16S"),
        ("hot_spring", "Iceland geothermal 16S"),
        ("algal_bloom", "Lake Taihu cyanobacterial bloom"),
        ("algal_bloom", "Baltic Sea HAB 16S"),
    ];

    biomes
        .iter()
        .enumerate()
        .map(|(i, (biome, title))| BiomeProject {
            accession: format!("PRJNA_SYN_{i:04}"),
            title: (*title).to_string(),
            biome: (*biome).to_string(),
            organism: "metagenome".to_string(),
        })
        .collect()
}

/// Biome diversity parameters for synthetic community generation (Exp126).
///
/// Returns (`biome_name`, `estimated_n_species`, `estimated_pielou_j`).
#[must_use]
pub fn biome_diversity_params() -> Vec<(&'static str, usize, f64)> {
    vec![
        ("gut", 300, 0.55),
        ("gut_infant", 200, 0.40),
        ("oral", 500, 0.70),
        ("oral_disease", 350, 0.60),
        ("soil_forest", 1000, 0.85),
        ("soil_agriculture", 800, 0.80),
        ("marine_sediment_shelf", 600, 0.75),
        ("marine_sediment_abyssal", 400, 0.65),
        ("vent_mar", 150, 0.30),
        ("vent_epr", 100, 0.25),
        ("rhizosphere_rice", 700, 0.78),
        ("rhizosphere_arabidopsis", 500, 0.72),
        ("freshwater_lake", 450, 0.68),
        ("freshwater_river", 600, 0.75),
        ("wastewater_sludge", 300, 0.50),
        ("wastewater_digester", 200, 0.45),
        ("coral_reef", 400, 0.65),
        ("coral_bleaching", 300, 0.55),
        ("deep_sea_trench", 80, 0.20),
        ("deep_sea_hadal", 60, 0.15),
        ("permafrost_thaw", 250, 0.50),
        ("permafrost_arctic", 200, 0.45),
        ("biofilm_hospital", 50, 0.15),
        ("biofilm_marine", 80, 0.20),
        ("hot_spring_yellowstone", 120, 0.35),
        ("hot_spring_iceland", 150, 0.40),
        ("algal_bloom_taihu", 60, 0.12),
        ("algal_bloom_baltic", 80, 0.18),
    ]
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn load_from_dir(dir: &std::path::Path) -> (Vec<BiomeProject>, bool) {
        super::super::load_json_array_or_fallback(
            &dir.join("biome_16s_projects.json"),
            "projects",
            BiomeProject::from_json_obj,
            |p| !p.accession.is_empty(),
            gen_synthetic_biome_projects,
        )
    }

    #[test]
    fn from_json_obj_parses_fields() {
        let obj = r#"{"accession": "PRJNA123456", "title": "Gut microbiome 16S", "biome": "gut", "organism": "metagenome"}"#;
        let p = BiomeProject::from_json_obj(obj);
        assert_eq!(p.accession, "PRJNA123456");
        assert_eq!(p.biome, "gut");
    }

    #[test]
    fn forced_synthetic() {
        let temp = TempDir::new().unwrap();
        let (projects, is_real) = load_from_dir(temp.path());
        assert!(!is_real);
        assert_eq!(projects.len(), 28);
        assert!(projects[0].accession.starts_with("PRJNA_SYN_"));
    }

    #[test]
    fn biome_diversity_across_projects() {
        let (projects, _) = load_biome_projects();
        let biomes: std::collections::HashSet<&str> =
            projects.iter().map(|p| p.biome.as_str()).collect();
        assert!(biomes.len() >= 10);
    }

    #[test]
    fn synthetic_fallback_via_public_api() {
        let (projects, is_real) = load_biome_projects();
        assert!(!projects.is_empty());
        if !is_real {
            assert_eq!(projects.len(), 28);
            assert!(!projects[0].biome.is_empty());
        }
    }

    #[test]
    fn diversity_params_count() {
        assert_eq!(biome_diversity_params().len(), 28);
    }

    #[test]
    fn diversity_params_ranges() {
        for (name, n_species, j) in biome_diversity_params() {
            assert!(!name.is_empty());
            assert!(n_species > 0, "{name}: n_species should be > 0");
            assert!(
                (0.0..=1.0).contains(&j),
                "{name}: Pielou J should be in [0,1], got {j}"
            );
        }
    }
}
