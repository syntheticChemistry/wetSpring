// SPDX-License-Identifier: AGPL-3.0-or-later
//! NCBI data loading for Phase 35 experiments.
//!
//! Reads JSON data produced by `scripts/fetch_ncbi_phase35.py`. Falls back to
//! synthetic data generation when NCBI data files are absent (offline / CI mode).
//!
//! Data lives in `data/ncbi_phase35/` relative to the crate manifest directory.

use std::path::PathBuf;

/// Vibrio genome assembly record (Exp121).
#[derive(Debug, Clone)]
pub struct VibrioAssembly {
    /// NCBI accession (e.g., "GCF_000006745.1").
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

/// 16S BioProject record with biome classification (Exp126).
#[derive(Debug, Clone)]
pub struct BiomeProject {
    /// BioProject accession (e.g., "PRJNA000000").
    pub accession: String,
    /// Project title.
    pub title: String,
    /// Biome classification (gut, soil, marine, vent, etc.).
    pub biome: String,
    /// Primary organism or target.
    pub organism: String,
}

fn data_dir() -> PathBuf {
    if let Ok(root) = std::env::var("WETSPRING_DATA_ROOT") {
        return PathBuf::from(root).join("ncbi_phase35");
    }
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        return PathBuf::from(manifest).join("../data/ncbi_phase35");
    }
    PathBuf::from("data/ncbi_phase35")
}

/// Extract a JSON string value from a line like `"key": "value"`.
fn json_str_value(json: &str, key: &str) -> String {
    let needle = format!("\"{key}\":");
    if let Some(pos) = json.find(&needle) {
        let rest = &json[pos + needle.len()..];
        let rest = rest.trim_start();
        if rest.starts_with('"') {
            let inner = &rest[1..];
            if let Some(end) = inner.find('"') {
                return inner[..end].to_string();
            }
        }
    }
    String::new()
}

/// Extract a JSON integer value from a line like `"key": 12345`.
fn json_int_value(json: &str, key: &str) -> u64 {
    let needle = format!("\"{key}\":");
    if let Some(pos) = json.find(&needle) {
        let rest = &json[pos + needle.len()..];
        let rest = rest.trim_start();
        let num_str: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        return num_str.parse().unwrap_or(0);
    }
    0
}

/// Split JSON array into individual object strings (minimal parser).
fn split_json_objects(array_content: &str) -> Vec<String> {
    let mut objects = Vec::new();
    let mut depth = 0;
    let mut start = None;

    for (i, ch) in array_content.char_indices() {
        match ch {
            '{' => {
                if depth == 0 {
                    start = Some(i);
                }
                depth += 1;
            }
            '}' => {
                depth -= 1;
                if depth == 0 {
                    if let Some(s) = start {
                        objects.push(array_content[s..=i].to_string());
                    }
                    start = None;
                }
            }
            _ => {}
        }
    }
    objects
}

/// Load Vibrio assemblies from NCBI data or generate synthetic fallback.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
pub fn load_vibrio_assemblies() -> (Vec<VibrioAssembly>, bool) {
    let path = data_dir().join("vibrio_assemblies.json");
    if let Ok(content) = std::fs::read_to_string(&path) {
        if let Some(arr_start) = content.find("\"assemblies\"") {
            let rest = &content[arr_start..];
            if let Some(bracket) = rest.find('[') {
                let arr = &rest[bracket..];
                let objects = split_json_objects(arr);
                let assemblies: Vec<VibrioAssembly> = objects.iter().map(|obj| {
                    VibrioAssembly {
                        accession: json_str_value(obj, "accession"),
                        organism: json_str_value(obj, "organism"),
                        genome_size_bp: json_int_value(obj, "genome_size_bp"),
                        gene_count: json_int_value(obj, "gene_count") as u32,
                        scaffold_count: json_int_value(obj, "scaffold_count") as u32,
                        isolation_source: json_str_value(obj, "isolation_source"),
                    }
                }).filter(|a| !a.accession.is_empty())
                .collect();

                if !assemblies.is_empty() {
                    return (assemblies, true);
                }
            }
        }
    }

    // Synthetic fallback: 150 assemblies mirroring real Vibrio diversity
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
    let sources = ["clinical", "environmental", "aquaculture", "marine", "estuarine"];

    let mut assemblies = Vec::with_capacity(150);
    for i in 0..150_u32 {
        rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let sp = &species[(i as usize) % species.len()];
        let size_var = (rng >> 33) as f64 / (u32::MAX as f64) * 0.2 - 0.1;
        let genome_size = ((sp.1 as f64) * (1.0 + size_var)) as u64;
        rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let gene_var = (rng >> 33) as f64 / (u32::MAX as f64) * 0.15 - 0.075;
        let gene_count = ((sp.2 as f64) * (1.0 + gene_var)) as u32;
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
    (assemblies, false)
}

/// Load Campylobacterota assemblies or generate synthetic fallback.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
pub fn load_campylobacterota() -> (Vec<CampyAssembly>, bool) {
    let path = data_dir().join("campylobacterota_assemblies.json");
    if let Ok(content) = std::fs::read_to_string(&path) {
        if let Some(arr_start) = content.find("\"assemblies\"") {
            let rest = &content[arr_start..];
            if let Some(bracket) = rest.find('[') {
                let arr = &rest[bracket..];
                let objects = split_json_objects(arr);
                let assemblies: Vec<CampyAssembly> = objects.iter().map(|obj| {
                    CampyAssembly {
                        accession: json_str_value(obj, "accession"),
                        organism: json_str_value(obj, "organism"),
                        genus: json_str_value(obj, "genus"),
                        genome_size_bp: json_int_value(obj, "genome_size_bp"),
                        gene_count: json_int_value(obj, "gene_count") as u32,
                        isolation_source: json_str_value(obj, "isolation_source"),
                    }
                }).filter(|a| !a.accession.is_empty())
                .collect();

                if !assemblies.is_empty() {
                    return (assemblies, true);
                }
            }
        }
    }

    // Synthetic fallback: 80 assemblies across genera and ecosystems
    let genera = [
        ("Campylobacter", "Campylobacter jejuni", 1_700_000_u64, 1700_u32, "gut"),
        ("Campylobacter", "Campylobacter coli", 1_800_000, 1750, "food"),
        ("Helicobacter", "Helicobacter pylori", 1_600_000, 1550, "gut"),
        ("Helicobacter", "Helicobacter hepaticus", 1_800_000, 1700, "gut"),
        ("Sulfurimonas", "Sulfurimonas denitrificans", 2_200_000, 2100, "vent"),
        ("Sulfurimonas", "Sulfurimonas autotrophica", 2_100_000, 2000, "vent"),
        ("Sulfurospirillum", "Sulfurospirillum multivorans", 3_200_000, 3000, "sediment"),
        ("Arcobacter", "Arcobacter butzleri", 2_300_000, 2200, "water"),
        ("Nautilia", "Nautilia profundicola", 1_700_000, 1600, "vent"),
    ];

    let mut rng = 99_u64;
    let mut assemblies = Vec::with_capacity(80);
    for i in 0..80_u32 {
        let g = &genera[(i as usize) % genera.len()];
        rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let var = (rng >> 33) as f64 / (u32::MAX as f64) * 0.1 - 0.05;
        assemblies.push(CampyAssembly {
            accession: format!("GCF_CAM_{i:04}"),
            organism: format!("{} strain SYN{i:03}", g.1),
            genus: g.0.to_string(),
            genome_size_bp: ((g.2 as f64) * (1.0 + var)) as u64,
            gene_count: ((g.3 as f64) * (1.0 + var)) as u32,
            isolation_source: g.4.to_string(),
        });
    }
    (assemblies, false)
}

/// Load 16S BioProject records or generate synthetic fallback.
pub fn load_biome_projects() -> (Vec<BiomeProject>, bool) {
    let path = data_dir().join("biome_16s_projects.json");
    if let Ok(content) = std::fs::read_to_string(&path) {
        if let Some(arr_start) = content.find("\"projects\"") {
            let rest = &content[arr_start..];
            if let Some(bracket) = rest.find('[') {
                let arr = &rest[bracket..];
                let objects = split_json_objects(arr);
                let projects: Vec<BiomeProject> = objects.iter().map(|obj| {
                    BiomeProject {
                        accession: json_str_value(obj, "accession"),
                        title: json_str_value(obj, "title"),
                        biome: json_str_value(obj, "biome"),
                        organism: json_str_value(obj, "organism"),
                    }
                }).filter(|p| !p.accession.is_empty())
                .collect();

                if !projects.is_empty() {
                    return (projects, true);
                }
            }
        }
    }

    // Synthetic fallback: 28 BioProjects across 14 biomes
    let biomes = [
        ("gut", "Human gut microbiome 16S survey", 300, 0.55),
        ("gut", "Infant gut longitudinal 16S", 200, 0.40),
        ("oral", "Human oral microbiome 16S", 500, 0.70),
        ("oral", "Periodontal disease oral 16S", 350, 0.60),
        ("soil", "Temperate forest soil 16S survey", 1000, 0.85),
        ("soil", "Agricultural soil microbiome", 800, 0.80),
        ("marine_sediment", "Continental shelf sediment 16S", 600, 0.75),
        ("marine_sediment", "Abyssal plain sediment 16S", 400, 0.65),
        ("vent", "Mid-Atlantic Ridge vent 16S", 150, 0.30),
        ("vent", "East Pacific Rise vent chimney 16S", 100, 0.25),
        ("rhizosphere", "Rice paddy rhizosphere 16S", 700, 0.78),
        ("rhizosphere", "Arabidopsis root microbiome 16S", 500, 0.72),
        ("freshwater", "Lake Erie freshwater 16S", 450, 0.68),
        ("freshwater", "Amazon river microbiome 16S", 600, 0.75),
        ("wastewater", "Activated sludge 16S survey", 300, 0.50),
        ("wastewater", "Anaerobic digester 16S", 200, 0.45),
        ("coral", "Great Barrier Reef coral 16S", 400, 0.65),
        ("coral", "Caribbean coral bleaching 16S", 300, 0.55),
        ("deep_sea", "Mariana Trench sediment 16S", 80, 0.20),
        ("deep_sea", "Hadal zone water column 16S", 60, 0.15),
        ("permafrost", "Siberian permafrost thaw 16S", 250, 0.50),
        ("permafrost", "Arctic permafrost core 16S", 200, 0.45),
        ("biofilm", "Hospital drain biofilm 16S", 50, 0.15),
        ("biofilm", "Ship hull biofilm 16S", 80, 0.20),
        ("hot_spring", "Yellowstone hot spring 16S", 120, 0.35),
        ("hot_spring", "Iceland geothermal 16S", 150, 0.40),
        ("algal_bloom", "Lake Taihu cyanobacterial bloom", 60, 0.12),
        ("algal_bloom", "Baltic Sea HAB 16S", 80, 0.18),
    ];

    let projects: Vec<BiomeProject> = biomes.iter().enumerate().map(|(i, (biome, title, _n_species, _evenness))| {
        BiomeProject {
            accession: format!("PRJNA_SYN_{i:04}"),
            title: (*title).to_string(),
            biome: (*biome).to_string(),
            organism: "metagenome".to_string(),
        }
    }).collect();

    (projects, false)
}

/// Biome diversity parameters for synthetic community generation (Exp126).
/// Returns (biome_name, estimated_n_species, estimated_pielou_j).
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
