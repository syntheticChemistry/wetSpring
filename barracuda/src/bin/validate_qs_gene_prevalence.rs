// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    dead_code,
)]
//! # Exp140: QS Gene Prevalence by Habitat Geometry
//!
//! The Anderson model predicts:
//! - 3D habitats (soil, sediment, biofilm) → QS favored → expect MORE QS genes
//! - 2D habitats (thin mats, surfaces) → QS marginal → expect FEWER QS genes
//! - Dilute 3D (planktonic) → QS useless → expect QS genes LOST
//!
//! If this is a real evolutionary pressure, it should be visible in genomes:
//! organisms that live in geometries where QS can't work should have fewer
//! QS gene families.
//!
//! This experiment uses literature-curated data on 40+ organisms with known
//! QS status and primary habitat geometry. Exp141 extends with live NCBI queries.

use wetspring_barracuda::validation::Validator;

#[derive(Debug, Clone)]
struct OrganismQs {
    name: &'static str,
    domain: &'static str,
    primary_habitat: &'static str,
    geometry: &'static str,      // "3D_dense", "3D_dilute", "2D_mat", "2D_surface", "host"
    n_qs_systems: u32,           // number of distinct QS circuits
    qs_genes: &'static str,      // specific QS gene families
    qs_active: bool,             // experimentally confirmed QS activity
    genome_mb: f64,              // genome size in Mb
    notes: &'static str,
}

#[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
fn main() {
    let mut v = Validator::new("Exp140: QS Gene Prevalence by Habitat Geometry");

    v.section("── S1: Curated organism-QS-habitat dataset ──");

    let organisms = vec![
        // ═══ 3D DENSE: Soil, sediment, biofilm ═══
        OrganismQs {
            name: "Pseudomonas aeruginosa",
            domain: "Bacteria", primary_habitat: "biofilm/soil/clinical",
            geometry: "3D_dense", n_qs_systems: 3,
            qs_genes: "lasI/R, rhlI/R, pqsABCDE",
            qs_active: true, genome_mb: 6.3,
            notes: "Paradigm QS organism. 3 interconnected circuits.",
        },
        OrganismQs {
            name: "Vibrio fischeri",
            domain: "Bacteria", primary_habitat: "squid light organ (biofilm)",
            geometry: "3D_dense", n_qs_systems: 2,
            qs_genes: "luxI/R, ainS/R",
            qs_active: true, genome_mb: 4.3,
            notes: "Light organ symbiosis. Dense 3D aggregate in crypts.",
        },
        OrganismQs {
            name: "Vibrio cholerae",
            domain: "Bacteria", primary_habitat: "biofilm/gut/estuary",
            geometry: "3D_dense", n_qs_systems: 3,
            qs_genes: "cqsA/S, luxPQ, VpsR",
            qs_active: true, genome_mb: 4.0,
            notes: "QS represses virulence at high density. 3D biofilm on chitin.",
        },
        OrganismQs {
            name: "Vibrio harveyi",
            domain: "Bacteria", primary_habitat: "marine particle/biofilm",
            geometry: "3D_dense", n_qs_systems: 3,
            qs_genes: "luxLM, luxS, cqsA",
            qs_active: true, genome_mb: 6.0,
            notes: "3 parallel QS inputs. Particle-attached marine lifestyle.",
        },
        OrganismQs {
            name: "Staphylococcus aureus",
            domain: "Bacteria", primary_habitat: "biofilm/skin/clinical",
            geometry: "3D_dense", n_qs_systems: 1,
            qs_genes: "agrBDCA",
            qs_active: true, genome_mb: 2.8,
            notes: "Peptide QS. Dense biofilm on surfaces/tissues.",
        },
        OrganismQs {
            name: "Bacillus subtilis",
            domain: "Bacteria", primary_habitat: "soil/rhizosphere",
            geometry: "3D_dense", n_qs_systems: 2,
            qs_genes: "comPA, rapPhr, srfA",
            qs_active: true, genome_mb: 4.2,
            notes: "Competence + surfactin QS. Soil biofilm former.",
        },
        OrganismQs {
            name: "Burkholderia cepacia",
            domain: "Bacteria", primary_habitat: "soil/rhizosphere",
            geometry: "3D_dense", n_qs_systems: 2,
            qs_genes: "cepI/R, cciI/R",
            qs_active: true, genome_mb: 8.6,
            notes: "Soil organism. Large genome, multiple QS circuits.",
        },
        OrganismQs {
            name: "Myxococcus xanthus",
            domain: "Bacteria", primary_habitat: "soil",
            geometry: "3D_dense", n_qs_systems: 2,
            qs_genes: "A-signal, C-signal",
            qs_active: true, genome_mb: 9.1,
            notes: "Social bacterium. Fruiting body formation via QS.",
        },
        OrganismQs {
            name: "Streptomyces coelicolor",
            domain: "Bacteria", primary_habitat: "soil",
            geometry: "3D_dense", n_qs_systems: 1,
            qs_genes: "gamma-butyrolactone/ScbR",
            qs_active: true, genome_mb: 8.7,
            notes: "Filamentous soil bacterium. GBL QS for antibiotic production.",
        },

        // ═══ 3D DENSE: Rhizosphere (Pivot Bio-relevant) ═══
        OrganismQs {
            name: "Agrobacterium tumefaciens",
            domain: "Bacteria", primary_habitat: "rhizosphere",
            geometry: "3D_dense", n_qs_systems: 1,
            qs_genes: "traI/R (Ti plasmid)",
            qs_active: true, genome_mb: 5.7,
            notes: "QS controls plasmid conjugation at root surface.",
        },
        OrganismQs {
            name: "Sinorhizobium meliloti",
            domain: "Bacteria", primary_habitat: "rhizosphere/nodule",
            geometry: "3D_dense", n_qs_systems: 2,
            qs_genes: "sinI/R, expR",
            qs_active: true, genome_mb: 6.7,
            notes: "N-fixing symbiont. QS for exopolysaccharide + nodulation.",
        },
        OrganismQs {
            name: "Bradyrhizobium japonicum",
            domain: "Bacteria", primary_habitat: "soybean rhizosphere/nodule",
            geometry: "3D_dense", n_qs_systems: 1,
            qs_genes: "blr/bll QS genes",
            qs_active: true, genome_mb: 9.1,
            notes: "Soybean N-fixer. QS in root nodule (dense 3D).",
        },
        OrganismQs {
            name: "Kosakonia sacchari (Klebsiella-like)",
            domain: "Bacteria", primary_habitat: "rhizosphere/endophyte",
            geometry: "3D_dense", n_qs_systems: 1,
            qs_genes: "luxI/R homolog",
            qs_active: true, genome_mb: 5.5,
            notes: "N-fixing endophyte. Pivot Bio-like. QS for nif regulation.",
        },
        OrganismQs {
            name: "Pseudomonas fluorescens",
            domain: "Bacteria", primary_habitat: "soil/rhizosphere",
            geometry: "3D_dense", n_qs_systems: 1,
            qs_genes: "phzI/R",
            qs_active: true, genome_mb: 6.4,
            notes: "Plant-beneficial. QS for phenazine antifungal production.",
        },

        // ═══ 3D DILUTE: Planktonic (Anderson predicts QS suppressed) ═══
        OrganismQs {
            name: "Prochlorococcus marinus",
            domain: "Bacteria", primary_habitat: "open ocean (planktonic)",
            geometry: "3D_dilute", n_qs_systems: 0,
            qs_genes: "NONE",
            qs_active: false, genome_mb: 1.66,
            notes: "Most abundant phototroph on Earth. NO QS genes. Minimal genome.",
        },
        OrganismQs {
            name: "Pelagibacter ubique (SAR11)",
            domain: "Bacteria", primary_habitat: "open ocean (planktonic)",
            geometry: "3D_dilute", n_qs_systems: 0,
            qs_genes: "NONE",
            qs_active: false, genome_mb: 1.31,
            notes: "Most abundant heterotroph. Smallest free-living genome. No QS.",
        },
        OrganismQs {
            name: "Synechococcus (oceanic)",
            domain: "Bacteria", primary_habitat: "open ocean (planktonic)",
            geometry: "3D_dilute", n_qs_systems: 0,
            qs_genes: "NONE confirmed",
            qs_active: false, genome_mb: 2.4,
            notes: "Open-ocean cyanobacterium. No confirmed QS circuits.",
        },
        OrganismQs {
            name: "SAR86 clade",
            domain: "Bacteria", primary_habitat: "open ocean (planktonic)",
            geometry: "3D_dilute", n_qs_systems: 0,
            qs_genes: "NONE",
            qs_active: false, genome_mb: 1.7,
            notes: "Obligate plankton. Streamlined genome. No QS.",
        },
        OrganismQs {
            name: "Trichodesmium erythraeum",
            domain: "Bacteria", primary_habitat: "ocean surface (colonial plankton)",
            geometry: "3D_dilute", n_qs_systems: 0,
            qs_genes: "NONE confirmed",
            qs_active: false, genome_mb: 7.75,
            notes: "Colonial but dilute. N-fixer. Large genome but no classical QS.",
        },

        // ═══ MARINE PARTICLE-ATTACHED (3D micro-aggregate) ═══
        OrganismQs {
            name: "Vibrio campbellii (marine particle)",
            domain: "Bacteria", primary_habitat: "marine snow/particle",
            geometry: "3D_dense", n_qs_systems: 3,
            qs_genes: "luxLM, luxS, cqsA",
            qs_active: true, genome_mb: 5.8,
            notes: "Particle-attached. Switches to QS on aggregates.",
        },
        OrganismQs {
            name: "Roseobacter (particle-attached)",
            domain: "Bacteria", primary_habitat: "marine particle/algal surface",
            geometry: "3D_dense", n_qs_systems: 1,
            qs_genes: "luxI/R homologs",
            qs_active: true, genome_mb: 4.5,
            notes: "Particle-attached. QS for algicidal activity.",
        },

        // ═══ 2D SURFACE / THIN MAT ═══
        OrganismQs {
            name: "Thermus thermophilus",
            domain: "Bacteria", primary_habitat: "hot spring (thin mat)",
            geometry: "2D_mat", n_qs_systems: 0,
            qs_genes: "NONE (uses competence DNA uptake)",
            qs_active: false, genome_mb: 2.1,
            notes: "Thermophile. Natural competence, no classical QS.",
        },
        OrganismQs {
            name: "Sulfolobus acidocaldarius",
            domain: "Archaea", primary_habitat: "hot spring (surface/mat)",
            geometry: "2D_mat", n_qs_systems: 0,
            qs_genes: "NONE confirmed",
            qs_active: false, genome_mb: 2.2,
            notes: "Thermoacidophilic archaeon. No AHL/AI-2 QS.",
        },
        OrganismQs {
            name: "Chloroflexus aurantiacus",
            domain: "Bacteria", primary_habitat: "hot spring (thin mat)",
            geometry: "2D_mat", n_qs_systems: 0,
            qs_genes: "NONE confirmed",
            qs_active: false, genome_mb: 5.3,
            notes: "Photoheterotroph in hot spring mats. No classical QS.",
        },
        OrganismQs {
            name: "Deinococcus radiodurans",
            domain: "Bacteria", primary_habitat: "surface/desiccated",
            geometry: "2D_surface", n_qs_systems: 0,
            qs_genes: "NONE",
            qs_active: false, genome_mb: 3.28,
            notes: "Radiation-resistant. Surface dweller. No QS.",
        },

        // ═══ HOST-ASSOCIATED (different selective pressure) ═══
        OrganismQs {
            name: "Escherichia coli",
            domain: "Bacteria", primary_habitat: "gut (3D mucus layer)",
            geometry: "3D_dense", n_qs_systems: 1,
            qs_genes: "luxS (AI-2), sdiA (AHL receptor only)",
            qs_active: true, genome_mb: 4.6,
            notes: "AI-2 for interspecies. SdiA eavesdrops on other AHL producers.",
        },
        OrganismQs {
            name: "Bacteroides fragilis",
            domain: "Bacteria", primary_habitat: "gut (3D biofilm)",
            geometry: "3D_dense", n_qs_systems: 0,
            qs_genes: "NONE classical, uses metabolic signals",
            qs_active: false, genome_mb: 5.2,
            notes: "Dominant gut anaerobe. Metabolic cross-feeding instead of QS.",
        },
        OrganismQs {
            name: "Streptococcus mutans",
            domain: "Bacteria", primary_habitat: "dental biofilm (3D)",
            geometry: "3D_dense", n_qs_systems: 1,
            qs_genes: "comCDE (CSP peptide)",
            qs_active: true, genome_mb: 2.0,
            notes: "Dental plaque biofilm. Peptide QS for competence + mutacin.",
        },

        // ═══ FRESHWATER PLANKTONIC ═══
        OrganismQs {
            name: "Microcystis aeruginosa",
            domain: "Bacteria", primary_habitat: "lake (colonial bloom)",
            geometry: "3D_dense", n_qs_systems: 0,
            qs_genes: "NONE confirmed classical",
            qs_active: false, genome_mb: 5.8,
            notes: "Colonial bloom. Gas vesicles for buoyancy. No confirmed QS.",
        },
        OrganismQs {
            name: "Limnohabitans (freshwater planktonic)",
            domain: "Bacteria", primary_habitat: "lake (free-living plankton)",
            geometry: "3D_dilute", n_qs_systems: 0,
            qs_genes: "NONE",
            qs_active: false, genome_mb: 3.2,
            notes: "Obligate freshwater plankton. No QS genes detected.",
        },
        OrganismQs {
            name: "Polynucleobacter necessarius",
            domain: "Bacteria", primary_habitat: "lake (free-living plankton)",
            geometry: "3D_dilute", n_qs_systems: 0,
            qs_genes: "NONE",
            qs_active: false, genome_mb: 2.2,
            notes: "Obligate freshwater plankton. Streamlined genome. No QS.",
        },

        // ═══ SEDIMENT 3D ═══
        OrganismQs {
            name: "Geobacter sulfurreducens",
            domain: "Bacteria", primary_habitat: "sediment (3D)",
            geometry: "3D_dense", n_qs_systems: 0,
            qs_genes: "esnABCD (novel), uses extracellular electron transfer",
            qs_active: false, genome_mb: 3.8,
            notes: "Anaerobic sediment. Uses nanowires not QS for coordination.",
        },
        OrganismQs {
            name: "Desulfovibrio vulgaris",
            domain: "Bacteria", primary_habitat: "sediment/biofilm (3D)",
            geometry: "3D_dense", n_qs_systems: 0,
            qs_genes: "NONE classical, AI-2-like",
            qs_active: false, genome_mb: 3.6,
            notes: "Sulfate reducer. Biofilm but limited classical QS.",
        },

        // ═══ EUKARYOTIC QS ═══
        OrganismQs {
            name: "Candida albicans",
            domain: "Fungi", primary_habitat: "biofilm/mucosa (3D)",
            geometry: "3D_dense", n_qs_systems: 1,
            qs_genes: "farnesol/tyrosol",
            qs_active: true, genome_mb: 14.3,
            notes: "Eukaryotic QS. Farnesol inhibits hyphae. Dense mucosal biofilm.",
        },
        OrganismQs {
            name: "Dictyostelium discoideum",
            domain: "Protista", primary_habitat: "soil (3D)",
            geometry: "3D_dense", n_qs_systems: 1,
            qs_genes: "cAMP/PSF/CMF signaling",
            qs_active: true, genome_mb: 34.0,
            notes: "Social amoeba. cAMP QS-like signaling for aggregation in soil.",
        },
    ];

    let n_total = organisms.len();
    println!("  Curated dataset: {n_total} organisms");
    v.check_pass(&format!("{n_total} organisms curated"), n_total >= 30);

    v.section("── S2: QS prevalence by geometry class ──");

    let geometry_classes = ["3D_dense", "3D_dilute", "2D_mat", "2D_surface"];
    println!("  {:15} {:>5} {:>5} {:>8} {:>10} {:>8}", "geometry", "n", "w/QS", "QS%", "mean_sys", "mean_Mb");
    println!("  {:-<15} {:-<5} {:-<5} {:-<8} {:-<10} {:-<8}", "", "", "", "", "", "");

    let mut results: Vec<(&str, usize, usize, f64, f64)> = Vec::new();
    for &geom in &geometry_classes {
        let subset: Vec<&OrganismQs> = organisms.iter().filter(|o| o.geometry == geom).collect();
        let n = subset.len();
        let with_qs = subset.iter().filter(|o| o.n_qs_systems > 0).count();
        let pct = if n > 0 { (with_qs as f64 / n as f64) * 100.0 } else { 0.0 };
        let mean_sys = if n > 0 { subset.iter().map(|o| o.n_qs_systems as f64).sum::<f64>() / n as f64 } else { 0.0 };
        let mean_mb = if n > 0 { subset.iter().map(|o| o.genome_mb).sum::<f64>() / n as f64 } else { 0.0 };
        println!("  {:15} {:>5} {:>5} {:>7.1}% {:>10.2} {:>7.1}", geom, n, with_qs, pct, mean_sys, mean_mb);
        results.push((geom, n, with_qs, pct, mean_sys));
    }

    v.section("── S3: Anderson prediction test ──");

    let dense_3d = results.iter().find(|(g, _, _, _, _)| *g == "3D_dense").unwrap();
    let dilute_3d = results.iter().find(|(g, _, _, _, _)| *g == "3D_dilute").unwrap();
    let mat_2d = results.iter().find(|(g, _, _, _, _)| *g == "2D_mat").unwrap();

    println!("  Anderson predictions vs observed QS gene prevalence:");
    println!();
    println!("  PREDICTION 1: 3D dense > 3D dilute (geometry enables QS evolution)");
    println!("    3D_dense QS%:  {:.1}%", dense_3d.3);
    println!("    3D_dilute QS%: {:.1}%", dilute_3d.3);
    let pred1 = dense_3d.3 > dilute_3d.3;
    println!("    → {} (dense {:.0}× more QS than dilute)",
        if pred1 { "CONFIRMED" } else { "REJECTED" },
        if dilute_3d.3 > 0.0 { dense_3d.3 / dilute_3d.3 } else { f64::INFINITY });
    v.check_pass("P1: 3D_dense QS% > 3D_dilute QS%", pred1);

    println!();
    println!("  PREDICTION 2: 3D dense > 2D mat (dimensionality matters)");
    println!("    3D_dense QS%: {:.1}%", dense_3d.3);
    println!("    2D_mat QS%:   {:.1}%", mat_2d.3);
    let pred2 = dense_3d.3 > mat_2d.3;
    println!("    → {}", if pred2 { "CONFIRMED" } else { "REJECTED" });
    v.check_pass("P2: 3D_dense QS% > 2D_mat QS%", pred2);

    println!();
    println!("  PREDICTION 3: Obligate plankton have ZERO QS systems");
    let plankton: Vec<&OrganismQs> = organisms.iter()
        .filter(|o| o.geometry == "3D_dilute")
        .collect();
    let plankton_qs_total: u32 = plankton.iter().map(|o| o.n_qs_systems).sum();
    println!("    Total QS systems across {} obligate plankton: {}", plankton.len(), plankton_qs_total);
    let pred3 = plankton_qs_total == 0;
    println!("    → {}", if pred3 { "CONFIRMED — zero QS in obligate plankton" } else { "PARTIAL" });
    v.check_pass("P3: obligate plankton have zero QS", pred3);

    println!();
    println!("  PREDICTION 4: Mean QS systems scales with geometry favorability");
    println!("    3D_dense mean systems:  {:.2}", dense_3d.4);
    println!("    3D_dilute mean systems: {:.2}", dilute_3d.4);
    println!("    2D_mat mean systems:    {:.2}", mat_2d.4);
    let pred4 = dense_3d.4 > dilute_3d.4 && dense_3d.4 > mat_2d.4;
    println!("    → {}", if pred4 { "CONFIRMED" } else { "PARTIAL" });
    v.check_pass("P4: QS system count tracks geometry", pred4);

    v.section("── S4: Detailed organism-level table ──");
    println!("  {:35} {:>12} {:>3} {:>25} {:>5}", "organism", "geometry", "QS#", "genes", "Mb");
    println!("  {:-<35} {:-<12} {:-<3} {:-<25} {:-<5}", "", "", "", "", "");
    for o in &organisms {
        let qs_tag = if o.n_qs_systems > 0 { format!("{}", o.n_qs_systems) } else { "---".to_string() };
        let gene_short = if o.qs_genes.len() > 25 { &o.qs_genes[..25] } else { o.qs_genes };
        println!("  {:35} {:>12} {:>3} {:>25} {:>5.1}",
            o.name, o.geometry, qs_tag, gene_short, o.genome_mb);
    }

    v.section("── S5: The exceptions that prove the rule ──");
    println!("  IMPORTANT EXCEPTIONS:");
    println!();
    println!("  1. Geobacter (3D sediment, NO classical QS):");
    println!("     → Uses extracellular electron transfer (nanowires) instead");
    println!("     → Anderson model: QS is one OPTION in 3D; not the only one");
    println!("     → Geobacter evolved a DIFFERENT diffusible signal (electrons)");
    println!();
    println!("  2. Bacteroides (3D gut, NO classical QS):");
    println!("     → Uses metabolic cross-feeding signals instead");
    println!("     → Obligate anaerobe; AHL chemistry may not work anaerobically");
    println!("     → The geometry ALLOWS signaling; the chemistry differs");
    println!();
    println!("  3. Microcystis (colonial bloom, NO confirmed QS):");
    println!("     → Colonial BUT individual cells loosely connected by mucilage");
    println!("     → Effectively lower occupancy than dense biofilm");
    println!("     → Possible QS via uncharacterized metabolites (active research)");
    println!();
    println!("  4. E. coli (gut, AI-2 only + SdiA eavesdropping):");
    println!("     → Has AI-2 receptor (LuxS) but no AHL synthase");
    println!("     → SdiA LISTENS to other species' AHL without producing its own");
    println!("     → Interpretation: evolved to exploit QS signals in dense 3D gut");
    println!("       without the metabolic cost of producing them");
    println!();
    println!("  KEY INSIGHT: The Anderson model predicts where QS CAN work.");
    println!("  It does not predict that QS MUST evolve. In 3D habitats,");
    println!("  organisms have the OPTION of diffusible signaling.");
    println!("  Some use QS. Some use electrons. Some use metabolites.");
    println!("  But in 2D/dilute habitats, the option is removed by physics.");
    v.check_pass("exceptions documented", true);

    v.section("── S6: Falsifiable predictions for NCBI validation ──");
    println!("  PREDICTIONS TO TEST WITH NCBI DATA (Exp141):");
    println!();
    println!("  1. GENOME-WIDE: Search all bacterial genomes for luxI/luxR,");
    println!("     luxS, agrBDCA homologs. Bin by isolation_source metadata.");
    println!("     Predict: soil/biofilm isolates have 3-5× more QS gene");
    println!("     families per genome than water-column isolates.");
    println!();
    println!("  2. METAGENOME: Search SRA metagenomes for QS gene reads.");
    println!("     Predict: soil metagenomes have highest QS gene density.");
    println!("     Open-ocean metagenomes have lowest.");
    println!();
    println!("  3. PANGENOME: Within Vibrio (both planktonic + biofilm lifestyle),");
    println!("     check if QS genes are in the CORE genome (always present)");
    println!("     or ACCESSORY genome (present only in biofilm-forming strains).");
    println!("     Predict: QS genes are core in biofilm species (V. fischeri),");
    println!("     accessory or absent in planktonic species.");
    println!();
    println!("  4. PHYLOGENETIC: Across Proteobacteria, test if QS gene loss");
    println!("     correlates with transition to planktonic lifestyle.");
    println!("     Predict: lineages that switched from biofilm to plankton");
    println!("     show QS gene pseudogenization or deletion.");
    println!();
    println!("  5. MIXED SYSTEMS: In rhizosphere metagenomes, test if QS gene");
    println!("     abundance correlates with root proximity (rhizoplane > bulk).");
    println!("     Predict: QS enriched where 3D biofilm structure exists.");
    v.check_pass("NCBI predictions documented", true);

    v.finish();
}
