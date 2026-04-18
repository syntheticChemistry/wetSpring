// SPDX-License-Identifier: AGPL-3.0-or-later
//! SHA-256 provenance for Python baseline scripts under `scripts/`.
//!
//! This registry lists every `*.py` file in the repository root `scripts/`
//! directory. Recompute digests after editing any script:
//!
//! ```text
//! sha256sum scripts/*.py
//! ```
//!
//! Validation-binary linkage (commit, category) lives in `provenance::python_baselines` in the
//! `provenance` module when the `gpu` feature is enabled; this module is script-content integrity only.

/// One Python baseline file pinned by content hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BaselineProvenance {
    /// Path relative to the wetSpring workspace root (e.g. `scripts/validate_exp001.py`).
    pub script: &'static str,
    /// Lowercase hex SHA-256 of the script bytes at registry update time.
    pub sha256: &'static str,
    /// One-line description of what the script is for.
    pub description: &'static str,
    /// Exact command to reproduce the baseline output (if applicable).
    ///
    /// `None` for helper/download scripts that are not directly run for
    /// validation. When set, the command should be runnable from the
    /// workspace root with the documented Python environment.
    pub command: Option<&'static str>,
    /// Git short SHA of the commit when the baseline was last validated.
    ///
    /// `None` for scripts whose baselines are not tied to a specific commit
    /// (download helpers, literature reference scripts with stable output).
    pub commit: Option<&'static str>,
    /// ISO 8601 date when the baseline was last validated (e.g. `"2026-04-17"`).
    ///
    /// `None` when the script output is deterministic and commit-independent.
    pub date: Option<&'static str>,
}

/// Every Python file in `scripts/` — order is lexicographic by path for stable diffs.
pub const PROVENANCE_REGISTRY: &[BaselineProvenance] = &[
    BaselineProvenance {
        script: "scripts/alamin2024_placement.py",
        sha256: "54de38f7dad4d4d07fa95f541d4b5592ec199a9962f1aa581e81e10e0b91a704",
        description: "Phylogenetic placement baseline (Alamin et al. 2024).",
        command: None,
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/algae_timeseries_baseline.py",
        sha256: "ec53e6b2458e05ad9ca2d029c4d876970a5681fe48c862a1b643ce184fdf3e5e",
        description: "Algae growth and bloom time-series baseline.",
        command: None,
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/anderson2014_viral_metagenomics.py",
        sha256: "fb23c9652b58b2d8162b5fe15a0a6ce1a7667ccec5316008131fdedae9be8120",
        description: "Viral metagenomics baseline (Anderson et al. 2014).",
        command: None,
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/anderson2015_rare_biosphere.py",
        sha256: "15a31f41d7d625269be425b7193cf849494e3c7de6e1d7bcb56ae348d483186c",
        description: "Rare biosphere abundance baseline (Anderson et al. 2015).",
        command: None,
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/anderson2017_population_genomics.py",
        sha256: "5e4a769e3329cb72b68ca3acf4253bb93f2242efd16311b34abbfdd9e59cecf6",
        description: "Population genomics baseline (Anderson et al. 2017).",
        command: None,
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/barracuda_cpu_v4_baseline.py",
        sha256: "98a0eb028fd98338b71b5250c6c9baaf91842b6205497d222ce9ff1c86fb6746",
        description: "barraCuda CPU v4 numeric parity baseline.",
        command: None,
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/benchmark_python_baseline.py",
        sha256: "24ef52de7ac56380a55e094658eda45bf935316ecc8e84b5ab7ba0d9987ba0f2",
        description: "Python scientific stack timing benchmark harness.",
        command: Some("python3 scripts/benchmark_python_baseline.py"),
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/benchmark_rust_vs_python.py",
        sha256: "46604acdecd1458ff23dcba1c2d3549dd5f8f1076a3007e763e974efbc94d73e",
        description: "Rust versus Python cross-language benchmark harness.",
        command: Some("python3 scripts/benchmark_rust_vs_python.py"),
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/bloom_surveillance_baseline.py",
        sha256: "ccd0cbdd4f2d7f69c8808c37552cb2fdd9b9a2d7ee8d4e2d6879520b03b1262e",
        description: "Harmful algal bloom surveillance scoring baseline.",
        command: None,
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/boden2024_phosphorus_phylogenomics.py",
        sha256: "11b8c5a2a8219c30b78f8d3f64e0eb1b2a1e4fc1340ddb6f17c0b9c6693de80f",
        description: "Phosphorus-cycle phylogenomics baseline (Boden et al. 2024).",
        command: None,
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/bruger2018_cooperation.py",
        sha256: "3b32ecc4a6d297535017d7255c94291b63b39799f3a748eeefe39d6a98c2adcf",
        description: "Microbial cooperation game-theory baseline (Bruger & Basler 2018).",
        command: None,
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/download_priority1.py",
        sha256: "5d9adf95c2bf44df14882e38e78a82f3b9cfeb3159f0e5909bafaf0a6d313ae6",
        description: "NCBI dataset download helper (priority tier 1).",
        command: None,
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/download_priority2.py",
        sha256: "c143a332ea7dad9a42c90ccd636da9fdabbfbaf6abc4d975295bd488b3c417ab",
        description: "NCBI dataset download helper (priority tier 2).",
        command: None,
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/epa_pfas_ml_baseline.py",
        sha256: "b50075efd62d60c25b24ff93239693d46a6fb66abfc2155a272bd62c97927f37",
        description: "EPA-style PFAS machine-learning prediction baseline.",
        command: None,
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/exp008_pfas_ml_baseline.py",
        sha256: "c957d9ec59cf23884bdf1ffad457a6aeabdfa1b8071e212c795f8bc8a7f7b472",
        description: "Exp008 PFAS decision-tree and sklearn baseline.",
        command: None,
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/fajgenbaum_pathway_scoring.py",
        sha256: "6e906b0159e0c26a0f962c3cc5197a136ca5edf0ea7310a0bbbd2e199862f979",
        description: "Metabolic pathway scoring (Fajgenbaum-style) baseline.",
        command: None,
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/felsenstein_pruning_baseline.py",
        sha256: "cc88aee8c74e5eb12ff33b324f333f96f60f63b5eba9b3878591244781155c92",
        description: "Felsenstein pruning and likelihood on a fixed tree baseline.",
        command: None,
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/feng2024_pore_diversity.py",
        sha256: "5dd58505694e385be72d663367e39d8d6dad7b4b5dc415bc2e49a00b1eb826bc",
        description: "Soil pore habitat and diversity baseline (Feng et al. 2024).",
        command: None,
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/fernandez2020_bistable.py",
        sha256: "afaddbd41db0192f0836a76504b3cb3c0366be6ac79ef93ebc379421260ec2af",
        description: "Bistable gene regulation baseline (Fernandez et al. 2020).",
        command: None,
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/fetch_ncbi_phase35.py",
        sha256: "0c55e7fd85ede4e689ff6016fb62037f50cbd838344b4e810d48a30c316218ad",
        description: "Phase 35 NCBI sequence fetch utility.",
        command: None,
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/generate_peak_baselines.py",
        sha256: "e2f88fb9261ad247d919a0c19d58a635fe899995b85a698e3daff1521db3a3a4",
        description: "Generates chromatography peak-detection JSON baselines.",
        command: None,
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/gillespie_baseline.py",
        sha256: "8ed2243125a5f72901eac94a4b4e82edd485ca7fab2a3c57a8d79bc40de2f720",
        description: "Gillespie stochastic simulation algorithm baseline.",
        command: Some("python3 scripts/gillespie_baseline.py"),
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/hsueh2022_phage_defense.py",
        sha256: "c8591a143d84a3bbee9f3b7f241d084c225cc387e88568a046eff361868747ab",
        description: "Phage defense systems baseline (Hsueh et al. 2022).",
        command: None,
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/islam2014_brandt_farm.py",
        sha256: "72a4703cfbb702933cc4fbe2c596083b7c6f886ac4906e439c6873734b6dfa57",
        description: "Farm digester biogas baseline (Islam et al. 2014, Brandt farm).",
        command: None,
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/liang2015_longterm_tillage.py",
        sha256: "dd447d48edb8593f6017e3363196febc5c93b7ab60f6ee20f311eb72252faca4",
        description: "Long-term tillage soil microbiome baseline (Liang et al. 2015).",
        command: None,
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/liu2009_neighbor_joining.py",
        sha256: "b3012bb9563dfb72ec9b6e87966a5a5565cd3f2c6dc5866024c1193c5f9d9ac0",
        description: "Neighbor-joining phylogeny baseline (Liu et al. 2009).",
        command: None,
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/liu2014_hmm_baseline.py",
        sha256: "cd366113f13a7cd0e5aace9cc0522dc3bb3f6d269ffff21fc9da15a204c601bc",
        description: "Profile HMM alignment baseline (Liu et al. 2014).",
        command: None,
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/martinez2023_pore_geometry.py",
        sha256: "b75181ca308568dee8f8a2a02719f2d03ace6bf2eeacd6af650d73161734ff1b",
        description: "Soil pore geometry baseline (Martinez et al. 2023).",
        command: None,
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/massbank_spectral_baseline.py",
        sha256: "9ca49c6fae456c8887acafd47b66cc4f2763a099c6d88b1fdb92fc3197e027a9",
        description: "MassBank LC-MS/MS spectral library matching baseline.",
        command: None,
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/mateos2023_sulfur_phylogenomics.py",
        sha256: "83d2adec8e22a565f09953e671eb83274fe3894b5a1dff39f8f28cbb0889305f",
        description: "Sulfur-cycle phylogenomics baseline (Mateos et al. 2023).",
        command: None,
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/mhatre2020_capacitor.py",
        sha256: "c2494c5b437d1bdee18b50c435ede87df053b64408418bac22348aa3e024100e",
        description: "Bacterial capacitor and redox baseline (Mhatre et al. 2020).",
        command: None,
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/moulana2020_pangenomics.py",
        sha256: "dfa4af917716477c618d3e24b61ea20e64709b3a87afd0da794d2efe4c8094e0",
        description: "Pangenome selection statistics baseline (Moulana & Lake 2020).",
        command: None,
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/mukherjee2024_colonization.py",
        sha256: "2056f32a550fc155b4de009aa86ec64d01fda1355905ee78ee6fcf2a377a074b",
        description: "Colonization dynamics baseline (Mukherjee et al. 2024).",
        command: None,
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/newick_parse_baseline.py",
        sha256: "9c7214b25760c86e7a6f1ec394ad050d2df8046f6bb25475c21e8d891de1472b",
        description: "Newick tree parsing and tree metric baseline.",
        command: None,
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/nmf_drug_disease_pipeline.py",
        sha256: "9c9cc26a5cf320b8f22aaf50ca7b45cf9e8f83ac06a5a79d60feb8bda90fdac0",
        description: "NMF drug–disease matrix factorization pipeline baseline.",
        command: None,
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/pfas_tree_export.py",
        sha256: "4f816795f6a03b36e25ef52683ed00239d3c6d3222a3137b2c3b0cb7fd24e5b4",
        description: "PFAS phylogenetic tree export helper.",
        command: None,
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/phylohmm_introgression_baseline.py",
        sha256: "25bde4f249085a300aedd00d26c1a09eb2b6e53a8b8531cc3efc7794d350fa64",
        description: "Phylo-HMM introgression detection baseline.",
        command: None,
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/phynetpy_rf_baseline.py",
        sha256: "b3f0f0de914e2d74dc20b187d3ec7f95e23fe1a09e362136d9b0232d7f6f4e47",
        description: "PhyNetPy random-forest phylogenetic baseline.",
        command: None,
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/python_anaerobic_biogas_baseline.py",
        sha256: "5429ecaf2827b1d98d95cf4f77ea8909254086cdc180eecfcec729cfc6bc7b88",
        description: "Anaerobic digester biogas ODE baseline (ADM1-style).",
        command: None,
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/rabot2018_structure_function.py",
        sha256: "dcf5a170ccd6fbc5400967ba42234e0658d000c90b89d9c2214316ed83e500f5",
        description: "Wetland structure–function baseline (Rabot et al. 2018).",
        command: None,
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/rf_distance_baseline.py",
        sha256: "fc8f5969c1e04431afed30c32d221acb1c8bd9916d8c71c50bd47b47277f070e",
        description: "Robinson–Foulds tree distance baseline.",
        command: None,
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/run_exp002.py",
        sha256: "7fbcd4bf2f3ec2b6b73a574b49b5afd19f2d3fe1ba4dd293e302784ee0fef0ff",
        description: "Exp002 paired-end merge and quality pipeline runner.",
        command: None,
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/sate_alignment_baseline.py",
        sha256: "5584d52b2f63249c0756c6f3763e61eaf3d7ba00faf65b916301380294de9eac",
        description: "SATé-style iterative alignment baseline.",
        command: None,
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/search_ncbi_datasets.py",
        sha256: "00f1e1133cdd14d44f0aad5219f5fcb1c55f32d6bd414d894b954147ad87803f",
        description: "NCBI Datasets API search helper.",
        command: None,
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/smith_waterman_baseline.py",
        sha256: "c71fd31338975395eb5f509a4b6b505867c4862cc7dfd356b5b38aa8a07640a3",
        description: "Smith–Waterman local pairwise alignment baseline.",
        command: Some("python3 scripts/smith_waterman_baseline.py"),
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/spectral_match_baseline.py",
        sha256: "e3f6f4ff4071c4fe4f93623ca0d083c446988568112b74ae455c3e5ae1eb9241",
        description: "MS2 cosine similarity and spectral matching baseline.",
        command: None,
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/srivastava2011_multi_signal.py",
        sha256: "67e91e853506274e790b5ff5f8db40f71a69d5eb16ec02845100d07c407c55d8",
        description: "Multi-signal regulatory integration baseline (Srivastava et al. 2011).",
        command: None,
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/tecon2017_biofilm_aggregate.py",
        sha256: "c07ccf97f27777d7eaf421f12a201961acb62b15b622c7679335372b7e45c6be",
        description: "Biofilm aggregate imaging baseline (Tecon et al. 2017).",
        command: None,
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/tolerances.py",
        sha256: "98933addce61553339390a20e40394fea6e0eee7dc6703baf4c53816bbe5d5ff",
        description: "Centralized float tolerances for Python versus Rust validation.",
        command: None,
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/transe_knowledge_graph.py",
        sha256: "a8d567aaf686a847c011cc6857da9ae2d406a9a959142821c9ffa4ebbe2eab9d",
        description: "TransE knowledge-graph embedding baseline.",
        command: None,
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/validate_exp001.py",
        sha256: "1b9fcc3c6658f48e6c4fc633ef60ed78c58703033574f655e859e30c08a43dff",
        description: "Exp001 FASTQ and diversity validation against Rust.",
        command: Some("python3 scripts/validate_exp001.py"),
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/validate_public_16s_python.py",
        sha256: "e6a5ab3c66d4b7cd28b04566c829f8437697749f7e7e32f344d4671f9957e6a1",
        description: "Public 16S amplicon validation for Track 1.",
        command: Some("python3 scripts/validate_public_16s_python.py"),
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/validate_track2.py",
        sha256: "658134aa4c67b848853c9b15d7effcd5717510b01c6b1905702ca07facbde9d7",
        description: "Track 2 PFAS and analytical chemistry validation harness.",
        command: Some("python3 scripts/validate_track2.py"),
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/wang2021_rawr_bootstrap.py",
        sha256: "3924050f2b396a09e9a1d9d2f50eec9ef389abaaf4bfca0fcaf584986e83997f",
        description: "RAWR bootstrap richness baseline (Wang et al. 2021).",
        command: None,
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/wang2025_tillage_microbiome.py",
        sha256: "b947317f2aaa8378c285cca2e622d39217a797e0e8edf0570b406fa42ae491fa",
        description: "Tillage soil microbiome baseline (Wang et al. 2025).",
        command: None,
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/waters2008_qs_ode.py",
        sha256: "81d532bb914621bd75fec209e048dfa1671b994da36c951824f4b08e25a7d1da",
        description: "Quorum-sensing regulatory ODE baseline (Waters & Bassler).",
        command: Some("python3 scripts/waters2008_qs_ode.py"),
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/zheng2023_dtl_reconciliation.py",
        sha256: "5554075856376f1c46b709c6a9c976e360831fab314b36e45546338c78913937",
        description: "Duplication–transfer–loss reconciliation baseline (Zheng et al. 2023).",
        command: None,
        commit: None,
        date: None,
    },
    BaselineProvenance {
        script: "scripts/zuber2016_meta_analysis.py",
        sha256: "b3112e70f47ae972e4b3a96521e9c8086c16f92bb7bdfaad00bb7bdf10476765",
        description: "Wetland methane meta-analysis baseline (Zuber et al. 2016).",
        command: None,
        commit: None,
        date: None,
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provenance_registry_no_duplicate_scripts() {
        let mut scripts: Vec<&str> = PROVENANCE_REGISTRY.iter().map(|p| p.script).collect();
        scripts.sort_unstable();
        for w in scripts.windows(2) {
            assert_ne!(
                w[0], w[1],
                "duplicate script in PROVENANCE_REGISTRY: {}",
                w[0]
            );
        }
    }

    #[test]
    fn provenance_registry_sha256_nonempty() {
        for p in PROVENANCE_REGISTRY {
            assert!(!p.sha256.is_empty(), "empty sha256 for {}", p.script);
            assert_eq!(
                p.sha256.len(),
                64,
                "sha256 must be 64 hex chars: {}",
                p.script
            );
            assert!(
                p.sha256.chars().all(|c| c.is_ascii_hexdigit()),
                "non-hex in sha256: {}",
                p.script
            );
        }
    }

    #[test]
    fn provenance_registry_descriptions_nonempty() {
        for p in PROVENANCE_REGISTRY {
            assert!(
                !p.description.trim().is_empty(),
                "empty description for {}",
                p.script
            );
        }
    }

    #[test]
    fn provenance_registry_scripts_end_with_py() {
        for p in PROVENANCE_REGISTRY {
            assert!(
                std::path::Path::new(p.script)
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("py")),
                "script path must end with .py: {}",
                p.script
            );
        }
    }

    #[test]
    fn provenance_registry_commands_reference_own_script() {
        for p in PROVENANCE_REGISTRY {
            if let Some(cmd) = p.command {
                assert!(
                    cmd.contains(p.script),
                    "command for {} should reference the script path: got '{cmd}'",
                    p.script
                );
            }
        }
    }
}
