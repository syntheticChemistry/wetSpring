# wetSpring Experiments

Experiment protocols and results for BarraCUDA/ToadStool validation against
published tools and open data. Each experiment establishes a baseline using
existing tools (Galaxy, QIIME2, asari, FindPFAS, scipy), then validates the
Rust CPU and Rust GPU implementations against that baseline.

**Updated**: 2026-02-20 (Exp063: GPU Random Forest batch inference)

---

## Experiment Index

| Exp | Name | Track | Status | Baseline Tool | Rust Modules Validated | Checks |
|-----|------|-------|--------|---------------|----------------------|--------|
| 001 | [Galaxy Bootstrap](001_galaxy_bootstrap.md) | 1 | DONE | Galaxy 24.1 / QIIME2 / DADA2 | io::fastq, bio::quality, bio::merge_pairs, bio::derep, bio::diversity | 28 |
| 002 | Phytoplankton 16S | 1 | DONE | QIIME2 / DADA2 | (extends Exp001 with PRJNA1195978) | — |
| 003 | Phage Assembly | 1 | DONE | SPAdes / Pharokka (conda) | (assembly/annotation, no Rust module yet) | — |
| 004 | Rust FASTQ + Diversity | 1 | DONE | Exp001 baseline | io::fastq, bio::diversity, bio::kmer, bio::derep | 18 |
| 005 | [asari LC-MS](005_asari_bootstrap.md) | 2 | DONE | asari 1.13.1 | io::mzml | 7 |
| 006 | [PFΔScreen Validation](006_pfascreen_validation.md) | 2 | DONE | FindPFAS / pyOpenMS | bio::tolerance_search, bio::spectral_match, bio::kmd | 10 |
| 007 | Rust mzML + PFAS | 2 | DONE | Exp005/006 baselines | io::mzml, io::ms2, bio::kmd, bio::spectral_match | — |
| 008 | [PFAS ML Water Monitoring](008_pfas_ml_water_monitoring.md) | 2 | DONE (Phase 3) | Michigan DEQ PFAS surface water (3,719 records) | bio::decision_tree | 7 |
| 009 | [Feature Pipeline](009_feature_pipeline_validation.md) | 2 | DONE | asari 1.13.1 (MT02) | bio::eic, bio::signal, bio::feature_table | 9 |
| 010 | [Peak Detection](010_peak_detection_validation.md) | cross | DONE | scipy.signal.find_peaks | bio::signal | 17 |
| 011 | 16S Pipeline End-to-End | 1 | DONE | DADA2/UCHIME/RDP/UniFrac | bio::dada2, bio::chimera, bio::taxonomy, bio::unifrac, bio::derep, bio::diversity | 37 |
| 012 | [Algae Pond 16S](012_algae_pond_16s_validation.md) | 1 | DONE | PRJNA488170 (real NCBI data) | io::fastq, bio::quality, bio::derep, bio::dada2, bio::chimera, bio::taxonomy, bio::unifrac, bio::diversity | 29 |
| 013 | [VOC Peak Validation](013_voc_peak_validation.md) | 1/cross | DONE | Reese 2019 Table 1 (PMC6761164) | bio::signal, bio::tolerance_search | 22 |
| 014 | [Public Data Benchmarks](014_public_data_benchmarks.md) | 1 | DONE | 22 samples, 4 BioProjects vs paper ground truth | io::fastq, bio::quality, bio::derep, bio::dada2, bio::diversity | 202 |
| 015 | [Pipeline Benchmark](015_pipeline_benchmark.md) | 1 | DONE | Rust CPU vs Galaxy/QIIME2 DADA2-R | — | Benchmark |
| 016 | [GPU Pipeline Parity](016_gpu_pipeline_parity.md) | 1 | DONE | CPU vs GPU math parity (10 samples, 4 BioProjects) | bio::quality_gpu, bio::dada2_gpu, bio::chimera_gpu, bio::taxonomy_gpu, bio::diversity_gpu, bio::streaming_gpu | 88 |
| 017 | [Extended Algae Validation](017_extended_algae_validation.md) | 1 | DONE | PRJNA382322 (Nannochloropsis outdoor pilot, 162K reads) | io::fastq, bio::quality, bio::derep, bio::dada2, bio::chimera, bio::taxonomy, bio::unifrac, bio::diversity | 29 |
| 018 | [PFAS Library Validation](018_pfas_library_validation.md) | 2 | DONE | 175 PFAS (Jones Lab Zenodo 14341321) + 22 hardcoded | bio::tolerance_search, bio::spectral_match, bio::kmd | 26 |
| 019 | [Phylogenetic Validation](019_phylogenetic_validation.md) | 1b | DONE (Phase 1) | PhyNetPy gene trees (1,284 Newick), SATe 16S (Dryad) | bio::unifrac (Newick parser), bio::robinson_foulds | 30 |
| 020 | [Waters 2008 QS/c-di-GMP ODE](020_waters2008_qs_ode.md) | 1 | DONE | scipy.integrate.odeint baseline | bio::ode, bio::qs_biofilm | 16 |
| 021 | [Robinson-Foulds Validation](021_robinson_foulds_validation.md) | 1b | DONE | dendropy RF distance baseline | bio::robinson_foulds | 23 |
| 022 | [Massie 2012 Gillespie SSA](022_massie2012_gillespie.md) | 1 | DONE | numpy SSA ensemble baseline | bio::gillespie | 13 |
| 023 | [Fernandez 2020 Bistable Switching](023_fernandez2020_bistable.md) | 1 | DONE | scipy ODE bifurcation | bio::bistable, bio::ode | 14 |
| 024 | [Srivastava 2011 Multi-Signal QS](024_srivastava2011_multi_signal.md) | 1 | DONE | scipy ODE multi-signal | bio::multi_signal, bio::ode | 19 |
| 025 | [Bruger & Waters 2018 Cooperation](025_bruger2018_cooperation.md) | 1 | DONE | scipy ODE game theory | bio::cooperation, bio::ode | 20 |
| 026 | [Liu 2014 HMM Primitives](026_liu2014_hmm.md) | 1b | DONE | numpy HMM sovereign | bio::hmm | 21 |
| 027 | [Mhatre 2020 Capacitor](027_mhatre2020_capacitor.md) | 1 | DONE | scipy ODE capacitor | bio::capacitor, bio::ode | 18 |
| 028 | [Smith-Waterman Alignment](028_smith_waterman_alignment.md) | 1b | DONE | Pure Python SW | bio::alignment | 15 |
| 029 | [Felsenstein Pruning](029_felsenstein_pruning.md) | 1b/c | DONE | Pure Python JC69 | bio::felsenstein | 16 |
| 030 | [Hsueh 2022 Phage Defense](030_hsueh2022_phage_defense.md) | 1 | DONE | scipy ODE deaminase | bio::phage_defense, bio::ode | 12 |
| 031 | [Wang 2021 RAWR Bootstrap](031_wang2021_rawr_bootstrap.md) | 1b | DONE | Pure Python resampling | bio::bootstrap | 11 |
| 032 | [Alamin & Liu 2024 Placement](032_alamin2024_placement.md) | 1b | DONE | Pure Python placement | bio::placement | 12 |
| 033 | [Liu 2009 Neighbor-Joining](033_liu2009_neighbor_joining.md) | 1b | DONE | Pure Python NJ | bio::neighbor_joining | 16 |
| 034 | [Zheng 2023 DTL Reconciliation](034_zheng2023_dtl_reconciliation.md) | 1b | DONE | Pure Python DTL | bio::reconciliation | 14 |
| 035 | [BarraCUDA CPU Parity v2](035_barracuda_cpu_parity_v2.md) | cross | DONE | CPU v1 extension | batch/flat APIs (5 domains) | 18 |
| 036 | [PhyNetPy RF Distances](036_phynetpy_rf_distances.md) | 1b | DONE | PhyNetPy gene trees | bio::robinson_foulds | 15 |
| 037 | [PhyloNet-HMM Discordance](037_phylohmm_discordance.md) | 1b | DONE | PhyloNet-HMM | bio::hmm | 10 |
| 038 | [SATé Pipeline Benchmark](038_sate_pipeline_benchmark.md) | 1b | DONE | SATé pipeline | bio::alignment, bio::neighbor_joining | 17 |
| 039 | [Algal Pond Time-Series](039_algae_timeseries.md) | 1 | DONE | Cahill proxy | bio::diversity, time-series | 11 |
| 040 | [Bloom Surveillance](040_bloom_surveillance.md) | 1 | DONE | Smallwood proxy | 16S pipeline, bio::diversity | 15 |
| 041 | [EPA PFAS ML](041_epa_pfas_ml.md) | 2 | DONE | Jones F&T proxy | bio::decision_tree | 14 |
| 042 | [MassBank Spectral](042_massbank_spectral.md) | 2 | DONE | Jones MS proxy | bio::spectral_match | 9 |
| 043 | [BarraCUDA CPU Parity v3](043_barracuda_cpu_v3.md) | cross | DONE | 18-domain coverage | 9 new domains (84 total) | 45 |
| 044 | [BarraCUDA GPU v3](044_barracuda_gpu_v3.md) | cross | DONE | GPU parity | diversity, spectral, stats | 14 |
| 045 | [ToadStool Bio Absorption](045_toadstool_bio_absorption.md) | cross/GPU | DONE | ToadStool cce8fe7c | SmithWatermanGpu, TreeInferenceGpu, GillespieGpu | 10 |
| 046 | [GPU Phylo Composition](046_gpu_phylo_composition.md) | GPU | DONE | CPU Felsenstein | FelsensteinGpu → bootstrap + placement | 15 |
| 047 | [GPU HMM Batch Forward](047_gpu_hmm_forward.md) | GPU | DONE | CPU HMM forward | HmmGpuForward (local WGSL) | 13 |
| 048 | [CPU vs GPU Benchmark](048_cpu_gpu_benchmark_phylo_hmm.md) | GPU | DONE | CPU baselines | Felsenstein, Bootstrap, HMM batch | 6 |
| 049 | [GPU ODE Parameter Sweep](049_gpu_ode_parameter_sweep.md) | GPU | DONE | `qs_biofilm::run_scenario` | ODE sweep (local WGSL), pow_f64 polyfill | 7 |
| 050 | [GPU Bifurcation Eigenvalues](050_gpu_bifurcation_eigenvalues.md) | GPU | DONE | Power iteration | BatchedEighGpu Jacobian eigenvalues | 5 |
| 051 | [Rare Biosphere Diversity](051_rare_biosphere.md) | 1c | DONE | Anderson 2015 rare biosphere | bio::diversity, bio::ani | 35 |
| 052 | [Viral Metagenomics](052_viral_metagenomics.md) | 1c | DONE | Anderson 2014 viral dN/dS | bio::dnds, bio::snp | 22 |
| 053 | [Sulfur Phylogenomics](053_sulfur_phylogenomics.md) | 1c | DONE | Mateos 2023 molecular clock | bio::molecular_clock | 15 |
| 054 | [Phosphorus Phylogenomics](054_phosphorus_phylogenomics.md) | 1c | DONE | Boden 2024 phosphorus clock | bio::molecular_clock | 13 |
| 055 | [Population Genomics](055_population_genomics.md) | 1c | DONE | Anderson 2017 ANI/SNP | bio::ani, bio::snp | 24 |
| 056 | [Pangenomics](056_pangenomics.md) | 1c | DONE | Moulana 2020 pangenome | bio::pangenome | 24 |
| 057 | [BarraCUDA CPU Parity v4](057_barracuda_cpu_v4.md) | cross | DONE | Track 1c CPU | 5 new domains (128 total) | 44 |
| 058 | [GPU Track 1c Promotion](058_gpu_track1c_promotion.md) | GPU | DONE | CPU Track 1c | ANI, SNP, dN/dS, pangenome WGSL shaders | 27 |
| 059 | [23-Domain Benchmark](059_23_domain_benchmark.md) | cross | DONE | Rust vs Python | 22.5× overall speedup | 20 |
| 060 | [Cross-Substrate metalForge](060_cross_substrate_metalforge.md) | cross | DONE | CPU↔GPU parity | metalForge substrate-independence | 20 |
| 061 | [Random Forest Inference](061_random_forest_inference.md) | ML | DONE | RF majority vote | bio::random_forest, bio::random_forest_gpu | 13 |
| 062 | [GBM Inference](062_gbm_inference.md) | ML | DONE | GBM sigmoid/softmax | bio::gbm | 16 |
| 063 | GPU Random Forest Batch | GPU/ML | DONE | CPU RF | rf_batch_inference.wgsl (SoA layout) | 13 |

---

## Results Directory Structure

```
experiments/
├── README.md                           ← this file
├── 001_galaxy_bootstrap.md             ← Galaxy/QIIME2 setup + 16S baseline
├── 005_asari_bootstrap.md              ← asari LC-MS setup + MT02 baseline
├── 006_pfascreen_validation.md         ← FindPFAS setup + PFAS baseline
├── 009_feature_pipeline_validation.md  ← Rust feature pipeline vs asari
├── 010_peak_detection_validation.md    ← Rust peak detection vs scipy
│
└── results/
    ├── 001_galaxy_bootstrap/
    │   ├── validation_report.json      ← DADA2/taxonomy summary stats
    │   ├── dada2-stats.tsv             ← per-sample DADA2 stats
    │   └── manifest.tsv                ← sample manifest
    ├── 002_phytoplankton/
    │   ├── exp002_report.json          ← 2,273 ASVs, 10 samples
    │   └── diversity_report.json       ← diversity metrics
    ├── 003_phage/
    │   └── exp003_report.json          ← assembly + annotation stats
    ├── 005_asari/
    │   ├── exp005_report.json          ← 5,951 features, 8 files
    │   ├── preferred_Feature_table.tsv ← asari feature table (5,951 rows)
    │   └── project.json                ← asari project config
    ├── 006_pfascreen/
    │   └── exp006_report.json          ← 25 PFAS precursors
    ├── 010_peak_baselines/
    │   ├── scipy_baselines.json        ← scipy provenance summary
    │   ├── single_gaussian.dat         ← test vector (200 points, 1 peak)
    │   ├── three_chromatographic.dat   ← test vector (500 points, 3 peaks)
    │   ├── noisy_with_spikes.dat       ← test vector (1000 points, 3 peaks)
    │   ├── overlapping_peaks.dat       ← test vector (200 points, 1 peak)
    │   └── monotonic_no_peaks.dat      ← test vector (100 points, 0 peaks)
    ├── 013_voc_baselines/
    │   └── reese2019_table1.tsv        ← 14 VOC compounds from Reese 2019
    ├── paper_benchmarks/
    │   ├── README.md                    ← benchmark strategy
    │   ├── humphrey2023_bacteriome.tsv  ← 18 OTUs, core genera
    │   ├── humphrey2023_metrics.json    ← community profile, validation targets
    │   ├── carney2016_crash_agents.json ← crash agents, detection methods
    │   ├── reese2019_voc_biomarkers.json← 14 VOC compounds, RI values
    │   └── reichardt2020_spectroradiometric.json ← organisms, methods
    ├── ncbi_dataset_search/
    │   └── ncbi_search_results.json     ← NCBI Entrez search results
    ├── 019_phylogenetic/
    │   └── newick_parse_python_baseline.json ← dendropy Newick parse stats
    ├── 021_rf_baseline/
    │   └── rf_python_baseline.json           ← dendropy RF distance baseline
    ├── 022_gillespie/
    │   └── gillespie_python_baseline.json    ← numpy SSA ensemble stats
    └── track2_validation_report.json   ← combined Track 2 validation
```

---

## Validation Binaries

Each validation binary uses the `Validator` harness with provenance tables,
hardcoded expected values from the baseline experiments, and tolerance
thresholds from `src/tolerances.rs`.

| Binary | Experiment | Checks | Command |
|--------|------------|--------|---------|
| `validate_fastq` | 001 | 28 | `cargo run --bin validate_fastq` |
| `validate_diversity` | 001/004 | 27 | `cargo run --bin validate_diversity` |
| `validate_mzml` | 005 | 7 | `cargo run --bin validate_mzml` |
| `validate_pfas` | 006 | 10 | `cargo run --bin validate_pfas` |
| `validate_features` | 009 | 9 | `cargo run --bin validate_features` |
| `validate_peaks` | 010 | 17 | `cargo run --bin validate_peaks` |
| `validate_diversity_gpu` | — | 38 | `cargo run --features gpu --bin validate_diversity_gpu` |
| `validate_16s_pipeline` | 011 | 37 | `cargo run --bin validate_16s_pipeline` |
| `validate_algae_16s` | 012 | 29 | `cargo run --bin validate_algae_16s` |
| `validate_voc_peaks` | 013 | 22 | `cargo run --bin validate_voc_peaks` |
| `validate_public_benchmarks` | 014 | 202 | `cargo run --bin validate_public_benchmarks` |
| `benchmark_pipeline` | 015 | — | `cargo run --release --bin benchmark_pipeline` |
| `validate_16s_pipeline_gpu` | 016 | 88 | `cargo run --features gpu --release --bin validate_16s_pipeline_gpu` |
| `validate_extended_algae` | 017 | 29 | `cargo run --bin validate_extended_algae` |
| `validate_pfas_library` | 018 | 21 | `cargo run --bin validate_pfas_library` |
| `validate_newick_parse` | 019 | 30 | `cargo run --bin validate_newick_parse` |
| `validate_qs_ode` | 020 | 16 | `cargo run --bin validate_qs_ode` |
| `validate_rf_distance` | 021 | 23 | `cargo run --bin validate_rf_distance` |
| `validate_gillespie` | 022 | 13 | `cargo run --bin validate_gillespie` |
| `validate_pfas_decision_tree` | 008 | 7 | `cargo run --bin validate_pfas_decision_tree` |
| `validate_gpu_phylo_compose` | 046 | 15 | `cargo run --features gpu --release --bin validate_gpu_phylo_compose` |
| `validate_gpu_hmm_forward` | 047 | 13 | `cargo run --features gpu --release --bin validate_gpu_hmm_forward` |
| `benchmark_phylo_hmm_gpu` | 048 | 6 | `cargo run --features gpu --release --bin benchmark_phylo_hmm_gpu` |
| `validate_gpu_ode_sweep` | 049-050 | 12 | `cargo run --features gpu --bin validate_gpu_ode_sweep` |
| `validate_rare_biosphere` | 051 | 35 | `cargo run --bin validate_rare_biosphere` |
| `validate_viral_metagenomics` | 052 | 22 | `cargo run --bin validate_viral_metagenomics` |
| `validate_sulfur_phylogenomics` | 053 | 15 | `cargo run --bin validate_sulfur_phylogenomics` |
| `validate_phosphorus_phylogenomics` | 054 | 13 | `cargo run --bin validate_phosphorus_phylogenomics` |
| `validate_population_genomics` | 055 | 24 | `cargo run --bin validate_population_genomics` |
| `validate_pangenomics` | 056 | 24 | `cargo run --bin validate_pangenomics` |
| `validate_barracuda_cpu_v4` | 057 | 44 | `cargo run --release --bin validate_barracuda_cpu_v4` |
| `validate_gpu_track1c` | 058 | 27 | `cargo run --features gpu --bin validate_gpu_track1c` |
| `validate_gpu_23_domain_benchmark` | 059 | 20 | `cargo run --features gpu --bin validate_gpu_23_domain_benchmark` |
| `validate_gpu_cross_substrate` | 060 | skip | `cargo run --features gpu --bin validate_gpu_cross_substrate` |
| `validate_barracuda_cpu_v5` | 061-062 | 29 | `cargo run --release --bin validate_barracuda_cpu_v5` |
| `validate_gpu_rf` | 063 | 13 | `cargo run --features gpu --bin validate_gpu_rf` |
| `benchmark_cpu_gpu` | — | — | `cargo run --release --features gpu --bin benchmark_cpu_gpu` |

**Total validation checks**: 1,501 (1,241 CPU + 260 GPU)
**Rust unit/integration tests**: 582 lib + integration + doc
**Validation binaries**: 29 CPU + 12 GPU
**Benchmark infrastructure**: `bench.rs` harness with RAPL + nvidia-smi energy profiling, JSON output

---

## How to Add a New Experiment

1. Create `experiments/NNN_descriptive_name.md` with: date, status, objective,
   baseline tool, dataset, protocol, acceptance criteria.
2. Run the baseline tool and save results to `experiments/results/NNN_name/`.
3. Create `src/bin/validate_NNN.rs` with provenance table and `Validator` checks.
4. Add the `[[bin]]` entry to `Cargo.toml`.
5. Update this README with the new experiment row.
