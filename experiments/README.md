# wetSpring Experiments

Experiment protocols and results for BarraCuda/ToadStool validation against
published tools and open data. Each experiment establishes a baseline using
existing tools (Galaxy, QIIME2, asari, FindPFAS, scipy), then validates the
Rust CPU and Rust GPU implementations against that baseline.

**Updated**: 2026-02-28 (Phase 73: 229 experiments, 5,743+ checks (1,833+ GPU on RTX 4070, 60 NPU on AKD1000), ToadStool S68+ (`e96576ee`), 79 primitives consumed, barracuda always-on, 1,199+ tests, 92 named tolerances, clippy pedantic CLEAN, 0 Passthrough, V73 deep debt reduction + V72 five-tier validation chain, 50/50 three-tier, 52/52 papers)

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
| 009 | [Feature Pipeline](009_feature_pipeline_validation.md) | 2 | DONE | asari 1.13.1 (MT02) | bio::eic, bio::signal, bio::feature_table | 8 |
| 010 | [Peak Detection](010_peak_detection_validation.md) | cross | DONE | scipy.signal.find_peaks | bio::signal | 17 |
| 011 | 16S Pipeline End-to-End | 1 | DONE | DADA2/UCHIME/RDP/UniFrac | bio::dada2, bio::chimera, bio::taxonomy, bio::unifrac, bio::derep, bio::diversity | 37 |
| 012 | [Algae Pond 16S](012_algae_pond_16s_validation.md) | 1 | DONE | PRJNA488170 (real NCBI data) | io::fastq, bio::quality, bio::derep, bio::dada2, bio::chimera, bio::taxonomy, bio::unifrac, bio::diversity | 34 |
| 013 | [VOC Peak Validation](013_voc_peak_validation.md) | 1/cross | DONE | Reese 2019 Table 1 (PMC6761164) | bio::signal, bio::tolerance_search | 22 |
| 014 | [Public Data Benchmarks](014_public_data_benchmarks.md) | 1 | DONE | 22 samples, 4 BioProjects vs paper ground truth | io::fastq, bio::quality, bio::derep, bio::dada2, bio::diversity | 202 |
| 015 | [Pipeline Benchmark](015_pipeline_benchmark.md) | 1 | DONE | Rust CPU vs Galaxy/QIIME2 DADA2-R | — | Benchmark |
| 016 | [GPU Pipeline Parity](016_gpu_pipeline_parity.md) | 1 | DONE | CPU vs GPU math parity (10 samples, 4 BioProjects) | bio::quality_gpu, bio::dada2_gpu, bio::chimera_gpu, bio::taxonomy_gpu, bio::diversity_gpu, bio::streaming_gpu | 88 |
| 017 | [Extended Algae Validation](017_extended_algae_validation.md) | 1 | DONE | PRJNA382322 (Nannochloropsis outdoor pilot, 162K reads) | io::fastq, bio::quality, bio::derep, bio::dada2, bio::chimera, bio::taxonomy, bio::unifrac, bio::diversity | 35 |
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
| 035 | [BarraCuda CPU Parity v2](035_barracuda_cpu_parity_v2.md) | cross | DONE | CPU v1 extension | batch/flat APIs (5 domains) | 18 |
| 036 | [PhyNetPy RF Distances](036_phynetpy_rf_distances.md) | 1b | DONE | PhyNetPy gene trees | bio::robinson_foulds | 15 |
| 037 | [PhyloNet-HMM Discordance](037_phylohmm_discordance.md) | 1b | DONE | PhyloNet-HMM | bio::hmm | 10 |
| 038 | [SATé Pipeline Benchmark](038_sate_pipeline_benchmark.md) | 1b | DONE | SATé pipeline | bio::alignment, bio::neighbor_joining | 17 |
| 039 | [Algal Pond Time-Series](039_algae_timeseries.md) | 1 | DONE | Cahill proxy | bio::diversity, time-series | 11 |
| 040 | [Bloom Surveillance](040_bloom_surveillance.md) | 1 | DONE | Smallwood proxy | 16S pipeline, bio::diversity | 15 |
| 041 | [EPA PFAS ML](041_epa_pfas_ml.md) | 2 | DONE | Jones F&T proxy | bio::decision_tree | 14 |
| 042 | [MassBank Spectral](042_massbank_spectral.md) | 2 | DONE | Jones MS proxy | bio::spectral_match | 9 |
| 043 | [BarraCuda CPU Parity v3](043_barracuda_cpu_v3.md) | cross | DONE | 18-domain coverage | 9 new domains (84 total) | 45 |
| 044 | [BarraCuda GPU v3](044_barracuda_gpu_v3.md) | cross | DONE | GPU parity | diversity, spectral, stats | 14 |
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
| 057 | [BarraCuda CPU Parity v4](057_barracuda_cpu_v4.md) | cross | DONE | Track 1c CPU | 5 new domains (128 total) | 44 |
| 058 | [GPU Track 1c Promotion](058_gpu_track1c_promotion.md) | GPU | DONE | CPU Track 1c | ANI, SNP, dN/dS, pangenome WGSL shaders | 27 |
| 059 | [23-Domain Benchmark](059_23_domain_benchmark.md) | cross | DONE | Rust vs Python | 22.5× overall speedup | 20 |
| 060 | [Cross-Substrate metalForge](060_cross_substrate_metalforge.md) | cross | DONE | CPU↔GPU parity | metalForge substrate-independence | 20 |
| 061 | [Random Forest Inference](061_random_forest_inference.md) | ML | DONE | RF majority vote | bio::random_forest, bio::random_forest_gpu | 13 |
| 062 | [GBM Inference](062_gbm_inference.md) | ML | DONE | GBM sigmoid/softmax | bio::gbm | 16 |
| 063 | GPU Random Forest Batch | GPU/ML | DONE | CPU RF | rf_batch_inference.wgsl (SoA layout) | 13 |
| 064 | [BarraCuda GPU Parity v1](064_barracuda_gpu_parity_v1.md) | cross/GPU | DONE | CPU reference | diversity, BC, ANI, SNP, dN/dS, pangenome, RF, HMM | 26 |
| 065 | [metalForge Full Cross-System](065_metalforge_full_cross_system.md) | cross | DONE | CPU↔GPU parity | Full portfolio substrate-independence proof | 35 |
| 066 | [CPU vs GPU Scaling Benchmark](066_cpu_vs_gpu_scaling_all_domains.md) | GPU | DONE | CPU vs GPU timing | benchmark_all_domains_cpu_gpu | Benchmark |
| 067 | [ToadStool Dispatch Overhead Profiling](067_dispatch_overhead_profiling.md) | GPU | DONE | Dispatch profiling | benchmark_dispatch_overhead | Benchmark |
| 068 | [Pipeline Caching Optimization](068_pipeline_caching_optimization.md) | GPU | DONE | Optimization | benchmark_dispatch_overhead (reuses) | Benchmark |
| 069 | [Python vs Rust CPU vs GPU Three-Tier Benchmark](069_python_vs_rust_cpu_vs_gpu.md) | cross | DONE | Three-tier timing | benchmark_three_tier | Benchmark |
| 070 | [BarraCuda CPU 25-Domain Pure Rust Math Proof](070_barracuda_cpu_25_domain_proof.md) | cross | COMPLETE | 25 domains consolidated | validate_barracuda_cpu_full | 50 |
| 071 | [BarraCuda GPU Math Portability Proof](071_barracuda_gpu_portability_proof.md) | GPU | COMPLETE | GPU math portability | validate_barracuda_gpu_full | 24 |
| 075 | [Pure GPU Analytics Pipeline](075_pure_gpu_analytics_pipeline.md) | GPU | DONE | Pure GPU pipeline | validate_pure_gpu_pipeline | 31 |
| 076 | [Cross-Substrate Pipeline](076_metalforge_cross_substrate_pipeline.md) | cross/GPU | DONE | Cross-substrate pipeline | validate_cross_substrate_pipeline | 17 |
| 077 | [ToadStool Bio Rewire](077_toadstool_bio_rewire.md) | GPU/cross | DONE | ToadStool bio primitive rewire | (all GPU binaries) | 451 (re-validated) |
| 087 | GPU Extended Domains (EIC/PCoA/Kriging/Rarefaction) | GPU | DONE | — | validate_gpu_extended | 50+ |
| 088 | metalForge PCIe Direct Transfer | metalForge | DONE | — | validate_pcie_direct | 32 |
| 089 | ToadStool Streaming Dispatch | streaming | DONE | — | validate_streaming_dispatch | 25 |
| 090 | Pure GPU Streaming Pipeline | GPU/streaming | DONE | — | validate_pure_gpu_streaming | 80 |
| 091 | Streaming vs Round-Trip Benchmark | GPU/benchmark | DONE | — | benchmark_streaming_vs_roundtrip | 2 |
| 092 | CPU vs GPU All 16 Domains | GPU/parity | DONE | — | validate_cpu_vs_gpu_all_domains | 48 |
| 093 | metalForge Full v3 (16 domains) | metalForge | DONE | — | validate_metalforge_full_v3 | 28 |
| 094 | [Cross-Spring Evolution Validation](094_cross_spring_evolution.md) | cross/GPU | DONE | CPU baselines | validate_cross_spring_evolution | 39 |
| 095 | [Cross-Spring Scaling Benchmark](095_cross_spring_scaling_benchmark.md) | GPU/benchmark | DONE | — | benchmark_cross_spring_scaling | Benchmark |
| 096 | Local WGSL Compile + Dispatch | GPU | DONE | — | validate_local_wgsl_compile | 10 |
| 099 | CPU/GPU expanded parity | cross/GPU | DONE | — | validate_cpu_gpu_expanded | 27 |
| 100 | metalForge v4 (16 domains) | metalForge | DONE | — | validate_metalforge_v4 | 28 |
| 101 | Pure GPU complete pipeline | GPU/streaming | DONE | — | validate_pure_gpu_complete | 52 |
| 102 | BarraCuda CPU v8 (25 domains) | cross | DONE | — | validate_barracuda_cpu_v8 | 175 |
| 103 | metalForge v5 (25/25 papers) | metalForge | DONE | — | validate_metalforge_v5 | 58 |
| 104 | metalForge v6 (25/25 papers) | metalForge | DONE | — | validate_metalforge_v6 | 24 |
| 105 | Pure GPU streaming v2 | streaming | DONE | — | validate_pure_gpu_streaming_v2 | 27 |
| 106 | Streaming ODE + phylo | streaming | DONE | — | validate_streaming_ode_phylo | 45 |
| 157 | Fajgenbaum pathway scoring | 3 | DONE | Published equations | validate_fajgenbaum_pathway | 8 |
| 158 | MATRIX pharmacophenomics | 3 | DONE | Lancet Haematology 2025 | validate_matrix_pharmacophenomics | 9 |
| 159 | Yang 2020 NMF drug repurposing | 3 | DONE | PMC | validate_nmf_drug_repurposing | 7 |
| 160 | Gao 2020 repoDB NMF | 3 | DONE | PMC | validate_repodb_nmf | 9 |
| 161 | ROBOKOP KG embedding | 3 | DONE | KG infrastructure | validate_knowledge_graph_embedding | 7 |
| 162 | Cross-spring S57 evolution | cross | DONE | S57 rewire | benchmark_cross_spring_evolution | 66 |
| 163 | BarraCuda CPU v9 (Track 3) | cross | DONE | CPU + Track 3 | validate_barracuda_cpu_v9 | 66 |
| 164 | GPU drug repurposing | GPU | DONE | GPU Track 3 | validate_gpu_drug_repurposing | 48 |
| 165 | metalForge drug repurposing | metalForge | DONE | Substrate-independent | validate_metalforge_drug_repurposing | 25 |
| 166 | Modern systems S66 | cross/GPU | DONE | Exp166 benchmark | benchmark_modern_systems_df64 | 19 |
| 167 | Diversity fusion GPU extension | GPU | DONE | CPU ↔ GPU parity | validate_gpu_diversity_fusion | 18 |
| 168 | Cross-spring S62/S66 validation | cross/GPU | DONE | S62+DF64 rewire | validate_cross_spring_s62 | ~25 |
| 169 | Modern cross-spring benchmark | cross/GPU | DONE | V44 rewire benchmark | benchmark_cross_spring_modern | 12 |
| 183 | Cross-Spring Evolution Benchmark (S66) | cross/GPU | DONE | S66 rewire benchmark | benchmark_cross_spring_s65 | 36 |
| 184 | [Real NCBI Sovereign Pipeline](184_real_ncbi_sovereign_pipeline.md) | 1 | DONE | Real NCBI 16S sovereign | validate_real_ncbi_pipeline | 25 |
| 185 | [Cold Seep Sovereign Pipeline](185_cold_seep_sovereign_pipeline.md) | 1c | DONE | Cold seep metagenomes | validate_cold_seep_pipeline | 8 |
| 186 | [Dynamic Anderson W(t)](186_dynamic_anderson.md) | cross/GPU | DONE | Community evolution | validate_dynamic_anderson | 7 |
| 187 | [DF64 Anderson Large Lattice](187_df64_anderson_large_lattice.md) | cross/GPU | DONE | DF64 L=24+ lattice | validate_df64_anderson | 4 |
| 188 | [NPU Sentinel Real Stream](188_npu_sentinel_real_stream.md) | NPU | DONE | Real sensor stream | validate_npu_sentinel_stream | 10 |
| 189 | [Cross-Spring Evolution S68](189_cross_spring_evolution_s68.md) | cross/GPU | DONE | S68 universal precision | benchmark_cross_spring_s68 | varies |
| 190 | [BarraCuda CPU v10 — V59 Science](190_barracuda_cpu_v10.md) | CPU/cross | DONE | V59 science domains | validate_barracuda_cpu_v10 | 75 |
| 191 | [GPU V59 Science Parity](191_gpu_v59_science.md) | GPU | DONE | Diversity + Anderson | validate_gpu_v59_science | 29 |
| 192 | [metalForge V59 Cross-Substrate](192_metalforge_v59_science.md) | metalForge | DONE | CPU↔GPU parity | validate_metalforge_v59_science | 36 |
| 193 | [NPU Hardware Validation](193_npu_hardware_validation.md) | NPU | DONE | Real AKD1000 DMA | validate_npu_hardware | 7 sections |
| 194 | [NPU Live ESN](194_npu_live_esn.md) | NPU | DONE | Sim↔hardware ESN | validate_npu_live | 23 |
| 195 | [Funky NPU Explorations](195_npu_funky_explorations.md) | NPU | DONE | AKD1000 novelties | validate_npu_funky | 14 |
| 196a | [Nanopore Signal Bridge](196a_nanopore_signal_bridge.md) | field genomics | DONE | POD5/NRS reader | io::nanopore, validate_nanopore_signal_bridge | 28 |
| 196b | [Simulated 16S Pipeline](196b_simulated_16s_pipeline.md) | field genomics | DONE | Nanopore→16S | bio::dada2, bio::taxonomy, validate_nanopore_simulated_16s | 11 |
| 196c | [NPU Int8 Quantization](196c_npu_int8_quantization.md) | field genomics | DONE | Community→NPU | bio::esn, validate_nanopore_int8_quantization | 13 |
| 196 | Nanopore Signal Bridge (full POD5) | field genomics | PARTIAL (196a-c done) | Real POD5 data | — | — |
| 197 | NPU Adaptive Sampling | field genomics | PLANNED | MinKNOW feedback | — | — |
| 198 | Field Bloom Sentinel E2E | field genomics | PLANNED | MinION → NPU | — | — |
| 199 | Soil 16S Field Pipeline | field genomics | PLANNED | MinION → Anderson | — | — |
| 200 | Soil Health NPU Classifier | field genomics | PLANNED | NPU soil class | — | — |
| 201 | AMR Gene Detection | field genomics | PLANNED | Long-read AMR | — | — |
| 202 | AMR Threat NPU Classifier | field genomics | PLANNED | NPU AMR class | — | — |
| 203 | [biomeOS Science Pipeline](203_biomeos_science_pipeline.md) | cross/IPC | DONE | wetspring-server | ipc::server, ipc::dispatch, ipc::protocol, ipc::metrics | 29 |
| 204 | [Capability Discovery](204_capability_discovery.md) | cross/IPC | DONE | Songbird | ipc::songbird | (within 203) |
| 205 | [Sovereign Fallback](205_sovereign_fallback.md) | cross/IPC | DONE | Three-tier routing | ncbi::nestgate | (within 203) |
| 206 | [BarraCuda CPU v11](206_barracuda_cpu_v11_ipc_dispatch.md) | cross/IPC | DONE | Direct function calls | ipc::dispatch, bio::diversity, bio::qs_biofilm | 64 |
| 207 | [BarraCuda GPU v4](207_barracuda_gpu_v4_ipc_science.md) | cross/GPU/IPC | DONE | CPU diversity/QS/Anderson | ipc::dispatch, ToadStool GPU primitives | 54 |
| 208 | [metalForge v7](208_metalforge_v7_mixed_nucleus.md) | cross/IPC/metalForge | DONE | CPU direct calls | ipc::dispatch, forge routing model, NUCLEUS atomics | 74 |

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
| `validate_features` | 009 | 8 | `cargo run --bin validate_features` |
| `validate_peaks` | 010 | 17 | `cargo run --bin validate_peaks` |
| `validate_diversity_gpu` | — | 38 | `cargo run --features gpu --bin validate_diversity_gpu` |
| `validate_16s_pipeline` | 011 | 37 | `cargo run --bin validate_16s_pipeline` |
| `validate_algae_16s` | 012 | 34 | `cargo run --bin validate_algae_16s` |
| `validate_voc_peaks` | 013 | 22 | `cargo run --bin validate_voc_peaks` |
| `validate_public_benchmarks` | 014 | 202 | `cargo run --bin validate_public_benchmarks` |
| `benchmark_pipeline` | 015 | — | `cargo run --release --bin benchmark_pipeline` |
| `validate_16s_pipeline_gpu` | 016 | 88 | `cargo run --features gpu --release --bin validate_16s_pipeline_gpu` |
| `validate_extended_algae` | 017 | 35 | `cargo run --bin validate_extended_algae` |
| `validate_pfas_library` | 018 | 26 | `cargo run --bin validate_pfas_library` |
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
| `benchmark_23_domain_timing` | 059 | — | `cargo run --release --bin benchmark_23_domain_timing` |
| `validate_cross_substrate` | 060 | 20 | `cargo run --features gpu --bin validate_cross_substrate` |
| `validate_barracuda_cpu_v5` | 061-062 | 29 | `cargo run --release --bin validate_barracuda_cpu_v5` |
| `validate_gpu_rf` | 063 | 13 | `cargo run --features gpu --bin validate_gpu_rf` |
| `validate_barracuda_gpu_v1` | 064 | 26 | `cargo run --features gpu --release --bin validate_barracuda_gpu_v1` |
| `validate_metalforge_full` | 065 | 35 | `cargo run --features gpu --release --bin validate_metalforge_full` |
| `benchmark_all_domains_cpu_gpu` | 066 | — | `cargo run --release --features gpu --bin benchmark_all_domains_cpu_gpu` |
| `benchmark_dispatch_overhead` | 067/068 | — | `cargo run --release --features gpu --bin benchmark_dispatch_overhead` |
| `benchmark_three_tier` | 069 | — | `cargo run --release --features gpu --bin benchmark_three_tier` |
| `benchmark_cpu_gpu` | — | — | `cargo run --release --features gpu --bin benchmark_cpu_gpu` |
| `validate_barracuda_cpu_full` | 070 | 50 | `cargo run --release --bin validate_barracuda_cpu_full` |
| `validate_barracuda_gpu_full` | 071 | 24 | `cargo run --features gpu --release --bin validate_barracuda_gpu_full` |
| `validate_gpu_streaming_pipeline` | 072 | 17 | `cargo run --features gpu --release --bin validate_gpu_streaming_pipeline` |
| `validate_dispatch_overhead_proof` | 073 | 21 | `cargo run --features gpu --release --bin validate_dispatch_overhead_proof` |
| `validate_substrate_router` | 074 | 20 | `cargo run --features gpu --release --bin validate_substrate_router` |
| `validate_pure_gpu_pipeline` | 075 | 31 | `cargo run --features gpu --release --bin validate_pure_gpu_pipeline` |
| `validate_cross_substrate_pipeline` | 076 | 17 | `cargo run --features gpu --release --bin validate_cross_substrate_pipeline` |
| *(ToadStool Bio Rewire)* | 077 | 451 | All GPU binaries re-validated after 8-module rewire |
| *(ODE GPU Sweep Readiness)* | 078 | 10 | Flat param APIs for 5 ODE modules (unit tests) |
| `validate_barracuda_cpu_v6` | 079 | 48 | `cargo run --release --bin validate_barracuda_cpu_v6` |
| `validate_dispatch_routing` (forge) | 080 | 35 | `cd metalForge/forge && cargo run --bin validate_dispatch_routing` |
| *(K-mer GPU Histogram)* | 081 | 4 | Flat histogram + sorted pairs (unit tests) |
| *(UniFrac CSR Flat Tree)* | 082 | 4 | CSR tree + sample matrix (unit tests) |
| *(Taxonomy NPU Int8)* | 083 | 3 | Int8 quantization + argmax parity (unit tests) |
| `validate_metalforge_full_v2` | 084 | 35+ | `cargo run --features gpu --release --bin validate_metalforge_full_v2` |
| `validate_barracuda_cpu_v7` | 085 | 43 | `cargo run --release --bin validate_barracuda_cpu_v7` |
| `validate_metalforge_pipeline` | 086 | 45 | `cargo run --release --bin validate_metalforge_pipeline` |
| `validate_gpu_extended` | 087 | 50+ | `cargo run --features gpu --bin validate_gpu_extended` |
| `validate_pcie_direct` | 088 | 32 | `cargo run --bin validate_pcie_direct` |
| `validate_streaming_dispatch` | 089 | 25 | `cargo run --bin validate_streaming_dispatch` |
| `validate_pure_gpu_streaming` | 090 | 80 | `cargo run --features gpu --release --bin validate_pure_gpu_streaming` |
| `benchmark_streaming_vs_roundtrip` | 091 | 2 | `cargo run --features gpu --release --bin benchmark_streaming_vs_roundtrip` |
| `validate_cpu_vs_gpu_all_domains` | 092 | 48 | `cargo run --features gpu --release --bin validate_cpu_vs_gpu_all_domains` |
| `validate_metalforge_full_v3` | 093 | 28 | `cargo run --features gpu --release --bin validate_metalforge_full_v3` |
| `validate_cross_spring_evolution` | 094 | 39 | `cargo run --features gpu --bin validate_cross_spring_evolution` |
| `benchmark_cross_spring_scaling` | 095 | — | `cargo run --release --features gpu --bin benchmark_cross_spring_scaling` |
| `validate_local_wgsl_compile` | 096 | 10 | `cargo run --features gpu --bin validate_local_wgsl_compile` |
| `validate_cpu_gpu_expanded` | 099 | 27 | `cargo run --features gpu --bin validate_cpu_gpu_expanded` |
| `validate_metalforge_v4` | 100 | 28 | `cargo run --features gpu --bin validate_metalforge_v4` |
| `validate_pure_gpu_complete` | 101 | 52 | `cargo run --features gpu --bin validate_pure_gpu_complete` |
| `validate_barracuda_cpu_v8` | 102 | 175 | `cargo run --release --bin validate_barracuda_cpu_v8` |
| `validate_metalforge_v5` | 103 | 58 | `cargo run --features gpu --bin validate_metalforge_v5` |
| `validate_metalforge_v6` | 104 | 24 | `cargo run --features gpu --bin validate_metalforge_v6` |
| `validate_pure_gpu_streaming_v2` | 105 | 27 | `cargo run --features gpu --bin validate_pure_gpu_streaming_v2` |
| `validate_streaming_ode_phylo` | 106 | 45 | `cargo run --features gpu --bin validate_streaming_ode_phylo` |
| `validate_fajgenbaum_pathway` | 157 | 8 | `cargo run --bin validate_fajgenbaum_pathway` |
| `validate_matrix_pharmacophenomics` | 158 | 9 | `cargo run --bin validate_matrix_pharmacophenomics` |
| `validate_nmf_drug_repurposing` | 159 | 7 | `cargo run --bin validate_nmf_drug_repurposing` |
| `validate_repodb_nmf` | 160 | 9 | `cargo run --bin validate_repodb_nmf` |
| `validate_knowledge_graph_embedding` | 161 | 7 | `cargo run --bin validate_knowledge_graph_embedding` |
| `benchmark_cross_spring_evolution` | 162 | 66 | `cargo run --features gpu --bin benchmark_cross_spring_evolution` |
| `validate_barracuda_cpu_v9` | 163 | 66 | `cargo run --release --bin validate_barracuda_cpu_v9` |
| `validate_gpu_drug_repurposing` | 164 | 48 | `cargo run --features gpu --bin validate_gpu_drug_repurposing` |
| `validate_metalforge_drug_repurposing` | 165 | 25 | `cargo run --features gpu --bin validate_metalforge_drug_repurposing` |
| `benchmark_modern_systems_df64` | 166 | 19 | `cargo run --features gpu --bin benchmark_modern_systems_df64` |
| `validate_gpu_diversity_fusion` | 167 | 18 | `cargo run --features gpu --bin validate_gpu_diversity_fusion` |
| `validate_cross_spring_s62` | 168 | ~25 | `cargo run --features gpu --release --bin validate_cross_spring_s62` |
| `benchmark_cross_spring_modern` | 169 | 12 | `cargo run --release --features gpu --bin benchmark_cross_spring_modern` |
| `benchmark_cross_spring_s65` | 183 | 36 | `cargo run --release --features gpu --bin benchmark_cross_spring_s65` |
| `benchmark_cross_spring_s68` | 189 | varies | `cargo run --release --features gpu --bin benchmark_cross_spring_s68` |
| `validate_barracuda_cpu_v10` | 190 | 75 | `cargo run --release --bin validate_barracuda_cpu_v10` |
| `validate_gpu_v59_science` | 191 | 29 | `cargo run --features gpu --release --bin validate_gpu_v59_science` |
| `validate_metalforge_v59_science` | 192 | 36 | `cargo run --features gpu --release --bin validate_metalforge_v59_science` |
| `validate_soil_qs_pore_geometry` | 170 | 26 | `cargo run --release --bin validate_soil_qs_pore_geometry` |
| `validate_soil_pore_diversity` | 171 | 27 | `cargo run --release --bin validate_soil_pore_diversity` |
| `validate_soil_distance_colonization` | 172 | 23 | `cargo run --release --bin validate_soil_distance_colonization` |
| `validate_notill_brandt_farm` | 173 | 14 | `cargo run --release --bin validate_notill_brandt_farm` |
| `validate_notill_meta_analysis` | 174 | 20 | `cargo run --release --bin validate_notill_meta_analysis` |
| `validate_notill_longterm_tillage` | 175 | 19 | `cargo run --release --bin validate_notill_longterm_tillage` |
| `validate_soil_biofilm_aggregate` | 176 | 23 | `cargo run --release --bin validate_soil_biofilm_aggregate` |
| `validate_soil_structure_function` | 177 | 16 | `cargo run --release --bin validate_soil_structure_function` |
| `validate_tillage_microbiome_2025` | 178 | 15 | `cargo run --release --bin validate_tillage_microbiome_2025` |
| `validate_soil_qs_cpu_parity` | 179 | 49 | `cargo run --release --bin validate_soil_qs_cpu_parity` |
| `validate_soil_qs_gpu` | 180 | 23 | `cargo run --features gpu --release --bin validate_soil_qs_gpu` |
| `validate_soil_qs_streaming` | 181 | 52 | `cargo run --features gpu --release --bin validate_soil_qs_streaming` |
| `validate_soil_qs_metalforge` | 182 | 14 | `cargo run --features gpu --release --bin validate_soil_qs_metalforge` |
| `validate_nanopore_signal_bridge` | 196a | 28 | `cargo run --release --bin validate_nanopore_signal_bridge` |
| `validate_nanopore_simulated_16s` | 196b | 11 | `cargo run --release --bin validate_nanopore_simulated_16s` |
| `validate_nanopore_int8_quantization` | 196c | 13 | `cargo run --release --bin validate_nanopore_int8_quantization` |
| `validate_science_pipeline` | 203 | 29 | `cargo run --features ipc --bin validate_science_pipeline` |
| `wetspring_server` | 203 | — | `cargo run --features ipc --bin wetspring_server` (biomeOS primal) |
| `validate_barracuda_cpu_v11` | 206 | 64 | `cargo run --features ipc --release --bin validate_barracuda_cpu_v11` |
| `validate_barracuda_gpu_v4` | 207 | 54 | `cargo run --features gpu,ipc --release --bin validate_barracuda_gpu_v4` |
| `validate_metalforge_v7_mixed` | 208 | 74 | `cargo run --features ipc --release --bin validate_metalforge_v7_mixed` |
| `benchmark_cross_spring_modern_s68plus` | 210 | 24 | `cargo run --features gpu --release --bin benchmark_cross_spring_modern_s68plus` |
| `benchmark_progression_cpu_gpu_stream` | 211 | 16 | `cargo run --features gpu --release --bin benchmark_progression_cpu_gpu_stream` |

**Total validation checks**: 5,061+
**Rust tests**: 1,103 (933 barracuda lib + 44 IPC + 60 integration + 19 doc + 47 forge)
**Binaries**: 178 validate + 17 benchmark + 1 server = 196 total
**ToadStool primitives**: 79 consumed (barracuda always-on, zero fallback code — S68+ `e96576ee`)
**Papers**: 52 (25 Tracks 1-2 + 5 Track 3 + 9 Track 4 + 1 cross-spring + 9 extensions + 3 reference)
**Local WGSL shaders**: 0 (all absorbed by ToadStool S63)
**GPU modules**: 42 total (all lean on upstream primitives)
**Benchmark infrastructure**: `bench.rs` harness with RAPL + nvidia-smi energy profiling, JSON output

---

## How to Add a New Experiment

1. Create `experiments/NNN_descriptive_name.md` with: date, status, objective,
   baseline tool, dataset, protocol, acceptance criteria.
2. Run the baseline tool and save results to `experiments/results/NNN_name/`.
3. Create `src/bin/validate_NNN.rs` with provenance table and `Validator` checks.
4. Add the `[[bin]]` entry to `Cargo.toml`.
5. Update this README with the new experiment row.
