# wetSpring Control Experiment Status

**Date:** February 25, 2026
**Status:** Phase 45 — 168 experiments, 3,300+ validation checks, all PASS (759 barracuda + 47 forge = 806 Rust tests), ToadStool S62+DF64 aligned, 49 primitives + 2 BGL helpers + 1 WGSL extension (barracuda always-on), 70 named tolerance constants, 0 ad-hoc tolerances, 7/9 P0-P3 delivered, 0 Passthrough, V40 catch-up complete

---

## Experiment Status

| Exp | Name | Track | Status | Checks |
|-----|------|-------|--------|--------|
| 001 | Galaxy Bootstrap (QIIME2/DADA2) | 1 | COMPLETE | 28 |
| 002 | Phytoplankton 16S (PRJNA1195978) | 1 | COMPLETE | — |
| 003 | Phage Assembly (SPAdes/Pharokka) | 1 | COMPLETE | — |
| 004 | Rust FASTQ + Diversity | 1 | COMPLETE | 55 |
| 005 | asari LC-MS Bootstrap | 2 | COMPLETE | 7 |
| 006 | PFAScreen Validation (FindPFAS) | 2 | COMPLETE | 10 |
| 007 | Rust mzML + PFAS | 2 | COMPLETE | — |
| 008 | PFAS ML Water Monitoring | 2 | COMPLETE (Phase 3) | 7 |
| 009 | Feature Pipeline (asari MT02) | 2 | COMPLETE | 8 |
| 010 | Peak Detection (scipy baseline) | cross | COMPLETE | 17 |
| 011 | 16S Pipeline End-to-End | 1 | COMPLETE | 37 |
| 012 | Algae Pond 16S (PRJNA488170) | 1 | COMPLETE | 34 |
| 013 | VOC Peak Validation (Reese 2019) | 1/cross | COMPLETE | 22 |
| 014 | Public Data Benchmarks (4 BioProjects) | 1 | COMPLETE | 202 |
| 015 | Pipeline Benchmark (Rust vs Galaxy) | 1 | COMPLETE | Benchmark |
| 016 | GPU Pipeline Parity (CPU↔GPU) | 1 | COMPLETE | 88 |
| 017 | Extended Algae (PRJNA382322) | 1 | COMPLETE | 35 |
| 018 | PFAS Library (Jones Lab Zenodo) | 2 | COMPLETE | 26 |
| 019 | Phylogenetic Validation (Phase 1) | 1b | COMPLETE | 30 |
| 020 | Waters 2008 QS/c-di-GMP ODE | 1 | COMPLETE | 16 |
| 021 | Robinson-Foulds Validation | 1b | COMPLETE | 23 |
| 022 | Massie 2012 Gillespie SSA | 1 | COMPLETE | 13 |
| 023 | Fernandez 2020 Bistable Switching | 1 | COMPLETE | 14 |
| 024 | Srivastava 2011 Multi-Signal QS | 1 | COMPLETE | 19 |
| 025 | Bruger & Waters 2018 Cooperation | 1 | COMPLETE | 20 |
| 026 | Liu 2014 HMM Primitives | 1b | COMPLETE | 21 |
| 027 | Mhatre 2020 Phenotypic Capacitor | 1 | COMPLETE | 18 |
| 028 | Smith-Waterman Alignment | 1b | COMPLETE | 15 |
| 029 | Felsenstein Pruning Likelihood | 1b/c | COMPLETE | 16 |
| 030 | Hsueh 2022 Phage Defense Deaminase | 1 | COMPLETE | 12 |
| 031 | Wang 2021 RAWR Bootstrap | 1b | COMPLETE | 11 |
| 032 | Alamin & Liu 2024 Placement | 1b | COMPLETE | 12 |
| 033 | Liu 2009 Neighbor-Joining (SATé core) | 1b | COMPLETE | 16 |
| 034 | Zheng 2023 DTL Reconciliation | 1b | COMPLETE | 14 |
| 035 | BarraCuda CPU Parity v2 | cross | COMPLETE | 18 |
| 036 | PhyNetPy RF distances (Exp019 Phase 2) | 1b | COMPLETE | 15 |
| 037 | PhyloNet-HMM discordance (Exp019 Phase 3) | 1b | COMPLETE | 10 |
| 038 | SATe pipeline benchmark (Exp019 Phase 4) | 1b | COMPLETE | 17 |
| 039 | Algal pond time-series (Cahill proxy) | 1 | COMPLETE | 11 |
| 040 | Bloom surveillance (Smallwood proxy) | 1 | COMPLETE | 15 |
| 041 | EPA PFAS ML (Jones F&T proxy) | 2 | COMPLETE | 14 |
| 042 | MassBank spectral (Jones MS proxy) | 2 | COMPLETE | 9 |
| 043 | BarraCuda CPU Parity v3 | cross | COMPLETE | 45 |
| 044 | BarraCuda GPU v3 | cross | COMPLETE | 14 |
| 045 | ToadStool Bio Absorption | cross/GPU | COMPLETE | 10 |
| 046 | GPU Phylogenetic Composition | GPU | COMPLETE | 15 |
| 047 | GPU HMM Batch Forward | GPU | COMPLETE | 13 |
| 048 | CPU vs GPU Benchmark (Phylo + HMM) | GPU | COMPLETE | 6 |
| 049 | GPU ODE Parameter Sweep | GPU | COMPLETE | 7 |
| 050 | GPU Bifurcation Eigenvalue Analysis | GPU | COMPLETE | 5 |
| 051 | Anderson 2015 Rare Biosphere | 1c | COMPLETE | 35 |
| 052 | Anderson 2014 Viral Metagenomics | 1c | COMPLETE | 22 |
| 053 | Mateos 2023 Sulfur Phylogenomics | 1c | COMPLETE | 15 |
| 054 | Boden 2024 Phosphorus Phylogenomics | 1c | COMPLETE | 13 |
| 055 | Anderson 2017 Population Genomics | 1c | COMPLETE | 24 |
| 056 | Moulana 2020 Pangenomics | 1c | COMPLETE | 24 |
| 057 | BarraCuda CPU Parity v4 (Track 1c) | cross | COMPLETE | 44 |
| 058 | GPU Track 1c (ANI + SNP + Pangenome + dN/dS) | GPU | COMPLETE | 27 |
| 059 | 25-Domain Rust vs Python Benchmark | cross | COMPLETE | Benchmark |
| 060 | metalForge Cross-Substrate Validation | cross/GPU | COMPLETE | 20 |
| 061 | Random Forest Ensemble Inference | cross | COMPLETE | 13 |
| 062 | GBM Inference (Binary + Multi-Class) | cross | COMPLETE | 16 |
| 063 | GPU Random Forest Batch Inference | GPU | COMPLETE | 13 |
| 064 | BarraCuda GPU Parity v1 (all GPU domains) | cross/GPU | COMPLETE | 26 |
| 065 | metalForge Full Cross-System Validation | cross/GPU | COMPLETE | 35 |
| 066 | CPU vs GPU Scaling Benchmark (all GPU domains) | GPU | COMPLETE | Benchmark |
| 067 | ToadStool Dispatch Overhead Profiling | GPU | COMPLETE | Benchmark |
| 068 | Pipeline Caching Optimization | GPU | COMPLETE | Optimization/Benchmark |
| 069 | Python → Rust CPU → GPU Three-Tier Benchmark | cross | COMPLETE | Benchmark |
| 070 | BarraCuda CPU 25-Domain Pure Rust Math Proof | cross | COMPLETE | 50 |
| 071 | BarraCuda GPU Math Portability Proof | GPU | COMPLETE | 24 |
| 072 | GPU Streaming Pipeline Proof | GPU | COMPLETE | 17 |
| 073 | Compute Dispatch Overhead Proof | GPU | COMPLETE | 21 |
| 074 | metalForge Substrate Router | cross/GPU | COMPLETE | 20 |
| 075 | Pure GPU Analytics Pipeline | GPU | COMPLETE | 31 |
| 076 | Cross-Substrate Pipeline | cross/GPU | COMPLETE | 17 |
| 077 | ToadStool Bio Rewire | GPU/cross | COMPLETE | 451 (re-validated) |
| 078 | ODE GPU Sweep Readiness | cross/GPU | COMPLETE | 10 (round-trip + parity) |
| 079 | BarraCuda CPU v6 — ODE Flat Param | CPU/cross | COMPLETE | 48 (flat RT + ODE + Python) |
| 080 | metalForge Dispatch Routing | cross/dispatch | COMPLETE | 35 (7 sections × 5 configs) |
| 081 | K-mer GPU Histogram Prep | GPU/refactor | COMPLETE | 4 (round-trip + GPU sizing) |
| 082 | UniFrac Flat Tree (CSR) | GPU/refactor | COMPLETE | 4 (CSR + parity + matrix) |
| 083 | Taxonomy NPU Quantization | NPU/refactor | COMPLETE | 3 (int8 RT + parity + sizing) |
| 084 | metalForge Full Cross-Substrate v2 | metalForge | COMPLETE | 35+ (12 domains CPU ↔ GPU) |
| 085 | BarraCuda CPU v7 — Tier A Layouts | CPU/layout | COMPLETE | 43 (kmer/unifrac/taxonomy flat) |
| 086 | metalForge Pipeline Proof | metalForge | COMPLETE | 45 (5-stage dispatch + parity) |
| 087 | GPU Extended Domains (EIC/PCoA/Kriging/Rarefaction) | GPU | COMPLETE | 50+ (4 new GPU domains) |
| 088 | metalForge PCIe Direct Transfer | metalForge | COMPLETE | 32 (6 paths + buffer contracts) |
| 089 | ToadStool Streaming Dispatch | streaming | COMPLETE | 25 (5 patterns + determinism) |
| 090 | Pure GPU Streaming Pipeline | GPU/streaming | COMPLETE | 80 (4 modes: RT, stream, parity, scaling) |
| 091 | Streaming vs Round-Trip Benchmark | GPU/benchmark | COMPLETE | 2 (parity + Bray-Curtis error) |
| 092 | CPU vs GPU All 16 Domains | GPU/parity | COMPLETE | 48 (16 domains head-to-head) |
| 093 | metalForge Full v3 (16 domains) | metalForge | COMPLETE | 28 (16 domains substrate-independent) |
| 094 | Cross-Spring Evolution Validation | GPU/parity | COMPLETE | 39 (5 neuralSpring primitives CPU↔GPU) |
| 095 | Cross-Spring Scaling Benchmark | GPU/benchmark | COMPLETE | 7 (scaling across 3 Springs) |
| 096 | ToadStool Bio Op Absorption Validation | GPU/shader | COMPLETE | 10 (4 upstream ops validated) |
| 097 | Structural Evolution Pass | code quality | COMPLETE | — (22-file refactor: flat layouts, DRY models, zero-clone APIs) |
| 098 | Upstream GPU Fixes | GPU/shader | COMPLETE | — (3 ToadStool bugs: SNP BGL, ODE f64, Jacobi eigenvectors) |
| 099 | CPU vs GPU Expanded + metalForge | GPU/metalForge | COMPLETE | 27 (k-mer, UniFrac, ODE, phage defense, mixed-HW pipeline) |
| 100 | metalForge v4 ODE Domains + NPU | GPU/metalForge | COMPLETE | 28 (bistable, multi-signal, phage defense, NPU routing, PCIe pipeline) |
| 101 | Pure GPU Promotion Complete (13 modules) | GPU/parity | COMPLETE | 38 (13 modules CPU↔GPU, 2 new WGSL shaders) |
| 102 | BarraCuda CPU v8 — Pure GPU Domains | CPU/cross | COMPLETE | 84 (13 GPU-promoted domains, known-value validation) |
| 103 | metalForge Cross-Substrate v5 | metalForge | COMPLETE | 38 (13 new GPU domains, substrate-independent) |
| 104 | metalForge Cross-Substrate v6 | metalForge | COMPLETE | 24 (5 gap domains: QS ODE, UniFrac, DADA2, K-mer, Felsenstein — **25/25 papers three-tier**) |
| 105 | Pure GPU Streaming v2 — Analytics | streaming | COMPLETE | 27 (alpha div + Bray-Curtis + spectral cosine + full pipeline) |
| 106 | GPU Streaming — ODE + Phylogenetics | streaming | COMPLETE | 45 (6 pre-warmed primitives: ODE sweep, phage, bistable, multi-signal, Felsenstein, UniFrac) |
| 107 | Spectral Cross-Spring (Anderson/QS) | cross | COMPLETE | 25 (Anderson 1D/2D/3D, Almost-Mathieu, Lanczos, QS-disorder analogy) |
| 108 | Vibrio QS parameter landscape | `validate_vibrio_qs_landscape` | PASS | 8 |
| 109 | Large-scale phylo placement | `validate_phylo_placement_scale` | PASS | 11 |
| 110 | Cross-ecosystem pangenome | `validate_cross_ecosystem_pangenome` | PASS | 17 |
| 111 | MassBank GPU spectral scale | `validate_massbank_gpu_scale` | PASS | 14 |
| 112 | Real-bloom GPU surveillance | `validate_real_bloom_gpu` | PASS | 23 |
| 113 | QS-disorder from real diversity | `validate_qs_disorder_real` | PASS | 5 |
| 114 | NPU QS phase classifier | `validate_npu_qs_classifier` | PASS | 13 |
| 115 | NPU phylogenetic placement | `validate_npu_phylo_placement` | PASS | 9 |
| 116 | NPU genome binning | `validate_npu_genome_binning` | PASS | 9 |
| 117 | NPU spectral screening | `validate_npu_spectral_screen` | PASS | 8 |
| 118 | NPU bloom sentinel | `validate_npu_bloom_sentinel` | PASS | 11 |
| 119 | NPU QS-disorder classifier | `validate_npu_disorder_classifier` | PASS | 9 |
| 120 | Cross-spring evolution benchmark | `benchmark_cross_spring_evolution` | PASS | 9 |
| 121 | NCBI Vibrio QS landscape | `validate_ncbi_vibrio_qs` | PASS (GPU) | 14 |
| 122 | 2D Anderson spatial QS | `validate_anderson_2d_qs` | PASS (GPU) | 12 |
| 123 | Temporal ESN bloom cascade | `validate_temporal_esn_bloom` | PASS | 9 |
| 124 | NPU spectral triage | `validate_npu_spectral_triage` | PASS | 10 |
| 125 | NCBI Campylobacterota pangenome | `validate_ncbi_pangenome` | PASS | 11 |
| 126 | Global QS-disorder atlas | `validate_ncbi_qs_atlas` | PASS (GPU) | 90 |
| 127 | 3D Anderson dimensional QS sweep | `validate_anderson_3d_qs` | PASS (GPU) | 17 |
| 128 | Vent chimney geometry QS | `validate_vent_chimney_qs` | PASS (GPU) | 12 |
| 129 | Dimensional QS phase diagram | `validate_dimensional_phase_diagram` | PASS (GPU) | 12 |
| 130 | Thick biofilm 3D QS extension | `validate_biofilm_3d_qs` | PASS (GPU) | 9 |
| 131 | Finite-size scaling (L=6→10) | `validate_finite_size_scaling` | PASS (GPU) | 11 |
| 132 | Geometry zoo (6 shapes) | `validate_geometry_zoo` | PASS (GPU) | 11 |
| 133 | Cave/hot spring/rhizosphere QS | `validate_ecosystem_geometry_qs` | PASS (GPU) | 17 |
| 134 | Cross-ecosystem QS atlas (28×5) | `validate_cross_ecosystem_atlas` | PASS (GPU) | 11 |
| 135 | Mapping sensitivity (9 α values) | `validate_mapping_sensitivity` | PASS (GPU) | 8 |
| 136 | Square-cubed law scaling | `validate_square_cubed_scaling` | PASS (GPU) | 6 |
| 137 | Planktonic/fluid 3D dilution | `validate_planktonic_dilution` | PASS (GPU) | 10 |
| 138 | Eukaryote vs bacteria scaling | `validate_eukaryote_scaling` | PASS (GPU) | 11 |
| 139 | QS distance scaling (bacteria vs humans) | `validate_qs_distance_scaling` | PASS (GPU) | 6 |
| 140 | QS gene prevalence by habitat geometry | `validate_qs_gene_prevalence` | PASS | 7 |
| 141 | NCBI QS habitat query (live data) | `validate_ncbi_qs_habitat` | PASS (NCBI) | 6 |
| 142 | Producer vs receiver QS (NCBI live) | `validate_producer_receiver_qs` | PASS (NCBI) | 8 |
| 143 | Anderson anomaly hunter (NP solutions) | `validate_anderson_anomalies` | PASS | 5 |
| 144 | Cold seep QS gene catalog (299K genes) | `validate_cold_seep_qs_catalog` | PASS | 8 |
| 145 | Cold seep QS type vs geometry | `validate_cold_seep_qs_geometry` | PASS | 5 |
| 146 | luxR phylogeny × geometry overlay | `validate_luxr_phylogeny_geometry` | PASS | 5 |
| 147 | Mechanical wave Anderson framework | `validate_mechanical_wave_anderson` | PASS | 6 |
| 148 | QS wave × localization (PRE 2020) | `validate_qs_wave_localization` | PASS | 6 |
| 149 | Burst statistics as Anderson (SciRep 2019) | `validate_burst_statistics_anderson` | PASS | 6 |
| 150 | Finite-size scaling v2 (disorder-averaged) | `validate_finite_size_scaling_v2` | PASS (GPU) | 14 |
| 151 | Correlated disorder lattices (biofilm) | `validate_correlated_disorder` | PASS (GPU) | 8 |
| 152 | Physical comm pathways vs Anderson (8 modes) | `validate_physical_comm_anderson` | PASS | 9 |
| 153 | Nitrifying community QS (npj Biofilms 2021) | `validate_nitrifying_qs` | PASS | 12 |
| 154 | Marine interkingdom QS (10 organisms) | `validate_marine_interkingdom_qs` | PASS | 6 |
| 155 | Myxococcus C-signal critical density (PNAS) | `validate_myxococcus_critical_density` | PASS | 7 |
| 156 | Dictyostelium cAMP relay (non-Hermitian) | `validate_dictyostelium_relay` | PASS | 8 |
| 157 | Fajgenbaum pathway scoring (JCI 2019) | `validate_fajgenbaum_pathway` | PASS | 8 |
| 158 | MATRIX pharmacophenomics (Lancet 2025) | `validate_matrix_pharmacophenomics` | PASS | 9 |
| 159 | NMF drug-disease factorization (Yang 2020) | `validate_nmf_drug_repurposing` | PASS | 7 |
| 160 | repoDB NMF reproduction (Gao 2020) | `validate_repodb_nmf` | PASS | 9 |
| 161 | Knowledge graph embedding (ROBOKOP) | `validate_knowledge_graph_embedding` | PASS | 7 |
| 162 | Cross-spring evolution (S54-S62) | `validate_cross_spring_s57` | PASS | 66 |
| 163 | BarraCuda CPU v9 — Track 3 drug repurposing | `validate_barracuda_cpu_v9` | PASS | 27 |
| 164 | GPU drug repurposing — GEMM NMF, TransE, PeakDetect | `validate_gpu_drug_repurposing` | PASS | 8 |
| 165 | metalForge drug repurposing — CPU↔GPU parity | `validate_metalforge_drug_repurposing` | PASS | 9 |
| 166 | Modern systems benchmark (S62+DF64) | `benchmark_modern_systems_df64` | PASS | 19 |
| 167 | Diversity fusion GPU extension (Write phase) | `validate_gpu_diversity_fusion` | PASS | 18 |

---

## Totals

| Category | Count |
|----------|-------|
| Experiments completed | 165 |
| CPU validation checks | 1,476 |
| GPU validation checks | 702 |
| Dispatch validation checks | 80 |
| Layout fidelity checks | 35 |
| Transfer/streaming checks | 57 |
| Cross-spring checks | 39 |
| Local WGSL checks | 10 |
| Pure GPU promotion checks | 38 |
| metalForge v6 three-tier checks | 24 |
| Pure GPU streaming v2 checks | 27 |
| Streaming ODE + phylogenetics checks | 45 |
| Cross-spring spectral theory checks | 25 |
| NPU reservoir deployment checks | 59 |
| Cross-spring evolution checks | 9 |
| NCBI-scale + 2D Anderson + temporal ESN checks | 146 |
| 3D Anderson dimensional QS checks | 50 |
| Geometry verification + cross-ecosystem checks | 50 |
| Why analysis: mapping, scaling, dilution, eukaryote checks | 35 |
| Empirical validation: QS distance, gene prevalence, NCBI habitat query | 19 |
| Anderson as null hypothesis: producer/receiver, anomalies | 13 |
| Extension papers: cold seep, phylogeny, waves (Exp144-149) | 36 |
| Phase 39: finite-size, correlated, comm, nitrifying, marine, myxo, dicty (Exp150-156) | 66 |
| Drug repurposing: Fajgenbaum, MATRIX, NMF, repoDB, ROBOKOP (Exp157-161) | 40 |
| Drug repurposing: CPU v9, GPU, metalForge (Exp163-165) | 44 |
| Phase 44: modern systems S62+DF64, diversity fusion (Exp166-167) | 37 |
| **Total validation checks** | **3,300+** |
| Rust tests | 806 (759 barracuda + 47 forge) |
| BarraCuda CPU parity | 380/380 (25 domains + 6 ODE flat + 3 layout + 13 GPU-promoted) |
| BarraCuda GPU parity | 29 domains (Exp064/087/092/101) |
| metalForge cross-system | 37 domains CPU↔GPU proven (Exp103+104+165), **30/30 papers three-tier** |
| metalForge dispatch routing | 35 checks across 5 configs (Exp080) |
| ToadStool primitives consumed | 44 (barracuda always-on, zero fallback — S62) |
| ToadStool session alignment | S62 (660+ WGSL, cpu-math gate, PeakDetect, TransE, SpMM, NMF, ODE bio, ridge, Anderson) |
| Cross-spring shader provenance | 35+ hotSpring, 22+ wetSpring, 14+ neuralSpring, 5+ airSpring, 500+ native |

---

## Python Baseline Status

| Script | Tool | Date | Status |
|--------|------|------|--------|
| `benchmark_python_baseline.py` | QIIME2/DADA2-R | Feb 2026 | GREEN |
| `validate_public_16s_python.py` | BioPython + NCBI | Feb 2026 | GREEN |
| `waters2008_qs_ode.py` | scipy.integrate.odeint | Feb 2026 | GREEN (35/35) |
| `gillespie_baseline.py` | numpy SSA ensemble | Feb 2026 | GREEN (8/8) |
| `rf_distance_baseline.py` | dendropy RF distance | Feb 2026 | GREEN (10/10) |
| `newick_parse_baseline.py` | dendropy tree stats | Feb 2026 | GREEN (10/10) |
| `pfas_tree_export.py` | sklearn DecisionTree | Feb 2026 | GREEN (acc=0.989) |
| `exp008_pfas_ml_baseline.py` | sklearn RF+GBM | Feb 2026 | GREEN (RF F1=0.978) |
| `fernandez2020_bistable.py` | scipy ODE (bifurcation) | Feb 2026 | GREEN |
| `srivastava2011_multi_signal.py` | scipy ODE (multi-signal) | Feb 2026 | GREEN |
| `bruger2018_cooperation.py` | scipy ODE (game theory) | Feb 2026 | GREEN |
| `liu2014_hmm_baseline.py` | numpy HMM (sovereign) | Feb 2026 | GREEN |
| `mhatre2020_capacitor.py` | scipy ODE (capacitor) | Feb 2026 | GREEN |
| `smith_waterman_baseline.py` | pure Python (sovereign) | Feb 2026 | GREEN |
| `felsenstein_pruning_baseline.py` | pure Python (sovereign) | Feb 2026 | GREEN |
| `hsueh2022_phage_defense.py` | scipy ODE (phage defense) | Feb 2026 | GREEN |
| `wang2021_rawr_bootstrap.py` | pure Python (bootstrap) | Feb 2026 | GREEN |
| `alamin2024_placement.py` | pure Python (placement) | Feb 2026 | GREEN |
| `liu2009_neighbor_joining.py` | pure Python (NJ) | Feb 2026 | GREEN |
| `zheng2023_dtl_reconciliation.py` | pure Python (DTL) | Feb 2026 | GREEN |
| `phynetpy_rf_baseline.py` | PhyNetPy gene trees | Feb 2026 | GREEN |
| `phylohmm_introgression_baseline.py` | PhyloNet-HMM | Feb 2026 | GREEN |
| `sate_alignment_baseline.py` | SATe pipeline | Feb 2026 | GREEN |
| `algae_timeseries_baseline.py` | Cahill proxy | Feb 2026 | GREEN |
| `bloom_surveillance_baseline.py` | Smallwood proxy | Feb 2026 | GREEN |
| `epa_pfas_ml_baseline.py` | Jones F&T proxy | Feb 2026 | GREEN |
| `massbank_spectral_baseline.py` | Jones MS proxy | Feb 2026 | GREEN |
| `benchmark_rust_vs_python.py` | 18-domain timing (Exp043) | Feb 2026 | GREEN |
| `anderson2015_rare_biosphere.py` | diversity/rarefaction (Exp051) | Feb 2026 | GREEN |
| `anderson2014_viral_metagenomics.py` | dN/dS + diversity (Exp052) | Feb 2026 | GREEN |
| `mateos2023_sulfur_phylogenomics.py` | clock/reconciliation (Exp053) | Feb 2026 | GREEN |
| `boden2024_phosphorus_phylogenomics.py` | clock/reconciliation (Exp054) | Feb 2026 | GREEN |
| `anderson2017_population_genomics.py` | ANI/SNP (Exp055) | Feb 2026 | GREEN |
| `moulana2020_pangenomics.py` | pangenome/enrichment (Exp056) | Feb 2026 | GREEN |
| `barracuda_cpu_v4_baseline.py` | 5 Track 1c domain timing (Exp057) | Feb 2026 | GREEN |

---

## Remaining Work

### Deferred (not blocking)
- Exp002 raw data: 70 FASTQ pairs from SRA (Galaxy bootstrap, not needed for validation)
- Trimmomatic/pyteomics baselines: superseded by sovereign Rust implementations

### Completed
- Exp019 Phases 2-4 (Phylogenetic): All COMPLETE
- Exp008 Full ML Pipeline: All COMPLETE
- Tolerance centralization: **DONE** — 50 named constants in `tolerances.rs` (ODE dt, Galaxy ranges, bootstrap LL, phage population)
- Ad-hoc tolerance elimination: **DONE** — all validation binaries use `tolerances::*` constants
- Code quality hardening: **DONE** — `deny(unsafe_code)`, `deny(expect_used, unwrap_used)`, pedantic + nursery clippy (0 warnings, both default and GPU features)
- Blanket lint tightening: **DONE** — removed `similar_names` blanket allow; targeted per-function `#[allow]` for domain-appropriate names
- Unsafe code evolution: **DONE** — env var tests use `Mutex`-serialized helpers with centralized `set_env`/`remove_env`
- Rust edition 2024: **DONE** — MSRV 1.85, `f64::midpoint()`, `usize::midpoint()`, `const fn` promotions
- metalForge forge crate: **DONE** — `wetspring-forge` (24 tests, substrate discovery + dispatch)
- GPU workgroup constants: **DONE** — all GPU modules use named `WORKGROUP_SIZE` matching ToadStool shaders
- Hardware abstraction: **DONE** — `HardwareInventory::from_content()`, injectable `/proc` parsing
- I/O streaming: **DONE** — zero-copy FASTQ (`FastqRefRecord`), mzML buffer reuse (`DecodeBuffer`)
- Determinism tests: **DONE** — 16 bitwise-exact tests across non-stochastic algorithms
- Fuzz testing: **DONE** — 4 harnesses (FASTQ, mzML, MS2, XML) via cargo-fuzz
- Doc strictness: **DONE** — `-D missing_docs -D rustdoc::broken_intra_doc_links` pass
- Math extraction: **DONE** — `bio::special` → `crate::special` (top-level, re-export shim removed)
- Math delegation: **DONE** — `crate::special::{erf, ln_gamma, regularized_gamma_lower}` delegate to `barracuda::special` when `gpu` feature active; no duplicate math
- wgpu feature tightening: **DONE** — `default-features = false, features = ["wgsl", "vulkan-portability"]`; no `renderdoc-sys` (C dep)
- Validation binary refactoring: **DONE** — `validate_cpu_vs_gpu_all_domains.rs` refactored from monolithic main() to 16 domain-specific helper functions
- Hardcoded path evolution: **DONE** — NPU device, PCI slots, benchmark output dirs all configurable via env vars (`WETSPRING_NPU_DEVICE`, `WETSPRING_GPU_PCI_SLOTS`, `WETSPRING_BENCHMARK_DIR`, `WETSPRING_PYTHON_BASELINE`)
- Negative-case tests: **DONE** — 8 new tests for malformed/missing/truncated input (FASTQ 5, mzML 3)
- Provenance manifest: **DONE** — `scripts/BASELINE_MANIFEST.md` maps 40 Python scripts to Rust binaries with SHA-256 hashes
- XML parser zero-copy: **DONE** — `xml_unescape` returns `Cow<str>`, text event path avoids double allocation
- Doc examples: **DONE** — `dada2::denoise`, `merge_pairs::merge_pair`, `parse_mzml`, `bray_curtis`, `smith_waterman` (19 doc tests total)
- Binary refactoring: **DONE** — `metalforge_full_v2` (12 helpers), `metalforge_full_v3` (16 helpers); 8 stale `too_many_lines` allows removed
- Absorption batch APIs: **DONE** — `snp::call_snps_batch`, `quality::filter_reads_flat`, `pangenome::analyze_batch`

---

## Track Coverage

### Track 1: Microbial Ecology (16S rRNA)
**Status:** Comprehensive. 9 experiments (001, 004, 011, 012, 014, 017, 020, 039, 040)
cover the full 16S pipeline from FASTQ to diversity metrics, validated against
QIIME2, DADA2, and 4 BioProjects with 22 samples.

### Track 1b: Comparative Genomics & Phylogenetics
**Status:** Comprehensive. 12 experiments covering the full phylogenetic
pipeline: Newick parsing (Exp019), Robinson-Foulds (Exp021), HMM (Exp026),
Smith-Waterman (Exp028), Felsenstein pruning (Exp029), bootstrap (Exp031),
placement (Exp032), Neighbor-Joining tree construction (Exp033), DTL
reconciliation (Exp034), PhyNetPy RF (Exp036), PhyloNet-HMM (Exp037), and
SATe pipeline (Exp038). This provides a complete
toolkit from sequence alignment through tree construction, evaluation,
statistical confidence, and cophylogenetic analysis.

### Track 1c: Deep-Sea Metagenomics (Anderson)
**Status:** Comprehensive. 6 experiments (051-056) plus CPU parity (Exp057).
Covers ANI, SNP calling, dN/dS, molecular clock, pangenome analysis,
phylogenomics, and rare biosphere diversity — all validated against Python
baselines and proved as pure Rust math via BarraCuda CPU v4.

### Track 2: Analytical Chemistry (LC-MS, PFAS)
**Status:** Comprehensive. 7 experiments (005, 006, 009, 013, 018, 041, 042) cover
mzML parsing, feature extraction, peak detection, PFAS screening, and library
matching. Exp008 adds sovereign ML for environmental monitoring.

---

## Code Quality (Feb 24, 2026)

```
cargo fmt --check              → clean (0 diffs, both crates)
cargo clippy --pedantic        → 0 warnings (pedantic + nursery, default features)
cargo clippy --features gpu    → 0 warnings (pedantic + nursery, GPU features)
cargo doc --features gpu       → clean (0 warnings, strict: -D missing_docs -D broken_intra_doc_links)
cargo test --lib               → 755 passed, 0 failed, 1 ignored (hardware-dependent)
cargo test --tests             → 60 integration (23 bio + 16 determinism + 21 I/O)
cargo test --doc               → 19 passed, 0 failed (5 API examples)
cargo llvm-cov --lib           → 95.75% line coverage
#![deny(unsafe_code)]          → enforced crate-wide (edition 2024; env-var tests use Mutex-serialized helpers)
#![deny(expect_used, unwrap_used)] → enforced crate-wide (test modules #[allow])
partial_cmp().unwrap()         → 0 (all migrated to f64::total_cmp)
inline tolerance literals      → 0 (70 named constants in tolerances.rs; V39 added 8)
blanket similar_names          → removed; targeted #[allow] per-function where domain-appropriate
GPU workgroup sizes            → named constants in all *_gpu.rs (match WGSL shaders)
shared math (crate::special)   → delegates to barracuda::special when gpu active; sovereign otherwise
hardware detection             → injectable (from_content / parse_*), no direct /proc in library
SPDX headers                   → all .rs files
max file size                  → all under 1000 LOC (fastq.rs: 913 largest)
external C dependencies        → 0 (flate2 rust_backend; wgpu default-features = false)
XML parser allocations         → Cow<str> for xml_unescape; 1 allocation per text event (was 2)
provenance headers             → all 158 binaries (commit, command, hardware)
duplicate math                 → 0 (crate::special delegates to ToadStool barracuda::special when gpu enabled)
Python baselines               → scripts/requirements.txt (pinned numpy, scipy, sklearn)
barracuda_cpu                  → 380/380 checks PASS (25 domains + 6 ODE flat + 3 layout + 13 GPU-promoted)
barracuda_gpu                  → 702 GPU checks PASS (770 with --features gpu, 9 ignored)
fuzz harnesses                 → 4 (FASTQ, mzML, MS2, XML)
zero-copy I/O                  → FastqRefRecord, DecodeBuffer reuse, streaming iterators
ToadStool alignment            → S62+DF64 (49 primitives, barracuda always-on, zero fallback code)
deprecated APIs                → 0 (parse_fastq → FastqIter::open in all binaries)
```

## BarraCuda CPU Parity

The `validate_barracuda_cpu` v1-v8 binaries prove pure Rust math matches
Python across all algorithmic domains:
- v1 (Exp035): 9 core domains
- v2 (Exp035): +5 batch/flat APIs
- v3 (Exp043): +9 domains (QS, phage, bootstrap, placement, decision tree, spectral, diversity, k-mer, pipeline)
- v4 (Exp057): +5 Track 1c domains (ANI, SNP, dN/dS, molecular clock, pangenome)
- v5 (Exp061/062): +2 ML domains (Random Forest, GBM)
- v6 (Exp079): +6 ODE flat parameter round-trip
- v7 (Exp085): +3 Tier A layout fidelity (kmer, unifrac, taxonomy)
- v8 (Exp102): +13 GPU-promoted domains (cooperation, capacitor, kmd, gbm, merge_pairs, signal, feature_table, robinson_foulds, derep, chimera, neighbor_joining, reconciliation, molecular_clock)

Combined: 380/380 CPU parity checks. This is the bridge to pure GPU execution.

```
Total CPU time: ~85ms (release build, all domains)
```

## BarraCuda GPU Parity

Exp064 + Exp101 consolidate ALL GPU-eligible domains into validation binaries,
proving pure GPU math matches CPU reference truth across the full portfolio:

- Diversity (Shannon, Simpson, Bray-Curtis) — via `FusedMapReduceF64`
- ANI — via `barracuda::ops::bio::ani::AniBatchF64` (ToadStool)
- SNP — via `barracuda::ops::bio::snp::SnpCallingF64` (ToadStool)
- dN/dS — via `barracuda::ops::bio::dnds::DnDsBatchF64` (ToadStool)
- Pangenome — via `barracuda::ops::bio::pangenome::PangenomeClassifyGpu` (ToadStool)
- Random Forest — via `barracuda::ops::bio::rf_inference::RfBatchInferenceGpu` (ToadStool)
- HMM forward — via `barracuda::ops::bio::hmm::HmmBatchForwardF64` (ToadStool)
- **13 new GPU domains** (Exp101): cooperation, capacitor, kmd, gbm, merge_pairs, signal, feature_table, robinson_foulds, derep, chimera, neighbor_joining, reconciliation, molecular_clock

29 GPU domains total. All bio primitives now flow through ToadStool's
absorbed or composed shaders, benefiting from cross-spring precision evolution.

## metalForge Cross-System Proof

Exp065 + Exp103 + Exp104 extends to ALL 37 metalForge domains, proving
substrate-independence: for every GPU-eligible algorithm, the metalForge
router can dispatch to CPU or GPU and get the same answer. This is the
foundation for CPU/GPU/NPU routing in production.

## ToadStool Evolution (Feb 24, 2026 — S62 Aligned)

### Write → Absorb → Lean Status

Following hotSpring's pattern for ToadStool integration:

| Phase | Count | Status |
|-------|:-----:|--------|
| **Lean** (consumed upstream) | 49 primitives (always-on, zero fallback code) | S62+DF64: PeakDetectF64, TranseScoreF64, ComputeDispatch, SparseGemmF64, TopK added to 44 prior |
| **Write** (local WGSL, pending absorption) | **0** — all retired | ODE shaders use `generate_shader()`; local WGSL deleted |
| **CPU math** (`crate::special`) | 3 functions delegating on GPU | `erf`, `ln_gamma`, `regularized_gamma_lower` → `barracuda::special::*` when `gpu` active; sovereign fallback for no-GPU |
| **CPU-only** (no GPU path) | 1 module (phred) | Pure GPU promotion complete (Exp101) |
| **Removed** (leaning upstream) | NMF (482 lines), ODE systems (715 lines), Anderson builder (~115 lines), ridge (~100 lines) | ~1,312 lines deleted in V30 lean |
| **metalForge** (absorption eng.) | 56 tolerances, SoA patterns, `#[repr(C)]` | Shaping all modules for ToadStool absorption |

### Feb 22 Rewire: 8 Bio Primitives Absorbed

ToadStool sessions 31d/31g absorbed all 8 wetSpring bio WGSL shaders. On Feb 22,
wetSpring rewired all 8 GPU modules to delegate to `barracuda::ops::bio::*`,
deleted the local shaders (25 KB), and verified 633 tests pass with 0 clippy
warnings. Two ToadStool bugs found and fixed during validation:

1. **SNP binding layout** — `is_variant` (binding 2) was declared `read_only` but
   the shader writes to it; extra phantom binding 6. Fixed in ToadStool `snp.rs`.
2. **AdapterInfo propagation** — wetSpring's `GpuF64::new()` used
   `WgpuDevice::from_existing_simple()` which sets synthetic adapter info, breaking
   ToadStool's RTX 4070 Ada Lovelace detection and f64 exp/log polyfill. Fixed to
   use `WgpuDevice::from_existing()` with real `AdapterInfo`.

### Cross-Spring Evolution (S62)

ToadStool `barracuda` is the convergence hub for all springs (660+ WGSL shaders):

| Spring | Contribution | Key Primitives |
|--------|-------------|-----------|
| **hotSpring** | Precision shaders (`df64_core.wgsl`, `Fp64Strategy`), lattice QCD, spectral theory | CG solver, Lanczos, Anderson, Hofstadter, Hermite/Laguerre, ESN reservoir |
| **wetSpring** | Bio ODE systems, NMF, genomics shaders, math_f64, PeakDetect, TransE | NMF (S58), 5 ODE bio (S58), ridge (S59), Anderson (S59), PeakDetect (S62), TransE (S60), erf/ln_gamma/trapz (always-on) |
| **neuralSpring** | Graph theory, ML inference, eigensolvers, TensorSession | graph_laplacian, effective_rank, numerical_hessian, belief_propagation, boltzmann_sampling, ValidationHarness |
| **airSpring** | IoT, precision agriculture | Richards PDE, moving_window, Kriging, pow_f64/acos fixes |

Cross-spring benefits measured: upstream ODE integrators are **10-43% faster** than
local (ToadStool optimizes across all springs' usage). hotSpring's `Fp64Strategy`
gives wetSpring f64 on all GPUs. neuralSpring's graph primitives enable wetSpring
community network analysis. 12 provenance tags track origins across springs.

### Streaming & Dispatch Validation (Feb 22, 2026)

| Exp | Binary | Checks | What it proves |
|-----|--------|:------:|----------------|
| 072 | `validate_gpu_streaming_pipeline` | 17 | Pre-warmed FMR eliminates per-stage dispatch; 1.22x streaming speedup |
| 073 | `validate_dispatch_overhead_proof` | 21 | Streaming beats individual at all batch sizes; overhead quantified |
| 074 | `validate_substrate_router` | 20 | GPU↔NPU↔CPU routing; PCIe topology; fallback parity |

### ToadStool Bio Rewire (Feb 22, 2026)

| Exp | Binary | Checks | What it proves |
|-----|--------|:------:|----------------|
| 077 | (all GPU binaries) | 451 | Full revalidation after 8-module rewire to ToadStool primitives |

Bugs found and fixed: SNP binding layout (ToadStool), AdapterInfo propagation (wetSpring).

### ODE Flat API + Dispatch Routing (Feb 22, 2026)

| Exp | Binary | Checks | What it proves |
|-----|--------|:------:|----------------|
| 079 | `validate_barracuda_cpu_v6` | 48 | GPU-compatible flat param APIs preserve bitwise ODE math across all 6 bio models |
| 080 | `validate_dispatch_routing` (forge) | 35 | Forge router correctly classifies 11 workloads across 5 substrate configs |

### Tier B → A Module Refactoring (Feb 22, 2026)

| Exp | Module | Tests | What it proves |
|-----|--------|:-----:|----------------|
| 081 | `kmer` | 4 | Histogram (4^k) + sorted pairs GPU layouts, round-trip fidelity |
| 082 | `unifrac` | 4 | CSR flat tree + sample matrix, UniFrac parity through flat path |
| 083 | `taxonomy` | 3 | Int8 affine quantization, argmax parity with f64 for NPU inference |
| 084 | metalForge full | 35+ | 12-domain cross-substrate (extends Exp065: +SW, Gillespie, DT, spectral) |
| 085 | Tier A layouts | 43 | kmer histogram/sorted-pairs RT, unifrac CSR RT, taxonomy int8 parity |
| 086 | metalForge pipeline | 45 | 5-stage dispatch routing + CPU/NPU parity + flat buffer readiness |
| 087 | GPU Extended Domains | 50+ | EIC, PCoA, Kriging, Rarefaction — 4 new GPU domains (--features gpu) |
| 088 | metalForge PCIe Direct | 32 | 6 paths + buffer contracts (CPU-only binary) |
| 089 | ToadStool Streaming Dispatch | 25 | 5 patterns + determinism (CPU-only binary) |
| 090 | Pure GPU Streaming Pipeline | 80 | 4 modes: round-trip, streaming, parity, batch scaling (--features gpu) |
| 091 | Streaming vs Round-Trip Benchmark | 2 | CPU ↔ RT ↔ streaming parity + Bray-Curtis error (--features gpu) |
| 092 | CPU vs GPU All 16 Domains | 48 | 16 domains CPU↔GPU parity (--features gpu) |
| 093 | metalForge Full v3 (16 domains) | 28 | 16 domains substrate-independent (--features gpu) |
| 094 | Cross-Spring Evolution Validation | 39 | 5 neuralSpring primitives CPU↔GPU parity (--features gpu) |
| 095 | Cross-Spring Scaling Benchmark | 7 | Cross-spring scaling at realistic sizes (--release --features gpu) |
| 096 | ToadStool Bio Op Absorption | 10 | 4 upstream ops validated (--features gpu), Lean phase complete |
| 097–104 | (See above experiment table) | | metalForge v4-v6, CPU v8, GPU promotion |
| 105 | Pure GPU Streaming v2 — Analytics | 27 | Pre-warmed Bray-Curtis + spectral cosine + full analytics pipeline (--features gpu) |
| 106 | GPU Streaming — ODE + Phylogenetics | 45 | 6 pre-warmed primitives, multi-dispatch proof, zero recompilation (--features gpu) |
| 107 | Spectral Cross-Spring (Anderson/QS) | 25 | Anderson 1D/2D/3D, Almost-Mathieu, Lanczos, QS-disorder analogy (--features gpu) |

### ODE Lean + Cross-Spring Benchmark (Phase 38 lean)

| Benchmark | Checks | What it proves |
|-----------|:------:|----------------|
| `benchmark_ode_lean_crossspring` | 11 | 5 ODE systems → upstream `generate_shader()` WGSL, CPU parity (4/5 exact, 1 clamping-divergent), upstream 21–51% faster (S62), linear batch scaling (--release --features gpu) |

5 local WGSL files deleted (30,424 bytes). All GPU modules use `BatchedOdeRK4<S>::generate_shader()`.

### Structural Evolution (Phase 23, Exp097)

| Target | Change | Validation |
|--------|--------|------------|
| ODE trajectory | `Vec<Vec<f64>>` → flat `Vec<f64>` + `state_at()`/`states()` | 740 tests, all ODE scenarios pass |
| Gillespie trajectory | `Vec<Vec<i64>>` → flat `Vec<i64>` + `states_iter()` | SSA convergence pass |
| DADA2 error model | 5 functions unified; GPU delegates | Denoising parity preserved |
| UniFrac distance matrix | N×N → condensed upper-triangle | UniFrac distance parity |
| Adapter trim | `(FastqRecord, bool)` → `Option<FastqRecord>` | FASTQ trimming pass |
| PCoA coordinates | `Vec<Vec<f64>>` → flat + `coord()` | PCoA eigendecomposition pass |
| ODE GPU polyfill | Hardcoded → `dev.needs_f64_exp_log_workaround()` | ODE GPU sweep pass |

740 tests. 48/48 CPU-GPU. 39/39 cross-spring. 0 clippy warnings.

### Handoff Documents

| Document | Location | Purpose |
|----------|----------|---------|
| **V33 — CPU-math lean (barracuda always-on)** | `wateringHole/handoffs/WETSPRING_TOADSTOOL_V33_CPUMATH_LEAN_FEB25_2026.md` | Phase 41, zero fallback code, ~177 lines dual-path removed |
| V32 — S62 lean: PeakDetect, TransE | `wateringHole/handoffs/WETSPRING_TOADSTOOL_V32_S62_LEAN_FEB24_2026.md` | Phase 41, S62 lean, paper queue fully GPU-covered |
| V31 — Absorption targets | `wateringHole/handoffs/WETSPRING_TOADSTOOL_V31_ABSORPTION_TARGETS_FEB24_2026.md` | Phase 41, absorption targets, cross-spring insights |
| V30 — S59 lean: NMF, ridge, ODE, Anderson | `wateringHole/handoffs/WETSPRING_TOADSTOOL_V30_S59_LEAN_FEB24_2026.md` | Phase 41, S59 lean, ~1,312 lines removed, 42 primitives |
| Shader evolution | `wateringHole/CROSS_SPRING_SHADER_EVOLUTION.md` | 660+ WGSL shader provenance (cross-spring, S62) |
| Archive | `wateringHole/handoffs/archive/` | V7-V29 (fossil record) |
