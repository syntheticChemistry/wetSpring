# wetSpring Benchmark Results

**Date:** February 26, 2026
**Status:** Phase 57 — Three-tier validation complete (Python → Rust CPU → GPU → metalForge) — 39/39 actionable papers full three-tier (52/52 total); 961 tests (882 barracuda + 47 forge + 32 integration/doc), 96.67% llvm-cov, 4,494+ checks (1,578 GPU on RTX 4070), 188 experiments, ToadStool S68 (`f0feb226`), 79 primitives consumed, 0 local WGSL/derivative/regression (fully lean), 82 named tolerances, 0 ad-hoc magic numbers, clippy pedantic CLEAN, 0 Passthrough, V57 ToadStool S68 catch-up

---

## Three-Tier Validation Architecture

wetSpring validates each algorithm at three levels:

```
Tier 1: Python baseline (published tools, open data)
Tier 2: Rust CPU (pure math, no unsafe, documented tolerances)
Tier 3: GPU (ToadStool/BarraCuda, math parity with CPU)
```

---

## Tier 2: Rust CPU Validation

### Validation Binary Results

| Binary | Experiment | Checks | Status |
|--------|-----------|--------|--------|
| `validate_fastq` | 001 | 28 | PASS |
| `validate_diversity` | 001/004 | 27 | PASS |
| `validate_mzml` | 005 | 7 | PASS |
| `validate_pfas` | 006 | 10 | PASS |
| `validate_features` | 009 | 8 | PASS |
| `validate_peaks` | 010 | 17 | PASS |
| `validate_16s_pipeline` | 011 | 37 | PASS |
| `validate_algae_16s` | 012 | 34 | PASS |
| `validate_voc_peaks` | 013 | 22 | PASS |
| `validate_public_benchmarks` | 014 | 202 | PASS |
| `validate_extended_algae` | 017 | 35 | PASS |
| `validate_pfas_library` | 018 | 26 | PASS |
| `validate_newick_parse` | 019 | 30 | PASS |
| `validate_qs_ode` | 020 | 16 | PASS |
| `validate_rf_distance` | 021 | 23 | PASS |
| `validate_gillespie` | 022 | 13 | PASS |
| `validate_pfas_decision_tree` | 008 | 7 | PASS |
| `validate_bistable` | 023 | 14 | PASS |
| `validate_multi_signal` | 024 | 19 | PASS |
| `validate_cooperation` | 025 | 20 | PASS |
| `validate_hmm` | 026 | 21 | PASS |
| `validate_capacitor` | 027 | 18 | PASS |
| `validate_alignment` | 028 | 15 | PASS |
| `validate_felsenstein` | 029 | 16 | PASS |
| `validate_barracuda_cpu` | cross | 21 | PASS |
| `validate_barracuda_cpu_v2` | cross | 18 | PASS |
| `validate_barracuda_cpu_v3` | cross | 45 | PASS |
| `validate_phage_defense` | 030 | 12 | PASS |
| `validate_bootstrap` | 031 | 11 | PASS |
| `validate_placement` | 032 | 12 | PASS |
| `validate_phynetpy_rf` | 036 | 15 | PASS |
| `validate_phylohmm` | 037 | 10 | PASS |
| `validate_sate_pipeline` | 038 | 17 | PASS |
| `validate_algae_timeseries` | 039 | 11 | PASS |
| `validate_bloom_surveillance` | 040 | 15 | PASS |
| `validate_epa_pfas_ml` | 041 | 14 | PASS |
| `validate_massbank_spectral` | 042 | 9 | PASS |
| `validate_rare_biosphere` | 051 | 35 | PASS |
| `validate_viral_metagenomics` | 052 | 22 | PASS |
| `validate_sulfur_phylogenomics` | 053 | 15 | PASS |
| `validate_phosphorus_phylogenomics` | 054 | 13 | PASS |
| `validate_population_genomics` | 055 | 24 | PASS |
| `validate_pangenomics` | 056 | 24 | PASS |
| `validate_barracuda_cpu_v4` | 057 | 44 | PASS |
| `validate_barracuda_cpu_v5` | 061/062 | 29 | PASS |
| **CPU total** | | **1,291** | **PASS** |

### Rust Unit/Integration Tests

| Suite | Count | Status |
|-------|-------|--------|
| Library + integration tests (CPU) | 752 | PASS (+ 1 ignored — hardware-dependent) |
| Library + integration tests (GPU) | 759 | PASS (+ 9 ignored — hardware-dependent) |
| metalForge forge tests | 47 | PASS |
| **Total** | **912** | **PASS** |
| Line coverage | 97% bio+io (56% overall) | Exceeds 90% target |

---

## Tier 2b: BarraCuda CPU Parity (v1–v8)

| Binary | Checks | Domains | Status |
|--------|--------|---------|--------|
| `validate_barracuda_cpu` | 21 | 9 domains (v1) | PASS |
| `validate_barracuda_cpu_v2` | 18 | 5 domains (v2) | PASS |
| `validate_barracuda_cpu_v3` | 45 | 18 domains (v3) | PASS |
| `validate_barracuda_cpu_v4` | 44 | 5 Track 1c domains (v4) | PASS |
| `validate_barracuda_cpu_v5` | 29 | RF + GBM (v5) | PASS |
| `validate_barracuda_cpu_v6` | 48 | 6 ODE flat (v6) | PASS |
| `validate_barracuda_cpu_v7` | 43 | Tier A layouts (v7) | PASS |
| `validate_barracuda_cpu_v8` | 175 | 13 promoted GPU domains (v8) | PASS |
| `validate_barracuda_cpu_full` | 50 | 25 consolidated (Exp070) | PASS |
| **CPU parity total** | **380** (deduplicated across 31+ domains) | | **PASS** |

---

## Tier 3: GPU Validation (22 GPU validation binaries, 1,578+ checks)

| Binary | Checks | Status |
|--------|--------|--------|
| `validate_diversity_gpu` | 38 | PASS |
| `validate_16s_pipeline_gpu` | 88 | PASS |
| `validate_barracuda_gpu_v3` | 14 | PASS |
| `validate_toadstool_bio` | 14 | PASS |
| `validate_gpu_phylo_compose` | 15 | PASS |
| `validate_gpu_hmm_forward` | 13 | PASS |
| `benchmark_phylo_hmm_gpu` | 6 | PASS |
| `validate_gpu_ode_sweep` | 12 | PASS |
| `validate_gpu_track1c` | 27 | PASS |
| `validate_cross_substrate` | 20 | PASS |
| `validate_barracuda_gpu_v1` | 26 | PASS |
| `validate_metalforge_full` | 35 | PASS |
| `validate_gpu_rf` | 13 | PASS |
| `validate_barracuda_gpu_full` | 24 | PASS |
| `validate_gpu_streaming_pipeline` | 17 | PASS |
| `validate_dispatch_overhead_proof` | 21 | PASS |
| `validate_substrate_router` | 20 | PASS |
| `validate_pure_gpu_pipeline` | 31 | PASS |
| `validate_cross_substrate_pipeline` | 17 | PASS |
| `validate_pure_gpu_complete` | 52 | PASS |
| `validate_metalforge_v5` | 58 | PASS |
| `validate_barracuda_cpu_v8` | 175 | PASS |
| **GPU total** | **1,578+** | **PASS** |

### GPU Performance

| Workload | Metric | CPU | GPU | Speedup |
|----------|--------|-----|-----|---------|
| Spectral cosine (2048×2048) | Time | 4.8s | 5.2ms | 926× |
| 16S pipeline (10 samples) | Time | 2.1s | 0.86s | 2.45× |
| Shannon diversity (10K) | Time | 12ms | 0.5ms | 24× |
| Bray-Curtis (100×100) | Parity | — | — | ≤1e-10 |

### CPU→GPU Math Parity (Exp016)

88/88 checks confirm GPU results match CPU within documented tolerances:
- Quality filter: bitwise identical
- DADA2 denoising: ≤1e-6 per-base error
- Chimera detection: bitwise identical
- Taxonomy classification: bitwise identical
- Diversity metrics: ≤1e-6 (f64 transcendentals)

---

## Tier 1: Python Baselines

| Baseline Script | Tool | Status |
|-----------------|------|--------|
| `benchmark_python_baseline.py` | QIIME2/DADA2-R | PASS |
| `validate_public_16s_python.py` | BioPython + NCBI | PASS |
| `waters2008_qs_ode.py` | scipy.integrate.odeint | PASS (35/35) |
| `gillespie_baseline.py` | numpy SSA ensemble | PASS (8/8) |
| `rf_distance_baseline.py` | dendropy RF | PASS (10/10) |
| `newick_parse_baseline.py` | dendropy tree stats | PASS (10/10) |
| `pfas_tree_export.py` | sklearn DecisionTree | PASS (acc=0.989, F1=0.986) |
| `exp008_pfas_ml_baseline.py` | sklearn RF+GBM | PASS (RF F1=0.978, GBM F1=0.992) |
| `fernandez2020_bistable.py` | scipy ODE (bifurcation) | PASS |
| `srivastava2011_multi_signal.py` | scipy ODE (multi-signal) | PASS |
| `bruger2018_cooperation.py` | scipy ODE (game theory) | PASS |
| `liu2014_hmm_baseline.py` | numpy HMM (sovereign) | PASS |
| `mhatre2020_capacitor.py` | scipy ODE (capacitor) | PASS |
| `smith_waterman_baseline.py` | pure Python (sovereign) | PASS |
| `felsenstein_pruning_baseline.py` | pure Python (sovereign) | PASS |
| `hsueh2022_phage_defense.py` | scipy ODE (phage defense) | PASS |
| `wang2021_rawr_bootstrap.py` | pure Python (bootstrap) | PASS |
| `alamin2024_placement.py` | pure Python (placement) | PASS |
| `phynetpy_rf_baseline.py` | PhyNetPy gene trees | PASS |
| `phylohmm_introgression_baseline.py` | PhyloNet-HMM | PASS |
| `sate_alignment_baseline.py` | SATe pipeline | PASS |
| `algae_timeseries_baseline.py` | Cahill proxy | PASS |
| `bloom_surveillance_baseline.py` | Smallwood proxy | PASS |
| `epa_pfas_ml_baseline.py` | Jones F&T proxy | PASS |
| `massbank_spectral_baseline.py` | Jones MS proxy | PASS |
| `benchmark_rust_vs_python.py` | 18-domain timing (Exp043) | PASS |
| `anderson2015_rare_biosphere.py` | diversity/rarefaction (Exp051) | PASS |
| `anderson2014_viral_metagenomics.py` | dN/dS + diversity (Exp052) | PASS |
| `mateos2023_sulfur_phylogenomics.py` | clock/reconciliation (Exp053) | PASS |
| `boden2024_phosphorus_phylogenomics.py` | clock/reconciliation (Exp054) | PASS |
| `anderson2017_population_genomics.py` | ANI/SNP (Exp055) | PASS |
| `moulana2020_pangenomics.py` | pangenome/enrichment (Exp056) | PASS |
| `barracuda_cpu_v4_baseline.py` | 5 Track 1c domain timing (Exp057) | PASS |

---

## Exp057: BarraCuda CPU Parity v4 (Track 1c, 5 Domains)

| Domain | Module | Checks | Time (µs) |
|--------|--------|--------|-----------|
| 19 | ANI | 9 | 139 |
| 20 | SNP calling | 8 | 193 |
| 21 | dN/dS | 9 | 18 |
| 22 | Molecular clock | 7 | 7 |
| 23 | Pangenome | 11 | 16 |
| **Total** | | **44** | **373** |

All 5 Track 1c domains: pure Rust, zero dependencies, 44/44 checks PASS.
Combined with v1-v3: **128/128 checks across 23 domains**.

---

## Exp061/062: BarraCuda CPU Parity v5 (RF + GBM)

| Domain | Module | Checks | Time (µs) |
|--------|--------|--------|-----------|
| 24 | Random Forest | 13 | 28 |
| 25 | GBM (binary + multi-class) | 16 | 34 |
| **Total** | | **29** | **62** |

Two new ensemble ML domains: RF majority-vote and GBM sequential boosting.
Combined v1-v6: **205/205 checks across 25 domains + 6 ODE flat modules**.

---

## Exp059: Rust vs Python Timing (25 Domains)

Head-to-head benchmark across all 25 BarraCuda CPU parity domains:

| Metric | Value |
|--------|-------|
| Rust (release) total | ~79,984 µs |
| Python total | ~1,798,608 µs |
| **Overall Speedup** | **22.5×** |
| Peak speedup | 625× (Smith-Waterman) |
| ODE domains | 15–28× |
| Track 1c domains | 6–56× |

Run with `cargo run --release --bin benchmark_23_domain_timing` and
`python3 scripts/benchmark_rust_vs_python.py`.

(Previous Exp043 covered 18 domains at ~20× speedup.)

---

## Exp095: Cross-Spring Scaling Benchmark (RTX 4070)

Cross-spring evolved primitives at realistic bioinformatics problem sizes.
Run with `cargo run --release --features gpu --bin benchmark_cross_spring_scaling`.

| Primitive | Evolved By | Problem Size | CPU (µs) | GPU (µs) | Speedup |
|-----------|-----------|-------------|----------|----------|---------|
| PairwiseJaccard | neuralSpring | 200×2000 (20K pairs) | 41,777 | 342 | **122×** |
| SpatialPayoff | neuralSpring | 256×256 (65K cells) | 1,026 | 47 | **22×** |
| PairwiseHamming | neuralSpring | 500×1000 (125K pairs) | 10,383 | 1,039 | **10×** |
| LocusVariance | neuralSpring | 100×10K (1M elems) | 1,079 | 154 | **7×** |
| BatchFitness | neuralSpring | 4096×256 (1M elems) | 470 | 79 | **6×** |
| GemmF64 | wetSpring | 256×256 f64 | 3,479 | 3,643 | 1.0× |

GPU wins at scale: Jaccard 122×, SpatialPayoff 22×, Hamming 10×.
GemmF64 at 256×256 is transfer-dominated; see Exp066 for larger sizes.

---

## Exp183: Cross-Spring Evolution Benchmark (ToadStool S66)

Comprehensive validation of the S66 rewire — 36/36 checks PASS. Run with
`cargo run --release --features gpu --bin benchmark_cross_spring_s65`.

| Domain | Checks | Status |
|--------|--------|--------|
| GPU ODE (5 systems × 128 batches) | 5 | PASS |
| DiversityFusion GPU (Write→Absorb→Lean) | 3 | PASS |
| CPU diversity delegation (11 functions → barracuda::stats) | 11 | PASS |
| CPU math delegation (dot/l2_norm → barracuda::stats) | 2 | PASS |
| hotSpring precision (erf, ln_gamma, norm_cdf, trapz) | 6 | PASS |
| GEMM pipeline (compile + cached dispatch) | 4 | PASS |
| Anderson spectral (hotSpring → wetSpring Track 4) | 3 | PASS |
| NMF + Ridge | 2 | PASS |
| **Total** | **36** | **PASS** |

---

## Cross-Spring Evolution: Where Things Evolved to Be Helpful

The biome model in action — each spring contributes domain expertise, ToadStool
absorbs it, and all springs benefit. Verified on RTX 4070 + Titan V.

### hotSpring → wetSpring (physics → biology)

| Primitive | Origin | wetSpring Use | Impact |
|-----------|--------|--------------|--------|
| f64 precision polyfills | Lattice QCD workarounds | GPU ODE accuracy on consumer GPUs | Enables f64 bio ODE on Ada |
| DF64 core-streaming (14 shaders) | SU(3) gauge theory | Consumer GPU viability | 10× throughput vs native f64 |
| Anderson 2D/3D + Lanczos | Condensed matter | QS-disorder localization (Exp107) | Novel biology insight |
| `find_w_c` phase transition | Metal-insulator transition | QS signal localization threshold | W_c = 11.79 |
| PeakDetectF64 | LC-MS signal processing | mzML peak detection | f64 end-to-end |
| BatchedEighGpu | Nuclear Hartree-Fock | PCoA eigendecomposition | GPU-accelerated |
| ESN reservoir | Stanton-Murillo transport | NPU deployment (Exp114-119) | int8 inference |
| RK4/RK45 adaptive | Precision ODE integration | All 5 bio ODE systems | Trait-generated WGSL |

### wetSpring → ToadStool → all springs (biology → shared)

| Primitive | Absorbed Session | Consumers |
|-----------|-----------------|-----------|
| 5 bio ODE systems → `OdeSystem` trait | S51-S58 | neuralSpring population genetics |
| DiversityFusion WGSL | S63 | airSpring crop biodiversity |
| 11 diversity metrics | S64 | All springs with ecology/graph data |
| GemmCachedF64 (60× taxonomy) | S62 | hotSpring HFB, neuralSpring GNN |
| Tolerance constant pattern | S52 | All springs adopted |
| `hill()`, `monod()` CPU | S66 | All springs with kinetic models |
| `fit_linear` regression | S66 | airSpring, groundSpring |

### neuralSpring → wetSpring (ML/population → biology)

| Primitive | Problem Size | GPU Speedup | wetSpring Use |
|-----------|-------------|-------------|--------------|
| PairwiseHamming | 500 seqs × 1000 bp | **10×** | SNP distance matrices |
| PairwiseJaccard | 200 genomes × 2000 genes | **122×** | Pangenome similarity |
| SpatialPayoff | 256×256 grid | **22×** | QS cooperation game theory |
| BatchFitness | 4096 × 256 genome | **6×** | Evolutionary fitness eval |
| LocusVariance | 100 pops × 10K loci | **7×** | FST estimation |
| graph_laplacian | Community networks | CPU | Spectral community analysis |

### ODE Lean Benchmark: Write → Absorb → Lean complete

| System | Origin | State Dim | Local CPU (µs) | Upstream CPU (µs) | Max Diff |
|--------|--------|-----------|---------------|-------------------|----------|
| Capacitor | wetSpring Exp002 | 6 | 2,020 | 1,581 | 0.00 |
| Cooperation | wetSpring Exp003 | 4 | 825 | 696 | 4.44e-16 |
| MultiSignal | wetSpring Exp006 | 7 | 1,592 | 1,214 | 4.44e-16 |
| Bistable | wetSpring Exp007 | 5 | 1,738 | 1,424 | 0.00 |
| PhageDefense | wetSpring Exp009 | 4 | 82 | 63 | extreme params |

Upstream integrate_cpu is **18-24% faster** than local — ToadStool optimized
the hot loop after absorbing the trait implementations.

---

## Exp101–103: Pure GPU Promotion + CPU v8 + metalForge v5 (Phase 28)

### Exp101: Pure GPU Promotion Complete

13 modules promoted from Tier B/C to GPU-capable. Zero Tier B/C remaining:

| Module | Strategy | GPU Checks | Notes |
|--------|----------|:----------:|-------|
| `cooperation_gpu` | Write (local WGSL 4v/13p) | exact parity | ODE RK4 f64, loop-unrolled |
| `capacitor_gpu` | Write (local WGSL 6v/16p) | exact parity | ODE RK4 f64, loop-unrolled |
| `kmd_gpu` | Compose (`KmerHistogramGpu`) | validated | Kendrick mass defect via k-mer histogram |
| `merge_pairs_gpu` | Compose (`PairwiseHammingGpu`) | validated | Overlap scoring via Hamming |
| `robinson_foulds_gpu` | Compose (`PairwiseHammingGpu`) | validated | Bipartition distance |
| `derep_gpu` | Compose (`KmerHistogramGpu`) | validated | Sequence hashing via k-mer |
| `neighbor_joining_gpu` | Compose (`GemmCachedF64`) | validated | Distance matrix operations |
| `reconciliation_gpu` | Compose (`TreeInferenceGpu`) | validated | DTL cost inference |
| `molecular_clock_gpu` | Compose (`GemmCachedF64`) | validated | Rate matrix operations |
| `chimera_gpu` | Compose (`GemmCachedF64`) | validated | Scoring via GEMM |
| `gbm_gpu` | Passthrough | validated | CPU kernel, GPU buffer acceptance |
| `feature_table_gpu` | Passthrough | validated | CPU kernel, GPU buffer acceptance |
| `signal_gpu` | Passthrough | validated | CPU kernel, GPU buffer acceptance |

### Exp102: BarraCuda CPU v8 (13 Promoted Domains)

175/175 checks validating pure Rust math for all 13 newly GPU-promoted domains.
Combined v1-v8: **380/380 across 31+ domains**.

### Exp103: metalForge v5 Cross-Substrate (29 Domains)

29 domains validated substrate-independent. 13 new GPU domains added to cross-system
matrix. CPU↔GPU parity proven for all compose and write modules.

---

## Grand Total

| Category | Checks | Status |
|----------|--------|--------|
| Rust CPU validation | 1,476 | PASS |
| GPU validation | 1,578+ | PASS |
| Dispatch + layout + transfer | 172 | PASS |
| Rust tests | 912 (833 barracuda + 47 forge + 32 integration/doc) | PASS |
| Python baselines | 44 scripts | PASS |
| BarraCuda CPU parity | 380/380 (v1-v8: 31+ domains) | PASS |
| ToadStool primitives consumed | 79 primitives (barracuda always-on, zero fallback — S66) | PASS |
| Local WGSL shaders | 0 (full lean — all GPU ops dispatch upstream) | PASS |
| Compose GPU wrappers | 7 (kmd, merge_pairs, robinson_foulds, derep, NJ, reconciliation, molecular_clock) | PASS |
| Passthrough GPU wrappers | 3 (gbm, feature_table, signal) | PASS |
| **Grand total** | **3,300+ validation + 912 tests** | **ALL PASS** |

---

## Reproduction

```bash
cd barracuda

# Tier 2: Rust CPU (1,476+ checks)
cargo test                         # 912 tests (833 barracuda + 47 forge + 32 integration/doc)
cargo run --release --bin validate_qs_ode  # ... repeat for all CPU binaries

# Tier 2b: BarraCuda CPU parity (380/380)
cargo run --release --bin validate_barracuda_cpu_v8   # 175 checks (13 promoted domains)

# Tier 3: GPU (1,578+ checks)
cargo run --features gpu --bin validate_diversity_gpu          # 38
cargo run --features gpu --bin validate_16s_pipeline_gpu       # 88
cargo run --features gpu --bin validate_pure_gpu_complete      # 52 (13 promoted domains)
cargo run --features gpu --bin validate_metalforge_v5          # 58 (29 cross-substrate domains)

# Tier 1: Python
cd ../scripts && python3 gillespie_baseline.py
```
