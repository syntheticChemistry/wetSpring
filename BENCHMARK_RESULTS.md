# wetSpring Benchmark Results

**Date:** March 1, 2026
**Status:** Phase 85 — Comprehensive sweep GREEN (Paper → CPU → GPU → Streaming → metalForge → NPU → NUCLEUS) — 52/52 papers, 50/50 three-tier; 1,210 tests (962 barracuda lib + 60 integration + 22 doc + 166 forge), 6,626+ checks (1,945+ GPU on RTX 4070, 60 NPU on AKD1000), 259 experiments, ToadStool S70+++ (`1dd7e338`), 93 primitives consumed, 26 CPU domains, 21 GPU domains, Python parity proven (15 domains bit-identical to SciPy), 0 local WGSL (fully lean), 97 named tolerances, clippy pedantic CLEAN. **V85:** Exp256-258 (EMP Atlas 30K samples, NUCLEUS Data Pipeline, Tower-Node — all 6 primals READY, IPC 3.2× overhead, bit-identical dispatch)

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
| Library + integration + IPC tests (CPU) | 955 | PASS (+ 1 ignored — hardware-dependent) |
| metalForge forge tests | 113 | PASS |
| **Total** | **1,210** (962 + 60 integration + 22 doc + 166 forge) | **PASS** |
| Line coverage | 95.46% line / 93.54% fn / 94.99% branch | Exceeds 90% target |

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
| `validate_barracuda_cpu_v9` | 27 | Track 3 drug repurposing (v9, Exp163) | PASS |
| `validate_barracuda_cpu_v10` | 75 | V59 science extensions (v10, Exp190) | PASS |
| `validate_barracuda_cpu_v11` | 64 | IPC dispatch math fidelity (v11, Exp206) | PASS |
| `validate_soil_qs_cpu_parity` | 49 | Track 4 soil QS pure Rust (Exp179) | PASS |
| **CPU parity total** | **546** (deduplicated across 36+ domains) | | **PASS** |

---

## Tier 3: GPU Validation (22 GPU validation binaries, 1,783+ checks)

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
| `validate_barracuda_gpu_v4` | 54 | PASS (Exp207: IPC GPU dispatch) |
| `validate_soil_qs_gpu` | 23 | PASS (Exp180: Track 4 soil QS) |
| `validate_gpu_v59_science` | 29 | PASS (Exp191: V59 science parity) |
| `validate_metalforge_v7_mixed` | 75 | PASS (Exp208: NUCLEUS mixed hardware) |
| **GPU total** | **1,783+** | **PASS** |

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
| Rust (release) total | ~51,358 µs |
| Python total | ~1,713,242 µs |
| **Overall Speedup** | **33.4×** |
| Peak speedup | 625× (Smith-Waterman) |
| ODE domains | 15–29× |
| Track 1c domains | 6–56× |

Run with `cargo run --release --bin benchmark_23_domain_timing` and
`python3 scripts/benchmark_rust_vs_python.py`.

(Previous Exp043 covered 18 domains at ~20× speedup.)

---

## Exp253: Python vs Rust Benchmark v3 (V84)

| Metric | Value |
|--------|-------|
| Domains | 15 domains paper parity proof |
| Checks | 35/35 PASS |
| Binary | `benchmark_python_vs_rust_v3` |

Validates Rust CPU matches Python across 15 domains with published-equation parity.

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

## Phase 62: Comprehensive Validation Sweep (Feb 27, 2026)

Full pipeline chain validated green — Python baselines through metalForge cross-substrate:

### Tier 1 → Tier 2: Python → BarraCuda CPU (Pure Rust Math)

| Benchmark | Python (µs) | Rust CPU (µs) | Speedup |
|-----------|-------------|---------------|---------|
| 23-domain total | 1,713,242 | 51,358 | **33.4×** |
| ODE integration (D01) | 461,002 | 8,283 | **56×** |
| Gillespie SSA (D02) | 717,988 | 18,700 | **38×** |
| Smith-Waterman (D04) | 7,500 | 12 | **625×** |
| Phage defense (D11) | 172,884 | 8,357 | **21×** |

Pure Rust, zero unsafe, zero dependencies. `cargo run --release --bin benchmark_23_domain_timing`.

### Tier 2 → Tier 3: BarraCuda CPU → GPU (Math Portability)

| Binary | Checks | Status | Wall-clock |
|--------|--------|--------|------------|
| `validate_barracuda_gpu_full` (Exp071) | 24/24 | PASS | 4.7s |
| `validate_barracuda_gpu_v1` (Exp064) | 26/26 | PASS | 4.0s |
| `validate_barracuda_gpu_v4` (Exp207) | 54/54 | PASS | 13.8s |
| `validate_soil_qs_gpu` (Exp180) | 23/23 | PASS | 3.4s |
| `validate_gpu_v59_science` (Exp191) | 29/29 | PASS | — |

GPU dispatch threshold (via `GpuF64::dispatch_threshold()`) routes small workloads to CPU,
large workloads to GPU. IPC layer adds zero numeric drift (Exp206: 64/64 EXACT_F64).

### Tier 3 → Pure GPU: ToadStool Unidirectional Streaming

| Binary | Checks | Status | Key Result |
|--------|--------|--------|------------|
| `validate_pure_gpu_streaming` (Exp090) | 80/80 | PASS | 441–837× vs round-trip |
| `validate_pure_gpu_streaming_v2` (Exp105) | 27/27 | PASS | Multi-domain analytics |
| `validate_streaming_ode_phylo` (Exp106) | 45/45 | PASS | ODE + phylo chained |
| `validate_pure_gpu_complete` (Exp101) | 52/52 | PASS | 13 modules promoted |

Streaming architecture: CPU → GPU → GPU → GPU → CPU (2 transfers vs 6 round-trip).
`GpuPipelineSession` pre-warms pipelines, `execute_to_buffer()` keeps data on GPU.

### Tier 4: metalForge Cross-Substrate (GPU → NPU → CPU)

| Binary | Checks | Status | Substrates |
|--------|--------|--------|------------|
| `validate_metalforge_v5` (Exp103) | 52/52 | PASS | CPU↔GPU 29 domains |
| `validate_metalforge_v6` (Exp104) | 24/24 | PASS | Three-tier complete |
| `validate_metalforge_v7_mixed` (Exp208) | 75/75 | PASS | NUCLEUS atomics, PCIe bypass |
| `validate_soil_qs_metalforge` (Exp182) | 14/14 | PASS | Track 4 cross-substrate |

metalForge routes workloads via `Capability` matching: `F64Compute` → GPU, `QuantizedInference` → NPU,
`ShaderDispatch` → GPU streaming. PCIe bypass topology modeled (NPU → GPU direct buffer transfer).

### Cross-Spring Evolution Benchmarks

| Binary | Checks | Status |
|--------|--------|--------|
| `benchmark_cross_spring_s68` (Exp189) | 28/28 | PASS |
| `benchmark_cross_spring_s65` (Exp183) | 36/36 | PASS |
| `benchmark_cross_spring_modern` (Exp169) | 20/20 | PASS |
| `benchmark_modern_systems_df64` (Exp166) | 19/19 | PASS |
| `benchmark_streaming_vs_roundtrip` (Exp091) | 2/2 | PASS |

---

## Grand Total

| Category | Checks | Status |
|----------|--------|--------|
| Rust CPU validation | 1,642 | PASS |
| GPU validation | 1,783+ | PASS |
| Dispatch + layout + transfer | 172 | PASS |
| IPC dispatch parity (CPU+GPU+metalForge) | 193 (Exp206-208) | PASS |
| Pure GPU streaming | 152 (Exp090+101+105+106) | PASS |
| Rust tests | 1,103 (977 barracuda lib + 60 integration + 19 doc + 47 forge) | PASS |
| Python baselines | 44 scripts | PASS |
| BarraCuda CPU parity | 546/546 (v1-v11: 36+ domains) | PASS |
| ToadStool primitives consumed | 93 primitives (barracuda always-on, zero fallback — S70+++) | PASS |
| Local WGSL shaders | 0 (full lean — all GPU ops dispatch upstream) | PASS |
| Compose GPU wrappers | 7 (kmd, merge_pairs, robinson_foulds, derep, NJ, reconciliation, molecular_clock) | PASS |
| Passthrough GPU wrappers | 0 (all promoted — S66 lean cycle) | PASS |
| **Grand total** | **6,569+ validation + 1,210 tests** | **ALL PASS** |

---

## Reproduction

```bash
cd barracuda

# Unit tests (977 lib + 60 integration + 19 doc + 47 forge = 1,103)
cargo test --features ipc

# Tier 2: BarraCuda CPU parity (546/546 across 36+ domains)
cargo run --release --bin validate_barracuda_cpu_full          # 50 (Exp070)
cargo run --release --bin validate_barracuda_cpu_v9            # 27 (Exp163)
cargo run --release --bin validate_barracuda_cpu_v10           # 75 (Exp190)
cargo run --features ipc --release --bin validate_barracuda_cpu_v11  # 64 (Exp206)
cargo run --release --bin validate_soil_qs_cpu_parity          # 49 (Exp179)

# Tier 3: GPU parity (1,783+ checks)
cargo run --features gpu --release --bin validate_barracuda_gpu_full  # 24 (Exp071)
cargo run --features gpu --release --bin validate_barracuda_gpu_v1   # 26 (Exp064)
cargo run --features gpu,ipc --release --bin validate_barracuda_gpu_v4  # 54 (Exp207)
cargo run --features gpu --release --bin validate_pure_gpu_complete   # 52 (Exp101)

# Tier 3b: Pure GPU streaming
cargo run --features gpu --release --bin validate_pure_gpu_streaming     # 80 (Exp090)
cargo run --features gpu --release --bin validate_pure_gpu_streaming_v2  # 27 (Exp105)
cargo run --features gpu --release --bin validate_streaming_ode_phylo    # 45 (Exp106)

# Tier 4: metalForge cross-substrate
cargo run --features gpu --release --bin validate_metalforge_v5         # 52 (Exp103)
cargo run --features gpu,ipc --release --bin validate_metalforge_v7_mixed  # 75 (Exp208)

# V84: Paper→CPU→GPU→Streaming pipeline
cargo run --release --bin validate_paper_math_control_v3       # 27 (Exp251, 32 papers)
cargo run --release --bin validate_barracuda_cpu_v19           # 42 (Exp252, 7 new domains)
cargo run --release --bin benchmark_python_vs_rust_v3          # 35 (Exp253, 15-domain parity)
cargo run --features gpu --release --bin validate_barracuda_gpu_v11  # 25 (Exp254, GPU portability)
cargo run --features gpu --release --bin validate_pure_gpu_streaming_v8  # 43 (Exp255, 6-stage pipeline)
cargo run --release --bin validate_emp_anderson_atlas                    # 35 (Exp256, 30K EMP samples)
cargo run --release --bin validate_nucleus_data_pipeline                 # 9  (Exp257, three-tier routing)
cargo run --release --features ipc --bin validate_nucleus_tower_node     # 13 (Exp258, all primals READY)

# Benchmarks
cargo run --release --bin benchmark_23_domain_timing           # Python→Rust CPU (33.4×)
cargo run --features gpu,json --release --bin benchmark_three_tier  # Python→CPU→GPU
cargo run --features gpu --release --bin benchmark_streaming_vs_roundtrip  # Streaming vs RT

# Tier 1: Python baselines
cd ../scripts && python3 benchmark_rust_vs_python.py && python3 benchmark_python_baseline.py
```
