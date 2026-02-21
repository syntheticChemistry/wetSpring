# Handoff v6: wetSpring → ToadStool / BarraCUDA Team

**Date:** February 21, 2026
**From:** wetSpring (ecoPrimals — Life Science & Analytical Chemistry)
**To:** ToadStool / BarraCUDA core team
**License:** AGPL-3.0-or-later
**Supersedes:** All prior handoffs (v1–v5, archived in `archive/handoffs/`)
**ToadStool reviewed:** barracuda v0.2.0 at commit cce8fe7c (bio absorption confirmed)

---

## Executive Summary

wetSpring has completed **76 experiments** with **1,742 validation checks** (all PASS)
across **25 algorithmic domains**, proving the full Python → Rust CPU → GPU → multi-substrate
portability path for life science, analytical chemistry, and environmental monitoring.

**Since v5 handoff (Feb 20):**

1. **Consolidated math proofs** — 25-domain CPU (50 checks, Exp070), 11-domain GPU (24 checks, Exp071)
2. **Streaming pipeline** — `GpuPipelineSession` pre-warmed FMR delivers 1.27× speedup (Exp072)
3. **Dispatch overhead quantified** — streaming beats individual dispatch at all batch sizes (Exp073)
4. **Substrate router** — GPU↔NPU↔CPU routing with PCIe topology awareness, fallback parity (Exp074)
5. **Pure GPU 5-stage pipeline** — diversity→BC→PCoA→stats→spectral, 0.1% overhead (Exp075)
6. **Cross-substrate pipeline** — GPU→NPU→CPU data flow with latency profiling (Exp076)
7. **Test suite grown** — 610 tests (547 lib + 50 integration + 13 doc), up from 552
8. **Binary count** — 73 total (50 CPU + 18 GPU validate + 5 benchmark)

**wetSpring totals:** 1,742 checks (1,291 CPU + 451 GPU), 610 tests, 93.5% line coverage,
76 experiments, 73 binaries, 25 CPU domains, 9 local WGSL shaders, 151 source files

---

## Part 1: What ToadStool Has Absorbed

| # | wetSpring Finding | ToadStool Primitive | Absorption Date |
|---|-------------------|---------------------|:---------------:|
| 1 | Bray-Curtis f64 | `ops::bray_curtis_f64::BrayCurtisF64` | Feb 16 |
| 2 | log_f64 coefficient bug | Fixed in `math_f64.wgsl` | Feb 16 |
| 3 | Shannon entropy FMR | `FusedMapReduceF64::shannon_entropy()` | Feb 16 |
| 4 | Simpson index FMR | `FusedMapReduceF64::simpson_index()` | Feb 16 |
| 5 | Smith-Waterman GPU | `ops::bio::smith_waterman::SmithWatermanGpu` | Feb 20 |
| 6 | Gillespie SSA GPU | `ops::bio::gillespie::GillespieGpu` | Feb 20 |
| 7 | Tree Inference GPU | `ops::bio::tree_inference::TreeInferenceGpu` | Feb 20 |
| 8 | Felsenstein GPU | `ops::bio::felsenstein::FelsensteinGpu` | Feb 20 |
| 9 | GemmF64::WGSL constant | `ops::linalg::gemm_f64::GemmF64::WGSL` | Feb 20 |

All 6 original bio primitive requests are addressed. Lean status confirmed.

---

## Part 2: Nine WGSL Shaders Ready for Absorption

All shaders live at `barracuda/src/shaders/`. Each has a CPU reference function,
documented tolerances, and validated GPU checks.

### 2.1 Pipeline Shaders (2)

#### `quality_filter.wgsl` — 88 pipeline checks (Exp016)

| Binding | Type | Purpose |
|---------|------|---------|
| `@group(0) @binding(0)` | `uniform` | `QualityFilterParams` (min_quality, window_size, etc.) |
| `@group(0) @binding(1)` | `storage, read` | `qual_data: array<u32>` — Phred scores |
| `@group(0) @binding(2)` | `storage, read` | `read_offsets: array<u32>` |
| `@group(0) @binding(3)` | `storage, read` | `read_lengths: array<u32>` |
| `@group(0) @binding(4)` | `storage, read_write` | `results: array<u32>` — trim positions |

- **Entry point:** `@compute @workgroup_size(256) fn quality_filter`
- **Dispatch:** `ceil(n_reads / 256)` workgroups
- **CPU reference:** `bio::quality::quality_trim()`
- **f64 transcendentals:** None
- **Polyfill:** Not needed (u32 only)
- **Proposed primitive:** `ParallelFilter<T>` — generalizes per-element filtering

#### `dada2_e_step.wgsl` — 88 pipeline checks (Exp016)

| Binding | Type | Purpose |
|---------|------|---------|
| `@group(0) @binding(0)` | `uniform` | `Dada2Params` (n_seqs, n_centers, max_len) |
| `@group(0) @binding(1)` | `storage, read` | `bases: array<u32>` — encoded sequences |
| `@group(0) @binding(2)` | `storage, read` | `quals: array<u32>` — quality scores |
| `@group(0) @binding(3)` | `storage, read` | `lengths: array<u32>` — per-sequence lengths |
| `@group(0) @binding(4)` | `storage, read` | `center_indices: array<u32>` |
| `@group(0) @binding(5)` | `storage, read` | `log_err: array<f64>` — precomputed log error table |
| `@group(0) @binding(6)` | `storage, read_write` | `scores: array<f64>` — output E-step scores |

- **Entry point:** `@compute @workgroup_size(256) fn e_step`
- **Dispatch:** `ceil(n_seqs * n_centers / 256)` workgroups
- **CPU reference:** `bio::dada2::e_step()`
- **f64 transcendentals:** None (log values precomputed on CPU)
- **Polyfill:** Not needed
- **Proposed primitive:** `BatchPairReduce<f64>` — each thread reduces one (sequence, center) pair

### 2.2 Bioinformatics Shaders (5)

#### `hmm_forward_f64.wgsl` — 13/13 GPU checks (Exp047)

| Binding | Type | Purpose |
|---------|------|---------|
| `@group(0) @binding(0)` | `uniform` | `HmmParams` (n_states, n_obs, n_seqs) |
| `@group(0) @binding(1)` | `storage, read` | `log_trans: array<f64>` — transition matrix |
| `@group(0) @binding(2)` | `storage, read` | `log_emit: array<f64>` — emission matrix |
| `@group(0) @binding(3)` | `storage, read` | `log_pi: array<f64>` — initial distribution |
| `@group(0) @binding(4)` | `storage, read` | `observations: array<u32>` — input symbols |
| `@group(0) @binding(5)` | `storage, read_write` | `log_alpha_out: array<f64>` — forward matrix |
| `@group(0) @binding(6)` | `storage, read_write` | `log_lik_out: array<f64>` — per-sequence log-likelihood |

- **Entry point:** `@compute @workgroup_size(256) fn main`
- **Dispatch:** `ceil(n_seqs / 256)` workgroups — one thread per sequence
- **CPU reference:** `bio::hmm::forward_log()`
- **f64 transcendentals:** `exp`, `log` (in `log_sum_exp2`)
- **Polyfill:** **YES** — `ShaderTemplate::for_driver_auto(source, true)`
- **Proposed primitive:** `HmmBatchForwardF64`

#### `batched_qs_ode_rk4_f64.wgsl` — 7/7 GPU checks (Exp049)

| Binding | Type | Purpose |
|---------|------|---------|
| `@group(0) @binding(0)` | `uniform` | `OdeConfig` (n_states, n_steps, dt) |
| `@group(0) @binding(1)` | `storage, read` | `initial_states: array<f64>` |
| `@group(0) @binding(2)` | `storage, read` | `batch_params: array<f64>` — per-trajectory parameters |
| `@group(0) @binding(3)` | `storage, read_write` | `output_states: array<f64>` |

- **Entry point:** `@compute @workgroup_size(256) fn main`
- **Dispatch:** `ceil(n_trajectories / 256)` workgroups — one thread per trajectory
- **CPU reference:** `bio::ode::rk4_integrate()`
- **f64 transcendentals:** `pow_f64` (Hill function)
- **Polyfill:** **YES** — `ShaderTemplate::for_driver_auto(source, true)`
- **Proposed primitive:** Fix upstream `BatchedOdeRK4F64` (currently broken on Ada Lovelace)

#### `ani_batch_f64.wgsl` — 7/7 GPU checks (Exp058)

| Binding | Type | Purpose |
|---------|------|---------|
| `@group(0) @binding(0)` | `uniform` | `AniParams` (n_pairs, seq_len) |
| `@group(0) @binding(1)` | `storage, read` | `seq_a: array<u32>` — encoded sequences |
| `@group(0) @binding(2)` | `storage, read` | `seq_b: array<u32>` |
| `@group(0) @binding(3)` | `storage, read_write` | `ani_out: array<f64>` — ANI scores |
| `@group(0) @binding(4)` | `storage, read_write` | `aligned_out: array<u32>` — aligned count |
| `@group(0) @binding(5)` | `storage, read_write` | `identical_out: array<u32>` — identical count |

- **Entry point:** `@compute @workgroup_size(256) fn main`
- **Dispatch:** `ceil(n_pairs / 256)` — one thread per pair
- **CPU reference:** `bio::ani::ani_pairwise()`
- **f64 transcendentals:** None (integer counting + f64 division)
- **Polyfill:** Not needed
- **Proposed primitive:** `AniBatchF64`

#### `snp_calling_f64.wgsl` — 5/5 GPU checks (Exp058)

| Binding | Type | Purpose |
|---------|------|---------|
| `@group(0) @binding(0)` | `uniform` | `SnpParams` (n_positions, n_sequences) |
| `@group(0) @binding(1)` | `storage, read` | `sequences: array<u32>` — encoded alignment |
| `@group(0) @binding(2)` | `storage, read_write` | `is_variant: array<u32>` — SNP flag |
| `@group(0) @binding(3)` | `storage, read_write` | `ref_allele: array<u32>` — reference base |
| `@group(0) @binding(4)` | `storage, read_write` | `depth_out: array<u32>` — read depth |
| `@group(0) @binding(5)` | `storage, read_write` | `alt_freq_out: array<f64>` — allele frequency |

- **Entry point:** `@compute @workgroup_size(256) fn main`
- **Dispatch:** `ceil(n_positions / 256)` — one thread per position
- **CPU reference:** `bio::snp::call_snps()`
- **f64 transcendentals:** None
- **Polyfill:** Not needed
- **Proposed primitive:** `SnpCallingF64`

#### `dnds_batch_f64.wgsl` — 9/9 GPU checks (Exp058)

| Binding | Type | Purpose |
|---------|------|---------|
| `@group(0) @binding(0)` | `uniform` | `DnDsParams` (n_pairs, codon_len) |
| `@group(0) @binding(1)` | `storage, read` | `seq_a: array<u32>` — codon-encoded sequences |
| `@group(0) @binding(2)` | `storage, read` | `seq_b: array<u32>` |
| `@group(0) @binding(3)` | `storage, read` | `genetic_code: array<u32>` — codon→amino acid table |
| `@group(0) @binding(4)` | `storage, read_write` | `dn_out: array<f64>` — non-synonymous substitutions |
| `@group(0) @binding(5)` | `storage, read_write` | `ds_out: array<f64>` — synonymous substitutions |
| `@group(0) @binding(6)` | `storage, read_write` | `omega_out: array<f64>` — dN/dS ratio |

- **Entry point:** `@compute @workgroup_size(64) fn main`
- **Dispatch:** `ceil(n_pairs / 64)` — one thread per pair
- **CPU reference:** `bio::dnds::nei_gojobori_pairwise()`
- **f64 transcendentals:** `log` (Jukes-Cantor correction)
- **Polyfill:** **YES** — `ShaderTemplate::for_driver_auto(source, true)`
- **Proposed primitive:** `DnDsBatchF64`

### 2.3 Pangenome & ML Shaders (2)

#### `pangenome_classify.wgsl` — 6/6 GPU checks (Exp058)

| Binding | Type | Purpose |
|---------|------|---------|
| `@group(0) @binding(0)` | `uniform` | `PanParams` (n_genes, n_genomes) |
| `@group(0) @binding(1)` | `storage, read` | `presence: array<u32>` — gene presence matrix (bit-packed) |
| `@group(0) @binding(2)` | `storage, read_write` | `class_out: array<u32>` — 0=unique, 1=accessory, 2=core |
| `@group(0) @binding(3)` | `storage, read_write` | `count_out: array<u32>` — genome count per gene |

- **Entry point:** `@compute @workgroup_size(256) fn main`
- **Dispatch:** `ceil(n_genes / 256)` — one thread per gene
- **CPU reference:** `bio::pangenome::classify_genes()`
- **f64 transcendentals:** None
- **Polyfill:** Not needed
- **Proposed primitive:** `PangenomeClassifyGpu`

#### `rf_batch_inference.wgsl` — 13/13 GPU checks (Exp063)

| Binding | Type | Purpose |
|---------|------|---------|
| `@group(0) @binding(0)` | `uniform` | `RfParams` (n_samples, n_trees, n_nodes, n_features, n_classes) |
| `@group(0) @binding(1)` | `storage, read` | `node_features: array<u32>` — split feature per node |
| `@group(0) @binding(2)` | `storage, read` | `node_thresh: array<f64>` — split threshold per node |
| `@group(0) @binding(3)` | `storage, read` | `node_children: array<u32>` — (left, right) indices |
| `@group(0) @binding(4)` | `storage, read` | `features: array<f64>` — input samples (SoA layout) |
| `@group(0) @binding(5)` | `storage, read_write` | `predictions: array<u32>` — per-tree leaf class |

- **Entry point:** `@compute @workgroup_size(256) fn main`
- **Dispatch:** `ceil(n_samples * n_trees / 256)` — one thread per (sample, tree)
- **CPU reference:** `bio::random_forest::RandomForest::predict()`
- **f64 transcendentals:** None (threshold comparisons only)
- **Polyfill:** Not needed
- **SoA layout:** Node arrays are flat; `features` is SoA (feature × sample). This enables
  coalesced memory access on GPU — all threads reading feature `i` access contiguous memory.
- **Proposed primitive:** `RfBatchInferenceGpu`

---

## Part 3: NVVM Driver Profile Bug (Still Open)

RTX 4070 (Ada Lovelace) NVVM cannot compile native f64 `exp()`, `log()`, `pow()`.
ToadStool's driver profile incorrectly reports `needs_f64_exp_log_workaround() = false`.

**Affected shaders:** `hmm_forward_f64.wgsl`, `batched_qs_ode_rk4_f64.wgsl`, `dnds_batch_f64.wgsl`

**Fix:** `needs_f64_exp_log_workaround()` should return `true` for all Ada Lovelace GPUs
(RTX 4060/4070/4080/4090). Currently, wetSpring forces polyfill via
`ShaderTemplate::for_driver_auto(source, true)` as a workaround.

**Impact if fixed upstream:** 3 of 9 local shaders can drop forced polyfill. All Springs benefit.

---

## Part 4: BarraCUDA CPU Math Extraction

wetSpring consolidated local math into `bio::special` (Phase 15). These functions
duplicate barracuda upstream primitives and are shaped for extraction:

| Local Function | File | BarraCUDA Target | Notes |
|----------------|------|------------------|-------|
| `erf()`, `normal_cdf()` | `bio/special.rs` | `barracuda::special::erf` | `mul_add` chains, FMA-ready |
| `ln_gamma()` | `bio/special.rs` | `barracuda::special::ln_gamma` | Lanczos, Horner form |
| `regularized_gamma_lower()` | `bio/special.rs` | `barracuda::special::regularized_gamma_p` | 1e-15 convergence |
| `integrate_peak()` | `bio/eic.rs` | `barracuda::numerical::trapz` | Trapezoidal integration |

**Consumers:** `bio::dada2` (via `regularized_gamma_lower`), `bio::pangenome` (via `normal_cdf`)

**Proposal:** Add `[features] math = []` to barracuda crate gating `numerical`, `special`,
`stats`, `optimize`, `sample` modules without pulling in wgpu/akida/toadstool-core. This
enables Springs to lean on shared CPU math without forcing the GPU stack.

**Why this matters:** hotSpring always-GPU and accepts the full dependency. wetSpring keeps
barracuda optional (`--features gpu`). Both would benefit from a granular `math` feature.

---

## Part 5: Streaming Pipeline Findings

wetSpring validated the ToadStool dispatch model through 5 experiments (Exp072-076):

### 5.1 Streaming Dispatch (Exp072-073)

`GpuPipelineSession` pre-compiles all pipelines at session init (27ms warmup).
Subsequent dispatches reuse cached pipelines and pooled buffers:

```
Traditional:  CPU → GPU → CPU → GPU → CPU → GPU → CPU  (6 PCIe transfers)
Streaming:    CPU → GPU ───→ GPU ───→ GPU → CPU          (2 PCIe transfers)
```

| Metric | Individual Dispatch | Streaming | Improvement |
|--------|:------------------:|:---------:|:-----------:|
| First FMR call | 110ms | 5µs | **22,000×** (warmup amortized) |
| Pipeline (4 stages) | baseline | 1.27× faster | **21% overhead eliminated** |
| Batch [64] streaming advantage | baseline | faster | Pipeline setup amortized |
| Batch [4096] streaming advantage | baseline | faster | Buffer reuse dominates |

**Key insight for ToadStool:** The 27ms warmup cost vanishes when dispatching multiple
stages. At 4+ stages, streaming always wins. Pipeline caching is the correct default.

### 5.2 Substrate Router (Exp074)

A `SubstrateRouter` dispatches workloads to the optimal hardware:

| Workload Type | Batch Size | Substrate | Rationale |
|---------------|:----------:|-----------|-----------|
| Batch-parallel analytics | ≥64 | GPU | Embarrassingly parallel, GPU dominates |
| Batch-parallel analytics | <64 | CPU | Dispatch overhead exceeds compute |
| Inference (taxonomy, anomaly) | any | NPU → CPU | AKD1000 if detected, else CPU fallback |
| Sequential (NJ, chimera) | any | CPU | Inherently sequential |

Validated: GPU math parity when routing through different substrates. NPU detected via
`/dev/akida0` with graceful CPU fallback. PCIe topology-aware.

### 5.3 Pure GPU Analytics Pipeline (Exp075)

Five bioinformatics stages running entirely on GPU, single upload and readback:

```
Alpha Diversity (FMR) → Bray-Curtis → PCoA (BatchedEighGpu) → Stats (FMR) → Spectral Cosine (GEMM+FMR)
```

| Stage | CPU Check | GPU Check | Parity |
|-------|:---------:|:---------:|:------:|
| Shannon/Simpson/Observed | ✅ | ✅ | ≤ 1e-10 |
| Bray-Curtis matrix | ✅ | ✅ | ≤ 1e-10 |
| PCoA eigendecomposition | ✅ | ✅ | ≤ 1e-6 |
| Variance/correlation | ✅ | ✅ | ≤ 1e-10 |
| Spectral cosine similarity | ✅ | ✅ | ≤ 1e-6 |

Pipeline overhead: **0.1%** — virtually all GPU time is useful compute.
31/31 checks PASS.

### 5.4 Cross-Substrate Pipeline (Exp076)

Heterogeneous pipeline:
`GPU (diversity + Bray-Curtis)` → `NPU (classification, with CPU fallback)` → `CPU (aggregation)`

Per-stage latencies profiled. Math parity maintained across all three substrates.
17/17 checks PASS.

---

## Part 6: PCIe Hardware Topology (Eastgate System)

| Device | PCIe Bus | Gen/Width | Bandwidth | Role |
|--------|----------|-----------|:---------:|------|
| RTX 4070 12GB | 01:00.0 | Gen4 x16 | 32 GB/s | Primary GPU — ToadStool dispatch |
| Titan V 12GB | 05:00.0 | Gen3 x16 | 16 GB/s | Secondary GPU — native f64 (Volta) |
| AKD1000 NPU | 08:00.0 | PCIe 2.0 x1 | 500 MB/s | Inference — taxonomy/anomaly detection |

**P2P DMA paths:**
- GPU ↔ GPU: Via PCIe switch or CPU bridge (IOMMU permitting)
- GPU ↔ NPU: Through CPU bridge (AKD1000 on Gen2 x1)
- GPU → CPU → NPU: Default path; 2 hops

**Absorption opportunity:** ToadStool device discovery could detect this topology
via `lspci -vvv` parsing and pre-compute optimal transfer paths.

---

## Part 7: Lessons Learned (Relevant to ToadStool/BarraCUDA Evolution)

### 7.1 Dispatch Engineering

| Finding | Experiment | Impact |
|---------|:----------:|--------|
| Pipeline caching reduces overhead 38% | Exp068 | Default to cached pipelines |
| Pre-warmed FMR: 5µs vs 110ms cold | Exp072 | Always warm FMR at session init |
| Streaming beats individual at all batch sizes | Exp073 | Streaming should be the default dispatch mode |
| 5-stage pipeline: 0.1% overhead | Exp075 | Pipeline composition has negligible cost |

### 7.2 Ada Lovelace f64 Workarounds

| Issue | Workaround | Proper Fix |
|-------|-----------|------------|
| NVVM f64 `exp`/`log`/`pow` crash | `for_driver_auto(_, true)` forced polyfill | Fix driver profile: `needs_f64_exp_log_workaround() = true` for Ada |
| naga rejects `enable f64;` | Omit from WGSL; `ShaderTemplate` handles | naga upstream fix (low priority) |
| Bare `f32` in f64 builtins | `max(x, f64(0.0))` explicit cast | naga type promotion fix |

### 7.3 Absorption Engineering

| Pattern | Why It Works | Example |
|---------|-------------|---------|
| SoA flat arrays | Coalesced GPU memory access | `rf_batch_inference.wgsl` features buffer |
| `#[repr(C)]` params | WGSL uniform matches Rust layout | All 9 shader param structs |
| Precomputed lookup tables | Avoid transcendentals in shader | `dada2_e_step.wgsl` log_err table |
| Named tolerances | Cross-Spring reproducibility | 32 constants in `tolerances.rs` |
| `mul_add` chains | FMA-friendly on CPU SIMD and GPU | `bio::special::erf` polynomial |
| Provenance headers | Audit trail for all validation | All 73 binaries |
| Zero `unsafe` | Prevents undefined behavior in hot paths | `#![forbid(unsafe_code)]` crate-wide |

### 7.4 What Makes a Shader Absorbable

Checklist for ToadStool team reviewing wetSpring shaders:

1. **WGSL source in-tree** — `barracuda/src/shaders/<name>.wgsl`
2. **Binding layout documented** — group/binding/type/purpose table (see Part 2)
3. **Dispatch geometry documented** — workgroup size, grid formula
4. **CPU reference function identified** — `bio::<module>::<function>()`
5. **Validation binary exists** — `validate_<domain>.rs` or `validate_barracuda_gpu_*.rs`
6. **Tolerance justified** — references `tolerances.rs` constant name
7. **Polyfill status documented** — whether `for_driver_auto(_, true)` is needed
8. **Edge cases tested** — empty input, single element, max batch size

All 9 wetSpring shaders meet criteria 1–7. Edge case coverage varies (documented per shader).

---

## Part 8: Updated Totals

| Metric | v5 (Feb 20) | v6 (Feb 21) | Delta |
|--------|:-----------:|:-----------:|:-----:|
| Experiments | 63 | **76** | +13 |
| CPU checks | 1,241 | **1,291** | +50 |
| GPU checks | 260 | **451** | +191 |
| Total checks | 1,501 | **1,742** | +241 |
| Tests | 552 | **610** | +58 |
| CPU domains | 25 | **25** | — |
| GPU domains (consolidated) | 8 | **11** | +3 |
| Local WGSL shaders | 9 | **9** | — |
| Binaries | 61 | **73** | +12 |
| Source files | — | **151** | — |
| Line coverage | 93.5% | **93.5%** | — |

---

## Part 9: Recommended Absorption Priority

| Priority | Item | Type | Status | Payoff |
|:--------:|------|------|--------|--------|
| **P0** | NVVM driver profile fix | Bug | Open | Unblocks all Ada Lovelace f64 transcendentals |
| **P1** | `HmmBatchForwardF64` | Shader | 13/13 PASS | Batch HMM forward for sequence analysis |
| **P1** | Fix `BatchedOdeRK4F64` | Shader | 7/7 PASS | Unblocks all ODE parameter sweep workloads |
| **P1** | `[features] math = []` | Crate | Proposal | Enables CPU math sharing without GPU stack |
| **P2** | `AniBatchF64` | Shader | 7/7 PASS | Pairwise genomic identity |
| **P2** | `SnpCallingF64` | Shader | 5/5 PASS | Population genomics |
| **P2** | `DnDsBatchF64` | Shader | 9/9 PASS | Evolutionary selection analysis |
| **P2** | `PangenomeClassifyGpu` | Shader | 6/6 PASS | Gene family core/accessory/unique classification |
| **P2** | `RfBatchInferenceGpu` | Shader | 13/13 PASS | ML ensemble at field-scale batch sizes |
| **P3** | `ParallelFilter<T>` | Pattern | 88 checks | Generalizes per-element quality filtering |
| **P3** | `BatchPairReduce<f64>` | Pattern | 88 checks | Generalizes DADA2 E-step and similar pair reductions |
| **P3** | Streaming dispatch default | Design | Exp072-075 | Pipeline session as default (not opt-in) dispatch mode |
| **P3** | PCIe topology detection | Feature | Exp074/076 | `lspci` parsing for device discovery and P2P paths |

---

## Part 10: Reproduction

```bash
cd barracuda

# Full test suite (610 tests)
cargo test --features gpu

# Consolidated CPU proof (1,291 checks across 50 binaries)
cargo run --release --bin validate_barracuda_cpu_full      # 50 checks, 25 domains

# Consolidated GPU proof (451 checks across 18 binaries)
cargo run --features gpu --release --bin validate_barracuda_gpu_full   # 24 checks, 11 domains

# Streaming pipeline validation
cargo run --features gpu --release --bin validate_gpu_streaming_pipeline  # 17 checks
cargo run --features gpu --release --bin validate_dispatch_overhead_proof # 21 checks

# Cross-substrate validation
cargo run --features gpu --release --bin validate_substrate_router        # 20 checks
cargo run --features gpu --release --bin validate_pure_gpu_pipeline       # 31 checks
cargo run --features gpu --release --bin validate_cross_substrate_pipeline # 17 checks

# metalForge full portfolio
cargo run --features gpu --release --bin validate_metalforge_full         # 35 checks

# 25-domain benchmark (Rust vs Python)
cargo run --release --bin benchmark_23_domain_timing
```

---

## Part 11: Files Changed Since v5

| Path | Change |
|------|--------|
| `barracuda/src/bin/validate_barracuda_cpu_full.rs` | New — 25-domain CPU consolidation (Exp070) |
| `barracuda/src/bin/validate_barracuda_gpu_full.rs` | New — 11-domain GPU consolidation (Exp071) |
| `barracuda/src/bin/validate_gpu_streaming_pipeline.rs` | New — streaming proof (Exp072) |
| `barracuda/src/bin/validate_dispatch_overhead_proof.rs` | New — overhead quantification (Exp073) |
| `barracuda/src/bin/validate_substrate_router.rs` | New — substrate router (Exp074) |
| `barracuda/src/bin/validate_pure_gpu_pipeline.rs` | New — 5-stage pure GPU (Exp075) |
| `barracuda/src/bin/validate_cross_substrate_pipeline.rs` | New — cross-substrate (Exp076) |
| `barracuda/Cargo.toml` | Added 7 new `[[bin]]` entries |
| `experiments/070_*.md` through `experiments/076_*.md` | New experiment documentation |
| `barracuda/EVOLUTION_READINESS.md` | Updated with streaming/dispatch/router results |
| `metalForge/ABSORPTION_STRATEGY.md` | Updated with streaming pipeline proof table |
| `metalForge/PCIE_TOPOLOGY.md` | Updated with cross-substrate verification entries |
| `README.md` | Updated metrics (73 binaries, 610 tests, 151 source files) |
| `CONTROL_EXPERIMENT_STATUS.md` | Added Exp070-076 to main table |
| `BENCHMARK_RESULTS.md` | Updated binary counts and reproduction commands |
| `whitePaper/STUDY.md` | Added §3.7 GPU Streaming & Cross-Substrate Validation |
| `whitePaper/README.md` | Added Consolidation, Streaming & Cross-Substrate section |

---

## Archived Handoffs

| Version | Date | Location |
|---------|------|----------|
| v1 | Feb 16 | `archive/handoffs/HANDOFF_WETSPRING_TO_TOADSTOOL_FEB_16_2026.md` |
| v2 | Feb 17 | `archive/handoffs/HANDOFF_SPRINGS_TO_TOADSTOOL_FEB_17_2026.md` |
| v3 | Feb 19 | `archive/handoffs/HANDOFF_WETSPRING_TO_TOADSTOOL_FEB_19_2026_v3.md` |
| v4 | Feb 20 | `archive/handoffs/HANDOFF_V4_FEB_20_2026.md` |
| v5 | Feb 20 | `HANDOFF_WETSPRING_TO_TOADSTOOL_FEB_20_2026.md` (root, historical) |
| **v6** | **Feb 21** | **This document** |

---

*Generated from wetSpring validation pipeline. Last updated: February 21, 2026.*
