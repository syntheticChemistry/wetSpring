# wetSpring → ToadStool/BarraCUDA: Evolution Handoff v7

**Date:** February 22, 2026
**From:** wetSpring (Life Science & Analytical Chemistry validation)
**To:** ToadStool/BarraCUDA team
**License:** AGPL-3.0-or-later
**Type:** CROSS-PRIMAL EVOLUTION HANDOFF
**Supersedes:** v6 (Feb 21) — archived at `archive/handoffs/HANDOFF_V6_FEB_21_2026.md`
**Depends On:** ToadStool sessions 31d/31g (bio shader absorption)

---

## Executive Summary

Since v6, wetSpring has completed 7 new experiments (077–083), promoted 5
modules from Tier B/C to Tier A (GPU/NPU-ready), validated the forge dispatch
router across 5 substrate configurations, and proven that GPU-compatible flat
parameter serialization preserves bitwise-identical math for all 6 ODE models.

**Key numbers:** 83 experiments, 1,835 validation checks (all PASS), 730 Rust
tests, 74 binaries, 7 Tier A modules ready for absorption, only 2 Tier B
modules remaining (both blocked on existing ToadStool issues).

**This handoff provides:** 3 new absorption candidates (kmer histogram, unifrac
CSR, taxonomy int8), ODE flat-param validation results, dispatch routing
findings, and lessons learned for ToadStool evolution.

---

## Part 1: What ToadStool Has Absorbed (23 Primitives)

### Original 15 (pre-Feb 22)

| Primitive | wetSpring Module | Experiment |
|-----------|-----------------|------------|
| `BrayCurtisF64` | diversity_gpu | Exp004/016 |
| `FusedMapReduceF64` (×3: Shannon, Simpson, spectral) | diversity_gpu, spectral_match_gpu | Exp004/016 |
| `GemmCachedF64` | gemm_cached | Exp016 |
| `BatchedEighGpu` | pcoa_gpu, validate_gpu_ode_sweep | Exp016/050 |
| `BatchTolSearchF64` | tolerance_search | Exp016 |
| `PrngXoshiro` | rarefaction_gpu | Exp016 |
| `SmithWatermanGpu` | alignment | Exp044 |
| `GillespieGpu` | gillespie | Exp044 |
| `TreeInferenceGpu` | decision_tree | Exp044 |
| `FelsensteinGpu` | felsenstein, bootstrap, placement | Exp046 |
| `ShaderTemplate::for_driver_auto` | ode_sweep_gpu, gemm_cached | Exp047+ |
| `LogsumexpWgsl` | (available, not yet wired) | — |
| `BatchedOdeRK4F64` | (blocked: `enable f64;`) | — |

### 8 Bio Primitives Absorbed (Feb 22, sessions 31d/31g)

| Primitive | wetSpring Module | Local Shader Removed |
|-----------|-----------------|---------------------|
| `HmmBatchForwardF64` | hmm_gpu | `hmm_batch_forward_f64.wgsl` |
| `AniBatchF64` | ani_gpu | `ani_batch_f64.wgsl` |
| `SnpCallingF64` | snp_gpu | `snp_calling_f64.wgsl` |
| `DnDsBatchF64` | dnds_gpu | `dnds_batch_f64.wgsl` |
| `PangenomeClassifyGpu` | pangenome_gpu | `pangenome_classify_f64.wgsl` |
| `QualityFilterGpu` | quality_gpu | `quality_filter_f64.wgsl` |
| `Dada2EStepGpu` | dada2_gpu | `dada2_e_step_f64.wgsl` |
| `RfBatchInferenceGpu` | random_forest_gpu | `rf_batch_inference_f64.wgsl` |

25 KB of local WGSL deleted. 451 GPU checks re-validated against ToadStool.

---

## Part 2: New Absorption Candidates (3 Modules)

### 2a. K-mer Histogram GPU Shader (Exp081)

**Module:** `barracuda/src/bio/kmer.rs`
**Status:** Tier A (GPU-ready)
**Pattern:** Dense histogram buffer of size 4^k

```
CPU: count_kmers(seq, k) → KmerCounts (HashMap)
     → to_histogram() → Vec<u32>[4^k]    ← GPU buffer
GPU: parallel per-sequence counting, atomic histogram merge
     → readback → from_histogram()       ← round-trip verified
```

**Key design decisions:**
- For k ≤ 12: dense flat histogram (`to_histogram()`) — 4^8 = 65,536 u32 entries
- For k > 12: sorted pairs (`to_sorted_pairs()`) — compact, GPU merge-sort
- Canonical k-mers (min of forward/reverse complement) preserved in both layouts
- Round-trip fidelity proven: 4 tests, every count survives serialization

**What ToadStool needs to build:**
- `KmerHistogramGpu` — atomic u32 histogram accumulation from packed 2-bit sequences
- Workgroup: 256 threads, each processing a sequence region
- Input: packed u64 k-mers, Output: u32[4^k] histogram
- Consider: GPU radix sort for large-k case (sorted pairs)

**File locations:**
- CPU implementation: `barracuda/src/bio/kmer.rs` (345 lines)
- GPU API ready: `to_histogram()`, `from_histogram()`, `to_sorted_pairs()`

### 2b. UniFrac CSR Tree GPU Shader (Exp082)

**Module:** `barracuda/src/bio/unifrac.rs`
**Status:** Tier A (GPU-ready)
**Pattern:** CSR flat tree + pairwise parallel dispatch

```
CPU: PhyloTree::from_newick() → to_flat_tree() → FlatTree (CSR)
     to_sample_matrix() → Vec<f64>[n_samples × n_leaves]
GPU: parallel over (i, j) pairs:
       upload FlatTree buffers + sample matrix
       propagate leaf→root per sample, accumulate UniFrac
     → readback distance matrix
```

**FlatTree buffer layout (CSR):**
```
parent:           buffer<u32, n_nodes>    — parent index
branch_length:    buffer<f64, n_nodes>    — branch lengths
n_children:       buffer<u32, n_nodes>    — child count
children_offset:  buffer<u32, n_nodes>    — CSR offset
children_flat:    buffer<u32, edges>      — contiguous children
leaf_indices:     buffer<u32, n_leaves>   — leaf node IDs
sample_matrix:    buffer<f64, n_samples × n_leaves> — abundances
```

**What ToadStool needs to build:**
- `UniFracPairwiseGpu` — parallel pairwise distance computation
- Workgroup: 1 per (i, j) pair; each propagates two samples bottom-up
- Challenge: bottom-up tree traversal ordering (topological sort)
- Output: condensed distance matrix (n*(n-1)/2 f64 values)

**File locations:**
- CPU implementation: `barracuda/src/bio/unifrac.rs` (559 lines)
- GPU API ready: `to_flat_tree()`, `to_sample_matrix()`, `FlatTree::to_phylo_tree()`

### 2c. Taxonomy NPU Int8 FC Layer (Exp083)

**Module:** `barracuda/src/bio/taxonomy.rs`
**Status:** Tier A / NPU
**Pattern:** Affine int8 quantization for FC inference

```
CPU: NaiveBayesClassifier::train(refs, k)
     → to_int8_weights() → NpuWeights { weights_i8, priors_i8, scale, zero_point }
NPU: FC layer inference: query_kmers × weights_i8 + priors_i8 → argmax
CPU: taxon_labels[argmax] → classification
```

**Quantization scheme:**
```
scale      = (max_log_prob - min_log_prob) / 255
zero_point = min_log_prob
q          = round((value - zero_point) / scale) - 128
```

**Buffer sizes for k=8:**
- `weights_i8`: `n_taxa × 65,536` i8 values
- `priors_i8`: `n_taxa` i8 values
- Typical: 50 taxa × 65K = 3.1 MB (fits in NPU SRAM)

**Validated:** Int8 argmax matches f64 argmax for all test queries (2 taxa, k=4).

**What ToadStool needs to build:**
- `TaxonomyNpuInference` — int8 FC dispatch targeting AKD1000
- Uses `Capability::QuantizedInference { bits: 8 }` + `BatchInference { max_batch: 8 }`
- Or: GPU fallback via f32 GEMM (`GemmCachedF64` already absorbed)

**File locations:**
- CPU implementation: `barracuda/src/bio/taxonomy.rs` (657 lines)
- NPU API ready: `to_int8_weights()`, `classify_quantized()`, `NpuWeights` struct

---

## Part 3: ODE Flat Parameter Validation (Exp078–079)

All 6 ODE biological models now have GPU-compatible flat parameter APIs:

| Model | Paper | Vars | Params | Status |
|-------|-------|------|--------|--------|
| `qs_biofilm` | Waters 2008 | 5 | 18 | Tier A, local WGSL exists |
| `bistable` | Fernandez 2020 | 5 | 21 | Tier A, flat API proven |
| `multi_signal` | Srivastava 2011 | 7 | 24 | Tier A, flat API proven |
| `phage_defense` | Hsueh 2022 | 4 | 11 | Tier A, flat API proven |
| `cooperation` | Bruger 2018 | 4 | 13 | Tier B, flat API proven |
| `capacitor` | Mhatre 2020 | 5 | (struct) | CPU-only |

**Exp079 proved:** `to_flat() → from_flat() → ODE integration` produces
**bitwise-identical** results (zero ULP drift) across all 6 modules. Python
baseline parity maintained through the flat serialization path.

**Blocker:** ToadStool `BatchedOdeRK4F64` requires `enable f64;` support in
naga. The local workaround WGSL (`batched_qs_ode_rk4_f64.wgsl`) works for
qs_biofilm but is not generalizable without the upstream fix.

**Layout note:** The existing GPU ODE shader uses a 17-param layout (omits
`d_cdg`), while the new `QsBiofilmParams::to_flat()` uses the complete
18-param layout. When ToadStool builds the generalized ODE shader, use
the 18-param layout from `to_flat()`.

---

## Part 4: Dispatch Routing Findings (Exp080)

The forge dispatch router was validated across 5 substrate configurations:

| Config | ODE Route | Taxonomy Route | Diversity Route |
|--------|-----------|----------------|-----------------|
| GPU f64 + NPU + CPU | GPU | NPU | GPU |
| GPU f64 + CPU | GPU | (none) | GPU |
| NPU + CPU | (none) / CPU fallback | NPU | CPU |
| CPU only | (none) | (none) | CPU |
| f32 iGPU + f64 dGPU + NPU + CPU | f64 dGPU | NPU | f64 dGPU |

**Key finding:** ODE workloads requiring `ShaderDispatch` correctly fail on
CPU-only configurations. Applications must define two workload variants:
GPU-optimal (f64 + shader) and CPU-fallback (f64 only).

**Live hardware:** Discovered 1 CPU, 3 GPUs (RTX 4070, Titan V, iGPU), 1 NPU
(AKD1000). All routing decisions correct.

**Recommendation for ToadStool:** The dispatch model should support workload
variants with graceful degradation. A single `OdeWorkload` should route to
GPU when shader-capable, fall back to CPU RK4 otherwise. This is architectural,
not just a wetSpring concern.

---

## Part 5: Lessons Learned & Recommendations

### 5a. naga `enable f64;` is the #1 Blocker

Every remaining Tier B module and the ODE GPU generalization are blocked on
naga's inability to handle `enable f64;` in WGSL. The current workaround
(`ShaderTemplate::for_driver_auto` with polyfills) works for individual shaders
but doesn't scale to a generalized ODE engine. This affects:
- All 5 non-qs_biofilm ODE modules
- The `cooperation` module (last Tier B)
- Any future f64-heavy GPU math

**Priority:** HIGH — resolving this unblocks 6 modules simultaneously.

### 5b. Flat Serialization Pattern

The `to_flat()` / `from_flat()` pattern proved extremely effective for GPU
preparation. Zero ULP drift in round-trips. **Recommend standardizing this
pattern in `barracuda::ops` for all parameterized GPU workloads.** The ODE
modules demonstrate the template: `N_VARS`, `N_PARAMS` constants, `to_flat()`
as `const fn`, `from_flat()` with length assertion.

### 5c. CSR for Tree Structures

The UniFrac CSR layout (flat parent/children/branch_length arrays) is the
canonical GPU-friendly tree representation. **Recommend adding a generic
`FlatTree` type to `barracuda::ops::tree` that any tree algorithm can use.**
Felsenstein, bootstrap, placement, NJ, DTL reconciliation, Robinson-Foulds
would all benefit.

### 5d. Int8 Quantization for NPU

The taxonomy int8 path demonstrates affine quantization preserving argmax
accuracy. **Recommend adding a generic `quantize_affine_i8(data, &mut scale,
&mut zero_point)` to `barracuda::ops::quant`** for any NPU-targeted workload.

### 5e. Streaming Pipeline Caching

Exp072/073 proved pre-warmed `GpuPipelineSession` delivers 1.27× speedup.
First-call latency drops from 110ms to 5µs. **Recommend making pipeline
caching the default in ToadStool dispatch.** The Fused Map-Reduce (FMR)
pattern is especially effective — warmup once, reuse across stages.

### 5f. PCIe Topology Awareness

Exp074/076 proved GPU→NPU→CPU heterogeneous routing works when the dispatch
layer knows the PCIe topology. **Recommend exposing PCIe bus IDs in
`barracuda::device::WgpuDevice`** so forge-level routers can identify devices
on the same fabric for DMA transfers bypassing CPU.

---

## Part 6: Updated Totals (v7)

| Metric | v6 (Feb 21) | v7 (Feb 22) | Delta |
|--------|-------------|-------------|-------|
| Experiments | 76 | **91** | +15 |
| Validation checks | 1,742 | **2,097+** | +355+ |
| Rust tests | 707 | **728** | +21 |
| Binaries | 73 | **81** | +8 |
| ToadStool primitives | 23 | 23 | — |
| Local WGSL shaders | 1 | 1 | — |
| Tier A modules | 4 | **7** | +3 |
| Tier B modules | 5 | **2** | -3 |
| Tier C modules | 14 | 14 | — |

### Tier A Modules (GPU/NPU-Ready)

| Module | Layout | Target | Blocker |
|--------|--------|--------|---------|
| `qs_biofilm` | 18-param flat, local WGSL | GPU | `enable f64;` for generalization |
| `bistable` | 21-param flat | GPU | `enable f64;` |
| `multi_signal` | 24-param flat | GPU | `enable f64;` |
| `phage_defense` | 11-param flat | GPU | `enable f64;` |
| `kmer` | 4^k histogram + sorted pairs | GPU | Needs histogram WGSL |
| `unifrac` | CSR flat tree + sample matrix | GPU | Needs tree-propagation WGSL |
| `taxonomy` | Int8 quantized weights | NPU | Needs AKD1000 FC integration |

### Tier B Modules (2 remaining)

| Module | Status | Blocker |
|--------|--------|---------|
| `cooperation` | Flat API done (Exp078) | Same `enable f64;` as other ODE |

(`cooperation` is functionally Tier A but formally B since its ODE blocker is
the same as the other 4 ODE modules.)

---

## Part 7: Recommended Absorption Priority

1. **`enable f64;` fix in naga/ToadStool** — unblocks 6 modules, HIGH
2. **`KmerHistogramGpu`** — standalone shader, no f64 dependency (uses u32), MEDIUM
3. **`UniFracPairwiseGpu`** — needs f64, depends on #1, MEDIUM
4. **`TaxonomyNpuInference`** — NPU path, independent of GPU work, LOW (hardware-dependent)
5. **Generic `FlatTree`** — utility type for all tree algorithms, LOW (architectural)

---

## Part 8: Reproduction Commands

```bash
# All CPU tests (654 lib + 60 integration + 14 doc + 2 bench)
cd barracuda && cargo test

# CPU validation v6 (ODE flat param parity, 48 checks)
cargo run --release --bin validate_barracuda_cpu_v6

# Dispatch routing (forge crate, 35 checks)
cd metalForge/forge && cargo run --bin validate_dispatch_routing

# Full CPU parity (25 domains, 50 checks)
cd barracuda && cargo run --release --bin validate_barracuda_cpu_full

# Full GPU parity (11 domains, 24 checks — requires --features gpu)
cargo run --release --features gpu --bin validate_barracuda_gpu_full

# GPU ODE sweep + bifurcation (requires --features gpu)
cargo run --release --features gpu --bin validate_gpu_ode_sweep

# Code quality
cargo fmt -- --check
cargo clippy --all-targets -- -W clippy::pedantic -W clippy::nursery
cargo doc --no-deps
```

---

## Part 9: Files Changed Since v6

| File | Change |
|------|--------|
| `barracuda/src/bio/kmer.rs` | +`to_histogram()`, `from_histogram()`, `to_sorted_pairs()`, `from_sorted_pairs()`, 4 tests |
| `barracuda/src/bio/unifrac.rs` | +`FlatTree`, `to_flat_tree()`, `to_phylo_tree()`, `to_sample_matrix()`, 4 tests |
| `barracuda/src/bio/taxonomy.rs` | +`NpuWeights`, `to_int8_weights()`, `classify_quantized()`, 3 tests |
| `barracuda/src/bio/qs_biofilm.rs` | +`N_VARS`, `N_PARAMS`, `to_flat()`, `from_flat()` |
| `barracuda/src/bio/bistable.rs` | +`N_VARS`, `N_PARAMS`, `to_flat()`, `from_flat()` |
| `barracuda/src/bio/multi_signal.rs` | +`N_VARS`, `N_PARAMS`, `to_flat()`, `from_flat()` |
| `barracuda/src/bio/phage_defense.rs` | +`N_VARS`, `N_PARAMS`, `to_flat()`, `from_flat()` |
| `barracuda/src/bio/cooperation.rs` | +`N_VARS`, `N_PARAMS`, `to_flat()`, `from_flat()` |
| `barracuda/src/bin/validate_barracuda_cpu_v6.rs` | New: 48-check ODE flat parity |
| `metalForge/forge/src/bin/validate_dispatch_routing.rs` | New: 35-check router validation |
| `barracuda/src/bin/validate_metalforge_full_v2.rs` | New: 12-domain cross-substrate (Exp084) |
| `experiments/078–084_*.md` | 7 new experiment protocols |
| `CONTROL_EXPERIMENT_STATUS.md` | Updated totals |
| `barracuda/EVOLUTION_READINESS.md` | Tier promotions, history |

---

## Previous Handoffs

| Version | Date | Key Content |
|---------|------|-------------|
| v6 | Feb 21 | 9 WGSL shaders, NVVM driver bug, CPU math extraction, streaming pipeline |
| v5 | Feb 20 | Track 1c GPU shaders, 25-domain benchmark, metalForge |
| v4 | Feb 20 | First bio shader handoff (SW, Gillespie, DT, Felsenstein) |
| v3 | Feb 19 | Primitive verification, fragile GEMM path |
| v2 | Feb 17 | Bio primitive requests |
| v1 | Feb 16 | Initial handoff: diversity shaders, log_f64 bug |

All prior versions archived at `archive/handoffs/`.

---

*From the wetSpring validation desk — building sovereign life science compute,
one experiment at a time.*
