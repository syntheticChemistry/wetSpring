# Handoff v4: wetSpring → ToadStool / BarraCUDA Team

**Date:** February 20, 2026
**From:** wetSpring (ecoPrimals — Life Science & Analytical Chemistry)
**To:** ToadStool / BarraCUDA core team
**License:** AGPL-3.0-or-later
**Supersedes:** All prior handoffs (Feb 16, 17, 19 v1/v2/v3)
**ToadStool reviewed:** barracuda v0.2.0 at commit cce8fe7c (bio absorption confirmed)

---

## Executive Summary

This handoff follows a comprehensive review of ToadStool's evolution since
the Feb 19 v3 handoff. ToadStool has absorbed wetSpring's findings and
built new primitives that address several of our original P0-P1 requests.

**What changed since v3:**

1. **ToadStool absorption confirmed** — Bray-Curtis f64, log_f64 fix, and
   all 11 originally-requested primitives are present and working
2. **P0 (ODE) addressed** — `numerical::rk45` (CPU) + `ops::rk_stage` (GPU)
   + `ops::md::integrators::rk4` provide RK4/RK45 integration
3. **P1 (PRNG) addressed** — `ops::prng_xoshiro_wgsl` and
   `ops::random_uniform_wgsl` provide GPU random numbers
4. **P1 (LogSumExp) addressed** — `ops::logsumexp_wgsl` provides numerically
   stable log-sum-exp reduction
5. **No breaking API changes** — wetSpring compiles clean against barracuda
   v0.2.0 (465 tests, 0 clippy warnings, GPU features compile)
6. **3 of 6 original requests are now covered** — remaining 3 detailed below

**wetSpring totals:** 1,257 validation checks (1,035 CPU + 222 GPU), 465 tests, 50 experiments

---

## ToadStool Evolution Review

### What ToadStool Has Absorbed

| wetSpring Finding | ToadStool Action | Date |
|-------------------|-----------------|------|
| Bray-Curtis f64 shader | `ops::bray_curtis_f64::BrayCurtisF64` | Feb 16 |
| log_f64 atanh coefficient bug | Halved coefficients in `math_f64.wgsl` | Feb 16 |
| Shannon/Simpson via FMR | `FusedMapReduceF64::shannon_entropy()`, `simpson_index()` | Feb 16 |
| PCoA pipeline pattern | `BrayCurtisF64` → `BatchedEighGpu` | Feb 16 |
| f64 precision requirements | `ShaderTemplate::for_driver_auto()` exp/log workaround | Feb 16 |

### What ToadStool Has Built (addressing wetSpring requests)

| Original Request | Priority | ToadStool Solution | Notes |
|-----------------|----------|-------------------|-------|
| BatchedRK4F64 | **P0** | `numerical::rk45::rk45_solve` (CPU adaptive), `ops::rk_stage::RkIntegrator` (GPU hybrid), `ops::md::integrators::rk4::Rk4` (GPU fixed-step) | CPU path has `Rk45Config` for tolerances; GPU path uses `rk_stage_f64.wgsl`; falls back to CPU for N<128 state dims |
| GPU PRNG | **P1** | `ops::prng_xoshiro_wgsl::PrngXoshiro`, `ops::random_uniform_wgsl` | xoshiro256++ in WGSL; supports seeded initialization; uniform distribution |
| LogSumExp / HMM | **P1** | `ops::logsumexp_wgsl` | Numerically stable log-sum-exp reduction for HMM forward-backward |

### Absorbed in cce8fe7c (Feb 20 Session 21)

All 3 remaining requests were absorbed in ToadStool commit `cce8fe7c`:

| Request | Priority | Status | Module |
|---------|----------|--------|--------|
| Smith-Waterman alignment | **P1** | **ABSORBED** — 3/3 GPU checks | `ops::bio::smith_waterman::SmithWatermanGpu` |
| Decision Tree GPU inference | **P2** | **ABSORBED** — 6/6 GPU parity | `ops::bio::tree_inference::TreeInferenceGpu` |
| Felsenstein phylogenetic likelihood | **P2** | **ABSORBED** — available | `ops::bio::felsenstein::FelsensteinGpu` |
| Gillespie SSA GPU | **P1** | **ABSORBED** — NVVM driver skip | `ops::bio::gillespie::GillespieGpu` |
| GemmF64::WGSL public constant | **P0** | **ABSORBED** — include_str! eliminated | `ops::linalg::gemm_f64::GemmF64::WGSL` |

**All 6 original requests are now addressed. Zero outstanding primitive requests.**

---

## Request Status Update

### P0: Batched RK4/RK45 — ADDRESSED

ToadStool now provides three integration paths:

1. **`numerical::rk45::rk45_solve`** — Pure-CPU adaptive Dormand-Prince with
   configurable tolerances (`Rk45Config`). Suitable for small ODE systems.

2. **`ops::rk_stage::RkIntegrator::integrate_fixed`** — Fixed-step RK4 with
   hybrid CPU/GPU execution (CPU for N<128, GPU for larger state vectors).

3. **`ops::md::integrators::rk4::Rk4`** — GPU-native RK4 with position +
   velocity + acceleration tensors. MD-focused but generalizable.

**wetSpring integration path:** wetSpring's `bio::ode::rk4_integrate` remains
the sovereign CPU reference (validated against Python scipy). For GPU parameter
sweeps of the Waters 2008 model, wetSpring should use `RkIntegrator::integrate_fixed`
with the ODE right-hand-side evaluated on CPU and state updates parallelized on GPU.

**What's still needed:** The GPU `RkIntegrator` evaluates f(t,y) on CPU and
parallelizes state updates on GPU. For true batched parameter sweeps
(N independent ODE instances with different rate constants), a
`BatchedRK4F64` wrapper that manages N parallel `RkIntegrator` sessions
would be valuable. The building blocks exist; the orchestration layer is
the remaining work.

### P1: GPU PRNG — ADDRESSED

ToadStool's `PrngXoshiro` provides GPU-resident random number generation
via xoshiro256++. This is a better choice than wetSpring's sovereign
Lehmer LCG (`Lcg64`) for GPU work because xoshiro has better statistical
properties and the WGSL shader is already written.

**Gillespie GPU integration path:**
1. Use `PrngXoshiro` to generate uniform random numbers per trajectory
2. Derive exponential variates via inverse CDF: `tau = -ln(u) / rate`
3. Reaction selection via `SumReduceF64` prefix sum on propensities
4. State update: one thread per trajectory, independent SSA

**What's still needed:** The SSA orchestration shader that combines PRNG +
exponential sampling + reaction selection + state update into a single
per-trajectory kernel. The building blocks are all present in ToadStool.

Proposed shader structure (for ToadStool team):

```wgsl
// gillespie_ssa_f64.wgsl — parallel trajectory SSA
//
// Bindings:
//   @group(0) @binding(0) var<uniform> params: GillespieParams;
//   @group(0) @binding(1) var<storage, read> stoich: array<i32>;  // R×S
//   @group(0) @binding(2) var<storage, read_write> states: array<f64>;  // T×S
//   @group(0) @binding(3) var<storage, read_write> prng_state: array<u64>; // T×4
//   @group(0) @binding(4) var<storage, read_write> times: array<f64>;  // T
//
// Each thread runs one complete trajectory to t_max.
// T = n_trajectories, S = n_species, R = n_reactions
```

### P1: LogSumExp — ADDRESSED

`ops::logsumexp_wgsl` provides exactly what wetSpring requested for HMM
forward-backward. No further action needed from ToadStool.

**wetSpring integration path:** When `bio::hmm` is implemented for
Exp019 Phase 3 (PhyloNet-HMM, Liu 2014), use `LogSumExp` for the
forward and backward probability recursions.

---

## Remaining Shader Designs

### Shader Design 1: Smith-Waterman Banded Alignment (P1)

**Purpose:** Banded local alignment with affine gap penalties for
SATe/phylogenetic sequence alignment (Liu 2009, Exp019 Phase 2).

**Design:**
- Anti-diagonal wavefront parallelism: each thread computes one cell in
  the anti-diagonal, with workgroup barriers between diagonals
- Band width parameter `w` limits memory to O(n×w) instead of O(n²)
- Affine gaps: three matrices (H, E, F) tracked per cell
- f64 for score accumulation (integer scores would also work)

```wgsl
// smith_waterman_banded_f64.wgsl
//
// Bindings:
//   @group(0) @binding(0) var<uniform> params: SwParams;
//     - n: u32           // query length
//     - m: u32           // target length
//     - band_width: u32  // band half-width
//     - gap_open: f64    // affine gap open penalty
//     - gap_extend: f64  // affine gap extend penalty
//   @group(0) @binding(1) var<storage, read> query: array<u32>;
//   @group(0) @binding(2) var<storage, read> target: array<u32>;
//   @group(0) @binding(3) var<storage, read> subst_matrix: array<f64>;  // 4×4
//   @group(0) @binding(4) var<storage, read_write> h_matrix: array<f64>;
//   @group(0) @binding(5) var<storage, read_write> e_matrix: array<f64>;
//   @group(0) @binding(6) var<storage, read_write> f_matrix: array<f64>;
//   @group(0) @binding(7) var<storage, read_write> result: SwResult;
//
// Dispatch: one workgroup per anti-diagonal, threads within band.
// Synchronize via workgroupBarrier() between diagonals.
// Traceback on CPU (branching-heavy, not GPU-friendly).
```

**Batching:** For phylogenetic pipelines, batch B alignment pairs:
extend bindings with pair offsets, dispatch B×diag workgroups.

### Shader Design 2: Decision Tree / Random Forest GPU Inference (P2)

**Purpose:** Batch inference for N samples through M decision trees,
enabling real-time PFAS monitoring at field scale.

**Design:**
- Each thread processes one (sample, tree) pair
- Tree structure is flat arrays: `feature_indices`, `thresholds`,
  `left_children`, `right_children`, `predictions`
- Traversal: follow left/right until leaf (max depth ~20, tight loop)
- Random forest: dispatch N×M threads, reduce predictions per sample

```wgsl
// tree_inference_f64.wgsl
//
// Bindings:
//   @group(0) @binding(0) var<uniform> params: TreeParams;
//     - n_samples: u32
//     - n_nodes: u32
//     - n_features: u32
//     - n_trees: u32     // 1 for single tree, M for forest
//   @group(0) @binding(1) var<storage, read> samples: array<f64>;      // N×F
//   @group(0) @binding(2) var<storage, read> feature_idx: array<u32>;  // nodes
//   @group(0) @binding(3) var<storage, read> thresholds: array<f64>;   // nodes
//   @group(0) @binding(4) var<storage, read> left_child: array<i32>;   // nodes
//   @group(0) @binding(5) var<storage, read> right_child: array<i32>;  // nodes
//   @group(0) @binding(6) var<storage, read> predictions: array<u32>;  // nodes
//   @group(0) @binding(7) var<storage, read_write> output: array<u32>; // N×M
//
// Thread (sample_id, tree_id):
//   node = 0
//   while left_child[node] >= 0:
//     if samples[sample_id * n_features + feature_idx[node]] <= thresholds[node]:
//       node = left_child[node]
//     else:
//       node = right_child[node]
//   output[sample_id * n_trees + tree_id] = predictions[node]
//
// Random forest majority vote: separate reduction dispatch or CPU.
```

wetSpring's `bio::decision_tree` has the exact flat-array structure
(`from_arrays`) that maps directly to these GPU buffers. The sklearn
JSON export (`experiments/results/008_pfas_ml/decision_tree_exported.json`)
contains 65 nodes × 28 features, validated on 744 samples with 100% parity.

### Shader Design 3: Felsenstein Phylogenetic Likelihood (P2)

**Purpose:** Postorder tree traversal computing per-site likelihoods for
maximum-likelihood phylogenetics.

**Design:**
- Level-order parallelism: process all nodes at the same depth simultaneously
- Each thread handles one (site, node) pair
- Conditional likelihoods propagated from leaves to root
- Tip nodes: identity mapping from observed nucleotide
- Internal nodes: matrix exponentiation × child likelihoods

```wgsl
// felsenstein_f64.wgsl
//
// Multi-dispatch: one dispatch per tree level (bottom-up).
//
// Bindings:
//   @group(0) @binding(0) var<uniform> params: FelsensteinParams;
//     - n_sites: u32
//     - n_nodes_this_level: u32
//     - n_states: u32    // 4 for DNA, 20 for protein
//   @group(0) @binding(1) var<storage, read> node_ids: array<u32>;
//   @group(0) @binding(2) var<storage, read> left_child: array<i32>;
//   @group(0) @binding(3) var<storage, read> right_child: array<i32>;
//   @group(0) @binding(4) var<storage, read> branch_lengths: array<f64>;
//   @group(0) @binding(5) var<storage, read> transition_probs: array<f64>;  // 4×4 per branch
//   @group(0) @binding(6) var<storage, read_write> likelihoods: array<f64>; // N×S×4
//
// Thread (site, node):
//   For each state s:
//     L_left = sum_j P(s->j, t_left) * likelihood[left_child][site][j]
//     L_right = sum_k P(s->k, t_right) * likelihood[right_child][site][k]
//     likelihood[node][site][s] = L_left * L_right
//
// Root: site_likelihood = sum_s pi_s * likelihood[root][site][s]
// Total: sum over sites of log(site_likelihood) — use LogSumExp for stability
```

**Key:** The matrix-vector product per node per site is a 4×4 × 4 operation
(DNA). For protein (20×20), this becomes the bottleneck and benefits
significantly from GPU parallelism. `LogSumExp` (now available in ToadStool)
handles the final site-likelihood summation.

---

## Local Extensions Absorption Status

| Local Extension | Location | ToadStool Status | Action |
|----------------|----------|-----------------|--------|
| `QualityFilterCached` | `quality_gpu.rs` | Not absorbed | Keep local; propose `ParallelFilter<T>` |
| `Dada2Gpu` (E-step) | `dada2_gpu.rs` | Not absorbed | Keep local; propose `BatchPairReduce<f64>` |
| `GemmCached` pipeline | `gemm_cached.rs` | Not absorbed | Keep local; `include_str!` path still fragile |
| `StreamingGpu` | `streaming_gpu.rs` | Not absorbed; `StatefulPipeline` is related | Keep local; evaluate `StatefulPipeline` fit |

**Note on `GemmCached`:** The `include_str!` cross-crate path
(`../../../../phase1/toadstool/crates/barracuda/src/shaders/linalg/gemm_f64.wgsl`)
is the highest-risk coupling in wetSpring. ToadStool's `GemmF64` could resolve
this by either:
1. Exposing `pub const GEMM_F64_WGSL: &str` from `barracuda::shaders::linalg`
2. Adding `GemmF64::new(device)` → `GemmF64::execute()` with cached pipeline

Option 2 is preferred — it would let wetSpring delete `gemm_cached.rs` entirely.

---

## Proven Claims (unchanged from v3)

| Claim | Evidence | Files |
|-------|----------|-------|
| Rust matches Python for 16S pipeline | 1,035/1,035 CPU checks | 27 validation binaries |
| GPU matches CPU | 140/140 checks, 88/88 pipeline parity | `validate_diversity_gpu`, `validate_16s_pipeline_gpu`, `validate_barracuda_gpu_v3` |
| 926× spectral cosine GPU speedup | Benchmark with 2048 spectra | `benchmark_cpu_gpu` |
| RK4 ODE matches scipy | 16/16 checks, 4 biological scenarios | `validate_qs_ode` |
| Gillespie SSA converges | 13/13 checks, mean within 0.2% | `validate_gillespie` |
| RF distance matches dendropy | 23/23 checks, 10 tree topologies | `validate_rf_distance` |
| Decision tree 100% parity | 7/7 checks, 744/744 predictions | `validate_pfas_decision_tree` |
| Zero unsafe, zero clippy | Full audit | `cargo clippy --pedantic --nursery` |

---

## Reproduction Commands

```bash
cd barracuda

# Full test suite (465 tests)
cargo test

# Lint clean
cargo fmt --check
cargo clippy -- -W clippy::pedantic -W clippy::nursery

# All CPU validation binaries (1,035 checks)
for bin in validate_16s_pipeline validate_algae_16s validate_diversity \
  validate_extended_algae validate_fastq validate_features validate_peaks \
  validate_pfas validate_pfas_library validate_public_benchmarks \
  validate_voc_peaks validate_qs_ode validate_gillespie \
  validate_rf_distance validate_newick_parse validate_pfas_decision_tree \
  validate_bistable validate_multi_signal validate_cooperation validate_hmm \
  validate_capacitor validate_alignment validate_felsenstein validate_phage_defense \
  validate_bootstrap validate_placement validate_barracuda_cpu \
  validate_barracuda_cpu_v2 validate_barracuda_cpu_v3 validate_phynetpy_rf validate_phylohmm validate_sate_pipeline \
  validate_algae_timeseries validate_bloom_surveillance validate_epa_pfas_ml validate_massbank_spectral; do
  cargo run --bin "$bin"
done

# GPU validation (requires SHADER_F64 GPU)
cargo run --features gpu --bin validate_diversity_gpu
cargo run --features gpu --bin validate_16s_pipeline_gpu
cargo run --features gpu --bin validate_barracuda_gpu_v3
```

---

## Recommended ToadStool Evolution Priority (Updated)

| Priority | Primitive | Status | Payoff |
|----------|-----------|--------|--------|
| ~~**P0**~~ | ~~BatchedRK4F64~~ | **ADDRESSED** | `ops::rk_stage`, `numerical::rk45` |
| ~~**P1**~~ | ~~GPU PRNG~~ | **ADDRESSED** | `ops::prng_xoshiro_wgsl` |
| ~~**P1**~~ | ~~LogSumExp~~ | **ADDRESSED** | `ops::logsumexp_wgsl` |
| **P1** | Smith-Waterman banded | NOT YET | Phylogenetic alignment (see §Shader Design 1) |
| **P1** | Gillespie SSA orchestration | BUILDING BLOCKS READY | PRNG + reduce exist; need SSA kernel |
| **P2** | Tree/RF inference GPU | NOT YET | ML at field scale (see §Shader Design 2) |
| **P2** | Felsenstein pruning | NOT YET | ML phylogenetics (see §Shader Design 3) |
| **Absorb** | `GemmF64::cached()` | NOT YET | Eliminates `include_str!` fragility |
| **Absorb** | `ParallelFilter<T>` | NOT YET | Generalizes `QualityFilterCached` |
| **Absorb** | `BatchPairReduce<f64>` | NOT YET | Generalizes `Dada2Gpu` E-step |

---

## What's Left in wetSpring

| Item | Status | Blocking? |
|------|--------|-----------|
| Exp019 Phases 2-4 (gene tree RF, PhyloNet-HMM, SATe 16S) | **COMPLETE** (Exp036–038) | — |
| Exp008 full RF ensemble in Rust | Future work | No |
| Wire `RkIntegrator` for GPU ODE sweeps | ToadStool primitive ready | No |
| Wire `PrngXoshiro` for GPU Gillespie | SSA orchestration needed | No |
| Tolerance centralization (inline → tolerances.rs) | Debt | No |

Nothing blocks ToadStool evolution. All Rust CPU modules are validated and
ready for GPU promotion when the corresponding primitives and orchestration exist.

---

## Session Update: Feb 20, 2026 (Evening)

### New Experiments (036-042)

| Exp | Paper/Proxy | Domain | Checks |
|-----|-------------|--------|--------|
| 036 | PhyNetPy RF distances (Exp019 Phase 2) | Gene tree RF validation | 15 |
| 037 | PhyloNet-HMM discordance (Exp019 Phase 3) | Introgression detection | 10 |
| 038 | SATe pipeline benchmark (Exp019 Phase 4) | Divide-and-conquer alignment | 17 |
| 039 | Algal pond time-series (Cahill proxy) | Time-series anomaly | 11 |
| 040 | Bloom surveillance (Smallwood proxy) | Metagenomic surveillance | 15 |
| 041 | EPA PFAS ML (Jones F&T proxy) | Fate-and-transport ML | 14 |
| 042 | MassBank spectral (Jones MS proxy) | MS spectral library | 9 |

### BarraCUDA CPU Parity Validation

`validate_barracuda_cpu` proves pure Rust math matches Python across all 9
algorithmic domains in ~56ms (release build):

| Domain | Time (µs) | Status |
|--------|-----------|--------|
| QS ODE (48h integration) | ~9,000 | PASS |
| Capacitor ODE (48h) | ~11,000 | PASS |
| Gillespie SSA (100 reps) | ~29,000 | PASS |
| HMM (forward, Viterbi, posterior) | <1 | PASS |
| Smith-Waterman (40bp + batch) | ~13 | PASS |
| Felsenstein (20bp, 3 taxa) | ~1 | PASS |
| Diversity (Shannon + Simpson) | <1 | PASS |
| Peak detection (100 pts) | ~1 | PASS |
| Game theory (100h ODE) | ~7,500 | PASS |
| Robinson-Foulds (4 taxa) | ~3 | PASS |

### New GPU Promotion Candidates

| Module | GPU Strategy | Priority | Refactor Status |
|--------|-------------|----------|-----------------|
| `bio::felsenstein` | Site-parallel (1 workgroup/site) | **P0** | `FlatTree` GPU-ready layout done |
| `bio::placement` | Edge-parallel (1 workgroup/edge) | **P0** | Depends on `FlatTree` |
| `bio::bootstrap` | Replicate-parallel (1 wg/replicate) | **P0** | Index-based resampling needed |
| `bio::alignment` | Anti-diagonal wavefront parallelism | **P1** | `score_batch`/`align_batch` added |
| `bio::hmm` | Sequence-parallel (1 wg/sequence) | **P1** | `forward_batch`/`viterbi_batch` added |
| `bio::bistable` + 4 more ODE | Parameter-sweep via `ops::rk_stage` | **P1** | Ready (uses `bio::ode`) |

### Exp043/044 Results (Feb 20, 2026)

| Experiment | Checks | Status |
|------------|--------|--------|
| Exp043 BarraCUDA CPU Parity v3 | 45/45 | PASS |
| Exp044 BarraCUDA GPU v3 | 14/14 | PASS |

**CPU:** All 18 domains validated on CPU (v1: 21 + v2: 18 + v3: 45 = 84 total checks).
Domains 10–18 in v3: multi-signal QS, phage defense, bootstrap, placement,
decision tree, spectral matching, extended diversity (Pielou, BC, Chao1),
k-mer counting, integrated pipeline.

**GPU:** 140 total GPU checks (diversity 38 + 16S pipeline 88 + v3: 14). GPU
parity confirmed for Pielou, Bray-Curtis, spectral cosine, variance,
correlation, weighted dot, determinism. Tier B domains (ODE, SW, HMM,
bootstrap, Felsenstein, placement) are CPU-validated and ready for ToadStool
shader development.

### Exp046–050 Results (Feb 20, 2026 — GPU composition)

| Experiment | Checks | Domain | Status |
|------------|--------|--------|--------|
| Exp046 GPU Phylo Composition | 15/15 | Felsenstein + bootstrap + placement | PASS |
| Exp047 GPU HMM Forward | 13/13 | Batch HMM (local WGSL shader) | PASS |
| Exp048 CPU vs GPU Benchmark | 6/6 | Felsenstein + Bootstrap + HMM timing | PASS |
| Exp049 GPU ODE Sweep | 7/7 | 64-batch QS parameter sweep (local WGSL) | PASS |
| Exp050 GPU Bifurcation Eigen | 5/5 | Jacobian eigenvalues via BatchedEighGpu | PASS |

**NVVM Finding**: RTX 4070 NVVM cannot compile native f64 `exp()`, `log()`, or
`pow()`. Driver profile reports `needs_f64_exp_log_workaround() = false` (incorrect).
Workaround: `ShaderTemplate::for_driver_auto(source, true)` to force polyfill.

**naga Finding**: `enable f64;` rejected by naga (wgpu 22). Bare f32 literals
in f64 builtins (e.g., `max(x, 0.0)`) fail type check. Fix: `f64()` casts.

### ToadStool Absorption Candidates

1. **`hmm_forward_f64.wgsl`** — HMM batch forward in log-space (13/13 checks)
2. **`batched_qs_ode_rk4_f64.wgsl`** — ODE RK4 sweep with `pow_f64` polyfill
3. **NVVM driver profile**: `needs_f64_exp_log_workaround()` should return
   `true` for Ada Lovelace (RTX 4070/4080/4090)
4. **`BatchedOdeRK4F64::integrate`**: should use `compile_shader_f64` not `compile_shader`

### Updated Totals

- **50 experiments**, **1,257 validation checks** (1,035 CPU + 222 GPU)
- **465 Rust tests**, **0 clippy warnings**, **0 fmt diffs**
- **84/84 BarraCUDA CPU parity checks** across 18 domains
- **222 GPU checks** across 9 GPU validation binaries
- **metalForge/** updated with NVVM findings + absorption strategy
- **4 local WGSL shaders** (Write → Absorb → Lean candidates)

---

## Part 7: Code to Absorb (GPU-Ready Patterns)

Following hotSpring's pattern, these are the specific Rust patterns
written for ToadStool absorption:

### 1. `FlatTree` (felsenstein.rs) — GPU-native tree layout

```rust
pub struct FlatTree {
    pub n_leaves: usize,
    pub n_internal: usize,
    pub n_sites: usize,
    pub left_child: Vec<i32>,       // negative = leaf -(idx+1)
    pub right_child: Vec<i32>,
    pub leaf_states: Vec<u8>,       // column-major [site * n_leaves + leaf]
    pub trans_left: Vec<f64>,       // precomputed 4x4 per node [node*16 + from*4 + to]
    pub trans_right: Vec<f64>,
}
```

Maps directly to WGSL storage buffers. One workgroup per site computes
pruning in post-order over the flat node array.

### 2. Batch APIs (hmm.rs, alignment.rs)

```rust
pub fn forward_batch(model: &HmmModel, sequences: &[&[usize]]) -> Vec<ForwardResult>
pub fn score_batch(pairs: &[(&[u8], &[u8])], params: &ScoringParams) -> Vec<i32>
```

Each element is independent — maps to one GPU workgroup per element.

### 3. metalForge CPU Baselines

Python benchmark at `metalForge/benchmarks/profile_gpu_candidates.py` with
JSON output at `metalForge/benchmarks/cpu_baseline.json`. Establishes the
baseline that GPU must beat:

| Workload | Python CPU (µs) |
|----------|----------------|
| Felsenstein 1K sites | 6,623 |
| Felsenstein 10K sites | 70,603 |
| SW 100×100 | 74,333 |
| Bootstrap 100 reps | 65,607 |
| HMM forward T=1000 | 1,664 |

Rust CPU will be 10-100× faster; GPU target is 10-1000× beyond Rust CPU
for batch workloads.
