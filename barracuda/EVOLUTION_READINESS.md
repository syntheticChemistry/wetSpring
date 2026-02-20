# wetSpring Evolution Readiness

**Date:** February 20, 2026
**Pattern:** Write → Absorb → Lean (inherited from hotSpring)
**Status:** 34 CPU + 15 GPU modules, 4 local WGSL shaders, 15 ToadStool primitives consumed

---

## Absorption Tiers

| Tier | Meaning | Action |
|------|---------|--------|
| **Absorbed** | ToadStool has the primitive; wetSpring consumes upstream | Lean on upstream |
| **A** | Local code ready for handoff — GPU-friendly, validated | Write handoff doc |
| **B** | CPU-validated, needs GPU-friendly refactoring | Refactor for absorption |
| **C** | CPU-only, no GPU path planned | Keep local |

---

## CPU Modules (34)

| Module | Domain | GPU Tier | ToadStool Primitive | Notes |
|--------|--------|----------|-------------------|-------|
| `alignment` | Smith-Waterman | Absorbed | `SmithWatermanGpu` | Exp044 |
| `bistable` | ODE toggle switch | **A** | — | Map to `BatchedOdeRK4F64` |
| `bootstrap` | Phylo resampling | Absorbed | Compose `FelsensteinGpu` | Exp046 |
| `capacitor` | Signal peak | C | — | Too small for GPU |
| `chimera` | Chimera detection | C | — | Sequential per-read |
| `cooperation` | Game theory QS | C | — | CPU-only model |
| `dada2` | Error model | **A** | Local WGSL | `dada2_e_step.wgsl` |
| `decision_tree` | PFAS ML | Absorbed | `TreeInferenceGpu` | Exp044 |
| `derep` | Dereplication | C | — | Hash-based, CPU-optimal |
| `diversity` | α/β diversity | Absorbed | `BrayCurtisF64`, `FMR` | Exp004/016 |
| `eic` | Extracted ion chromatogram | C | — | I/O-bound |
| `feature_table` | OTU table | C | — | Sparse matrix |
| `felsenstein` | Pruning likelihood | Absorbed | `FelsensteinGpu` | Exp046 |
| `gillespie` | Stochastic SSA | Absorbed | `GillespieGpu` | Exp044 |
| `hmm` | Hidden Markov Model | **A** | Local WGSL | `hmm_forward_f64.wgsl` |
| `kmd` | Kendrick mass defect | C | — | Lookup table |
| `kmer` | K-mer counting | **B** | — | Needs lock-free hash GPU primitive |
| `merge_pairs` | Read merging | C | — | Sequential per-pair |
| `multi_signal` | Multi-signal QS | **B** | — | Maps to ODE sweep |
| `neighbor_joining` | NJ tree construction | C | — | Sequential algorithm |
| `ode` | RK4 integrator | **A** | Local WGSL | `batched_qs_ode_rk4_f64.wgsl` |
| `pcoa` | PCoA ordination | Absorbed | `BatchedEighGpu` | Exp016 |
| `phage_defense` | CRISPR/RM model | **B** | — | Maps to ODE sweep |
| `phred` | Quality scoring | C | — | Per-base lookup |
| `placement` | Phylo placement | Absorbed | Compose `FelsensteinGpu` | Exp046 |
| `qs_biofilm` | QS/c-di-GMP ODE | **A** | Local WGSL | `batched_qs_ode_rk4_f64.wgsl` |
| `quality` | Read quality | **A** | Local WGSL | `quality_filter.wgsl` |
| `reconciliation` | DTL reconciliation | C | — | Tree traversal |
| `robinson_foulds` | Tree distance | C | — | Per-node comparison |
| `signal` | Signal processing | C | — | FFT-based, small data |
| `spectral_match` | Spectral cosine | Absorbed | `FMR` spectral cosine | Exp016 |
| `taxonomy` | Naive Bayes classify | **B** | — | NPU candidate (FC model) |
| `tolerance_search` | Tolerance search | Absorbed | `BatchTolSearchF64` | Exp016 |
| `unifrac` | UniFrac distance | **B** | — | Needs tree traversal GPU primitive |

---

## GPU Modules (15)

| Module | Wraps | ToadStool Primitive | Status |
|--------|-------|-------------------|--------|
| `chimera_gpu` | `chimera` GPU scoring | `FMR` | Lean |
| `dada2_gpu` | DADA2 E-step | Local WGSL | **A** — handoff candidate |
| `diversity_gpu` | α/β diversity | `BrayCurtisF64`, `FMR` | Lean |
| `eic_gpu` | EIC extraction | `FMR` | Lean |
| `gemm_cached` | Matrix multiply | `GemmCachedF64` | Lean |
| `hmm_gpu` | HMM forward | Local WGSL | **A** — handoff candidate |
| `kriging` | Spatial interpolation | `KrigingF64` | Lean |
| `ode_sweep_gpu` | ODE parameter sweep | Local WGSL | **A** — handoff candidate |
| `pcoa_gpu` | PCoA eigenvalues | `BatchedEighGpu` | Lean |
| `quality_gpu` | Quality filtering | Local WGSL | **A** — handoff candidate |
| `rarefaction_gpu` | Rarefaction curves | `PrngXoshiro` | Lean |
| `spectral_match_gpu` | Spectral cosine | `FMR` | Lean |
| `stats_gpu` | Variance/correlation | `FMR` | Lean |
| `streaming_gpu` | Streaming pipeline | Multiple | Lean |
| `taxonomy_gpu` | Taxonomy scoring | `FMR` | Lean |

---

## Local WGSL Shader Inventory

| Shader | Domain | Lines | Status | Absorption Target |
|--------|--------|-------|--------|------------------|
| `dada2_e_step.wgsl` | DADA2 error model | ~60 | **A** | New ToadStool primitive |
| `quality_filter.wgsl` | Read quality | ~40 | **A** | New ToadStool primitive |
| `hmm_forward_f64.wgsl` | HMM batch forward | ~65 | **A** | `HmmBatchForwardF64` |
| `batched_qs_ode_rk4_f64.wgsl` | QS ODE sweep | ~120 | **A** | Fix upstream `BatchedOdeRK4F64` |

### Shader Compilation Notes

All 4 local shaders require `ShaderTemplate::for_driver_auto(source, true)` on
RTX 4070 (Ada Lovelace) to force f64 transcendental polyfills. The driver profile
incorrectly reports `needs_f64_exp_log_workaround() = false` for Ada Lovelace.

**naga quirks:**
- `enable f64;` not supported — omit from all WGSL
- Bare f32 literals in f64 builtins fail type check — use `f64(0.0)`
- `pow()` on f64 crashes NVVM — use `pow_f64()` polyfill

---

## ToadStool Primitives Consumed (15)

| Primitive | Module(s) | Exp |
|-----------|----------|-----|
| `BrayCurtisF64` | diversity_gpu | 004/016 |
| `FusedMapReduceF64` (Shannon) | diversity_gpu | 004/016 |
| `FusedMapReduceF64` (Simpson) | diversity_gpu | 004/016 |
| `FusedMapReduceF64` (spectral cosine) | spectral_match_gpu | 016 |
| `GemmCachedF64` | gemm_cached | 016 |
| `BatchedEighGpu` | pcoa_gpu, validate_gpu_ode_sweep | 016/050 |
| `BatchTolSearchF64` | tolerance_search | 016 |
| `PrngXoshiro` | rarefaction_gpu | 016 |
| `SmithWatermanGpu` | alignment (via barracuda) | 044 |
| `GillespieGpu` | gillespie (via barracuda) | 044 |
| `TreeInferenceGpu` | decision_tree (via barracuda) | 044 |
| `FelsensteinGpu` | felsenstein, bootstrap, placement | 046 |
| `ShaderTemplate::for_driver_auto` | hmm_gpu, ode_sweep_gpu | 047/049 |
| `LogsumexpWgsl` | (available, not yet wired) | — |
| `BatchedOdeRK4F64` | (blocked: `enable f64;` + `pow` crash) | — |

---

## Absorption Queue (handoff to ToadStool)

### Ready Now (Tier A)

1. **`hmm_forward_f64.wgsl`** — 13/13 GPU checks (Exp047)
   - Batch HMM forward in log-space, 1 thread per sequence
   - Uses `exp_f64`, `log_f64` polyfills via `for_driver_auto(_, true)`
   - Proposed primitive: `HmmBatchForwardF64`

2. **`batched_qs_ode_rk4_f64.wgsl`** — 7/7 GPU checks (Exp049)
   - Fix for upstream `BatchedOdeRK4F64` (`enable f64;` removal, `pow_f64` polyfill)
   - Upstream should use `compile_shader_f64` not `compile_shader`

3. **NVVM driver profile fix** — Ada Lovelace RTX 40-series
   - `needs_f64_exp_log_workaround()` should return `true`
   - Affects all f64 transcendentals: `exp`, `log`, `pow`

### Needs Refactoring (Tier B)

4. **`kmer`** — Needs lock-free hash table GPU primitive (P3)
5. **`unifrac`** — Needs tree traversal GPU primitive (P3)
6. **`taxonomy`** — NPU candidate (Naive Bayes → FC model → int8)
7. **`multi_signal`** / **`phage_defense`** — Map to ODE sweep pattern

---

## Validation Coverage by Tier

| Tier | Modules | CPU Checks | GPU Checks |
|------|---------|------------|------------|
| Absorbed | 12 | 450+ | 200 |
| A (handoff ready) | 6 | 180+ | 32 |
| B (needs refactor) | 5 | 80+ | 0 |
| C (CPU-only) | 11 | 325+ | 0 |
| **Total** | **34** | **1,035** | **200** |

---

## Write → Absorb → Lean History

| Date | Event |
|------|-------|
| Feb 16 | Handoff v1: diversity shaders, log_f64 bug, BrayCurtis pattern |
| Feb 17 | Handoff v2: bio primitives requested (SW, Gillespie, Felsenstein, DT) |
| Feb 19 | Handoff v3: primitive verification, fragile GEMM path eliminated |
| Feb 20 | ToadStool absorbs 4 bio primitives (commit cce8fe7c) |
| Feb 20 | Exp046: FelsensteinGpu composed for bootstrap + placement |
| Feb 20 | Exp047: Local HMM shader written + validated (absorption candidate) |
| Feb 20 | Exp049: Local ODE shader written (upstream fix candidate) |
| Feb 20 | Exp050: BatchedEighGpu validated for bifurcation (bit-exact) |
