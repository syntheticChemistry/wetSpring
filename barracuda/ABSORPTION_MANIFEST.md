# Absorption Manifest: wetSpring → ToadStool

**Date:** February 21, 2026  
**Pattern:** Write → Absorb → Lean (inherited from hotSpring)  
**Status:** 5 absorbed, 9 Tier A handoff candidates, 4 CPU math extraction candidates

---

## Absorbed by ToadStool (Lean)

These modules have been absorbed upstream. wetSpring consumes the ToadStool
primitive directly via `barracuda::*` re-exports. Local code has been deleted
or rewired.

| Primitive | Origin Module | ToadStool Location | Absorbed Date | Handoff |
|-----------|--------------|-------------------|---------------|---------|
| `SmithWatermanGpu` | `bio::alignment` | `ops::bio::smith_waterman` | Feb 20 | v2 |
| `GillespieGpu` | `bio::gillespie` | `ops::bio::gillespie` | Feb 20 | v2 |
| `TreeInferenceGpu` | `bio::decision_tree` | `ops::bio::tree_inference` | Feb 20 | v2 |
| `FelsensteinGpu` | `bio::felsenstein` | `ops::bio::felsenstein` | Feb 20 | v2 |
| `GemmCachedF64` | `bio::gemm_cached` | `ops::linalg::gemm_f64` | Feb 20 | v3 |

**Validation**: All 5 absorbed primitives pass GPU parity checks in
`validate_toadstool_bio.rs` (Exp045, 10/10).

---

## Active Local Extensions (Write)

These are validated local implementations ready for or approaching ToadStool
absorption. Each has a CPU reference, GPU validation, and documented binding
layout.

### 9 Local WGSL Shaders (Tier A — handoff submitted)

| Shader | Domain | GPU Checks | Binding Layout | Absorption Target |
|--------|--------|:----------:|----------------|-------------------|
| `quality_filter.wgsl` | Read quality trimming | 88 (pipeline) | 1 storage(r) + 1 storage(rw) | `ParallelFilter<T>` |
| `dada2_e_step.wgsl` | DADA2 error model | 88 (pipeline) | 2 storage(r) + 1 storage(rw) | `BatchPairReduce<f64>` |
| `hmm_forward_f64.wgsl` | HMM batch forward | 13 (Exp047) | 3 storage(r) + 1 storage(rw) + 1 uniform | `HmmBatchForwardF64` |
| `batched_qs_ode_rk4_f64.wgsl` | ODE parameter sweep | 7 (Exp049) | 2 storage(r) + 1 storage(rw) + 1 uniform | Fix upstream `BatchedOdeRK4F64` |
| `ani_batch_f64.wgsl` | ANI pairwise identity | 7 (Exp058) | 2 storage(r) + 1 storage(rw) | `AniBatchF64` |
| `snp_calling_f64.wgsl` | SNP calling | 5 (Exp058) | 2 storage(r) + 1 storage(rw) | `SnpCallingF64` |
| `dnds_batch_f64.wgsl` | dN/dS (Nei-Gojobori) | 9 (Exp058) | 3 storage(r) + 1 storage(rw) + 1 uniform | `DnDsBatchF64` |
| `pangenome_classify.wgsl` | Pangenome classification | 6 (Exp058) | 2 storage(r) + 1 storage(rw) | `PangenomeClassifyGpu` |
| `rf_batch_inference.wgsl` | Random Forest batch | 13 (Exp063) | 4 storage(r) + 1 storage(rw) | `RfBatchInferenceGpu` |

**Total GPU checks from local shaders:** 236

**Handoff doc:** `../wateringHole/handoffs/WETSPRING_TOADSTOOL_TIER_A_SHADERS_FEB21_2026.md`

### CPU Math Functions (extraction candidates)

| Local Implementation | File | Upstream Primitive | Status |
|---------------------|------|-------------------|--------|
| `erf()`, `normal_cdf()` | `bio/special.rs` | `barracuda::special::erf` | Consolidated, `mul_add` optimized |
| `ln_gamma()` | `bio/special.rs` | `barracuda::special::ln_gamma` | Lanczos, Horner form |
| `regularized_gamma_lower()` | `bio/special.rs` | `barracuda::special::regularized_gamma_p` | Series, 1e-15 convergence |
| `integrate_peak()` | `bio/eic.rs` | `barracuda::numerical::trapz` | Trapezoidal integration |

**Blocked:** Awaiting `barracuda` `[features] math = []` feature gate that
exposes CPU math without pulling in wgpu/akida/toadstool-core.

---

## Composed from ToadStool Primitives

These workflows compose existing ToadStool primitives — no new absorption
needed, just documentation of the composition pattern.

| Workflow | Composed From | Exp | Checks |
|----------|-------------|-----|:------:|
| Bootstrap resampling | `FelsensteinGpu` per replicate | 046 | 15 |
| Phylogenetic placement | `FelsensteinGpu` per edge | 046 | 15 |
| ODE parameter sweep | Local WGSL + `for_driver_auto` | 049 | 7 |
| Bifurcation eigenvalues | CPU Jacobian + `BatchedEighGpu` | 050 | 5 |
| 5-stage GPU pipeline | `FMR` + `BrayCurtis` + `BatchedEigh` + `FMR` + `FMR` | 075 | 31 |
| Cross-substrate pipeline | GPU `FMR` → NPU classify → CPU aggregate | 076 | 17 |

---

## Keeps Local (no absorption path)

These modules are wetSpring-specific and stay local. They are validation
infrastructure, I/O parsers, or CPU-optimal algorithms.

| Module | Why Local |
|--------|----------|
| `validation.rs` | Spring-specific validation framework |
| `tolerances.rs` | Spring-specific tolerance constants |
| `error.rs` | Spring-specific error types |
| `encoding.rs` | Sovereign base64 (zero dependencies) |
| `bench/` | Benchmark harness + power monitoring |
| `io::fastq` | Streaming FASTQ parser (CPU-optimal, I/O-bound) |
| `io::mzml` | Streaming mzML parser (CPU-optimal, I/O-bound) |
| `io::ms2` | Streaming MS2 parser (CPU-optimal, I/O-bound) |
| `io::xml` | Sovereign XML subset parser |
| `bio::chimera` | Sequential per-read (CPU-optimal) |
| `bio::derep` | Hash-based dereplication (CPU-optimal) |
| `bio::kmer` | K-mer counting, 2-bit encoding (CPU-optimal) |
| `bio::merge_pairs` | Sequential per-pair merging (CPU-optimal) |
| `bio::phred` | Per-base quality lookup (CPU-optimal) |
| `bio::gbm` | Sequential boosting inference (CPU-optimal) |
| `bio::molecular_clock` | Small tree traversal (CPU-optimal) |
| `bio::neighbor_joining` | Sequential NJ algorithm (CPU-optimal) |
| `bio::reconciliation` | DTL tree traversal (CPU-optimal) |
| `bio::robinson_foulds` | Per-node tree comparison (CPU-optimal) |
| `bio::signal` | 1D peak detection (CPU-optimal) |
| `bio::eic` | EIC extraction (I/O-bound) |
| `bio::feature_table` | Sparse feature matrix (CPU-optimal) |
| `bio::kmd` | Lookup-table KMD (CPU-optimal) |

---

## Blocked (needs upstream evolution)

| Blocker | Impact | Resolution Path |
|---------|--------|-----------------|
| `barracuda::math` feature gate | 4 CPU math functions can't lean upstream | Propose `[features] math = []` |
| NVVM driver profile | f64 transcendentals crash on Ada Lovelace | Fix `needs_f64_exp_log_workaround()` |
| Lock-free GPU hash primitive | `bio::kmer` can't promote to GPU | ToadStool new primitive |
| Tree traversal GPU primitive | `bio::unifrac` can't promote to GPU | ToadStool new primitive |
| NPU substrate support | `bio::taxonomy` NPU inference path | ToadStool NPU dispatch |

---

## ToadStool Primitives Consumed (15)

| Primitive | Module(s) | Validation |
|-----------|----------|------------|
| `BrayCurtisF64` | `diversity_gpu` | Exp004/016 |
| `FusedMapReduceF64` (Shannon) | `diversity_gpu` | Exp004/016 |
| `FusedMapReduceF64` (Simpson) | `diversity_gpu` | Exp004/016 |
| `FusedMapReduceF64` (spectral cosine) | `spectral_match_gpu` | Exp016 |
| `GemmCachedF64` | `gemm_cached` | Exp016 |
| `BatchedEighGpu` | `pcoa_gpu`, `validate_gpu_ode_sweep` | Exp016/050 |
| `BatchTolSearchF64` | `tolerance_search` | Exp016 |
| `PrngXoshiro` | `rarefaction_gpu` | Exp016 |
| `SmithWatermanGpu` | `alignment` (via barracuda) | Exp044 |
| `GillespieGpu` | `gillespie` (via barracuda) | Exp044 |
| `TreeInferenceGpu` | `decision_tree` (via barracuda) | Exp044 |
| `FelsensteinGpu` | `felsenstein`, `bootstrap`, `placement` | Exp046 |
| `ShaderTemplate::for_driver_auto` | `hmm_gpu`, `ode_sweep_gpu`, Track 1c, RF | Exp047+ |
| `LogsumexpWgsl` | (available, not yet wired) | — |
| `BatchedOdeRK4F64` | (blocked: `enable f64;` + `pow` crash) | — |

---

## Absorption History

| Date | Event |
|------|-------|
| Feb 16 | Handoff v1: diversity shaders, BrayCurtis pattern |
| Feb 17 | Handoff v2: bio primitives requested (SW, Gillespie, Felsenstein, DT) |
| Feb 19 | Handoff v3: primitive verification, fragile GEMM path eliminated |
| Feb 20 | ToadStool absorbs 4 bio primitives (commit `cce8fe7c`) |
| Feb 20 | Bootstrap/placement composed from FelsensteinGpu |
| Feb 20 | HMM, ODE local shaders written + validated |
| Feb 20 | Track 1c: 4 new WGSL shaders (ANI, SNP, dN/dS, pangenome) |
| Feb 20 | RF batch inference shader (SoA layout) |
| Feb 21 | Phase 15: Quality hardening, tolerance centralization |
| Feb 21 | Phase 17: Absorption engineering, `bio::special` consolidated |
| Feb 21 | Exp064-076: GPU consolidation, streaming, cross-substrate proofs |
| Feb 21 | Handoff v6 + ABSORPTION_MANIFEST created |

---

## How to Use This Document

1. **Before writing new GPU code**: Check if ToadStool already has a primitive
   (Absorbed section) or if another Spring has already written a handoff
2. **Before absorption**: Verify all GPU checks pass, binding layout is documented,
   dispatch geometry is documented, CPU reference is validated
3. **After absorption**: Move from "Active Local" to "Absorbed", update handoff
   date, delete local shader copy, rewire to upstream import
4. **Cross-Spring**: Check `../hotSpring/barracuda/ABSORPTION_MANIFEST.md` for
   physics primitives that may have analogous bio patterns
