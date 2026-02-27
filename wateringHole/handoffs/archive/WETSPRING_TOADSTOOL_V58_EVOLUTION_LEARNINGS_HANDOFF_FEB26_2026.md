# wetSpring → ToadStool/BarraCUDA Handoff: V58 Evolution Learnings & Absorption Candidates

**Date:** February 26, 2026
**From:** wetSpring V58
**To:** ToadStool/BarraCUDA team
**ToadStool pin:** S68 (`f0feb226`) — 700 shaders, 0 f32-only, universal precision
**wetSpring:** 961 tests, 882 barracuda lib, 96.67% llvm-cov, 82 named tolerances, 189 experiments, 175 binaries, 4,494+ validation checks (1,578 GPU on RTX 4070)

---

## Executive Summary

This handoff captures everything wetSpring has learned that is relevant to
ToadStool's evolution, including absorption candidates, cross-spring patterns,
feature-gate hygiene, universal precision experience, and benchmark data.
V57 caught wetSpring up to S68; this document is the forward-looking complement.

---

## Part 1: Cross-Spring Evolution — What Worked

### The Biome Model

Springs don't import each other. Each leans on ToadStool independently.
ToadStool absorbs what works; all Springs benefit. This architecture has
proven itself across 5 Springs:

```
hotSpring   → precision f64, DF64 streaming, plasma physics shaders
wetSpring   → bio ODE, QS, diversity, metagenomics, drug repurposing
neuralSpring → EA fitness, spatial payoff, locus variance, Hamming, Jaccard
airSpring   → atmospheric / climate primitives
groundSpring → geospatial / soil physics
```

### Cross-Spring Primitive Flow

Primitives migrate naturally:

| Origin Spring | Primitive | Absorbed By | Consumed By |
|---------------|-----------|------------|-------------|
| wetSpring | 8 bio shaders (HMM, ANI, SNP, dN/dS, pangenome, QF, DADA2, RF) | ToadStool S31d/g | wetSpring, neuralSpring |
| neuralSpring | 5 EA/population genetics (Hamming, Jaccard, SpatialPayoff, BatchFitness, LocusVariance) | ToadStool S39 | wetSpring, neuralSpring |
| hotSpring | precision f64 polyfills, DF64 core-streaming | ToadStool S62+DF64 | wetSpring, hotSpring |
| wetSpring | diversity_fusion_f64 (fused Shannon+Simpson+evenness) | ToadStool S63 | wetSpring |
| wetSpring | 5 ODE bio systems (bistable, capacitor, cooperation, multi_signal, phage_defense) | ToadStool S58 | wetSpring |

### What This Proves

1. **Write → Absorb → Lean works at scale.** wetSpring went from 12 local WGSL
   shaders to zero, with all math delegated to ToadStool.
2. **Cross-spring pollination is real.** hotSpring's precision work improved
   wetSpring's numerical accuracy; wetSpring's bio patterns informed neuralSpring's
   population genetics.
3. **Universal precision architecture (S67/S68) unblocks DF64 for bio.** The
   same ODE shader that runs at F64 today can run at DF64 tomorrow with a
   single `Precision` argument change.

---

## Part 2: Universal Precision — wetSpring Experience

### V57 Rewire Results

wetSpring rewired 6 GPU modules from `compile_shader_f64` to
`compile_shader_universal(source, Precision::F64)`:

| Module | Shader Source | Status |
|--------|-------------|--------|
| `bistable_gpu` | `BatchedOdeRK4::<BistableOde>::generate_shader()` | Rewired, validated |
| `phage_defense_gpu` | `BatchedOdeRK4::<PhageDefenseOde>::generate_shader()` | Rewired, validated |
| `cooperation_gpu` | `BatchedOdeRK4::<CooperationOde>::generate_shader()` | Rewired, validated |
| `capacitor_gpu` | `BatchedOdeRK4::<CapacitorOde>::generate_shader()` | Rewired, validated |
| `multi_signal_gpu` | `BatchedOdeRK4::<MultiSignalOde>::generate_shader()` | Rewired, validated |
| `gemm_cached` | `GemmF64::WGSL` | Rewired, validated |

All 882 tests pass after rewire. The change is backward-compatible.

### DF64 Opportunity for Bio

The ODE systems are excellent DF64 candidates because:

- **Stiff dynamics:** QS bistable switching involves sharp transitions where
  accumulated FP error matters. DF64's ~48-bit mantissa (vs 23 for f32) would
  let RK4 handle stiffer scenarios at longer timesteps.
- **Parameter sweeps:** ODE sweep over 1000+ parameter combinations benefits
  from DF64 throughput on FP32 cores (5888 FP32 vs 92 FP64 on RTX 4070).
- **Already universal-precision-ready:** After V57, changing `Precision::F64`
  to `Precision::Df64` is the only code change needed.

### Recommended ToadStool Action

1. Validate `BatchedOdeRK4` with `Precision::Df64` — the `OdeSystem` trait
   generates f64 WGSL, and `compile_shader_universal` should rewrite to DF64
   automatically.
2. Add a DF64 test case in ToadStool's ODE test suite using one of the
   wetSpring bio systems (e.g., `PhageDefenseOde` — simplest, 4 vars, 11 params).

---

## Part 3: Feature-Gate Audit Findings

### The Bug (fixed, contributed upstream in V57)

`numerical/mod.rs` (`wgsl_hessian_column()`) and `stats/mod.rs`
(`WGSL_HISTOGRAM`, `WGSL_BOOTSTRAP_MEAN_F64`) referenced
`crate::shaders::precision` without `#[cfg(feature = "gpu")]`. Since
`crate::shaders` is gated behind `gpu`, this breaks `default-features = false`.

### Systematic Audit Recommendation

The f32→f64 evolution in S68 (291 shaders) added `downcast_f64_to_f32()` calls
across multiple modules. We recommend a systematic grep:

```bash
rg 'crate::shaders' --type rust -l | while read f; do
  if ! rg '#\[cfg\(feature = "gpu"\)\]' "$f" -q; then
    echo "UNGATED: $f"
  fi
done
```

Any file that references `crate::shaders::` but sits in a non-GPU module
(`error`, `linalg`, `numerical`, `special`, `tolerances`, `validation`, `stats`)
needs `#[cfg(feature = "gpu")]` on the referencing items.

### Why This Matters

wetSpring is the primary `default-features = false` consumer. Any new
shader-dependent code in CPU modules will break wetSpring's build. The CI
should include a `--no-default-features` test target.

---

## Part 4: Absorption Candidates for ToadStool

### Already Absorbed (Lean)

79 ToadStool primitives consumed. 0 local WGSL. 0 local derivative math.
0 local regression math. wetSpring is fully lean.

### Patterns Worth Absorbing

| Pattern | Location | Benefit |
|---------|----------|---------|
| **Validator harness** | `barracuda/src/validation.rs` | Structured f64 comparison with named tolerances, provenance tables, section headers, markdown output |
| **tolerances.rs** | `barracuda/src/tolerances.rs` | 82 named tolerance constants with paper/algorithm provenance — complementary to ToadStool's 12 `barracuda::tolerances` constants |
| **OdeSystem trait implementations** | `barracuda/src/bio/ode_systems.rs` | 5 bio ODE systems ready for `generate_shader()` — demonstrates the trait pattern at scale |
| **metalForge bridge** | `metalForge/forge/src/bridge.rs` | Multi-substrate dispatch (GPU↔NPU↔CPU), 47 tests — pattern for universal hardware routing |

### Domain-Specific Primitives

These are validated in wetSpring but currently not candidates for ToadStool
absorption (domain-specific, not general-purpose):

| Primitive | Domain | Reason |
|-----------|--------|--------|
| `soil_qs_*` | Soil microbiology | Too application-specific; wetSpring-only |
| `anderson_*` | Anderson localization QS | Cross-domain with hotSpring; potential baseCamp paper candidate |
| `drug_repurposing_*` | Pharmacophenomics | Track 3 specific; wetSpring-only |

---

## Part 5: Benchmark Data for ToadStool Reference

### GPU vs CPU Crossover Points (RTX 4070)

| Domain | Crossover N | GPU Speedup at N=large | Source |
|--------|------------|----------------------|--------|
| Spectral cosine | ~100 pairs | 926× at 2.1M | Exp087 |
| Bray-Curtis | ~50 | 200×+ at 500 | Exp092 |
| Smith-Waterman | ~10 pairs | 625× at 1000 | Exp059 |
| ODE parameter sweep | ~100 combos | 30×+ at 1000 | Exp049 |
| GEMM (32×32) | Always | Proportional to matrix size | Exp066 |
| HMM forward | ~50 seqs | 15× at 500 | Exp047 |

### Streaming vs Round-Trip

Streaming dispatch (zero CPU round-trips) gives 441–837× throughput over
per-dispatch round-trips at batch sizes ≥ 1000 (Exp091). This validates
ToadStool's streaming architecture design.

### Rust vs Python

Overall 22.5× speedup across 25 domains. Peak 625× (Smith-Waterman).
Minimum 2.4× (pipeline orchestration). GPU adds another 10–926× on top.

---

## Part 6: What We Learned About ToadStool's Architecture

### What Works Exceptionally Well

1. **`BatchedOdeRK4` + `OdeSystem` trait** — elegant, extensible, correct.
   5 bio ODE systems integrated with zero boilerplate.
2. **`FusedMapReduceF64`** — the Swiss army knife. Used for Shannon, Simpson,
   spectral cosine, KMD, neighbor-joining, molecular clock, merge pairs.
3. **`compile_shader_f64()` → `compile_shader_universal()`** — backward-compatible
   evolution. No downstream breakage.
4. **Cross-spring primitive sharing** — the `ops::bio::*` namespace lets
   neuralSpring and wetSpring share bio shaders without coupling.

### Rough Edges (minor, all worked around)

1. **SNP shader binding mismatch** — `is_variant` buffer declared `read_only`
   upstream but needs `read_write` for write-back. wetSpring wraps with
   `catch_unwind` (graceful skip). Fix contributed in V40.
2. **Feature-gate leaks** — addressed in V57 (see Part 3).
3. **`ShaderTemplate` name collision** — ODE derivative function names avoid
   `_f64` suffix to prevent `ShaderTemplate` from rewriting them. Documented
   in all 5 `OdeSystem` implementations.

---

## Part 7: Cross-Spring Evolution Timeline (S39 → S68)

| Session | What Happened | Who Benefits |
|---------|--------------|-------------|
| S39 | neuralSpring 5 EA primitives absorbed | wetSpring + neuralSpring |
| S40 | `KmerHistogramGpu`, `UniFracPropagateGpu`, `TaxonomyFcF64` | wetSpring |
| S41 | `BatchedOdeRK4F64` ODE fix | wetSpring + hotSpring |
| S52 | `barracuda::tolerances` module | All Springs |
| S54 | `graph_laplacian`, `effective_rank`, `numerical_hessian` | wetSpring |
| S56 | `boltzmann_sampling` | wetSpring |
| S57 | Formatting + DRY cleanup | All |
| S58 | `BatchedOdeRK4` trait + 5 bio ODE systems, NMF | wetSpring |
| S59 | `ridge_regression`, `anderson_3d_correlated`, `erf/ln_gamma/etc.` | wetSpring |
| S60 | `TranseScoreF64`, `SparseGemmF64`, `TopK` | wetSpring Track 3 |
| S62 | `PeakDetectF64`, DF64 core-streaming | wetSpring + hotSpring |
| S63 | `diversity_fusion_f64` absorbed | wetSpring |
| S64 | `stats::diversity`, `stats::{dot, l2_norm}` | wetSpring |
| S65 | `ComputeDispatch`, `Fp64Strategy`, DF64 GEMM | hotSpring + wetSpring |
| S66 | `hill`, `monod`, `fit_linear`, `percentile`, `mean`, `shannon_from_frequencies` | wetSpring |
| S67 | Universal Precision Architecture | All Springs |
| S68 | Dual-Layer Universal Precision, 291 f32→f64, 700 shaders | All Springs |

---

## Part 8: Recommended Next Steps for ToadStool

### Immediate

1. **Merge V57 feature-gate fix** if not already merged.
2. **Add `--no-default-features` CI target** to catch future gate leaks.

### Short-Term

1. **DF64 validation for `BatchedOdeRK4`** — test with wetSpring bio systems.
2. **GPU Lanczos kernel** — enables Anderson localization at L=24+ (Exp187).
3. **Audit `crate::shaders` references** in CPU-only modules (Part 3).

### Medium-Term

1. **Absorb `Validator` harness pattern** — wetSpring's structured validation
   with provenance tables and named tolerances could become a ToadStool utility.
2. **DF64 bio benchmarks** — once `BatchedOdeRK4` + DF64 works, benchmark
   ODE sweep throughput: F64 (native) vs DF64 (FP32 cores) on RTX 4070.
3. **Cross-spring benchmark suite** — a ToadStool-level binary that exercises
   primitives from all Springs in a single run.

### Long-Term

1. **metalForge pattern in ToadStool** — substrate-independent dispatch
   (GPU↔NPU↔CPU) as a first-class ToadStool feature.
2. **Tolerance module evolution** — merge wetSpring's 82 domain-specific
   constants with ToadStool's 12 infrastructure constants.

---

## Part 9: Files Changed in V58

| File | Change |
|------|--------|
| All status docs | Metrics synchronized: 189 experiments, 961 tests, 175 binaries, S68 |
| `experiments/README.md` | Added Exp184-189 to index and binary tables |
| `barracuda/EVOLUTION_READINESS.md` | Updated shader generation notes to `compile_shader_universal` |
| `wateringHole/README.md` | Updated shader count to 700+ (S68 universal precision) |
| Multiple `.md` files | 188→189 experiments, 912→961 tests, 173→175 binaries, S66→S68 |

---

## Part 10: Pin History

| Version | ToadStool Pin | Session | Key Changes |
|---------|--------------|---------|-------------|
| V58 | `f0feb226` | S68 | Evolution learnings handoff, full doc sync |
| V57 | `f0feb226` | S68 | Universal precision catch-up, feature-gate fix |
| V56 | `045103a7` | S66 | Science pipeline, NCBI, NestGate, biomeOS |
| V55 | `045103a7` | S66 | Deep debt, idiomatic Rust |
| V54 | `045103a7` | S66 | Codebase audit, supply-chain |
| V53 | `045103a7` | S66 | Cross-spring evolution benchmarks |
| V44 | `02207c4a` | S62+DF64 | Complete cross-spring rewire |
| V40 | `02207c4a` | S62+DF64 | 55-commit catch-up |
