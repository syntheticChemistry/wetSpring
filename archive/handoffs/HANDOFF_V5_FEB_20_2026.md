# Handoff v5: wetSpring → ToadStool / BarraCUDA Team

**Date:** February 20, 2026
**From:** wetSpring (ecoPrimals — Life Science & Analytical Chemistry)
**To:** ToadStool / BarraCUDA core team
**License:** AGPL-3.0-or-later
**Supersedes:** All prior handoffs (v1–v4, archived in `archive/handoffs/`)
**Supplemented by:** `wateringHole/handoffs/WETSPRING_TOADSTOOL_TIER_A_SHADERS_FEB21_2026.md`
**ToadStool reviewed:** barracuda v0.2.0 at commit cce8fe7c (bio absorption confirmed)

---

## Executive Summary

wetSpring has completed 63 experiments with 1,501 validation checks (all PASS)
across 25 algorithmic domains, proving Python → Rust CPU → GPU portability for
life science, analytical chemistry, and environmental monitoring workloads.

**Since v4 handoff:**

1. **GPU Track 1c promotion** — 4 new WGSL shaders (ANI, SNP, dN/dS, pangenome),
   27/27 GPU checks (Exp058)
2. **25-domain Rust vs Python benchmark** — 22.5× overall, peak 625× (Exp059)
3. **metalForge cross-substrate validation** — 20/20 CPU↔GPU parity (Exp060)
4. **ML ensemble inference** — Random Forest (Exp061) + GBM (Exp062) in pure Rust
5. **GPU RF batch inference** — SoA WGSL shader, 13/13 GPU checks (Exp063)
6. **Total local WGSL shaders** increased from 4 to **9**, all validated

**wetSpring totals:** 1,501 validation checks (1,241 CPU + 260 GPU), 552 tests
(539 lib + 13 doc), 93.5% line coverage, 63 experiments, 61 binaries,
25 CPU domains, 9 local WGSL shaders

---

## What ToadStool Has Absorbed (Unchanged)

| wetSpring Finding | ToadStool Primitive | Date |
|-------------------|---------------------|------|
| Bray-Curtis f64 | `ops::bray_curtis_f64::BrayCurtisF64` | Feb 16 |
| log_f64 coefficient bug | Fixed in `math_f64.wgsl` | Feb 16 |
| Shannon/Simpson FMR | `FusedMapReduceF64::shannon_entropy()`, `simpson_index()` | Feb 16 |
| Smith-Waterman GPU | `ops::bio::smith_waterman::SmithWatermanGpu` | Feb 20 |
| Gillespie SSA GPU | `ops::bio::gillespie::GillespieGpu` | Feb 20 |
| Tree Inference GPU | `ops::bio::tree_inference::TreeInferenceGpu` | Feb 20 |
| Felsenstein GPU | `ops::bio::felsenstein::FelsensteinGpu` | Feb 20 |
| GemmF64::WGSL constant | `ops::linalg::gemm_f64::GemmF64::WGSL` | Feb 20 |

**All 6 original primitive requests are addressed.**

---

## New Shaders Ready for Absorption (9 total)

### Original Pipeline Shaders (4)

| Shader | GPU Checks | Exp | Proposed Primitive |
|--------|:----------:|-----|-------------------|
| `hmm_forward_f64.wgsl` | 13/13 | 047 | `HmmBatchForwardF64` |
| `batched_qs_ode_rk4_f64.wgsl` | 7/7 | 049 | Fix upstream `BatchedOdeRK4F64` |
| `dada2_e_step.wgsl` | 88 (pipeline) | 016 | `BatchPairReduce<f64>` |
| `quality_filter.wgsl` | 88 (pipeline) | 016 | `ParallelFilter<T>` |

### Track 1c Bioinformatics Shaders (4, new since v4)

| Shader | GPU Checks | Exp | Proposed Primitive |
|--------|:----------:|-----|-------------------|
| `ani_batch_f64.wgsl` | 7/7 | 058 | `AniBatchF64` |
| `snp_calling_f64.wgsl` | 5/5 | 058 | `SnpCallingF64` |
| `dnds_batch_f64.wgsl` | 9/9 | 058 | `DnDsBatchF64` |
| `pangenome_classify.wgsl` | 6/6 | 058 | `PangenomeClassifyGpu` |

### ML Ensemble Shader (1, new since v4)

| Shader | GPU Checks | Exp | Proposed Primitive |
|--------|:----------:|-----|-------------------|
| `rf_batch_inference.wgsl` | 13/13 | 063 | `RfBatchInferenceGpu` |

### Shader Handoff Notes

Each shader includes:
- **Binding layout**: `@group(0) @binding(N)` with buffer types and sizes
- **Dispatch geometry**: workgroup size and grid formula
- **CPU reference**: corresponding `bio::*` function for validation
- **Tolerance**: documented in each experiment `.md` file
- **f64 polyfill requirements**: which shaders need `for_driver_auto(_, true)`

All shader source files are at `barracuda/src/shaders/`.

---

## NVVM Driver Profile Bug (Unchanged)

RTX 4070 (Ada Lovelace) NVVM cannot compile native f64 `exp()`, `log()`, `pow()`.
Driver profile incorrectly reports `needs_f64_exp_log_workaround() = false`.

**Fix**: `needs_f64_exp_log_workaround()` should return `true` for Ada Lovelace
(RTX 4070/4080/4090). This affects 5 of 9 local shaders:
- `hmm_forward_f64.wgsl` (exp, log)
- `batched_qs_ode_rk4_f64.wgsl` (pow)
- `dnds_batch_f64.wgsl` (log)
- `rf_batch_inference.wgsl` (none — thresholds only, no transcendentals)
- All Track 1c shaders via `ShaderTemplate::for_driver_auto`

---

## New CPU Modules (Since v4)

| Module | Domain | Exp | Checks |
|--------|--------|-----|:------:|
| `random_forest` | RF ensemble (majority vote, N trees) | 061 | 13 |
| `gbm` | GBM binary (sigmoid) + multi-class (softmax) | 062 | 16 |

These extend the ML pipeline: decision tree → Random Forest → GBM,
covering the three dominant ensemble methods without Python dependency.

---

## metalForge Cross-Substrate Validation (Exp060)

20/20 checks proving identical results on CPU and GPU for Track 1c algorithms.
This validates the metalForge thesis: **math is substrate-independent**.

| Algorithm | CPU Time | GPU Time | Checks | Status |
|-----------|----------|----------|:------:|--------|
| ANI | reference | parity | 5/5 | CPU=GPU |
| SNP | reference | parity | 5/5 | CPU=GPU |
| Pangenome | reference | parity | 5/5 | CPU=GPU |
| dN/dS | reference | parity | 5/5 | CPU=GPU |

---

## Updated Totals

| Metric | v4 (prev) | v5 (current) |
|--------|:---------:|:------------:|
| Experiments | 50 | **63** |
| CPU checks | 1,035 | **1,241** |
| GPU checks | 222 | **260** |
| Total checks | 1,257 | **1,501** |
| Tests | 465 | **552** |
| CPU domains | 18 | **25** |
| Local WGSL shaders | 4 | **9** |
| Binaries | 35 | **61** |

---

## Recommended Absorption Priority (Updated)

| Priority | Primitive | Status | Payoff |
|----------|-----------|--------|--------|
| **P1** | `HmmBatchForwardF64` | Shader ready (13/13) | HMM batch forward for sequence analysis |
| **P1** | Fix `BatchedOdeRK4F64` | Shader ready (7/7) | Unblocks all ODE sweep workloads |
| **P1** | NVVM driver profile fix | Bug identified | Unblocks all f64 transcendental shaders |
| **P2** | `AniBatchF64` | Shader ready (7/7) | Pairwise genomic identity |
| **P2** | `SnpCallingF64` | Shader ready (5/5) | Population genomics |
| **P2** | `DnDsBatchF64` | Shader ready (9/9) | Selection analysis |
| **P2** | `PangenomeClassifyGpu` | Shader ready (6/6) | Gene family analysis |
| **P2** | `RfBatchInferenceGpu` | Shader ready (13/13) | ML ensemble at field scale |
| **P3** | `ParallelFilter<T>` | Pattern established | Generalizes quality filtering |
| **P3** | `BatchPairReduce<f64>` | Pattern established | Generalizes DADA2 E-step |

---

## Reproduction

```bash
cd barracuda

# Full test suite (552 tests)
cargo test

# All CPU validation binaries (1,241 checks)
cargo run --release --bin validate_barracuda_cpu_v5  # RF + GBM (29 checks)
cargo run --release --bin validate_barracuda_cpu_v4  # Track 1c (44 checks)
# ... plus 27 more CPU binaries

# GPU validation (260 checks, requires --features gpu)
cargo run --features gpu --release --bin validate_gpu_track1c    # 27 checks
cargo run --features gpu --release --bin validate_cross_substrate # 20 checks
cargo run --features gpu --release --bin validate_gpu_rf          # 13 checks
# ... plus 9 more GPU binaries

# 25-domain benchmark
cargo run --release --bin benchmark_23_domain_timing
```
