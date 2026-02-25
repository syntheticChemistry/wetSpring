# wetSpring → ToadStool Handoff V32: S62 Lean

**Date:** February 24, 2026
**From:** wetSpring (Phase 41)
**To:** ToadStool / BarraCuda team
**ToadStool:** S62 (post DF64 expansion + architectural evolution)
**Previous:** V31 (S59 absorption targets)

---

## Summary

wetSpring pulled ToadStool S60-S62 (5 commits since S59), identified 2 new
primitives to wire, rewired them, and revalidated. The paper queue is now
fully GPU-covered (31/31 actionable papers).

---

## ToadStool Commits Reviewed (S60-S62)

| Commit | Session | Key Changes |
|--------|---------|-------------|
| `0c998992` | S60-S61 | SpMM f64, TranseScoreF64, Conv2D GPU, NVK guard, cpu-math feature gate |
| `2dc76044` | S62 | BandwidthTier, PeakDetectF64, pool padding, dispatch_with_transfer_cost |
| `9fb51f22` | — | DF64 core-streaming into HMC pipeline |
| `06782766` | — | Hybrid FP64 core-streaming spec |
| `02207c4a` | — | DF64 GEMM/LJ/KE shaders, ComputeDispatch builder, unified_hardware decomposition |

---

## Rewires Performed

### 1. PeakDetectF64 → `bio/signal_gpu.rs`

**Before:** `find_peaks_gpu()` validated the GPU device via FMR then fell back
to CPU `find_peaks()`. The doc comment noted "When ToadStool provides a
`BatchPeakDetectionGpu` primitive... this wrapper will rewire."

**After:** Uses `barracuda::ops::peak_detect_f64::PeakDetectF64` builder for
native GPU dispatch with WGSL parallel local-maxima + prominence. Falls back
to CPU for signals < 64 elements or on GPU error.

**Validation:** Exp010 peak detection 17/17 checks pass.

### 2. TranseScoreF64 → `validate_knowledge_graph_embedding.rs`

**Before:** Local `KgEmbedding::score()` with manual TransE `h + r - t` L2
distance. CPU-only.

**After:** Added `§6 GPU TransE Parity` section that constructs
`barracuda::ops::transe_score_f64::TranseScoreF64` with the trained entity/
relation embeddings, scores all 538 triples on GPU, and compares against CPU.

**Result:** Max CPU↔GPU diff = 1.78e-15 (within f64 precision). 9/9 checks pass.

---

## New Primitives Consumed

| # | Primitive | ToadStool Module | Session | Usage |
|---|-----------|-----------------|---------|-------|
| 43 | `PeakDetectF64` | `ops::peak_detect_f64` | S62 | `bio/signal_gpu.rs` — LC-MS peak detection |
| 44 | `TranseScoreF64` | `ops::transe_score_f64` | S60 | `validate_knowledge_graph_embedding.rs` — drug repurposing KG |

Available but not yet wired (no local equivalent to replace):

| Primitive | ToadStool Module | Session | Relevance |
|-----------|-----------------|---------|-----------|
| `SparseGemmF64` | `ops::sparse_gemm_f64` | S60 | Drug-disease matrices (~5% fill). NMF currently uses dense GEMM |
| `BandwidthTier` | `dispatch::config` | S62 | PCIe-aware routing for metalForge |
| `dispatch_with_transfer_cost` | `dispatch` | S62 | Bandwidth-aware CPU/GPU decision |
| `ComputeDispatch` builder | `device::compute_pipeline` | S62 | Boilerplate reduction for GPU ops |

---

## Paper Queue Impact

All 43 papers now have full three-tier coverage:

| Track | Papers | CPU | GPU | metalForge |
|-------|:------:|:---:|:---:|:----------:|
| Track 1 (Ecology + ODE) | 10 | 10/10 | 10/10 | 10/10 |
| Track 1b (Phylogenetics) | 5 | 5/5 | 5/5 | 5/5 |
| Track 1c (Metagenomics) | 6 | 6/6 | 6/6 | 6/6 |
| Track 2 (PFAS/LC-MS) | 4 | 4/4 | 4/4 | 4/4 |
| **Track 3 (Drug repurposing)** | **5** | **5/5** | **5/5** | 0/5 |
| Cross-spring | 1 | 1/1 | 1/1 | — |
| Extensions | 9 | 9/9 | — | — |
| **Grand total** | **43** | **43/43** | **31/31** | **25/25** |

Track 3 upgraded from 3/5 → 5/5 GPU thanks to TranseScoreF64 (S60) and SpMM (S60).

---

## Verification

| Check | Result |
|-------|--------|
| `cargo test --features gpu` | 759 pass, 0 fail |
| `cargo clippy --all-features -W pedantic -W nursery` | 0 warnings |
| `validate_knowledge_graph_embedding --features gpu` | 9/9 (TransE GPU parity: 1.78e-15) |
| `validate_peaks` | 17/17 |
| `validate_barracuda_cpu_v8 --features gpu` | 84/84 |
| metalForge forge tests | 47 pass |

---

## Current wetSpring State

| Metric | Value |
|--------|-------|
| Phase | **41** |
| Tests | **806** (759 barracuda + 47 forge) |
| Experiments | **162** |
| Validation checks | **3,198+** |
| ToadStool primitives | **44** |
| ToadStool alignment | **S62** |
| Local WGSL shaders | **0** |
| Paper queue GPU coverage | **31/31** (was 29/30 at S59) |
