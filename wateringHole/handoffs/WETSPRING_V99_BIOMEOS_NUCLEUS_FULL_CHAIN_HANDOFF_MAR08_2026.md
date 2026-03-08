# wetSpring V99 — biomeOS/NUCLEUS Integration + Full Chain Handoff

**Date:** March 8, 2026
**From:** wetSpring
**To:** barraCuda, toadStool, coralReef, biomeOS
**License:** AGPL-3.0-or-later
**Covers:** V98+ → V99 (Exp319–326)

---

## Executive Summary

- wetSpring now runs as a **biomeOS science primal** via JSON-RPC IPC (Unix sockets)
- Deploy graph `wetspring_deploy.toml` added to biomeOS — Tower→ToadStool→wetSpring
- V99 chain: CPU v25 (46/46) → GPU v14 (27/27) → metalForge v17 (29/29) = **102/102 PASS**
- V98 regression: 173/173 PASS (no regressions introduced)
- Total: **300 experiments, 305 binaries, 8,886+ validation checks**

---

## Part 1: biomeOS Integration (Exp321–322)

### Exp321: biomeOS/NUCLEUS V98+ Integration — 42/42 PASS

Validated wetSpring as a deployable biomeOS primal:

| Area | What was validated |
|------|--------------------|
| NUCLEUS env | `XDG_RUNTIME_DIR`, biomeos socket dir, deploy graph existence |
| IPC server | Bind, health.check (5 capabilities), substrate, version |
| Science methods | `science.diversity` (Shannon=ln(4), Simpson=0.75, Bray-Curtis), `science.qs_model` (4 scenarios), `science.full_pipeline` |
| Brain module | `brain.observe`, `brain.attention`, `brain.urgency` |
| Protocol | JSON-RPC 2.0 error codes (-32601), wrong version rejection, empty counts error, 10-request connection multiplexing |
| Metrics | total_calls, success_count, error_count, snapshot via RPC |
| Discovery | Songbird graceful fallback |
| Performance | IPC overhead: ~0.1ms vs sub-µs direct (~2000x), acceptable for orchestration |

### Exp322: Cross-Primal Pipeline — 22/22 PASS

End-to-end cross-primal data flows through biomeOS capability routing:

| Pipeline | Data flow |
|----------|-----------|
| airSpring → wetSpring | FAO-56 ET₀ + Hargreaves + Makkink → QS biofilm dynamics |
| wetSpring → neuralSpring | Bray-Curtis distance → graph Laplacian → effective_rank |
| hotSpring → wetSpring | Anderson 3D → Lanczos eigenvalues → level spacing → phase interpretation |
| groundSpring → wetSpring | Bootstrap CI (5000 resamples) + jackknife → diversity confidence |
| Full pipeline | health → diversity → QS → pipeline → metrics (5-stage IPC) |

---

## Part 2: V99 Chain (Exp323–326)

### Exp323: barraCuda CPU v25 — 46/46 PASS

5 domains, pure Rust:

| Domain | Spring | Checks | Key validations |
|--------|--------|:------:|-----------------|
| D55: Bio | wetSpring | 12 | Shannon, Simpson, Chao1, Pielou, Felsenstein, HMM, QS, cooperation, BC |
| D56: Cross-Spring | airSpring+hotSpring+neuralSpring | 10 | 6 ET₀ methods, Anderson spectral, graph Laplacian, effective_rank |
| D57: Statistics | groundSpring | 10 | mean, variance, Pearson, Spearman, jackknife, bootstrap CI, fit_linear |
| D58: Precision | hotSpring+barraCuda | 6 | erf, norm_cdf, NMF (KL divergence) |
| D59: IPC Math | biomeOS+wetSpring | 8 | Pipeline math, QS scenarios, BC condensed, PCoA |

### Exp324: barraCuda GPU v14 — 27/27 PASS

4 domains, GPU parity:

| Domain | Spring | Checks | Key validations |
|--------|--------|:------:|-----------------|
| G26: Diversity GPU | wetSpring | 13 | 4×Shannon, 4×Simpson, BC condensed — all FusedMapReduceF64 |
| G27: Anderson | hotSpring+neuralSpring | 4 | Weak/strong disorder, phase interpretation |
| G28: Cross-Domain | all Springs | 5 | Var, Pearson, jackknife GPU→CPU statistics |
| G29: ToadStool Dispatch | toadStool+barraCuda | 5 | fp64_strategy, 10K-element Shannon, dot/Hybrid skip, provenance (28 shaders) |

### Exp326: metalForge v17 — 29/29 PASS

5 domains, NUCLEUS + biomeOS:

| Domain | Spring | Checks | Key validations |
|--------|--------|:------:|-----------------|
| MF22: Diversity | wetSpring | 6 | Soil/pharma Shannon, BC, Simpson, Chao1 |
| MF23: Cross-Primal | airSpring+hotSpring+wetSpring | 5 | ET₀, Anderson, QS, cooperation |
| MF24: Statistics | groundSpring | 5 | mean, bootstrap CI, Pearson, fit_linear, erf |
| MF25: NUCLEUS Probes | biomeOS | 5 | XDG_RUNTIME_DIR, Tower/Node/Nest socket scan, biomeOS binary |
| MF26: biomeOS Graph | biomeOS+all Springs | 8 | Deploy graphs, capability_registry, pipeline math, JK, cross-track |

---

## Part 3: Deploy Graph

`biomeOS/graphs/wetspring_deploy.toml`:

```
Tower (BearDog + Songbird)
  → ToadStool (GPU compute, optional)
    → wetSpring (science primal)
      → validate_wetspring_atomic (health_check)
```

9 capabilities mapped:

| Capability | JSON-RPC method |
|------------|----------------|
| `science.diversity` | `science.diversity` |
| `science.anderson` | `science.anderson` |
| `science.qs_model` | `science.qs_model` |
| `science.ncbi_fetch` | `science.ncbi_fetch` |
| `science.full_pipeline` | `science.full_pipeline` |
| `brain.observe` | `brain.observe` |
| `brain.attention` | `brain.attention` |
| `brain.urgency` | `brain.urgency` |
| `metrics.snapshot` | `metrics.snapshot` |

---

## Part 4: Cross-Spring Shader Evolution

### wetSpring-authored shaders consumed by ecosystem

| Shader | Consumer Springs |
|--------|-----------------|
| `fused_map_reduce_f64.wgsl` | airSpring, hotSpring, neuralSpring |
| `bray_curtis_f64.wgsl` | airSpring |
| `anderson_3d_f64.wgsl` | hotSpring, groundSpring |

### Shaders consumed by wetSpring from other springs

| Shader | Origin | Used for |
|--------|--------|----------|
| `df64_core.wgsl` | hotSpring | FP32-core f64 emulation |
| `gemm_f64.wgsl` | neuralSpring | NMF drug repurposing |
| `precision_routing.wgsl` | barraCuda | Fp64Strategy::Hybrid detection |

---

## Part 5: Recommendations for Upstream

### barraCuda

1. **`FitResult` API**: Consider exposing `.slope()` and `.intercept()` methods alongside `.params[0]`/`.params[1]` — current API requires index knowledge.
2. **NMF module path**: `barracuda::linalg::nmf` is consistent but `wetspring_barracuda::bio::nmf` doesn't exist — consider re-exporting for domain discovery.
3. **HmmModel fields**: `log_pi`/`log_trans`/`log_emit` naming is technically correct (log-space) but undiscoverable — doc aliases would help.

### toadStool

1. **Absorb wetspring_deploy.toml pattern**: biomeOS now has a science primal deploy graph; toadStool should reference this for `capability_call` routing to Springs.
2. **IPC overhead is 2000x direct**: For latency-sensitive paths, consider batched `capability_call` or streaming JSON-RPC.
3. **Provenance tracking**: 28 shaders in registry, 22 cross-spring — the provenance API works well for tracking Write→Absorb→Lean cycles.

### biomeOS

1. **wetspring_deploy.toml committed**: The deploy graph is in `phase2/biomeOS/graphs/`. Ready for `biomeos deploy graphs/wetspring_deploy.toml`.
2. **Capability registry up to date**: `config/capability_registry.toml` already has `science.diversity`, `science.anderson`, etc. mapped to `wetspring`.
3. **Brain module**: `brain.observe/attention/urgency` registered — biomeOS can route attention/urgency queries.

### coralReef

No specific action items — coralReef shader compilation works transparently via toadStool proxy. The `Fp64Strategy::Hybrid` path on consumer GPUs (RTX 4070) correctly routes through DF64 for FusedMapReduceF64.

---

## Part 6: Status Summary

| Metric | Value |
|--------|-------|
| Experiments | 300 |
| Binaries | 305 |
| Validation checks | 8,886+ |
| Tests | 1,047 |
| Papers reproduced | 52 |
| V99 chain | 166/166 PASS |
| V98 regression | 173/173 PASS |
| biomeOS capabilities | 9 registered |
| Cross-spring shaders | 28 (22 cross-spring) |
| Upstream pins | barraCuda `a898dee`, toadStool S130+ `bfe7977b`, coralReef Iter 10 `d29a734` |

---

**License:** AGPL-3.0-or-later
