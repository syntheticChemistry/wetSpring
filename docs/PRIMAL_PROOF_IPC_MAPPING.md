<!-- SPDX-License-Identifier: CC-BY-SA-4.0 -->

# wetSpring — Primal Proof IPC Mapping

**Last Updated:** May 16, 2026 (V171 — Live composition health: runtime probing replaces deferred checks. Wave 20 schema parity self-check. Registry 452.)

When `--features primal-proof` is enabled, wetSpring routes cross-primal calls
through IPC rather than linking primal libraries. This document maps each domain
operation to its IPC method, precision query, and fallback behavior.

---

## Tier 2 Pre-Flight Methods

| IPC Method | Module | What It Does |
|------------|--------|-------------|
| `toadstool.validate` | `ipc/toadstool_validate.rs` | Pre-flight workload capability check before dispatch |
| `toadstool.list_workloads` | `ipc/toadstool_validate.rs` | Query registered workloads by filter |
| `barracuda.precision.route` | `ipc/precision_route.rs` | Domain-specific precision strategy advisory |
| `compute.dispatch.submit` | `ipc/compute_dispatch.rs` | Submit GPU compute to barraCuda |
| `compute.performance_surface` | `ipc/performance_surface.rs` | Query performance characteristics |

---

## Domain Operation → Precision Mapping

Each science domain maps to a `barracuda.precision.route` query. The `domain`
parameter selects the operation class; `hardware_hint` is discovered at runtime.

| Science Domain | Route `domain` | Typical Precision | Tolerance |
|----------------|---------------|-------------------|-----------|
| Diversity (Shannon, Simpson) | `stats.diversity` | F64 | 1e-9 |
| Alignment (Smith-Waterman) | `bio.alignment` | F64 | exact |
| Taxonomy (Naive Bayes) | `ml.classification` | F64 | 1e-6 |
| Phylogenetics (Robinson-Foulds) | `bio.phylogenetics` | exact integer | 0 |
| NMF (drug-disease) | `linalg.nmf` | F64 | 1e-8 |
| Anderson spectral | `physics.spectral` | DF64 | 1e-10 |
| QS/biofilm ODE | `bio.ode` | F64 | 1e-9 |
| Kinetics (Gompertz) | `bio.kinetics` | F64 | 1e-6 |
| Time series | `stats.timeseries` | F64 | 1e-9 |
| LTEE mutation accumulation | `bio.genomics` | F64 | 1e-11 |

### How routing works

1. Caller invokes `precision_route::route(domain, hardware_hint)`
2. barraCuda returns `PrecisionAdvice` (recommended tier, FMA safety, compiler needs)
3. Caller selects shader/compute path based on advice
4. If barraCuda unavailable, fall back to F64 CPU (safe default)

---

## Cross-Primal IPC Surface (19 methods)

| Method | Module | Feature Gate |
|--------|--------|-------------|
| `health.check` | `ipc/handlers.rs` | `ipc + barracuda-lib` |
| `health.liveness` | `ipc/handlers.rs` | `ipc + barracuda-lib` |
| `health.readiness` | `ipc/handlers.rs` | `ipc + barracuda-lib` |
| `science.diversity` | `ipc/handlers.rs` | `ipc + barracuda-lib` |
| `science.anderson` | `ipc/handlers.rs` | `ipc + barracuda-lib` |
| `science.qs_model` | `ipc/handlers.rs` | `ipc + barracuda-lib` |
| `science.ncbi_fetch` | `ipc/handlers.rs` | `ipc + barracuda-lib` |
| `science.full_pipeline` | `ipc/handlers.rs` | `ipc + barracuda-lib` |
| `science.kinetics` | `ipc/handlers.rs` | `ipc + barracuda-lib` |
| `science.alignment` | `ipc/handlers.rs` | `ipc + barracuda-lib` |
| `science.taxonomy` | `ipc/handlers.rs` | `ipc + barracuda-lib` |
| `science.phylogenetics` | `ipc/handlers.rs` | `ipc + barracuda-lib` |
| `science.nmf` | `ipc/handlers.rs` | `ipc + barracuda-lib` |
| `science.timeseries` | `ipc/handlers.rs` | `ipc + barracuda-lib` |
| `science.timeseries_diversity` | `ipc/handlers.rs` | `ipc + barracuda-lib` |
| `provenance.begin` | `ipc/provenance.rs` | `ipc` |
| `provenance.record` | `ipc/provenance.rs` | `ipc` |
| `provenance.complete` | `ipc/provenance.rs` | `ipc` |
| `security.audit_log` | `ipc/skunkbat.rs` | `ipc` |

---

## Outbound IPC Clients (Tier 2)

These modules call external primals via IPC. Gated behind `#[cfg(feature = "ipc")]`.

| Target Primal | Client Module | Methods Called |
|---------------|--------------|---------------|
| toadStool | `ipc/toadstool_validate.rs` | `toadstool.validate`, `toadstool.list_workloads` |
| barraCuda | `ipc/precision_route.rs` | `barracuda.precision.route` |
| barraCuda | `ipc/compute_dispatch.rs` | `compute.dispatch.submit` |
| barraCuda | `ipc/performance_surface.rs` | `compute.performance_surface` |
| barraCuda | `ipc/barracuda_route.rs` | Compute routing strategy |
| skunkBat | `ipc/skunkbat.rs` | `security.audit_log` |
| Songbird | `ipc/songbird.rs` | Capability resolution |

---

## Feature Gate Matrix

| Feature | What links | IPC clients | IPC server |
|---------|-----------|-------------|------------|
| `(none)` | lib only | no | no |
| `ipc` | lib + clients | yes | no |
| `ipc + barracuda-lib` | lib + clients + server | yes | yes |
| `primal-proof` | lib + clients (IPC-first) | yes | no |

When `primal-proof` is on without `barracuda-lib`, all science calls route
through IPC to a deployed barraCuda primal. The library fallback is never linked.

---

## Fallback Behavior

All IPC clients follow the same pattern:

1. `discover()` — find socket via env var or XDG runtime directory
2. If not found → return `Err(No{Primal})` — caller falls back to library path
3. If found → `rpc_call()` with timeout from `ipc/timeouts.rs`
4. Parse typed response or return `Err(Protocol|Transport)`

No `unsafe`. No hardcoded paths. `family_id`-aware socket names.
