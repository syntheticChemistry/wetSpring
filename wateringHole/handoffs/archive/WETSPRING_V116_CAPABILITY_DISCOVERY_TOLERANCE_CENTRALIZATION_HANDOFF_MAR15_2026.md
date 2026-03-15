# wetSpring V116 → BarraCUDA/ToadStool Capability Discovery + Tolerance Centralization Handoff

**Date:** March 15, 2026
**From:** wetSpring V116 (376 experiments, 5,707+ checks, 1,662 tests, 354 binaries)
**To:** BarraCUDA/ToadStool team
**Authority:** wateringHole (ecoPrimals Core Standards)
**Supersedes:** V115 Deep Audit Evolution Handoff (Mar 15)
**Pins:** barraCuda v0.3.5 (`03986ce`), toadStool S130+, coralReef Phase 10
**License:** AGPL-3.0-or-later

---

## Executive Summary

- **`capability.list` JSON-RPC handler implemented** — any primal or orchestrator can query wetSpring's full capability set at runtime
- **Capability domains expanded** from 11 ecology-only to 14 domains across 4 families (ecology, provenance, brain, metrics) with 19 methods total
- **15 inline tolerance literals centralized** across 10 validation binaries — zero remaining bare tolerance constants in validation code
- **3 validation binaries refactored** from hardcoded primal names/paths to capability-based discovery with standard fallback chain
- **metalForge forge lint parity** achieved — `missing_docs`, `pedantic`, `nursery` match barracuda strictness
- **All 31 IPC tests pass**, zero clippy warnings, 19 files changed (342 insertions, 130 deletions)

---

## Part 1: capability.list Handler (Spring-as-Niche Standard)

### What was implemented

wetSpring now responds to `capability.list` JSON-RPC calls, returning its full
capability manifest at runtime. This is the capability-based discovery primitive
that Songbird and biomeOS orchestrators use to locate science primals.

**dispatch.rs** — new match arm:
```rust
"capability.list" => handlers::handle_capability_list(),
```

**handlers/mod.rs** — response structure:
```json
{
  "primal": "wetSpring",
  "version": "0.1.0",
  "domain": "ecology",
  "capabilities": ["health.check", "capability.list", "science.diversity", ...],
  "domains": [
    {
      "name": "ecology.diversity",
      "description": "Alpha/beta diversity metrics (Shannon, Simpson, Chao1, Bray-Curtis)",
      "methods": ["science.diversity"]
    },
    ...
  ]
}
```

### toadStool action

When toadStool implements `capability.list` for its own IPC surface, adopt the
same response shape: `{ primal, version, domain, capabilities, domains }`.
This enables uniform primal introspection across the ecosystem.

### barraCuda action

If barraCuda exposes math capabilities via IPC (e.g., `math.gemm`, `math.eigh`,
`shader.compile`), register them using the same domain hierarchy. Natural domains:
- `math.linalg` — GEMM, eigh, SVD, Cholesky
- `math.stats` — variance, correlation, covariance, MapReduce
- `math.spatial` — Kriging, interpolation
- `shader.*` — compile, list, provenance

---

## Part 2: Capability Domain Expansion (11 → 14 domains, 19 methods)

### capability_domains.rs changes

| Family | Domains | Methods |
|--------|:-------:|:-------:|
| ecology | 11 | 11 |
| provenance | 1 | 3 (begin, record, complete) |
| brain | 1 | 3 (observe, attention, urgency) |
| metrics | 1 | 1 (snapshot) |
| **Total** | **14** | **19** |

New test coverage:
- `all_domains_have_recognised_prefix` — validates 4 domain families
- `domains_cover_all_four_families` — ensures ecology + provenance + brain + metrics
- `total_capability_count_matches_registry` — 19 methods match `capability_registry.toml`
- `all_methods_returns_flat_list` — `all_methods()` introspection

### toadStool action

The provenance domain methods (`provenance.begin`, `provenance.record`,
`provenance.complete`) are wetSpring's interface to the provenance trio. When
toadStool proxies these (e.g., for GPU compute provenance), the method names
and domain structure should be consistent.

---

## Part 3: Inline Tolerance Centralization (15 replacements, 10 binaries)

### Pattern: inline literal → named constant

| Binary | Before | After |
|--------|--------|-------|
| `validate_barracuda_cpu_v26.rs` | `1e-10` | `tolerances::PYTHON_PARITY` |
| `benchmark_python_vs_rust_v5.rs` | `0.001` | `tolerances::ODE_DEFAULT_DT` |
| `validate_anderson_qs_environments_v1.rs` | `1e-12`, `1e-15` | `tolerances::ANALYTICAL_F64`, `tolerances::EXACT_F64` |
| `validate_phage_defense.rs` | `1e-10` | `tolerances::PYTHON_PARITY` |
| `validate_pure_gpu_complete.rs` | `1e-5`, `1e-6` | `tolerances::GEMM_GPU_MAX_ERR`, `tolerances::GPU_VS_CPU_F64` |

Total: 15 replacements, zero remaining inline tolerance literals in validation code.

### barraCuda action

When barraCuda tests use tolerance thresholds, centralize them similarly.
wetSpring's `tolerances` module (180+ constants) is organized by domain:
bio, gpu, spectral, instrument, analytical. Each constant has a doc comment
explaining its scientific justification.

---

## Part 4: Capability-Based Primal Discovery (3 binaries)

### Pattern: hardcoded name → runtime discovery

Three validation binaries were refactored from:
```rust
// Before: hardcoded primal name
if service.name == "NestGate" { ... }
let socket = format!("/tmp/beardog-{}.sock", family);
```

To:
```rust
// After: capability-based discovery
if service.capability == "data.ncbi" { ... }
fn discover_socket(env_var: &str, primal: &str) -> PathBuf {
    // env var → XDG_RUNTIME_DIR → BIOMEOS_SOCKET_DIR → temp_dir()
}
```

### toadStool action

This is the standard primal discovery pattern for the ecosystem. A primal only
knows its own identity and discovers others at runtime via:
1. Environment variable (e.g., `BEARDOG_SOCKET`, `TOADSTOOL_SOCKET`)
2. `$XDG_RUNTIME_DIR/{primal}.sock`
3. `$BIOMEOS_SOCKET_DIR/{primal}.sock`
4. `std::env::temp_dir()/{primal}.sock`

toadStool should ensure its socket follows this pattern and that any hardcoded
references in dispatch code use the same fallback chain.

---

## Part 5: 8 GPU Primitive Opportunities (from V115 audit, carried forward)

These were identified in V115 and remain actionable for barraCuda/toadStool:

| # | Primitive | wetSpring Module | Current State | Proposed barraCuda Op |
|---|-----------|-----------------|---------------|----------------------|
| 1 | SparseGemmF64 | `nmf.rs` | CPU matmul (dense) | Sparse GEMM for rank-k NMF |
| 2 | AdaptiveOdeGpu | `ode.rs` | `rk45_integrate` CPU | RK45 adaptive step-size on GPU |
| 3 | BatchedChimeraGpu | `chimera_gpu.rs` | Compose (multi-op) | Fused chimera scoring shader |
| 4 | BatchedReconciliationGpu | `reconciliation_gpu.rs` | Compose (multi-op) | Fused DTL reconciliation |
| 5 | GillespieBatchedSSA | `gillespie_gpu.rs` | Single-trajectory GPU | Multi-trajectory batched SSA |
| 6 | BatchedNewickGpu | `unifrac_gpu.rs` | CPU tree traversal | GPU parallel tree operations |
| 7 | NanoporeBasecallGpu | `io/nanopore.rs` | CPU signal bridge | GPU-accelerated basecalling |
| 8 | EsnGpuInference | `esn.rs` | NPU-targeted int8 | GPU fallback for non-NPU |

### barraCuda action

Priorities 1–3 are highest impact (NMF is compute-hot in Track 3 drug
repurposing; adaptive ODE unlocks stiff biological systems; chimera is
pipeline-critical in DADA2). The ops should be f64-canonical WGSL with
`compile_shader_universal()` and `PrecisionRoutingAdvice`.

---

## Part 6: Audit False-Positive Resolution

The V115 audit initially reported:
- 4 instances of `panic!()` in production code
- ~20 instances of `#[expect(clippy::unwrap_used)]` in production code

**Resolution:** All instances are exclusively within `#[cfg(test)] mod tests`
blocks. Production code has zero violations. The crate-level
`#![forbid(unsafe_code)]`, `#![deny(clippy::expect_used, unwrap_used)]`
enforcements are working as intended.

### toadStool/barraCuda action

When auditing your own crates, note that `#[expect()]` on `mod tests` is the
correct pattern — it suppresses clippy inside tests while the crate-level
`#![deny]` enforces the rule in production code.

---

## Quality Gates

| Check | Result |
|-------|--------|
| `cargo check --workspace` | Clean |
| `cargo clippy --workspace -- -D warnings` | Zero warnings (pedantic + nursery) |
| `cargo test --features gpu,ipc,json -p wetspring-barracuda` | 31 IPC tests pass |
| Files changed | 19 (342 insertions, 130 deletions) |
| Inline tolerance literals | 0 remaining |
| Hardcoded primal names | 0 remaining |
| Production `panic!()`/`unwrap()` | 0 |

---

## Files Changed

```
barracuda/src/ipc/capability_domains.rs   — domain expansion + tests
barracuda/src/ipc/dispatch.rs             — capability.list route + tests
barracuda/src/ipc/handlers/mod.rs         — handle_capability_list() + CAPABILITIES
metalForge/forge/src/lib.rs               — lint parity (missing_docs, pedantic, nursery)
metalForge/forge/src/nest/discovery.rs    — doc correction (temp_dir fallback)
barracuda/src/bin/validate_barracuda_cpu_v26.rs       — tolerance centralization
barracuda/src/bin/benchmark_python_vs_rust_v5.rs      — tolerance centralization
barracuda/src/bin/validate_anderson_qs_environments_v1.rs — tolerance centralization
barracuda/src/bin/validate_phage_defense.rs           — tolerance centralization
barracuda/src/bin/validate_pure_gpu_complete.rs       — tolerance centralization
(+ 5 more validation binaries)
barracuda/src/bin/validate_primal_pipeline_v1.rs      — capability-based discovery
barracuda/src/bin/validate_workload_routing_v1.rs     — capability-based discovery
barracuda/src/bin/validate_petaltongue_live_v1.rs     — capability-based discovery
```
