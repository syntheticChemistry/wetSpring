# Ecosystem Leverage Guide — wetSpring V128

> **V145 note (Apr 17, 2026):** This guide was written at V128. Current
> metrics: **363** binaries, **1,700** tests, **150+** barraCuda primitives
> (v0.3.12), **42** niche capabilities, **22** barraCuda IPC consumed
> capabilities (primal proof Level 5). Exp403: Tier 2 IPC-WIRED validation
> (live UDS to 5 primals). PG-09: barraCuda IPC evaporation surface.
> See `handoffs/WETSPRING_V145_PRIMAL_PROOF_TIER2_HANDOFF_APR17_2026.md`
> and `docs/PRIMAL_GAPS.md` for current composition status.

**Date:** March 18, 2026
**Scope:** What wetSpring absorbs from the ecoPrimals ecosystem and what it contributes back.

---

## Absorption Surface — What wetSpring Leverages

### barraCuda (Pure Math Primal)

| Primitive | wetSpring Usage |
|-----------|-----------------|
| `FusedMapReduceF64` | Shannon, Simpson, observed, evenness, Bray-Curtis (GPU) |
| `GemmF64` | Pairwise spectral cosine similarity (GPU) |
| `BatchedEighGpu` | PCoA ordination (GPU eigendecomposition) |
| `KrigingF64` | Spatial interpolation |
| `VarianceF64`, `CorrelationF64`, `CovarianceF64`, `WeightedDotF64` | Statistical compute (GPU) |
| `BrayCurtisF64` | Beta diversity GPU path |
| `QualityFilterGpu` | Quality filtering shader |
| `Dada2EStepGpu` | DADA2 denoising E-step (GPU) |
| `kahan_sum` | Compensated summation — delegated, zero local math |
| `WgpuDevice` | GPU device abstraction |

**Unwired but available:** `SparseGemmF64`, `TranseScoreF64`, `TopK`, `LogsumexpWgsl`, `monod()`, `haldane()`.

### toadStool (Compute Orchestration)

wetSpring's `metalForge/forge` modules (`bridge`, `dispatch`, `probe`, `inventory`, `streaming`) form the absorption seam. The `compute_dispatch.rs` module implements the `DispatchOutcome<T>` protocol for clean transport vs application error separation.

### sweetGrass (Resilience Patterns)

| Pattern | File |
|---------|------|
| `RetryPolicy` (exponential backoff, jitter) | `ipc/resilience.rs` |
| `CircuitBreaker` (threshold, cooldown) | `ipc/resilience.rs` |

### healthSpring (Operational Standards)

| Pattern | File |
|---------|------|
| `health.liveness` probe | `ipc/handlers/mod.rs` |
| `health.readiness` probe | `ipc/handlers/mod.rs` |
| `PRIMAL_NAME` / `PRIMAL_DOMAIN` | `lib.rs` |

### airSpring (Code Quality Patterns)

| Pattern | File |
|---------|------|
| `cast` module (9 safe numeric helpers) | `cast.rs` — adopted V128 |
| `mul_add()` FMA sweep | Multiple files — adopted V128 |
| `OrExit<T>` zero-boilerplate exit | `validation/mod.rs` |

### groundSpring (Ecosystem Compliance)

| Pattern | File |
|---------|------|
| `deny.toml` C-dep ban list (ecoBin) | `deny.toml` — adopted V128 |
| `unlicensed = "deny"` policy | `deny.toml` |
| `yanked = "deny"` policy | `deny.toml` |

### rhizoCrypt (Provenance)

| Pattern | File |
|---------|------|
| Session-scoped provenance DAGs | `ipc/provenance.rs` |
| `provenance.begin` / `record` / `complete` | `ipc/dispatch.rs` |

### Squirrel (AI Integration)

| Pattern | File |
|---------|------|
| 8 typed MCP tool definitions | `ipc/mcp.rs` |
| `ai.ecology_interpret` dispatch | `ipc/dispatch.rs` |

---

## Contribution Surface — What wetSpring Gives Back

### Patterns Pioneered

| Pattern | Adopted By |
|---------|------------|
| `DispatchOutcome<T>` error separation | groundSpring V112+, airSpring v0.9+ |
| Python baseline provenance registry | Unique to wetSpring; recommended for all springs |
| `OnceLock` GPU probe (prevent SIGSEGV) | groundSpring V113+ |
| Centralized tolerance registry | Recommended upstream for barraCuda |
| MCP tool definitions for AI | Recommended for all primals |
| `ChaosEngine` fault injection testing | Used in resilience test suite |

### Active Handoffs to Upstream

- **barraCuda:** `monod()` + `haldane()` absorption candidates, tolerance registry recommendation
- **toadStool:** Forge module absorption seam, session-level provenance tracking
- **Ecosystem:** IPC resilience patterns, MCP integration, `FAMILY_ID`-aware socket discovery

---

## Discovery Architecture

wetSpring discovers all primals at runtime via capability-based socket discovery:

```
FAMILY_ID / BIOMEOS_FAMILY_ID → socket name suffix (default: "default")
{PRIMAL}_SOCKET env override → XDG_RUNTIME_DIR/biomeos/ → temp_dir fallback
```

No hardcoded primal knowledge beyond `PRIMAL_NAME` ("wetspring") and `PRIMAL_DOMAIN` ("science.ecology"). All inter-primal communication uses JSON-RPC 2.0 over Unix sockets with the 4-format capability standard.

---

## Compliance Posture

| Item | Status |
|------|--------|
| `#![forbid(unsafe_code)]` | Enforced crate-wide |
| `unlicensed = "deny"` | Active in deny.toml |
| `yanked = "deny"` | Active in deny.toml |
| C-dep ban list | 14 banned crates in deny.toml |
| SCYBORG Provenance Trio | AGPL-3.0 + ORC + CC-BY-SA 4.0 |
| Zero external math | All math via barraCuda or sovereign `special` module |
| Zero mocks in production | All mocks isolated to `#[cfg(test)]` |
| Proptest coverage | 7 modules (bio + IPC) |
