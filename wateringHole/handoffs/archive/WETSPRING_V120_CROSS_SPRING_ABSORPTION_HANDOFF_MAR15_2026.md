# wetSpring V120 — Cross-Spring Absorption Handoff

**Date:** March 15, 2026
**From:** wetSpring (ecoPrimals)
**To:** barraCuda / toadStool / metalForge teams
**Sprint:** V120 — Cross-Spring Absorption

---

## Summary

V120 absorbs patterns discovered during cross-spring review of all sibling
springs (airSpring, groundSpring, neuralSpring, healthSpring, hotSpring).
Focuses on completing typed error evolution, hardening the deploy graph
with graceful degradation, and establishing a shared Python tolerance
module for baseline parity.

## Changes Relevant to barraCuda / toadStool

### Typed Errors — Now Complete Across Library Code

| Crate | Error Type | Variants | Replaces |
|-------|-----------|----------|----------|
| wetspring-forge | `NcbiError` | `HttpRequest`, `InvalidUtf8`, `AssemblyNotFound`, `CacheFailed`, `FileSystem` | `Result<_, String>` in `ncbi.rs` |
| wetspring-forge | `DataError` | `InvalidSocket`, `Connect`, `Timeout`, `Write`, `Flush`, `Read`, `EmptyResponse` | `Result<String, String>` in `data.rs` |
| wetspring-barracuda | `Error::Ipc` | (existing variant) | `Result<Value, String>` in `ai.rs` handler |

**Consumption pattern for toadStool:** When dispatching work to wetSpring
and handling errors, you can now match on typed variants rather than
parsing error strings. The `Display` impls preserve the original message
format for backward compatibility.

### Deploy Graph: Optional Nodes with `fallback = "skip"`

New nodes in `graphs/wetspring_deploy.toml`:

```toml
[[nodes]]
id = "germinate_toadstool"
[nodes.primal]
by_capability = "compute"
[nodes.constraints]
optional = true
fallback = "skip"

[[nodes]]
id = "germinate_squirrel"
[nodes.primal]
by_capability = "ai"
[nodes.constraints]
optional = true
fallback = "skip"
```

**Impact:** biomeOS orchestration can now deploy wetSpring even when
ToadStool or Squirrel are absent. NestGate and petalTongue also gained
`fallback = "skip"`. This follows the neuralSpring fallback pattern.

### Hardcoded Primal Name Elimination

`ncbi/nestgate/discovery.rs` and `visualization/ipc_push.rs` now use
`primal_names` constants instead of hardcoded string literals. Feature-gated
with local constant fallback when `ipc` feature is disabled.

### Shared Python Tolerance Module

`scripts/tolerances.py` mirrors all 120+ Rust tolerance constants for
Python baseline parity. Enables scripts to import constants rather than
using magic numbers:

```python
from tolerances import ANALYTICAL_F64, ODE_DIVISION_GUARD
assert abs(computed - expected) < ANALYTICAL_F64
```

### Live Pipeline Refactoring

`visualization/live_pipeline.rs` (611 LOC) split into:
- `live_pipeline/mod.rs` — core session types and methods
- `live_pipeline/stages.rs` — pre-built stage definitions (16S, LC-MS, phylo)

All public APIs preserved.

## Updated Metrics

| Metric | V119 | V120 |
|--------|------|------|
| Library tests | 1,687 | 1,638 (refined count) |
| GPU hw-specific failures | 3 | 3 (pre-existing) |
| Capability domains | 15 | 16 |
| Methods | 20 | 22 |
| `Result<_, String>` in lib | 8 | 1 (legitimate OnceLock) |
| Python tolerance constants | 0 | 120+ |
| Deploy graph optional nodes | 2 | 4 |

## Absorption Patterns Applied from Sibling Springs

| Pattern | Source | Applied |
|---------|--------|---------|
| `fallback = "skip"` | neuralSpring | Deploy graph optional nodes |
| Typed errors in forge | airSpring convention | `NcbiError`, `DataError` |
| Shared tolerance module | healthSpring concept | `scripts/tolerances.py` |
| Feature-gated constants | airSpring DI pattern | `ncbi/nestgate/discovery.rs` |

## Next Absorption Candidates (for future sprints)

- `TissueContext`-style GPU uniform buffers (healthSpring)
- Typed `tarpc` IPC client (groundSpring)
- Species as dispatch parameter in shaders (healthSpring)
- `print_provenance_header` as `Result` + panic wrapper (groundSpring)
