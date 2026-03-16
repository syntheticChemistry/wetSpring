# wetSpring V123 — Zero-Panic Validation + Dual-Format Discovery Handoff

**Date:** March 16, 2026
**From:** wetSpring V123
**To:** barraCuda / toadStool / biomeOS teams
**Scope:** Zero-panic validation (groundSpring V109), dual-format capability discovery (neuralSpring/ludoSpring), centralized RPC error extraction (healthSpring V29), Python dep hardening (groundSpring V109)

---

## Executive Summary

V123 absorbs four cross-ecosystem patterns from sibling springs. The biggest change is the
zero-panic validation transformation — 1,039 `.expect()` and 632 `.unwrap()` calls replaced
with `OrExit` across 192 validation/benchmark binaries. IPC layer enriched with dual-format
capability discovery and centralized error extraction.

## Changes (V122 → V123)

### 1. Zero-Panic Validation (`OrExit` trait)

New `validation::OrExit<T>` trait provides panic-free error handling for validation binaries:

```rust
use wetspring_barracuda::validation::OrExit;

let val = barracuda::stats::covariance(&x, &y).or_exit("covariance");
let gpu = GpuF64::new().await.or_exit("GPU init");
```

- `Result<T, E: Display>::or_exit("context")` → stderr + `process::exit(1)`
- `Option<T>::or_exit("context")` → stderr + `process::exit(1)`
- `Validator::finish_with_code()` → returns `ExitCode` without diverging
- `Validator::all_passed()` → `const fn` for composable queries

**Impact:** 192 binaries transformed. Zero `clippy::expect_used`/`clippy::unwrap_used`
lint suppressions remain. Pattern available for adoption by all springs.

### 2. Dual-Format Capability Discovery

`capability.list` now returns Format B alongside Format A:

**Format A (flat, backward-compatible):**
```json
{"capabilities": ["science.diversity", "science.anderson", ...]}
```

**Format B (rich, biomeOS Pathway Learner):**
```json
{
  "operation_dependencies": {"science.diversity": ["abundance_table"], ...},
  "cost_estimates": {"science.diversity": {"latency_ms": 0.5, "cpu": "low", "memory_bytes": 4096}, ...},
  "semantic_mappings": {"diversity": "science.diversity", ...}
}
```

Songbird registration enriched with `niche`, `niche_description`, `required_dependencies`.

### 3. Centralized RPC Error Extraction

New `protocol::extract_rpc_error(response) -> Option<(i64, String)>` replaces ad-hoc
`response.contains("error")` checks. Songbird client migrated.

### 4. Python Dependency Hardening

`requirements.txt` pinned with upper bounds: `numpy>=1.24,<3`, `scipy>=1.12,<2`, etc.

---

## barraCuda Primitive Consumption (V123)

No new primitives consumed. V123 is a quality/infrastructure evolution.

### Existing consumption (unchanged from V122):
- 150+ barraCuda math primitives across stats, linalg, special, ops, nn, spectral, nautilus
- `FusedMapReduceF64` for GPU diversity
- Lanczos eigensolver for Anderson localization
- `BytesMut` zero-copy I/O
- Non-blocking `submit_commands`

### Patterns for absorption (new in V123):
| Pattern | Location | Absorption Target |
|---------|----------|-------------------|
| `OrExit<T>` trait | `validation/mod.rs` | barraCuda test utils (if desired) |
| `extract_rpc_error()` | `ipc/protocol.rs` | barraCuda IPC protocol (or shared crate) |
| Dual-format capability discovery | `ipc/handlers/mod.rs` | toadStool/biomeOS capability parsing |

---

## Evolution Requests to barraCuda

### Track 3 GPU (when ready upstream):
- `SparseGemmF64` — drug repurposing pathway scoring
- `TranseScoreF64` — knowledge graph embedding inference
- `TopK<f64>` — ranked candidate selection

### Stats additions (groundSpring V109 surfaced):
- Upper-bound Python pinning already adopted; no new stats needed

---

## Quality Metrics

| Metric | V122 | V123 |
|--------|------|------|
| Library tests | 1,703 | 1,703 |
| `clippy::expect_used` suppressions | 117+ | **0** |
| `.expect()` in validation binaries | ~1,144 | **0** |
| `.unwrap()` in validation binaries | ~350 | **0** |
| `OrExit` trait | — | ✓ |
| Dual-format discovery | — | ✓ |
| `extract_rpc_error()` | — | ✓ |
| Python dep upper bounds | partial | ✓ |
| Clippy warnings | 0 | 0 |
| `#[allow()]` in codebase | 0 | 0 |
