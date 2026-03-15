# wetSpring V119 → ToadStool/BarraCUDA Deep Debt Evolution Handoff

**Date:** March 15, 2026
**From:** wetSpring V119 (376 experiments, 5,707+ checks, 1,687 tests, 354 binaries)
**To:** ToadStool/BarraCUDA team
**Authority:** wateringHole (ecoPrimals Core Standards)
**Supersedes:** V118 Absorption Handoff (Mar 15) — archived
**Pins:** barraCuda v0.3.5 (`03986ce`), toadStool S130+, coralReef Phase 10
**License:** AGPL-3.0-or-later

---

## Executive Summary

V119 is a codebase maturity sprint. No new science — this is about evolving from "works" to "idiomatic, discoverable, and ready for ecosystem-wide absorption."

- **Niche architecture**: `niche.rs` self-knowledge + `wetspring-ecology.yaml` BYOB manifest — wetSpring now knows what it is, what it can do, and what it depends on
- **Typed errors**: `VaultError`, `NestError`, `SongbirdError`, `AssemblyError` replace `Result<_, String>` — error handling is now domain-specific and machine-parseable
- **Domain refactoring**: 7 files over 500 LOC split into 26 submodules by domain boundary — net −3,496 lines, zero API changes
- **`primal_names.rs`**: All hardcoded primal name strings replaced with constants — no binary knows any other primal's name at compile time
- **Squirrel AI wired**: `ai.ecology_interpret` capability with graceful degradation — the primal can now ask AI questions without failing if AI is unavailable
- **`proptest` adopted**: 4 property-based tests for stochastic algorithms — Gillespie, bootstrap, diversity, cooperation
- **15 IPC capability domains** / 20 methods registered for Songbird discovery
- **Zero production panics, zero unsafe, zero clippy warnings, zero TODO/FIXME** — codebase is clean

---

## Part 1: What Changed (for absorption analysis)

### Niche Self-Knowledge (`niche.rs` + BYOB)

wetSpring now has a machine-readable self-description:

```rust
pub const NICHE_NAME: &str = "wetspring";
pub const CAPABILITIES: &[&str] = &[
    "science.diversity", "science.qs_model", "science.anderson",
    "science.kinetics", "science.alignment", "science.taxonomy",
    "science.phylogenetics", "science.nmf", "science.timeseries",
    "science.timeseries_diversity", "science.ncbi_fetch",
    "science.full_pipeline", "provenance.begin", "provenance.record",
    "provenance.complete", "brain.observe", "brain.attention",
    "brain.urgency", "metrics.snapshot", "ai.ecology_interpret",
];
pub const DEPENDENCIES: &[NicheDependency] = &[...]; // 7 primals
```

Plus `#[cfg(feature = "json")]` functions for `operation_dependencies()`, `cost_estimates()`, and `ecology_semantic_mappings()`.

**toadStool action:** Consider a `niche` trait or convention that all springs implement. This enables biomeOS to query any primal's capabilities, dependencies, and costs at runtime without hardcoded knowledge.

### Typed Error Enums

Before V119: `Result<_, String>` scattered throughout vault, forge, and IPC code.
After V119:

| Crate | Error Type | Variants | Replaced |
|-------|-----------|:--------:|:--------:|
| barracuda (vault) | `VaultError` | 7 | 5 `String` returns |
| forge (nest) | `NestError` | 6 | 10 methods |
| forge (songbird) | `SongbirdError` | 4 | 3 functions |
| forge (assembly) | `AssemblyError` | 4 | 4 functions |
| barracuda (ipc) | `crate::error::Error` | existing | 1 function |
| barracuda (bio) | `crate::error::Error` | existing | 1 function |

All implement `std::error::Error` + `Display`. `NestError` derives `PartialEq, Eq` for direct comparison in test assertions.

**toadStool action:** When absorbing forge, the typed errors are ready for integration. `NestError` maps directly to Nest transport failures; `SongbirdError` maps to Songbird discovery failures.

### Domain-Organized Refactoring (7 files → 26 submodules)

| Original | LOC | New Structure | Domain Logic |
|----------|:---:|---------------|-------------|
| `streaming_gpu.rs` | 670 | `mod.rs` + `stages.rs` + `analytics.rs` | GPU pipeline session, per-stage ops, streaming analytics |
| `chimera.rs` | 531 | `mod.rs` + `detection.rs` + `kmer_sketch.rs` | Chimera detection algorithm, k-mer similarity |
| `signal.rs` | 532 | `mod.rs` + `peak_detect.rs` + `prominence.rs` + `smoothing.rs` | Peak finding, prominence calculation, (future) smoothing |
| `msa.rs` | 565 | `mod.rs` + `alignment.rs` + `scoring.rs` | MSA algorithms, score-to-distance conversion |
| `mzxml/mod.rs` | 583 | `mod.rs` + `parser.rs` + `types.rs` | XML streaming parser, spectrum types |
| `mzml/decode.rs` | 580 | `mod.rs` + `base64.rs` + `compression.rs` | Binary decode orchestration, base64, zlib |
| `handlers/expanded.rs` | 485 | `expanded.rs` + 6 domain files | Per-domain IPC handlers (kinetics, drug, alignment, taxonomy, phylogenetics, anderson) |

All public APIs preserved. Internal fields use `pub(super)`. Tests remain in `mod.rs`.

**toadStool action:** The per-domain handler structure (`handlers/kinetics.rs`, `handlers/drug.rs`, etc.) is the pattern for all future science handlers. When adding new IPC methods, create a new domain file rather than growing a monolith.

### Hardcoding Elimination (`primal_names.rs`)

```rust
pub const SELF: &str = "wetspring";
pub const BIOMEOS: &str = "biomeos";
pub const SONGBIRD: &str = "songbird";
pub const SQUIRREL: &str = "squirrel";
pub const NESTGATE: &str = "nestgate";
pub const PETALTONGUE: &str = "petaltongue";
pub const RHIZOCRYPT: &str = "rhizocrypt";
```

8 IPC modules and 4 validation binaries updated. No binary or library code contains literal primal name strings anymore.

**toadStool action:** This pattern should be standard across all primals. Each primal knows its own name via `primal_names::SELF` and discovers others via `ipc::discover`. Consider promoting to a shared `primal_names` crate or biomeOS convention.

### Squirrel AI Integration

New capability domain `ecology.ai_assist` with method `ai.ecology_interpret`:

1. `discover_squirrel()` finds the Squirrel socket via `discover_socket("SQUIRREL_SOCKET", "squirrel")`
2. If found, forwards query via JSON-RPC `ai.query` to Squirrel
3. If unavailable/timeout/error, returns `Ok` with status `"unavailable"` / `"timeout"` / `"error"` — **never returns Err**

This graceful degradation pattern (from `provenance.rs`) ensures the primal remains operational even when external AI is down.

**toadStool action:** The graceful degradation pattern is reusable for any optional dependency. Consider making it a trait or helper: `try_external_call(socket, method, params) -> Value` that automatically wraps failures as status objects.

### proptest Adoption

4 property-based tests validate stochastic algorithm invariants:

| Module | Property | Strategy |
|--------|----------|----------|
| `gillespie` | Steady-state convergence for any 2-reaction system | Random rates, populations |
| `bootstrap` | 95% CI contains true mean | Random data vectors |
| `diversity` | Rarefaction monotonicity + Shannon bounds (0 ≤ H ≤ ln(S)) | Random abundance vectors |
| `cooperation` | Total population bounded in cooperation ODE | Random parameters |

**toadStool action:** `proptest` is a powerful tool for validating numerical code that can't be exhaustively tested. Consider adopting for barraCuda's stochastic primitives (Gillespie, Monte Carlo, bootstrap).

### Clone Reduction

- `PhyloTree::into_flat_tree(self)`: zero-copy consuming method for leaf label transfer (avoids cloning when tree is consumed)
- `Box<dyn Fn>` retained for Gillespie `PropensityFn`: documented rationale — heterogeneous reaction systems require dynamic dispatch from JSON-configured models; generics would require a single closure type for all reactions

---

## Part 2: Updated Primitive Consumption (V119)

All V118 primitive consumption unchanged. New additions:

| Domain | New in V119 |
|--------|-------------|
| ipc | `discover_squirrel()`, `primal_names::*` constants |
| capability | `ai.ecology_interpret` (15th domain, 20th method) |
| niche | `CAPABILITIES`, `DEPENDENCIES`, `cost_estimates()`, `ecology_semantic_mappings()` |

---

## Part 3: Updated Absorption Targets (V119)

V118 targets still apply. V119 adds:

| Target | Priority | What | Why |
|--------|:--------:|------|-----|
| Shared `niche` convention | P1 | Machine-readable self-description trait for all primals | Enables biomeOS to query capabilities without hardcoded manifests |
| Shared `primal_names` convention | P2 | Standardized approach to primal name constants | Prevents string drift across springs |
| `proptest` for barraCuda | P3 | Property-based testing for stochastic primitives | Catches edge cases that unit tests miss |
| Typed error hierarchy | P2 | Common error trait or crate for springs and primals | Enables structured error propagation through IPC |

### V118 Targets (still open)

| Target | Priority | Status |
|--------|:--------:|--------|
| `ComputeDispatch` for ODE modules | P3 | Not started |
| DF64 GEMM adoption (`Fp64Strategy::Hybrid`) | P3 | Not started |
| `BandwidthTier` wiring | P3 | Not started |
| 8 GPU primitive opportunities (compose → lean) | P2-P3 | Not started |
| Tolerance standardization | P2 | wetSpring ready, waiting for cross-spring adoption |
| `crate::special` extraction | P3 | Blocked on `barracuda::math` CPU-only feature gate |

---

## Part 4: Learnings for toadStool/barraCuda Evolution

### Niche self-knowledge is the foundation for BYOB

The `niche.rs` pattern proves that a primal can describe itself in code (not just YAML). The `CAPABILITIES` array mirrors `capability_domains.rs` exactly — a test enforces this. This means biomeOS can discover capabilities by asking the primal directly, not by reading a manifest file.

### Typed errors enable structured IPC error codes

With `VaultError::ConsentExpiredOrRevoked` instead of `Err("consent expired".to_string())`, the IPC layer could map error variants to JSON-RPC error codes (e.g., `-32001` for consent errors, `-32002` for blob errors). This is the path to machine-parseable error responses.

### Domain refactoring by boundary, not by size

The 7 refactored files were split by domain boundary (detection vs sketch, stages vs analytics, parser vs types) rather than by arbitrary line count. This means each submodule has a clear reason to exist and a clear API surface. The pattern: `mod.rs` holds types + re-exports + tests, domain files hold implementations.

### `#[expect(reason)]` as documentation

The `#[expect(clippy::expect_used, reason = "validation binary: ...")]` pattern documents WHY a lint is suppressed, not just THAT it is. This is strictly better than `#[allow(clippy::expect_used)]` because the compiler warns if the suppression becomes unnecessary. Note: `reason` is not supported at crate level (`#![...]`) without triggering unfulfilled lint warnings.

### Graceful degradation is a pattern, not a special case

The Squirrel AI handler follows the same pattern as the provenance handler: try the external call, catch any failure, return Ok with a status object. This should be the default for any optional dependency. The primal should never fail because an optional external service is down.

---

## Quality Gates (at handoff time)

| Check | Result |
|-------|--------|
| `cargo fmt --check` | Zero violations |
| `cargo clippy --workspace --all-features` | Zero warnings |
| `cargo test --workspace` | 1,687 passed, 0 failed, 2 ignored |
| `cargo test --features gpu --lib` | 1,404 passed, 3 failed (pre-existing GPU HW tolerance), 42 ignored |
| Production `panic!()` | 0 |
| Production `unsafe` | 0 (`#![forbid(unsafe_code)]`) |
| TODO/FIXME/HACK in production | 0 |
| `Result<_, String>` in library | 8 (down from ~25) |
| Local WGSL shaders | 0 |
| Named tolerances | 200+ |
| barraCuda primitives consumed | 150+ |
| Files over 500 LOC | `mzml/decode/mod.rs` (561) — single remaining, orchestration logic |

---

## Recommended toadStool Actions (V119 priorities)

1. **Define `niche` convention** — standardize machine-readable self-description across all primals (capabilities, dependencies, costs, semantic mappings)
2. **Define shared `primal_names` pattern** — prevent string drift, enable compile-time primal identity
3. **Adopt typed error hierarchy** — `VaultError` / `NestError` patterns are ready for cross-primal error propagation
4. **Wire Squirrel discovery** — the `discover_socket` pattern is proven; ensure Squirrel's IPC endpoint matches the convention
5. **Adopt `proptest`** — validate stochastic barraCuda primitives with property-based testing
6. **Absorb tolerance naming convention** — 200+ constants with scientific provenance, hierarchy-tested (V118 action, still open)
7. **Add `barracuda::math` CPU-only feature gate** — unblock `crate::special` extraction from springs (V118 action, still open)
