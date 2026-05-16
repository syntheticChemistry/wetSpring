# wetSpring V169 — Wave 17 Neural API Signal Adoption Handoff

**From:** wetSpring
**To:** primalSpring (coordination), primal teams, spring teams
**Date:** May 16, 2026
**Audit:** Response to primalSpring Wave 17 Cross-Cutting directive

---

## Summary

V169 adopts the Wave 17 Neural API Signal Elevation. `nest.store` and
`nest.commit` signal dispatch are wired with automatic fallback to pre-v3.56
multi-call sequences. `primal.announce` and `signal.dispatch` added to
consumed capabilities. Registry synced to 451 methods. GAP-GS-015 verified.
7/7 CI cross-sync integration tests pass. 252 lib tests pass.

---

## What Was Adopted

### 1. `nest.store` Signal Dispatch (composition collapse)

**File:** `barracuda/src/facade/provenance.rs` — `try_tier2_inner()`

The 5-call provenance sequence:
```
dag.session.create → dag.event.append → dag.dehydrate → session.commit → braid.create
```
now collapses to a single `signal.dispatch("nest.store", ...)` call when
biomeOS v3.56+ is available. Automatic fallback to the multi-call sequence
for older biomeOS.

New function: `try_nest_store_signal()` — sends signal with session type,
witnesses, agents, and result/params hashes. Returns `Some` on success
(session_id or braid_id present in response), `None` to trigger fallback.

### 2. `nest.commit` Signal Dispatch (session finalization)

**File:** `barracuda/src/ipc/provenance/mod.rs` — `complete_session()`

The 3-phase completion sequence:
```
dehydrate → commit → braid.create
```
now collapses to a single `signal.dispatch("nest.commit", ...)` call.
Automatic fallback to the multi-call sequence for older biomeOS.

New functions: `signal_dispatch()` (generic signal dispatch helper),
`try_nest_commit_signal()` (nest.commit-specific wrapper).

### 3. `primal.announce` + `signal.dispatch` Consumed

**File:** `barracuda/src/niche.rs` — `CONSUMED_CAPABILITIES`

Added:
- `primal.announce` — replaces 3-call registration pattern (method.register +
  capability.register + lifecycle.register). Fallback to `method.register` on
  pre-v3.57 biomeOS.
- `signal.dispatch` — enables composition collapse for the 14 atomic signals.

Note: wetSpring does not currently emit `method.register` / `capability.register` /
`lifecycle.register` in sequence — registration goes through Songbird's
`discovery.register`. The `primal.announce` consumption prepares for when
biomeOS handles registration directly via the announce signal.

### 4. Registry Synced to 451 Methods

**File:** `barracuda/tests/ci_cross_sync.rs`

- Threshold bumped from 400 → 440 (expected 451+)
- Domain prefixes: added `primal.` and `signal.` to recognized list
- Test renamed: `consumed_includes_biomeos_v351_lifecycle` →
  `consumed_includes_biomeos_lifecycle_and_signals`
- Doc comment updated: 413 → 451

### 5. GAP-GS-015 Verified

`ALL_CAPS` and `BTSP_EXTRA_CAPS` re-export from primalSpring
`composition/mod.rs` confirmed working. `cargo check --features guidestone
--lib -p wetspring-barracuda` passes cleanly.

### 6. IPC Health Status Updated

**File:** `barracuda/src/ipc/handlers/mod.rs`

`biome_os` health status now reports:
- `primal_announce: "adopted"`
- `signal_dispatch: "adopted"`
- `wave17: "signal_elevation_adopted"`

### 7. Capability Registry Updated

**File:** `capability_registry.toml`

Wave 17 signal adoption block added: adopted signals (nest.store, nest.commit),
pending (primal.announce), registry_sync count 451.

---

## Build Gate

| Check | Result |
|-------|--------|
| `cargo clippy --features ipc --lib -- -W clippy::pedantic -W clippy::nursery` | exit 0 |
| `cargo test --features ipc --lib` | 252 pass, 0 fail |
| `cargo test --features ipc --test ci_cross_sync` | 7/7 pass |
| `cargo check --features guidestone --lib` | exit 0 (GAP-GS-015) |

---

## Remaining Glacial Priorities

Per the upstream directive, wetSpring still needs:

| Priority | Status |
|----------|--------|
| Thread 4 (Environmental Genomics) expression + targets | PENDING — needs foundation work |
| B7 → lithoSpore module 6 maintenance | ACTIVE — Tier 2 COMPLETE, handoff in progress |
| `primal.announce` live adoption | PREPARED — consumed, will activate when biomeOS v3.57+ is available |
| PG-02 / PG-04 (deployment-only) | HOLDING — upstream NestGate / NeuralSpring |

---

## What Stays as `ctx.call()`

Per Wave 17 standard, domain-specific math calls remain as direct IPC:
- `stats.mean`, `stats.variance`, `stats.std_dev`, `stats.correlation`
- `tensor.matmul`, `tensor.create`, `tensor.transpose`
- `linalg.solve`, `linalg.eigenvalues`, `linalg.svd`
- `spectral.fft`, `spectral.power_spectrum`
- `compute.dispatch` (toadStool workload submission)

Only orchestration sequences collapse to signals.

---

## Upstream Stale Data — Reiterated

| Document | Stale | Should Be |
|----------|-------|-----------|
| CROSS_SPRING_PARITY_SCORECARD wetSpring row | gS L4, 1,613 tests, LTEE "STARTED" | gS L5, 1,962 tests, LTEE Tier 2 COMPLETE |
| CROSS_SPRING_PARITY_SCORECARD | 48 consumed | 50 consumed (Wave 17) |
| plasmidBin manifest.toml | tests=1902 | 1,962 |

---

## Metrics (V169)

| Metric | Value |
|--------|-------|
| guideStone | **Level 5** — live NUCLEUS 30/31 pass |
| barraCuda | **v0.4.0** |
| coralReef | **v0.1.0** (in niche DEPENDENCIES) |
| Lib tests | 1,962 (0 failures) |
| Integration tests | 7/7 ci_cross_sync pass |
| Consumed capabilities | **50** (33 canonical + 15 legacy + 2 Wave 17) |
| Signal adoption | nest.store, nest.commit (dispatch with fallback) |
| Registry sync | **451** methods |
| Primal gaps | 2 open (deployment-only), 20 resolved/closed |
| Doc files synced | 14 |

---

*Submitted to primalSpring via `wateringHole/handoffs/` per
`NUCLEUS_SPRING_ALIGNMENT.md` feedback protocol.*
