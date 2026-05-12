# wetSpring V157 — Deep Debt Evolution + Upstream Handoff

**Date:** 2026-05-10
**From:** wetSpring
**To:** primalSpring, primal teams (barraCuda, toadStool, NestGate, BearDog, sweetGrass, rhizoCrypt, loamSpine, skunkBat), downstream springs
**primalSpring version:** v0.9.25
**Commits:** 778c82e (V157) + 7294935 (V156)

---

## Summary

V156–V157 complete the post-interstadial evolution wave triggered by the
primalSpring audit (May 10, 2026). V156 wired skunkBat audit logging,
absorbed biomeOS v3.51 capabilities, and created CI cross-sync tests.
V157 centralized IPC timeouts, evolved the GPU API, expanded the shared
validation harness, and ran a comprehensive deep debt audit.

**Build gate:** `cargo build + fmt --check + clippy --workspace --all-targets +
test --workspace --lib --tests` — **zero warnings, 1,594 passed, 0 failed.**

---

## V157 Changes

### IPC Timeouts Centralized
New `ipc::timeouts` module with 8 semantic constants:
- `DISCOVERY` (5s) — songbird
- `STANDARD_RPC` (10s) — provenance trio, general transport
- `COMPUTE` (30s) — toadStool/barraCuda dispatch
- `AI_INFERENCE` (30s) — Squirrel
- `CONNECTION` (120s) — server-side long-lived
- `FACADE_SHORT/STANDARD/RENDER` (5/10/15s) — HTTP gateway tiers

All 11 scattered `Duration::from_secs(N)` constants across IPC + facade now
reference the canonical module. Semantic tier names document intent.

### GPU API Evolution
Removed deprecated `submit_and_poll` from `pairwise_l2_gpu.rs` per barraCuda
BREAKING_CHANGES Sprint 42. `read_buffer_f32` handles its own poll cycle
internally via `submit_and_map<T>`. This resolves the GPU API drift gap flagged
by both projectNUCLEUS and foundation composition validation docs.

### Shared Validation Harness
`validation::timing` expanded:
- `BenchRowEvolved` — absorbs s87's local `Timing` struct for cross-spring evolution
- `print_bench_evolved_table` — 4-column box-drawing table with evolved column
- `print_kv_box` — reusable ASCII summary box for `╔═╗` blocks
- `timing` module now public for direct import by bins

### Bug Fix
Fixed broken `bench` wrapper in `validate_cross_spring_evolution_s87.rs`. The
function had invalid syntax (`bench_print(label, f)(result, ms)`) that went
undetected because the binary requires `--features gpu`.

### Deep Debt Audit Results
| Dimension | Status |
|-----------|--------|
| Unsafe code | ZERO across entire workspace |
| Production mocks | ZERO — all isolated to `#[cfg(test)]` |
| Hardcoded URLs | ALL env-configurable (7 env vars) |
| `todo!()` / `unimplemented!()` | ZERO in production paths |
| TODO/FIXME/HACK comments | ZERO in `.rs` files |

---

## V156 Changes

### skunkBat Audit Logging (JH-5 Forwarding)
- `SKUNKBAT` primal constant added to `primal_names.rs`
- New `NicheDependency` (role: audit, required: false)
- All 7 deploy graphs updated with `[[graph.nodes]]` entry
- Capabilities: `audit.event`, `audit.forward`

### biomeOS v3.51 Absorbed
- `composition.status` and `method.register` added to `CONSUMED_CAPABILITIES`
- `composition.science_health` handler includes `biome_os` status block

### CI Cross-Sync Test
New `ci_cross_sync.rs` with 6 integration tests:
1. Local capability registry matches dispatch table
2. Niche capabilities superset of dispatched methods
3. Dependencies include required infrastructure primals
4. Consumed capabilities include biomeOS v3.51 lifecycle methods
5. Consumed capabilities use recognized ecosystem domain prefixes
6. Canonical `primalSpring/config/capability_registry.toml` accessible (≥300 methods)

### Primal Gap Triage
- PG-16 closed: std_dev N-1 vs N convention documented as intentional
- PG-20 closed: `uds_send.py` established as permanent `socat` alternative
- PG-21 closed: same resolution as PG-20
- **12 gaps open, 10 resolved** (was 15/7)

---

## For Upstream Primal Teams

### barraCuda
- `submit_and_poll` migration complete on wetSpring side. One remaining call in
  `pairwise_l2_gpu.rs` removed. GPU build still blocked on `AniBatchF64` + other
  `domain-genomics` feature imports — needs barraCuda feature gate alignment.
- wetSpring consumes 150+ barraCuda primitives, 44 GPU modules, 0 local WGSL.
- Timeout for compute dispatch: `ipc::timeouts::COMPUTE` (30s).

### toadStool
- IPC timeout for compute dispatch is now `ipc::timeouts::COMPUTE` (30s).
- `performance_surface` module wired and validated.
- ComputeDispatch: all calls route through `compute_dispatch.rs` with centralized
  timeout from `ipc::timeouts`.

### skunkBat (NEW)
- Declared as dependency with `required: false`, role: `audit`.
- Added to all 7 deploy graphs (order 2, after beardog).
- Consumed capabilities: `audit.event`, `audit.forward`.
- Integration is declaration-only — actual event forwarding awaits skunkBat IPC.

### NestGate
- Real NCBI pipeline (Exp184) validated with 5 accessions.
- Data fetch handlers use env-configurable URLs (`WETSPRING_NCBI_*`).
- Bulk dataset ingestion (Tara, HMP, KBS-LTER, cold seep) blocked on NestGate
  data provisioning + LAN infrastructure.

### Provenance Trio (rhizoCrypt, loamSpine, sweetGrass)
- IPC timeout: `ipc::timeouts::STANDARD_RPC` (10s) for all trio calls.
- Facade provenance: `ipc::timeouts::FACADE_SHORT` (5s).
- Circuit breaker epoch-based pattern in place.

### songbird
- Discovery timeout: `ipc::timeouts::DISCOVERY` (5s).
- NAT traversal wired in niche.

### BearDog
- TLS dependency declared. All deploy graphs place beardog at order 1.
- skunkBat depends_on beardog in all graphs.

---

## For Spring Teams

### Patterns to Absorb
1. **Centralized timeout module** — `ipc::timeouts` with semantic tier names.
   Other springs with scattered Duration constants should consider the same.
2. **CI cross-sync test** — validates local capabilities against canonical
   `primalSpring/config/capability_registry.toml`. Portable pattern.
3. **Shared validation harness** — `BenchRowEvolved` + table helpers reduce
   boilerplate in cross-spring validation binaries.
4. **`#[expect(reason)]` policy** — zero `#[allow()]` in production. Every lint
   suppression carries a reason string.

### NUCLEUS Composition Patterns
- **7 deploy graphs** validated with `graph_validate.rs` (schema check).
- **skunkBat integration pattern**: order 2, after beardog, before compute.
- **biomeOS v3.51 lifecycle**: `composition.status` and `method.register` consumed
  but deferred to runtime via `CompositionContext`.
- **CI cross-sync**: local capability surface validated against canonical registry.

### neuralAPI / biomeOS Deployment
- wetSpring science facade (`wetspring_science_facade`) binds `FACADE_BIND`
  (default `127.0.0.1:3100`), CORS via `FACADE_CORS_ORIGIN`.
- Science health endpoint includes biomeOS status block.
- All timeouts documented in `ipc::timeouts` module doc comments.

---

## Gardens Review (projectNUCLEUS + foundation)

### projectNUCLEUS
- 11 wetSpring workloads under `workloads/wetspring/`.
- `wetspring-exp001-python-baseline` FAIL: unprovisioned data / wrong CWD.
- GPU API drift gap (submit_and_poll) — **resolved this wave**.
- Workload TOML paths still use literal machine paths.

### foundation
- wetSpring mapped to threads 1, 3, 4, 5, 6, 7.
- Data targets in `data/targets/thread01_wcm_targets.toml` etc.
- GPU build breakage (`pairwise_l2_gpu.rs` API) — **resolved this wave**.
- Kokkos parity referenced for hotSpring/plasma thread, not wetSpring-specific.

---

## Remaining Open Gaps (12)

See `docs/PRIMAL_GAPS.md` for full details. Key owners:
- **barraCuda**: GPU feature alignment, API surface drift
- **toadStool**: ComputeDispatch session management
- **biomeOS**: composition.reload, method.register live wiring
- **NestGate**: data provisioning for bulk datasets
- **primalSpring**: upstream manifest absorption of skunkBat dependency
