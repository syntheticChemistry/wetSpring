# Experiment 373: V113 — Provenance Trio + Expanded Capabilities + biomeOS Deploy Graph

**Date:** 2026-03-15
**Status:** PASS
**Version:** V113
**Predecessor:** Exp372 (V112 streaming I/O + pedantic)

---

## Hypothesis

Integrating the provenance trio (rhizoCrypt → loamSpine → sweetGrass) via
biomeOS `capability.call`, expanding the IPC surface from 9 to 19 capabilities,
implementing the cross-spring time series exchange format, and creating a
biomeOS deploy graph will:

1. Enable every wetSpring experiment to produce immutable, attributed records
2. Allow other springs to consume wetSpring science via biomeOS graph composition
3. Provide structured time series exchange with airSpring, healthSpring, etc.
4. Position wetSpring as a deployable niche per the SPRING_AS_NICHE_DEPLOYMENT_STANDARD

## Method

### Phase 1: Provenance Trio Integration

Created `ipc/provenance.rs` following `SPRING_PROVENANCE_TRIO_INTEGRATION_PATTERN`:
- Neural API socket discovery (4-tier cascade)
- `capability.call` abstraction (JSON-RPC 2.0 over Unix socket)
- Three-phase completion: dehydrate → commit → attribute
- Full graceful degradation (standalone mode preserved)
- IPC handlers: `provenance.begin`, `provenance.record`, `provenance.complete`

### Phase 2: Expanded Science Capabilities

Created `ipc/handlers/expanded.rs` wrapping existing barracuda library functions:
- `science.kinetics` — Gompertz and first-order biogas production
- `science.alignment` — Smith-Waterman via `bio::alignment`
- `science.taxonomy` — Naive Bayes via `bio::taxonomy::NaiveBayesClassifier`
- `science.phylogenetics` — Robinson-Foulds via `bio::robinson_foulds`
- `science.nmf` — NMF (Lee & Seung multiplicative update, self-contained)

### Phase 3: Cross-Spring Time Series

Created `ipc/timeseries.rs` implementing `ecoPrimals/time-series/v1`:
- `science.timeseries` — ingest and analyze (mean, variance, trend)
- `science.timeseries_diversity` — diversity metrics on abundance series
- Builder functions for outbound payloads
- Schema version validation

### Phase 4: biomeOS Deploy Graph

Created `graphs/wetspring_deploy.toml`:
- Sequential germination: BearDog → Songbird → Provenance Trio → wetSpring
- Optional enrichment nodes: nestGate, petalTongue
- 19 capabilities registered
- Validation phase with health check

### Phase 5: Wiring

- Updated `ipc/mod.rs`: added `provenance` and `timeseries` sub-modules
- Updated `ipc/handlers/mod.rs`: added `expanded` sub-module, expanded CAPABILITIES (9 → 19)
- Updated `ipc/dispatch.rs`: added 9 new dispatch routes

## Results

| Gate | Result |
|------|--------|
| `cargo check --features ipc,json` | PASS |
| `cargo clippy` (pedantic + nursery) | ZERO new warnings |
| `cargo test` (1,326 tests) | 0 failures |
| Provenance degradation (no trio) | Graceful — all handlers return valid JSON |
| Time series schema validation | Rejects wrong versions |
| Deploy graph structure | Follows SPRING_AS_NICHE_DEPLOYMENT_STANDARD |

## Key Design Decisions

1. **Zero compile-time coupling**: Provenance uses `capability.call` over Unix socket, not crate imports
2. **NMF is self-contained**: No upstream barracuda NMF module yet; Lee & Seung MU with `mul_add`
3. **Deploy graph uses optional nodes**: nestGate and petalTongue marked optional — graph works without them
4. **19 capabilities**: Full surface including brain, provenance, and time series

## Files Changed

| File | Change |
|------|--------|
| `barracuda/src/ipc/mod.rs` | Added `provenance`, `timeseries` modules; expanded capability table |
| `barracuda/src/ipc/provenance.rs` | **NEW** — provenance trio integration |
| `barracuda/src/ipc/timeseries.rs` | **NEW** — cross-spring time series exchange |
| `barracuda/src/ipc/handlers/mod.rs` | Added `expanded` module; CAPABILITIES 9 → 19 |
| `barracuda/src/ipc/handlers/expanded.rs` | **NEW** — kinetics, alignment, taxonomy, phylogenetics, NMF |
| `barracuda/src/ipc/dispatch.rs` | Added 9 dispatch routes |
| `graphs/wetspring_deploy.toml` | **NEW** — biomeOS deploy graph |
