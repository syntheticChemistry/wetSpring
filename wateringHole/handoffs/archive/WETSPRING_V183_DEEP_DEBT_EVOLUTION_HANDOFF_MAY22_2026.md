# wetSpring V183 — Deep Debt Evolution + Doc Alignment + Handoff

**Date:** 2026-05-22
**From:** wetSpring
**To:** primalSpring (audit), barraCuda (math surface), toadStool (compute.fan_out), Songbird (capability.resolve), biomeOS (Neural API), all springs (composition patterns)
**Priority:** INFORMATIONAL — structural evolution, no blocking asks

---

## Deep Debt Evolution (V183)

Systematic resolution of 5 tracks of post-V182 technical debt.

### Track A: Large File Refactoring

6 oversize experiment files (>800L) refactored into idiomatic Rust directory modules:

| File | Before | After |
|------|--------|-------|
| `experiments/mod.rs` | 987L | 501L mod + 497L registry_table |
| `exp_cpu_vs_gpu_all_domains` | 921L | 108L mod + bio/chem/ml_ode (317/210/300L) |
| `exp_cross_spring_s57` | 912L | 112L mod + s54/s56/gpu (173/235/319L) |
| `exp_cross_spring_evolution_s87` | 864L | 134L mod + core/extended (297/407L) |
| `bench_cross_spring_s65` | 829L | 178L mod + ode_gpu/delegation (154/171L) |
| `exp_anderson_qs_environments_v1` | 823L | 635L mod + visualization (205L) |

No production modules exceed 800L. All files compile cleanly.

### Track B: Discovery Evolution

**B1 — Songbird-first resolution:** `discover_by_capability` now uses `#[cfg(feature = "ipc")]` to gate Songbird RPC. Static `capability_to_primal` table demoted to bootstrap-only fallback. Callers unchanged.

**B2 — Deploy graph convergence:** `NUCLEUS_GRAPH_REL_PATH` and `NUCLEUS_GRAPH_NAME` constants centralized in `primal_names.rs`. Hardcoded `wetspring_science_nucleus.toml` strings in `ipc/handlers/mod.rs` and `facade/provenance.rs` replaced with constants.

**B3 — Neural API consolidation:** 3 duplicate `neural_api_socket()` implementations eliminated. `facade/provenance.rs`, `facade/dark_forest.rs`, and `facade/grammar.rs` now delegate to canonical `ipc::provenance::neural_api_socket()` (connect-probe + negative cache).

**B5 — Discovery deduplication:** `visualization/ipc_push.rs` standalone fallback now delegates to shared `ncbi::nestgate::discovery::discover_standalone()` instead of duplicating resolution logic.

### Track C: Python Baselines

3 missing baselines written (all passing):

| Script | Experiment | Checks |
|--------|-----------|--------|
| `gonzales_ic50_baseline.py` | Exp280 (Gonzales 2014 IC50) | 23/23 |
| `gonzales_pk_decay_baseline.py` | Exp281 (Fleck/Gonzales 2021 PK) | 12/12 |
| `matrix_pharmacophenomics_baseline.py` | Exp158 (Fajgenbaum MATRIX) | 10/10 |

**Total Python baselines: 55** (was 52).

### Track D: Tenaillon Pipeline

`graphs/tenaillon_2016_fan_out.toml` created — 27 shards of 10 clones (264 total), ready for toadStool `compute.fan_out` when that primitive is available. Pipeline already supports `WETSPRING_CLONE_OFFSET` / `WETSPRING_MAX_CLONES` env vars for batched execution.

### Track E: Notebook UniBin Alignment

8 notebooks updated from `cargo run --bin validate_*` to `wetspring validate --scenario`:
- 5 general notebooks (02-benchmark, 03-gonzales, 05-composition, cahill-algae, liao-biogas)
- 3 paper notebooks (gonzales-jak, r-industry, fajgenbaum-drug)
- `NOTEBOOK_PATTERN.md` updated with UniBin workflow section

---

## Doc Alignment (V183)

### Root docs updated
- `CONTEXT.md` — scenario count 337 → 345
- `CONTROL_EXPERIMENT_STATUS.md` — full V183 rewrite with UniBin control chains
- `experiments/results/README.md` — UniBin reproduction commands
- `specs/README.md` — V183 date, 345 scenarios, 43 niche
- `PAPER_REVIEW_QUEUE.md` — 52 consumed (was 51), V183 status

### baseCamp updated
- `README.md` — V183 status banner (1,962 tests, 386 exps, 2 gaps)
- 7 faculty briefings — 33 old `cargo run --bin` commands → UniBin equivalents
- `PENDING_DATASETS.md` — Tenaillon fan_out status, V183 date

---

## Composition Patterns for NUCLEUS Deployment

For upstream teams absorbing wetSpring's patterns:

### Neural API Discovery (for biomeOS)
```
canonical: ipc::provenance::neural_api_socket()
cascade:   NEURAL_API_SOCKET env → BIOMEOS_SOCKET_DIR → XDG_RUNTIME_DIR/biomeos/ → temp_dir
probe:     connect-probe with 50ms timeout + negative cache (DEAD_SOCKET_CACHE)
```

### Capability-Based Discovery (for Songbird)
```
discover_by_capability("tensor") →
  1. [ipc] Songbird capability.resolve RPC (primary)
  2. Static capability_to_primal bootstrap table (fallback)
  3. discover_primal → env → XDG → temp cascade
```

### Deploy Graph Pair (for biomeOS orchestration)
```
graphs/wetspring_deploy.toml           — base niche composition (primals + gates + IPC)
graphs/wetspring_science_nucleus.toml  — science NUCLEUS (neural API, facade, provenance trio)
```

### Atomic Composition Testing Pattern (for all springs)
```
Exp400: proto-nucleate (136/136) — can the graph even form?
Exp401: IPC parity (43/43)      — do IPC results match direct calls?
Exp402: niche gate (63/63)       — are all niche capabilities surfaced?
Exp403: primal proof (5 primals) — do live primals respond correctly?
certify: Level 5 NUCLEUS (38/38) — full sovereign validation
```

---

## For Upstream Teams

### toadStool
- `graphs/tenaillon_2016_fan_out.toml` ready for `compute.fan_out` consumption
- 27 shards × ~45min each; GPU required per shard
- Pipeline env vars documented in experiment header

### Songbird
- wetSpring now gates `resolve_via_songbird` under `#[cfg(feature = "ipc")]`
- When Songbird is live, `discover_by_capability` uses it as primary resolver
- Static table remains as bootstrap for non-IPC builds

### barraCuda
- No new shader asks; 150+ primitives consumed, 0 local WGSL
- GillespieGpu NVVM f64 failure documented (skip, not block)

### biomeOS
- Neural API socket discovery consolidated to one implementation
- All 3 facade paths (provenance, dark_forest, grammar) delegate correctly
- Connect-probe + dead socket cache pattern available for adoption

### All Springs
- Directory module pattern proven for experiment organization (Track A)
- Python baseline pattern documented (tolerances.py shared module)
- UniBin `Validator::bridge_into` adapter pattern for `primalspring::ValidationResult`

---

## Active Gap Posture

| # | Gap | Priority | Status |
|---|-----|----------|--------|
| WS-1 | Ionic contract negotiation | HIGH | IonicContractRegistry available upstream (Wave 37); wire pending |
| WS-2 | Cross-spring RootPulse exchange | HIGH | nest.sync RESOLVED; E2E pending biomeOS 6-node test |
| WS-4 | petalTongue client WASM | MEDIUM | Blocked upstream |
| WS-9 | Cross-tier parity (L3) | MEDIUM | L1/L2 done; L3 pending Nest deploy |
| WS-11 | Variant caller calibration | HIGH | v3 deployed; 5/5 Tenaillon clones validated; 259 remaining awaiting fan_out |

---

## For primalSpring Audit

Requesting audit confirmation:
1. V183 deep debt resolution aligns with Wave 38 expectations
2. 345 scenarios + 55 Python baselines consistent with registry 445
3. Songbird-first discovery pattern approved for spring adoption
4. Tenaillon fan_out graph format compatible with toadStool spec
5. No regressions in sporePrint validation surface
