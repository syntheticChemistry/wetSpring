# WETSPRING V141 — Audit Remediation + Primal Evolution Handoff

| Field | Value |
|-------|-------|
| Date | 2026-04-11 |
| From | wetSpring V141 |
| To | primalSpring, biomeOS, barraCuda, toadStool, spring teams |
| License | AGPL-3.0-or-later |
| Supersedes | V140 composition audit + primal evolution handoffs |

## Executive Summary

V141 closes the audit remediation loop opened by V140's ecosystem review.
Capability surfaces are reconciled and cross-checked in CI. Proto-nucleate
alignment is validated at test time against `primalSpring` graphs. Provenance
run commands and a `TensorSession` proof-of-concept round out the technical
changes. Seven composition-specific gaps are documented for primal team
feedback.

### Three-Tier Validation Narrative (crystallised)

```
Tier 1  Python validates Rust (science fidelity)
        58 scripts → 1,946 tests, 5,800+ checks, 356 binaries

Tier 2  Rust + Python validate NUCLEUS composition (primal patterns)
        97/97 proto-nucleate (guard constant), 7 deploy graphs,
        21 domains (8 families), cross-check tests in CI

Tier 3  Composition enables ecoBin harvest (deployment)
        cargo build --release → infra/plasmidBin/
        biomeOS deploy --graph wetspring_science_nucleus.toml
```

Python was the validation target for Rust. Now Rust AND Python are validation
targets for ecoPrimal NUCLEUS composition patterns. ecoBins are harvested to
`infra/plasmidBin/` as composition tiers pass.

## Canonical Metrics

| Metric | V140 | V141 |
|--------|------|------|
| Library tests | 1,580 | 1,607 (barracuda all-features) |
| Total tests | ~1,940 | 1,946 |
| Validation binaries | 356 | 356 |
| IPC domain families | 5 | 8 |
| Domain prefixes | 18 | 21 |
| Domain methods | 30 | 41 |
| Niche capabilities | 45 | 45 |
| Dispatch methods | 37 | 42 |
| Composition health handlers | 5 | 5 |
| Deploy graphs | 7 | 7 (canonical) |
| Proto-nucleate test-time check | — | CI reads primalSpring TOML |
| Provenance run commands | — | 8 key scripts |
| TensorSession PoC | — | `alpha_diversity_session` |
| Primal composition gaps | implicit | 7 in `docs/PRIMAL_GAPS.md` |
| Named tolerances | 242 | 242 (8 with commands) |
| Clippy | 0 | 0 |
| Unsafe | 0 | 0 |

## What Changed (V141)

### 1. Capability Domain Alignment

The domain registry (`capability_domains.rs`) now has 21 domain prefixes
across 8 families (was 18/5). New domains: `data` (3 methods), `vault` (3),
`composition` (5). Total domain methods: 41 (was 30).

Two cross-check tests enforce consistency:
- `handlers_capabilities_covered_by_domains_or_documented` — every handler
  capability is covered by a domain (or explicitly documented as meta).
- `niche_capabilities_superset_of_handlers` — niche advertises at least
  everything the handler surface dispatches.

Capability drift is now a CI failure, not a review finding.

### 2. Proto-Nucleate Test-Time Validation

`niche.rs::proto_nucleate_node_names_match_niche_dependencies` reads
`primalSpring/graphs/downstream/wetspring_lifescience_proto_nucleate.toml`
at test time. Validates all niche dependencies appear as graph nodes, and
the graph declares correct `owner` and `pattern`. Gracefully degrades if
`primalSpring` is not co-located (prints warning, does not fail).

### 3. Provenance Run Commands

`BaselineProvenance` now has `command: Option<&'static str>`. Eight key
validation scripts carry exact reproduction commands. Test enforces that
commands reference their script paths.

### 4. TensorSession Proof-of-Concept

`diversity_gpu::alpha_diversity_session` demonstrates the `GpuContext` →
`TensorSession` bridge for alpha diversity. Explicitly `f32` — the `f64`
`FusedMapReduceF64` path remains the science-grade route. This validates
the barraCuda session API for fused multi-op pipelines.

### 5. EXPECTED_CHECKS Guard

`validate_composition_nucleus_v1.rs` now has `const EXPECTED_CHECKS: u32 = 97`
with a runtime `assert_eq!` before `v.finish()`. Drifts in check count are
caught at binary exit.

### 6. Upstream barraCuda Fix

`barraCuda/crates/barracuda/src/tolerances/precision.rs`: `for_precision_tier`
and its test are now `#[cfg(feature = "gpu")]`. Fixes `E0433` when building
without the `gpu` feature.

## Primal Evolution — Per-Primal Status

### Required Primals

| Primal | Role | wetSpring Usage | Next Step |
|--------|------|----------------|-----------|
| **BearDog** | security | `probe_capability("security")`, consent gate in vault | Stable consent/token contract for NUCLEUS |
| **Songbird** | discovery | `probe_capability("discovery")`, niche registration | **PG-03**: `capability.resolve` so routing is not name-locked |
| **rhizoCrypt** | DAG | `discover_rhizocrypt()`, provenance probe | **PG-02**: typed IPC client for DAG sessions |
| **loamSpine** | commit | `discover_loamspine()`, ledger probe | **PG-02**: real ledger commit across primal boundary |
| **sweetGrass** | provenance | `discover_sweetgrass()`, braid fields | **PG-02**: braid IPC; implement or deprecate aspirational `integration.sweetgrass.braid` |

### Optional Primals

| Primal | Role | wetSpring Usage | Next Step |
|--------|------|----------------|-----------|
| **NestGate** | storage | vault/data paths, Nest health probe | **PG-04**: end-to-end storage IPC |
| **toadStool** | compute | `discover_toadstool()`, compute probe | **PG-05**: compute IPC for NUCLEUS deployments (springs use barraCuda directly today) |
| **petalTongue** | viz | `discover_petaltongue()`, scenario builders | Stable viz IPC contract for NUCLEUS user-facing graphs |
| **Squirrel** | AI | `ai.ecology_interpret` → Unix socket JSON-RPC | Neural API alignment: context limits, timeouts, biomeOS routing |

### barraCuda (path dependency, not IPC primal)

- **Absorption:** Zero local WGSL, zero duplicate derivative math. 44 GPU modules
  lean on upstream. 5 ODE systems generated via `BatchedOdeRK4::generate_shader()`.
- **TensorSession:** PoC validates the session API; promotion to science-grade
  awaits `f64` session support or explicit `f32` tolerance tiers.
- **Upstream fix:** `precision.rs` feature-gate landed in V141.

## Composition Gaps (docs/PRIMAL_GAPS.md)

| ID | Gap | Owner | Status |
|----|-----|-------|--------|
| PG-01 | Proto-nucleate not parsed at build/CI time | wetSpring + primalSpring | **Mitigated** (test reads TOML) |
| PG-02 | Provenance trio: no typed IPC clients | rhizoCrypt, loamSpine, sweetGrass | Open |
| PG-03 | Discovery is name-based, not capability-first | Songbird, biomeOS | Open |
| PG-04 | NestGate: vault/data paths are local, not cross-primal | NestGate | Open |
| PG-05 | toadStool: no runtime compute IPC from springs | toadStool | Open |
| PG-06 | Ionic bond negotiation: metadata only | primalSpring Track 4 | Open |
| PG-07 | Niche vs handlers capability drift | wetSpring | **Resolved V141** |

## NUCLEUS Deployment via biomeOS + Neural API

wetSpring's deploy graphs define the canonical `[[graph.node]]` pattern with
`by_capability`, `order`, and `depends_on` fields. The deployment story:

```
primalSpring owns the proto-nucleate graph
  → wetSpring validates niche alignment at test time
    → composition health handlers report readiness
      → biomeOS deploy reads the deploy graph
        → Neural API exposes /capability/discover + /health endpoints
          → ecoBin harvested to plasmidBin for release
```

For this to become real:
1. **Songbird** needs `capability.resolve` (PG-03) so biomeOS routes by
   capability, not by primal name.
2. **Provenance trio** needs typed IPC (PG-02) so braid/DAG/commit cross the
   process boundary.
3. **toadStool** needs compute IPC (PG-05) so coralReef-compiled shaders can
   be dispatched through the orchestrator instead of only through barraCuda
   path dependencies.
4. **biomeOS** needs to consume the deploy graph TOML and expose the Neural API
   socket per `neural-api-{family_id}.sock` convention.

## What Spring Teams Should Absorb

Other springs evolving toward composition can adopt these patterns:

- **`EXPECTED_CHECKS` guard pattern:** Enforce documented check counts with a
  runtime constant. Prevents silent drift between documentation and binary.
- **Capability cross-check tests:** Three-way consistency (niche → handlers →
  domains) enforced in CI. Template: `capability_domains::tests`.
- **Proto-nucleate test-time validation:** `include_str!` or `read_to_string`
  the primalSpring TOML in tests. Graceful degradation when not co-located.
- **`docs/PRIMAL_GAPS.md`:** Composition-specific gap tracking per ecosystem
  convention. Separates primal integration gaps from general architecture debt.
- **Provenance `command` field:** Extend provenance registries with exact
  reproduction commands. Enables one-click baseline re-creation.
- **`check_all.sh` with pedantic + nursery:** Match CI strictness locally.
  Include `--all-features` test step.

## Handoff Artifacts

| File | Purpose |
|------|---------|
| `barracuda/src/ipc/capability_domains.rs` | 21 domains, 41 methods, cross-check tests |
| `barracuda/src/niche.rs` | Proto-nucleate cross-validation test |
| `barracuda/src/provenance_registry.rs` | 8 run commands on `BaselineProvenance` |
| `barracuda/src/bio/diversity_gpu.rs` | `alpha_diversity_session` TensorSession PoC |
| `barracuda/src/bin/validate_composition_nucleus_v1.rs` | `EXPECTED_CHECKS = 97` guard |
| `docs/PRIMAL_GAPS.md` | 7 composition gaps for primal team feedback |
| `scripts/check_all.sh` | Aligned with CI: pedantic/nursery, all-features |
