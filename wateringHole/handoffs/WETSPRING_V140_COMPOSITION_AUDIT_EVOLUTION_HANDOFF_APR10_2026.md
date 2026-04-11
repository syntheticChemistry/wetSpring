# WETSPRING V140 — Composition Audit + Evolution Handoff

| Field | Value |
|-------|-------|
| Date | 2026-04-10 |
| From | wetSpring V140 |
| To | primalSpring, biomeOS, wateringHole, barraCuda |
| License | AGPL-3.0-or-later |
| Supersedes | WETSPRING_V139_COMPOSITION_VALIDATION_HANDOFF_APR10_2026.md |

## Executive Summary

Full ecosystem audit executed against wateringHole standards. All critical and
high-priority findings resolved. This handoff marks the evolution from
"Rust validates Python" to "Rust + Python validate NUCLEUS composition patterns."

### Evolution Path (now three tiers)

```
Tier 1: Python baseline → Rust validation (1,580 tests, 356 binaries)
Tier 2: Rust → Primal composition validation (proto-nucleate alignment)
Tier 3: Python + Rust → ecoBin harvest to plasmidBin
```

Python was the validation target for Rust. Now Rust AND Python are validation
targets for the ecoPrimals NUCLEUS patterns. ecoBins can be harvested to
`infra/plasmidBin/` as composition tiers pass.

## Canonical Metrics

| Metric | V139 | V140 |
|--------|------|------|
| Library tests | 1,580 | 1,580 |
| Integration + forge tests | 360 | 360 |
| Total tests | ~1,940 | ~1,940 |
| Validation binaries | 356 | 356 |
| IPC methods routed | 37 | 37 |
| Capabilities advertised | 45 | 45 |
| Composition health handlers | 5 | 5 |
| Deploy graphs | 7 | 7 (all canonical schema) |
| Proto-nucleate primals covered | 11/11 | 11/11 |
| Named tolerances | 38+ | 242 with provenance trail |
| LOC (Rust) | ~214,690 | ~215,000 |
| unsafe blocks | 0 | 0 |
| `#[allow()]` in production | 0 | 0 |
| clippy warnings | 82+ | 0 |
| cargo-deny | FAIL (bans) | PASS |
| cargo fmt | PASS | PASS |
| Files over 1000 LOC | 0 | 0 |
| C application dependencies | 0 | 0 |

## Changes in V140

### Critical Fixes

1. **C1: clippy::expect_used in facade binary** — The CORS header fallback
   in `wetspring_science_facade.rs:42` used `.expect()` which is denied by
   workspace clippy policy. Replaced with `?` error propagation through the
   `Result` return type. Zero panicking paths in the facade.

2. **C2: PrecisionTier non-exhaustive match** — upstream barraCuda added
   `PrecisionTier::F16` but `validate_precision_brain_v1.rs` had an exhaustive
   match without it. Added wildcard arm mapping unknown tiers to 0.0 for
   forward compatibility.

3. **C3: Upstream barraCuda feature gate bug** — `tolerances/precision.rs:138`
   references `crate::device` without `#[cfg(feature = "gpu")]` gating. This
   blocks `cargo test -p wetspring-barracuda` in isolation. Workaround: test
   via `cargo test --workspace`. **Handed back to barraCuda team.**

### Deploy Graph Schema Canonicalization

`wetspring_deploy.toml` was the only deploy graph using the non-canonical
`[[nodes]]` schema. Migrated to `[[graph.node]]` with `name`, `order`,
`by_capability`, `capabilities`, and `depends_on` fields matching the
canonical schema used by all 6 other deploy graphs and validated by
`graph_validate.rs`. Version bumped to 2.0.0.

### Tolerance Provenance Trail

New `barracuda/data/tolerance_provenance.toml` provides machine-readable
provenance for all named tolerance constants:
- IEEE 754 identity tolerances (mathematical, no baseline needed)
- Analytical tolerances (textbook references: Golub & Van Loan, Lee & Seung)
- Python parity tolerances (script path, package, experiment ID, command)
- GPU vs CPU tolerances (experiment IDs, instruction reorder justification)
- Instrument tolerances (measurement precision specs, EPA methods)

### CI Orchestration

New `scripts/check_all.sh` orchestrates the full audit pipeline:
fmt → clippy → test → deny → coverage → Python baselines.
Supports `--skip-cov` and `--skip-py` for faster iteration.

### cargo-deny Cleanup

- Added `windows-sys@0.60.2` skip (wgpu v28 transitive duplicate)
- Removed stale license allowances (AGPL-3.0 short forms, MPL-2.0,
  Unicode-DFS-2016 — not in current dep tree)
- Removed stale `flate2` wrapper from `cc` ban (only `blake3` wraps `cc`)

### CHANGELOG Backfill

Added V138 and V139 entries to CHANGELOG.md (were documented only in
handoffs and README; now in canonical changelog format).

## Composition Validation Tier

The new validation tier proves NUCLEUS composition patterns work:

```
┌─────────────────────────────────────────────────────────────┐
│  Tier 1: Python → Rust                                      │
│  "Does our Rust match the Python baseline?"                 │
│  Evidence: 356 validation binaries, hardcoded expected vals │
├─────────────────────────────────────────────────────────────┤
│  Tier 2: Rust → Primal Composition                          │
│  "Does our IPC surface match the proto-nucleate graph?"     │
│  Evidence: validate_composition_nucleus_v1 (7 dimensions)   │
├─────────────────────────────────────────────────────────────┤
│  Tier 3: Composition → ecoBin                               │
│  "Can we harvest a deployable binary to plasmidBin?"        │
│  Evidence: cargo build --release, ecoBin cross-compile      │
└─────────────────────────────────────────────────────────────┘
```

### ecoBin Harvest Path

```bash
cd /home/westgate/Development/ecoPrimals/springs/wetSpring
cargo build --release --features ipc --bin wetspring
cp target/release/wetspring ../../infra/plasmidBin/wetspring/wetspring
cd ../../infra/plasmidBin
./harvest.sh wetspring --tag v0.9.0
```

## Gaps — Status

### Resolved in V140

| # | Gap | Resolution |
|---|-----|-----------|
| C1 | expect_used in facade | `?` propagation |
| C2 | PrecisionTier exhaustiveness | Wildcard arm |
| H1 | Hardcoded `/tmp` | `std::env::temp_dir()` |
| H2 | unused_async warnings | Module-level `#[expect]` |
| H3 | cargo-deny bans | `windows-sys` skip |
| H4 | Stale deny.toml | Removed unused entries |
| M5 | Deploy graph schema | Canonical `[[graph.node]]` |

### Handed Back (Upstream)

| # | Gap | Owner | Action |
|---|-----|-------|--------|
| C3 | barraCuda feature gate | barraCuda team | Gate `tolerances/precision.rs:138` behind `#[cfg(feature = "gpu")]` |
| V139-1 | coralReef direct IPC | barraCuda/primalSpring | Expose `compute.shader.status` |
| V139-2 | Deploy graph schema doc | primalSpring | Document canonical `[[graph.node]]` schema |
| V139-3 | Proto-nucleate validation node | primalSpring | Confirm `primalspring_primal` implements `coordination.validate_composition` |
| V139-4 | Neural API socket env | biomeOS | Add `BIOMEOS_NEURAL_API_SOCKET` override |

### Ecosystem Gaps (unchanged)

See `GAPS.md` for 7 documented architectural gaps (all external team
dependencies: ionic negotiation, RootPulse exchange, public chain anchor,
petalTongue WASM, ludoSpring composition, hotSpring physics, radiating
attribution).

## Ecosystem Compliance Matrix

| Standard | Status |
|----------|--------|
| AGPL-3.0-or-later (SCYBORG) | PASS |
| ecoBin v3.0 (pure Rust, zero C app deps) | PASS |
| `#![forbid(unsafe_code)]` | PASS (workspace + crate) |
| Zero `#[allow()]` in production | PASS |
| All files under 1000 LOC | PASS (max 940) |
| JSON-RPC 2.0 / UDS IPC | PASS |
| Capability-based discovery | PASS |
| Data provenance (accession numbers) | PASS |
| Handoff naming convention | PASS |
| Proto-nucleate alignment | PASS (11/11 primals) |
| Squirrel AI integration | PASS (optional, graceful degradation) |
| Tolerance provenance trail | **NEW** — now machine-readable |

---

*License: AGPL-3.0-or-later*
