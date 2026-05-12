<!-- SPDX-License-Identifier: CC-BY-SA-4.0 -->

# wetSpring V164b — Upstream Deep Debt & Foundation Sync Handoff

**From:** wetSpring V164b
**To:** primalSpring (upstream audit), all spring teams, projectNUCLEUS, foundation, lithoSpore
**Date:** May 12, 2026
**Commits:** wetSpring `aeaa436` (V164b), foundation `80db7e9`

---

## Deliverables This Increment

### 1. Foundation Thread Sync (10/10 Active)

Three threads had expressions authored on disk by other springs but not wired
into `lineage/THREAD_INDEX.toml`:

| Thread | Expression | Change |
|--------|-----------|--------|
| 3 (Immunology) | `IMMUNO_DRUG_DISCOVERY.md` | Wired expression + sources + targets, "mapped" → "active" |
| 8 (Human Health) | `SOVEREIGN_HEALTH.md` | "seeded" → "active" (already had expression + data) |
| 9 (Gaming) | `GAMING_CREATIVE_SCIENCE.md` | Wired expression (already had sources/targets) |
| 10 (Provenance) | `PROVENANCE_ECONOMICS.md` | Wired expression + sources + targets, "mapped" → "active" |

Foundation is now **10/10 threads active** with all expressions, sources, and
targets indexed. `expressions/README.md` updated to list all 9 expression
documents (excluding README itself).

### 2. Pending Datasets Tracker

Created `specs/PENDING_DATASETS.md` — consolidated view across wetSpring:

- **4 data downloads pending:** EPA UCMR 5, EPA PFOS surface water, Jones PFAS 175 compounds, LTEE PRJNA294072
- **9 LTEE queue datasets:** B1-B6, B8, E1, E5 (B7 Tier 1 complete)
- **2 hardware items:** AKD1000 NPU, Titan V GPU
- **3 GPU shaders not written:** kmer histogram, UniFrac tree propagation, DF64 Phase 2

### 3. Deep Debt Audit Confirmed Clean

Full codebase audit across 743 `.rs` files (344 barracuda bins + 22 forge bins):

| Category | Finding |
|----------|---------|
| Unsafe code | Zero — `#![forbid(unsafe_code)]` on all crate roots |
| `#[allow(` | Zero — all using `#[expect(reason = "...")]` |
| Production mocks | Zero |
| `extern crate` / `try!` / `#[macro_use]` | Zero |
| `todo!()` / `unimplemented!()` / FIXME | Zero in compiled code |
| Hardcoded addresses | All env-configurable (`FACADE_BIND`, `WETSPRING_TCP_ADDR`, API URL defaults) |
| `deny.toml` | 15 C-binding crates banned (ring, openssl-sys, libz-sys, etc.) |
| Large files >800L | 5 validation binaries, all with justified `#[expect(clippy::too_many_lines)]` |

### 4. Doc Sync (13 files updated)

All docs synchronized to V164b: README, CONTEXT, GAPS, PRIMAL_GAPS, specs/README,
BARRACUDA_REQUIREMENTS, whitePaper/README, baseCamp/README, EXTENSION_PLAN,
EVOLUTION_READINESS, ABSORPTION_MANIFEST, experiments/README, sporeprint/validation-summary.

Reconciled counts: 384 experiments, 366 binaries (344+22), 1,962 lib tests,
63/63 papers, LTEE B7 TIER 1 COMPLETE.

---

## Current State Summary

| Metric | Value |
|--------|-------|
| Tests | 1,962 lib + 97 integration + 18 IPC, 0 failed |
| Experiments | 384 indexed (381 completed + 3 proposed) |
| Binaries | 366 (344 barracuda + 22 forge) |
| Validation checks | 5,900+ |
| Papers | 63/63 main queue DONE |
| LTEE | B7 TIER 1 COMPLETE, 9 queued (B1-B6, B8, E1, E5) |
| Coverage | 91.20% line / 90.30% function |
| GuideStone | Level 4 (38/38 pass, 4 skip) |
| Primal gaps | 4 open (all external: PG-02 provenance trio, PG-03 Songbird, PG-04 NestGate, PG-05 toadStool) |
| Foundation threads | 10/10 active |
| Tier 4 | IPC-first defaults (`default = []`) |

---

## Remaining Debt

### External (Blocked on Upstream Primals)

| Gap | Owner | Status |
|-----|-------|--------|
| PG-02 Provenance trio | rhizoCrypt + loamSpine + sweetGrass | Awaiting trio endpoints going live |
| PG-03 Capability discovery | Songbird | `capability.resolve` not shipped |
| PG-04 NestGate deploy | NestGate | Content pipeline not live |
| PG-05 toadStool dispatch | toadStool | Full sovereign dispatch pending |

### Internal (Low Priority)

- **Fuzz targets stale:** `fuzz_mzml.rs` and `fuzz_ms2.rs` reference `parse_mzml`/`parse_ms2` (removed APIs); need update to use iterator patterns matching `fuzz_fastq.rs`
- **LTEE queue:** 9 papers queued (B1-B6, B8, E1, E5) — not blocking but feeds lithoSpore
- **GPU shaders:** 3 not written (kmer histogram, UniFrac tree propagation, DF64 Phase 2)
- **guideStone L5:** Target path documented, needs expanded certification coverage

---

## For Upstream Primals (primalSpring Audit)

### NestGate Team
wetSpring's `ncbi/nestgate/` module is wired and tested but gates on NestGate's
content pipeline going live. The sovereign NCBI genomics pipeline (Exp380 B7
Tenaillon 264-genome) requires NestGate for production data fetching. IPC path:
`ncbi.fetch` → NestGate → NCBI API. Currently falls back to direct HTTP.

### Songbird Team
wetSpring registers 42 niche capabilities at startup via `method.register`. The
`capability.resolve` call (PG-03) for runtime discovery is wired but Songbird
hasn't shipped the resolve endpoint. wetSpring uses hardcoded socket paths with
`FAMILY_ID` fallback.

### Provenance Trio (rhizoCrypt + loamSpine + sweetGrass)
Science facade routes (`/api/v1/provenance/{result_id}`) attempt provenance
queries through the trio. Currently returns structured gap reports when trio
is unavailable. Ready to consume when trio endpoints ship.

### toadStool Team
Performance surface and compute dispatch wired. `toadStool.performance_surface`
and `compute.dispatch` IPC calls exist but are `check_skip` in guideStone when
toadStool is not present.

---

## For Spring Teams (River Delta)

### Patterns Available for Absorption

1. **`--format json` on UniBin subcommands** — `OutputFormat` enum with `clap::ValueEnum`,
   passed to handler functions, calls `ValidationResult::to_json()` for machine-readable
   output. Avoids `std::env::set_var` (which is unsafe). Other springs implementing
   `--format json` should follow this pattern.

2. **Foundation thread expression authoring** — `ENVIRONMENTAL_GENOMICS.md` follows
   the template: framing → paper lineage → jelly strings (provenance gaps) → NUCLEUS
   blueprints → spring alignment → petalTongue vision → scyBorg publication chain.

3. **Pending datasets consolidation** — `specs/PENDING_DATASETS.md` tracks all
   undownloaded data, hardware needs, and GPU shader gaps in one place. Other
   springs with similar scatter should consolidate.

4. **Deep debt audit methodology** — systematic grep-based audit across all
   `.rs` files for unsafe, `#[allow(`, mocks, hardcoded values, deprecated
   patterns, unfinished markers. Confirms compliance posture for upstream audit.

### What wetSpring Needs from Other Springs

- **groundSpring B2+B1 LTEE data** for cross-validation against wetSpring B7
- **neuralSpring ML surrogates** for LTEE prediction comparison
- **hotSpring Anderson RMT** for cross-thread spectral validation

---

## For projectNUCLEUS

- 12 workload TOMLs in `workloads/wetspring/` (including new LTEE B7 workload)
- All workloads gate-agnostic (`$SPRINGS_ROOT` paths)
- `--format json` enables Tier 2 structured ingestion from `validate` and `certify`
- Next: Exp380 Rust validation binary (`validate_ltee_b7_mutation_accumulation`)
  when Tier 2 pipeline is built

## For lithoSpore

- B7 expected values JSON at `experiments/results/ltee_b7_expected_values.json`
- 10 validation targets: non-mutator rate, Ts:Tv, mutation spectrum, accumulation curve
- 4 foundation targets in `data/targets/thread04_enviro_targets.toml` with `validated = false`
- Module 6 (`ltee-genomics`) can ingest when Tier 2 Rust binary validates

## For foundation

- 10/10 threads now active
- Thread 4 expression authored with 8 jelly strings and 4 NUCLEUS blueprints
- 40 Thread 4 targets (36 original + 4 LTEE B7)
- Expressions README updated
- Thread 5 ML Surrogates files committed (from other spring work)
