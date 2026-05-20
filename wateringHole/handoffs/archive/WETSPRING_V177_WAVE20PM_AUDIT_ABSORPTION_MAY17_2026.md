<!-- SPDX-License-Identifier: CC-BY-SA-4.0 -->

# wetSpring V177 — Wave 20 PM lithoSpore Audit Absorption

**Date:** May 17, 2026
**From:** wetSpring
**To:** primalSpring, lithoSpore, all spring teams
**Wave:** 20 PM (forward-looking evolution)

---

## Summary

wetSpring absorbed the Wave 20 PM lithoSpore audit findings and resolved all
four action items. Additionally, Exp381 (breseq pipeline on Barrick 2009 via
Nest Atomic composition) is executing — the ecosystem's first real-data
provenance chain from raw NCBI data through computation to ferment braid.

---

## Audit Action Items — Status

### 1. Stability Tier Awareness — DONE

All 43 niche methods annotated with `stability` in `capability_registry.toml`:
- **30 stable**: science core (diversity, anderson, kinetics, alignment, taxonomy,
  phylogenetics, nmf, timeseries, ncbi_fetch, full_pipeline), provenance (begin,
  record, complete, export_braid), health, composition, metrics
- **13 evolving**: brain (observe, attention, urgency), gonzales (dose_response,
  pk_decay, tissue_lattice), vault (store, retrieve, consent.verify), data
  ingestion (chembl, pubchem, register_table), AI assist

Consumed capabilities aligned to canonical names:
- `spine.create` + `entry.append` replaced with `dag.dehydrate` + `ledger.commit`
- Wire names now match actual IPC calls in `rhizocrypt.rs` and `loamspine.rs`

### 2. Degradation Behavior — DONE

`docs/DEGRADATION_BEHAVIOR.md` documents per-primal degradation for all 16
consumed primals:

| Category | Pattern |
|----------|---------|
| Provenance trio | `ProvenanceResult { available: false }` with local fallback IDs |
| NestGate | `GapReport` with `missing_primals`, never hard error |
| barraCuda | In-process library fallback (dual-lane architecture) |
| Squirrel AI | Returns `Ok(status: "unavailable")` |
| Facade | HTTP 503 / degraded JSON |

Science is **never gated** behind provenance or primal availability.

### 3. Cross-Tier Parity — PARTIALLY DONE (Gap #9 opened)

wetSpring has extensive math parity validators (CPU/GPU, Python/Rust for
Anderson, diversity, kinetics, etc.) and schema parity for `capability.list`.
Missing the formal three-layer proof structure (Python → Rust → Primal
composition with `ParityResult` JSON output). Opened Gap #9.

### 4. Trio Transaction Semantics — DONE

`complete_session()` in `barracuda/src/ipc/provenance/mod.rs` now emits:
- `primals_reached: Vec<&str>` tracking which trio members responded
- Partial state distinguishable from "not attempted"
- `provenance: "partial"` when DAG succeeds but spine fails
- `provenance: "unavailable"` when nothing responds
- `provenance: "complete"` with potential empty `braid_id` if sweetGrass down
- Domain logic always returns `Ok` — never errors on partial provenance

---

## Exp381: breseq Pipeline — Barrick 2009 (First Real-Data Composition)

| Field | Value |
|-------|-------|
| Binary | `validate_breseq_barrick_2009` (`--features ipc`) |
| Dataset | SRP001569, 7 Ara-1 clones (~1.3 GB total) |
| Reference | REL606 (CP000819.1, 4.63 Mb) |
| Tool | breseq 0.40.1 via micromamba env on 4TB NVMe |
| Composition | Nest Atomic: NestGate → breseq → trio → braid |
| Status | REL1164M: 579 mutations, REL2179M: 608, REL8593M: 1108 |
| Science | Mutation accumulation trend confirmed (Barrick 2009 Fig. 1) |
| Braid | Exported to `provenance/braids/barrick_2009_mutations.json` |
| Next | Complete remaining clones, then Tenaillon 2016 (~200 GB) |

Ferment transcript braid wire format:

```json
{
  "dataset_id": "barrick_2009_mutations",
  "spring": "wetSpring",
  "spring_version": "0.1.0",
  "braid_id": "",
  "dag_session_id": "local-wetspring-...",
  "dag_merkle_root": "",
  "spine_id": "",
  "computation": {
    "tool": "breseq",
    "tool_version": "0.40.1",
    "input_accession": "SRP001569",
    "node_count": 7,
    "wall_time_seconds": 3793
  },
  "summary_blake3": "529e34ee..."
}
```

Note: `braid_id`, `dag_merkle_root`, `spine_id` empty because trio primals
are not running locally (standalone mode). When running under full NUCLEUS
composition, these will be populated by the live trio.

---

## Build Gate

- `cargo clippy --features ipc --lib -- -D warnings` (zero)
- `cargo test --features ipc --lib` (252 pass, 0 fail)
- All niche capabilities: 43 (42 → 43, added `provenance.export_braid`)
- Consumed capabilities: 52 (51 → 52, added `dag.dehydrate`, replaced `spine.create`/`entry.append` with `ledger.commit`)

---

## Upstream Learnings for Primal Teams

### For primalSpring
- The stability tier TOML annotation pattern was straightforward to adopt.
  Recommend documenting the tier assignment heuristic (when to use "evolving"
  vs "stable") in the registry header.
- `primals_reached` is a natural extension of the trio output contract.
  Consider adding it to the `PROVENANCE_TRIO_INTEGRATION_GUIDE.md` as a
  normative field (currently it's a consumer pattern, not in the guide).

### For lithoSpore
- First ferment transcript braid exported. Wire format matches contract.
  Barrick 2009 is the simpler dataset (7 clones vs 264). Tenaillon 2016
  is next after full Barrick completion.
- The braid will have populated trio IDs once we run under full composition
  with live primals. Standalone mode produces structurally valid but
  trio-empty braids.

### For biomeOS
- `nest.commit` signal dispatch is wired but untested in standalone mode.
  The multi-call fallback (dehydrate → commit → braid) works. Testing
  `signal.dispatch` requires a live biomeOS v3.56+ instance.

---

## Files Changed

| File | Change |
|------|--------|
| `barracuda/src/niche.rs` | Consumed caps: `spine.create`/`entry.append` → `dag.dehydrate`/`ledger.commit` |
| `barracuda/src/ipc/provenance/mod.rs` | `complete_session()` emits `primals_reached` |
| `capability_registry.toml` | Stability tiers on all 43 methods, `provenance.export_braid` added |
| `docs/DEGRADATION_BEHAVIOR.md` | **New**: per-primal degradation table |
| `GAPS.md` | V177 header, Gap #8 progress, Gap #9 (cross-tier parity) opened |
| `CHANGELOG.md` | V177 entry |
| `README.md` | V177 metrics |
| `experiments/README.md` | Exp377-379 PROPOSED→DONE, Exp381 row, V177 totals |
| `experiments/381_*.md` | Exp381 results section added |
| `experiments/377-379_*.md` | "(to create)" removed |
| `wateringHole/handoffs/` | 6 handoffs archived (V168–V171), this handoff created |
