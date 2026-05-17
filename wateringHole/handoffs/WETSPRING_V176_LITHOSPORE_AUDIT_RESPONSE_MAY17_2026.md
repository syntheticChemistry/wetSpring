# wetSpring V176 — lithoSpore Audit Response

**Date:** May 17, 2026
**From:** wetSpring
**To:** lithoSpore, primalSpring, Provenance Trio teams
**Scope:** Ferment transcript braid export, trio capability reconciliation, remaining pipeline work

---

## Response to lithoSpore Audit

### What We Did (V176)

1. **New `provenance.export_braid` IPC method** — produces the ferment transcript
   braid JSON in the exact wire format defined by the lithoSpore contract:

   ```json
   {
     "dataset_id": "tenaillon_2016_genomes",
     "spring": "wetSpring",
     "spring_version": "0.1.0",
     "braid_id": "braid-...",
     "dag_session_id": "dag-...",
     "dag_merkle_root": "...",
     "spine_id": "spine-...",
     "computation": {
       "tool": "breseq",
       "tool_version": "0.38.1",
       "input_accession": "PRJNA294072",
       "input_blake3": "...",
       "output_blake3": "...",
       "wall_time_seconds": 86400,
       "node_count": 264
     },
     "summary_blake3": "...",
     "timestamp": "..."
   }
   ```

   The handler internally calls `complete_session()` to run the full
   dehydrate → commit → braid sequence, then packages the identifiers
   with the computation metadata provided by the caller.

2. **New `braid_handoff.rs` module** — `FermentTranscriptBraid` and
   `ComputationMetadata` structs with serde Serialize/Deserialize,
   `from_session_result()` constructor, `to_json()` export, and
   `handle_export_braid()` IPC handler. 5 tests: serialization roundtrip,
   field extraction from session result, parameter validation (2), graceful
   degradation.

3. **loamSpine capability routing fixed** — corrected from
   `capability_call(socket, "session", "commit", ...)` to
   `capability_call(socket, "ledger", "commit", ...)` per the
   `PROVENANCE_TRIO_INTEGRATION_GUIDE.md` v2.0 canonical method reference.
   This was a silent bug — biomeOS would have routed to the wrong capability
   domain if a `session` provider existed alongside `ledger`.

4. **Guard constants updated** — niche capabilities 42 → 43, capability
   domains provenance methods 3 → 4, dispatch/wire method count 37 → 38,
   CI cross-sync threshold 452 → 453.

### What's Still Needed (wetSpring-Owned)

| Item | Priority | Status | Notes |
|------|----------|--------|-------|
| breseq pipeline binary | HIGH | Not started | Needs breseq 0.38+ installed, SRA toolkit for NCBI download |
| Tenaillon 2016 processing | HIGH | Not started | ~200 GB raw data, 264 genomes, ~24h compute |
| Barrick 2009 processing | HIGH | Not started | ~15 GB raw data, 19 genomes, ~2h compute |
| First braid handoff to lithoSpore | HIGH | Blocked by above | Produces `provenance/braids/tenaillon_2016_genomes.json` |
| Good 2017 metagenomic | MEDIUM | Not started | ~50 GB |
| Blount 2012 replay | MEDIUM | Not started | ~30 GB |

### lithoSpore Request Acknowledgment

| # | lithoSpore Request | wetSpring Response |
|---|-------------------|-------------------|
| 1 | Degradation behavior | **Already done** — all trio interactions degrade gracefully, never panic, documented in `composition_health.rs` |
| 2 | Freeze method names | **Aligned** — using canonical names from trio integration guide v2.0. loamSpine routing fixed. |
| 3 | Trio transaction semantics | **Partial** — wetSpring treats partial provenance as success with reduced coverage. Trio teams need to define rollback semantics. |
| 4 | UDS socket ownership | **Not our scope** — forwarded to biomeOS/songBird. wetSpring uses `discover::discover_socket()` cascade. |
| 5 | `nest.store` signal dispatch | **Annotated** — Wave 17 signals wired in graph TOMLs and `try_nest_commit_signal()`. Waiting on biomeOS signal routing. |
| 6 | Produce ferment transcript braids | **Wire format done** (V176). Pipeline not started — requires breseq + SRA data. |
| 7 | `spore.instantiate` | **Not our scope** — forwarded to biomeOS. |
| 8 | Wave 20 `capability.list` | **Done** — returns complete inventory with `count` field. |

### Trio Adoption Status Update

Per `SWEETGRASS_SPRING_BRAID_PATTERNS.md` adoption checklist:

| Item | Status |
|------|--------|
| `ipc/provenance/` directory with per-primal modules | **Done** — `rhizocrypt.rs`, `loamspine.rs`, `sweetgrass.rs`, `braid_handoff.rs` |
| `rhizocrypt.rs` — session create, event append, dehydration | **Wired** — `begin_session()`, `record_step()`, `dehydrate()` |
| `loamspine.rs` — session commit | **Wired** — `commit_session()` via `ledger.commit` (fixed V176) |
| `sweetgrass.rs` — braid create with agents | **Wired** — `create_attribution_braid()` + `record_experiment_provenance()` |
| Session lifecycle in handlers | **Wired** — `provenance.begin/record/complete` IPC handlers |
| Graceful degradation | **Done** — all paths return `"unavailable"` when trio absent |
| Deploy graph with trio nodes | **Done** — `graphs/wetspring_deploy.toml` |
| Ferment transcript export | **Done** (V176) — `provenance.export_braid` handler |

**Updated status:** rhizoCrypt **Wired**, loamSpine **Wired** (fixed), sweetGrass **Wired**, Braid Export **Wired**. Overall: **Operational** (pending live trio deployment for end-to-end validation).

---

## Metrics (V176)

| Metric | Value |
|--------|-------|
| Niche capabilities | 43 |
| Consumed capabilities | 53 (registry 453) |
| Provenance methods | 4 (begin, record, complete, export_braid) |
| Lib tests | 252 pass (0 failures) |
| Clippy | zero warnings (pedantic + nursery) |
