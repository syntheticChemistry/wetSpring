# wetSpring GAPS — Known Architectural Gaps

Intentional gaps scaffolded by the wetSpring science NUCLEUS deployment.
Each gap is documented so it feeds back to the owning team via wateringHole
and primalSpring evolution tracking.

Last updated: 2026-05-19 (V179 — Wave 23 E2E absorption. ParityResult struct + barrick_2009_parity.json
with formal L1 vs L2 cross-tier comparison. Consumer socket pattern absorbed in Python + shell.
Barrick 2009 systems study v2: 7/7 clones, 486 sovereign vs 569 breseq (0.85 ratio). Adaptive GPU
dispatch: CPU for 36bp reads, GPU for pileup+calling. Tenaillon 2016 workspace ready (590GB, 312
accessions, 524 FASTQs).)

---

## 1. Ionic Contract Negotiation

**Owner:** primalSpring Track 4 (`BondingConstraint + BondingPolicy`)

**Status:** Scaffolded — bonding metadata declared, no negotiation protocol.

The facade declares its bonding capabilities in `bonding_metadata.json` and
exposes them via `GET /api/v1/system/composition`. However, there is no
automated protocol for:

- Establishing a new ionic bond (capability scoping, duration, attribution)
- Modifying an existing bond (scope expansion/reduction)
- Terminating a bond (cleanup, final provenance seal)
- Verifying a bond's validity during request dispatch

primalSpring's `BondType::Ionic` and `TrustModel::Contractual` define the
abstract model. The concrete negotiation protocol (handshake, contract
serialization, mutual verification) is the gap.

**Impact:** External researchers cannot self-service ionic bond creation.
Manual setup required via ecosystem maintainers.

---

## 2. Cross-Spring Data Exchange via RootPulse

**Owner:** RootPulse team

**Status:** Not started — NestGate stores locally, no remote pull protocol.

The `data.fetch.*` handlers store fetched data in NestGate with BLAKE3 hashes
and provenance sessions. The `vault.*` handlers provide consent-gated storage.
However:

- No protocol for another spring's NUCLEUS to pull provenance-wrapped data
  subsets from this spring's NestGate
- No RootPulse sync mechanism for cross-spring provenance chain exchange
- No differential sync (only full fetch or nothing)

**Impact:** Each spring operates as a silo. Ionic bonds can declare shared
capabilities but cannot exchange stored data with provenance continuity.

---

## 3. Public Chain Anchor

**Owner:** loamSpine team (Paper 20 Section 6 — marked "Needed")

**Status:** Not started — provenance braids exist but have no public anchor.

The provenance trio produces:
- rhizoCrypt ephemeral DAG sessions
- loamSpine immutable ledger commits
- sweetGrass semantic braids (PROV-O aligned)

Missing: a mechanism to anchor these braids to a public verifiable ledger
(timestamping service, blockchain anchor, or similar). Without this, the
provenance chain is verifiable only within the ecosystem trust boundary.

**Impact:** External verification of provenance requires trusting the
ecosystem's own ledger. Public anchoring would provide independent
verification.

---

## 4. petalTongue Client-Side WASM

**Owner:** petalTongue team

**Status:** Server-side SVG rendering works; client WASM not yet built.

The facade currently renders grammar expressions server-side via petalTongue
RPC, returning SVG strings. The endgame (Phase 3 of web deployment) is:

- Compile petalTongue grammar engine to WASM
- Ship to browser as a client-side renderer
- Eliminate server round-trip for visualization
- Enable offline rendering with cached data

**Impact:** All grammar rendering requires live HPC connection. No offline
interactive visualization. Plotly.js remains the primary client renderer.

---

## 5. ludoSpring / esotericWebb Interactive Composition

**Owner:** ludoSpring team, esotericWebb team

**Status:** Not started — explorer uses Plotly.js; no game/composition engine.

The current explorer is a Plotly.js chart viewer with parameter sliders.
The endgame is:

- **ludoSpring:** Interactive scenes with real physics, navigation, and
  exploration (gaming engine for science)
- **esotericWebb:** Composition contracts for combining visualizations,
  data streams, and interaction modes
- **petalTongue:** Grammar-of-graphics rendering within ludoSpring scenes

Together these would replace the static chart explorer with a living,
navigable data environment.

**Impact:** Data exploration is limited to 2D charts. No 3D navigation,
no physics simulation overlays, no interactive data manipulation.

---

## 6. hotSpring Physics Simulations

**Owner:** hotSpring team

**Status:** Not started — same NUCLEUS pattern, different science domain.

hotSpring would use the same deploy graph pattern proven by wetSpring to
run real physics simulations:

- Anderson localization in physical systems (not just biological analogy)
- Wave propagation in disordered media
- Thermal transport in heterogeneous materials

The NUCLEUS deployment pattern, facade architecture, and provenance wrapping
are directly transferable.

**Impact:** Physics simulations are only available as analytical math in
wetSpring's Anderson modules. No numerical physics simulation engine.

---

## 7. Radiating Attribution Calculator

**Owner:** sweetGrass team + sunCloud team
(Paper 20 Section 6 — marked "Low, Phase 4")

**Status:** Not started.

When data flows through ionic bonds across multiple springs, attribution
must "radiate" — each downstream use must credit upstream contributors
proportionally. This requires:

- Attribution weight calculation based on derivation depth
- Credit distribution across the NFT vertex DAG
- Integration with sunCloud's economic model

**Impact:** Novel Ferment Transcript vertices record derivation chains but
do not compute or distribute attribution weights. The gAIa commons model
requires this for its economic sustainability.

---

## 8. Ferment Transcript Braid Pipeline (lithoSpore Audit — V176)

**Owner:** wetSpring
**Priority:** High — lithoSpore's highest-priority request for wetSpring

lithoSpore ships ~3.4 MB summary statistics on USB but the raw upstream data
(Tenaillon 2016: ~200 GB, Barrick 2009: ~15 GB) cannot travel with the artifact.
The ferment transcript pattern requires wetSpring to:

1. Run the computation (breseq pipeline on raw NCBI/SRA data)
2. Record provenance via the trio (DAG + spine + braid)
3. Export a portable `FermentTranscriptBraid` JSON for lithoSpore

**V176 progress:** `provenance.export_braid` method wired, `FermentTranscriptBraid`
struct implemented, loamSpine capability routing fixed to canonical `ledger.commit`.

**V177 progress — Exp381 executing:** `validate_breseq_barrick_2009` binary built.
Full Nest Atomic composition: `NestGate` → `breseq` → trio → braid. Environment:
`micromamba breseq-env` on 4TB NVMe with `breseq 0.40.1`, `samtools`, `bowtie2`, `R`.
Reference genome `CP000819.1` (REL606, 4.63 Mb). 7 SRA runs (SRP001569) downloaded.
2/7 clones processed (REL1164M: 579 mutations, REL8593M: 1108 mutations) — mutation
accumulation trend confirmed. Ferment transcript braid exported to
`/mnt/4tb-work/.../provenance/braids/barrick_2009_mutations.json`. Re-running for
full 7/7 completion.

**Composition approach:** This is a Nest Atomic composition, not a standalone script.
Data acquisition via NestGate `storage.fetch_external` (HTTPS → cache). Computation
orchestrated as a NUCLEUS workload (primalSpring `spring_deploy_manifest.toml` already
declares `science.ncbi_fetch` for wetSpring). Provenance recorded via trio during
pipeline execution. Braid exported for lithoSpore handoff. Ionic bonding model applies
for external collaborator access (Barrick Lab, UT Austin).

**Workspace:** `/mnt/4tb-work/ecoPrimals/springs/wetSpring/datasets/ltee/` (3.4 TB free).

| Dataset | Size | Status |
|---------|------|--------|
| Barrick 2009 (7 clones) | ~1.3 GB | **In progress** — 2/7 done, re-running for 7/7 |
| Tenaillon 2016 (264 genomes) | ~200 GB | Spec in Exp380, next after Barrick |
| Good 2017 (metagenomic) | ~50 GB | In paper queue |
| Blount 2012 (replay seq) | ~30 GB | In paper queue |

---

## 9. Cross-Tier Parity (Wave 20 PM → Wave 23)

**Owner:** wetSpring
**Priority:** Medium — lithoSpore pattern adoption
**Status:** L1 vs L2 documented, L3 pending

`ParityResult` struct added to `validation::parity` module. Formal parity
JSON exported to `experiments/results/barrick_2009_parity.json` with per-clone
L1 (breseq 0.40.1) vs L2 (sovereign Rust) comparison.

**Barrick 2009 parity (7 clones):**
- Total: L1=569, L2=486 (0.85 ratio), 0 position matches
- Early generations (2K-10K): L2 over-calls (mapping artifacts in repeats)
- Late generations (30K-40K): L2 under-calls (0.75 freq threshold rejects
  sub-clonal sweeps that breseq's polymorphism mode captures)
- Zero position overlap indicates coordinate representation mismatch

**Remaining:**
- L2 vs L3 (local vs IPC-composed) — requires live trio
- Position-level matching with ±5bp window tolerance
- Adaptive frequency threshold per generation
- lithoSpore `expected_values.json` contract for Barrick 2009

---

## 10. Stale Socket Detection (Wave 22 — RESOLVED)

**Owner:** wetSpring (consumer side), biomeOS (server side), plasmidBin (infra)
**Priority:** RESOLVED — absorbed from primalSpring Wave 22 audit
**Status:** Implemented

Discovery functions used `path.exists()` to check UDS sockets, which returned
true for stale sockets (file exists, no listener). This caused ~100ms timeouts
per stale socket during trio discovery.

**Resolution:**
- Rust: `socket_is_alive()` connect-probe with 50ms timeout replaces `path.exists()`
  in all 6 discovery tiers. `DEAD_SOCKET_CACHE` for session-level negative caching.
- Python: `tools/uds_send.py` — `socket_is_alive()` + `_dead_sockets` set
- Shell: `tools/nucleus_composition_lib.sh` — `socket_is_alive()` + `_DEAD_SOCKETS`
  associative array. `tools/composition_nucleus.sh` — `wait_for_socket()` upgraded
  from `-S` file-type check to connect-probe.
- Upstream report filed: `WETSPRING_UPSTREAM_BIOMEOS_STALE_SOCKETS_MAY18_2026.md`
- Server-side cleanup confirmed across all 14 primals (primalSpring audit)

---

## 11. Variant Caller Parity (Active — Calibration Gap)

**Owner:** wetSpring
**Priority:** High — directly affects lithoSpore braid quality
**Status:** In progress — v1 refinement deployed, parity improving

Sovereign Rust pipeline over-calls variants vs breseq baseline. Root cause:
simple frequency thresholds vs breseq's Bayesian mixture model with base
quality weighting.

**v0 (permissive):** ~34,000 variants per clone (8,500x over breseq)
**v1 (refined):** ~60-78 variants per clone (15-19x over breseq)

Remaining work: quality-weighted binomial model in SnpCallingF64 shader,
mapper-level deduplication of repetitive region alignments.

Parity evolution tracked in `provenance/braids/barrick_2009_refined_v1.json`
with exact parameter derivations anchored — no magic numbers.

---

## Summary Table

| # | Gap | Owner | Priority | Phase |
|---|-----|-------|----------|-------|
| 1 | Ionic contract negotiation | primalSpring Track 4 | High | 2 |
| 2 | Cross-spring data exchange | RootPulse | High | 2 |
| 3 | Public chain anchor | loamSpine | Medium | 3 |
| 4 | Client WASM renderer | petalTongue | Medium | 3 |
| 5 | Interactive composition | ludoSpring + esotericWebb | Low | 4 |
| 6 | Physics simulations | hotSpring | Low | 4 |
| 7 | Radiating attribution | sweetGrass + sunCloud | Low | 4 |
| 8 | Ferment transcript pipeline | wetSpring | **High** | 2 |
| 9 | Cross-tier parity | wetSpring | Medium | 2 |
| 10 | Stale socket detection | wetSpring + biomeOS | **RESOLVED** | — |
| 11 | Variant caller parity | wetSpring | **High** | 2 |

---

*This document is maintained by wetSpring and fed back to wateringHole
for ecosystem-wide gap tracking. Each gap discovered during the science
NUCLEUS deployment informs the primal teams responsible for resolution.*
