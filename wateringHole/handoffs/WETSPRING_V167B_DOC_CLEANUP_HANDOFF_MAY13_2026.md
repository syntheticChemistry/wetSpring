# wetSpring V167b — Doc Cleanup + Debris Audit Handoff

**From:** wetSpring
**To:** primalSpring (coordination), primal teams, spring teams
**Date:** May 13, 2026
**Audit:** Post-V167 doc sync + debris sweep

---

## Summary

V167b is a documentation-only pass. All 18 doc files synced from V166→V167
banner. Internal metric inconsistencies resolved (binary counts, test counts,
gap counts, guideStone Level). Rust source debris audit found zero
TODO/FIXME/HACK items; 4 stale version/wording references in `.rs` fixed.
Zero compilation regressions. Upstream handoff for primal team evolution below.

---

## Doc Sync — V167 Banner Consistency

18 files updated from V166 banner to V167:

| File | Key Fix |
|------|---------|
| `README.md` | Current release → V167, guideStone L5, PG-03+PG-05 resolved |
| `GAPS.md` | Last updated → V167 |
| `CONTEXT.md` | Binary count 364→367 |
| `barracuda/README.md` | V167 banner, CPU modules 49→41, binaries 333→367, tests 1902→1962 |
| `barracuda/EVOLUTION_READINESS.md` | V167, gS L5, gaps 4/18→2/20, experiments 260→384 |
| `barracuda/ABSORPTION_MANIFEST.md` | V167, gS L4→L5, gaps 4/18→2/20 |
| `sporeprint/validation-summary.md` | Date bump, gS L5 status line added |
| `docs/PRIMAL_PROOF_IPC_MAPPING.md` | V167 |
| `whitePaper/README.md` | V167, gS L5 |
| `whitePaper/baseCamp/README.md` | V167, gS L5 |
| `whitePaper/baseCamp/EXTENSION_PLAN.md` | V167 |
| `experiments/README.md` | V167 banner + stale mid-file totals (5800→5900, 1592→1962, 340→367) |
| `specs/PAPER_REVIEW_QUEUE.md` | V167 |
| `specs/BARRACUDA_REQUIREMENTS.md` | V167, gS L5 |
| `specs/PENDING_DATASETS.md` | V167 |
| `specs/README.md` | V167, gS L5 |
| `wateringHole/README.md` | Archive count 168→184 |

---

## Rust Source Debris — 4 Fixes

| File | Fix |
|------|-----|
| `bin/wetspring_guidestone.rs` | Reference table: "Level 4, 67/67" → "Level 5, 38/38 NUCLEUS + 4 env skip" |
| `ipc/discover.rs` | PG-03 doc: "when Songbird implements" → "PG-03 RESOLVED, Songbird wired as primary" |
| `bin/validate_ltee_b7_v1.rs` | Version string: "V165" → "V167" |
| `facade/provenance.rs` | guideStone constants: "29/29 PASS v0.1.0" → "38/38 PASS v0.2.0" |

---

## Debris Audit — Clean

| Category | Result |
|----------|--------|
| TODO in `.rs` | **0** |
| FIXME in `.rs` | **0** |
| HACK in `.rs` | **0** |
| V166 in `.rs` | **0** (post-fix) |
| "Level 4" in `.rs` | **0** (post-fix) |
| Orphan `.toml` workloads | **0** |
| Python scripts | **58** — all registered in `provenance_registry.rs` as reproducibility anchors (Write→Absorb→Lean) |
| Shell scripts | **19** — CI/validation tooling, no orphans |
| `#[allow()]` | **0** |
| `unsafe` | **0** (`forbid` workspace-level) |

---

## Primal Evolution — What wetSpring Learned for Upstream Teams

### For bearDog team
- `crypto.hash` uses `{ data: base64 }` — confirmed correct. However, `crypto.sign`
  uses `{ message: base64 }`. Downstream springs may confuse the two. Recommend
  a note in `CRYPTO_WIRE_CONTRACT.md` clarifying the field name difference.

### For skunkBat team
- `security.audit_log` is the read-side (cursor-based polling). Springs that
  only emit audit events use `audit.event` and `audit.forward`. This distinction
  is not obvious from `capability_registry.toml`. Recommend adding a
  "direction: emit|read" column to the registry.

### For Songbird team
- `capability.resolve` is live and working as fallback-first pattern. wetSpring
  wires it as primary with static table fallback. The pattern is reusable
  by all 8 springs.

### For NestGate team
- `content.*` vs `storage.*` naming is confirmed intentional per biomeOS v3.53.
  PG-04 remains deployment-only — wetSpring code is ready, waiting for live NestGate.

### For Provenance Trio (rhizoCrypt, loamSpine, sweetGrass)
- Alias resolution (GAP-36) works at wire level. PG-02 remains deployment-only.
  wetSpring method names (`dag.*`, `session.commit`, `braid.create`) align with
  alias targets.

### For lithoSpore
- B7 Tenaillon Tier 2 is COMPLETE (27/27 PASS). `ltee_b7_expected_values.json`
  structurally aligned with `module6_breseq.json`. 9 LTEE papers queued. Ready
  for lithoSpore `ltee-genomics` module to consume.

### For projectNUCLEUS / neuralAPI / biomeOS
- wetSpring holds on full NUCLEUS composition per directive. 136/136 proto-nucleate
  alignment. 7 deploy graphs validated. `--format json` output for Tier 2 ingestion.
  UniBin musl binary verified for plasmidBin deployment.

### For foundation
- 10/10 threads active. Thread 4 (expression + targets) seeded with 36 targets.
  Thread 5 (LTEE B7 Tenaillon) Tier 2 complete.

### Composition Patterns for NUCLEUS Deployment
- **IPC-first defaults** (`default = []`) — no primal linked at compile time.
- **Fallback-first discovery** — `resolve_via_songbird()` → static table → structured gap report.
- **`validate_parity_or_skip`** — graceful skip when primal absent, exit 0 in NUCLEUS mode.
- **Feature gates** — `ipc`, `guidestone`, `primal-proof` isolate surfaces cleanly.
- **guideStone L5 pattern** — 38 checks validate full domain science via IPC,
  4 environmental skips for absent infrastructure (not code gaps).

---

## Metrics (V167b)

| Metric | Value |
|--------|-------|
| guideStone | **Level 5** (primal proof) |
| Lib tests | 1,962 (0 failures) |
| Clippy pedantic+nursery | 0 warnings |
| Primal gaps | 2 open (deployment-only), 20 resolved/closed |
| Papers | 63/63 (+ 9 LTEE queued) |
| LTEE B7 | Tier 2 COMPLETE (27/27 PASS) |
| `unsafe` | 0 (`forbid` at workspace level) |
| C dependencies | 0 (deny.toml enforced) |
| Doc files synced | 18 |
| Source debris fixes | 4 |

---

*Submitted to primalSpring via `wateringHole/handoffs/` per
`NUCLEUS_SPRING_ALIGNMENT.md` feedback protocol.*
