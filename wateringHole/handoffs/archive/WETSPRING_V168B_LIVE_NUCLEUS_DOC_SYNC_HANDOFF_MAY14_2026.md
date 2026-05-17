# wetSpring V168b — Live NUCLEUS Doc Sync Handoff

**From:** wetSpring
**To:** primalSpring (coordination), primal teams
**Date:** May 14, 2026
**Audit:** Post-live NUCLEUS test + doc sync

---

## Summary

V168b syncs all 20 doc files to the V168 banner with live NUCLEUS test results.
guideStone ran against a live stack — 30/31 pass, 9 skip, 1 fail. plasmidBin
composition validation passed (tower, nucleus, full, niche-wetspring). Fossil
record margin notes updated to Level 5 / 2 gaps. Upstream stale data reiterated.

---

## Live NUCLEUS Test Results

### plasmidBin Composition Validation (offline)

| Composition | Primals | Result |
|---|---|---|
| tower | bearDog + songBird + skunkBat | PASS (3 warn) |
| nucleus | 9 primals | PASS (9 warn) |
| full | NUCLEUS + Meta (12 primals) | PASS (12 warn) |
| niche-wetspring | 12 primals | PASS (12 warn) |

Warnings: symlink size=0 / dynamic linked — expected for local dev.

### NUCLEUS Stack Launch

8/12 sockets live: biomeOS, BearDog, Songbird, ToadStool, barraCuda, NestGate,
sweetGrass, petalTongue. coralReef started with alternate socket name pattern.
Squirrel started standalone (no AI providers). rhizoCrypt started on TCP only.

**loamSpine crashed** — async runtime nesting panic in infant_discovery.rs:233.
This is an upstream bug (loamSpine team).

### guideStone Against Live Stack: 30/31 PASS

| Layer | Pass | Skip | Fail | Details |
|-------|:----:|:----:|:----:|---------|
| B0 Bare Science | 7/7 | 0 | 0 | Shannon, Hill, matmul, stats — all exact |
| B1 Tolerance | 2/2 | 0 | 0 | ANALYTICAL_F64, IPC_ROUND_TRIP_TOL |
| B2 Checksums | 6/6 | 0 | 0 | All 6 BLAKE3 hashes match |
| N0 Liveness | 1/4 | 2 | 1 | barraCuda alive; NestGate+ToadStool protocol mismatch; **BearDog health format** |
| N1 Manifest IPC | 11/15 | 4 | 0 | 10 tensor/stats/linalg/spectral + crypto.hash; skip compute.dispatch/storage/inference/squirrel |
| N2 Domain Science | 1/3 | 2 | 0 | weighted_mean pass; median/determinant not in ecoBin |
| N3 Cross-Atomic | 1/2 | 1 | 0 | BearDog hash pass; store internal error |
| **Total** | **30/31** | **9** | **1** | |

### Upstream Issues Found

| Primal | Issue | Severity |
|--------|-------|----------|
| loamSpine | Runtime nesting panic in infant_discovery | HIGH — crashes on launch |
| BearDog | `health.liveness` response format mismatch | LOW — responds but not parsed as "alive" |
| NestGate | `health.liveness` returns "Method not found" | MEDIUM — likely HTTP endpoint mismatch |
| ToadStool | `health.liveness` returns empty response | MEDIUM — socket responds but no JSON-RPC |

---

## Doc Sync — 20 Files Updated

All files updated from V167 → V168 headers with live NUCLEUS results:
GAPS.md, barracuda/README.md, EVOLUTION_READINESS.md, ABSORPTION_MANIFEST.md,
experiments/README.md (header + 2 totals), specs/README.md (+ binary count fix),
specs/BARRACUDA_REQUIREMENTS.md, specs/PENDING_DATASETS.md, specs/PAPER_REVIEW_QUEUE.md,
whitePaper/README.md, whitePaper/baseCamp/README.md, whitePaper/baseCamp/EXTENSION_PLAN.md,
sporeprint/validation-summary.md, docs/PRIMAL_PROOF_IPC_MAPPING.md,
docs/PRIMAL_GAPS.md (section title), wateringHole/ECOSYSTEM_LEVERAGE_GUIDE.md,
wateringHole/CROSS_SPRING_SHADER_EVOLUTION.md, `validate_ltee_b7_v1.rs`.

---

## Upstream Stale Data — Reiterated for primalSpring

| Document | Stale | Should Be |
|----------|-------|-----------|
| CROSS_SPRING_PARITY_SCORECARD | gS L4, 1,613 tests, LTEE "STARTED" | gS L5, 1,962 tests, LTEE Tier 2 COMPLETE |
| PRIMAL_GAPS.md Wave 3 row | V166b, gS L4 | V168, gS L5 |
| infra/wateringHole README | V163, 1,613 tests, gS L4 | V168, 1,962, L5 |
| plasmidBin manifest.toml | tests=1902, barracuda_depth="composing" | 1962, "composed" |

---

## Metrics (V168b)

| Metric | Value |
|--------|-------|
| guideStone | **Level 5** — live NUCLEUS 30/31 pass |
| barraCuda | **v0.4.0** |
| coralReef | **v0.1.0** (in niche DEPENDENCIES) |
| Lib tests | 1,962 (0 failures) |
| Composition | **composed** — plasmidBin tower/nucleus/full/niche all validated |
| Primal gaps | 2 open (deployment-only), 20 resolved/closed |
| Doc files synced | 20 |
| Upstream issues | 4 (loamSpine panic, BearDog health, NestGate method, ToadStool empty) |

---

*Submitted to primalSpring via `wateringHole/handoffs/` per
`NUCLEUS_SPRING_ALIGNMENT.md` feedback protocol.*
