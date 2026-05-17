# wetSpring V168 — Compute Trio Absorption Handoff

**From:** wetSpring
**To:** primalSpring (coordination), barraCuda team, coralReef team, strandGate team
**Date:** May 14, 2026
**Audit:** "Ecosystem Status Update — May 14, 2026"

---

## Summary

V168 responds to the compute trio evolution wave. barraCuda v0.4.0 contract pin
absorbed. coralReef v0.1.0 declared as niche dependency. All doc references
to v0.3.12 updated to v0.4.0 across 18 files. plasmidBin deployment evolution
confirmed. sourDough internalization noted for future adoption.

---

## Actions Taken

### 1. barraCuda v0.4.0 Absorption

- `upstream_contract.rs` pin: `0.3.12` → `0.4.0`
- Cargo.toml comment updated
- Contract test passes: `version_ge("0.4.0", "0.4.0")` verified
- 18 doc files updated from v0.3.12 → v0.4.0 (all non-archived)
- `ipc/protocol.rs` doc comment updated
- Archived handoffs (fossil record) intentionally preserved at v0.3.12

New upstream capabilities available to wetSpring:
- Precision/E2E validation framework
- VFIO sovereign dispatch (AMD live, NVIDIA FECS-gated)
- `session::dispatch` module (from latest barraCuda pull)

### 2. coralReef v0.1.0 Niche Declaration

Added to `niche.rs::DEPENDENCIES`:
```rust
NicheDependency {
    name: CORALREEF,
    role: "shader",
    required: false,
    capability: "shader",
}
```

Already present in wetSpring's IPC surface:
- `discover.rs`: `discover_coralreef()` function
- `CONSUMED_CAPABILITIES`: `shader.compile.wgsl` (coralReef sovereign shader compiler)
- Deploy graphs: `wetspring_deploy.toml`, `wetspring_science_nucleus.toml`
- plasmidBin: `cells/wetspring_cell.toml` lists coralreef in requires

### 3. plasmidBin Deployment Evolution — Confirmed

wetSpring's plasmidBin integration verified:
- `cells/wetspring_cell.toml`: NUCLEUS base + life science domain + petalTongue live mode
- `manifest.toml`: `evolution = "composed"`, niche = 12 primals (full NUCLEUS + Meta)
- `cells_manifest.toml`: cell registered with biology domain
- `tools/composition_nucleus.sh`: references `$ECO_ROOT/infra/plasmidBin`

### 4. sourDough Internalization — Noted

sourDough subcommand mapping (v0.3.0–v0.6.0) replaces shell automation patterns.
Contract doc at `primalSpring/docs/SOURDOUGH_DEPLOYMENT_INTERNALIZATION.md` (not
yet published — found at `infra/wateringHole/handoffs/SOURDOUGH_V030_*`).
No local wetSpring changes needed until sourDough binary is available.

---

## Upstream Stale Data — For primalSpring Coordination

These upstream documents contain stale wetSpring data:

| Document | Location | Stale Field | Current Value |
|----------|----------|-------------|---------------|
| CROSS_SPRING_PARITY_SCORECARD | primalSpring/docs/ | gS Level | **L4** → should be **L5** |
| CROSS_SPRING_PARITY_SCORECARD | primalSpring/docs/ | Tests | **1,613** → should be **1,962** |
| CROSS_SPRING_PARITY_SCORECARD | primalSpring/docs/ | LTEE B7 | **STARTED / Rust NO / expected_values NO** → should be **Tier 2 COMPLETE / Rust YES / expected_values YES** |
| CROSS_SPRING_PARITY_SCORECARD | primalSpring/docs/ | Phase 32 PG | Lists PG-03, PG-05 open → should be **RESOLVED** |
| PRIMAL_GAPS.md (Wave 3) | primalSpring/docs/ | Version/gS | **V166b, gS L4** → should be **V168, gS L5** |
| infra/wateringHole README | infra/wateringHole/ | Version | **V163** → should be **V168** |
| infra/wateringHole README | infra/wateringHole/ | Tests | **1,613** → should be **1,962** |
| infra/wateringHole README | infra/wateringHole/ | gS Level | **L4** → should be **L5** |
| infra/wateringHole README | infra/wateringHole/ | LTEE B7 | **STARTED** → should be **COMPLETE** |
| plasmidBin manifest.toml | infra/plasmidBin/ | tests | **1902** → should be **1,962** |
| plasmidBin manifest.toml | infra/plasmidBin/ | barracuda_depth | **composing** → should be **composed** |

---

## Upstream Blockers — No Change for wetSpring

Per audit, none of these block spring evolution work:
- Songbird TURN client (geo-delocalized Tier 2)
- BearDog FIDO2 (SoloKey witness)
- genomeBin Tier 3 (USB packaging)
- sporePrint pipeline (liveSpore publish)

---

## Open Gaps — Unchanged

| ID | Status | Blocker |
|----|--------|---------|
| PG-02 | Deployment-only | Provenance trio live stack |
| PG-04 | Deployment-only | NestGate live deployment |

---

## Metrics (V168)

| Metric | Value |
|--------|-------|
| guideStone | **Level 5** (primal proof) |
| barraCuda | **v0.4.0** (was v0.3.12) |
| coralReef | **v0.1.0** (now in niche DEPENDENCIES) |
| Lib tests | 1,962 (0 failures) |
| Contract test | `upstream_contract::pinned_version_matches_linked` PASS |
| Clippy pedantic+nursery | 0 warnings |
| Primal gaps | 2 open (deployment-only), 20 resolved/closed |
| Papers | 63/63 (+ 9 LTEE queued) |
| Composition status | **composed** |
| `unsafe` | 0 (`forbid` at workspace level) |

---

*Submitted to primalSpring via `wateringHole/handoffs/` per
`NUCLEUS_SPRING_ALIGNMENT.md` feedback protocol.*
