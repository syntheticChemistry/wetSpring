# wetSpring V170 — Wave 20 Schema Standardization Handoff

**From:** wetSpring
**To:** primalSpring (coordination), primal teams, spring teams
**Date:** May 16, 2026
**Audit:** Wave 20 Schema Standardization + E2E Validation

---

## Summary

V170 absorbs all Wave 20 common checklist items for wetSpring:

1. **`count` field** added to `capability.list` response
2. **Registry synced** to 452 methods (`primal.list` added)
3. **`primal.announce`** moved from `[signals].pending` to adopted
4. Tests updated and passing: 252 lib, 7 ci_cross_sync, 18 ipc_roundtrip

---

## Changes

### `capability.list` — Canonical Envelope Compliance

`handle_capability_list()` now returns `"count": caps.len()` alongside the
existing `"capabilities"` string array. The canonical subset is:

```json
{
  "capabilities": ["capability.list", "identity.get", "health.check", ...],
  "count": 42,
  "primal": "wetspring"
}
```

Extra fields (`methods`, `provided_capabilities`, `consumed_capabilities`,
`domain`, `version`, `cost_estimates`, `semantic_mappings`,
`operation_dependencies`) are preserved unchanged.

### Registry Sync — 452

- `capability_registry.toml` `[signals].registry_sync` bumped 451 → 452
- `ci_cross_sync.rs` module doc 451 → 452
- `ci_cross_sync.rs` `canonical_registry_accessible_and_nontrivial` threshold
  440 → 442 (floor with 452 expected)

### `primal.announce` — Fully Adopted

Previously listed as `[signals].pending` in `capability_registry.toml`
despite being consumed in `niche::CONSUMED_CAPABILITIES` (since V169).
Now correctly listed in `[signals].adopted`. `[signals].wave` bumped 17 → 20.

### Test Coverage

| Test | Suite | What It Validates |
|------|-------|-------------------|
| `capability_list_wire_standard_l2` | dispatch.rs (unit) | `count == capabilities.len()` |
| `capability_list_includes_all_domains` | ipc_roundtrip.rs (integration) | `count` present and consistent over socket |
| `canonical_registry_accessible_and_nontrivial` | ci_cross_sync.rs (integration) | Registry has 452+ methods |

---

## What We Don't Do Yet (Correctly Excluded)

| Item | Status | Why |
|------|--------|-----|
| `primal.list` handler | N/A | This is a biomeOS method; springs don't serve it |
| `--provenance-dir` | DEFERRED | Useful for B7 Tier 3 when sweetGrass TCP resolves |
| `s_schema_standard` scenario | CANDIDATE | wetSpring has ci_cross_sync covering same intent |
| `nest.commit` E2E (`s_nest_commit_live`) | WIRED | Already signal-path dispatched since V169 |

---

## Remaining Wave 20 Observations for Upstream

### For primalSpring

- **wetSpring scorecard**: Should now show V170, gS L5, 1,962 tests, LTEE
  B7 Tier 2 COMPLETE, 50 consumed capabilities, registry 452, `count` present.
- **Schema validation scenario**: `ci_cross_sync.rs` 7 tests achieve the same
  goal as `s_schema_standard` — validates surface parity, consumed domains,
  registry nontrivial. Adding a formal `s_schema_standard` is optional.

### For All Springs

- **`count` is one line**: `"count": caps.len()` in the JSON builder. Takes
  5 minutes. wetSpring's pattern: assign capabilities to a local variable,
  reference it for both `"capabilities"` and `"count"`.
- **`primal.list` is biomeOS**: Springs don't need to implement it — just
  bump registry cross-sync thresholds to 452.
- **`primal.announce` stale `pending` flags**: If your `capability_registry.toml`
  or similar config still lists `primal.announce` as pending/candidate, update
  it. The method has been in `CONSUMED_CAPABILITIES` since Wave 17.

---

## Metrics (V170)

| Metric | Value |
|--------|-------|
| guideStone | **Level 5** — 30/31 live NUCLEUS pass |
| Registry sync | **452** methods |
| Consumed capabilities | **50** (33 canonical + 15 legacy + 2 signal) |
| `capability.list` `count` | **Present** (matches `capabilities.len()`) |
| `primal.announce` | **Adopted** |
| Lib tests | 1,962 (0 failures) |
| CI cross-sync | 7/7 pass |
| IPC roundtrip | 18/18 pass |
| Clippy | **ZERO warnings** (pedantic+nursery, both crates) |
| Papers | 63/63 DONE, 9 LTEE queued |

---

*Submitted to primalSpring via `wateringHole/handoffs/` per
`NUCLEUS_SPRING_ALIGNMENT.md` feedback protocol.*
