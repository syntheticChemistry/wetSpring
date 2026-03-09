# Exp327: petalTongue Visualization V1

**Phase:** V100 (petalTongue integration)
**Date:** 2026-03-09
**Binary:** `validate_visualization_v1`
**Features:** `json`

## Status: PASS (45/45)

## Scope

Control validation for the new `barracuda/src/visualization/` module — wetSpring's
petalTongue-compatible scenario export system.

## Domains

| Domain | Checks | Description |
|--------|--------|-------------|
| V1 Schema | 11 | DataChannel serialization: channel_type tags, required fields, skip_serializing_if |
| V2 Scenarios | 19 | ecology, ordination, dynamics, anderson, benchmarks — structure, channel counts |
| V3 IPC Params | 4 | Scenario serialization, domain field, push_render failure on missing socket |
| V4 Full Chain | 7 | full_pipeline_scenario merges ecology+dynamics, edges, JSON round-trip |
| V5 Metadata | 4 | ScientificRange, family=wetspring, semver version, mode=live-ecosystem |

## Architecture

```
visualization/
  types.rs         — DataChannel enum (6 variants), ScientificRange, EcologyScenario
  ipc_push.rs      — PetalTonguePushClient (discover socket, push_render, push_append, push_gauge)
  mod.rs           — scenario_to_json, scenario_with_edges_json
  scenarios/
    mod.rs         — scaffold, node, edge, channel helpers + full_pipeline_scenario
    ecology.rs     — Shannon, Simpson, Chao1, Pielou, rarefaction, Bray-Curtis heatmap
    ordination.rs  — PCoA scatter + eigenvalue scree
    dynamics.rs    — QS biofilm ODE (5 state vars) + bistable switch
    chemistry.rs   — EIC chromatograms, peak detection, KMD scatter
    anderson.rs    — level spacing gauge, W(t) disorder curves
    benchmarks.rs  — Galaxy vs CPU vs GPU three-tier bar charts
```

## Key Design Decisions

- No petalTongue crate dependency — JSON schema + IPC only
- All scenarios built from live barraCuda math, zero mocks
- Feature-gated on `json` (serde + serde_json)
- Follows healthSpring reference pattern (DataChannel, IPC push, scenario builders)
- Capability-based socket discovery (PETALTONGUE_SOCKET → XDG → /tmp)

## Chain Position

CPU v25 (46/46) → GPU v14 (27/27) → metalForge v17 (29/29) → **Visualization V1 (45/45)**
