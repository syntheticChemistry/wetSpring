# fossilRecord: guidestone_prokaryotic_may2026

## What

Snapshot of the pre-extinction `wetspring_guidestone` binary — the prokaryotic
(single-file) certification binary that validated wetSpring composition
correctness before the eukaryotic UniBin evolution.

## When

- **Fossilized**: 2026-05-09
- **Last known state**: 38/38 NUCLEUS checks (4 skip), Level 4 readiness
- **primalSpring version**: v0.9.25

## Why

primalSpring v0.9.25 introduced the Interstadial Primordial Extinction wave,
requiring all springs to evolve from prokaryotic (one binary per concern) to
eukaryotic (single UniBin with absorbed certification and scenarios).

## What Supersedes

- `wetspring_unibin certify` — layered certification (L0–L6)
- `wetspring_unibin validate` — two-tier scenario validation
- `certification/` library module — reusable certification layers
- `validation/scenarios/` — registered ScenarioMeta scenarios

## Contents

- `src/wetspring_guidestone.rs` — original single-file guidestone binary

## Provenance

- Binary: `wetspring_guidestone`
- Cargo.toml entry: `[[bin]] name = "wetspring_guidestone"` (retained for backward compatibility)
- Required features: `guidestone`
- Upstream: primalSpring v0.9.17 → v0.9.25
