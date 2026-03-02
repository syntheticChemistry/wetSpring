# Exp290: biomeOS Graph v2 — Tower/Node/Nest + Sovereign Pipeline

**Status:** PASS (27/27 checks)
**Date:** 2026-03-02
**Binary:** `validate_biomeos_graph_v2`
**Command:** `cargo run -p wetspring-forge --bin validate_biomeos_graph_v2`
**Feature gate:** none

## Purpose

Validates the full biomeOS coordination layer at V92D state: socket discovery,
primal orchestration, vault-aware routing, and sovereign fallback. Proves that
NUCLEUS atomics (Tower, Node, Nest) correctly orchestrate mixed hardware
dispatch while maintaining provenance and determinism guarantees.

## Sections (S1–S7)

| Section | Checks | Description |
|---------|--------|-------------|
| S1 | 5 | Tower graph — substrate discovery, capability matching |
| S2 | 3 | Node dispatch graph — workload→substrate routing DAG |
| S3 | 3 | Nest coordination — storage protocol, sovereign fallback |
| S4 | 8 | Pipeline topologies — GPU-only, mixed NPU→GPU→CPU, CPU-only |
| S5 | 4 | Absorption evolution — origin tracking, lean verification |
| S6 | 3 | Sovereign mode — zero external dependencies |
| S7 | 3 | Determinism — rerun-identical routing, stable origin summary |

## biomeOS Coordination

| Component | Role | Validated |
|-----------|------|-----------|
| **Songbird** | Tower mesh discovery | S1 (conditional) |
| **NestGate** | Distributed storage | S3 (sovereign fallback) |
| **Tower** | Substrate graph | S1, S2 |
| **Node** | Compute dispatch | S2, S6 |
| **Nest** | Storage protocol | S3, S6 |

## Chain

metalForge v13 (Exp289) → **biomeOS Graph v2 (this)** → ToadStool absorption
