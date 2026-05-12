<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# wetSpring V160 — Tier 4 Handler-Level primal-proof Complete

**Date:** May 11, 2026
**Version:** V160
**From:** wetSpring
**To:** primalSpring, barraCuda, projectNUCLEUS, foundation teams

---

## Summary

V160 resolves PG-09 (barraCuda IPC evaporation). All 5 handlers that call
`barracuda::*` directly are now `primal-proof` wired with IPC-first routing
and graceful in-process fallback. The Tier 4 Cargo structure is confirmed:
`barracuda` is `optional = true` with `barracuda-lib` feature gate (default on),
matching the ludoSpring exemplar pattern.

**Metrics:** 1,962 lib tests, 97 integration, 0 failed. 7 primal gaps open,
15 resolved/closed. Zero clippy warnings. Zero unsafe code.

**plasmidBin binary:** 1.4M stripped release (`infra/plasmidBin/springs/wetspring`).

---

## Handler-Level primal-proof Wiring (Complete)

All handlers that call `barracuda::*` are now covered:

| Handler | File | IPC Method | Version |
|---------|------|------------|---------|
| `handle_diversity` | `science.rs` | `stats.diversity` | V159 |
| `handle_anderson` | `science.rs` | `spectral.anderson_3d` | V159 |
| `handle_qs_model` | `science.rs` | `compute.ode_rk4` | V159 |
| `handle_nmf` | `drug.rs` | `linalg.nmf` | V160 |
| `handle_dose_response` | `gonzales.rs` | `stats.hill_sweep` | V160 |

Handlers NOT wired (confirmed no `barracuda::` calls — use only `crate::bio::*`):
`handle_kinetics`, `handle_alignment`, `handle_taxonomy`, `handle_phylogenetics`,
`handle_brain_observe`, `handle_brain_attention`, `handle_brain_urgency`,
`handle_pk_decay`, `handle_tissue_lattice`, `handle_hormesis`, `handle_biome_atlas`,
`handle_disorder_sweep`, `handle_cross_species`, `handle_ai_ecology_interpret`.

---

## Tier 4 Cargo Structure

```toml
# Already in place since V152:
barracuda = { path = "...", optional = true }

[features]
default = ["barracuda-lib"]
barracuda-lib = ["dep:barracuda"]
primal-proof = ["ipc"]
```

- `cargo build --features ipc` → in-process library (default, `barracuda-lib` on)
- `cargo build --features ipc,primal-proof --no-default-features` → IPC-only sovereign
- UniBin builds and validates without barraCuda source when `barracuda-lib` is off
  and `primal-proof` routes all compute through IPC

---

## Gap Status (7 open, 15 resolved/closed)

**Resolved this wave:**
- PG-09 (barraCuda IPC evaporation) — 5/5 handlers wired

**Remaining wetSpring-internal (1):**
- PG-12 (Exp403 legacy surface — v0.9.17 migration, low priority)

**External teams (5):**
- PG-02 (Provenance trio IPC readiness)
- PG-04 (NestGate live deployment)
- PG-05 (toadStool sovereign dispatch)
- PG-06 (Ionic bond protocol spec)
- PG-18 (Trio UDS connection reset)

**Mixed (2):**
- PG-03 (Songbird capability.resolve — wetSpring abstraction wired, waiting Songbird)
- PG-10 (primalSpring spectral/linalg routing in method_to_capability_domain)

**barraCuda (1):**
- PG-17 (tensor.matmul handle-based API — inline data path or handle-aware parity)

---

## Upstream Team Guidance

### barraCuda
- wetSpring's `barracuda_route.rs` is the reference IPC routing module for
  handler-level dispatch. Other springs can replicate this pattern.
- PG-17 remains: `tensor.matmul` requires pre-created handles. Consider adding
  an inline-data convenience path for the `primal-proof` use case.
- New IPC method needed: `stats.hill_sweep` (batch Hill equation evaluation
  across dose ranges). Currently `handle_dose_response` calls `barracuda::stats::hill`
  per-point in a loop — the IPC method should accept a batch for efficiency.

### primalSpring
- PG-10: `method_to_capability_domain()` should add `"spectral" | "linalg"` to
  the `"tensor"` match arm. wetSpring works around this, but it's a correctness
  issue for any spring using the helper.
- wetSpring V160 can be cited as Tier 4 handler-level wiring complete. The
  `primal-proof` feature flag pattern is validated.

### projectNUCLEUS
- plasmidBin binary ready: 1.4M stripped, `infra/plasmidBin/springs/wetspring`.
- 11 workload TOMLs are gate-agnostic and ready for dispatch.
- When transitioning from `$SPRINGS_ROOT` source builds to fetched binaries,
  the `wetspring` binary is the IPC server entry point.

### foundation
- Thread 04 (environmental genomics) targets remain the primary seeding
  candidates: 63/63 paper reproductions, cold seep metagenomics, Fajgenbaum
  pathway scoring, real NCBI 16S pipelines.
- Foundation seeding work is next priority after this Tier 4 closure.

---

*This document is maintained by wetSpring and fed back to primalSpring via
the wateringHole handoff protocol.*
