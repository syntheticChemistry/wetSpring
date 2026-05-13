<!-- SPDX-License-Identifier: CC-BY-SA-4.0 -->

# wetSpring V166 — Tier 2 Convergence Wave Response

**From:** wetSpring V166
**To:** primalSpring (upstream audit), all spring teams, projectNUCLEUS, lithoSpore
**Date:** May 13, 2026
**Context:** Response to "Tier 2 Convergence Wave" audit (May 13, 2026)

---

## Audit Compliance — Priority Checklist

| Priority | Audit Requirement | wetSpring Status |
|----------|------------------|-----------------|
| **1** | Wire `toadstool.validate` | **DONE** (V165b) — `ipc/toadstool_validate.rs`, contract aligned V166 |
| **2** | Wire `barracuda.precision.route` | **DONE** (V165b) — `ipc/precision_route.rs` |
| **3** | LTEE handoff for lithoSpore | **DONE** — B7 Tier 2 (27/27 PASS), `expected_values.json` aligned with `module6_breseq.json` |
| **4** | plasmidBin deployment | **VERIFIED** — musl static binary builds and runs standalone |
| **5** | Surface gaps upstream | **This document** |

wetSpring addressed Priorities 1-2 proactively (V165b, May 12) before this audit
was issued. V166 closes the remaining contract gaps and documentation.

---

## What Changed in V166

### Contract Alignment

`ValidateResult` now includes `dry_run` echo field per NUCLEUS spec.
`WorkloadEntry` now includes `last_run` timestamp per primalSpring
`LIVE_SCIENCE_API.md`. Tests updated to validate both fields.

### IPC Mapping Documentation

New `docs/PRIMAL_PROOF_IPC_MAPPING.md` documents:
- 10 domain operations mapped to `barracuda.precision.route` queries
- 19 inbound science IPC methods
- 7 outbound IPC client modules
- Feature gate matrix (none / ipc / ipc+barracuda-lib / primal-proof)
- Fallback behavior when primal sockets unavailable

### plasmidBin Readiness

```
target: x86_64-unknown-linux-musl
features: guidestone
binary: wetspring_unibin
subcommands: version, validate --list, status --format json, certify, serve
```

Binary builds, runs standalone, produces structured JSON output. Ready for
`sources.toml` registration and `auto-harvest.yml` integration.

---

## LTEE B7 Handoff — lithoSpore Module 6

| Artifact | Path | Status |
|----------|------|--------|
| Python baseline | `notebooks/papers/tenaillon-ltee-mutation.ipynb` | Tier 1 complete |
| Expected values | `experiments/results/ltee_b7_expected_values.json` | Aligned with `module6_breseq.json` |
| Rust validator | `barracuda/src/bin/validate_ltee_b7_v1.rs` | 27/27 PASS |
| Experiment doc | `experiments/380_ltee_b7_tenaillon_mutation_accumulation.md` | Tier 2 complete |
| Workload TOML | `projectNUCLEUS/workloads/wetspring/wetspring-ltee-b7-mutation-accumulation.toml` | Ready |

Structural diff between `ltee_b7_expected_values.json` and lithoSpore
`validation/expected/module6_breseq.json`: **zero mismatches**. Same top-level
keys, same target structure, same provenance format.

---

## Remaining Gaps (All External)

| Gap | Blocked On | Notes |
|-----|-----------|-------|
| PG-02 Provenance Trio | rhizoCrypt + loamSpine + sweetGrass endpoints going live | Client stubs ready |
| PG-03 Capability Discovery | Songbird `capability.resolve` (Pass 14) | Discovery module ready |
| PG-04 NestGate Content | NestGate `content.put`/`content.get` going live | Client pattern from neuralSpring available |
| PG-05 toadStool Live | toadStool production socket deployment | Client wired, tested with mock responses |

No internal gaps. No internal debt. Zero clippy warnings (new code). Zero unsafe.

---

## Upstream Observations

### primalSpring LIVE_SCIENCE_API.md vs NUCLEUS spec discrepancy

The canonical `primalSpring/docs/LIVE_SCIENCE_API.md` marks `barracuda.precision.route`
as **NOT IMPLEMENTED**, while `projectNUCLEUS/specs/LIVE_SCIENCE_API.md` marks it
as **IMPLEMENTED (649 tests)**. wetSpring wired against the NUCLEUS spec.
Recommend syncing the canonical doc.

### lithoSpore module 6 — Tier 2 alignment

lithoSpore `UPSTREAM_GAPS.md` reports module 6 (ltee-breseq) as **Tier 2 PASS (8/8)**
sourcing from wetSpring B7. Our `expected_values.json` is structurally aligned.
The "sovereign genomics pipeline" gap (264 NCBI genomes) remains an NCBI data
download task, not a code gap.

### foundation barraCuda CPU parity benchmarks

foundation now has Python baselines for variance, velocity Verlet, and spectral
eigenvalues in `benchmarks/barracuda_cpu_parity/`. These are directly relevant
to wetSpring's precision routing — the tolerance thresholds in our IPC mapping
doc align with foundation's measured ULP and absolute error bounds.

---

## Absorption Patterns for Other Springs

### toadstool.validate client (copy-paste ready)

Reuse `ipc/toadstool_validate.rs` pattern. Key: reuse your existing
`compute_dispatch::discover()` for socket location. Response parsing is
forward-compatible (unknown fields ignored via `serde_json::Value`).

### Precision mapping documentation

Follow the table format in `docs/PRIMAL_PROOF_IPC_MAPPING.md`. Map each of
your domain operations to a `barracuda.precision.route` domain string. Document
your tolerance expectations so barraCuda can optimize shader selection.

### plasmidBin musl build

```bash
cargo build --release --target x86_64-unknown-linux-musl --bin your_unibin --features guidestone
```

Verify with `version`, `validate --list`, `status --format json`.

---

## Current Metrics

| Metric | Value |
|--------|-------|
| Tests | 1,962 lib + 97 integration + 18 IPC |
| Binaries | 367 (345 barracuda + 22 forge) |
| Experiments | 384 indexed |
| Papers | 63/63 + LTEE B7 TIER 2 COMPLETE |
| Coverage | 91.20% line / 90.30% function |
| GuideStone | Level 4 (38/38 pass, 4 skip) |
| Primal gaps | 4 open (all external), 18 resolved/closed |
| Foundation | 10/10 threads active |
| Tier | 2 (structurally complete, contract aligned) |
| Clippy | 0 warnings (new code) |
| Unsafe | 0 (`forbid(unsafe_code)`) |
| plasmidBin | musl binary verified |
