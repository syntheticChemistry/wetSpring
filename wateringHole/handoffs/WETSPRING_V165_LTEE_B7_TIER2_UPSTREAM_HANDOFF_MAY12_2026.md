<!-- SPDX-License-Identifier: CC-BY-SA-4.0 -->

# wetSpring V165 — LTEE B7 Tier 2 + Audit Response

**From:** wetSpring V165
**To:** primalSpring (audit response), lithoSpore, projectNUCLEUS, all spring teams
**Date:** May 12, 2026
**Context:** Response to "River Delta Evolution — All 7 Springs" audit

---

## Audit Items — Resolution Status

The May 12 audit listed four priorities for wetSpring. Three were already
completed in V164/V164b. This increment completes the fourth.

| Audit Priority | Audit Status | Actual Status |
|---------------|-------------|---------------|
| LTEE B7 → lithoSpore module 6 | "STARTED (Exp380)" | **TIER 2 COMPLETE** — Python baseline (V164) + Rust binary (V165, 27/27 PASS) |
| Close PG-02 through PG-05 | "4 open — push toward resolution" | **4 open — all blocked on upstream primals** (documented below) |
| Thread 4 expression + targets | "Sources exist, expression missing" | **DONE** (V164): `ENVIRONMENTAL_GENOMICS.md` + 40 targets wired |
| gS L5 push | "After PGs close" | **Preparing** — expanded certification coverage |
| `--format json` flag | "Add structured JSON output" | **DONE** (V164): `OutputFormat` enum on `certify/validate/status` |

---

## LTEE B7 Tier 2 — Complete (V165)

`validate_ltee_b7_v1.rs` — 27/27 checks PASS:

| Section | Checks | Result |
|---------|--------|--------|
| Population structure | 3 | 12 populations, 264 genomes, 4,629,812 bp genome |
| Mutation rate | 3 | 8.9×10⁻¹¹ /bp/gen, genome-wide rate, 100× mutator |
| Accumulation curve | 10 | 9 time points + 50K aggregate (R² = 0.999985) |
| Mutation spectrum | 9 | 6-class + Ts:Tv + G:C→A:T + sum |
| Model validation | 2 | R² and slope vs rate |

**Artifacts for lithoSpore:**
- `experiments/results/ltee_b7_expected_values.json` — 10 targets + curve
- `barracuda/src/bin/validate_ltee_b7_v1.rs` — Tier 2 Rust binary
- `notebooks/papers/tenaillon-ltee-mutation.ipynb` — Tier 1 Python baseline

**lithoSpore module 6 (`ltee-genomics`)** can now BLAKE3-hash and ingest both
Tier 1 (Python) and Tier 2 (Rust) validation paths.

---

## PG-02 through PG-05 — Readiness Report

All four gaps are externally owned. wetSpring's integration surface is
wired and tested — we are ready to consume when upstream ships.

### PG-02: Provenance Trio (rhizoCrypt + loamSpine + sweetGrass)

**wetSpring readiness:**
- `facade/provenance.rs` — provenance query handler wired
- `/api/v1/provenance/{result_id}` — REST endpoint live
- `/api/v1/validation/chain/{paper_id}` — validation chain endpoint live
- `ipc/handlers/provenance.rs` — JSON-RPC handler for trio calls
- Deploy graphs include trio nodes with `check_skip` behavior

**Blocked on:** trio endpoints going live (rhizoCrypt DAG, loamSpine
attestation, sweetGrass braid). When live, wetSpring can pipe science
results through the full provenance chain.

### PG-03: Songbird Capability Discovery

**wetSpring readiness:**
- 42 niche capabilities registered at startup via `method.register`
- `ipc/songbird.rs` — discovery module wired
- `FAMILY_ID` / `BIOMEOS_FAMILY_ID` env for multi-instance socket resolution

**Blocked on:** `capability.resolve` method not shipped by Songbird.
wetSpring falls back to hardcoded socket paths with family_id.

### PG-04: NestGate Content Pipeline

**wetSpring readiness:**
- `ncbi/nestgate/mod.rs` — content fetch module with NestGate transport
- `ncbi/nestgate/discovery.rs` — socket discovery with family_id
- `ncbi/efetch.rs` — sovereign NCBI pipeline (direct HTTP fallback)
- Exp380 B7 pipeline designed for NestGate-cached genome fetching

**Blocked on:** NestGate content pipeline (`content.put`, `content.get`)
going live for data caching. Pass 12 Songbird VPS relay unblocks NestGate
extracellular content. wetSpring falls back to direct NCBI HTTP.

### PG-05: toadStool Sovereign Dispatch

**wetSpring readiness:**
- `toadstool.list_workloads` already wired (confirmed working)
- 12 workload TOMLs in `projectNUCLEUS/workloads/wetspring/`
- `--format json` on all validation subcommands for Tier 2 ingestion
- `compute.dispatch.submit` IPC path wired

**Blocked on:** `toadstool.validate` (Pass 14). When shipped, wetSpring's
Tier 2 convergence is automatic — binary + JSON output + workload TOMLs
are all in place.

---

## For lithoSpore Team (Spinning Up)

You need these artifacts from wetSpring for module 6 (`ltee-genomics`):

1. **Expected values:** `experiments/results/ltee_b7_expected_values.json`
   - 10 validation targets with tolerances
   - Mutation accumulation curve (9 time points)
   - Provenance metadata

2. **Rust binary:** `cargo run --release --bin validate_ltee_b7_v1`
   - 27/27 checks, exit 0 = PASS
   - Add `--format json` when toadStool Tier 2 lands

3. **Python notebook:** `notebooks/papers/tenaillon-ltee-mutation.ipynb`
   - Tier 1 baseline with full derivation

4. **Foundation targets:** `data/targets/thread04_enviro_targets.toml`
   - 4 LTEE B7 targets (`validated = false` → update when you confirm)

---

## For All Spring Teams

### Pattern: `--format json` Without `unsafe`

wetSpring solved `--format json` without `std::env::set_var` (which is
unsafe in Rust 2024). Pattern:

```rust
#[derive(clap::ValueEnum, Clone, Debug)]
enum OutputFormat { Text, Json }
// Pass to handler → call result.to_json() when Json
```

### Pattern: Tier 2 LTEE Binary

Each LTEE reproduction follows:
1. Constants from paper (population structure, rates, spectrum)
2. Accumulation model (linear, power-law, etc.)
3. R² goodness-of-fit validation
4. Slope vs theoretical rate comparison
5. Provenance section with paper/BioProject/baseline references

groundSpring's B1-B3 and hotSpring's B2 follow similar structures.
