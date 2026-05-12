# wetSpring V164 — Downstream Seeding Sprint Handoff

**Date**: May 12, 2026
**From**: wetSpring V164
**To**: primalSpring coordination, lithoSpore, foundation, projectNUCLEUS

---

## Summary

wetSpring completes all three audit targets from the Downstream Seeding
Sprint (May 12, 2026):

1. **Foundation Thread 4 expression + targets** — elevates Thread 4 from
   "seeded" to "active"
2. **LTEE B7 Tier 1 complete** — expected values JSON ready for lithoSpore
   module 6
3. **projectNUCLEUS workload TOML** — Exp380 B7 pipeline wired for dispatch
4. **--format json** — UniBin CLI supports structured output for Tier 2
   ingestion

---

## Deliverables

### For foundation

| Artifact | Path | Status |
|----------|------|--------|
| Thread 4 expression | `gardens/foundation/expressions/ENVIRONMENTAL_GENOMICS.md` | **NEW** — active |
| THREAD_INDEX.toml | Thread 4 `expression` + `status = "active"` | **UPDATED** |
| Data sources | `data/sources/thread04_enviro.toml` (23 sources) | Pre-existing |
| Data targets | `data/targets/thread04_enviro_targets.toml` (36 targets) | Pre-existing |

The expression covers:
- 3 external paper lineages (Anderson 6, Waters 7, Liu 6 papers)
- 7 baseCamp papers (01, 03, 04, 05, 06, 09, 16)
- 8 jelly strings with NUCLEUS structural solutions
- 4 composition blueprints (sovereign 16S, PFAS sentinel, LTEE B7, NPU field)
- Spring alignment table (6 springs contributing)
- petalTongue vision (4 dashboard concepts)
- scyBorg publication chain

**Thread status**: 8/10 threads now have expressions (up from 7/10).

### For lithoSpore

| Artifact | Path | Status |
|----------|------|--------|
| Expected values JSON | `experiments/results/ltee_b7_expected_values.json` | **NEW** |
| Python baseline | `notebooks/papers/tenaillon-ltee-mutation.ipynb` | **NEW** |

10 validation targets for module 6 (breseq comparison):
- `n_populations`: 12 (exact)
- `n_genomes`: 264 (exact)
- `genome_length_bp`: 4,629,812 bp (±100)
- `nonmutator_rate_per_bp_per_gen`: 8.9×10⁻¹¹ (±1.0×10⁻¹¹)
- `nonmutator_mutations_at_50k`: 20.6 (±2.3)
- `ts_tv_ratio`: 1.7 (±0.3)
- `gc_to_at_fraction`: 0.68 (±0.05)
- `mutator_rate_multiplier`: 100× (±50)
- `mutation_spectrum`: 6-class distribution
- `accumulation_model`: near-linear

Mutation accumulation curve (9 time points, 0–50K generations) included.

### For projectNUCLEUS

| Artifact | Path | Status |
|----------|------|--------|
| Workload TOML | `workloads/wetspring/wetspring-ltee-b7-mutation-accumulation.toml` | **NEW** |
| --format json | `wetspring_unibin validate --format json` | **NEW** |
| --format json | `wetspring_unibin certify --format json` | **NEW** |
| --format json | `wetspring_unibin status --format json` | **NEW** |

The workload TOML targets `validate_ltee_b7_mutation_accumulation` (Tier 2
binary, not yet built). When the Rust validation binary ships, NUCLEUS can
dispatch the full B7 pipeline.

`--format json` enables structured output for all three primary UniBin
subcommands. `validate` leverages primalSpring's `ValidationResult::to_json()`.
`status` outputs a JSON object with version, domain, guidestone level,
scenario counts. `validate --list --format json` outputs the full scenario
registry as a JSON array.

---

## Remaining Work (wetSpring)

| Item | Status | Blocked On |
|------|--------|-----------|
| B7 Tier 2: Rust validation binary | Pending | Needs genome download infrastructure |
| B7 Tier 3: lithoSpore integration | Pending | Tier 2 + lithoSpore module 6 scaffold |
| PG-02 (provenance trio) | Blocked | rhizoCrypt/loamSpine/sweetGrass not live |
| PG-03 (Songbird canonical names) | Blocked | Songbird method discovery not live |
| PG-04 (NestGate content pipeline) | Blocked | NestGate not live |
| PG-05 (toadStool workload dispatch) | Blocked | toadStool dispatch not live |
| GuideStone L5 | Blocked | 4 skips require live infrastructure |

---

## Patterns for Absorption

### --format json Pattern

Other springs adding `--format json` to validation binaries:

```rust
// In your CLI definition (clap derive):
#[derive(Clone, Copy, Default, clap::ValueEnum)]
pub enum OutputFormat {
    #[default]
    Text,
    Json,
}

// In your validate subcommand:
#[arg(long, value_enum, default_value_t = OutputFormat::Text)]
format: OutputFormat,

// In your validate handler:
if matches!(format, OutputFormat::Json) {
    if let Ok(json) = result.to_json() {
        println!("{json}");
    }
} else {
    result.finish();  // human-readable
}
```

primalSpring's `ValidationResult::to_json()` handles the serialization.
No unsafe code needed (no env var setting).

### Foundation Expression Pattern

Follow `specs/EXPRESSION_AUTHORING_GUIDE.md`. Required sections:
1. Header (thread number, cross-threads, license)
2. Framing (why this expression exists)
3. Paper lineage (per-paper: citation, organism, key results)
4. Jelly strings (provenance gaps → NUCLEUS solutions)
5. Data targets (pointer to TOML manifests)
6. NUCLEUS composition blueprints (deploy graph architecture)
7. Spring alignment (who contributes what)
8. petalTongue vision (how science becomes live surfaces)
9. scyBorg publication (how results get published)

Then update `THREAD_INDEX.toml` with the expression path and status.
