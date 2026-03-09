# wetSpring V101 — Deep Debt Evolution + Absorption Handoff

**Date:** March 9, 2026
**From:** wetSpring
**To:** barraCuda, toadStool, coralReef
**License:** AGPL-3.0-or-later
**Covers:** V101 deep debt resolution, tolerance evolution, idiomatic Rust modernization

---

## Executive Summary

- All bare float tolerance literals in validation binaries promoted to named `tolerances::` constants
- 179 named tolerance constants (was 164), all scientifically documented with provenance
- 3 validation binaries refactored: long `main` functions decomposed into domain-focused helpers
- All `cargo clippy -W pedantic` warnings resolved (was 8, now 0)
- `ABSORPTION_MANIFEST.md` corrected: `reconciliation_gpu` uses `FusedMapReduceF64` (not `TreeInferenceGpu`)
- Total: **1,455 tests, 316 binaries, 9,060+ checks, 334 experiments — ALL PASS**

---

## Part 1: Tolerance Evolution

### New Named Constants

| Constant | Value | Module | Justification |
|----------|-------|--------|---------------|
| `SSA_SINGLE_RUN_ABSOLUTE` | 30.0 | `tolerances::bio::phylogeny` | Gillespie SSA single-realization variance for mean ~50 reactions; `sqrt(50) ≈ 7.07`, 30.0 = ~4σ |
| `BOOTSTRAP_ESTIMATE_SMALL` | 0.5 | `tolerances::bio::phylogeny` | Bootstrap support estimates at 100 replicates with small trees; ±0.5 support units |

### Bare Literal Migration

All remaining bare float literals in validation binaries were replaced with named constants:

| Binary | Old | New | Count |
|--------|-----|-----|:-----:|
| `validate_barracuda_cpu_v22` | `30.0` | `tolerances::SSA_SINGLE_RUN_ABSOLUTE` | 1 |
| `validate_barracuda_cpu_v22` | `0.5` | `tolerances::BOOTSTRAP_ESTIMATE_SMALL` | 1 |
| `validate_cross_spring_evolution_modern` | `0.01` | `tolerances::RIDGE_TEST_TOL` | 2 |
| `validate_bio_brain_s79` | `0.0` | `tolerances::EXACT` | 6 |
| `validate_cpu_gpu_viz_math` | `0.0` | `tolerances::EXACT` | 1 |

**Result:** Zero inline tolerance literals remain in the codebase.

---

## Part 2: Clippy Pedantic Cleanup

### `too_many_lines` — Refactored 3 Binaries

| Binary | Before | After | Strategy |
|--------|:------:|:-----:|----------|
| `validate_cpu_gpu_viz_math` | 144-line `main` | 4 helpers: `validate_alpha_diversity`, `validate_beta_and_ordination`, `validate_kmd`, `validate_ode_determinism` | Domain-coherent extraction |
| `dump_wetspring_scenarios` | 267-line `main` + `build_scenarios` | `CheckTracker` struct, `build_core_scenarios`, `build_analysis_scenarios`, `output_scenarios`, `run_stream_demo` | Responsibility separation |
| `validate_visualization_v1` | 287-line `main` | `validate_schema`, `validate_scenario_builders`, `validate_ipc`, `validate_full_chain` | Validation phase separation |

### `similar_names` — Intentional CPU/GPU Pairs

`validate_cpu_gpu_viz_math` uses `_cpu`/`_gpu` suffix bindings (e.g., `h_cpu`/`h_gpu`) by design for
CPU vs GPU comparison. Resolved with `#![allow(clippy::similar_names)]` at crate level.

### `doc_markdown` — Backtick Fixes

- `validate_biomeos_petaltongue_full`: `DataChannel JSON` → `` `DataChannel` JSON ``
- New tolerance constants: backticked `sqrt()` and code references

---

## Part 3: Absorption Manifest Correction

`reconciliation_gpu` was incorrectly mapped to `TreeInferenceGpu` (does not exist in barraCuda).
Corrected to `FusedMapReduceF64` — the actual primitive used for batch reconciliation cost
computation. Added note: full `BatchReconcileGpu` is a candidate for upstream absorption.

---

## Part 4: Current Primitive Inventory

wetSpring consumes **150+ primitives** from standalone barraCuda v0.3.3 `a898dee`:

| Category | Count | Examples |
|----------|:-----:|---------|
| GPU compute | 30 | `FusedMapReduceF64`, `BrayCurtisF64`, `BatchedEighGpu`, `BatchedOdeRK4F64` |
| GPU ODE traits | 5 | `generate_shader()` for phage, bistable, multi-signal, cooperation, capacitor |
| Stats | 15 | `shannon`, `simpson`, `chao1`, `pielou`, `hill`, `norm_cdf`, `norm_ppf`, `mean`, `percentile` |
| Linalg | 8 | `gemm_f64`, `matmul`, `outer_product`, `determinant` |
| Special | 6 | `erf`, `erfc`, `ln_gamma`, `gamma_inc`, `normal_cdf`, `dot` |
| Numerical | 5 | `integrate_cpu`, `rk45_integrate`, `gradient_1d`, `find_roots_bisection` |
| Spectral | 4 | `anderson_3d`, `lanczos`, `lanczos_eigenvalues`, `disordered_laplacian` |
| Tolerances | 1 | `tolerances` module pattern (adopted from hotSpring) |
| Dispatch | 1 | `ComputeDispatch` (264 ops across GPU pipeline) |

### Absorption Targets for barraCuda Team

| Target | Priority | Rationale |
|--------|----------|-----------|
| `BatchReconcileGpu` | P2 | DTL reconciliation currently uses `FusedMapReduceF64` as workaround |
| `viz::DataChannel` generics | P3 | petalTongue types could be shared across springs |
| Tolerance hierarchy pattern | P3 | Other springs could adopt the `tolerances/` module structure |

---

## Part 5: Code Quality Snapshot

| Metric | Value |
|--------|-------|
| `cargo fmt --check` | Clean (0 diffs) |
| `cargo clippy -W pedantic` | **0 warnings** |
| `cargo doc --no-deps` | **0 warnings** |
| Tests | 1,455 (1,047 lib + 203 forge + 27 doc + 178 integration) |
| Coverage | 93.94% line / 91.85% fn / 93.60% branch |
| Named tolerances | 179 (scientifically justified, hierarchy-tested) |
| Binaries | 316 (294 barracuda + 22 forge) |
| Inline tolerance literals | 0 |
| `unsafe` code | 0 (enforced crate-wide, `allow` only in test env-var manipulation) |
| TODO/FIXME | 0 |
| External C deps | 0 (`flate2` uses `rust_backend`) |
| Max file size | All under 1000 LOC |
| Local WGSL | 0 (fully lean) |

---

## Part 6: Cross-Spring Learnings for barraCuda

### Tolerance Module Pattern

wetSpring's `tolerances/` hierarchy (179 constants across `bio/`, `gpu/`, `spectral/`, `instrument/`)
has proven effective for eliminating magic numbers. Each constant is `pub const` with a doc comment
explaining the physical or statistical basis, applicable domain, and provenance (which experiment
or paper established it). This pattern scales — hotSpring should consider adopting it for plasma
physics tolerances, and barraCuda itself could provide a tolerance module template.

### Validation Binary Pattern

The `hotSpring` pattern (explicit pass/fail, exit code 0/1, provenance header) continues to work
well at scale. At 316 binaries, the pattern has not shown strain. Key discipline: every bare
literal gets a named constant, every constant gets a documentation comment, every binary gets
a provenance table.

### petalTongue Integration

The visualization module (`barracuda/src/visualization/`) demonstrates how science→visualization
can be wired through IPC without coupling. The `DataChannel` types (Timeseries, Heatmap, Bar,
Scatter, Distribution, Gauge, Spectrum) are generic enough for cross-spring reuse. StreamSession
lifecycle management (initial→append→update→replace→close) handles progressive rendering.

---

## Verification

```bash
cargo test --workspace --no-fail-fast     # 1,455 passed, 0 failed
cargo fmt --all -- --check                # 0 diffs
cargo clippy --workspace --all-targets -- -W clippy::pedantic  # 0 warnings
cargo doc --workspace --no-deps           # 0 warnings
```
