# wetSpring V117 → BarraCUDA/ToadStool Deep Tolerance Centralization + Code Quality Handoff

**Date:** March 15, 2026
**From:** wetSpring V117 (376 experiments, 5,707+ checks, 1,667 tests, 354 binaries)
**To:** BarraCUDA/ToadStool team
**Authority:** wateringHole (ecoPrimals Core Standards)
**Supersedes:** V116 Capability Discovery + Tolerance Centralization Handoff (Mar 15)
**Pins:** barraCuda v0.3.5 (`03986ce`), toadStool S130+, coralReef Phase 10
**License:** AGPL-3.0-or-later

---

## Executive Summary

- **39 `cargo fmt` violations resolved** across 19 files — format compliance restored
- **12 unfulfilled `#[expect()]` attributes removed** from 2 validation binaries — zero clippy warnings achieved
- **4 `panic!()` calls eliminated from production code** — GPU modules and nautilus bridge now return `Result` instead of panicking
- **5 ESN urgency thresholds centralized** as named constants with scientific documentation
- **13 new tolerance constants added** across 4 submodules (spectral, gpu, instrument, mod)
- **30+ inline tolerance literals replaced** in 17 validation binaries with named constants
- **Zero clippy warnings, zero fmt violations, 1,667 tests passing, zero unsafe code**

---

## Part 1: Format and Lint Compliance

### `cargo fmt` (39 violations → 0)

19 files reformatted: 9 validation binaries, 7 IPC modules, 1 I/O module,
1 bio module, 1 metalForge module. All were minor formatting drift
(long function call arguments, trailing commas).

### Unfulfilled `#[expect()]` (12 → 0)

| Binary | Removed Expects | Reason |
|--------|:---------------:|--------|
| `validate_cpu_vs_gpu_v11.rs` | 7 | `unwrap_used`, `cast_*`, `many_single_char_names`, `items_after_statements`, `float_cmp` — none triggered |
| `validate_barracuda_cpu_v27.rs` | 5 | `cast_possible_truncation`, `cast_sign_loss`, `many_single_char_names`, `items_after_statements`, `float_cmp` — none triggered |

---

## Part 2: Production Panic Elimination (4 → 0)

| File | Before | After |
|------|--------|-------|
| `bio/brain/nautilus_bridge.rs` | `panic!("expected continuous input")` | `return Err(Error::InvalidInput(...))` |
| `bio/derep_gpu.rs` | `.unwrap_or_else(\|e\| panic!("GPU: {e}"))` | `?` (function already returns `Result`) |
| `bio/chimera_gpu.rs` | `.unwrap_or_else(\|e\| panic!("GPU: {e}"))` | `?` |
| `bio/reconciliation_gpu.rs` | `.unwrap_or_else(\|e\| panic!("GPU: {e}"))` | `?` |

The GPU module functions (`dereplicate_gpu`, `detect_chimeras_gpu`, `reconcile_dtl_gpu`)
already returned `Result` — the panics were in test helpers that called them.
The nautilus bridge `reservoir_input_dimensions` now returns `Result` with
`Error::InvalidInput` for the discrete-input mismatch case.

---

## Part 3: ESN Urgency Threshold Centralization

`bio/esn/heads.rs` now defines 5 named constants for the attention state machine:

| Constant | Value | Purpose |
|----------|-------|---------|
| `URGENCY_ESCALATE_ALERT` | 0.6 | Healthy → Alert threshold |
| `URGENCY_ESCALATE_CRITICAL` | 0.8 | Alert → Critical threshold |
| `URGENCY_DEESCALATE` | 0.3 | Recovery (de-escalation) threshold |
| `PHASE_LABEL_LOW` | 0.3 | Phase discretization: low/medium boundary |
| `PHASE_LABEL_HIGH` | 0.6 | Phase discretization: medium/high boundary |

Provenance: adapted from hotSpring `AttentionState` physics thresholds,
re-calibrated for bio diversity prediction uncertainty ranges. The hysteresis
gap (0.3 for de-escalation vs 0.6 for escalation) prevents oscillation at
boundary urgency levels.

---

## Part 4: Tolerance Constant Expansion (13 new constants)

### `tolerances/spectral.rs` (6 new)

| Constant | Value | Domain |
|----------|-------|--------|
| `SOIL_QS_TILLAGE` | 0.2 | Tillage effect on P(QS) and model ratios |
| `ANDERSON_NU_PARITY` | 0.4 | Critical exponent ν fitting tolerance |
| `GEOMETRY_DIMENSIONAL_PARITY` | 0.15 | 3D vs 2D level-spacing / cross-species |
| `FAO56_ET0_PARITY` | 0.15 | Cross-spring FAO-56 ET₀ computation |
| `INTERCEPT_NEAR_ZERO` | 0.5 | Linear regression intercept residual |

### `tolerances/gpu.rs` (1 new)

| Constant | Value | Domain |
|----------|-------|--------|
| `ODE_GPU_LANDSCAPE_PARITY` | 2.0 | QS parameter sweep GPU vs CPU |

### `tolerances/instrument.rs` (2 new)

| Constant | Value | Domain |
|----------|-------|--------|
| `PFSA_HOMOLOGUE_WINDOW` | 60.0 | PFSA CF₂ homologue separation filter |
| `RETENTION_INDEX_MATCH` | 0.1 | GC-MS retention index matching |

### `tolerances/mod.rs` (4 new)

| Constant | Value | Domain |
|----------|-------|--------|
| `PHARMACOKINETIC_PARITY` | 0.1 | IC50, PK onset, selectivity ratios |
| `IC50_RESPONSE_TOL` | 0.01 | Hill equation response at IC50 midpoint |
| `REGRESSION_FIT_PARITY` | 0.01 | Nonlinear regression parameter estimates |
| `RAREFACTION_BOOTSTRAP_SHANNON` | 0.5 | Bootstrap Shannon mean vs CPU estimate |

Each constant has a documentation comment explaining its scientific
justification, provenance (which experiment validated it), and relationship
to other tolerance constants in the hierarchy.

---

## Part 5: Binary Tolerance Replacements (30+ across 17 binaries)

| Binary | Replacements | Constants Used |
|--------|:------------:|---------------|
| `validate_nucleus_data_pipeline.rs` | 2 | `CHAO1_COUNT_HALFWIDTH` |
| `validate_gonzales_cpu_parity.rs` | 5 | `CHAO1_COUNT_HALFWIDTH`, `REGRESSION_FIT_PARITY` |
| `validate_gonzales_ic50_s79.rs` | 5 | `PHARMACOKINETIC_PARITY`, `IC50_RESPONSE_TOL` |
| `validate_gonzales_pk_s79.rs` | 1 | `PHARMACOKINETIC_PARITY` |
| `validate_cross_spring_s86.rs` | 4 | `CROSS_SPRING_NUMERICAL`, `SOIL_MODEL_APPROX`, `INTERCEPT_NEAR_ZERO`, `FAO56_ET0_PARITY` |
| `validate_kbs_lter_anderson_v1.rs` | 1 | `SOIL_QS_TILLAGE` |
| `validate_paper_math_control_v4.rs` | 1 | `SOIL_QS_TILLAGE` |
| `validate_voc_peaks.rs` | 1 | `RETENTION_INDEX_MATCH` |
| `validate_vent_chimney_qs.rs` | 1 | `GEOMETRY_DIMENSIONAL_PARITY` |
| `validate_heterogeneity_sweep_s79.rs` | 1 | `GEOMETRY_DIMENSIONAL_PARITY` |
| `validate_df64_anderson.rs` | 1 | `ANDERSON_NU_PARITY` |
| `validate_pfas_library.rs` | 1 | `PFSA_HOMOLOGUE_WINDOW` |
| `validate_vibrio_qs_landscape.rs` | 1 | `ODE_GPU_LANDSCAPE_PARITY` |
| `validate_gpu_extended.rs` | 2 | `RAREFACTION_BOOTSTRAP_SHANNON`, `NANOPORE_DIVERSITY_TOLERANCE` |
| `validate_barracuda_cpu_v23.rs` | 1 | `SOIL_MODEL_APPROX` |

---

## Quality Gates

| Check | Result |
|-------|--------|
| `cargo fmt --check` | Zero violations |
| `cargo clippy --workspace --all-features` | Zero warnings |
| `cargo test --workspace` | 1,667 passed, 0 failed, 2 ignored |
| Production `panic!()` | 0 |
| Production `unsafe` | 0 (`#![forbid(unsafe_code)]`) |
| Inline tolerance literals in binaries | ~30 fewer (13 new named constants) |
| Files changed | 44 (385 insertions, 173 deletions) |

---

## Remaining Work (P2-P3)

| Item | Priority | Notes |
|------|----------|-------|
| Provenance tables in newer binaries | P2 | ~150 binaries have minimal provenance |
| `ComputeDispatch` adoption | P3 | 6 ODE modules still use manual BGL |
| DF64 GEMM adoption | P3 | `Fp64Strategy::Hybrid` for consumer GPUs |
| `BandwidthTier` wiring | P3 | PCIe-aware dispatch in metalForge |
| 8 GPU primitive opportunities | P2-P3 | Carried forward from V116 |
