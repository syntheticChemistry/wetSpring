# Exp312 — Cross-Spring Provenance Rewire & Validation

| Field    | Value |
|----------|-------|
| Date     | Mar 7, 2026 |
| Phase    | V97e |
| Status   | PASS (31/31 provenance checks, 1346 unit/doc tests, 0 failures) |
| Command  | `cargo run --features gpu --bin validate_cross_spring_provenance` |

## Purpose

Complete the rewiring of wetSpring to modern barraCuda/toadStool/coralReef APIs,
validate cross-spring shader evolution, and benchmark the modern systems.

## Rewiring Summary

### Builder Pattern Migration (Breaking API Changes)

| Dispatch | Old API | New API |
|----------|---------|---------|
| HMM Forward | 10 positional args | `HmmForwardArgs` struct |
| DADA2 E-step | 9 positional args | `Dada2DispatchArgs` (dimensions + buffers) |
| Gillespie SSA | 7 positional args | `GillespieModel` struct + 4 args |
| RK45 Adaptive | N/A (CPU only) | `Rk45DispatchArgs` (future GPU promotion) |

### Precision Routing

- Replaced coarse `Fp64Strategy` match with fine-grained `PrecisionRoutingAdvice`
- `GpuF64::precision_routing()` now available for shared-memory f64 safety
- `optimal_precision()` routes through `PrecisionRoutingAdvice` variants:
  - `F64Native` / `F64NativeNoSharedMem` → `Precision::F64`
  - `Df64Only` → `Precision::Df64`
  - `F32Only` → `Precision::F32`
- Handles new `Fp64Strategy::Sovereign` variant

### Provenance API

- New `wetspring_barracuda::provenance` module (GPU-gated)
- `shaders_authored()` — 5 wetSpring-originated shaders
- `shaders_consumed()` — 17 shaders consumed from all springs
- `shaders_from_other_springs()` — 12 shaders from hotSpring/neuralSpring/airSpring/groundSpring
- `cross_spring_report()` — full evolution timeline from barraCuda registry
- `wetspring_provenance_summary()` — wetSpring-focused summary

### Error Handling

- Fixed 8 pre-existing `unused_must_use` warnings (`.submit()` results now propagated)
- Fixed 4 `redundant_closure` lints in `validate_barracuda_gpu_v12.rs`

## Cross-Spring Evolution Validated

| Flow | Count | Examples |
|------|-------|---------|
| hotSpring → wetSpring | 5 | DF64 core, transcendentals, stress virial, Verlet, ESN readout |
| wetSpring → neuralSpring | 3 | Smith-Waterman, Gillespie SSA, HMM forward |
| neuralSpring → wetSpring | 2 | KL divergence, chi-squared |
| airSpring → wetSpring | 3 | Hargreaves ET₀, seasonal pipeline, moving window |
| groundSpring → wetSpring | 2 | Welford mean+variance, chi-squared CDF |

**Total**: 28 shaders in registry, 22 cross-spring, 17 consumed by wetSpring.

## Regression Results

| Check | Status |
|-------|--------|
| `cargo fmt` | PASS |
| `cargo clippy -D warnings` (default) | PASS |
| `cargo clippy -D warnings --features gpu` | PASS |
| `cargo doc -D warnings` (barracuda) | PASS |
| `cargo doc -D warnings` (forge) | PASS |
| Unit + doc tests | 1,346 pass, 0 fail |
| Provenance binary (31 checks) | 31 pass, 0 fail |

## Files Changed

- `barracuda/src/bio/hmm_gpu.rs` — HmmForwardArgs builder pattern
- `barracuda/src/bio/dada2_gpu.rs` — Dada2DispatchArgs builder pattern
- `barracuda/src/gpu.rs` — PrecisionRoutingAdvice, Fp64Strategy::Sovereign
- `barracuda/src/provenance.rs` — new provenance module
- `barracuda/src/lib.rs` — register provenance module
- `barracuda/src/bin/validate_toadstool_bio.rs` — GillespieModel
- `barracuda/src/bin/validate_cpu_vs_gpu_all_domains.rs` — GillespieModel
- `barracuda/src/bin/validate_metalforge_full_v2.rs` — GillespieModel
- `barracuda/src/bin/validate_metalforge_full_v3.rs` — GillespieModel
- `barracuda/src/bin/validate_cross_spring_provenance.rs` — new Exp312 binary
- `barracuda/src/bin/validate_barracuda_gpu_v12.rs` — redundant closure fix
- `barracuda/src/bio/bistable_gpu.rs` — error propagation
- `barracuda/src/bio/capacitor_gpu.rs` — error propagation
- `barracuda/src/bio/cooperation_gpu.rs` — error propagation
- `barracuda/src/bio/gemm_cached.rs` — error propagation (2 sites)
- `barracuda/src/bio/multi_signal_gpu.rs` — error propagation
- `barracuda/src/bio/pairwise_l2_gpu.rs` — error propagation
- `barracuda/src/bio/phage_defense_gpu.rs` — error propagation
- `barracuda/Cargo.toml` — register provenance binary
