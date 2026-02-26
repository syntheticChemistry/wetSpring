# wetSpring → ToadStool V50 ODE Derivative Rewire Handoff

**Date:** February 26, 2026
**ToadStool pin:** `17932267` (S65)
**wetSpring phase:** 50 (V50)
**Status:** All 5 absorbed ODE systems now delegate derivative computation to barracuda

---

## Summary

V50 completes the ODE derivative lean: wetSpring's 5 biological ODE systems now
use `barracuda::numerical::ode_bio::*Ode::cpu_derivative` instead of local RHS
functions. This eliminates ~200 lines of duplicate derivative math (hill
functions, Monod kinetics, full RHS bodies) while preserving wetSpring's
integration framework (trajectory storage, clamping, steady-state analysis).

## Changes

### ODE derivative rewire (5 systems)

| System | File | Local removed | barracuda delegate |
|--------|------|--------------|-------------------|
| Capacitor | `bio/capacitor.rs` | `hill()`, `capacitor_rhs()` | `CapacitorOde::cpu_derivative` |
| Cooperation | `bio/cooperation.rs` | `hill()`, `coop_rhs()` | `CooperationOde::cpu_derivative` |
| Multi-Signal | `bio/multi_signal.rs` | `hill()`, `hill_repress()`, `multi_rhs()` | `MultiSignalOde::cpu_derivative` + cdg guard |
| Bistable | `bio/bistable.rs` | `hill()`, `bistable_rhs()` | `BistableOde::cpu_derivative` + cdg guard |
| Phage Defense | `bio/phage_defense.rs` | `monod()`, `defense_rhs()` | `PhageDefenseOde::cpu_derivative` |

### c-di-GMP convergence guard

Bistable and multi-signal systems retain a thin wrapper that prevents the
c-di-GMP derivative from going negative when c-di-GMP is near zero
(`< ODE_CDG_CONVERGENCE`). This is a fixed-step RK4 stability refinement
not present in barracuda's derivative (which is designed for both adaptive
RK45 and GPU batched integration where clamping handles it).

### Retained locally

- `rk4_integrate` / `OdeResult` / `steady_state_mean` — trajectory storage
  and analysis (barracuda's `BatchedOdeRK4::integrate_cpu` only returns
  final states; `rk45_solve` uses adaptive stepping)
- Per-system clamping bounds (`CLAMP` arrays)
- Param structs with ergonomic named fields, `Default` impls, domain docs
- `QsBiofilm` base model (monostable variant, not absorbed into barracuda)
- Scenario functions and simulation wrappers

### Additional improvements

- `http.rs`: `interpret_output` takes ownership (eliminates `stdout.clone()`)
- `http.rs`: `which_exists` rewritten as pure Rust PATH scan (no subprocess)
- `ncbi_data/vibrio.rs`: 4 new `try_load_json_array` error-path tests

## Verification

- 823 barracuda lib tests pass (0 failures, 1 ignored)
- 47 forge lib tests pass
- `cargo fmt --check` — clean
- `cargo clippy --lib -D pedantic -D nursery -D warnings` — clean
- `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --lib` — clean
- All ODE systems produce numerically identical results to pre-rewire

## What remains local (by design)

| Component | Reason |
|-----------|--------|
| `ode.rs` (RK4 integrator) | Trajectory + clamping not in barracuda's batched API |
| `qs_biofilm.rs` (base model) | Not absorbed — monostable variant |
| `tolerances.rs` (77 constants) | Domain-specific, complementary to `barracuda::tolerances` |
| `validation.rs` (Validator) | Simpler API for Python-baseline pattern |
| `ncbi/` (NCBI interaction) | wetSpring-specific data pipeline |

## ToadStool status

No new evolution requests. All 9/9 P0-P3 requests remain DONE.
ToadStool's `ode_bio` module (absorbed from wetSpring v24-v25) is the
canonical source for all 5 derivative implementations.
