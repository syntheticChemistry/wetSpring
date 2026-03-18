# wetSpring V129 Handoff — Deep Debt Evolution & Absorption Candidates

**Date:** 2026-03-18
**From:** wetSpring V129
**To:** barraCuda team, toadStool team, ecosystem
**License:** AGPL-3.0-or-later
**Supersedes:** WETSPRING_V128_ECOSYSTEM_ABSORPTION_HANDOFF_MAR18_2026.md

---

## Executive Summary

wetSpring V129 completes a comprehensive deep-debt evolution sprint: type-safe numeric
casts across 62 files (~170 migrated), unconditional primal identity module,
upstream `bingocube-nautilus` JSON serialization fix, and pure Rust binary discovery.
This handoff documents what changed, what the barraCuda/toadStool teams should absorb,
and learnings relevant to ecosystem evolution.

**Key metrics:** 1,548+ tests, 7 proptest modules, zero clippy warnings (pedantic+nursery),
zero unsafe code, zero mocks in production, zero TODO/FIXME, 15 safe cast helpers, zero
hardcoded primal name strings in library code, 62 files changed (net -51 lines).

---

## Part 1: What wetSpring V129 Evolved

### 1a. `cast` Module (15 Helpers, ~170 Migrations)

The `cast.rs` module was expanded from 9 to 15 safe numeric conversion helpers:

| Helper | Conversion | Semantics |
|--------|-----------|-----------|
| `usize_f64` | usize → f64 | Exact for < 2^53 |
| `f64_usize` | f64 → usize | Truncate, debug-assert non-negative |
| `usize_u32` | usize → u32 | Debug-assert fits |
| `i32_f64` | i32 → f64 | Exact (`f64::from`) |
| `u32_f64` | u32 → f64 | Exact (`f64::from`) |
| `f64_u32` | f64 → u32 | Truncate, debug-assert range |
| `u32_usize` | u32 → usize | Widening |
| `u64_usize` | u64 → usize | Saturate on 32-bit |
| `u64_f64` | u64 → f64 | Exact for < 2^53 |
| `usize_u64` | usize → u64 | Widening |
| `u64_u32` | u64 → u32 | **V129 new** — debug-assert fits |
| `i64_f64` | i64 → f64 | **V129 new** — exact for |v| < 2^53 |
| `u128_f64` | u128 → f64 | **V129 new** — exact for < 2^53 |
| `f64_i32` | f64 → i32 | **V129 new** — truncate, debug-assert |

**barraCuda action:** Consider absorbing this pattern as `barracuda::cast` for
ecosystem-wide consistency. All springs use `as` casts; a shared module
would eliminate scattered `#[expect(clippy::cast_*)]` across the ecosystem.

### 1b. `primal_names` Module — Unconditional Primal Identity

Previously, primal name constants lived in `ipc/primal_names.rs` behind
`#[cfg(feature = "ipc")]`, forcing `niche.rs` to duplicate the DEPENDENCIES
table for the non-IPC case.

V129 creates a top-level `primal_names.rs` (unconditionally available) and
has `ipc/primal_names.rs` re-export for backward compatibility.

**Ecosystem learning:** Primal name constants should never be feature-gated.
Every primal needs its identity for niche manifests, provenance, and
capability registration — all without requiring full IPC infrastructure.

### 1c. Upstream `bingocube-nautilus` Fix

Root cause identified and fixed in `primalTools/bingoCube/nautilus/src/lib.rs`:

- `to_json()` was serializing `observations_count: usize` but NOT the actual
  `Vec<BetaObservation>` — `from_json()` created capacity but zero elements
- Fix: Added `#[cfg_attr(feature = "json", derive(Serialize, Deserialize))]`
  to `BetaObservation` and serialize the actual observations vector
- The `json_roundtrip` test (pre-existing failure across all Nautilus consumers)
  now passes

**All Nautilus consumers should benefit** (hotSpring, neuralSpring, groundSpring).

### 1d. Pure Rust Binary Discovery

Replaced `Command::new("which")` with a pure Rust `find_on_path()` function
using `std::env::split_paths(&path_var)`. Applied in:

- `barracuda/src/bio/dorado.rs` (V128)
- `barracuda/src/bin/validate_nucleus_v4.rs` (V129)

**Ecosystem learning:** No spring should spawn `which` — it's not portable and
not sovereign. `std::env::split_paths` + `is_file()` is the pure Rust equivalent.

---

## Part 2: What barraCuda Should Absorb

### 2a. `monod()` and `haldane()` (12 Validation Binaries)

12 validation binaries define local Michaelis–Menten and Haldane kinetics:

```rust
fn monod(s: f64, mu_max: f64, ks: f64) -> f64 {
    mu_max * s / (ks + s)
}
fn haldane(s: f64, mu_max: f64, ks: f64, ki: f64) -> f64 {
    mu_max * s / (ks + s + s * s / ki)
}
```

**Recommendation:** Add `barracuda::stats::monod(s, mu_max, ks)` and
`barracuda::stats::haldane(s, mu_max, ks, ki)` — these are universal substrate
kinetics used across biochemistry, environmental science, and pharmacology.

**Affected binaries:**
- `validate_petaltongue_biogas_v1.rs`
- `validate_cpu_vs_gpu_v11.rs`
- `validate_metalforge_v19.rs`
- `validate_toadstool_dispatch_v4.rs`
- `validate_pure_gpu_streaming_v13.rs`
- `validate_cpu_vs_gpu_v10.rs`
- `validate_barracuda_cpu_v27.rs`
- `validate_paper_math_control_v6.rs`
- `validate_barracuda_cpu_v26.rs`
- `benchmark_python_vs_rust_v5.rs`
- `validate_metalforge_v18.rs`
- `validate_fungal_fermentation_digestate.rs`
- `validate_anaerobic_coffee_residues.rs`

### 2b. `cast` Module Pattern

See §1a. The `cast` module pattern eliminates a recurring ecosystem-wide problem:
scattered `#[expect(clippy::cast_*)]` attributes on every file that does numeric
conversion. A single shared module with documented precision semantics is better.

### 2c. `log_sum_exp` CPU Implementation

`barracuda/src/bio/numerics.rs` contains a CPU `log_sum_exp` because barraCuda
only provides a GPU equivalent. If barraCuda adds a CPU path, wetSpring can delegate.

---

## Part 3: What toadStool Should Know

### 3a. metalForge `forge` Dispatch Maturity

The `metalForge/forge/` crate is stable at v0.3.0:
- `probe.rs`: GPU (wgpu) + CPU (/proc) + NPU (/dev) discovery
- `dispatch.rs`: Capability-based workload routing
- `bridge.rs`: forge ↔ barracuda device bridge

All NPU device path resolution now goes through `wetspring_barracuda::niche::discover_npu_device()`.

### 3b. 5 ODE Systems via `generate_shader()`

All 5 biological ODE systems lean on `BatchedOdeRK4<S>::generate_shader()`:
Bistable, Capacitor, Cooperation, MultiSignal, PhageDefense.

Zero local WGSL. Upstream `integrate_cpu()` is 20-33% faster than local integrators.

### 3c. 40+ Unique Primitives Consumed

wetSpring consumes ~40 unique GPU primitives and 150+ total barraCuda operations.
See `barracuda/EVOLUTION_READINESS.md` for the full breakdown.

---

## Part 4: Ecosystem Learnings

### For All Springs

1. **Cast modules are high-value**: ~170 casts migrated, ~30 expect attributes removed.
   Payoff: clippy pedantic+nursery clean without scattered suppressions.

2. **Primal names must be unconditional**: Feature-gating identity constants
   forces code duplication. Move them to a top-level, always-available module.

3. **`Command::new("which")` is not sovereign**: Use `std::env::split_paths`.

4. **Serialization round-trip tests catch real bugs**: The nautilus JSON bug
   existed across all consumers. Every serializable struct needs a roundtrip test.

5. **`#[expect]` removal is a quality signal**: Removing unnecessary expect
   attributes proves the underlying code evolved past the need for suppressions.

### For barraCuda

- `monod` + `haldane` are the last remaining local math in wetSpring
- A shared `cast` module would benefit the entire ecosystem
- CPU `log_sum_exp` is a gap (GPU equivalent exists)

### For bingoCube/nautilus

- `BetaObservation` serialization fix needs to propagate to all consumers
- The `primalTools/bingoCube/nautilus` directory is not in a separate git repo —
  the fix was applied directly; track appropriately

---

## Action Items

| Actor | Action | Priority |
|-------|--------|----------|
| **barraCuda** | Absorb `monod()` + `haldane()` into `barracuda::stats` | Medium |
| **barraCuda** | Consider absorbing `cast` module pattern | Low |
| **barraCuda** | Add CPU `log_sum_exp` | Low |
| **bingoCube** | Verify nautilus JSON fix propagated to all consumers | High |
| **All springs** | Audit for `Command::new("which")` → `find_on_path` | Low |
| **All springs** | Move primal names out of feature-gated modules | Medium |

---

## License

AGPL-3.0-or-later. Part of the ecoPrimals sovereign compute fossil record.
