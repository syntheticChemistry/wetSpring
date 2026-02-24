# wetSpring → ToadStool Handoff V31: Absorption Targets & Evolution Insights

**Date:** February 24, 2026
**From:** wetSpring (life science & analytical chemistry biome)
**To:** ToadStool / BarraCuda team
**Phase:** 41 — Post S59-lean; documenting next absorption targets and cross-spring learnings

---

## Summary

This handoff identifies what wetSpring still has locally that ToadStool should
consider absorbing, what we learned during the S58-S59 lean cycle, and
cross-spring evolution patterns that benefit all springs.

---

## 1. Absorption Targets: What wetSpring Still Has Locally

### 1.1 HIGH — GPU Shader Candidates (Write phase active)

These 5 ODE GPU wrappers generate WGSL at runtime via `BatchedOdeRK4<S>::generate_shader()`.
The **scenario functions** (CPU-only validation logic) remain local, but the underlying
GPU dispatch is already lean. The next evolution step is for ToadStool to absorb the
full ODE sweep pattern with configurable batch scaling.

| Module | Vars | Params | Scenario Functions | GPU Wrapper |
|--------|:----:|:------:|---|---|
| `bistable_gpu` | 5 | 21 | `scenario_*`, `bifurcation_scan` | Uses `BatchedOdeRK4<BistableOde>` |
| `capacitor_gpu` | 6 | 16 | `scenario_*` | Uses `BatchedOdeRK4<CapacitorOde>` |
| `cooperation_gpu` | 4 | 13 | `cooperator_frequency`, `scenario_*` | Uses `BatchedOdeRK4<CooperationOde>` |
| `multi_signal_gpu` | 7 | 24 | `scenario_*` | Uses `BatchedOdeRK4<MultiSignalOde>` |
| `phage_defense_gpu` | 4 | 11 | `scenario_*` | Uses `BatchedOdeRK4<PhageDefenseOde>` |

**Suggestion:** Consider a `BatchedOdeSweep` high-level API in ToadStool that
wraps `BatchedOdeRK4` with parameter-space sweep, batched dispatch, and result
collection. This would let all springs replace their custom GPU wrappers.

### 1.2 MEDIUM — CPU Primitives Worth Absorbing

| Function | Location | Lines | Why Absorb |
|----------|----------|:-----:|---|
| `cosine_similarity` | Various binaries (was in `bio/nmf.rs`) | ~10 | Used by 10+ binaries for spectral matching; already in `barracuda::linalg::nmf` but could be a first-class `barracuda::linalg::cosine_similarity` |
| `top_k_cosine` | `validate_matrix_pharmacophenomics.rs` | ~20 | Cosine-based top-K on NMF latent factors; useful for recommendation/ranking |
| `integrate_peak` | `bio/eic.rs` | ~30 | Chromatographic peak area; already delegates to `barracuda::numerical::trapz` on GPU but with index-range slicing |
| `EsnConfig` / ESN training | `bio/esn.rs` | ~400 | Full ESN with reservoir, readout, NPU export; ToadStool has `esn_v2` but wetSpring's ridge-based training path is well-tested |

### 1.3 LOW — Feature-Gate Blocked (needs `math` feature)

These functions have GPU-path delegates already but keep local fallbacks because
`barracuda` is behind the `gpu` feature gate. If ToadStool adds a `math` feature
that doesn't pull in `wgpu`, these could lean fully upstream.

| Function | wetSpring | ToadStool | Status |
|----------|-----------|-----------|--------|
| `erf(x)` | `special.rs` | `barracuda::special::erf` | GPU: delegates; no-GPU: local A&S 7.1.26 |
| `ln_gamma(x)` | `special.rs` | `barracuda::special::ln_gamma` | GPU: delegates; no-GPU: local Lanczos |
| `regularized_gamma_lower(a,x)` | `special.rs` | `barracuda::special::regularized_gamma_p` | GPU: delegates; no-GPU: local series |
| `dot(a,b)` | `special.rs` | — | No upstream CPU equivalent; 3 lines |
| `l2_norm(xs)` | `special.rs` | — | No upstream CPU equivalent; 2 lines |

**Suggestion:** Add `dot` and `l2_norm` to `barracuda::linalg` or `barracuda::special`
as CPU-only helpers. They're used by ~15 wetSpring binaries.

---

## 2. What We Learned During S58-S59 Lean

### 2.1 Result Types Matter

ToadStool's NMF returns `Result<NmfResult, BarracudaError>` while wetSpring's
local version panicked on bad input. This is the correct evolution — all 3 NMF
binaries now handle the Result cleanly with `.expect()`. **Lesson:** Always use
`Result` in upstream primitives, even for "can't fail" operations.

### 2.2 Feature-Gate Architecture

The biggest friction in leaning upstream is the feature-gate coupling.
wetSpring depends on ToadStool `barracuda` only through `optional = true`
behind the `gpu` feature. This means:

- No-GPU builds can't access `barracuda::linalg::ridge_regression` even though
  it's pure CPU math
- We had to keep a local `#[cfg(not(feature = "gpu"))]` fallback for ESN ridge
- `special.rs` functions maintain dual implementations

**Suggestion:** Consider splitting `barracuda` into:
- `barracuda-core` (CPU math: linalg, numerical, special, spectral, validation)
- `barracuda-gpu` (WGSL shaders, device, ops, tensor)

This would let springs lean on CPU math without pulling in `wgpu`.

### 2.3 ODE System Trait is the Right Pattern

The `OdeSystem` trait → `BatchedOdeRK4<S>::generate_shader()` pattern is
excellent. It:
- Eliminates hand-written WGSL (we deleted 30KB of local shaders)
- Guarantees CPU↔GPU parity (same derivative function, different executor)
- Makes upstream 10-43% faster on CPU due to shared optimization
- Scales linearly with batch size

### 2.4 Tolerance Constants Should Live Upstream

wetSpring had 60 tolerance constants; after lean, 4 NMF constants were removed
(now 56). Consider absorbing commonly-used tolerances into `barracuda::tolerances`:
- `MATRIX_EPS` (1e-15) — used by ridge, pangenome, ESN
- `PYTHON_PARITY` (1e-10) — used by validation comparisons
- `BOX_MULLER_U1_FLOOR` (1e-10) — used by stochastic sampling

### 2.5 Provenance Tags Are Valuable

The 12 `ProvenanceTag` entries in `barracuda::provenance` track cross-spring
origins. wetSpring suggests adding tags for the S58-S59 absorptions:
- `PROV_NMF` — wetSpring → `linalg::nmf`
- `PROV_RIDGE` — wetSpring → `linalg::ridge`
- `PROV_ODE_BIO_SYSTEMS` — wetSpring → `numerical::ode_bio`
- `PROV_ANDERSON_CORRELATED` — wetSpring → `spectral::anderson`

---

## 3. Cross-Spring Evolution Patterns

### 3.1 The Write-Absorb-Lean Cycle Works

```
wetSpring Writes    ToadStool Absorbs    wetSpring Leans      Measured Benefit
─────────────────   ─────────────────    ─────────────────    ────────────────
bio/nmf.rs (482L)   linalg/nmf.rs (S58)  barracuda::linalg    Result type, shared opt
ode_systems (715L)  ode_bio/ (S58)       barracuda::numerical  10-43% faster CPU
esn ridge (100L)    linalg/ridge (S59)   barracuda::linalg    Cholesky-based, tested
anderson (115L)     spectral/ (S59)      barracuda::spectral  Identical algorithm
```

Total: ~1,412 lines written → absorbed → deleted. Net code reduction in wetSpring
while functionality is preserved (and improved) upstream.

### 3.2 hotSpring Precision → Everyone Benefits

hotSpring's `df64_core.wgsl` and `Fp64Strategy` (absorbed S58) provide f64 on
all GPU tiers. This directly benefits:
- wetSpring ODE bio shaders (require f64 for convergence)
- neuralSpring spectral methods (IPR, Lanczos need f64)
- airSpring Kriging (precision-sensitive spatial interpolation)

### 3.3 neuralSpring Graph Theory → wetSpring Uses

neuralSpring's S54 absorption (`graph_laplacian`, `effective_rank`,
`numerical_hessian`) enables wetSpring's:
- Community network analysis (species interaction graphs)
- Hessian-based ODE sensitivity analysis
- Effective rank of gene expression matrices

### 3.4 Shared Patterns That Emerged

| Pattern | Origin | Now Used By |
|---------|--------|-------------|
| `BatchedOdeRK4<S>` generic ODE | wetSpring+ToadStool | All springs with ODE models |
| `FusedMapReduceF64` | hotSpring | wetSpring (Shannon, Simpson, cosine), neuralSpring, airSpring |
| `ValidationHarness` | neuralSpring (S59) | ToadStool-wide; wetSpring may adopt |
| `ProvenanceTag` | ToadStool | Cross-spring tracking |
| Feature-gated delegation | wetSpring `special.rs` | Pattern for CPU/GPU dual-path |

---

## 4. Remaining Architecture Notes

### 4.1 Validation Binary Pattern

wetSpring has 152 validation binaries following the `hotSpring` pattern:
- Hardcoded expected values with provenance (script, commit, date)
- Explicit pass/fail with exit code 0/1
- Three-tier: Python baseline → Rust CPU → GPU → metalForge

Consider adding a `barracuda::validation::binary!` macro that standardizes:
```rust
validation_binary! {
    name: "Exp160: repoDB NMF",
    required_features: ["gpu"],
    checks: [
        abs("NMF converges", last_error, expected, 1e-6),
        rel("reconstruction error", rel_err, 0.85, 0.05),
    ],
}
```

### 4.2 Streaming Pipeline

wetSpring validated pure GPU streaming (Exp090, Exp105-106) showing 441-837×
speedup over round-trip. The key insight: pre-warming eliminates per-stage
dispatch overhead. ToadStool's `UnidirectionalPipeline` should expose a
`pre_warm()` API for this pattern.

### 4.3 metalForge Integration

wetSpring's `metalForge/forge/` crate (v0.3.0) discovers compute substrates
at runtime. The `ShaderOrigin` enum (`Absorbed`, `Local`, `CpuOnly`) was
useful for tracking absorption status. Consider making this a ToadStool
first-class concept.

---

## 5. Verification

All rewired systems validated:

| Binary | Checks | Exit |
|--------|:------:|:----:|
| `validate_repodb_nmf` | 9/9 | 0 |
| `validate_nmf_drug_repurposing` | 7/7 | 0 |
| `validate_matrix_pharmacophenomics` | 9/9 | 0 |
| `validate_correlated_disorder` | 9/9 | 0 |
| `benchmark_ode_lean_crossspring` | 11/11 | 0 |
| `cargo test --features gpu` | 759 pass | 0 |
| `cargo clippy --all-features -W pedantic -W nursery` | 0 warnings | 0 |

---

## 6. Current wetSpring State

| Metric | Value |
|--------|-------|
| Phase | **41** |
| Tests | **806** (759 barracuda + 47 forge) |
| Experiments | **162** |
| Validation checks | **3,198+** |
| Binaries | **152** |
| CPU modules | **46** |
| GPU modules | **42** |
| ToadStool primitives consumed | **42** |
| Local WGSL shaders | **0** |
| ToadStool alignment | **S59** |
| Tolerance constants | **56** |
| Lines removed in V30-V31 | **~1,412** |
