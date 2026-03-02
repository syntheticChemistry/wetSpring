# wetSpring → ToadStool/BarraCUDA Handoff V88 — Full Experiment Buildout + Barracuda API Learnings

**Date**: March 2, 2026
**From**: wetSpring (V88)
**To**: ToadStool/BarraCUDA team
**ToadStool pin**: S71+++ (`1dd7e338`)
**License**: AGPL-3.0-or-later
**Supersedes**: V87 (blueFish WhitePaper + Brain Architecture Review)

---

## Executive Summary

- **8 new experiments (Exp263-270)**: 427 validation checks across every compute layer — CPU pure math, CPU↔GPU parity, metalForge cross-system dispatch, NUCLEUS atomics, ToadStool primitive-level math, mixed-hardware PCIe bypass, and biomeOS graph coordination. All PASS.
- **Barracuda API deep dive**: Detailed findings on `barracuda::stats`, `linalg`, `special`, `numerical`, `spectral` API surfaces that matter for Spring consumers. Return types, struct layouts, and call chains documented below for the upstream team.
- **CPU-only workload routing clarified**: `dispatch::route` returning `None` for `ShaderOrigin::CpuOnly` workloads is correct behavior, not an error. Validation binaries now correctly differentiate GPU-routable from CPU-only workloads.
- **Mixed-hardware dispatch proven end-to-end**: 47-workload catalog through NUCLEUS Tower→Node→Nest with bandwidth-aware multi-GPU routing, NPU→GPU PCIe bypass, and sovereign fallback.

---

## Part 1: Barracuda API Learnings for Upstream

### Return Type Patterns

These findings document the actual API surface that Spring consumers encounter.
Upstream team should consider whether these patterns are intentional or should
be streamlined for consumer ergonomics.

| Function | Returns | Consumer Must |
|----------|---------|---------------|
| `stats::pearson_correlation(x, y)` | `Result<f64>` | `.unwrap()` or `?` |
| `stats::spearman_correlation(x, y)` | `Result<f64>` | `.unwrap()` or `?` |
| `special::ln_gamma(x)` | `Result<f64>` | `.unwrap()` or `?` |
| `numerical::trapz(y, dx)` | `Result<f64>` | `.unwrap()` or `?` |
| `stats::fit_linear(x, y)` | `Option<FitResult>` | `.unwrap()` |
| `stats::fit_exponential(x, y)` | `Option<FitResult>` | `.unwrap()` |

### `FitResult` Struct Layout

`FitResult` uses a generic `params: Vec<f64>` rather than named fields:

```
FitResult {
    model: String,          // e.g. "linear", "exponential"
    params: Vec<f64>,       // [slope, intercept] for linear; [rate, amplitude] for exponential
    r_squared: f64,
    rmse: f64,
}
```

Consumers must know the parameter ordering convention per model. Consider whether
named accessors (e.g., `fit.slope()`, `fit.intercept()`) would reduce consumer errors.

### `graph_laplacian` Signature

```rust
pub fn graph_laplacian(adjacency: &[f64], n: usize) -> Vec<f64>
```

Takes a flat row-major `n×n` adjacency matrix, not `Vec<Vec<f64>>`. This is the
correct design for GPU interop (flat arrays), but consumers constructing adjacency
matrices must flatten manually.

### Spectral Pipeline Chain

The spectral analysis pipeline requires three separate calls:

```rust
let csr = spectral::anderson_3d(l, w, seed);       // → SpectralCsrMatrix
let tridiag = spectral::lanczos(&csr);              // → LanczosTridiag
let eigenvalues = spectral::lanczos_eigenvalues(&tridiag); // → Vec<f64>
```

This is a good composable design. The intermediate types (`SpectralCsrMatrix`,
`LanczosTridiag`) enable consumers to inspect or branch between steps.

### `linalg::ridge_regression` Signature

```rust
pub fn ridge_regression(
    x: &[f64], y: &[f64],
    n_samples: usize, n_features: usize, n_outputs: usize,
    regularization: f64,
) -> RidgeResult
```

Flat arrays with explicit dimensions. Returns `RidgeResult { weights: Vec<f64> }`.

### `linalg::nmf` Signature

```rust
pub fn nmf(v: &[f64], m: usize, n: usize, config: &NmfConfig) -> (Vec<f64>, Vec<f64>)
```

Returns `(W, H)` as flat arrays. `NmfConfig { rank, max_iter, tolerance, objective }`.
`NmfObjective::KullbackLeibler` and `Euclidean` supported.

---

## Part 2: Experiment Results

### Exp263: BarraCuda CPU v20 (37/37 PASS)

20th CPU parity round. Vault DF64 + cross-domain pure Rust math validation.

### Exp264: CPU vs GPU v7 (22/22 PASS)

27-domain GPU parity proof (G17-G21 expansion).

### Exp265: metalForge v12 (63/63 PASS)

Extended cross-system dispatch with bandwidth-aware routing.

### Exp266: NUCLEUS v3 (106/106 PASS)

Tower→Node→Nest + Vault + biomeOS full lifecycle validation.

### Exp267: ToadStool Dispatch v3 — Pure Rust Math (41/41 PASS)

Validates pure Rust math across 6 `barracuda` domains:

| Section | Domain | Key Finding |
|---------|--------|-------------|
| S1 | `barracuda::stats` | Linear fit slope=2.0, r²=1.0; Pearson=1.0, Spearman=1.0 |
| S2 | `barracuda::linalg` | Laplacian row sums exactly zero; NMF fully non-negative |
| S3 | `barracuda::special` | erf(x) + erfc(x) = 1 to 1e-14 |
| S4 | `barracuda::numerical` | Hessian of x²+y² = diag(2,2) to 1e-4 |
| S5 | wetSpring `bio::diversity` | Delegates are bit-identical to `barracuda::stats` |
| S6 | `barracuda::spectral` | Anderson LSR=0.6687 (W=4, L=4) |

### Exp268: CPU vs GPU Pure Math (38/38 PASS)

Deepest GPU parity layer — not bio domains, but the underlying barracuda math:

| Section | Primitive | Finding |
|---------|-----------|---------|
| S1 | `FusedMapReduceF64` | Shannon/Simpson/Observed match at sizes 32-4096 |
| S2 | `BrayCurtisF64` | All 3 pairs within `GPU_VS_CPU_F64` tolerance |
| S3 | `BatchedEighGpu` | Eigenvalues + variance within tolerance |
| S5 | DF64 pack/unpack | Roundtrip error exactly 0.0 for π, e, 1e-15, 1e20, −π, 0 |
| S6 | `GpuPipelineSession` | 5 consecutive streaming runs bit-identical |

### Exp269: Mixed Hardware Dispatch (91/91 PASS)

NUCLEUS atomics through full hardware spectrum:

| Section | Focus | Finding |
|---------|-------|---------|
| S1 | Tower discovery | 3 GPUs detected with bandwidth tiers |
| S2 | NPU→GPU PCIe bypass | 0 CPU round-trips |
| S5 | 8-stage mixed pipeline | GPU→GPU→GPU→GPU→GPU→NPU→CPU→CPU |
| S6 | Write→Absorb→Lean | 96% absorption rate (45/47) |
| S7 | Full catalog dispatch | 45 GPU-routable + 2 CPU-only (correct) |

### Exp270: biomeOS Graph Coordination (29/29 PASS)

Full biomeOS layer: socket discovery, primal orchestration, 3 pipeline topologies,
sovereign mode (zero external primal dependencies).

---

## Part 3: CPU-Only Workload Routing

### Finding

`dispatch::route()` returns `None` for `ShaderOrigin::CpuOnly` workloads
(`ncbi_assembly_ingest`, `fastq_parsing`). This is correct: these workloads
are inherently CPU-bound I/O operations that don't need GPU substrate routing.

### Recommendation

Consider making this explicit in the `dispatch` module documentation or API:

```rust
// dispatch::route returns None for CPU-only workloads by design.
// Callers should check ShaderOrigin before routing.
```

Validation binaries now correctly partition workloads before routing:

```rust
if matches!(bw.origin, ShaderOrigin::CpuOnly) {
    // Not a routing failure — routing not applicable
    continue;
}
let decision = dispatch::route(&bw.workload, &substrates);
```

---

## Part 4: Absorption Targets

### From V87 (still open)

1. **Brain architecture generalization** — abstract hotSpring 4-layer brain to `barracuda::brain`
2. **ERI shader class** — 4-center integral pattern for computational chemistry
3. **`esn_v2` shape bug** — `InputMatrix` shape validation for `n_inputs != 1`

### New from V88

4. **`FitResult` named accessors** — optional ergonomic layer over `params: Vec<f64>`
5. **`dispatch::route` docs** — clarify `None` return for CPU-only workloads
6. **Spectral pipeline convenience** — optional `anderson_eigenvalues(l, w, seed)` combining the 3-step chain

---

## Part 5: Quality State

| Metric | Value |
|--------|-------|
| Experiments | 270 |
| Validation checks | 7,083+ (all PASS) |
| Tests | 1,249 (1 ignored: hardware-dependent) |
| Binaries | 253 (233 validate + 20 benchmark) |
| ToadStool primitives consumed | 93 |
| Local WGSL shaders | 0 |
| Clippy | CLEAN (pedantic + nursery, zero warnings) |
| Coverage | 95.86% line |
| Named tolerances | 97 |
| Unsafe code | 0 |

---

## Part 6: What Changed Since V87

| V87 | V88 |
|-----|-----|
| 262 experiments | 270 experiments (+8) |
| 6,656+ checks | 7,083+ checks (+427) |
| 1,247 tests | 1,249 tests (+2) |
| 238 binaries | 253 binaries (+15) |
| blueFish whitePaper launched | Full control validation across all layers |
| API surface documented informally | API surface documented with exact signatures |

---

*Archived handoff:* `archive/WETSPRING_TOADSTOOL_V87_BLUEFISH_BRAIN_ARCH_HANDOFF_MAR01_2026.md`
