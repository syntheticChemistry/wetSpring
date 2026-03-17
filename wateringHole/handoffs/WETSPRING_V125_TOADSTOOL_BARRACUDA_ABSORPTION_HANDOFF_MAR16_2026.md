# wetSpring V125 → toadStool/barraCuda Absorption Handoff

**Date:** 2026-03-16
**From:** wetSpring V125 (ecoPrimals)
**To:** toadStool (compute orchestration), barraCuda (math primitives)
**License:** AGPL-3.0-or-later
**Status:** Active handoff

---

## Executive Summary

wetSpring has consumed 150+ barraCuda primitives across stats, linalg, ops,
spectral, special, numerical, and device modules. Zero local WGSL remains.
All 5 ODE systems use `BatchedOdeRK4::generate_shader()`. The codebase is in
full **Lean** phase. This handoff documents patterns, IPC conventions, and
lessons learned that toadStool and barraCuda should absorb to benefit the
entire ecoPrimals ecosystem.

- **150+ barraCuda primitives** consumed (stats, linalg, ops, spectral, special, numerical, device)
- **44 GPU modules**, 12 Compose (wiring multiple primitives), 5 ODE Write→Lean
- **1,719+ tests**, 376 experiments, 5,707+ validation checks, 354 binaries
- **Zero** local WGSL, zero unsafe, zero `#[allow()]`, clippy pedantic+nursery clean

---

## Part 1: IPC Patterns for toadStool Absorption

### 1.1 Structured `IpcError` Enum

wetSpring V125 replaced `Error::Ipc(String)` with typed variants. **Recommend
barraCuda and toadStool adopt this pattern** so all primals have structured
error recovery:

```rust
pub enum IpcError {
    SocketPath(String),           // bind/create/remove failures
    Connect(String),              // connection refused / not found
    Transport(String),            // write/read/flush/timeout/shutdown
    Codec(String),                // serialize/deserialize failures
    RpcReject { code, message },  // JSON-RPC error from remote primal
    EmptyResponse,                // no response or missing result field
}
```

**Key benefit:** Callers can match on variant for structured recovery — retry
on `Connect`, degrade on `RpcReject`, abort on `SocketPath`. metalForge already
has `SongbirdError`/`NestError`/`DataError` with similar granularity; this
pattern unifies them under a single enum.

### 1.2 Dual-Format Capability Parsing

wetSpring V125 adds `extract_capabilities()` that parses both flat and structured
capability responses:

```rust
pub struct CapabilityInfo {
    pub capabilities: Vec<String>,      // Format A: flat list
    pub domains: Vec<CapabilityDomain>, // Format B: structured domains
    pub primal: Option<String>,
    pub version: Option<String>,
}
```

**toadStool action:** Consider promoting `extract_capabilities()` to barraCuda
or a shared IPC crate so all primals can parse capability responses without
reimplementing.

### 1.3 Generic Socket Discovery Convention

wetSpring V125 formalizes the `{PRIMAL_UPPER}_SOCKET` env var convention:

```rust
pub fn socket_env_var(primal: &str) -> String {
    format!("{}_SOCKET", primal.to_ascii_uppercase())
}
pub fn discover_primal(primal: &str) -> Option<PathBuf> {
    discover_socket(&socket_env_var(primal), primal)
}
```

**toadStool action:** Adopt this as the standard discovery API in the IPC
protocol spec. All springs and primals should be discoverable by name alone.

### 1.4 `compute.dispatch` Client

wetSpring V124 implemented a typed `compute.dispatch` client for toadStool
S156+ dispatch:

- `DispatchHandle` (job_id + optional compute_socket)
- `DispatchError` enum: `NoComputePrimal`, `Transport`, `Rpc`, `MalformedResponse`
- `submit()`, `result()`, `capabilities()` with auto-discovery
- Zero serde dependency (lightweight JSON extraction)

**toadStool action:** Validate this client against toadStool S156+ and publish
a reference implementation in the IPC protocol spec.

### 1.5 Centralized RPC Error Extraction

wetSpring V123 added `extract_rpc_error()` for structured JSON-RPC error parsing.
Used by Songbird, NestGate, and provenance clients. healthSpring V29 pattern.

**barraCuda action:** Consider promoting to shared IPC utilities.

---

## Part 2: barraCuda Primitive Consumption Map

### 2.1 Heavy-Use Modules

| barraCuda Module | wetSpring Usage | Delegation Count |
|------------------|----------------|:----------------:|
| `stats` | `shannon`, `simpson`, `chao1`, `bray_curtis`, `pearson_correlation`, `bootstrap_ci`, `jackknife_mean_variance`, `norm_cdf`, `mean`, `hill`, `fit_exponential`, `r_squared`, `AlphaDiversity` | ~48 |
| `ops` | `WrightFisherGpu`, `StencilCooperationGpu`, `HillGateGpu`, `GemmF64`, `BatchedEighGpu`, `LaplacianGpu`, `fused_map_reduce_f64`, `transe_score_f64`, `peak_detect_f64`, `sparse_gemm_f64`, `bray_curtis_f64` | ~45 |
| `spectral` | `anderson_3d`, `lanczos`, `lanczos_eigenvalues`, `level_spacing_ratio`, `GOE_R`, `POISSON_R`, `find_w_c`, `AndersonSweepPoint` | ~40 |
| `device` | `WgpuDevice`, `PhysicsDomain`, `PrecisionTier`, `PrecisionBrain`, `HardwareCalibration`, `DeviceCapabilities` | ~38 |
| `numerical` | `BatchedOdeRK4`, `OdeSystem`, `trapz`, `numerical_hessian`, `CooperationOde`, `MultiSignalOde` | ~17 |
| `shaders` | `Precision`, `provenance` (evolution_report, shader_count, cross_spring_shaders) | ~17 |
| `special` | `erf`, `erfc_f64`, `expm1_f64`, `log1p_f64`, `bessel_j0_minus1_f64`, `anderson_diagonalize`, `tridiagonal_ql` | ~15 |
| `linalg` | `nmf` (NmfConfig, NmfObjective), `graph_laplacian`, `ridge_regression`, `CsrMatrix` | ~12 |

### 2.2 ODE Systems (Write → Absorb → Lean)

All 5 ODE systems generate WGSL at runtime via `BatchedOdeRK4<S>::generate_shader()`:

| ODE System | Module | Status |
|------------|--------|--------|
| PhageDefenseOde | `bio/phage_defense_gpu.rs` | Lean |
| MultiSignalOde | `bio/multi_signal_gpu.rs` | Lean |
| CapacitorOde | `bio/capacitor_gpu.rs` | Lean |
| BistableOde | `bio/bistable_gpu.rs` | Lean |
| CooperationOde | `bio/cooperation_gpu.rs` | Lean |

### 2.3 Compose Phase (Wiring Multiple Primitives)

12 GPU modules compose multiple barraCuda primitives:

| Module | Primitives Combined |
|--------|-------------------|
| `kmd_gpu` | GemmF64 + FusedMapReduce |
| `merge_pairs_gpu` | SmithWatermanGpu + alignment ops |
| `robinson_foulds_gpu` | LaplacianGpu + tree ops |
| `derep_gpu` | Hamming + clustering |
| `neighbor_joining_gpu` | Distance + tree construction |
| `reconciliation_gpu` | DTL + tree ops |
| `molecular_clock_gpu` | Felsenstein + rate estimation |
| `chimera_gpu` | Alignment + voting |
| `gbm_gpu` | Decision tree ensemble |
| `feature_table_gpu` | Sparse ops + reduce |
| `streaming_gpu` | BrayCurtis + FMR pipeline |
| `taxonomy_gpu` | NaiveBayes + kmer |

---

## Part 3: Ecosystem Patterns for Adoption

### 3.1 Zero-Panic Validation

The `OrExit` trait replaces all `.expect()`/`.unwrap()` in validation binaries
with `process::exit(1)` + structured error message. No panics, no stack traces —
clean CI output. healthSpring, neuralSpring have also adopted this.

**barraCuda action:** Consider adding `OrExit` to barraCuda core for all springs.

### 3.2 Workspace `deny.toml`

Root-level `deny.toml` with:
- `wildcards = "deny"` — no `*` version specs
- `yanked = "deny"` — fail on yanked crates
- `confidence-threshold = 0.8`
- Advisory DB configured

groundSpring, airSpring, healthSpring all have this pattern.

### 3.3 Structured Tracing

`tracing` v0.1 replaces `eprintln!` in all server/IPC code. coralReef Phase 10
pattern. All springs converging on this.

### 3.4 `#[expect(lint, reason = "...")]`

Replaced all `#[allow()]` with self-documenting `#[expect(reason)]` across
276+ binaries. Unfulfilled expectations are compile errors. biomeOS Edition 2024
pattern.

### 3.5 Named Tolerances

214 centralized tolerance constants — zero inline magic numbers in validation
code. groundSpring/ludoSpring "Python tolerance mirror" pattern (46-60 constants).

---

## Part 4: What wetSpring Does NOT Need from barraCuda

| Item | Reason |
|------|--------|
| `TensorSession` | Not implemented; neuralSpring domain (ML ops) |
| `nn` module | No neural network workloads |
| `nautilus` module | Used via bingocube-nautilus, not barraCuda |

---

## Part 5: Recommended Upstream Actions

### For barraCuda

1. **Promote `IpcError` pattern** — shared typed IPC error enum for all springs
2. **Promote `extract_capabilities()`** — dual-format parsing utility
3. **Promote `OrExit` trait** — zero-panic validation for all consumers
4. **Promote `socket_env_var()`** — standard discovery convention
5. **Audit `eigh` feature gate** — `linalg::eigh` gated behind `gpu` but used by `ipc` feature path

### For toadStool

1. **Validate `compute.dispatch` client** — test against S156+ dispatch
2. **Publish reference IPC client** — based on wetSpring's typed implementation
3. **Adopt structured `IpcError`** — for dispatch result forwarding
4. **Formalize env var convention** — `{PRIMAL_UPPER}_SOCKET` as protocol standard

---

## Test Evidence

| Metric | Value |
|--------|-------|
| Library tests | 1,475 (117 IPC, 9 error, 252 metalForge) |
| Total tests | 1,719+ |
| Experiments | 376 |
| Validation checks | 5,707+ |
| Binaries | 354 |
| Clippy warnings | 0 (pedantic + nursery) |
| `unsafe` blocks | 0 (`#![forbid(unsafe_code)]`) |
| `#[allow()]` | 0 |
| Local WGSL | 0 |
| Named tolerances | 214 |
