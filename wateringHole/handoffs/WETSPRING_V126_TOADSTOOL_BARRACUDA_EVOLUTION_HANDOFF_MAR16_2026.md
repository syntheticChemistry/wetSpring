# wetSpring V126 → toadStool/barraCuda Evolution Handoff

**Date:** 2026-03-16
**From:** wetSpring V126 (376 experiments, 1,443+ tests, 354 binaries)
**To:** toadStool team, barraCuda team
**License:** AGPL-3.0-or-later

---

## Executive Summary

wetSpring has consumed 150+ barraCuda primitives across 6 tracks (microbial ecology,
analytical chemistry, deep-sea metagenomics, drug repurposing, soil QS, immunological
Anderson). All 42 GPU modules and 41 CPU modules are in Lean phase — zero local WGSL,
zero duplicate math. This handoff documents:

1. **IPC patterns** toadStool should absorb from wetSpring V125–V126
2. **barraCuda primitive consumption** — what wetSpring uses and how
3. **Gaps** — barraCuda features wetSpring can't yet leverage
4. **Ecosystem patterns** that emerged and should be standardized

---

## 1. IPC Patterns for Upstream Absorption

### 1.1 Structured `IpcError` Enum (V125)

wetSpring replaced opaque `Error::Ipc(String)` with typed variants:

```rust
pub enum IpcError {
    SocketPath(String),
    Connect(String),
    Transport(String),
    Codec(String),
    RpcReject { code: i64, message: String },
    EmptyResponse,
}
```

**Why toadStool should absorb:** Enables structured error recovery — retry on
`Connect`/`Transport`, degrade on `RpcReject`, abort on `SocketPath`. All springs
converging on this pattern (healthSpring, groundSpring, airSpring).

### 1.2 `IpcError` Query Helpers (V126)

```rust
impl IpcError {
    pub const fn is_retriable(&self) -> bool;
    pub fn is_timeout_likely(&self) -> bool;
    pub const fn is_method_not_found(&self) -> bool;
    pub const fn is_connection_error(&self) -> bool;
}
```

**Why toadStool should absorb:** Enables circuit-breaker / exponential-backoff logic
without brittle string matching. sweetGrass-compatible.

### 1.3 `DispatchOutcome<T>` Enum (V126)

```rust
pub enum DispatchOutcome<T> {
    Success(T),
    Protocol(DispatchError),       // retriable
    Application { code: i64, message: String }, // deterministic
}
```

**Why toadStool should absorb:** Callers can retry on `Protocol` (transient transport
failures) and report on `Application` (workload rejected). Separates "toadStool is
down" from "toadStool says no".

### 1.4 Health Probes (V126)

| Method | Response | Purpose |
|--------|----------|---------|
| `health.liveness` | `{"alive": true}` | Fast keep-alive (biomeOS polling) |
| `health.readiness` | `{"ready": true, "subsystems": {...}}` | Deep probe with per-subsystem status |
| `health.check` | Delegates to readiness | Backward compatible |

**toadStool action:** Adopt `health.liveness`/`health.readiness` as standard for all
primals. biomeOS orchestrator should poll liveness cheaply, readiness for routing.

### 1.5 Dual-Format Capability Parsing (V125)

`extract_capabilities()` parses both flat (`"capabilities": [...]`) and structured
(`"domains": [{"name": "...", "methods": [...]}]`) formats.

**toadStool action:** Standardize dual-format as the canonical response for
`capability.list` across all primals.

### 1.6 Generic Socket Discovery (V125)

```rust
pub fn socket_env_var(primal: &str) -> String;   // "songbird" → "SONGBIRD_SOCKET"
pub fn discover_primal(primal: &str) -> Option<PathBuf>;
```

**toadStool action:** Adopt as the standard discovery pattern. All primals should
follow `{PRIMAL_UPPER}_SOCKET` convention.

---

## 2. barraCuda Primitive Consumption Map

### 2.1 GPU Primitives (42 modules, all Lean)

| Primitive | wetSpring Modules | Usage Pattern |
|-----------|-------------------|---------------|
| `FusedMapReduceF64` | diversity, streaming, eic, spectral_match, chimera, derep, merge_pairs, neighbor_joining, molecular_clock, reconciliation, kmd, rarefaction (12) | Map-reduce over f64 arrays |
| `GemmF64` / `GemmCached` | gemm_cached, taxonomy, spectral_match, chimera, derep (5) | Matrix multiplication |
| `BrayCurtisF64` | diversity, streaming (2) | Beta diversity |
| `BatchedOdeRK4` | bistable, capacitor, cooperation, multi_signal, phage_defense (5) | ODE integration with `generate_shader()` |
| `BatchedOdeRK4F64` | ode_sweep (1) | Generic ODE sweep |
| `BatchedEighGpu` | pcoa (1) | Eigenvalue decomposition for PCoA |
| `PeakDetectF64` | signal (1) | 1D peak detection |
| `KrigingF64` | kriging (1) | Spatial interpolation |
| `VarianceF64` / `CorrelationF64` / `CovarianceF64` / `WeightedDotF64` | stats, eic (2) | Statistical aggregates |
| `BatchToleranceSearchF64` | tolerance_search (1) | ppm-tolerance m/z matching |
| `DiversityFusionGpu` | diversity_fusion (1) | Fused diversity pipeline |
| `BatchedMultinomialGpu` | rarefaction (1) | Multinomial subsampling |
| `KmerHistogramGpu` | kmer, kmd, derep (3) | K-mer counting |
| `UniFracPropagateGpu` | unifrac (1) | Phylogenetic diversity |
| `AniBatchF64` / `DnDsBatchF64` / `SnpCallingF64` / `PangenomeClassifyGpu` | ani, dnds, snp, pangenome (4) | Comparative genomics |
| `HmmBatchForwardF64` | hmm (1) | Hidden Markov Model forward |
| `QualityFilterGpu` / `Dada2EStepGpu` / `RfBatchInferenceGpu` | quality, dada2, random_forest (3) | Bioinformatics ML |
| `PairwiseL2Gpu` / `PairwiseHammingGpu` / `PairwiseJaccardGpu` | distance metrics (3) | Pairwise distance matrices |
| `SpatialPayoffGpu` / `BatchFitnessGpu` / `LocusVarianceGpu` | game theory (3) | Evolutionary dynamics |
| `WrightFisherGpu` / `StencilCooperationGpu` / `HillGateGpu` | population genetics (3) | Stochastic simulation |
| `SymmetrizeGpu` / `LaplacianGpu` | graph (2) | Graph operations |
| `BootstrapMeanGpu` / `KimuraGpu` / `HargreavesBatchGpu` / `JackknifeMeanGpu` | extended stats (4) | Statistical estimation |

### 2.2 CPU Primitives (41 modules, all Lean)

| Module | Functions Used |
|--------|---------------|
| `stats::diversity` | `shannon`, `simpson`, `chao1`, `bray_curtis`, `pielou_evenness`, `rarefaction_curve`, `alpha_diversity` |
| `stats` | `bootstrap_ci`, `mean`, `percentile`, `correlation`, `norm_cdf`, `pearson_correlation`, `jackknife_mean_variance`, `fit_linear` |
| `special` | `erf`, `ln_gamma`, `regularized_gamma_p` |
| `numerical` | `trapz`, `rk45::*`, `OdeSystem` traits, all 5 ODE systems |
| `linalg` | `ridge_regression`, `nmf`, `graph_laplacian` |
| `spectral` | `anderson_3d`, `lanczos`, `level_spacing_ratio`, `find_w_c` |
| `esn_v2` | `ESN`, `MultiHeadEsn`, `HeadConfig`, `ReservoirConfig`, `TrainResult` |

### 2.3 ODE Systems via `generate_shader()`

Five ODE systems use barraCuda's `BatchedOdeRK4::generate_shader()` for GPU-native
integration (zero local WGSL):

1. `BiStableOde` — Fernandez 2020 phenotypic switching
2. `CapacitorOde` — Mhatre 2020 capacitor dynamics
3. `CooperationOde` — Bruger & Waters 2018 cooperation
4. `MultiSignalOde` — Srivastava 2011 multi-input QS
5. `PhageDefenseOde` — Hsueh/Severin 2022 phage defense

---

## 3. Gaps — barraCuda Primitives wetSpring Can't Yet Leverage

| Primitive | barraCuda Session | Potential Use | Priority |
|-----------|-------------------|---------------|----------|
| `SparseGemmF64` | S60 | Drug repurposing NMF (CSR × dense, large matrices) | Medium |
| `TranseScoreF64` | S60 | Knowledge graph drug-disease scoring (Track 3) | Medium |
| `TopK` | S60 | Drug-disease pair ranking after NMF/TransE | Medium |
| `LogsumexpWgsl` | Early | HMM forward numerical stability improvement | Low |
| `ValidationHarness` | S59 | Richer validation API (could replace custom `Validator`) | Low |

**barraCuda action:** Prioritize `SparseGemmF64` + `TranseScoreF64` + `TopK` stabilization
for Track 3 drug repurposing GPU acceleration. wetSpring has CPU baselines ready.

---

## 4. Ecosystem Patterns That Emerged

### 4.1 Zero-Panic Validation (`OrExit` Trait)

```rust
pub trait OrExit<T> {
    fn or_exit(self, msg: &str) -> T;
}
```

Used in 47+ validation binaries. `std::process::exit(1)` instead of panic.
**Recommended:** Standardize across all springs.

### 4.2 Named Tolerance Constants

214 named constants in `tolerances.rs` (e.g., `PYTHON_PARITY`, `GPU_F32_PARITY`,
`ANDERSON_IPR_TOL`). Documented in `TOLERANCE_REGISTRY.md`.
**Recommended:** barraCuda adopt a shared tolerance vocabulary.

### 4.3 Capability-Based Discovery

All primal interactions use runtime discovery — no hardcoded paths:
- `WETSPRING_SOCKET`, `TOADSTOOL_SOCKET`, `SQUIRREL_SOCKET`
- `$XDG_RUNTIME_DIR/biomeos/{primal}-default.sock`
- `<temp_dir>/{primal}-default.sock`

### 4.4 Structured Tracing

Migrated all `eprintln!` to `tracing::info!`/`tracing::warn!` following
coralReef/neuralSpring pattern.

### 4.5 `deny.toml`

Workspace-level `deny.toml` with `wildcards = "deny"`, `yanked = "deny"`,
`confidence-threshold = 0.8`. **Recommended:** All springs adopt.

---

## 5. Evolution Phase Summary

| Phase | Count | Description |
|-------|:-----:|-------------|
| **Lean** | 150+ | GPU/CPU modules consuming upstream barraCuda directly |
| **Compose** | 7 | Wiring multiple barraCuda primitives (kmd, merge_pairs, robinson_foulds, derep, neighbor_joining, reconciliation, molecular_clock) |
| **Write** | 0 | Zero local WGSL shaders |
| **Passthrough** | 0 | All promoted to Lean/Compose |

---

## 6. Recommended Upstream Actions

### For barraCuda Team
1. **Stabilize** `SparseGemmF64`, `TranseScoreF64`, `TopK` for Track 3 absorption
2. **Consider** `IpcError`-style enum for barraCuda's own error types
3. **Adopt** named tolerance vocabulary from wetSpring for cross-spring consistency

### For toadStool Team
1. **Absorb** `DispatchOutcome<T>` pattern into toadStool's dispatch API
2. **Standardize** `health.liveness`/`health.readiness` probes for all primals
3. **Adopt** `IpcError` query helpers for circuit-breaker logic
4. **Enforce** `{PRIMAL_UPPER}_SOCKET` discovery convention in PCIe topology routing
5. **Standardize** dual-format `capability.list` response (flat + domains)

### For Ecosystem (biomeOS/sweetGrass)
1. **biomeOS:** Poll `health.liveness` for keep-alive, `health.readiness` for routing
2. **sweetGrass:** Wire `IpcError::is_retriable()` into provenance trio circuit breaker
3. **All springs:** Converge on `DispatchOutcome` for toadStool dispatch clients
