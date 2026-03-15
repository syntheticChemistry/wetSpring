# wetSpring V118 → ToadStool/BarraCUDA Absorption + Evolution Handoff

**Date:** March 15, 2026
**From:** wetSpring V117 (376 experiments, 5,707+ checks, 1,667 tests, 354 binaries)
**To:** ToadStool/BarraCUDA team
**Authority:** wateringHole (ecoPrimals Core Standards)
**Supersedes:** V117 Deep Tolerance Centralization + Code Quality Handoff (Mar 15)
**Pins:** barraCuda v0.3.5 (`03986ce`), toadStool S130+, coralReef Phase 10
**License:** AGPL-3.0-or-later

---

## Executive Summary

- wetSpring consumes **150+ barraCuda primitives** across 12+ domains (stats, spectral, ops, device, numerical, linalg, esn_v2, tensor, shaders, special, bio, unified_hardware)
- **47 GPU modules** (35 lean, 12 compose, 5 write-to-lean ODE) — zero local WGSL, fully lean on upstream
- **200+ named tolerance constants** in a 4-submodule hierarchy ready for cross-spring adoption
- **14 IPC capability domains** / 19 methods registered for Songbird discovery
- **Zero production panics, zero unsafe, zero clippy warnings** — codebase is clean for absorption analysis
- **6 absorption targets** identified for toadStool/barraCuda to evolve

---

## Part 1: Primitives Consumed by Domain

### stats
`mean`, `correlation`, `covariance`, `norm_cdf`, `hill`, `fit_exponential`, `r_squared`, `pearson_correlation`, `variance`, `BootstrapMeanGpu`, `KimuraGpu`, `HargreavesBatchGpu`, `JackknifeMeanGpu`

### spectral
`anderson_2d`, `anderson_3d`, `lanczos`, `lanczos_eigenvalues`, `level_spacing_ratio`, `GOE_R`, `POISSON_R`, `AndersonSweepPoint`, `find_w_c`

### special
`bessel_j0_minus1_f64`, `erfc_f64`, `expm1_f64`, `log1p_f64`, `anderson_diagonalize`, `tridiagonal_ql`, `erf`

### ops (GPU)
`FusedMapReduceF64`, `GemmF64`, `BatchedEighGpu`, `BrayCurtisF64`, `WeightedDotF64`, `BatchToleranceSearchF64`, `PeakDetectF64`, `CorrelationF64`, `CovarianceF64`, `VarianceF64`, `TranseScoreF64`, `kmd_grouping_f64`

### ops::bio (GPU)
`BatchedMultinomialGpu`, `DiversityFusionGpu`, `PairwiseL2Gpu`, `QualityFilterGpu`, `UniFracPropagateGpu`, `Dada2EStepGpu`, `HmmBatchForwardF64`, `GillespieGpu`, `SmithWatermanGpu`, `FelsensteinGpu`, `TreeInferenceGpu`, `RfBatchInferenceGpu`, `PangenomeClassifyGpu`, `SnpCallingF64`, `AniBatchF64`, `DnDsBatchF64`, `BatchFitnessGpu`, `PairwiseJaccardGpu`, `PairwiseHammingGpu`, `LocusVarianceGpu`, `KmerHistogramGpu`, `SpatialPayoffGpu`, `BatchedOdeRK4F64`

### device
`WgpuDevice`, `ComputeDispatch`, `Device`, `BufferPool`, `PooledBuffer`, `TensorContext`, `PhysicsDomain`, `PrecisionTier`, `FmaPolicy`, `domain_requires_separate_fma`

### numerical (ODE)
`PhageDefenseOde`, `MultiSignalOde`, `CapacitorOde`, `BistableOde`, `CooperationOde`, `ode_generic::BatchedOdeRK4`

### linalg
`nmf::NmfConfig`, `NmfObjective`, `NmfResult`, `cosine_similarity`

### esn_v2
`ESN`, `ESNConfig`, `ExportedWeights`, `HeadConfig`, `MultiHeadEsn`

### tensor / shaders / unified_hardware
`Tensor`, `Precision`, `provenance::*` (27 cross-spring shader records), `BandwidthTier`

---

## Part 2: Tolerance Architecture (200+ Named Constants)

wetSpring's tolerance system is a 4-submodule hierarchy designed for cross-spring adoption:

```
tolerances/
├── mod.rs           # Machine-precision, special-fn, Jacobi, NMF, ODE, DF64, pharma, bootstrap
├── bio/             # 10 submodules: alignment, anderson, brain, diversity, esn, misc, ode, parity, phylogeny
├── instrument.rs    # GC, m/z, retention time, PFAS (PFHxS, PFSA homologue window)
├── gpu.rs           # GPU vs CPU parity (f64, transcendental, log polyfill, ODE landscape)
└── spectral.rs      # Anderson, dynamic W(t), level spacing, Lyapunov, soil QS, FAO-56
```

**Key design patterns:**
- Every constant has a doc comment explaining scientific justification and provenance
- Constants are grouped by domain, not by tolerance magnitude
- Hierarchy-tested: `EXACT_F64 < ANALYTICAL_F64 < PYTHON_PARITY < GPU_VS_CPU_F64 < EMPIRICAL_TOLERANCE`
- All validation binaries reference named constants from `tolerances::*` — zero inline magic numbers

**toadStool action:** Consider a shared tolerance crate or barraCuda `tolerances` feature gate that springs can consume. The naming conventions and hierarchy pattern are ready for standardization across the ecosystem. hotSpring, airSpring, neuralSpring, and groundSpring could all adopt this structure.

---

## Part 3: GPU Evolution Status

| Category | Count | Description |
|----------|:-----:|-------------|
| **Lean** | 35 | Consume upstream barraCuda primitives directly |
| **Compose** | 12 | Wire multiple barraCuda primitives (kmd, merge_pairs, robinson_foulds, derep, NJ, reconciliation, molecular_clock, chimera, gbm, feature_table, streaming, taxonomy) |
| **Write → Lean (ODE)** | 5 | Use `BatchedOdeRK4<S>::generate_shader()` — WGSL deleted, lean on upstream |
| **Local WGSL** | 0 | All absorbed by barraCuda |
| **Tier B/C** | 0 | All promoted |

**What this means for toadStool/barraCuda:**
- wetSpring has no local shader maintenance burden — all evolution happens upstream
- The 12 "Compose" modules are the most interesting for absorption analysis: they wire multiple primitives into domain-specific pipelines (e.g., `chimera_gpu` composes `FusedMapReduceF64` + `GemmF64`)
- The ODE `generate_shader()` pattern is the gold standard — domain structs define their own WGSL at compile time via trait implementation

---

## Part 4: ESN/NPU Bridge

wetSpring maintains a bio-attention state machine in `bio/esn/heads.rs`:

| Constant | Value | Purpose |
|----------|:-----:|---------|
| `URGENCY_ESCALATE_ALERT` | 0.6 | Healthy → Alert |
| `URGENCY_ESCALATE_CRITICAL` | 0.8 | Alert → Critical |
| `URGENCY_DEESCALATE` | 0.3 | Recovery threshold |
| `PHASE_LABEL_LOW` | 0.3 | Phase discretization low/medium |
| `PHASE_LABEL_HIGH` | 0.6 | Phase discretization medium/high |
| `NUM_HEADS` | 36 | Gen 2 bio heads |
| `GROUP_SIZE` | 6 | Heads per group |

**Bio brain architecture:**
- 36-head `MultiHeadBioEsn` with `BioHeadGroupDisagreement` + `AttentionState` (Healthy/Alert/Critical)
- `BioNautilusBrain` bridges bingoCube/nautilus evolutionary reservoir to bio domains
- Int8 quantization pipeline: ESN → `ExportedWeights` → NPU deployment (AKD1000)
- 3 IPC methods: `brain.observe`, `brain.attention`, `brain.urgency`

**toadStool action:** The `AttentionState` hysteresis pattern (escalate at 0.6/0.8, de-escalate at 0.3) is a reusable pattern for any domain that needs regime classification with oscillation prevention. Consider promoting to `barracuda::esn_v2::attention` or `toadStool::brain`.

---

## Part 5: IPC Capability Architecture

14 domains, 19 methods registered for Songbird discovery:

| Domain Family | Domains | Methods |
|--------------|:-------:|:-------:|
| ecology.* | 11 | 12 (diversity, qs_model, anderson, kinetics, alignment, taxonomy, phylogenetics, nmf, timeseries, ncbi, pipeline) |
| provenance | 1 | 3 (begin, record, complete) |
| brain | 1 | 3 (observe, attention, urgency) |
| metrics | 1 | 1 (snapshot) |

**Implementation:**
- `capability.list` JSON-RPC handler returns full domain/method registry
- `discover.rs`: env var → `XDG_RUNTIME_DIR` → `temp_dir()` fallback for socket discovery
- JSON-RPC 2.0 over Unix sockets, 10-request multiplexing validated

**toadStool action:** The capability discovery pattern (env → XDG → temp fallback) is now standardized across wetSpring and should be consistent across all springs. Ensure toadStool's Songbird implementation matches this 3-tier fallback.

---

## Part 6: Absorption Targets

| Target | Priority | What | Why |
|--------|:--------:|------|-----|
| `ComputeDispatch` for ODE modules | P3 | 6 ODE modules still use manual buffer group layout instead of `ComputeDispatch` | Reduces boilerplate, enables dispatch-level optimization |
| DF64 GEMM adoption | P3 | `Fp64Strategy::Hybrid` for consumer GPUs where native f64 is slow | Unlock 2-4x speedup on RTX 4070 for GEMM-heavy workloads |
| `BandwidthTier` wiring | P3 | PCIe-aware dispatch in metalForge | Currently probed but not used for routing decisions |
| 8 GPU primitive opportunities | P2-P3 | Compose modules that could become lean if barraCuda adds fused variants | `chimera_gpu`, `derep_gpu`, `taxonomy_gpu` compose 2-3 primitives each |
| Tolerance standardization | P2 | Shared tolerance naming convention + hierarchy across springs | 200+ constants in wetSpring are ready for cross-spring adoption |
| `crate::special` extraction | P3 | 4 local math functions (`erf`, `ln_gamma`, `regularized_gamma_lower`, `integrate_peak`) duplicate barraCuda | Blocked on `barracuda::math` CPU-only feature gate |

---

## Part 7: Learnings for toadStool/barraCuda Evolution

### Streaming-only I/O
wetSpring V112 eliminated all buffered I/O APIs in favor of streaming iterators. This reduced memory usage for large FASTQ/mzML files from O(n) to O(1). The pattern: `Parser::stream(reader) -> impl Iterator<Item = Record>`. barraCuda's bio ops should expect streaming input.

### Tolerance naming convention
Pattern: `{DOMAIN}_{METRIC}_{QUALIFIER}`. Examples: `PYTHON_PARITY` (1e-10), `GPU_VS_CPU_F64` (1e-6), `SOIL_QS_TILLAGE` (0.2), `ANDERSON_NU_PARITY` (0.4). The hierarchy is strictly ordered from exact to empirical. This convention could become an ecosystem standard.

### hotSpring pattern reuse
wetSpring adopted hotSpring's `AttentionState` machine (Healthy/Alert/Critical with hysteresis), `Write → Absorb → Lean` cycle, and 3-tier validation (paper → CPU → GPU → streaming → metalForge). These patterns proved transferable across physics and biology domains without modification.

### Provenance trio wiring
`provenance.begin` / `provenance.record` / `provenance.complete` lifecycle manages session tracking. Every validation run could produce a provenance record linking: dataset accession → Python baseline → Rust result → GPU result → tolerance used → pass/fail.

### Zero-panic production code
V117 eliminated all 4 production `panic!()` calls. The pattern: GPU module functions return `Result`, test helpers use `?` instead of `.unwrap_or_else(|e| panic!())`. This should be the standard for all barraCuda-consuming code.

### ODE `generate_shader()` pattern
The cleanest GPU integration: domain structs implement a trait that generates WGSL at compile time. No local `.wgsl` files to maintain. wetSpring's 5 ODE systems (PhageDefense, Bistable, MultiSignal, Cooperation, Capacitor) all use this pattern via `BatchedOdeRK4<S>`. This is the gold standard for domain-specific GPU code.

---

## Quality Gates (at handoff time)

| Check | Result |
|-------|--------|
| `cargo fmt --check` | Zero violations |
| `cargo clippy --workspace --all-features` | Zero warnings |
| `cargo test --workspace` | 1,667 passed, 0 failed, 2 ignored |
| Production `panic!()` | 0 |
| Production `unsafe` | 0 (`#![forbid(unsafe_code)]`) |
| TODO/FIXME/HACK in production | 0 |
| Local WGSL shaders | 0 |
| Named tolerances | 200+ |
| barraCuda primitives consumed | 150+ |

---

## Recommended toadStool Actions

1. **Absorb tolerance naming convention** — promote `{DOMAIN}_{METRIC}_{QUALIFIER}` hierarchy as ecosystem standard
2. **Add `barracuda::math` CPU-only feature gate** — unblock `crate::special` extraction from springs
3. **Wire `ComputeDispatch` to ODE modules** — 6 modules ready for migration
4. **Evaluate `Fp64Strategy::Hybrid` for GEMM** — consumer GPU DF64 performance opportunity
5. **Standardize Songbird socket fallback** — env → XDG → temp pattern across all primals
6. **Consider fused bio ops** — `chimera_detect_fused`, `derep_fused`, `taxonomy_fused` would promote 3 compose modules to lean
