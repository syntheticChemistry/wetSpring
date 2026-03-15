# wetSpring V114 → barraCuda / toadStool: Deep Audit + Absorption Evolution

**Date:** 2026-03-12
**From:** wetSpring V114 (Deep Audit)
**To:** barraCuda / toadStool team
**License:** AGPL-3.0-or-later
**Covers:** V114 audit fixes, absorption status, upstream requests, evolution roadmap

---

## Executive Summary

V114 is a deep audit and debt resolution pass across the full wetSpring codebase. 15 binaries received missing `required-features` gates (fixing a broken default build), 52 clippy pedantic/nursery warnings were resolved, 4 deprecated batch parsers migrated to streaming, inline tolerances replaced with named constants, VRAM estimation evolved from hardcoded to capability-based, and taxonomy classification logic deduplicated. All quality gates now pass: fmt, clippy (pedantic+nursery), doc, test — zero warnings across all.

---

## Part 1: Absorption Status — Fully Lean

wetSpring is **fully lean** on barraCuda v0.3.5 (`0649cd0`):

| Metric | Value |
|--------|-------|
| barraCuda primitives consumed | 150+ |
| Local WGSL shaders | 0 |
| Local ODE derivative code | 0 |
| Local CPU math | 0 (all delegated to `barracuda::*`) |
| Unsafe code | 0 (`#![forbid(unsafe_code)]`) |
| Dual-path fallbacks | 0 |

### Namespaces Consumed

| Namespace | Example Usage |
|-----------|---------------|
| `barracuda::stats` | `norm_cdf`, `mean`, `covariance`, `bootstrap_ci`, `jackknife_mean_variance`, `pearson_correlation`, `spearman_correlation`, `fit_linear`, `correlation`, `diversity::shannon/simpson` |
| `barracuda::special` | `erf`, `ln_gamma`, `regularized_gamma_p`, `bessel_j0_minus1_f64`, `erfc_f64`, `expm1_f64`, `log1p_f64` |
| `barracuda::numerical` | `trapz`, `gradient_1d` |
| `barracuda::linalg` | `graph_laplacian`, `effective_rank` |
| `barracuda::spectral` | `anderson_3d`, `lanczos`, `lanczos_eigenvalues`, `level_spacing_ratio`, `find_w_c`, `GOE_R`, `POISSON_R` |
| `barracuda::ops` | `BrayCurtisF64`, `FusedMapReduceF64`, `GemmF64`, `CorrelationF64`, `CovarianceF64`, `VarianceF64`, `WeightedDotF64` |
| `barracuda::device` | `WgpuDevice`, `HardwareCalibration`, `PrecisionBrain`, `DeviceCapabilities`, `FmaPolicy`, `PhysicsDomain` |

---

## Part 2: Upstream Requests (Carried Forward from V113)

### P0 — Immediate

| # | Target | Justification |
|---|--------|---------------|
| 1 | `barracuda::io::biom` | BIOM format OTU table parser for EMP 28K-sample atlas (Exp364) |
| 2 | `barracuda::ncbi::entrez::esearch_ids()` | Expose accession list for batch fetch workflows |
| 3 | `barracuda::bio::kinetics::gompertz_fit()` | Nonlinear fitting for Track 6 biogas pipeline |

### P1 — Near-Term

| # | Target | Justification |
|---|--------|---------------|
| 4 | `barracuda::bio::anderson::temporal_w()` | Dynamic W(t) model for KBS LTER 30-year data (Exp366) |
| 5 | `barracuda::bio::qs::regulon_map()` | FNR/ArcAB/Rex regulon cross-reference (Exp367) |

### P2 — Unwired Primitives

Available but not yet consumed by wetSpring:

| Primitive | Session | Use Case | Priority |
|-----------|---------|----------|----------|
| `SparseGemmF64` | S60 | Track 3 drug repurposing NMF (CSR × dense) | Medium |
| `TranseScoreF64` | S60 | Track 3 knowledge graph drug-disease scoring | Medium |
| `TopK` | S60 | Track 3 drug-disease pair ranking | Medium |
| `ComputeDispatch` | S65 | Eliminate BGL/pipeline boilerplate in GPU modules | Medium |
| `BandwidthTier` | S62 | PCIe-aware metalForge dispatch routing | Low |
| `LogsumexpWgsl` | Early | HMM forward pass stability (currently stable without) | Low |

---

## Part 3: V114 Audit Fixes (For Your Awareness)

These are internal fixes that don't affect the barraCuda/toadStool API surface:

1. **15 binaries** received `required-features = ["gpu"]` and/or `["json"]` — they used `tokio::runtime::Runtime` or `serde_json` without feature gates.
2. **Deprecated batch parsers** (`parse_fastq`, `parse_ms2`) fully eliminated — all I/O now uses streaming iterators.
3. **VRAM estimation** evolved from hardcoded `12_u64` GB to `DeviceCapabilities::from_device(dev).max_buffer_size / (1024^3)` — capability-based.
4. **`argmax_with_priors`** unified in `bio::taxonomy` module (was duplicated in `streaming_gpu.rs` and `taxonomy_gpu.rs`).

---

## Part 4: API Observations (Feedback)

1. `api_key()` returns `Option<String>` — consider `api_key_or_empty()` for convenience.
2. `ncbi::entrez` is private — consider exposing `esearch_ids(db, term, retmax, api_key) -> Vec<String>`.
3. `DataChannel::Scatter` requires `point_labels: Vec<String>` even when empty — consider `#[serde(default)]`.
4. wgpu `AdapterInfo` does not expose VRAM — wgpu issue #2447. When resolved, barraCuda should surface it through `DeviceCapabilities`.

---

## Part 5: Python Baseline & Industry Benchmark Status

### Python Baselines for barraCuda CPU Parity

| Coverage | Domains |
|----------|---------|
| Full timing baseline | 23 domains (`scripts/benchmark_rust_vs_python.py`) |
| Track 1c baseline | 5 domains (`scripts/barracuda_cpu_v4_baseline.py`) |
| Track 6 baseline | 6 domains (`scripts/python_anaerobic_biogas_baseline.py`) |
| Paper-specific | ~20 additional per-paper scripts |
| **Gap** | ~19 domains lack dedicated Python timing baselines (derep, merge_pairs, quality_filter, kmd, unifrac, pcoa, norm_cdf, erf, etc.) |

### Industry GPU Benchmarks

- **Kokkos**: Not applicable to wetSpring (life science). Kokkos/LAMMPS validation is hotSpring's domain.
- **Galaxy vs GPU**: No direct comparison. Exp015 compares Galaxy vs Rust CPU only.
- **R industry**: V107 parity (53/53 PASS) vs R/vegan, R/DADA2, R/phyloseq — CPU only.
- **Gap**: No external industry tool vs Rust GPU benchmark exists for life science.

---

## Part 6: Evolution Readiness

All 47 CPU modules and 42 GPU modules are lean or compose. The only CPU-only module is `phred` (per-base lookup, no parallelism benefit).

### Next Wiring Targets

1. `ComputeDispatch` → eliminate BGL/pipeline boilerplate across GPU modules
2. `SparseGemmF64` + `TranseScoreF64` + `TopK` → Track 3 drug repurposing GPU promotion
3. `BandwidthTier` → metalForge PCIe-aware dispatch

### Hardware Learning Profile

The `output/hardware_capability_profile.json` format (Exp362) is ready for toadStool `hw-learn` module absorption. Contains: adapter name, precision tiers, domain routing, firmware inventory, FMA policy, bio workload thresholds, sovereign dispatch status.

---

## Action Items

- [ ] **barraCuda**: Expose `esearch_ids()` in `ncbi::entrez` (P0.2)
- [ ] **barraCuda**: Add `barracuda::io::biom` parser (P0.1)
- [ ] **barraCuda**: Add `bio::kinetics::gompertz_fit()` (P0.3)
- [ ] **toadStool**: Absorb hardware capability profile format from Exp362
- [ ] **wetSpring**: Wire `ComputeDispatch` in GPU modules (next evolution pass)
- [ ] **wetSpring**: Expand Python baselines to cover all 47 domains
