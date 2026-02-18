# wetSpring White Paper

**Status**: Working draft — reviewed for PII, suitable for public repository
**Purpose**: Document the validation of life science and analytical chemistry pipelines on consumer hardware using `BarraCUDA`
**Date**: February 2026 (updated 2026-02-18)

---

## Documents

| Document | Description | Audience |
|----------|-------------|----------|
| [METHODOLOGY.md](METHODOLOGY.md) | Two-track validation protocol and acceptance criteria | Methodology review |
| [STUDY.md](STUDY.md) | Full study narrative: replication + Rust + GPU | General / publication |

---

## What This Study Is

wetSpring validates published computational life science and analytical chemistry workflows — from raw sequencing reads (FASTQ) and mass spectrometry data (mzML) through diversity metrics, PFAS screening, and ecological analysis — on consumer hardware. Every pipeline passes through three phases:

- **Phase 1 (Galaxy Control)**: Run the published analysis in Galaxy/QIIME2/Python on our hardware. Validate against reference publications. Profile performance. Document tool versions.
- **Phase 2 (Rust Port)**: Re-implement critical I/O and compute stages using sovereign Rust code (zero external parser dependencies). Validate output against Phase 1 baselines.
- **Phase 3 (GPU Acceleration)**: Promote validated Rust code to WGSL f64 shaders via `ToadStool`/`BarraCUDA`. Validate GPU results against CPU baselines.

The evolution path is: **Python baseline → Rust validation → GPU acceleration → sovereign pipeline.**

The study answers three questions:
1. **Can published bioinformatics and analytical chemistry be independently reproduced?** (Answer: yes — 89 CPU validation checks pass across FASTQ, mzML, diversity, PFAS screening, feature extraction, and peak detection. 8 of 10 experiments complete with baselines from Galaxy, QIIME2, asari, FindPFAS, and scipy.)
2. **Can Rust replace the Python scientific stack for these domains?** (Answer: yes — 30 sovereign Rust modules cover the complete 16S pipeline (FASTQ → quality → merge → derep → DADA2 → chimera → taxonomy → diversity → UniFrac) and the full LC-MS feature pipeline (mzML → EIC → peaks → features). 284 tests, 0 clippy pedantic warnings, 0 production unwrap/expect. Only runtime dependency: `flate2` for gzip.)
3. **Can consumer GPUs accelerate ecological and analytical computation at f64 precision?** (Answer: yes — 38/38 GPU validation checks pass. 11 ToadStool primitives (wgpu v22). GPU spectral cosine achieves **1,077× speedup** over CPU at 200×200 matrix. Zero custom WGSL shaders.)

---

## Key Results

### Phase 2 (Rust Ports): 89/89 CPU checks pass

| Validation Binary | Modules Tested | Checks | Status |
|-------------------|----------------|:------:|--------|
| `validate_fastq` | `io::fastq`, quality, merge, derep | 28 | PASS |
| `validate_diversity` | `bio::diversity` + `bio::kmer` | 18 | PASS |
| `validate_mzml` | `io::mzml` + `io::xml` (sovereign XML) | 7 | PASS |
| `validate_pfas` | `io::ms2` + `bio::tolerance_search` + KMD | 10 | PASS |
| `validate_features` | `bio::eic` + `bio::signal` + `bio::feature_table` | 9 | PASS |
| `validate_peaks` | `bio::signal` vs scipy.signal.find_peaks | 17 | PASS |

### Phase 3 (GPU Acceleration): 38/38 GPU checks pass

| Primitive / Shader | Operation | Tolerance | Status |
|-------------------|-----------|-----------|--------|
| `FusedMapReduceF64` (ToadStool) | GPU Shannon entropy | `≤ 1e-10` vs CPU | 3/3 PASS |
| `FusedMapReduceF64` (ToadStool) | GPU Simpson index | `≤ 1e-6` vs CPU | 3/3 PASS |
| `BrayCurtisF64` (ToadStool, absorbed) | GPU all-pairs BC distance | `≤ 1e-10` vs CPU | 6/6 PASS |
| `BatchedEighGpu` (ToadStool) | GPU PCoA ordination | `≤ 1e-6` vs CPU | 5/5 PASS |
| `FusedMapReduceF64` (ToadStool) | GPU alpha diversity bundle | `≤ 1e-6` vs CPU | 6/6 PASS |
| `GemmF64` + `FusedMapReduceF64` | GPU spectral cosine matching | `≤ 1e-6` vs CPU | 8/8 PASS |
| `VarianceF64` (ToadStool) | GPU variance / std dev | `≤ 1e-10` vs CPU | 3/3 PASS |
| `CorrelationF64` + `CovarianceF64` | GPU Pearson r / covariance | `≤ 1e-10` vs CPU | 2/2 PASS |
| `WeightedDotF64` (ToadStool) | GPU weighted/plain dot product | `≤ 1e-10` vs CPU | 2/2 PASS |

**Discovery**: The Shannon shader initially required a portable `log_f64()` implementation because native `log(f64)` crashes the NVIDIA NVVM compiler. A 2× coefficient bug was found and fixed in ToadStool's `math_f64.wgsl`. Shannon and Simpson have since been rewired to ToadStool's `FusedMapReduceF64` (single-dispatch fused map-reduce), and the deprecated custom shaders were removed.

### Dependency Sovereignty

| Dependency | Role | Status |
|------------|------|--------|
| `flate2` | gzip decompression (FASTQ) | **Only runtime dependency** |
| `io::xml` | XML pull parsing (mzML) | **Sovereign** (replaced `quick-xml`) |
| `io::fastq` | FASTQ streaming parser | **Sovereign** (replaced `needletail`) |
| `encoding` | Base64 decoding (mzML arrays) | **Sovereign** (replaced `base64` crate) |
| `wgpu` + `tokio` | GPU compute (feature-gated) | Optional, `gpu` feature only |

### New in Feb 2026: Complete 16S Pipeline

| Stage | Module | Tests | Baseline |
|-------|--------|-------|----------|
| DADA2 denoising | `bio::dada2` | 9 | Galaxy/QIIME2 (Exp001) |
| Chimera detection | `bio::chimera` | 8 | UCHIME3 algorithm |
| Taxonomy classification | `bio::taxonomy` | 8 | RDP naive Bayes / SILVA 138 |
| UniFrac distance | `bio::unifrac` | 8 | Newick tree + weighted/unweighted |
| Feature pipeline GPU | `bio::eic_gpu` | — | ToadStool FusedMapReduceF64 + WeightedDotF64 |
| Rarefaction GPU | `bio::rarefaction_gpu` | — | ToadStool FusedMapReduceF64 + bootstrap CI |

### Three-Tier Performance Benchmark

| Workload | Python | Rust CPU | Rust GPU | GPU/CPU |
|----------|--------|----------|----------|---------|
| Shannon 1M | 7.1ms | 3.7ms | 4.3ms | 0.85× |
| Bray-Curtis 100×100 | 15.0ms | 1.04ms | 2.6ms | 0.40× |
| **Spectral cosine 200×200** | 8.8ms | 3,937ms | **3.7ms** | **1,077×** |

Full benchmark: [`BENCHMARK_RESULTS.md`](../BENCHMARK_RESULTS.md)

---

## Relation to Other Documents

- **[`CONTROL_EXPERIMENT_STATUS.md`](../CONTROL_EXPERIMENT_STATUS.md)**: Detailed experiment logs, validation counts, and evolution roadmap.
- **[`EVOLUTION_READINESS.md`](../EVOLUTION_READINESS.md)**: Module-by-module GPU promotion readiness assessment.
- **[`papers/PIPELINE_TARGETS.md`](../papers/PIPELINE_TARGETS.md)**: Target paper mapping — what each paper computes and how it maps to Rust modules.
- **[`HANDOFF_WETSPRING_TO_TOADSTOOL_FEB_16_2026.md`](../HANDOFF_WETSPRING_TO_TOADSTOOL_FEB_16_2026.md)**: Technical handoff to the `ToadStool`/`BarraCUDA` team — shader findings, `log_f64` bug fix, evolution roadmap.
- **`hotSpring/whitePaper/`** (sibling project): Computational physics validation following the same methodology.
- **`wateringHole/`** (parent ecosystem): Inter-primal standards, naming conventions, and shared documentation.

---

## Reproduction

```bash
# Phase 1 (Galaxy/Python, ~4 hours)
bash scripts/setup_tools.sh
bash scripts/download_data.sh
bash scripts/start_galaxy.sh
# Then run experiments/001-008 in Galaxy

# Phase 2 (Rust validation, ~2 minutes)
cd barracuda
cargo test --release                              # 284 unit + integration tests
cargo run --release --bin validate_fastq           # 28/28 PASS
cargo run --release --bin validate_diversity        # 18/18 PASS
cargo run --release --bin validate_mzml            # 7/7 PASS
cargo run --release --bin validate_pfas            # 10/10 PASS
cargo run --release --bin validate_features         # 9/9 PASS
cargo run --release --bin validate_peaks           # 17/17 PASS

# Phase 3 (GPU, requires SHADER_F64 GPU)
cargo run --release --features gpu --bin validate_diversity_gpu  # 38/38 PASS

# Performance benchmark (CPU vs GPU)
cargo run --release --features gpu --bin benchmark_cpu_gpu
```

No institutional access required. AGPL-3.0 licensed.

---

## Hardware

| Component | Specification |
|-----------|--------------|
| CPU | Intel i9-12900K (8P+8E, 24 threads) |
| RAM | 64 GB DDR5 |
| GPU | NVIDIA RTX 4070 12 GB (`SHADER_F64` confirmed) |
| OS | Pop!_OS 22.04 (Linux 6.17) |
| Rust | stable (1.82+) |
| wgpu | v22 (Vulkan backend) |

---

## Grand Total: 127/127 Quantitative Checks Pass

| Phase | Checks | Description |
|-------|:------:|-------------|
| Phase 2 (CPU) | 89 | 28 FASTQ + 18 diversity + 7 mzML + 10 PFAS + 9 features + 17 peaks |
| Phase 3 (GPU) | 38 | 3 Shannon + 3 Simpson + 6 BC + 5 PCoA + 6 alpha + 8 spectral + 7 stats |
| **Total** | **127** | **All pass** |

## Rust Module Inventory (30 modules, 284 tests)

| Domain | Modules |
|--------|---------|
| I/O | `io::fastq`, `io::mzml`, `io::ms2`, `io::xml`, `encoding` |
| 16S Pipeline | `bio::quality`, `bio::merge_pairs`, `bio::derep`, `bio::dada2`, `bio::chimera`, `bio::taxonomy`, `bio::kmer` |
| Diversity | `bio::diversity`, `bio::pcoa`, `bio::unifrac` |
| LC-MS | `bio::eic`, `bio::signal`, `bio::feature_table`, `bio::spectral_match`, `bio::tolerance_search`, `bio::kmd` |
| Spatial | `bio::kriging` |
| GPU | `bio::diversity_gpu`, `bio::pcoa_gpu`, `bio::spectral_match_gpu`, `bio::stats_gpu`, `bio::eic_gpu`, `bio::rarefaction_gpu` |
| Infrastructure | `validation`, `tolerances`, `error`, `gpu` |
