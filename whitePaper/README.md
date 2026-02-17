# wetSpring White Paper

**Status**: Working draft — reviewed for PII, suitable for public repository
**Purpose**: Document the validation of life science and analytical chemistry pipelines on consumer hardware using `BarraCUDA`
**Date**: February 2026

---

## Documents

| Document | Description | Audience |
|----------|-------------|----------|
| [METHODOLOGY.md](METHODOLOGY.md) | Two-track validation protocol and acceptance criteria | Methodology review |

---

## What This Study Is

wetSpring validates published computational life science and analytical chemistry workflows — from raw sequencing reads (FASTQ) and mass spectrometry data (mzML) through diversity metrics, PFAS screening, and ecological analysis — on consumer hardware. Every pipeline passes through three phases:

- **Phase 1 (Galaxy Control)**: Run the published analysis in Galaxy/QIIME2/Python on our hardware. Validate against reference publications. Profile performance. Document tool versions.
- **Phase 2 (Rust Port)**: Re-implement critical I/O and compute stages using sovereign Rust code (zero external parser dependencies). Validate output against Phase 1 baselines.
- **Phase 3 (GPU Acceleration)**: Promote validated Rust code to WGSL f64 shaders via `ToadStool`/`BarraCUDA`. Validate GPU results against CPU baselines.

The evolution path is: **Python baseline → Rust validation → GPU acceleration → sovereign pipeline.**

The study answers three questions:
1. **Can published bioinformatics and analytical chemistry be independently reproduced?** (Answer: yes — 36/36 CPU validation checks pass across FASTQ parsing, mzML/MS2 parsing, k-mer counting, diversity metrics, and PFAS fragment screening.)
2. **Can Rust replace the Python scientific stack for these domains?** (Answer: in progress — sovereign parsers for FASTQ, mzML, MS2, and XML are complete with zero external parser dependencies. All compute modules validate against Python baselines. 193 tests, 97.9% line coverage, 0 clippy pedantic warnings, 0 production unwrap/expect.)
3. **Can consumer GPUs accelerate ecological and analytical computation at f64 precision?** (Answer: yes — 17/17 GPU diversity validation checks pass. Shannon entropy, Simpson index, Bray-Curtis distance matrices, and PCoA ordination compute on GPU with `≤ 1e-10` error vs CPU. Shannon/Simpson use ToadStool's `FusedMapReduceF64`; PCoA uses `BatchedEighGpu`. GPU dispatch uses Result-based error propagation — no panics.)

---

## Key Results

### Phase 2 (Rust Ports): 36/36 CPU checks pass

| Validation Binary | Modules Tested | Checks | Status |
|-------------------|----------------|:------:|--------|
| `validate_fastq` | `io::fastq` (sovereign, gzip-aware) | 9 | PASS |
| `validate_diversity` | `bio::diversity` + `bio::kmer` | 14 | PASS |
| `validate_mzml` | `io::mzml` + `io::xml` (sovereign XML) | 7 | PASS |
| `validate_pfas` | `io::ms2` + `bio::tolerance_search` | 6 | PASS |

### Phase 3 (GPU Acceleration): 17/17 GPU checks pass

| Primitive / Shader | Operation | Tolerance | Status |
|-------------------|-----------|-----------|--------|
| `FusedMapReduceF64` (ToadStool) | GPU Shannon entropy | `≤ 1e-10` vs CPU | 3/3 PASS |
| `FusedMapReduceF64` (ToadStool) | GPU Simpson index | `≤ 1e-6` vs CPU | 3/3 PASS |
| `BrayCurtisF64` (ToadStool, absorbed) | GPU all-pairs BC distance | `≤ 1e-10` vs CPU | 6/6 PASS |
| `BatchedEighGpu` (ToadStool) | GPU PCoA ordination | `≤ 1e-6` vs CPU | 5/5 PASS |

**Discovery**: The Shannon shader initially required a portable `log_f64()` implementation because native `log(f64)` crashes the NVIDIA NVVM compiler. A 2× coefficient bug was found and fixed in ToadStool's `math_f64.wgsl`. Shannon and Simpson have since been rewired to ToadStool's `FusedMapReduceF64` (single-dispatch fused map-reduce), and the deprecated custom shaders were removed.

### Dependency Sovereignty

| Dependency | Role | Status |
|------------|------|--------|
| `flate2` | gzip decompression (FASTQ) | **Only runtime dependency** |
| `io::xml` | XML pull parsing (mzML) | **Sovereign** (replaced `quick-xml`) |
| `io::fastq` | FASTQ streaming parser | **Sovereign** (replaced `needletail`) |
| `encoding` | Base64 decoding (mzML arrays) | **Sovereign** (replaced `base64` crate) |
| `wgpu` + `tokio` + `bytemuck` | GPU compute (feature-gated) | Optional, `gpu` feature only |

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
cargo test --release                              # 193 unit + integration tests
cargo run --release --bin validate_fastq           # 9/9 PASS
cargo run --release --bin validate_diversity        # 14/14 PASS
cargo run --release --bin validate_mzml            # 7/7 PASS
cargo run --release --bin validate_pfas            # 6/6 PASS

# Phase 3 (GPU, requires SHADER_F64 GPU)
cargo run --release --features gpu --bin validate_diversity_gpu  # 17/17 PASS
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
| wgpu | 0.19 (Vulkan backend) |

---

## Grand Total: 48/48 Quantitative Checks Pass

| Phase | Checks | Description |
|-------|:------:|-------------|
| Phase 2 (CPU) | 36 | 9 FASTQ + 14 diversity/kmer + 7 mzML + 6 PFAS/MS2 |
| Phase 3 (GPU) | 12 | 3 Shannon + 3 Simpson + 4 Bray-Curtis (+ 2 capability) |
| **Total** | **48** | **All pass** |
