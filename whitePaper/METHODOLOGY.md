# Validation Methodology

**Purpose**: Document the two-track validation approach used in the wetSpring study
**See also**: [README.md](README.md) for key results

---

## 1. Two-Track Approach

wetSpring validates two complementary scientific domains that exercise different
computational primitives. Each track follows the same three-phase evolution path.

### Track 1: Life Science (Algae / Metagenomics)

Reproduce published 16S rRNA amplicon microbiome profiling for algae pond
management. This exercises: FASTQ parsing, quality filtering, k-mer counting,
diversity metrics (Shannon, Simpson, Chao1), beta diversity (Bray-Curtis),
and ordination (PCoA).

**Source papers**: Pond Crash Forensics; Biotic Countermeasures (Sandia National Labs)

### Track 2: PFAS Analytical Chemistry (codename blueFish)

Reproduce published PFAS non-target screening from LC-MS/HRMS data. This
exercises: mzML/MS2 parsing, peak detection, mass defect analysis, suspect
screening, and homologous series detection.

**Source papers**: Jones et al. (MSU), Michigan PFAS monitoring studies

### Why Two Tracks

| Track | Computational Primitives | GPU Shader Value |
|-------|-------------------------|------------------|
| Track 1 (Life Science) | Sequence I/O, k-mer ops, diversity metrics | Hash tables, reduce, distance matrices |
| Track 2 (Analytical Chemistry) | Spectral I/O, peak detection, mass math | Signal processing, tolerance search, spectral correlation |

Both tracks produce GPU kernels useful **far beyond** their original domain —
the ecoPrimals convergent evolution thesis in action.

---

## 2. Three-Phase Evolution

### Phase 1: Python / Galaxy Control

Run the published analysis using the original tools on our hardware. This
establishes:

- **Correctness**: Our hardware produces results matching published outputs
- **Performance baseline**: Traditional Python scientific stack
- **Bug inventory**: Upstream issues found during reproduction
- **Provenance**: Exact tool versions, dataset accessions, commands

Phase 1 uses Galaxy (for bioinformatics tools), QIIME2 (for diversity),
asari (for LC-MS), and `PFΔScreen` (for PFAS screening).

### Phase 2: Rust Validation

Re-implement critical I/O and compute stages in sovereign Rust. Compare:

- **Accuracy**: Output matches Phase 1 within documented tolerances
- **Dependencies**: Minimize external crates (target: zero parser dependencies)
- **Streaming**: Zero-copy / streaming I/O where possible
- **Determinism**: Same input → identical output, always

### Phase 3: GPU Acceleration

Promote validated Rust code to WGSL f64 shaders via `ToadStool` / `BarraCUDA`.
Compare:

- **Accuracy**: GPU results match CPU within `GPU_VS_CPU_*` tolerances
- **Throughput**: GPU vs CPU wall time for identical workloads
- **Precision**: IEEE 754 f64 via `SHADER_F64` — no f32 compromise
- **Vendor neutrality**: Vulkan backend, any GPU with f64 support

---

## 3. Workloads

### 3.1 FASTQ Parsing and Quality Assessment (Track 1)

**Tool**: `io::fastq` (sovereign, gzip-aware streaming parser)
**Validates against**: `FastQC` quality reports from Galaxy

| Check | Metric | Tolerance | Rationale |
|-------|--------|-----------|-----------|
| Record count | Exact match | 0 | Deterministic count |
| GC content | Percentage | ± 0.5% | Rounding differences |
| Mean quality score | Phred average | ± 0.5 | Float precision |
| Min/max length | Exact match | 0 | Deterministic |

### 3.2 Mass Spectrometry Parsing (Track 2)

**Tools**: `io::mzml` + `io::xml` (sovereign XML pull parser), `io::ms2`
**Validates against**: pyteomics spectrum extraction

| Check | Metric | Tolerance | Rationale |
|-------|--------|-----------|-----------|
| Spectrum count | Exact match | 0 | Deterministic |
| m/z values | Float comparison | ± 0.01 | IEEE rounding |
| Intensity values | Float comparison | ± 0.01 | IEEE rounding |
| MS level | Exact match | 0 | Integer attribute |

### 3.3 K-mer Counting (Track 1)

**Tool**: `bio::kmer` (2-bit canonical k-mer engine)
**Validates against**: DADA2 dereplication counts

| Check | Metric | Tolerance | Rationale |
|-------|--------|-----------|-----------|
| Unique k-mers | Exact match | 0 | Deterministic |
| Total k-mers | Exact match | 0 | Deterministic |
| Canonical equivalence | Verified | — | Forward == reverse complement |

### 3.4 Diversity Metrics (Track 1 — CPU and GPU)

**Tools**: `bio::diversity` (CPU), `bio::diversity_gpu` (GPU WGSL shaders)
**Validates against**: scikit-bio analytical formulas

#### CPU Validation

| Metric | Formula | Tolerance | Rationale |
|--------|---------|-----------|-----------|
| Shannon | `-Σ p_i ln(p_i)` | `1e-12` | Analytical f64 |
| Simpson | `Σ p_i²` | `1e-12` | Analytical f64 |
| Chao1 | `S + f1(f1-1)/(2(f2+1))` | `1e-12` | Analytical f64 |
| Observed features | Count of non-zero | 0 | Integer |
| Bray-Curtis | `Σ|a-b| / Σ(a+b)` | `1e-15` | Symmetry check |

#### GPU vs CPU Validation

| Metric | Primitive / Shader | Tolerance | Rationale |
|--------|-------------------|-----------|-----------|
| Shannon | `FusedMapReduceF64` (ToadStool) | `1e-10` | Transcendental `log_f64` in WGSL |
| Simpson | `FusedMapReduceF64` (ToadStool) | `1e-6` | Pure arithmetic, generous margin |
| Bray-Curtis | `BrayCurtisF64` (ToadStool, absorbed) | `1e-10` | f64 absolute value and division |
| PCoA eigenvalues | `BatchedEighGpu` (ToadStool) | `1e-6` | Jacobi rotation convergence |
| PCoA distances | Reconstructed from eigenvectors | `1e-6` | Accumulated eigenvector error |
| PCoA proportions | Computed from eigenvalues | `1e-6` | Division by positive sum |

The GPU tolerances are deliberately looser than CPU because:
1. WGSL f64 may use fused multiply-add differently than CPU
2. `log_f64` is a polynomial approximation (not hardware `log`)
3. Workgroup-level reduction ordering may differ from sequential
4. Jacobi eigendecomposition convergence introduces small residuals

### 3.5 PFAS Fragment Screening (Track 2)

**Tool**: `bio::tolerance_search` (ppm/Da tolerance search)
**Validates against**: ismembertolerance (MATLAB equivalent)

| Check | Metric | Tolerance | Rationale |
|-------|--------|-----------|-----------|
| Fragment matches | ppm window | ± 10 ppm | Standard HRMS |
| Known PFAS fragments | Detected count | 0 | Exact match |
| False positive rate | Specificity | 0 | No spurious matches |

---

## 4. Comparison Protocol

1. Run Python / Galaxy reference with documented tool versions
2. Run Rust code on identical input data, record outputs
3. Compare on identical metrics using `validation::check` framework
4. GPU validation: CPU result is the reference, GPU must match within tolerance
5. All checks produce structured pass/fail with explicit exit codes:
   - Exit 0: all checks pass
   - Exit 1: one or more checks fail
   - Exit 2: skip (e.g., `SHADER_F64` not available)
6. Tolerances are centralized in `src/tolerances.rs` with documented rationale
7. Deterministic: same input → identical output, always

---

## 5. Hardware

All experiments run on a single consumer workstation:

| Component | Specification |
|-----------|--------------|
| CPU | Intel i9-12900K (8P+8E, 24 threads) |
| RAM | 64 GB DDR5 |
| GPU | NVIDIA RTX 4070 12 GB (`SHADER_F64` confirmed, Vulkan) |
| OS | Pop!_OS 22.04 (Linux 6.17) |
| Rust | stable (1.82+) |
| wgpu | 0.19 |

---

## 6. Acceptance Criteria

### Phase 2 (CPU): 36/36 checks must pass

| Binary | Checks | Target |
|--------|:------:|--------|
| `validate_fastq` | 9 | All within FastQC tolerance |
| `validate_diversity` | 14 | All within analytical tolerance |
| `validate_mzml` | 7 | All within pyteomics tolerance |
| `validate_pfas` | 6 | All exact match (MS2 parsing + PFAS screening) |
| **Total** | **36** | **All pass** |

Current status: **36/36 pass.**

### Phase 3 (GPU): 17/17 checks must pass

| Binary | Checks | Target |
|--------|:------:|--------|
| `validate_diversity_gpu` | 17 | All within `GPU_VS_CPU_*` tolerances |

Current status: **17/17 pass.** Checks: 3 Shannon + 3 Simpson + 6 Bray-Curtis + 2 PCoA eigenvalues + 1 PCoA distances + 2 PCoA proportions. (Plus 2 capability checks: adapter detection, `SHADER_F64` support.)

---

## 7. Key Differences from hotSpring

| Aspect | hotSpring | wetSpring |
|--------|-----------|-----------|
| Domain | Nuclear physics | Life science + analytical chemistry |
| Phase A/B | Python reproduction → `BarraCUDA` re-execution | Galaxy/Python → Rust sovereign port |
| GPU workload | Large-matrix eigensolve, MD force evaluation | Diversity metrics, distance matrices, signal processing |
| Validation metric | chi2/datum | Pass/fail within documented tolerance |
| Data size | Small (52–2,042 nuclei) | Large (millions of reads, thousands of spectra) |
| Key insight | GPU-resident hybrid beats CPU for matrix physics | Sovereign parsers + GPU diversity validated at f64 |

Both projects prove the same thesis: **sovereign compute on consumer hardware
can replicate institutional results, then exceed them via Rust + GPU.**

---

## 8. Software Versions

| Component | Version | Notes |
|-----------|---------|-------|
| Rust | stable (1.82+) | MSRV set in `Cargo.toml` |
| wgpu | 0.19 | Vulkan backend, `SHADER_F64` |
| Galaxy | 24.1 | Local Docker instance |
| QIIME2 | 2026.1.0 | Galaxy plugin (q2-amplicon-2026.1) |
| asari | 1.13.1 | LC-MS feature extraction |
| `PFΔScreen` | Latest | PFAS non-target screening |
| flate2 | 1.0 | Only runtime dependency (gzip) |
