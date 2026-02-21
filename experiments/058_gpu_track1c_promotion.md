# Experiment 058: GPU Track 1c Promotion — ANI + SNP + Pangenome + dN/dS

**Date:** February 21, 2026
**Status:** COMPLETE
**Track:** GPU

---

## Purpose

Prove that the Track 1c bioinformatics math is truly portable from CPU to
GPU. Four WGSL compute shaders dispatch the same algorithms that run in
the CPU `bio::ani`, `bio::snp`, `bio::pangenome`, and `bio::dnds` modules —
validating CPU↔GPU parity for population genomics workloads.

This is the GPU half of the BarraCUDA portability proof:

```
Python baseline → CPU parity (Exp057, 23 domains) → [THIS] GPU promotion → ToadStool absorption
```

---

## WGSL Shaders Written

| Shader | Strategy | Threads | Transcendentals |
|--------|----------|---------|-----------------|
| `ani_batch_f64.wgsl` | One thread per sequence pair | N pairs | None (integer counting + f64 division) |
| `snp_calling_f64.wgsl` | One thread per alignment position | L positions | None (integer counting + f64 division) |
| `pangenome_classify.wgsl` | One thread per gene cluster | G genes | None (integer counting) |
| `dnds_batch_f64.wgsl` | One thread per coding sequence pair | N pairs | `log()` for Jukes-Cantor correction |

All shaders follow the Write → Absorb → Lean pattern: local wetSpring
shaders that are ToadStool absorption candidates.

---

## GPU Modules

| Module | CPU Module | API |
|--------|-----------|-----|
| `bio::ani_gpu::AniGpu` | `bio::ani` | `batch_ani(&[(seq_a, seq_b)])` |
| `bio::snp_gpu::SnpGpu` | `bio::snp` | `call_snps(&[&[u8]])` |
| `bio::pangenome_gpu::PangenomeGpu` | `bio::pangenome` | `classify(&[u8], n_genes, n_genomes)` |
| `bio::dnds_gpu::DnDsGpu` | `bio::dnds` | `batch_dnds(&[(seq_a, seq_b)])` |

---

## Validation Results: 27/27 PASS

### Section 1: GPU ANI Batch (7 checks)
- Identical sequences → 1.0 (matches CPU)
- Completely different → 0.0 (matches CPU)
- Half-match → 0.5 (matches CPU)
- Gap-excluded → ANI=1.0 (matches CPU)
- N-excluded → correct (matches CPU)
- Aligned counts exact match with CPU

### Section 2: GPU SNP Calling (5 checks)
- Variant count matches CPU
- Variant positions match CPU
- Depths match CPU at variant sites
- Alt frequencies match CPU (< 1e-6 tolerance)
- Non-variant positions have alt_freq = 0

### Section 3: GPU Pangenome Classification (6 checks)
- Core/accessory/unique counts match CPU exactly
- Genome presence counts match CPU
- Core genes verified: count = n_genomes

### Section 4: GPU dN/dS Batch — Nei-Gojobori 1986 (9 checks)
- Identical sequences → dN=0, dS=0 (matches CPU)
- Synonymous-only pair → dN=0, dS>0 (matches CPU, tol < 1e-4)
- Nonsynonymous pair → dN>0 (matches CPU, tol < 1e-4)
- Mixed-change pair → both dN and dS match CPU
- Full genetic code translation, codon site counting, pathway-averaged
  differences, and Jukes-Cantor correction all running on GPU

---

## Hardware

- NVIDIA GeForce RTX 4070
- SHADER_F64: YES
- dN/dS shader uses `log()` polyfill for Jukes-Cantor on NVVM drivers

---

## Data

All synthetic — no external data dependencies.
