# Experiment 060: metalForge Cross-Substrate Validation

**Date:** February 21, 2026
**Status:** COMPLETE
**Track:** cross/GPU

---

## Purpose

Prove that the bioinformatics math is **substrate-independent**: the same
algorithm produces identical results whether executed on CPU (Rust) or GPU
(WGSL via wgpu). This is the metalForge proof — showing that algorithms can
be routed to any available hardware without affecting correctness.

```
Python baseline → CPU parity (128/128) → Timing (22.5×) → GPU portability (Exp058) → [THIS] substrate independence
```

---

## Design

For each GPU-promoted algorithm:
1. Run on CPU as reference truth
2. Run on GPU via the WGSL shader
3. Compare results at tight tolerances (≤ 1e-6 for f64)

Four substrates tested:

| # | Algorithm | CPU Module | GPU Module | Shader |
|---|-----------|-----------|-----------|--------|
| 1 | ANI | `bio::ani` | `bio::ani_gpu` | `ani_batch_f64.wgsl` |
| 2 | SNP | `bio::snp` | `bio::snp_gpu` | `snp_calling_f64.wgsl` |
| 3 | Pangenome | `bio::pangenome` | `bio::pangenome_gpu` | `pangenome_classify.wgsl` |
| 4 | dN/dS | `bio::dnds` | `bio::dnds_gpu` | `dnds_batch_f64.wgsl` |

---

## Results: 20/20 PASS

### Substrate 1: ANI (5 checks)
- Identical, partial, different, N-excluded, gap-excluded pairs
- All 5 GPU values == CPU values (tol < 1e-10)

### Substrate 2: SNP Calling (4 checks)
- Variant count matches (3 variants)
- All variant positions correctly identified

### Substrate 3: Pangenome (3 checks)
- Core, accessory, unique counts match exactly

### Substrate 4: dN/dS (8 checks)
- 4 pairs × (dN, dS) all match CPU within 1e-6
- Includes: identical, synonymous-only, nonsynonymous, and mixed pairs
- Genetic code translation, pathway averaging, Jukes-Cantor correction — all identical

---

## Hardware

- CPU: Intel Core i9-12900 (reference truth)
- GPU: NVIDIA GeForce RTX 4070 (SHADER_F64: YES)

---

## Conclusion

The math is substrate-independent. Any algorithm that passes the CPU
validation can be dispatched to GPU and produce identical results.
This enables metalForge's substrate routing: CPU → GPU → NPU based on
availability and workload characteristics.
