# Exp215: CPU vs GPU v5 — V66 I/O Evolution Domains

**Track:** cross (GPU)
**Phase:** 66
**Status:** BUILT — awaiting GPU hardware validation
**Binary:** `validate_cpu_vs_gpu_v5_io_evolution`
**Features:** `gpu`

## Purpose

Validates that GPU math produces identical results to CPU for all domains
evolved during the V66 deep audit: byte-native FASTQ, bytemuck nanopore,
streaming MS2, and the full 16S pipeline through GPU dispatch.

## What It Tests

- Byte-native FASTQ → GPU diversity (Shannon, Simpson, observed)
- Quality filter → GPU dereplication
- MS2 streaming → GPU spectral cosine match
- Nanopore signal → GPU stats
- Full 16S pipeline (GPU-chained: FASTQ → quality → diversity → rarefaction)
- GPU dispatch threshold gating

## Key Findings

Binary built and compiles cleanly. Awaiting RTX 4070 GPU availability
for execution. All CPU reference values are established by Exp212.
Expected: identical GPU results within `tolerances::ANALYTICAL_F64`.
