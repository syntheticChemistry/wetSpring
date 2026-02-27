# Exp212: BarraCuda CPU v12 — Post-Audit Math Fidelity

**Track:** cross
**Phase:** 66
**Status:** COMPLETE — 55/55 checks PASS
**Binary:** `validate_barracuda_cpu_v12`
**Features:** none (CPU only)

## Purpose

Validates that the V66 deep audit (byte-native FASTQ, bytemuck nanopore,
streaming APIs, tolerance centralization, `total_cmp` migration) preserves
end-to-end math fidelity through the CPU pipeline.

## What It Tests

- I/O → diversity pipeline (byte-native FASTQ → Shannon/Simpson/observed)
- I/O → quality pipeline (FASTQ records → Phred → quality metrics)
- QS ODE integration (Waters 2005 bistable parameters)
- Dereplication after quality filter
- Merge pairs after adapter trim
- Nanopore signal → calibrated stats
- MS2 streaming → spectral cosine match
- Named tolerance enforcement (92 constants)
- CPU fallback parity with V11

## Key Findings

All 55 checks pass. The V66 I/O evolution does not introduce any numeric
drift in the CPU math pipeline. The `total_cmp` migration produces identical
sort orders to the previous `partial_cmp().unwrap()` pattern for non-NaN data.
