# Experiment 196c: NPU Int8 Quantization Pipeline

**Date:** February 27, 2026
**Phase:** 61
**Track:** Field Genomics (Sub-thesis 06)
**Status:** PASS (13/13 checks)
**Binary:** `validate_nanopore_int8_quantization`

---

## Objective

Validate the complete community profile → int8 quantization → ESN classifier →
NPU inference pipeline for field deployment. This closes the software path from
nanopore reads (Exp196a-b) through community analysis to neuromorphic edge
classification (bloom / healthy / stressed / AMR / PFAS).

## Background

The AKD1000 NPU operates on int8 arithmetic at 18.8K inferences/sec (Exp194).
Community profiles from the 16S pipeline are f64 abundance vectors. The
quantization pipeline must:

1. Map f64 community profiles to int8 without losing regime information
2. Preserve ESN classification agreement (bloom vs healthy vs stressed)
3. Maintain power budget (< 10 mW) for coin-cell field deployment

## Validation Sections

| Section | Checks | What It Validates |
|---------|:------:|-------------------|
| S1: f64→int8 fidelity | 4 | Affine quantization round-trip error, dynamic range, outlier clipping |
| S2: ESN classification | 4 | 3-class regime agreement (f64 vs int8), confidence correlation |
| S3: Bloom detection | 3 | Pre-bloom / active / post-bloom from quantized community profiles |
| S4: Power/latency | 2 | Estimated inference latency, energy per classification |

**Total:** 13/13 PASS

## Tolerance Constants

- `NPU_INT8_COMMUNITY` — maximum abs error for int8 community quantization
- `FIELD_ANDERSON_REGIME` — regime classification tolerance for field conditions

## Key Findings

- Affine int8 quantization preserves >95% of community profile information
- ESN classification agreement between f64 and int8 is 100% for 3-class regime
  (consistent with Exp116, Exp194)
- Bloom detection latency: ~53 µs per classification (from Exp194 hardware data)
- Power budget: 1.4 µJ/inference — coin-cell CR2032 → 11 years at 1 Hz

## Connection to Hardware

This experiment validates the SOFTWARE path. When MinION hardware arrives:
- Exp196a's parser reads real POD5 files
- Exp196b's pipeline processes real reads
- This experiment's quantization feeds real community profiles to the AKD1000
- The full loop closes: MinION → BarraCUDA → NPU → alert
