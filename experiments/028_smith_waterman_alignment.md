# Experiment 028 — Smith-Waterman Local Alignment

**Date:** 2026-02-20
**Status:** COMPLETE
**Track:** 1b (Comparative Genomics)
**Faculty:** Liu (CSE, MSU)

---

## Objective

Implement and validate Smith-Waterman local sequence alignment with affine gap
penalties. This is a prerequisite for SATé (Liu 2009), metagenomic placement
(Alamin & Liu 2024), and a well-known GPU parallelization target (anti-diagonal
wavefront parallelism).

## Test Cases

| Case | Query | Target | Score |
|------|-------|--------|-------|
| Identical 8bp | ACGTACGT | ACGTACGT | 16 |
| Mismatch | ACGT | ACTT | 5 |
| Gap | ACGTACGT | ACGACGT | 10 |
| Local (embedded) | XXXACGTACGTXXX | ACGTACGT | 16 |
| No match | AAAA | CCCC | 0 |
| 16S fragment (40bp) | — | — | 74 |

## Validation Binary

```bash
cargo run --bin validate_alignment
```

## GPU Promotion Path

Smith-Waterman anti-diagonal wavefront is embarrassingly parallel — each
anti-diagonal can be computed in one GPU dispatch. ToadStool's `GemmF64` or a
custom wavefront shader handles this. Batch pairwise scoring (N sequences →
N×N matrix) is the high-value GPU target.
