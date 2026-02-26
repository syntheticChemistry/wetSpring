# Exp187: DF64 Anderson at L=24+ — Extended Precision Large Lattice

**Date:** February 26, 2026
**Phase:** V55 — Science extensions
**Binary:** `validate_df64_anderson` (planned)
**Command:** `cargo run --release --features gpu --bin validate_df64_anderson`
**Status:** Protocol defined, implementation pending
**Depends on:** Exp184b (GPU L=14-20), hotSpring DF64 validation

## Purpose

Leverage hotSpring's DF64 (double-float) discovery to perform Anderson
localization analysis at lattice sizes L ≥ 24 with enhanced numerical
precision. At L=24, the Anderson matrix is 13,824 × 13,824 — Lanczos
convergence requires ~300+ iterations, and eigenvalue spacing near the
band center becomes very small, making DF64 precision valuable.

## Physics

At large L, finite-size corrections to W_c diminish, providing:
- More precise W_c estimate (current: 16.26 ± 0.95 from L=6-12)
- Refined critical exponent ν (literature: 1.57 ± 0.02)
- Test of single-parameter scaling theory at unprecedented L

DF64 provides ~30 digits of precision on FP32 cores at 2× FP32
throughput (demonstrated by hotSpring). For Lanczos, this means:
- More accurate tridiagonal matrix entries
- Better eigenvalue separation near band center
- Reduced ghost eigenvalues from Lanczos breakdown

## Target Lattice Sizes

| L | N = L³ | Matrix NNZ | Lanczos iters | Est. time (GPU) |
|---|--------|-----------|---------------|-----------------|
| 24 | 13,824 | ~96,768 | 300 | ~5 min |
| 28 | 21,952 | ~153,664 | 400 | ~15 min |
| 32 | 32,768 | ~229,376 | 500 | ~40 min |

## Validation Checks

### S1: DF64 vs F64 Parity (L=14 reference)
- [ ] DF64 eigenvalues match F64 to 12+ digits at L=14
- [ ] Level spacing ratio r agrees to ANALYTICAL_F64 tolerance
- [ ] DF64 Lanczos converges in same iteration count as F64

### S2: Large Lattice W_c Refinement
- [ ] W_c(L=24) within 2% of W_c(L=12) from Exp150
- [ ] W_c(L=28) within 1.5% of W_c(L=24) (convergence)
- [ ] Finite-size corrections scale as L^(-1/ν) with ν ≈ 1.57

### S3: Critical Exponent Refinement
- [ ] ν estimate from L=14-32 consistent with 1.57 ± 0.02
- [ ] Scaling collapse cost decreases with increasing L range
- [ ] Best ν from combined L=6-32 more precise than L=6-12

## Dependencies

- ToadStool DF64 Lanczos kernel (may need evolution request)
- hotSpring DF64 arithmetic validated (Exp: plasma MD, lattice QCD)
- barracuda::spectral DF64 variant (to be evolved from existing f64)

## Compute Estimate

- L=24 sweep (15 W points × 16 realizations): ~80 min GPU
- L=28 sweep: ~4 hours GPU
- L=32 sweep: ~10 hours GPU
- Best hardware: Northgate RTX 5090 or biomeGate RTX 4070

## Provenance

| Item | Value |
|------|-------|
| DF64 method | Dekker (1971), double-float arithmetic |
| hotSpring validation | DF64 plasma MD + lattice QCD reproduction |
| Target W_c | 16.54 ± 0.10, Slevin & Ohtsuki, PRL 82 (1999) |
| Target ν | 1.571 ± 0.004, Rodriguez et al., PRB 84 (2011) |
