# Exp344: CPU vs GPU v10

**Date:** March 2026
**Track:** 6 (Anaerobic QS / ADREC) — Chain
**Binary:** `validate_cpu_vs_gpu_v10`
**Status:** PASS (14 checks)

---

## Hypothesis

What this tier validates: all Track 6 math produces identical results on CPU and GPU substrates.

## Method

Execute the same computations on CPU and GPU backends. Compare outputs at floating-point tolerance. Ensure no substrate-specific divergence in numerical results.

## Results

All 14 checks PASS. See `cargo run --release --bin validate_cpu_vs_gpu_v10` for full output.

## Key Finding

Math is truly portable across compute substrates. Users can choose CPU or GPU without changing results.

## Modules Validated

- CPU backend
- GPU backend
- Cross-substrate numerical equivalence
- Track 6 anaerobic math (all domains)
