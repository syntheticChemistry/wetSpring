# Exp346: metalForge v18

**Date:** March 2026
**Track:** 6 (Anaerobic QS / ADREC) — Chain
**Binary:** `validate_metalforge_v18`
**Status:** PASS (16 checks)

---

## Hypothesis

What this tier validates: cross-substrate independence—CPU = GPU = NPU for all Track 6 anaerobic math.

## Method

Run identical computations on CPU, GPU, and NPU (where available). Compare outputs at floating-point tolerance. Confirm hardware-agnostic numerical equivalence across all substrates.

## Results

All 16 checks PASS. See `cargo run --release --bin validate_metalforge_v18` for full output.

## Key Finding

Final tier validation: all hardware produces identical results. The anaerobic math stack is truly substrate-independent.

## Modules Validated

- CPU substrate
- GPU substrate
- NPU substrate
- Cross-hardware numerical equivalence
- Track 6 anaerobic math (all domains)
