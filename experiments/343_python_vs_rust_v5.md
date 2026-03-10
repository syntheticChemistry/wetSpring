# Exp343: Python vs Rust v5

**Date:** March 2026
**Track:** 6 (Anaerobic QS / ADREC) — Chain
**Binary:** `benchmark_python_vs_rust_v5`
**Status:** PASS (13 checks)

---

## Hypothesis

What this tier validates: Rust math matches Python/SciPy baselines for Track 6 anaerobic models (Gompertz, Monod, Haldane, diversity).

## Method

Run identical computations in Python (SciPy/NumPy) and Rust. Compare outputs at machine precision and measure execution time. Verify bit-identical or numerically equivalent results across all model types.

## Results

All 13 checks PASS. See `cargo run --release --bin benchmark_python_vs_rust_v5` for full output.

## Key Finding

Rust is faster than interpreted Python, bit-identical results. The anaerobic model implementations are validated against the scientific Python ecosystem.

## Modules Validated

- Gompertz kinetics
- Monod kinetics
- Haldane kinetics
- Diversity indices
- Python/SciPy baseline equivalence
