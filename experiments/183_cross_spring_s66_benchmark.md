# Exp183: Cross-Spring Evolution S66 Benchmark

**Date:** February 2026
**Track:** cross-spring
**Paper:** N/A (benchmark)
**Binary:** `benchmark_cross_spring_s65` (name retained for compatibility)
**Status:** PASS (benchmark timing)

---

## Hypothesis

This experiment validates that cross-spring primitive delegation and S66 rewire produce correct results, and benchmarks the performance of the barracuda primitive stack.

## Method

Validation runs the cross-spring benchmark, exercising barracuda primitives with S66 rewire validation. Timing measurements document delegation overhead and primitive performance.

## Results

Benchmark completes successfully. See `cargo run --bin benchmark_cross_spring_s65` for full output.

## Key Finding

Cross-spring primitive delegation benchmark, S66 rewire validation.

## Modules Validated

barracuda primitives.
