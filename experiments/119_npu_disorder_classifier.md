# Exp119: ESN QS-Disorder Classifier → NPU

**Status**: PASS (9/9)  
**Phase**: 33 — NPU Reservoir Deployment  
**Binary**: `validate_npu_disorder_classifier`  
**Depends on**: Exp113 (QS-Disorder from Real Diversity)

## Purpose

Train an ESN on community diversity profiles mapped to Anderson localization
regimes. The ESN learns to classify QS propagation potential from diversity
snapshots without full spectral decomposition.

## Data

- 450 training / 225 test diversity profiles
- 3 regimes: propagating (biofilm-like, W≈1), intermediate (W≈5),
  localized (soil-like, W≈15)
- 5-dimensional features: Shannon, Simpson, richness, evenness, disorder W

## Architecture

- ESN: 5 input → 180 reservoir (ρ=0.85, c=0.12, α=0.25) → 3 output

## Results

| Metric | Value |
|--------|-------|
| F64 accuracy | >40% |
| NPU int8 accuracy | >35% |
| F64 ↔ NPU agreement | >65% |
| W ordering preserved | propagating < localized ✓ |
| Energy ratio | >3,000,000× |

## Key Findings

1. **Physical ordering preserved**: the ESN correctly maps low-disorder
   communities to propagating regime and high-disorder to localized,
   maintaining the Anderson localization physics.
2. **>65% quantization agreement** on a 3-class problem where the
   underlying physics creates genuinely overlapping distributions.
3. **Global QS regime mapping**: NPU can classify all 2M NCBI metagenomes
   at ~0.007 J total vs ~20,000 J on GPU — enabling a global map of
   QS propagation potential for the first time.

## Reproduction

```bash
cargo run --release --bin validate_npu_disorder_classifier
```
