# Exp114: ESN QS Phase Classifier → NPU

**Status**: PASS (13/13)  
**Phase**: 33 — NPU Reservoir Deployment  
**Binary**: `validate_npu_qs_classifier`  
**Depends on**: Exp108 (Vibrio QS Parameter Landscape)

## Purpose

Train an Echo State Network on QS ODE parameter sweep data, quantize the
readout to int8, and validate that NPU-quantized classification preserves
f64 argmax fidelity across the parameter landscape.

## Data

- 512 training samples from QS ODE simulations (varying µ_max, k_ai_prod,
  k_hapr_ai, k_dgc_basal, k_bio_max)
- 256 test samples from non-overlapping parameter space
- 3-class target: biofilm / planktonic / intermediate (from ODE steady state)

## Architecture

- ESN: 5 input → 200 reservoir (ρ=0.9, c=0.1, α=0.3) → 3 output
- Training: diagonal ridge regression (λ=10⁻⁶)
- Quantization: affine int8 on W_out → NpuReadoutWeights

## Results

| Metric | Value |
|--------|-------|
| F64 accuracy | 69.5% |
| NPU int8 accuracy | 69.5% |
| F64 ↔ NPU agreement | 100% (256/256) |
| NPU energy (est.) | 0.000089 J / 256 classifications |
| GPU energy (est.) | 0.80 J / 256 classifications |
| Energy ratio | ~9,000× reduction |
| NPU throughput | 1,538 Hz |

## Key Findings

1. **Perfect quantization fidelity**: int8 readout preserves argmax in 100%
   of test cases — the quantization error never changes the classification.
2. **~9,000× energy reduction**: NPU inference vs GPU ODE sweep.
3. **Real-time viability**: 1,538 Hz far exceeds bioreactor monitoring needs
   (typically 0.1–1 Hz sample rates).

## Reproduction

```bash
cargo run --release --bin validate_npu_qs_classifier
```
