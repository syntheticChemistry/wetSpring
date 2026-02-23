# Exp063: GPU Random Forest Batch Inference

**Binary**: `validate_gpu_rf`
**Status**: COMPLETE
**Track**: 1b (ML Ensembles)
**Checks**: 13

## Purpose

Validate GPU-accelerated Random Forest batch inference via ToadStool's
`RfBatchInferenceGpu`. Proves that GPU classification matches CPU reference
truth for ensemble voting across 6 feature vectors × 5 decision trees.

## Design

- **Input**: 6 synthetic feature vectors (3–5 features)
- **Model**: 5 decision trees with known split thresholds
- **Comparison**: CPU `random_forest::classify_batch` vs GPU `RfBatchInferenceGpu`
- **Metric**: Exact classification parity (argmax match on all samples)

## Results

- CPU and GPU produce identical class labels for all 6 inputs
- SoA (Structure of Arrays) layout validated for GPU dispatch
- Majority vote ensemble logic matches across substrates

## Reproduction

```bash
cargo run --features gpu --release --bin validate_gpu_rf
```
