# Exp083: Taxonomy NPU Int8 Quantization

**Status**: COMPLETE
**Date**: 2026-02-22
**Module**: `barracuda/src/bio/taxonomy.rs`
**New tests**: 3

## Purpose

Adds int8 affine quantization to the Naive Bayes taxonomy classifier,
producing NPU-compatible weight buffers for BrainChip AKD1000 FC-layer
inference. Validates that quantized classification preserves argmax
agreement with full f64 precision.

## APIs Added

| Type / Method | Description |
|--------------|-------------|
| `NpuWeights` struct | Int8 buffers: `weights_i8`, `priors_i8`, scale, zero_point |
| `to_int8_weights()` | Quantize `dense_log_probs` → `NpuWeights` |
| `classify_quantized(seq)` | Int8 scoring path (NPU simulation) |

## Quantization Scheme

Affine int8 mapping across the full log-probability range:

```
scale      = (max - min) / 255
zero_point = min
q          = round((value - zero_point) / scale) - 128
```

Dequantization: `real = (q + 128) × scale + zero_point`

For k=8, the weight matrix is `n_taxa × 65,536` int8 values.
The log-probability range is typically [-30, -1], giving ~0.11 per quantization step.

## NPU Dispatch Pattern

```
CPU: NaiveBayesClassifier::train() → to_int8_weights()
     ↓ export buffers
NPU: FC layer: Q_int8 × W_int8 + bias_int8 → argmax
     ↓ result
CPU: taxon_labels[argmax] → Lineage
```

The AKD1000 NPU supports int8 FC layers with batch inference (max_batch=8),
matching the forge dispatch capability `QuantizedInference { bits: 8 }`.

## Tests

| Test | What it validates |
|------|-------------------|
| `int8_quantization_round_trip` | Buffer sizes correct, scale positive |
| `quantized_classification_parity` | Int8 argmax matches f64 for both taxa |
| `npu_buffer_sizes` | k=8 produces 65,536 × n_taxa weight buffer |

## Tier Promotion

taxonomy: **B → A/NPU** (NPU-ready, pending AKD1000 FC dispatch integration)
