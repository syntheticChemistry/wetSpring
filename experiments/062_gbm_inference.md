# Exp062: Gradient Boosting Machine (GBM) Inference

**Date:** February 20, 2026
**Status:** COMPLETE — 16/16 checks PASS
**Track:** cross

## Purpose

Prove that GBM classification (binary + multi-class) can be performed in
pure Rust. GBM differs from RF in that trees are sequential (each corrects
the previous residual), so the math is fundamentally different.

## Design

### Binary GBM: `GbmClassifier`

1. Initialize cumulative score from `initial_prediction` (log-odds baseline)
2. For each tree: `score += learning_rate × tree.predict(features)`
3. Final: `class = sigmoid(score) >= 0.5 ? 1 : 0`

### Multi-Class GBM: `GbmMultiClassifier`

1. K sets of trees (one per class)
2. Each class chain produces a raw score
3. Softmax normalization → probabilities → argmax

### Components

| Module | Description |
|--------|-------------|
| `GbmTree` | Regression tree (f64 residuals, not class labels) |
| `GbmClassifier` | Binary classifier with sigmoid |
| `GbmMultiClassifier` | Multi-class with softmax |
| `GbmPrediction` | class, probability, raw_score |
| `GbmMultiPrediction` | class, probabilities[], raw_scores[] |

## Validation Results

**validate_barracuda_cpu_v5:** 16/16 checks PASS

Binary GBM (3 stumps):
- Negative classification: raw_score = −0.23, prob < 0.5, class 0
- Positive classification: raw_score = 0.33, prob > 0.5, class 1
- Sigmoid correctness: sigmoid(0.33) = 0.5818
- Batch prediction: [0, 1]
- Initial bias: log-odds = 5.0 → prob > 0.99

Multi-Class GBM (3 classes):
- Softmax sums to 1.0
- Correct class selection per score distribution

## GPU Strategy

GBM is inherently sequential across boosting rounds (tree N depends on
cumulative sum from trees 0..N-1). However, batch samples can be dispatched
in parallel within each round, and multi-class chains can run concurrently.
Candidate for ToadStool as `GbmBatchInferenceGpu`.

## Conclusion

GBM inference is pure sovereign math, proven correct for both binary and
multi-class modes. Combined with Exp061 (RF), wetSpring now covers the
two most important ensemble ML methods in pure Rust.
