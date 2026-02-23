# Exp123: Temporal ESN Bloom Cascade

**Status:** PASS (9 checks)
**Binary:** `validate_temporal_esn_bloom`
**Features:** CPU-only
**Date:** 2026-02-23

## Purpose

Compares stateful (memory-carrying) vs stateless ESN for temporal bloom phase detection across multi-window trajectories. Stateful ESN preserves reservoir state across windows within a trajectory; stateless resets each window. Validates that temporal context enables earlier pre-bloom detection.

## Design

Generate multi-window bloom trajectories (normal to pre-bloom to active to post-bloom to normal). Stateful ESN: carry reservoir state between consecutive windows; train on full trajectory, ridge-regress on states. Stateless baseline: reset reservoir per window (Exp118 approach). Compare detection latency and pre-bloom recall. NPU int8 quantization tested for both paths.

## Data Source

Synthetic multi-window diversity trajectories (50–100 windows per series). Features: Shannon, Simpson, richness, evenness, Bray-Curtis delta, temperature. Labeled phases for normal, pre-bloom, active-bloom, post-bloom.

## Key Results

- Stateful ESN detects pre-bloom 2–4 windows earlier than stateless.
- Pre-bloom recall: stateful > stateless; normal specificity > 90% for both.
- NPU int8 quantization preserves stateful advantage.
- Stateful-NPU recall exceeds stateless-f64 recall (memory + int8 beats f64 + no memory).

## Reproduction

```bash
cargo run --release --bin validate_temporal_esn_bloom
```
