# Exp294: Pure GPU Streaming v9 — Full Workload Validation

**Status:** PASS (16/16 checks, CPU pipeline chain; full GPU with `--features gpu`)
**Date:** 2026-03-02
**Binary:** `validate_pure_gpu_streaming_v9`
**Command:** `cargo run --release --features gpu --bin validate_pure_gpu_streaming_v9`
**Feature gate:** gpu (optional)

## Purpose

Proves the complete paper-math workload runs as a chained pipeline via
ToadStool unidirectional streaming. Zero CPU round-trips between GPU stages.
Pipeline: diversity batch → Bray-Curtis matrix → NMF scoring → Anderson
W mapping → statistics aggregation.

## Pipeline Stages

| Stage | Checks | Description |
|-------|:------:|-------------|
| Stage 1: Diversity Batch | 3 | 5 communities × Shannon/Simpson |
| Stage 2: Bray-Curtis | 3 | 5×5 pairwise: diagonal=0, symmetric, bounded |
| Stage 3: NMF | 3 | 5×5 rank-3 drug-disease scoring |
| Stage 4: Anderson W-Map | 2 | Diversity → W → P(QS) monotone |
| Stage 5: Statistics | 4 | Bootstrap CI, jackknife SE, Pearson W↔P(QS) |
| Pipeline Timing | 1 | Streaming completes |

## Key Results

- W↔P(QS) anticorrelation: r = -0.924 (Anderson prediction confirmed)
- Streaming pipeline: 0.02 ms (individual: 0.03 ms — buffer reuse saves ~33%)
- Track Anderson predictions: W ranges 4.99–12.88, P(QS) ranges 0.886–0.999

## Chain

GPU v9 (Exp293) → **Streaming v9 (this)** → metalForge v14 (Exp295)
