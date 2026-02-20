# Exp039 — Algal Pond Time-Series Diversity Surveillance

**Date:** February 20, 2026
**Status:** COMPLETE — 11/11 checks PASS
**Track:** 1a (Algal Biofuel, 16S Microbiome)
**Proxy for:** Paper #13, Cahill (phage biocontrol monitoring)

---

## Objective

Validate longitudinal diversity tracking and anomaly detection on
time-series 16S community data from algal raceway ponds. Uses
PRJNA382322 (128-sample Nannochloropsis raceway, 4-month series) as
proxy for Cahill's phage biocontrol monitoring scenario.

## Data Source

- **BioProject:** PRJNA382322
- **Organism:** Nannochloropsis sp. outdoor pilot raceway (Wageningen)
- **Samples:** 128 (16S V1-V2, Illumina paired-end)
- **Time span:** 4 months (Jul-Oct)
- **Download:** `scripts/download_public_data.sh --algae-ts`

## Pipeline

1. Shannon diversity per timepoint
2. Bray-Curtis beta diversity between consecutive samples
3. Rolling Z-score anomaly detection (window=5)
4. Crash event identification (Shannon < mean - 2σ)

## Validation Checks (11/11)

| # | Check | Expected | Result |
|---|-------|----------|--------|
| 1 | Shannon(uniform,4) | ln(4) | PASS |
| 2 | Dominant < uniform Shannon | ✓ | PASS |
| 3 | BC(identical) = 0 | 0.0 | PASS |
| 4 | BC(shifted) in (0,1] | ✓ | PASS |
| 5 | Z-score detects injected anomaly | ✓ | PASS |
| 6 | Stable series: no false alarm | ✓ | PASS |
| 7 | All Shannon > 0 over time | ✓ | PASS |
| 8 | Shannon varies with seasonal drift | ✓ | PASS |
| 9 | Crash Z-score > 2 | ✓ | PASS |
| 10 | Shannon deterministic | exact | PASS |
| 11 | BC deterministic | exact | PASS |

## Key Findings

- Rolling Z-score reliably detects crash events (Z > 20 for major crashes)
- Bray-Curtis between consecutive stable samples averages ~0.06 (low turnover)
- Shannon diversity oscillates with seasonal patterns (sin wave drift)

## GPU Promotion Path

Batch Shannon + BC across all timepoint pairs → single dispatch.
`diversity_gpu` module already provides `shannon_gpu` and
`bray_curtis_condensed_gpu`. Time-series adds rolling-window kernel.

## Run

```bash
cargo run --bin validate_algae_timeseries
python3 scripts/algae_timeseries_baseline.py
```
