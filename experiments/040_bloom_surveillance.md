# Exp040 — Bloom Event Detection & Surveillance

**Date:** February 20, 2026
**Status:** COMPLETE — 15/15 checks PASS
**Track:** 1a (Algal Biofuel, 16S Microbiome)
**Proxy for:** Paper #14, Smallwood (raceway metagenomic surveillance)

---

## Objective

Validate bloom event detection using diversity collapse signatures:
Shannon drop, Gini-Simpson drop, Berger-Parker dominance spike, and
Bray-Curtis shift. Uses PRJNA1224988 (175-sample cyanobacterial bloom
time series) as proxy for Smallwood's raceway surveillance.

## Data Source

- **BioProject:** PRJNA1224988
- **Organism:** Multi-year cyanobacterial bloom across oligotrophic lakes
- **Samples:** 175 (16S time series)
- **Download:** `scripts/download_public_data.sh --bloom-ts`

## Detection Algorithm

A bloom is detected when Shannon diversity drops below mean - 2σ of the
pre-bloom baseline. Supporting indicators:
- Berger-Parker dominance > 0.3 (single taxon >30% of community)
- Gini-Simpson < 0.3 (dominance concentration)
- Bray-Curtis shift > 0.5 (major community restructuring)
- Pielou evenness collapse

## Validation Checks (15/15)

| # | Check | Expected | Result |
|---|-------|----------|--------|
| 1 | Shannon(even) ≈ 2.98 | ✓ | PASS |
| 2 | Evenness(even) > 0.9 | ✓ | PASS |
| 3 | Dominance(even) < 0.1 | ✓ | PASS |
| 4 | Shannon drops during bloom | ✓ | PASS |
| 5 | Evenness drops during bloom | ✓ | PASS |
| 6 | Dominance spikes during bloom | ✓ | PASS |
| 7 | Gini-Simpson < 0.3 during bloom | ✓ | PASS |
| 8 | BC(pre,bloom) > 0.5 | ✓ | PASS |
| 9 | BC(self) = 0 | 0.0 | PASS |
| 10 | Shannon recovers post-bloom | ✓ | PASS |
| 11 | BC(pre,recovery) < 0.1 | ✓ | PASS |
| 12 | Bloom detected (Shannon < mean-2σ) | ✓ | PASS |
| 13 | Shannon deterministic | exact | PASS |
| 14 | Simpson deterministic | exact | PASS |
| 15 | BC deterministic | exact | PASS |

## Key Findings

- Bloom event produces Shannon collapse from 2.98 → 0.29 (10× reduction)
- Berger-Parker dominance spikes from 0.06 → 0.96 (near-monoculture)
- Recovery to pre-bloom diversity levels is rapid once bloom subsides
- Bray-Curtis shift during bloom: 0.91 (near-total community replacement)

## GPU Promotion Path

Multi-metric diversity surveillance: batch Shannon, Simpson, BC, evenness
in a single dispatch per timepoint. `diversity_gpu` + `streaming_gpu`
modules provide the foundation for real-time monitoring.

## Run

```bash
cargo run --bin validate_bloom_surveillance
python3 scripts/bloom_surveillance_baseline.py
```
