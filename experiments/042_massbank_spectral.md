# Exp042 — MassBank PFAS Spectral Matching Validation

**Date:** February 20, 2026
**Status:** COMPLETE — 9/9 checks PASS
**Track:** 2 (Analytical Chemistry / Mass Spectrometry)
**Proxy for:** Paper #21, Jones PFAS mass spectrometry detection

---

## Objective

Validate cosine similarity spectral matching on PFAS-like mass spectra,
extending to MassBank reference spectra (10K+ entries) when downloaded.
Tests the `spectral_match` module's ability to correctly identify:
- Self-matches (cosine = 1.0)
- Instrument variation tolerance (cosine > 0.99)
- Related PFAS family compounds (cosine > 0.3)
- Unrelated compounds (cosine < 0.3)

## Data Source

- **MassBank:** GitHub `MassBank/MassBank-data` (PFAS reference spectra)
- **Synthetic:** PFOS, PFOA, and caffeine mock spectra with realistic m/z patterns
- **Download:** `scripts/download_public_data.sh --massbank`

## Validation Checks (9/9)

| # | Check | Expected | Result |
|---|-------|----------|--------|
| 1 | cosine(PFOS, PFOS) | 1.000000 | PASS |
| 2 | cosine(PFOS, shifted) | 0.999662 | PASS |
| 3 | near_match > 0.99 | ✓ | PASS |
| 4 | family_match > 0.3 | ✓ | PASS |
| 5 | unrelated_match < 0.3 | ✓ | PASS |
| 6 | cosine symmetry | exact | PASS |
| 7 | diagonal all 1.0 | ✓ | PASS |
| 8 | all pairwise >= 0 | ✓ | PASS |
| 9 | cosine deterministic | exact | PASS |

## Key Findings

- Self-match gives exact 1.0 (bit-identical spectra)
- Instrument variation (±0.1 Da shift) gives 0.9997 similarity
- PFOS vs PFOA: 0.537 (partial fragment overlap, not full family match)
- PFOS vs caffeine: 0.0 (no matching peaks within 0.5 Da tolerance)
- All metric properties (symmetry, non-negativity, identity) hold exactly

## GPU Promotion Path

`spectral_match_gpu::cosine_vs_library_gpu` already exists for batch
library screening. Pairwise matrix computation parallelizes trivially.

## Run

```bash
cargo run --bin validate_massbank_spectral
python3 scripts/massbank_spectral_baseline.py
```
