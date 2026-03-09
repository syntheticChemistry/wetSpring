# Exp334: Science-to-Viz Pipeline

**Date:** March 9, 2026
**Binary:** `validate_science_viz_pipeline`
**Crate:** `wetspring-forge`
**Result:** 34/34 PASS

## Scope

End-to-end validation of the science compute → visualization pipeline:

| Domain | Checks |
|--------|:------:|
| P1 Ecology end-to-end | 5 |
| P2 Full pipeline scenario | 3 |
| P3 IPC diversity + viz flag | 3 |
| P4 Pangenome science→viz | 3 |
| P5 HMM science→viz | 5 |
| P6 Stochastic science→viz | 4 |
| P7 NMF science→viz | 4 |
| P8 Streaming pipeline roundtrip | 4 |
| P9 Existing scenario regression | 3 |

## Key Findings

- IPC `visualization: bool` parameter works: `handle_diversity` builds ecology scenario and includes JSON in response
- All 6 new domain builders produce valid JSON that round-trips through serde
- Existing scenario builders (dynamics, anderson, benchmark) remain functional
- Streaming pipeline scenario produces parseable multi-node graph with edges

## Chain Position

V101 → Exp333 (viz evolution) + Exp334 (science→viz pipeline)
