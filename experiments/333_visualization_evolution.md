# Exp333: Visualization Evolution

**Date:** March 9, 2026
**Binary:** `validate_viz_evolution_v1`
**Crate:** `wetspring-forge`
**Result:** 44/44 PASS

## Scope

Validates the full petalTongue visualization evolution in wetSpring V101:

| Domain | Checks |
|--------|:------:|
| V1 Spectrum DataChannel | 4 |
| V2 StreamSession lifecycle | 7 |
| V3 Songbird capabilities | 8 |
| V4 Pangenome scenario | 4 |
| V5 HMM scenario | 3 |
| V6 Stochastic scenario | 3 |
| V7 Similarity scenario | 4 |
| V8 Rarefaction scenario | 3 |
| V9 NMF scenario | 4 |
| V10 Streaming pipeline | 4 |

## Key Decisions

- **Spectrum DataChannel**: 7th channel type following healthSpring pattern and wateringHole guidance
- **StreamSession**: Session lifecycle (open→push→close) with backpressure awareness
- **Songbird**: 16 visualization capabilities announced, 7 channel types, streaming support
- **Builder pattern**: Each builder takes domain output structs, returns `(EcologyScenario, Vec<ScenarioEdge>)`

## Chain Position

V101 → Exp333 (viz evolution) + Exp334 (science→viz pipeline)
V100 → Exp327-332 (petalTongue + local evolution + mixed HW)
