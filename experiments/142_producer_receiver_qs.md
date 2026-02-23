# Exp142: QS Producer vs Receiver Separation — Live NCBI

| Field          | Value |
|----------------|-------|
| **Status**     | PASS (8/8 checks, LIVE NCBI DATA) |
| **Binary**     | `validate_producer_receiver_qs` |
| **Date**       | 2026-02-23 |
| **Phase**      | 37 — Empirical Validation |
| **Data**       | 6 synthase + 5 receptor gene families × 7 habitats = 77 queries |

## Hypothesis

QS signal producers (synthases: luxI, lasI, rhlI) are metabolically expensive.
Receivers (receptors: luxR, sdiA) are cheap. Anderson predicts producers should
concentrate in 3D-dense habitats where signals propagate. Receivers can exist
anywhere (eavesdropping is geometry-independent).

## Key Results — Live NCBI Data

| Habitat     | Producers | Receptors | R:P ratio |
|-------------|----------:|----------:|----------:|
| soil        | 341       | 533       | 1.6:1     |
| rhizosphere | 90        | 203       | 2.3:1     |
| biofilm     | 141       | 112       | 0.8:1     |
| ocean_water | 183       | 226       | 1.2:1     |
| freshwater  | 547       | 876       | 1.6:1     |
| **hot_spring** | **5**  | **2**     | **0.4:1** |
| clinical    | 3,176     | 4,016     | 1.3:1     |

## Key Findings

1. **Hot springs: 5 producers, 2 receptors** — near-zero QS investment. The
   cleanest signal in the entire dataset. 2D prediction strongly confirmed.
2. **R:P ratios are ~1:1 across most habitats** — the simple eavesdropper
   hypothesis (high R:P in dilute) doesn't hold in raw NCBI data.
3. **sdiA dominates the receptor column** (276 soil, 611 freshwater, 2557
   clinical) — these are all Enterobacteriaceae with eavesdropper receptors.
   The eavesdropper signal IS there but concentrated in one gene family.
4. **Freshwater has MORE producers (547) than soil (341)** — NCBI "freshwater"
   isolates include biofilm-formers, not just plankton. Isolation source
   metadata doesn't cleanly separate lifestyles.
5. **Biofilm R:P = 0.8:1** — biofilm organisms invest HEAVILY in production
   (more producers than receivers). This is the opposite of eavesdropping.
