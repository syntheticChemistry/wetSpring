# Exp354: Anderson QS Landscape v1 — Flagship Visualization

**Date:** 2026-03-10
**Track:** V110 — petalTongue Live + Anderson QS
**Binary:** `validate_petaltongue_anderson_v1`
**Required features:** `gpu`
**Status:** PASS (21/21)

---

## Hypothesis

Builds the "one picture that tells the whole story" dashboard. Computes diversity, disorder (W), and P(QS) propagation probability across 5 biomes (Algae Pond, Forest Soil, Anaerobic Digester, Deep-Sea Vent, Rhizosphere). Uses 6 of 9 DataChannel types: Bar (Shannon/Simpson comparison), Scatter (H' vs W mapping), TimeSeries (W → P(QS) curve + rarefaction curves), Gauge (per-biome P(QS)), FieldMap (10×10 spatial W lattice), Heatmap (Bray-Curtis matrix). Exports anderson_qs_landscape_full.json (21KB scenario).

## Method

5 biomes × diversity + Anderson W + P(QS). 6 DataChannel types. Full scenario export.

## Domains

| Domain | Description |
|--------|-------------|
| Bar | Shannon vs Simpson comparison |
| Scatter | H' vs W mapping |
| TimeSeries | W → P(QS) curve, rarefaction curves |
| Gauge | Per-biome P(QS) |
| FieldMap | 10×10 spatial W lattice |
| Heatmap | Bray-Curtis matrix |

## Results

All 21 checks PASS. See `cargo run --release --features gpu --bin validate_petaltongue_anderson_v1`.

## Key Finding

**Anaerobic Digester** (H'=0.25, W=18.57, P(QS)=0.245) → **QS SUPPRESSED**. All diverse biomes P(QS) > 0.88 → **QS ACTIVE**. Community structure determines signaling.
