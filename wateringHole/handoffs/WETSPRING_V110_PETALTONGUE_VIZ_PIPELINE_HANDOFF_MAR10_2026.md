# wetSpring V110: petalTongue Visualization Pipeline Handoff

**Date:** 2026-03-10
**From:** wetSpring (V110)
**To:** toadStool/barraCuda, petalTongue, biomeOS

---

## Summary

V110 wires wetSpring's 28 existing scenario builders to live petalTongue
visualization via IPC push + JSON export. Four new experiment binaries
produce human-readable dashboards and validate Anderson model evolution.

The `stream_ecology` module adds ecology-specific StreamSession methods
that compose real barraCuda math with IPC push — a pattern for upstream
absorption into `barracuda::bio::visualization`.

**Key scientific finding:** The Anderson W mapping must include oxygen as
a second disorder dimension (W = 3.5·H' + 8·O₂). The original inverse-
diversity model is wrong for cross-environment QS prediction (r = -0.575
vs r = +0.851 for O₂-modulated). Testable with real metatranscriptomic data.

## New Experiments

| Exp | Binary | Checks | Status |
|:---:|--------|:------:|:------:|
| 353 | `validate_petaltongue_live_v1` | 54/54 | PASS |
| 354 | `validate_petaltongue_anderson_v1` | 21/21 | PASS |
| 355 | `validate_petaltongue_biogas_v1` | 18/18 | PASS |
| 356 | `validate_anderson_qs_environments_v1` | 18/18 | PASS |

### Exp353: Live Ecology Dashboard

- Validates all 9 DataChannel types serialize correctly for petalTongue
- IPC push client discovers socket (or gracefully degrades to JSON export)
- StreamSession lifecycle with ecology data
- biomeOS binary discovery and NUCLEUS readiness probing
- Produces 3 loadable scenarios: ecology_dashboard.json, anderson_qs_landscape.json, amplicon_pipeline.json

### Exp354: Anderson QS Landscape (flagship)

- 5 biomes: Algae Pond, Forest Soil, Anaerobic Digester, Deep-Sea Vent, Rhizosphere
- Diversity → Disorder → P(QS) full pipeline
- 6 DataChannel types exercised: Bar, Scatter, TimeSeries, Gauge, FieldMap, Heatmap
- 10×10 spatial W lattice as FieldMap
- Key finding: Digester (H'=0.25, W=18.57) → QS SUPPRESSED; diverse biomes P(QS) > 0.88

### Exp355: Biogas Kinetics Dashboard

- Track 6: Gompertz, first-order, Monod, Haldane kinetics for 3 feedstocks
- Temperature/pH operational envelopes as Distribution
- Co-digestion mix (W=11.3, H'=1.90) → HEALTHY; monoculture → STRESSED

### Exp356: Anderson QS Cross-Environment Validation (key scientific finding)

- Tests 3 W parameterizations against literature QS prevalence for 10 environments
- **H1 (original, inverse diversity):** r = -0.575 — REFUTED for cross-environment use
- **H2 (signal dilution, W ∝ H'):** r = +0.812 — diversity IS disorder (signal scattering)
- **H3 (O₂-modulated, W = 3.5·H' + 8·O₂):** r = +0.851 — BEST FIT
- Key insight: anaerobic communities have lower effective W due to reduced transcriptional noise for QS operons (FNR/ArcAB/Rex regulation)
- Exports `anderson_qs_model_comparison.json` with per-model Bar and Scatter channels
- **Absorption target:** `barracuda::bio::anderson` should evolve `map_h_to_w` to accept O₂ parameter

## New Module: `stream_ecology.rs`

Ecology-specific StreamSession methods that compose barraCuda math with IPC:

| Method | Function |
|--------|----------|
| `push_diversity_frame` | Computes Shannon/Simpson/Pielou from counts, pushes Bar |
| `push_bray_curtis_update` | Recomputes BC matrix, pushes Heatmap |
| `push_rarefaction_point` | Appends depth/richness to TimeSeries |
| `push_anderson_w` | Updates W and P(QS) gauges |
| `push_kinetics_step` | Appends kinetics (t, y) to TimeSeries |

6 unit tests. Pattern is ready for upstream absorption as `barracuda::bio::visualization`.

## JSON Scenario Artifacts

All dashboards produce `output/*.json` loadable via `petaltongue ui --scenario`:

| File | Size | Content |
|------|------|---------|
| `ecology_dashboard.json` | ~5KB | Ecology + beta diversity + edges |
| `anderson_qs_landscape.json` | ~2KB | Quick Anderson dashboard |
| `anderson_qs_landscape_full.json` | ~21KB | Full 5-biome Anderson landscape |
| `amplicon_pipeline.json` | ~1KB | 16S pipeline stages |
| `biogas_kinetics_dashboard.json` | ~59KB | Full biogas kinetics (60 time points × 3 feedstocks × 4 models) |

## biomeOS / NUCLEUS Integration

- biomeOS binary discovery: checks `../../phase2/biomeOS/target/{release,debug}/biomeos`
- Primal scan: beardog, songbird, toadstool, nestgate, squirrel, petaltongue, biomeos
- Tower/Node/Nest readiness reported (currently standalone mode — primals discovered when installed)
- Graceful degradation: all experiments pass with or without biomeOS/NUCLEUS running

## Absorption Opportunities for Upstream

### For barraCuda
- `stream_ecology.rs` methods → `barracuda::bio::visualization` module
- Kinetics functions (Gompertz, Monod, Haldane) → `barracuda::bio::kinetics` (already noted in V109)
- **Anderson W evolution:** `barracuda::bio::anderson::map_h_to_w(h, o2)` — two-parameter W function replacing single-variable. GPU shader `AndersonDisorderF64` should accept O₂ as second uniform. Exp356 provides the coefficients (α=3.5, β=8.0) and test data for 10 environments.

### For petalTongue
- `ecology` domain theme validated with all 9 channel types
- ScientificRange patterns for ecology (diversity thresholds, W regimes)
- Composite scenario patterns (multi-node with edges)

### For biomeOS
- Primal discovery paths solidified (env var → XDG → temp fallback)
- NUCLEUS readiness reporting pattern (Tower/Node/Nest status)
- IPC graceful degradation pattern (socket found but refused → continue)

## Primitive Consumption (V110)

| Primitive | Source | Usage |
|-----------|--------|-------|
| `diversity::shannon` | barraCuda CPU | All 3 experiments |
| `diversity::simpson` | barraCuda CPU | Exp353, Exp354 |
| `diversity::pielou_evenness` | barraCuda CPU | Exp353, Exp354 |
| `diversity::chao1` | barraCuda CPU | Exp353 |
| `diversity::observed_features` | barraCuda CPU | Exp353 |
| `diversity::rarefaction_curve` | barraCuda CPU | Exp353, Exp354 |
| `diversity::bray_curtis_matrix` | barraCuda CPU | All 3 experiments |
| `stats::norm_cdf` | barraCuda CPU | Exp354, Exp355 (P(QS)) |
| `stats::covariance` | barraCuda CPU | Exp353 |
| `stats::mean` | barraCuda CPU | Exp353 |

## Validation Chain Position

```
CPU (Exp347) → GPU (Exp348) → ToadStool (Exp349)
→ Streaming (Exp350) → metalForge (Exp351) → NUCLEUS (Exp352)
→ petalTongue Live (Exp353) → Anderson Viz (Exp354) → Biogas Viz (Exp355)
→ Anderson Cross-Env (Exp356)
```

V110 extends the V109 validation chain into the visualization layer AND
evolves the Anderson model itself. The math is already proven (V109
CPU/GPU/mixed hardware); V110 proves the visualization pipeline makes
that math human-readable, and Exp356 refines the underlying physics
(W now two-dimensional: diversity + oxygen).
