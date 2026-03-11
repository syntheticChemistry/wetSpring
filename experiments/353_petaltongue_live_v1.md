# Exp353: petalTongue Live Ecology Dashboard v1

**Date:** 2026-03-10
**Track:** V110 — petalTongue Live + Anderson QS
**Binary:** `validate_petaltongue_live_v1`
**Required features:** `gpu`
**Status:** PASS (54/54)

---

## Hypothesis

First live visualization experiment. Validates all 9 DataChannel types serialize correctly, IPC push client discovers petalTongue (or gracefully degrades), StreamSession lifecycle with ecology data, scenario JSON export, biomeOS/NUCLEUS readiness probing, real barraCuda math (Shannon, Bray-Curtis, Anderson W, norm_cdf). Produces 3 petalTongue-loadable JSON scenarios: ecology_dashboard.json, anderson_qs_landscape.json, amplicon_pipeline.json.

## Method

Full live pipeline: DataChannel coverage (all 9 types), serialization, JSON export, UiConfig ecology theme, IPC discovery, StreamSession lifecycle, LivePipelineSession, Anderson QS visualization, real math validation, biomeOS/NUCLEUS readiness, live push integration.

## Domains

| Domain | Description |
|--------|-------------|
| DataChannel coverage | All 9 types serialize correctly |
| Serialization | Bar, Scatter, TimeSeries, Gauge, FieldMap, Heatmap, etc. |
| JSON export | ecology_dashboard, anderson_qs_landscape, amplicon_pipeline |
| UiConfig | Ecology theme |
| IPC discovery | petalTongue discovery or graceful degradation |
| StreamSession | Lifecycle with ecology data |
| LivePipelineSession | Live push integration |
| Anderson QS | Visualization |
| Real math | Shannon, Bray-Curtis, Anderson W, norm_cdf |
| biomeOS/NUCLEUS | Readiness probing |

## Results

All 54 checks PASS. See `cargo run --release --features gpu --bin validate_petaltongue_live_v1`.

## Key Finding

Live ecology dashboard validated. All DataChannel types, IPC discovery, and scenario export produce petalTongue-loadable JSON. biomeOS/NUCLEUS readiness integrated.
