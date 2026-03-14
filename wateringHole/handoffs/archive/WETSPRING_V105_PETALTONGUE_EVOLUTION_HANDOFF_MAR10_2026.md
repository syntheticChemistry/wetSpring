# wetSpring V105 — petalTongue Visualization Evolution Handoff

**Date**: 2026-03-10  
**From**: wetSpring  
**To**: barraCuda, toadStool, coralReef, petalTongue  
**Status**: Complete — clippy-clean, all tests pass

---

## Summary

wetSpring V105 evolves the visualization layer from static scenario dumps to a
live pipeline streaming architecture. Scientists bring their data; wetSpring
builds the visualization in real time through `petalTongue` IPC.

## What Changed

### Layer 1 — Infrastructure (petalTongue parity with healthSpring)

| Feature | Before | After |
|---------|--------|-------|
| IPC buffer | 4 KB | 64 KB |
| Channel types | 8 | 9 (+ Scatter3D) |
| Capabilities | 16 | 21 (+ 5 domains) |
| IPC methods | 5 | 9 (+ query_capabilities, subscribe_interactions, push_render_with_domain, dismiss_session) |

### Layer 2 — Live Pipeline Streaming

- `LivePipelineSession` wraps `StreamSession` with domain-aware stage progression
- Pre-built stages for 16S amplicon (5 stages), LC-MS (5 stages), phylogenetics (4 stages)
- Each stage reports progress via gauge, pushes data channels on completion
- JSON export fallback when petalTongue socket unavailable

### Layer 2 — New Scenario Builders (5 new modules)

| Module | Domain | Visualization |
|--------|--------|---------------|
| `scenarios::msa` | ecology | Conservation bar, pairwise identity heatmap, mean identity gauge |
| `scenarios::calibration` | measurement | Standard curve, R² gauge, predicted unknowns |
| `scenarios::spectroscopy` | measurement | JCAMP-DX spectra, peak bar charts |
| `scenarios::basecalling` | ecology | Pass rate, mean quality, read length distribution |
| `scenarios::neighbor_joining` | ecology | Distance heatmap, branch lengths, tree metrics |

### Layer 2 — Scientific Ranges

Added actionable thresholds to all scenarios missing them:
- **stochastic**: population stability (within/beyond 2 SD)
- **rarefaction**: richness vs Chao1 estimate (>50% / <50%)
- **HMM**: transition probability (normal 0–0.5, warning 0.5–1.0)
- **NMF**: reconstruction error (normal <0.1, warning 0.1–0.5)
- **streaming_pipeline**: quality pass rate (80–100% / 0–80%)

### Layer 3 — Sample-Parameterized Scenarios

- `EnvironmentalProfile` → diversity + taxonomy + ordination from real data
- `PfasScreeningProfile` → detection + quantitation from suspect screening
- `CalibrationProfile` → standard curve + predicted unknowns from lab data

### 3D Ordination

- `Scatter3D` wired into `ordination_scenario` for 3+ axis PCoA
- Point labels, axis labels, unit annotations — petalTongue renders with rotation

## Test Results

| Metric | Value |
|--------|-------|
| Lib tests (json feature) | 1,288 pass |
| Integration tests | 219 pass |
| Clippy warnings | 0 |
| New test count (V105) | +31 (from 1,200 to 1,231 lib + json viz) |

## Scenario Builder Count

| Category | Count |
|----------|-------|
| Existing (V104) | 28 |
| New (V105) | 5 |
| Sample-parameterized (V105) | 3 |
| **Total** | **33 + 3 profile builders** |

## Upstream Requests

### To petalTongue

1. **Scatter3D rendering** — wetSpring now emits `scatter3d` channels for PCoA
   3D. petalTongue should render with WebGL/wgpu rotation, zoom, point labels.

2. **`visualization.capabilities` RPC** — wetSpring IPC client now calls
   `visualization.capabilities`. petalTongue should respond with supported
   channel types and features.

3. **`visualization.interact.subscribe` RPC** — wetSpring can now subscribe
   to interactions. petalTongue should push click/selection/zoom events back.

4. **`visualization.dismiss` RPC** — wetSpring now sends dismiss on session
   close. petalTongue should tear down the session rendering.

### To barraCuda

5. **`seed_extend` / `profile_alignment`** — MSA scenario currently does
   progressive alignment via wetSpring's built-in MSA. When barraCuda adds
   seed-extend or profile HMM alignment, wetSpring will rewire.

### To toadStool

6. **Live pipeline hardware routing** — `LivePipelineSession` stages could
   benefit from `PrecisionRoutingAdvice` to automatically route f64 stages
   to native GPU vs CPU.

## No Breaking Changes

All existing scenario builders, IPC methods, and DataChannel types are
unchanged. The Scatter3D variant is additive; existing JSON consumers that
don't handle unknown channel types via `#[serde(other)]` may need updating.
