# Exp356: Anderson QS Cross-Environment Validation v1

**Date:** 2026-03-10
**Track:** V110
**Status:** PASS (18/18)
**Binary:** `validate_anderson_qs_environments_v1`
**Features:** `gpu`

## Hypothesis

The Anderson QS model's W parameterization (H' → W mapping) determines
whether the model correctly predicts QS prevalence across aerobic,
microaerobic, and anaerobic environments. Three alternative W functions
are tested against literature-based QS prevalence scores for 10 environments.

## Method

- **10 environments** with literature-parameterized community profiles:
  Lab E. coli, P. aeruginosa biofilm, human gut, anaerobic digester,
  oral biofilm, rhizosphere soil, ocean surface, bulk soil, hot spring,
  deep-sea hydrothermal vent.
- **Three W models**:
  - H1 (original): W = 20·exp(-0.3·H') — high diversity → low W → QS active
  - H2 (signal dilution): W = 4·H' — high diversity → high W → QS suppressed
  - H3 (O₂-modulated): W = 3.5·H' + 8·O₂ — diversity + oxygen both add disorder
- Pearson correlation and MAE against known QS biology scores

## Key Results

| Model | Pearson r | MAE | Correct direction? |
|-------|:---------:|:---:|:------------------:|
| H1 (inverse) | -0.575 | 0.418 | No — predicts MORE QS in diverse environments |
| H2 (dilution) | +0.812 | 0.331 | Yes — monoculture > polyculture |
| H3 (O₂-mod) | **+0.851** | **0.331** | Yes — captures anaerobic > aerobic |

## Key Finding

**The original model (H1) is wrong for cross-environment comparison.** It maps
diversity inversely to disorder, predicting more QS in diverse soil than in a
monoculture — contradicting known biology.

Signal dilution (H2) correctly captures that many species = many competing
signals = more scattering. Adding oxygen (H3) further improves the model
by capturing FNR/ArcAB/Rex-mediated QS upregulation under anaerobic conditions.

**Implication**: The Anderson QS model needs BOTH diversity AND oxygen as
disorder dimensions. This is testable with real 16S + metatranscriptomic data.

## Run

```bash
cargo run --release --features gpu --bin validate_anderson_qs_environments_v1
```

## petalTongue Scenario

```bash
petaltongue ui --scenario output/anderson_qs_model_comparison.json
```
