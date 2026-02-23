# Exp148: QS Traveling Wave × Anderson Localization

| Field          | Value |
|----------------|-------|
| **Status**     | PASS (6/6 checks) |
| **Binary**     | `validate_qs_wave_localization` |
| **Date**       | 2026-02-23 |
| **Phase**      | 38 — Extension Papers |

## Core Idea

Combine two complementary QS models:
- Meyer et al. (PRE 2020): QS as reaction-diffusion traveling wave
- Our model: Anderson localization determines WHETHER waves CAN propagate

Effective QS range = min(L_QS, ξ), where L_QS = reaction-limited range
and ξ = Anderson localization length.

## Combined QS Range Equation

    L_eff(W, d) = min( L_QS, ξ(W, d) )

    L_QS = √(D/k_deg) — chemistry-limited
    ξ(W,d) = a × |W - W_c|^(-ν) — Anderson-limited (ν ≈ 1.57 for d=3)

## Wave-Localization Regimes

| W | d | v_wave | ξ_rel | Regime |
|:---:|:---:|:------:|:-----:|--------|
| 2.0 | 3 | 1.00 | 100 | EXTENDED — wave-limited |
| 8.0 | 3 | 0.95 | 50 | EXTENDED — ξ shrinking |
| 14.0 | 3 | 0.80 | 10 | NEAR-CRITICAL |
| 16.5 | 3 | 0.30 | 3 | CRITICAL — Anderson transition |
| 20.0 | 3 | 0.00 | 1.5 | LOCALIZED — wave arrested |
| 5.0 | 2 | 0.00 | 5 | LOCALIZED (d=2) |

## Key Findings

- V. fischeri in light organ: W = 1.95, deep extended regime → L_eff = L_QS
  (matches Meyer et al.'s measured 100-200 µm activation range exactly)
- Soil biofilm at W = 12.8: wave speed reduced to ~22% of maximum
- QS traveling waves STOP at the Anderson transition (W = W_c)
- "Localized QS" (Jemielita 2019) IS Anderson localization
- "Synchronized QS" IS the extended regime
