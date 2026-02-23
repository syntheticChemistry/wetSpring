# Exp137: Planktonic & Mixed Fluid 3D — Dilution Effects

| Field          | Value |
|----------------|-------|
| **Status**     | PASS (10/10 checks, GPU-confirmed) |
| **Binary**     | `validate_planktonic_dilution` |
| **Date**       | 2026-02-23 |
| **Phase**      | 36c — Why Analysis |

## Hypothesis

Sea plankton is a living 3D biological mass. Does the Anderson model predict
QS in dilute 3D suspensions? Dilution increases effective disorder (signal
must traverse empty space), modeled as W_eff = W_base / occupancy.

## Design

- Occupancy sweep: 100%, 75%, 50%, 30%, 20%, 10%, 5%
- Sea plankton density mapping (10⁶ cells/mL → occupancy ~0.1%)
- Turnover rate analysis: early colonization (J=0.2) to climax (J=0.95)
- Dense biofilm vs floc vs marine snow vs free plankton

## Key Results

Dilution sweep:

| occupancy | W_eff | ⟨r⟩   | regime     |
|-----------|-------|--------|------------|
| 100%      | 13.0  | 0.4757 | QS-ACTIVE  |
| 75%       | 17.3  | 0.4343 | suppressed |
| 50%       | 26.0  | 0.4318 | suppressed |
| 10%       | 130.0 | 0.3524 | suppressed |

**QS breaks at occupancy ≤ 75%** (W_eff ≥ 17.3, exceeding W_c ≈ 16.5)

Biofilm temporal stages (turnover):

| Stage               | J    | W     | 2D     | 3D     |
|---------------------|------|-------|--------|--------|
| early_colonization  | 0.20 | 3.40  | ACTIVE | ACTIVE |
| growth_phase        | 0.50 | 7.75  | ACTIVE | ACTIVE |
| mature_biofilm      | 0.80 | 12.10 | ACTIVE | ACTIVE |
| climax_community    | 0.95 | 14.27 | ---    | ACTIVE |

## Key Findings

1. **Free-floating plankton (10⁶/mL) is QS-SUPPRESSED** — occupancy ~0.1%
   means W_eff >> W_c; signal can't traverse the gaps
2. **QS in marine bacteria requires ATTACHMENT** — flocs, marine snow,
   particle-attached communities (occupancy >75%)
3. **This matches biology**: Hmmer et al. 2002 — QS prevalence in marine
   bacteria scales with surface attachment, not cell density
4. **Turnover rate matters indirectly** — fast turnover → high steady-state
   diversity (high J) → needs 3D; early colonization (low J) → 2D works
5. **Prediction**: the critical biofilm maturation stage where 2D QS fails
   and 3D structure becomes essential is around J ≈ 0.9 (climax community)
