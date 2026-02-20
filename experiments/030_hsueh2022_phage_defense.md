# Experiment 030: Hsueh/Severin 2022 — Phage Defense Deaminase

**Paper**: Hsueh, Severin et al. "A Broadly Conserved Deoxycytidine Deaminase Protects Bacteria from Phage Infection" — *Nature Microbiology* 7:1210-1220, 2022

**Track**: 1 — Microbial Ecology & Signaling (Waters)

---

## Objective

Model the evolutionary arms race between bacteria with deoxycytidine deaminase (DCD) defense and lytic phage. The defense imposes a fitness cost but provides protection, creating a classic cost-benefit tradeoff in phage-bacteria coevolution.

## Model

4-variable ODE system:
- `Bd`: Defended bacteria (with DCD)
- `Bu`: Undefended bacteria (no DCD)
- P: Free phage
- R: Resources (nutrients, Monod kinetics)

Key dynamics:
- DCD deaminates cytosine in phage DNA → reduces burst size by `defense_efficiency` (default 90%)
- Defended bacteria pay growth cost (`defense_cost` = 15% of max growth rate)
- Resource-limited growth via Monod kinetics

## Scenarios

| # | Scenario | Key Result |
|---|----------|------------|
| 1 | No phage | Undefended outcompetes (no cost penalty) |
| 2 | Phage attack | Defended survives, undefended crashes |
| 3 | Pure defended | Defended persists at high density with phage |
| 4 | Pure undefended | Crashes under phage pressure |
| 5 | High cost (50%) | Defended still persists but at lower density |

## Baseline Provenance

- **Python script**: `scripts/hsueh2022_phage_defense.py`
- **Integrator**: `scipy.integrate.odeint` (adaptive LSODA)
- **Rust module**: `barracuda::bio::phage_defense`
- **Rust integrator**: RK4 fixed-step (dt=0.001)
- **Validation binary**: `validate_phage_defense` — 12/12 checks

## Acceptance Criteria

- [x] Without phage: undefended outcompetes defended
- [x] With phage: defended survives, undefended crashes to ~0
- [x] Defense efficiency modulates survival
- [x] All variables non-negative across all scenarios
- [x] Deterministic (bit-exact reruns)

## GPU Promotion Path

**Tier A — Rewire**: Population-level ODEs are parallel across parameter sweeps. Each (cost, efficiency) pair is independent → one workgroup per parameter set. ToadStool `ode_batch` primitive.

---

*Date*: 2026-02-20 | *Checks*: 12/12 PASS
