# Experiment 025 — Bruger & Waters 2018 Cooperative QS Game Theory

**Date:** 2026-02-20
**Status:** COMPLETE
**Track:** 1 (Microbial Ecology / Signaling)
**Faculty:** Waters (MMG, MSU)

---

## Objective

Reproduce the evolutionary game theory model from Bruger & Waters 2018
(AEM 84:e00402-18). Cooperators produce QS signal (public good) at a cost;
cheaters exploit the signal without producing it. The model shows how
cooperation persists despite cheater frequency advantage.

## Model

4-variable ODE: Nc (cooperator density), Nd (cheater density), AI (signal),
B (biofilm). Cooperators pay a production cost but contribute to shared signal
benefit and biofilm formation. Cheaters grow faster (no cost) but can't
produce signal independently.

## Scenarios

| # | Scenario | IC | Python f_coop | Python B_ss |
|---|----------|----|---------------|-------------|
| 1 | Equal start | 50/50 | 0.376 | 0.757 |
| 2 | Coop-dominated | 90/10 | 0.866 | 0.767 |
| 3 | Cheat-dominated | 10/90 | 0.073 | 0.534 |
| 4 | Pure coop | 100/0 | 1.000 | 0.767 |
| 5 | Pure cheat | 0/100 | 0.000 | 0.000 |

## Key Findings

- Cheaters always have frequency advantage (f_coop < initial fraction)
- But cooperators persist because they generate the public good
- Pure cheaters cannot form biofilm (tragedy of the commons)
- Biofilm output tracks cooperator frequency

## Validation Binary

```bash
cargo run --bin validate_cooperation
```

## Modules Validated

- `bio::cooperation` — game theory ODE, cooperator frequency analysis
- `bio::ode` — RK4 integrator
