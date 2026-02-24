# Exp157: Fajgenbaum Pathway Scoring — PI3K/AKT/mTOR → Sirolimus

| Field          | Value |
|----------------|-------|
| **Status**     | PASS (8/8 checks) |
| **Binary**     | `validate_fajgenbaum_pathway` |
| **Date**       | 2026-02-24 |
| **Phase**      | 39 — Drug Repurposing Track |
| **Paper**      | 39 (Fajgenbaum et al. JCI 2019) |

## Core Idea

Reproduce the computational drug-pathway matching that identified sirolimus
as a treatment for IL-6-blockade-refractory iMCD. PI3K/AKT/mTOR ranks as the
highest-activation pathway (0.92); sirolimus (mTOR inhibitor) correctly scores
above IL-6 blockers that had failed clinically.

## Key Findings

- PI3K/AKT/mTOR is the highest-activation pathway (0.92) in proteomic data
- Sirolimus and everolimus correctly rank #1 and #2 by pathway match score
- IL-6 blockers (tocilizumab, siltuximab) correctly rank below mTOR inhibitors
- The pathway scoring logic is a special case of NMF factorization (Papers 41-42)
- All data sources are open (DrugBank, KEGG, published case series)
