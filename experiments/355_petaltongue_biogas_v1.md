# Exp355: petalTongue Biogas Dashboard v1

**Date:** 2026-03-10
**Track:** V110 — petalTongue Live + Anderson QS
**Binary:** `validate_petaltongue_biogas_v1`
**Required features:** `gpu`
**Status:** PASS (18/18)

---

## Hypothesis

Track 6 anaerobic digestion monitoring dashboard. Streams Modified Gompertz, first-order, Monod, and Haldane kinetics for 3 feedstocks (Corn Stover, Coffee Residues, Co-Digestion Mix). Includes digester diversity with Anderson W gauges, temperature/pH operational envelopes. 5 scenario nodes, 18 data channels, 4 edges. The co-digestion mix (W=11.3, H'=1.90) is HEALTHY; monoculture feedstocks are STRESSED.

## Method

6 anaerobic digestion monitoring streams. 4 kinetics models × 3 feedstocks. Anderson W gauges, temperature/pH envelopes. 5 nodes, 18 channels, 4 edges.

## Domains

| Domain | Description |
|--------|-------------|
| Modified Gompertz | Kinetics for 3 feedstocks |
| First-order | Kinetics |
| Monod | Growth kinetics |
| Haldane | Inhibition kinetics |
| Anderson W | Digester diversity gauges |
| Operational | Temperature/pH envelopes |

## Results

All 18 checks PASS. See `cargo run --release --features gpu --bin validate_petaltongue_biogas_v1`.

## Key Finding

Co-digestion mix (W=11.3, H'=1.90) → **HEALTHY**. Monoculture feedstocks (Corn Stover, Coffee Residues) → **STRESSED**. Dashboard validates Track 6 biogas pipeline with petalTongue visualization.
