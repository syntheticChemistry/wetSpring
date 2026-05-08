# wetSpring V149 — primalSpring v0.9.17 Alignment Handoff

**Date:** April 20, 2026
**From:** wetSpring V149
**For:** primalSpring, barraCuda, Squirrel, ToadStool, all spring teams
**primalSpring version:** v0.9.17 (Phase 45)
**guideStone:** Level 4 — 38/38 pass, 4 skip, exit 0

---

## Summary

wetSpring's `wetspring_guidestone` now passes **38/38** checks with only **4 skips**
against a live 12-primal NUCLEUS deployed via `nucleus_launcher.sh`. This resolves
7 of the 11 skips from V148 (31/31, 11 skip) through corrected parameter names
and proper launcher configuration.

### What Changed (V148 → V149)

| Metric | V148 | V149 | Delta |
|--------|------|------|-------|
| Checks passed | 31/31 | 38/38 | +7 |
| Skips | 11 | 4 | -7 |
| Primals alive | 4 | 5 | +1 (Squirrel) |
| primalSpring | v0.9.16 | v0.9.17 | genomeBin v5.1, launcher |
| Open gaps | 14 | 10 | -4 resolved |

---

## Key Findings

### 1. PG-13 Resolved: Parameter Names, Not Missing Methods

The 6 "missing" barraCuda methods (`stats.variance`, `stats.correlation`,
`linalg.solve`, `linalg.eigenvalues`, `spectral.fft`, `spectral.power_spectrum`)
were always registered — the guideStone was sending empty `{}` params instead
of the correct parameter names. Corrected mappings:

| Method | Correct Params | Result |
|--------|---------------|--------|
| `stats.variance` | `{"data": [f64]}` | `{result, convention, denominator}` |
| `stats.correlation` | `{"x": [f64], "y": [f64]}` | `{result}` |
| `linalg.solve` | `{"matrix": [[f64]], "b": [f64]}` | `{result: [f64]}` |
| `linalg.eigenvalues` | `{"matrix": [[f64]]}` | `{result: [f64]}` |
| `spectral.fft` | `{"data": [f64]}` | `{real, imag, result, n}` |
| `spectral.power_spectrum` | `{"data": [f64]}` | `{result, n}` |

**Lesson for other springs:** Probe methods with correct params before
classifying as "Unknown method." The error "Missing required param" means
the method exists but needs different parameters.

### 2. PG-14 Partially Resolved: Squirrel Liveness

The `nucleus_launcher.sh` properly configures Squirrel with provider sockets
(`SERVICE_MESH_ENDPOINT`, `CRYPTO_SIGN_PROVIDER_SOCKET`, etc.). Squirrel
liveness now PASSES via the `ai` capability domain. `inference.complete`
still SKIPs because no Ollama backend is configured.

### 3. PG-15 Updated: compute.dispatch Exists

ToadStool's `compute.dispatch` IS registered and returns a proper error:
"Missing 'binary' array (compiled GPU binary bytes)". This is a legitimate
constraint — it needs actual compiled shader binary data, not a noop probe.

### 4. nucleus_launcher.sh Works End-to-End

All 12 primals start in dependency order with proper family-named sockets
and capability domain aliases. Socket contention from multiple NUCLEUS
instances (different `FAMILY_ID`) requires careful cleanup — stale PID files
and generic sockets from previous runs can confuse `pgrep` detection.

### 5. biomeOS Compiles

The upstream compile error (biomeos-unibin) is fixed in v0.9.17. biomeOS
Neural API starts and creates `neural-api-{family}.sock`.

---

## Remaining 4 Skips

| Skip | Reason | Blocks Level 5? |
|------|--------|-----------------|
| `compute.dispatch` | ToadStool needs compiled GPU binary | No — legitimate constraint |
| `inference.complete` | No Ollama backend configured | No — infrastructure dep |
| `stats.median` | Not registered in barraCuda | Yes |
| `linalg.determinant` | Not registered in barraCuda | Yes |

Only 2 method registrations (`stats.median`, `linalg.determinant`) remain
as actual code gaps. These are in N2 (extended domain science), not in the
15-method manifest. Level 5 certification may proceed with these as
documented exceptions.

---

## Recommendations

### For barraCuda team
- Register `stats.median` and `linalg.determinant` in the ecobin
- Document parameter schemas for all 32+ IPC methods (prevents PG-13 recurrence)

### For primalSpring team
- Document the `nucleus_launcher.sh` socket cleanup procedure for multi-family
  environments (stale PID files cause "already running" false positives)
- Consider adding `--force-clean` flag to the launcher

### For other spring teams
- Use `nucleus_launcher.sh` for deployment — it handles all env vars and
  dependency ordering
- Probe methods with populated params before classifying as missing
- Squirrel requires `SERVICE_MESH_ENDPOINT` and other provider sockets;
  the launcher sets these automatically

---

## Files Changed

| File | Change |
|------|--------|
| `barracuda/src/bin/wetspring_guidestone.rs` | 6 new parity checks with correct params |
| `validation/CHECKSUMS` | Regenerated BLAKE3 for updated guideStone |
| `README.md` | V149 status, 38/38, 10 gaps |
| `CONTEXT.md` | V149, v0.9.17, 5 primals |
| `specs/README.md` | V149 status, dates, counts |
| `docs/PRIMAL_GAPS.md` | PG-13 resolved, PG-14/15 updated |
| `CHANGELOG.md` | V149 entry |
| `whitePaper/baseCamp/README.md` | V149 status alignment |
| `whitePaper/baseCamp/EXTENSION_PLAN.md` | V149 status alignment |
| `experiments/README.md` | V149 status alignment |
| `wateringHole/ECOSYSTEM_LEVERAGE_GUIDE.md` | V149 note |
| `wateringHole/CROSS_SPRING_SHADER_EVOLUTION.md` | V149 note |

---

## Upstream Drift — Requests for primalSpring

1. **downstream_manifest.toml**: `guidestone_readiness = 3` for wetspring — should
   be `4`. Also missing `guidestone_properties` field (ludoSpring has it).
   Requested update:
   ```toml
   guidestone_readiness = 4
   guidestone_properties = { deterministic = true, traceable = true, self_verifying = true, env_agnostic = true, tolerance_documented = true }
   ```

2. **NUCLEUS_SPRING_ALIGNMENT.md**: Lists wetSpring at V147, Level 3. Should be
   V149, Level 4 (38/38 pass, 4 skip).

3. **`is_skip_error` adopted**: wetSpring now uses
   `primalspring::composition::is_skip_error` (v0.9.17) for all error
   classification, matching ludoSpring and neuralSpring patterns.

---

*Handed back to primalSpring per NUCLEUS_SPRING_ALIGNMENT.md feedback protocol.*
