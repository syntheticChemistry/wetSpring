# WETSPRING V147 — Ecosystem Evolution Handoff

**Date:** April 18, 2026
**From:** wetSpring V147
**To:** primalSpring, barraCuda, NestGate, BearDog, Songbird, toadStool, Squirrel,
       coralReef, biomeOS, and all downstream springs
**Purpose:** Primal use review, composition patterns learned, deployment feedback,
             and actionable items for ecosystem evolution.

---

## 1. Where wetSpring Stands

```
Python baseline (58 scripts, 64 papers)
  → Rust validation (1,594 lib tests, 342 binaries)         ← Level 2 DONE
    → barraCuda GPU (44 GPU modules, 150+ primitives)        ← Level 3-4 DONE
      → Primal composition (48 consumed IPC methods)          ← Level 5 IN PROGRESS
        → guideStone (Level 3: bare certified, N2 v0.9.15)   ← CURRENT
          → NUCLEUS deployment (plasmidBin, biomeOS)          ← Level 6 TARGET
```

**guideStone binary:** `wetspring_guidestone`
- Bare mode: 9/9 checks PASS, exit 2 (no NUCLEUS needed)
- N2 expanded: stats.variance, stats.median, stats.correlation,
  linalg.determinant, linalg.eigenvalues, spectral.fft
- CONSUMED_CAPABILITIES: 48 barraCuda (33 v0.9.15 canonical + 15 legacy) +
  21 other primals = 69 total consumed

---

## 2. What wetSpring Learned About Composition

### 2.1 The Evaporation Pattern Works

Migrating from `barracuda::stats::mean()` (library) to `ctx.call("tensor",
"stats.mean", params)` (IPC) is mechanically straightforward. The composition
API (`CompositionContext`, `validate_parity`, `validate_liveness`) handles
socket discovery, JSON-RPC marshalling, and result extraction cleanly.

**Key insight:** The gap is never in the primal — barraCuda already serves all
32+ methods. The gap is always in the spring's wiring.

### 2.2 Bare Mode Is the Foundation

The guideStone's bare mode (B0 + B1, no primals needed) proved more valuable
than expected. It validates the 5 certified properties using analytically
derivable baselines (Shannon, Hill, exact matmul, population variance). This
means a guideStone can certify its own science on any machine without
deploying NUCLEUS first. Springs should implement bare mode before wiring IPC.

### 2.3 Exit Code Semantics Matter

The correct convention (matching primalSpring standard):
- **Exit 0** = full NUCLEUS certification (all layers pass)
- **Exit 1** = at least one check failed
- **Exit 2** = bare-only (no primals discovered, bare properties valid)

wetSpring initially had bare mode exiting 0 (all bare checks passed). This is
wrong — exit 0 should only mean "NUCLEUS certified." Fixed in V147.

### 2.4 Tolerance Strategy

Two tolerance tiers serve different purposes:
- `tolerances::ANALYTICAL_F64` (1e-12): local f64 arithmetic chains
- `ps_tol::IPC_ROUND_TRIP_TOL` (1e-10): JSON serialization round-trip loss

Springs should use their own tolerances for bare mode and primalSpring's
`IPC_ROUND_TRIP_TOL` for all IPC parity checks. The two-tier model cleanly
separates "is my math right?" from "does IPC preserve precision?"

---

## 3. Gaps Found — Actionable for Upstream Teams

### 3.1 For primalSpring (composition API)

**PG-10: `method_to_capability_domain()` routing gap**
The match arm routes `tensor`, `stats`, `math`, `noise`, `activation`, `rng`,
`fhe`, `tolerances`, `validate`, `device` → `"tensor"` (barraCuda). But
`spectral` and `linalg` fall through to the default `_ => prefix` branch.

Fix: add `"spectral" | "linalg"` to the `"tensor"` match arm in
`primalspring::composition::method_to_capability_domain()`.

Workaround (wetSpring uses): pass `"tensor"` explicitly as the capability
domain in `validate_parity` calls for spectral/linalg methods.

**PG-11: downstream_manifest.toml drift**
wetSpring's entry lists 7 `validation_capabilities` and
`guidestone_readiness = 1`. Actual state: 15+ validated methods,
`GUIDESTONE_READINESS = 3`. Update the manifest to reflect reality.

### 3.2 For barraCuda

**v0.9.15 surface confirmation needed:**
wetSpring's N2 depends on these methods being served over JSON-RPC IPC:
- `stats.variance`, `stats.median`, `stats.correlation` (STATS)
- `linalg.determinant`, `linalg.eigenvalues` (LINALG)
- `spectral.fft` (SPECTRAL)

wetSpring's legacy surface also depends on `stats.weighted_mean` — is this
still served in v0.9.15 or replaced by `stats.normalize` + `stats.mean`?

**Response key normalization:** Some methods return `result` (scalar), others
return arrays or nested structures. A consistent `result_key` convention
(always `"result"` for scalar, `"data"` for arrays) would simplify
`validate_parity` across springs.

### 3.3 For NestGate

The cross-atomic pipeline (N3: BearDog hash → NestGate store → retrieve →
verify) depends on `storage.store` and `storage.retrieve` accepting and
returning consistent key-value structures. The JSON-RPC schema for these
methods should be documented if not already.

### 3.4 For Songbird / biomeOS

**PG-03: Name-based discovery** is still the primary gap. Socket resolution
uses `discover_primal()` with canonical name strings. True capability-based
discovery (`capability.resolve` → socket path) would decouple springs from
primal identity. This becomes critical when primals are deployed from
plasmidBin on clean machines where paths differ.

### 3.5 For toadStool

**PG-05: No active compute dispatch via IPC.** wetSpring uses barraCuda
directly. For sovereign dispatch (coralReef native), compute requests would
route through toadStool. This gap becomes relevant at NUCLEUS deployment.

### 3.6 For Squirrel / neuralSpring

wetSpring declares `inference.complete` in validation_capabilities but the
IPC schema is not yet standardized. Once Squirrel's `inference.complete`
method has a stable request/response schema, wetSpring's guideStone can
validate that AI inference over IPC matches local inference.

---

## 4. Composition Patterns for NUCLEUS Deployment via biomeOS

### 4.1 The Deploy Story (Clean Machine)

```
1. Clone infra/plasmidBin/          (static ecoBin binaries)
2. Start biomeOS                     (orchestrator)
3. biomeos deploy --graph nucleus.toml
   → spawns: beardog, songbird, toadstool, barracuda, coralreef,
             nestgate, rhizocrypt, loamspine, sweetgrass (9 UDS sockets)
4. Run primalspring_guidestone       (base certification — 6 layers)
   → exit 0: composition is sound
5. Run wetspring_guidestone          (domain certification)
   → Bare: 9/9 pass (no NUCLEUS needed)
   → NUCLEUS: N0 liveness → N1 manifest → N2 domain science → N3 cross-atomic
   → exit 0: peer-reviewed life science validated through NUCLEUS
```

### 4.2 Fragment Composition

wetSpring's deploy graph (`wetspring_science_nucleus.toml`) uses fragment
references: `tower_atomic`, `node_atomic`, `nest_atomic`, `meta_tier`.
Custom profiles should use `resolve = true` in `[graph.metadata]` to
inherit from fragments. This pattern is validated by Exp400 (136/136
proto-nucleate alignment checks).

### 4.3 Bonding Policy

Cross-atomic compositions declare bond type (`Covalent`, `Ionic`), trust
model, and encryption tiers per atomic boundary. wetSpring's facade
(`barracuda/src/facade/mod.rs`) reports bonding capabilities in composition
health responses.

### 4.4 Neural API Integration

biomeOS provides Neural API routing for cross-atomic IPC. The pattern:
1. Spring registers capabilities via `capability.list`
2. biomeOS routes incoming requests to the correct primal by capability
3. Spring calls primals by capability domain, not identity
4. Results flow back through Neural API with provenance metadata

wetSpring's provenance trio (rhizoCrypt DAG, loamSpine journal, sweetGrass
braids) is wired but endpoints are not yet live (PG-02).

---

## 5. What Sibling Springs Should Absorb

### 5.1 guideStone Binary Pattern

```rust
use primalspring::composition::{CompositionContext, validate_parity, validate_liveness};
use primalspring::validation::ValidationResult;
use primalspring::tolerances as ps_tol;

fn main() {
    let mut v = ValidationResult::new("myspring guideStone");

    // Bare mode (B0 + B1) — always runs, no NUCLEUS needed
    v.section("B0 — Bare Science");
    validate_bare_science(&mut v);  // analytical baselines only
    v.section("B1 — Tolerance Provenance");
    validate_tolerances(&mut v);

    // NUCLEUS mode (N0-N3) — requires live primals
    v.section("N0 — NUCLEUS Liveness");
    let mut ctx = CompositionContext::from_live_discovery_with_fallback();
    let alive = validate_liveness(&mut ctx, &mut v, &["tensor", "security"]);
    if alive == 0 {
        v.finish();
        std::process::exit(if v.exit_code() == 0 { 2 } else { 1 });
    }

    // Domain science IPC checks with analytical baselines
    v.section("N1-N2 — Domain Science");
    validate_parity(&mut ctx, &mut v,
        "my_check", "tensor", "stats.mean",
        serde_json::json!({"data": [1.0, 2.0, 3.0]}),
        "result", 2.0, ps_tol::IPC_ROUND_TRIP_TOL);

    v.finish();
    std::process::exit(v.exit_code());
}
```

### 5.2 CONSUMED_CAPABILITIES Structure

Organize consumed methods by primal surface and version:
1. v0.9.15 canonical surface (barraCuda 33 methods)
2. Legacy surface (pending migration)
3. Tower/Nest/Meta methods (other primals)

### 5.3 Analytical Baselines > Python Dependencies

For bare mode, use closed-form analytical baselines (Shannon H', exact
matmul, population variance) rather than reading Python output files.
This makes the guideStone self-contained and environment-agnostic.

---

## 6. Evolution Roadmap — What Comes Next

| Level | Status | Next Action |
|-------|--------|-------------|
| Level 3 (bare) | **CERTIFIED** | Maintain |
| Level 4 (NUCLEUS) | Blocked | Deploy barraCuda + BearDog + NestGate from plasmidBin |
| Level 5 (certified) | Blocked | Full N0-N3 pass against live NUCLEUS |
| Level 6 (deployment) | Blocked | biomeOS deploys graph, guideStone validates externally |

**Blockers for Level 4:**
1. Deploy barraCuda ecobin from plasmidBin (binary exists, needs socket startup)
2. Deploy BearDog + NestGate for cross-atomic pipeline
3. Verify spectral/linalg methods served over IPC (PG-10 routing fix helps)
4. Standardize `result_key` convention across barraCuda JSON-RPC responses

---

*This handoff follows `WETSPRING_{VERSION}_{TOPIC}_HANDOFF_{DATE}.md` convention
and feeds back to primalSpring per NUCLEUS_SPRING_ALIGNMENT.md feedback protocol.*
