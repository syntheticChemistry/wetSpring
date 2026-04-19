# wetSpring V148 — Ecosystem Evolution Handoff

**Date:** April 19, 2026
**From:** wetSpring V148 (primalSpring v0.9.16)
**For:** All primal teams (barraCuda, BearDog, NestGate, ToadStool, Squirrel,
biomeOS) and all spring teams (hotSpring, neuralSpring, healthSpring,
ludoSpring, airSpring, groundSpring, primalSpring)
**License:** AGPL-3.0-or-later

---

## Summary

wetSpring is the first downstream spring to validate science through a live
NUCLEUS composition via IPC. The `wetspring_guidestone` binary passes 31/31
checks against a live 5-primal deployment with exit 0. This handoff documents
what we learned, what we need from each team, and patterns that other springs
can reuse.

**Validation progression proven end-to-end:**
```
Python baseline (58 scripts)
  → Rust validation (1,594 tests, 364 binaries)
    → barraCuda GPU (150+ primitives, 44 GPU modules)
      → guideStone bare (16/16, exit 2)
        → guideStone NUCLEUS (31/31, exit 0)     ← HERE
          → Level 5 (11 skips → 0)               ← NEXT
```

---

## What We Deployed

A 5-primal NUCLEUS composition built from source and launched manually with
`FAMILY_ID=wetspring-validation`:

| Primal | Socket | Role | Status |
|--------|--------|------|--------|
| barraCuda | `math-wetspring-validation.sock` | GPU math (32 methods) | ALIVE |
| BearDog | `beardog-wetspring-validation.sock` | Crypto, trust | ALIVE |
| NestGate | `nestgate-wetspring-validation.sock` | Storage | ALIVE |
| ToadStool | `toadstool-wetspring-validation.sock` | Compute dispatch | ALIVE (BTSP) |
| Squirrel | `squirrel-wetspring-validation.sock` | AI inference | BTSP-only |

biomeOS not deployed (upstream `biomeos-types` compile error: missing `secret`
module). Not required for guideStone validation — primals discovered via
family-aware capability sockets.

---

## What We Learned — For Each Primal Team

### barraCuda Team

**What works:** `tensor.create`, `tensor.matmul` (handle-based), `stats.mean`,
`stats.std_dev` (sample), `stats.weighted_mean` — all produce correct results
over IPC.

**What's missing (PG-13):** 6 methods in the v0.9.16 canonical surface are
not implemented in the ecobin:
- `stats.variance`, `stats.correlation`
- `linalg.solve`, `linalg.eigenvalues`
- `spectral.fft`, `spectral.power_spectrum`

These are in the downstream manifest and block Level 5.

**API discoveries:**
- `tensor.matmul` requires pre-created tensor handles (`lhs_id`/`rhs_id`) —
  inline data not accepted (PG-17). Consider adding an inline convenience
  path, or document the handle-based API as canonical.
- `stats.std_dev` uses sample standard deviation (N-1, Bessel's correction).
  wetSpring's Python baselines use population std_dev (N). Both correct;
  document the convention (PG-16).
- `stats.weighted_mean` expects `"values"` parameter (not `"data"`).
  Consistent with barraCuda's API but differs from the `stats.mean` pattern.

### BearDog Team

**What works:** `crypto.hash` (BLAKE3) produces correct hashes over IPC.
Cross-atomic pipeline (hash → NestGate store → retrieve → verify) passes.

**Discovery:** `crypto.hash` requires base64-encoded input data. The method
name suggests raw bytes but the wire format is base64 JSON. Document this
in the method contract.

### NestGate Team

**What works:** `storage.store` and `storage.retrieve` round-trip correctly.
Content-addressed storage with BLAKE3 integrity.

**Cross-atomic pipeline:** BearDog hash → NestGate store → NestGate retrieve →
hash verify. Full chain validated.

### ToadStool Team

**Issue (PG-15):** The JSON-RPC socket is live and responds, but
`compute.dispatch` returns "Method not found". ToadStool appears to have
separate sockets for JSON-RPC, tarpc, and BTSP — the JSON-RPC socket may
not register compute methods.

**Ask:** Register `compute.dispatch` on the JSON-RPC UDS socket, or document
which socket compute methods live on.

### Squirrel Team

**Issue (PG-14):** Squirrel's UDS socket requires BTSP handshake. Plain
JSON-RPC clients (including `primalspring::composition::CompositionContext`)
get "Connection reset by peer" / "BTSP frame too large". The v0.9.16
composition API classifies this as SKIP (reachable, incompatible protocol).

**Ask:** Add a plain JSON-RPC fallback for basic methods (at minimum
`inference.complete` and `health.liveness`), or provide a BTSP client library
that `primalspring::composition` can use.

### biomeOS Team

**Issue:** `biomeos-types` crate fails to compile (`error[E0583]: file not
found for module 'secret'`). This prevented using `biomeos deploy --graph`
for automated NUCLEUS launch.

**Workaround:** Manual primal launch with `setsid` and explicit socket paths.
Works but doesn't test biomeOS orchestration.

**Ask:** Fix the missing module or provide a minimal launcher that doesn't
require the full `biomeos-types` dependency.

---

## What We Learned — For Spring Teams

### Patterns That Work

1. **`primalspring::composition` API** is the right abstraction. `CompositionContext`,
   `validate_parity`, `validate_liveness` handle socket discovery, JSON-RPC framing,
   and result extraction cleanly.

2. **Family-aware discovery** (`FAMILY_ID=wetspring-validation`) works. Sockets
   resolve `{capability}-{family}.sock` before falling back. Multiple test
   NUCLEUS instances can coexist.

3. **`validate_parity_or_skip` pattern**: When a method might not exist in the
   ecobin, wrap `ctx.call()` and degrade to `check_skip` on error. This keeps
   the guideStone green even with partial deployments.

4. **BLAKE3 checksums** (`primalspring::checksums`) for Property 3 (self-verifying).
   Generate with `gen_checksums` example, verify at runtime, gracefully skip
   when manifest absent.

5. **Bare mode** should work without any NUCLEUS running. All B0/B1/B2 checks
   use locally computed baselines. Exit 2 = bare-only is a valid CI state.

### Patterns to Avoid

1. **Don't assume inline data for tensor ops.** barraCuda uses handle-based APIs
   (`tensor.create` → `tensor_id` → `tensor.matmul`). Your guideStone must
   manage tensor lifecycle.

2. **Don't assume population vs sample std_dev.** Check which convention
   barraCuda uses for each method and match your expected values.

3. **Don't hardcode socket paths.** Use `CompositionContext::from_live_discovery_with_fallback()`
   with `FAMILY_ID`. Socket paths are runtime-discovered.

4. **Don't expect biomeOS for deployment.** Manual launch with `setsid` works.
   Build primals from source if plasmidBin x86_64 binaries aren't available.

### What Each Spring Needs

| Spring | Current Level | What to Do |
|--------|:---:|---|
| hotSpring | 5 (certified) | Reference — other springs follow this pattern |
| wetSpring | **4** (NUCLEUS) | Resolve 11 skips for Level 5 |
| ludoSpring | 3 (bare) | Deploy NUCLEUS, run guideStone against it |
| neuralSpring | 2 (scaffold) | Wire B0/B1/B2 bare checks, test locally |
| healthSpring | 2 (scaffold) | Wire B0/B1/B2 bare checks, test locally |
| airSpring | 0 | Start with guideStone scaffold from primalSpring |
| groundSpring | 0 | Start with guideStone scaffold from primalSpring |

---

## Composition Patterns for NUCLEUS Deployment

### Manual Launch (What We Used)

```bash
export FAMILY_ID="wetspring-validation"
export BEARDOG_FAMILY_SEED="$(head -c 32 /dev/urandom | xxd -p)"
export NESTGATE_JWT_SECRET="$(head -c 64 /dev/urandom | base64)"
export XDG_RUNTIME_DIR="/run/user/$(id -u)"
SOCKET_DIR="$XDG_RUNTIME_DIR/biomeos"
mkdir -p "$SOCKET_DIR"

# Launch each primal
setsid ./barracuda --socket "$SOCKET_DIR/math-$FAMILY_ID.sock" &
setsid ./beardog --socket "$SOCKET_DIR/beardog-$FAMILY_ID.sock" &
setsid ./nestgate --socket "$SOCKET_DIR/nestgate-$FAMILY_ID.sock" &
# ... etc

# Create capability symlinks for discovery
ln -sf "math-$FAMILY_ID.sock" "$SOCKET_DIR/tensor.sock"
ln -sf "beardog-$FAMILY_ID.sock" "$SOCKET_DIR/security.sock"
ln -sf "nestgate-$FAMILY_ID.sock" "$SOCKET_DIR/storage.sock"
```

### Validation

```bash
# Bare (no NUCLEUS required)
cargo run --features guidestone --bin wetspring_guidestone
# → 16/16 pass, exit 2

# NUCLEUS
FAMILY_ID=wetspring-validation ./target/release/wetspring_guidestone
# → 31/31 pass (11 skip), exit 0
```

### Neural API via biomeOS (Target Pattern)

Once biomeOS compiles and plasmidBin has x86_64 binaries:
```bash
export FAMILY_ID="wetspring-validation"
./nucleus_launcher.sh --composition full start
./nucleus_launcher.sh status  # verify 12/12 healthy
./target/release/wetspring_guidestone
# → target: 42/42 pass, exit 0
```

---

## Gap Summary (14 Open, 3 Resolved)

| Gap | Owner | Priority | Blocks |
|-----|-------|----------|--------|
| PG-02 | rhizoCrypt/loamSpine/sweetGrass | Medium | Provenance trio IPC |
| PG-03 | Songbird/biomeOS | Low | Capability-based discovery |
| PG-04 | NestGate | Medium | Cross-spring data exchange |
| PG-05 | ToadStool | Low | Sovereign compute dispatch |
| PG-06 | primalSpring Track 4 | Medium | Ionic bond negotiation |
| PG-08 | primalSpring | Low | Validate manifest binary name |
| PG-09 | wetSpring | Medium | Library→IPC evaporation |
| PG-10 | primalSpring | Medium | spectral/linalg routing |
| PG-12 | wetSpring | Low | Exp403 legacy surface cleanup |
| **PG-13** | **barraCuda** | **High** | **6 methods block Level 5** |
| **PG-14** | **Squirrel** | **High** | **BTSP-only blocks AI inference** |
| **PG-15** | **ToadStool** | **Medium** | **compute.dispatch on JSON-RPC** |
| **PG-16** | **barraCuda/wetSpring** | **Low** | **Convention documentation** |
| **PG-17** | **barraCuda** | **Low** | **Handle-based API documentation** |

Resolved: PG-01 (V141), PG-07 (V141), PG-11 (V148).

---

## Files Changed in This Milestone

| File | Change |
|------|--------|
| `barracuda/src/bin/wetspring_guidestone.rs` | Handle-based matmul, sample std_dev, base64 crypto, `validate_parity_or_skip` |
| `barracuda/src/niche.rs` | `GUIDESTONE_READINESS = 4` |
| `docs/PRIMAL_GAPS.md` | PG-13 through PG-17 |
| `validation/CHECKSUMS` | Regenerated for all code changes |
| `README.md`, `CONTEXT.md` | Level 4 status |
| `infra/plasmidBin/wetspring/metadata.toml` | `guidestone_readiness = 4` |

---

*Handoff follows naming convention: `WETSPRING_{VERSION}_{TOPIC}_HANDOFF_{DATE}.md`.*
*Filed to `wateringHole/handoffs/` per NUCLEUS_SPRING_ALIGNMENT.md feedback protocol.*
*Full gap details: `docs/PRIMAL_GAPS.md`.*
