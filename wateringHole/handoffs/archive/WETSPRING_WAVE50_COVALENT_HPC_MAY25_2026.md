# wetSpring Wave 50 — Covalent HPC Readiness

**Date**: May 25, 2026
**From**: wetSpring (southGate)
**To**: primalSpring, downstream springs
**Version**: V187

---

## Sound Off

```
wetSpring Wave 50: NUCLEUS 7/13 on southGate, cross-subnet reachable, covalent ready
```

## Compliance Status

| Mandate | Status |
|---------|--------|
| Post-primordial (zero `target/release/` for primals) | **DONE** (V186) |
| NUCLEUS sharing on gate | **7/13 health-responding** |
| Songbird federation TCP | **0.0.0.0:7700** (LAN-reachable) |
| SONGBIRD_PEERS seeded | Pending — Songbird process unstable (crashes after startup) |
| Cross-gate `capability.call` | Infrastructure ready, Songbird instability blocking |

## NUCLEUS State

| Primal | Status |
|--------|--------|
| Songbird | ALIVE (health: healthy), TCP :7700 bound — **crashes intermittently** |
| ToadStool | ALIVE (socket responding) |
| barraCuda | ALIVE (status: alive) |
| coralReef | ALIVE (alive: true) |
| NestGate | ALIVE (socket, no health.liveness method) |
| sweetGrass | ALIVE (alive: true) |
| petalTongue | ALIVE (alive: true) |
| biomeOS | Process running, socket not connecting |
| BearDog | Process running, socket timing out |
| Squirrel | Process running, socket not found |
| rhizoCrypt | Process running, socket not found |
| loamSpine | **FAILED** — Tokio runtime-in-runtime panic (known upstream) |
| skunkBat | Not in launcher primal list |

## Cross-Subnet Connectivity

| Field | Value |
|-------|-------|
| southGate LAN IP | 192.168.4.29 |
| Subnet | 192.168.4.0/22 |
| eastGate LAN IP | 192.168.1.144 |
| Ping latency | 4ms (via router) |
| TCP :7700 reachable | Yes (Songbird responds 400 Bad Request to raw JSON = alive) |
| TURN relay needed | **No** — routed natively |

The Wave 50 audit noted southGate is on a different subnet than eastGate/ironGate.
Testing confirms cross-subnet routing works through the existing network infrastructure.
No additional configuration needed for covalent mesh.

## Wave 50 Work Items

| Item | Status | Notes |
|------|--------|-------|
| Validate breseq pipeline against live NUCLEUS | **Ready** — NestGate alive, exp381 exists | Needs stable Songbird for provenance trio |
| WS-2 cross-spring data exchange via `nest.sync` | **Path identified** — NestGate alive, biomeOS has `nest.sync` graph | Need orchestration experiment |
| WS-11 variant caller re-measurement | **Pending** — needs toadStool on remote gate | Covalent compute via mesh |
| Document cross-subnet workaround | **Not needed** — routing works natively | Documented above |

## Upstream Blockers

| Issue | Component | Impact |
|-------|-----------|--------|
| Primal process instability | plasmidBin musl binaries on southGate | 7/13 health-responding vs eastGate 12/12 |
| Songbird crash after initial health check | Songbird | Cannot seed SONGBIRD_PEERS durably |
| BearDog socket timeout | BearDog | Tower Atomic incomplete |
| biomeOS socket not connecting | biomeOS | Neural API bootstrap mode |
| loamSpine Tokio panic | loamSpine | Provenance trio blocked |

## Ecosystem Pull (May 25)

Key upstream changes absorbed:
- airSpring: 5 files (Wave 49 post-primordial)
- groundSpring: 4 files
- healthSpring: Wave 49 handoff
- hotSpring: Exp222 reagent pipeline handoff
- ludoSpring: 2 files (39 ins, 29 del)
- barraCuda: showcase cleanup
- coralReef: 6 files (phase2 → primals path updates)
- loamSpine: showcase scripts removed
- nestGate: fuzz target removed
- petalTongue: showcase → fossilRecord
- rhizoCrypt: showcase utils removed
- skunkBat: RUN_ALL.sh removed
- squirrel: specs active → historical
- sweetGrass: showcase utils removed
- toadStool: showcase QUICK_START removed
- sourDough: GitHub Actions notify-plasmidbin workflow
- lithoSpore: CHASSIS.md spec
- sporePrint: 15 files (doc refresh)
