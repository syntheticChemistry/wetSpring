# wetSpring — Wave 60: Eukaryotic Gate Onboarding

**Date:** 2026-05-28
**Version:** V189
**Gate:** southGate (5800X3D, RTX 4060 + 3090s, 128GB)
**Upstream:** primalSpring Wave 60 — Eukaryotic Gate Onboarding

---

## Summary

southGate achieves 13/13 NUCLEUS processes. biomeOS discovers 1725
capabilities from 21 primal surfaces. Forgejo remotes configured for
periplasm sync. SSH key registration blocked (needs eastGate operator).
wetSpring is the P1 pattern node — proving the eukaryotic gate pattern.

## NUCLEUS Status: 13/13 Processes

| Primal | Status | Transport | Notes |
|--------|--------|-----------|-------|
| biomeOS | ALIVE | UDS + TCP:9000 | 1725 caps / 21 surfaces auto-discovered |
| BearDog | ALIVE | UDS + TCP:9100 | NODE_ID=southgate01 |
| Songbird | ALIVE | UDS + TCP:9200 | Stable (Wave 53 fix) |
| skunkBat | ALIVE | UDS + TCP:9140 | Manual start (launcher `--socket` bug) |
| ToadStool | ALIVE | UDS (compute) + TCP:9400 | Health via compute-nucleus01.sock |
| barraCuda | ALIVE | UDS (math) + TCP:9741 | Manual start (port conflict on 9740) |
| coralReef | ALIVE | UDS (shader) + TCP:9730 | |
| NestGate | ALIVE | UDS + TCP:9500 | 66 methods, BTSP-auth gated |
| rhizoCrypt | ALIVE | UDS (provenance) + TCP:9601 | Manual start (launcher `--unix` bug) |
| loamSpine | ALIVE | UDS + TCP:9700 | Tokio fix confirmed |
| sweetGrass | ALIVE | UDS + TCP:9850 | Manual start (launcher `--socket` bug) |
| petalTongue | BTSP-GATED | UDS + TCP:9900 | Process running, BTSP enforcement rejects probes |
| Squirrel | BTSP-GATED | UDS + TCP:9300 | Process running, BTSP enforcement rejects probes |

**11/13 health-responding** (unauthenticated). **13/13 processes running** with sockets.

## Forgejo Onboarding

| Step | Status |
|------|--------|
| SSH config for `git.primals.eco:2222` | DONE |
| Forgejo remotes on 29 repos | DONE (`cascade-pull.sh --ensure-remotes`) |
| SSH key registration | **BLOCKED** — needs API token from eastGate VPS operator |
| `git push forgejo main` | **BLOCKED** — Permission denied (publickey) |
| WaterFall 20-repo profile dry-run | DONE — `cascade-pull.sh --gate auto --dry-run` works |
| `GATE_NAME=southGate` env var | DONE — hostname `pop-os` doesn't match pattern |

## Blockers for eastGate / primalSpring

1. **Forgejo SSH key**: southGate's `id_ed25519_ecoPrimal.pub` needs registration
   via Forgejo API. We don't have root SSH to VPS or an API token.
   Key: `ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIHrLVoaIaDaUZVae2UCNhmA8YZ3dVo/FuMOdep+0ZnMV ecoPrimal@github`

2. **Manifest issues** (infra/wateringHole/ecosystem_manifest.toml):
   - `songbird` repo key → primals directory is `songBird/` (case mismatch)
   - `nestGate` local_path resolves to `ecoPrimals/.` (should be `primals/nestGate`)
   - `cellMembrane` not cloned on southGate (inner-only repo, needs Forgejo access first)

3. **Launcher `--socket` flag bug**: plasmidBin `nucleus_launcher.sh` passes
   `--socket` to sweetGrass, rhizoCrypt, skunkBat — these binaries only accept
   `--port`. Requires manual start for those 3.

## Pattern Node Status

wetSpring is P1 pattern node per Wave 60 directive:
- 13/13 NUCLEUS processes ✓
- Spring validates against live NUCLEUS ✓ (345 scenarios, 1,962 tests)
- Sync through periplasm: BLOCKED (SSH key registration)
- neuralSpring follows once pattern is proven ✓ (co-resident on southGate)

## Remaining

- SSH key registration for Forgejo push access
- `CASCADE_SYNC_SOURCE=forgejo` once SSH is live
- Run wetSpring experiments against live NUCLEUS for membrane pattern validation
