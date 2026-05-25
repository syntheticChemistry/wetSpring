# wetSpring Wave 49 — Post-Primordial Cleanup

**Date**: May 25, 2026
**From**: wetSpring (southGate)
**To**: primalSpring, downstream springs
**Version**: V186

---

## Wave 49 Compliance

### 1. Cut all primordial patterns — DONE

| Pattern | Action |
|---------|--------|
| `~/.local/bin/` primal stubs | None found (previously cleaned) |
| `cargo install`/`cargo build --release` for deployment | None found |
| `which beardog`/`which songbird` fallback logic | Removed from 4 experiment files |
| `target/release/` path discovery | Removed from 4 experiment files + `composition_nucleus.sh` |
| `wetspring` symlink in plasmidBin | Removed (springs != primals) |
| `phase2/biomeOS/` legacy paths | Updated to `primals/biomeOS/` in 2 files |

**Verification (clean):**

```
for p in beardog songbird biomeos toadstool barracuda coralreef nestgate \
         squirrel rhizocrypt loamspine sweetgrass skunkbat petaltongue; do
    w=$(which $p 2>/dev/null) && echo "STALE: $p -> $w"
done
# Output: (empty — all clean)
```

### 2. NUCLEUS from plasmidBin — DONE (partial)

- Started via `SONGBIRD_FEDERATION_PORT=7700 ./tools/nucleus_launcher.sh start`
- Auto-detected: `infra/plasmidBin/primals/` (flat layout, no triple subdir yet)
- 4/13 primals socket-ready: biomeOS, barraCuda, nestGate, sweetGrass
- Songbird DOWN: rejects `--security-socket` flag from launcher
- BearDog DOWN: socket gone from prior run
- loamSpine DOWN: Tokio runtime-in-runtime panic (known upstream)
- petalTongue DOWN: startup failure

## Code Changes

### New: `barracuda/src/validation/experiments/primal_binary.rs`

Shared post-primordial binary discovery:
1. `{NAME}_BIN` env var (explicit override)
2. `NUCLEUS_BIN_DIR` env var (set by nucleus_launcher.sh)
3. `infra/plasmidBin/primals/{triple}/` (triple-aware)
4. `infra/plasmidBin/primals/` (flat fallback)

No PATH search, no `target/release/`, no `which`.

### Updated: 4 experiment files

| File | Before | After |
|------|--------|-------|
| `exp_nucleus_tower_node.rs` | 50-line `discover_primal_bin` + `which` | 2-line delegate to `primal_binary::discover` |
| `exp_nucleus_v4.rs` | 30-line `discover_biomeos_bin` + `find_on_path` | 2-line delegate |
| `exp_metalforge_v17.rs` | 20-line `discover_biomeos_bin` + PATH scan | 1-line delegate |
| `exp_biomeos_nucleus_v98.rs` | 30-line `discover_biomeos_bin` + `which` | 1-line delegate |

### Updated: `tools/composition_nucleus.sh`

- `find_binary()`: plasmidBin-only, errors hard on missing
- Host-triple detection for `primals/{triple}/` layout
- Removed petalTongue hardcoded `target/release/` path

## Upstream Issues (blocking full NUCLEUS)

| Issue | Component | Workaround |
|-------|-----------|------------|
| `--security-socket` flag rejected | Songbird binary in plasmidBin | Needs plasmidBin Songbird update |
| BearDog socket cleanup on restart | BearDog | Manual socket cleanup |
| Tokio runtime-in-runtime panic | loamSpine | Skip health probe |
| Startup failure | petalTongue | `rm /run/user/$(id -u)/biomeos/petaltongue-*.sock` before restart |

## Ecosystem Pull Summary

Pulled all repos (May 25). Key changes:
- ludoSpring: 12 deletions (stale Wave 48 template)
- sporePrint: 17 files, 315 insertions (doc refresh)
- wateringHole: Wave 49 handoff added
- primalSpring: `26f5182` post-primordial + `8bd1221` 0.0.0.0 bind fix + `0dbfe35` s_covalent_mesh scenario
