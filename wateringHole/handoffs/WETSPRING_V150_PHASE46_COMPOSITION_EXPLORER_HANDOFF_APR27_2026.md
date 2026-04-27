# wetSpring V150 — Phase 46 Composition Explorer: Data Visualization Lane

**Date**: April 27, 2026
**From**: wetSpring V150
**To**: primalSpring, petalTongue, rhizoCrypt, loamSpine, sweetGrass, Songbird, all spring teams
**Trigger**: primalSpring Phase 46 — composition template abstraction
**Lane**: Data exploration & visualization (genome/protein scene graphs, DAG navigation, large data IPC, braid lineage)
**License**: AGPL-3.0-or-later

---

## Summary

wetSpring adopted the Phase 46 composition template library and built an
interactive gene explorer composition (`wetspring_composition.sh`) that
navigates an 8-gene cancer panel through petalTongue scene graphs, records
exploration steps via rhizoCrypt DAG and sweetGrass braids, and exercises
barraCuda IPC math on biological data. The composition ran successfully
against a live 8-primal NUCLEUS launched via `composition_nucleus.sh`.

**Key finding**: petalTongue handles complex scenes well (100 nodes, 41KB,
<1ms), but the entire provenance trio (rhizoCrypt, loamSpine, sweetGrass)
resets UDS connections on JSON-RPC requests — the data exploration lane
cannot track provenance until this is resolved.

---

## What Changed

| Change | Detail |
|--------|--------|
| **Composition adopted** | `wetspring_composition.sh` — gene panel explorer with 6 domain hooks |
| **Tools directory created** | `tools/` with composition_nucleus.sh, nucleus_composition_lib.sh, uds_send.py |
| **Python UDS shim** | `uds_send.py` — socat-free JSON-RPC transport (45 lines) |
| **5 new gaps** | PG-18 (trio UDS reset), PG-19 (scene format), PG-20/21 (socat), PG-22 (Songbird) |
| **15 gaps total** | PG-01–PG-22, 7 resolved |
| **Lib patched** | Local `nucleus_composition_lib.sh` falls back to python3 when socat absent |

---

## Composition Test Results

### petalTongue — Scene Graph Complexity

| Test | Nodes | Payload | Time | Result |
|------|-------|---------|------|--------|
| Gene list (8 genes) | 9 | ~4KB | <1ms | PASS |
| Gene list + detail panel | 15 | ~7KB | <1ms | PASS |
| Stress test (50 nodes) | 51 | 23KB | <1ms | PASS |
| Stress test (100 nodes) | 101 | 41KB | <1ms | PASS |
| Nested hierarchy (3 levels, 7 nodes) | 7 | ~3KB | <1ms | PASS |
| 6 concurrent sessions | 6 scenes | — | — | PASS |

**Finding**: petalTongue's `visualization.render.scene` handles biological
data visualization scenes well at this scale. The scene graph format requires
the `"Text"` single-key enum variant with explicit `x`/`y`/`content` fields
and `"color": {"r","g","b","a"}` object (not array). This is correctly
implemented in `make_text_node` from the composition lib.

**Recommendation for petalTongue**: Document the scene primitive schema more
explicitly. First-time users building scenes by hand will hit the "expected
map with a single key" error if they use `{"type":"text","text":"..."}`.

### barraCuda — IPC Math on Biological Data

| Method | Input | Result | Status |
|--------|-------|--------|--------|
| `stats.mean` | Gene sequence lengths [63,62,64,...] | 63.375 | PASS |
| `stats.std_dev` | Same | 0.744 (sample, N-1) | PASS |
| `stats.correlation` | Self-correlation | 1.0 | PASS |
| `spectral.fft` | 8-point on lengths | Correct complex output | PASS |

**Finding**: barraCuda IPC math works flawlessly for biological data analysis.
Sequence length statistics, correlation, and spectral analysis all return
correct results. No new gaps.

### Provenance Trio — rhizoCrypt / loamSpine / sweetGrass

| Method | Primal | Result |
|--------|--------|--------|
| `dag.session.create` | rhizoCrypt | Connection reset (PG-18) |
| `dag.event.append` | rhizoCrypt | Connection reset |
| `spine.create` | loamSpine | Connection reset |
| `braid.record` | sweetGrass | Connection reset |

**Finding**: All three provenance primals accept UDS connections but
immediately reset them when JSON-RPC is sent. This matches upstream PG-45.
The primals are running (pgrep confirms) and their sockets exist, but they
do not speak JSON-RPC on UDS. This completely blocks the data exploration
lane's provenance features.

**Impact**: The composition degrades gracefully — gene navigation works,
scene rendering works, math IPC works — but no DAG state, no ledger sealing,
no braid provenance. For wetSpring's data exploration lane, provenance is
the key differentiator, so this is the highest-priority gap.

### Other Primals

| Primal | Status | Notes |
|--------|--------|-------|
| BearDog | Alive | BTSP-aware, liveness passes |
| toadStool | Alive | Liveness passes |
| Songbird | Socket timeout | Never creates family-named socket (PG-22) |
| NestGate | Not launched | Not in default PRIMAL_LIST |
| Squirrel | Not launched | Not in default PRIMAL_LIST |

---

## Patterns Discovered

### 1. Python UDS Shim (socat replacement)

On systems without `socat`, a 45-line Python script provides identical
JSON-RPC-over-UDS transport. The local composition lib is patched to
auto-detect and fall back. **Candidate for upstream promotion into
`nucleus_composition_lib.sh`.**

### 2. Scene Complexity Is Not the Bottleneck

petalTongue handles 100-node scenes in <1ms. For biological data
visualization (genome browsers, phylogenetic trees), the scene primitive
set is sufficient. The bottleneck for rich visualization will be:
- Missing primitives: lines/edges for relationship graphs (gene→protein
  links). Currently only `Text` primitives are used; `Line`, `Rect`,
  `Circle` would enable actual graph visualization.
- Layout: the composition must compute all positions in bash; a layout
  engine in petalTongue would help.

### 3. Provenance Trio Is the Critical Path

For data exploration, the value is in the provenance trail: which gene
did I start from? What path did I take? What transformations were applied?
Without DAG/ledger/braid, the composition is just a viewer, not an explorer.
The provenance trio UDS fix is the single most impactful change for all
downstream compositions.

### 4. Large Dataset IPC Pattern

NestGate was not in the default PRIMAL_LIST. When added, the store/retrieve
pattern from the guideStone (V149) should work for moderate payloads. For
truly large datasets (genomic sequences, protein structures), a streaming
IPC pattern may be needed — this is an open design question.

---

## Upstream Drift (Unchanged from V149)

1. **downstream_manifest.toml**: `guidestone_readiness = 3` (should be 4),
   missing `guidestone_properties` field
2. **NUCLEUS_SPRING_ALIGNMENT.md**: wetSpring at V147/Level 3 (should be V150/Level 4)

---

## Files Changed

| File | Change |
|------|--------|
| `tools/wetspring_composition.sh` | New: gene panel explorer composition |
| `tools/nucleus_composition_lib.sh` | Copied + patched socat fallback |
| `tools/composition_nucleus.sh` | Copied from primalSpring |
| `tools/uds_send.py` | New: Python UDS shim |
| `docs/PRIMAL_GAPS.md` | PG-18..22 added, header updated |
| `README.md` | V150 status |
| `CONTEXT.md` | V150 status |
| `CHANGELOG.md` | V150 entry |
| `specs/README.md` | V150 status |
| `whitePaper/baseCamp/README.md` | V150 status |
| `whitePaper/baseCamp/EXTENSION_PLAN.md` | V150 status |
| `whitePaper/README.md` | V150 status |
| `experiments/README.md` | V150 status |
| `barracuda/ABSORPTION_MANIFEST.md` | V150 date |
| `GAPS.md` | V150 gap counts |
| `wateringHole/ECOSYSTEM_LEVERAGE_GUIDE.md` | V150 note |
| `wateringHole/CROSS_SPRING_SHADER_EVOLUTION.md` | V150 note |

---

*Handed back to primalSpring per NUCLEUS_SPRING_ALIGNMENT.md feedback protocol.*
