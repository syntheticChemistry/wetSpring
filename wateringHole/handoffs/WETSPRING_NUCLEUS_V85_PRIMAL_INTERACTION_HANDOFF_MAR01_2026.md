# wetSpring → NUCLEUS V85 Primal Interaction Handoff

**Date:** March 1, 2026
**From:** wetSpring V85 (Exp256-258, 6,626+ checks, 1,210+ Rust tests)
**To:** biomeOS, NestGate, Songbird, ToadStool

---

## What Happened

wetSpring V85 extended the validated science pipeline (V84: 256 experiments,
6,569+ checks, Python parity proven, GPU portability proven) with three new
experiments that exercise NUCLEUS primal interaction:

| Exp | Name | Checks | Result |
|-----|------|--------|--------|
| 256 | EMP-Scale Anderson Atlas | 35 | PASS — 30,002 samples, 14 biomes |
| 257 | NUCLEUS Data Pipeline | 9 | PASS — three-tier routing validated |
| 258 | NUCLEUS Tower-Node | 13 | PASS — all primals READY, 3.2× IPC overhead |

**Total V85:** 259 experiments, 6,626+ validation checks.

---

## What We Proved

### 1. The Science Scales

Exp256 processed 30,002 EMP-calibrated samples across 14 EMPO biome categories
in 55ms on CPU alone. The Anderson-QS classifier confirms Paper 01's prediction:
all natural 3D biomes produce extended states (r > midpoint).

The pipeline is:
```
Community → Shannon H' → Pielou J → W = 0.5 + 14.5×J → Anderson 3D → r → classify
```

This is the exact pipeline that needs to run on real EMP data through NUCLEUS.

### 2. The IPC Is Bit-Identical

Exp258 measured IPC dispatch overhead at 3.2× (0.86µs → 2.74µs per Shannon call).
This is negligible — Anderson spectral dominates at 500ms. JSON-RPC serialization
adds zero math error. The dispatch architecture is validated for production science.

### 3. All Primals Are Installed

Exp258 found all six primal binaries on Eastgate:
- biomeOS v0.1.0 (built)
- BearDog, Songbird, ToadStool, NestGate, Squirrel (all in `~/.local/bin/`)
- All four NUCLEUS modes READY: Tower, Node, Nest, Full

---

## What Each Primal Needs to Evolve

### biomeOS

1. **Batch `capability.call` dispatch** — wetSpring sends 30K samples per
   request. The Neural API should accept `science.diversity_batch` with an
   array of count vectors and return an array of results.
2. **Progress reporting** — for long-running science workloads, the Neural
   API should expose `metrics.progress` showing completion percentage.
3. **Multi-gate Plasmodium** — for SRA atlas (Axis 1C), DADA2 workloads
   need to distribute across towers. Plasmodium should route
   `science.dada2_process` to available Node Atomics.

### NestGate

1. **BIOM format parser** — EMP OTU tables are HDF5-BIOM format. NestGate
   needs a Rust HDF5 reader (hdf5-rust or pure Rust) or TSV conversion.
2. **HTTP bulk fetch** — `storage.fetch_url` for direct download from
   `ftp.microbio.me/emp/release1/` and Qiita.
3. **SRA prefetch integration** — for KBS LTER and longitudinal atlas,
   NestGate needs `ncbi.sra_prefetch` that wraps `prefetch` and `fasterq-dump`.
4. **Content-addressed storage** — EMP tables should be stored on Westgate
   ZFS via CAS (BearDog-signed, NestGate-managed).

### Songbird

1. **Multi-gate discovery** — when LAN mesh is active, Songbird should
   discover primals across gates (Eastgate ↔ Strandgate ↔ Westgate).
2. **Capability routing** — `discovery.find("science.diversity_batch")`
   should return the Node with available GPU capacity.

### ToadStool

1. **Anderson spectral batch** — accept array of (L, W, seed) tuples and
   return array of (r, regime) results in a single GPU dispatch.
2. **Diversity batch** — `science.diversity_batch` accepting N count vectors,
   fused into a single GPU launch for all N samples.
3. **Unidirectional streaming for atlas** — chain diversity → J → W → Anderson
   in a single GPU pipeline without CPU round-trips (Exp255 proves the pattern).

---

## Activation Sequence

```bash
# Step 1: Start NUCLEUS on Eastgate
biomeos nucleus start --mode node --node-id eastgate

# Step 2: Start wetSpring science primal
cd wetSpring && cargo run --release --features ipc --bin wetspring_server

# Step 3: Verify three-tier routing
cd wetSpring && cargo run --release --bin validate_nucleus_data_pipeline

# Step 4: Run EMP atlas through NUCLEUS
cd wetSpring && cargo run --release --bin validate_emp_anderson_atlas

# Step 5: Enable NestGate data routing
WETSPRING_DATA_PROVIDER=nestgate cargo run --release --bin validate_nucleus_data_pipeline
```

---

## Data Acquisition Path

```
NestGate                     wetSpring                  ToadStool
   │                            │                           │
   │ ← storage.fetch_url ──────┤                           │
   │   (EMP BIOM table)        │                           │
   │ ── BIOM → count vectors ──→                           │
   │                            │ ── science.diversity ────→│
   │                            │    (30K samples batch)    │
   │                            │ ←── Shannon, Simpson ─────│
   │                            │                           │
   │                            │ ── science.anderson ─────→│
   │                            │    (W values batch)       │
   │                            │ ←── r, regime ────────────│
   │                            │                           │
   │ ← storage.store ──────────┤ (results to Westgate ZFS) │
```

---

## Files Created

| File | Purpose |
|------|---------|
| `barracuda/src/bin/validate_emp_anderson_atlas.rs` | Exp256 binary |
| `barracuda/src/bin/validate_nucleus_data_pipeline.rs` | Exp257 binary |
| `barracuda/src/bin/validate_nucleus_tower_node.rs` | Exp258 binary |
| `experiments/256_emp_anderson_atlas.md` | Exp256 protocol |
| `experiments/257_nucleus_data_pipeline.md` | Exp257 protocol |
| `experiments/258_nucleus_tower_node.md` | Exp258 protocol |

---

## Science Ready

The math is proven (V84). The pipeline scales (V85). The primals are installed.
The IPC is bit-identical. The data is public. Start NUCLEUS and run real data.
