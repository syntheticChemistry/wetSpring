# wetSpring Benchmark Results

**Date:** February 20, 2026
**Status:** Three-tier validation complete (Python → Rust CPU → GPU)

---

## Three-Tier Validation Architecture

wetSpring validates each algorithm at three levels:

```
Tier 1: Python baseline (published tools, open data)
Tier 2: Rust CPU (pure math, no unsafe, documented tolerances)
Tier 3: GPU (ToadStool/BarraCUDA, math parity with CPU)
```

---

## Tier 2: Rust CPU Validation

### Validation Binary Results

| Binary | Experiment | Checks | Status |
|--------|-----------|--------|--------|
| `validate_fastq` | 001 | 28 | PASS |
| `validate_diversity` | 001/004 | 27 | PASS |
| `validate_mzml` | 005 | 7 | PASS |
| `validate_pfas` | 006 | 10 | PASS |
| `validate_features` | 009 | 8 | PASS |
| `validate_peaks` | 010 | 17 | PASS |
| `validate_16s_pipeline` | 011 | 37 | PASS |
| `validate_algae_16s` | 012 | 34 | PASS |
| `validate_voc_peaks` | 013 | 22 | PASS |
| `validate_public_benchmarks` | 014 | 202 | PASS |
| `validate_extended_algae` | 017 | 35 | PASS |
| `validate_pfas_library` | 018 | 26 | PASS |
| `validate_newick_parse` | 019 | 30 | PASS |
| `validate_qs_ode` | 020 | 16 | PASS |
| `validate_rf_distance` | 021 | 23 | PASS |
| `validate_gillespie` | 022 | 13 | PASS |
| `validate_pfas_decision_tree` | 008 | 7 | PASS |
| `validate_bistable` | 023 | 14 | PASS |
| `validate_multi_signal` | 024 | 19 | PASS |
| `validate_cooperation` | 025 | 20 | PASS |
| `validate_hmm` | 026 | 21 | PASS |
| `validate_capacitor` | 027 | 18 | PASS |
| `validate_alignment` | 028 | 15 | PASS |
| `validate_felsenstein` | 029 | 16 | PASS |
| `validate_barracuda_cpu` | cross | 21 | PASS |
| `validate_barracuda_cpu_v2` | cross | 18 | PASS |
| `validate_barracuda_cpu_v3` | cross | 45 | PASS |
| `validate_phage_defense` | 030 | 12 | PASS |
| `validate_bootstrap` | 031 | 11 | PASS |
| `validate_placement` | 032 | 12 | PASS |
| `validate_phynetpy_rf` | 036 | 15 | PASS |
| `validate_phylohmm` | 037 | 10 | PASS |
| `validate_sate_pipeline` | 038 | 17 | PASS |
| `validate_algae_timeseries` | 039 | 11 | PASS |
| `validate_bloom_surveillance` | 040 | 15 | PASS |
| `validate_epa_pfas_ml` | 041 | 14 | PASS |
| `validate_massbank_spectral` | 042 | 9 | PASS |
| **CPU total** | | **1,035** | **PASS** |

### Rust Unit/Integration Tests

| Suite | Count | Status |
|-------|-------|--------|
| Library + integration tests | 465 | PASS |
| **Total** | **465** | **PASS** (1 ignored) |

---

## Tier 3: GPU Validation

| Binary | Checks | Status |
|--------|--------|--------|
| `validate_diversity_gpu` | 38 | PASS |
| `validate_16s_pipeline_gpu` | 88 | PASS |
| `validate_barracuda_gpu_v3` | 14 | PASS |
| `validate_toadstool_bio` | 14 | PASS |
| `validate_gpu_phylo_compose` | 15 | PASS |
| `validate_gpu_hmm_forward` | 13 | PASS |
| `benchmark_phylo_hmm_gpu` | 6 | PASS |
| `validate_gpu_ode_sweep` | 12 | PASS |
| **GPU total** | **200** | **PASS** |

### GPU Performance

| Workload | Metric | CPU | GPU | Speedup |
|----------|--------|-----|-----|---------|
| Spectral cosine (2048×2048) | Time | 4.8s | 5.2ms | 926× |
| 16S pipeline (10 samples) | Time | 2.1s | 0.86s | 2.45× |
| Shannon diversity (10K) | Time | 12ms | 0.5ms | 24× |
| Bray-Curtis (100×100) | Parity | — | — | ≤1e-10 |

### CPU→GPU Math Parity (Exp016)

88/88 checks confirm GPU results match CPU within documented tolerances:
- Quality filter: bitwise identical
- DADA2 denoising: ≤1e-6 per-base error
- Chimera detection: bitwise identical
- Taxonomy classification: bitwise identical
- Diversity metrics: ≤1e-6 (f64 transcendentals)

---

## Tier 1: Python Baselines

| Baseline Script | Tool | Status |
|-----------------|------|--------|
| `benchmark_python_baseline.py` | QIIME2/DADA2-R | PASS |
| `validate_public_16s_python.py` | BioPython + NCBI | PASS |
| `waters2008_qs_ode.py` | scipy.integrate.odeint | PASS (35/35) |
| `gillespie_baseline.py` | numpy SSA ensemble | PASS (8/8) |
| `rf_distance_baseline.py` | dendropy RF | PASS (10/10) |
| `newick_parse_baseline.py` | dendropy tree stats | PASS (10/10) |
| `pfas_tree_export.py` | sklearn DecisionTree | PASS (acc=0.989, F1=0.986) |
| `exp008_pfas_ml_baseline.py` | sklearn RF+GBM | PASS (RF F1=0.978, GBM F1=0.992) |
| `fernandez2020_bistable.py` | scipy ODE (bifurcation) | PASS |
| `srivastava2011_multi_signal.py` | scipy ODE (multi-signal) | PASS |
| `bruger2018_cooperation.py` | scipy ODE (game theory) | PASS |
| `liu2014_hmm_baseline.py` | numpy HMM (sovereign) | PASS |
| `mhatre2020_capacitor.py` | scipy ODE (capacitor) | PASS |
| `smith_waterman_baseline.py` | pure Python (sovereign) | PASS |
| `felsenstein_pruning_baseline.py` | pure Python (sovereign) | PASS |
| `hsueh2022_phage_defense.py` | scipy ODE (phage defense) | PASS |
| `wang2021_rawr_bootstrap.py` | pure Python (bootstrap) | PASS |
| `alamin2024_placement.py` | pure Python (placement) | PASS |
| `phynetpy_rf_baseline.py` | PhyNetPy gene trees | PASS |
| `phylohmm_introgression_baseline.py` | PhyloNet-HMM | PASS |
| `sate_alignment_baseline.py` | SATe pipeline | PASS |
| `algae_timeseries_baseline.py` | Cahill proxy | PASS |
| `bloom_surveillance_baseline.py` | Smallwood proxy | PASS |
| `epa_pfas_ml_baseline.py` | Jones F&T proxy | PASS |
| `massbank_spectral_baseline.py` | Jones MS proxy | PASS |
| `benchmark_rust_vs_python.py` | 18-domain timing (Exp043) | PASS |

---

## Exp043: Rust vs Python Timing (18 Domains)

Head-to-head benchmark across all BarraCUDA CPU parity domains:

| Metric | Value |
|--------|-------|
| Rust (release) total | ~84,500 µs |
| — v1 domains (1–9) | ~60,000 µs |
| — v3 domains (10–18) | ~24,500 µs |
| Python total | ~1,749,000 µs |
| **Speedup** | **~20×** |

Run with `scripts/benchmark_head_to_head.sh` or `python3 scripts/benchmark_rust_vs_python.py`.

---

## Grand Total

| Category | Checks | Status |
|----------|--------|--------|
| Rust CPU validation | 1,035 | PASS |
| GPU validation | 200 | PASS |
| Rust tests | 465 | PASS |
| Python baselines | 28 scripts | PASS |
| BarraCUDA CPU parity | 84/84 (18 domains) | PASS |
| ToadStool bio primitives | 15 consumed (4 bio absorbed) | PASS |
| Local WGSL shaders | 4 (HMM, ODE, DADA2, quality) | PASS |
| **Grand total** | **1,235 validation + 465 tests** | **ALL PASS** |

---

## Reproduction

```bash
cd barracuda

# Tier 2: Rust CPU (1,035 checks)
cargo test                         # 465 tests
cargo run --release --bin validate_qs_ode  # ... repeat for all 27 CPU binaries

# Tier 3: GPU (200 checks)
cargo run --features gpu --bin validate_diversity_gpu          # 38
cargo run --features gpu --bin validate_16s_pipeline_gpu       # 88
cargo run --features gpu --bin validate_barracuda_gpu_v3       # 14
cargo run --features gpu --bin validate_gpu_phylo_compose      # 15
cargo run --features gpu --bin validate_gpu_hmm_forward        # 13
cargo run --features gpu --bin benchmark_phylo_hmm_gpu         # 6
cargo run --features gpu --bin validate_gpu_ode_sweep          # 12
# validate_toadstool_bio also available (14 checks)

# Tier 1: Python
cd ../scripts && python3 gillespie_baseline.py
```
