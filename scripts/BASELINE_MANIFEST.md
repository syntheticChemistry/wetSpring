# Python Baseline Manifest

**Generated:** February 23, 2026 (last significant script update)
**Last scripts commit:** `48fb787` (Phase 28 baseline freeze; hashes updated after adding Reproduction headers)

This manifest maps each Python baseline script to its Rust validation binary,
experiment number, and content hash for reproducibility verification.

## Provenance Contract

Every hardcoded expected value in a Rust validation binary traces back to a
Python baseline script listed here. To re-verify a baseline:

```bash
sha256sum scripts/<script>.py          # must match SHA-256 below
python3 scripts/<script>.py            # re-run; compare output
cargo run --bin <binary>               # Rust must match within tolerance
```

## Reproduction Environment

All baselines were generated and frozen on the following environment:

- **Python:** 3.11+ (CPython)
- **OS:** Ubuntu 22.04 / Debian 12 (x86_64)
- **NumPy:** 1.26+ (where used)
- **SciPy:** 1.12+ (where used — ODE, peaks, integration)
- **Galaxy:** quay.io/galaxy/galaxy:latest (Exp001 only, with qiime2-2026 conda env)
- **Baseline freeze:** commit `48fb787` (2026-02-23)

Standard reproduction for most scripts (no special environment):

```bash
cd /path/to/wetSpring
python3 scripts/<script>.py
```

For Exp001 (Galaxy/QIIME2), see `validate_exp001.py` for container instructions.
For Track 2 (asari/FindPFAS), see `validate_track2.py` for venv instructions.

## Automated Drift Verification

```bash
./scripts/verify_baseline_integrity.sh
```

Compares current SHA-256 hashes against this manifest. Exit 0 = no drift.

---

## Track 1: Microbial Ecology

| Script | Rust Binary | Exp | Tool | SHA-256 (first 16) |
|--------|-------------|-----|------|---------------------|
| `validate_exp001.py` | `validate_16s_pipeline` | 001 | QIIME2/DADA2 | `ee03e9b69bf9dc93` |
| `validate_public_16s_python.py` | `validate_public_benchmarks` | 014 | BioPython+NCBI | `aef16ecca3704ba8` |
| `waters2008_qs_ode.py` | `validate_qs_ode` | 020 | scipy.integrate | `939d4393f00915d7` |
| `gillespie_baseline.py` | `validate_gillespie` | 022 | numpy SSA | `065b9a98ce51e169` |
| `fernandez2020_bistable.py` | `validate_bistable` | 023 | scipy ODE | `9aca89bb4083c2f7` |
| `srivastava2011_multi_signal.py` | `validate_multi_signal` | 024 | scipy ODE | `2141570e2071391f` |
| `bruger2018_cooperation.py` | `validate_cooperation` | 025 | scipy ODE | `ad66ff587e825923` |
| `mhatre2020_capacitor.py` | `validate_capacitor` | 027 | scipy ODE | `66cbbc0532b114cc` |
| `hsueh2022_phage_defense.py` | `validate_phage_defense` | 030 | scipy ODE | `746d4092a4db2c81` |
| `algae_timeseries_baseline.py` | `validate_algae_timeseries` | 039 | Cahill proxy | `876c6b974de73346` |
| `bloom_surveillance_baseline.py` | `validate_bloom_surveillance` | 040 | Smallwood proxy | `e26bbb3c09d61c5b` |

## Track 1b: Comparative Genomics & Phylogenetics

| Script | Rust Binary | Exp | Tool | SHA-256 (first 16) |
|--------|-------------|-----|------|---------------------|
| `newick_parse_baseline.py` | `validate_newick_parse` | 019 | dendropy | `74aff99ae4510fcb` |
| `rf_distance_baseline.py` | `validate_rf_distance` | 021 | dendropy RF | `7f924879f755d3ca` |
| `liu2014_hmm_baseline.py` | `validate_hmm` | 026 | numpy HMM | `8d1e911ae205598e` |
| `smith_waterman_baseline.py` | `validate_alignment` | 028 | pure Python | `9f5019611d0ad0a3` |
| `felsenstein_pruning_baseline.py` | `validate_felsenstein` | 029 | pure Python | `f55e0afcf07fa109` |
| `wang2021_rawr_bootstrap.py` | `validate_bootstrap` | 031 | pure Python | `cb2b7b70d9e6a99a` |
| `alamin2024_placement.py` | `validate_placement` | 032 | pure Python | `8c20e9a30f90fed3` |
| `liu2009_neighbor_joining.py` | `validate_neighbor_joining` | 033 | pure Python | `84cfc553edf5bc9a` |
| `zheng2023_dtl_reconciliation.py` | `validate_reconciliation` | 034 | pure Python | `00faed6493798f42` |
| `phynetpy_rf_baseline.py` | `validate_phynetpy_rf` | 036 | PhyNetPy | `335605ef2bf7c2e3` |
| `phylohmm_introgression_baseline.py` | `validate_phylohmm` | 037 | PhyloNet-HMM | `d88f490dc32ad808` |
| `sate_alignment_baseline.py` | `validate_sate_pipeline` | 038 | SATe pipeline | `5684f28404761095` |

## Track 1c: Deep-Sea Metagenomics (Anderson)

| Script | Rust Binary | Exp | Tool | SHA-256 (first 16) |
|--------|-------------|-----|------|---------------------|
| `anderson2015_rare_biosphere.py` | `validate_rare_biosphere` | 051 | diversity/rarefaction | `3d77ed29f2c325c2` |
| `anderson2014_viral_metagenomics.py` | `validate_viral_metagenomics` | 052 | dN/dS + diversity | `1a127bc3da4a8c88` |
| `mateos2023_sulfur_phylogenomics.py` | `validate_sulfur_phylogenomics` | 053 | clock/reconciliation | `1bc6cba0dc975f7f` |
| `boden2024_phosphorus_phylogenomics.py` | `validate_phosphorus_phylogenomics` | 054 | clock/reconciliation | `589293509eff7097` |
| `anderson2017_population_genomics.py` | `validate_population_genomics` | 055 | ANI/SNP | `9464eb5b65ed6c2e` |
| `moulana2020_pangenomics.py` | `validate_pangenomics` | 056 | pangenome/enrichment | `a137dd02d9ff17a8` |
| `barracuda_cpu_v4_baseline.py` | `validate_barracuda_cpu_v4` | 057 | 5 Track 1c timing | `53cd6534b0e80795` |

## Track 2: Analytical Chemistry (LC-MS, PFAS)

| Script | Rust Binary | Exp | Tool | SHA-256 (first 16) |
|--------|-------------|-----|------|---------------------|
| `validate_track2.py` | `validate_mzml`, `validate_pfas` | 005-007 | mzML/PFAS | `95849562003f193f` |
| `pfas_tree_export.py` | `validate_pfas_decision_tree` | 008 | sklearn DT | `4480bd0ac240cfb8` |
| `exp008_pfas_ml_baseline.py` | `validate_epa_pfas_ml` | 008/041 | sklearn RF+GBM | `a8baf918c72bc4d2` |
| `generate_peak_baselines.py` | `validate_peaks` | 010 | scipy peaks | `cca76ac7ddcef699` |
| `epa_pfas_ml_baseline.py` | `validate_epa_pfas_ml` | 041 | Jones F&T proxy | `afb69af5a39e5c17` |
| `massbank_spectral_baseline.py` | `validate_massbank_spectral` | 042 | Jones MS proxy | `41f2f17d2fc01de5` |

## Benchmarks

| Script | Rust Binary | Exp | Tool | SHA-256 (first 16) |
|--------|-------------|-----|------|---------------------|
| `benchmark_rust_vs_python.py` | `benchmark_23_domain_timing` | 043/059 | 18-domain timing | `d8bb4236df46a5d7` |
| `benchmark_python_baseline.py` | `benchmark_three_tier` | 069 | Python baseline JSON | `a798edee0068fd1e` |

## Track 3: Drug Repurposing (Fajgenbaum / Every Cure)

| Script | Rust Binary | Exp | Tool | SHA-256 (first 16) |
|--------|-------------|-----|------|---------------------|
| `nmf_drug_disease_pipeline.py` | `validate_nmf_drug_repurposing` | 159 | NMF (Lee & Seung) | `c0532b512e044290` |
| `fajgenbaum_pathway_scoring.py` | `validate_fajgenbaum_pathway` | 157 | Pathway scoring | `7038642ff08b2132` |
| `transe_knowledge_graph.py` | `validate_knowledge_graph_embedding` | 161 | TransE (KG) | `2c362100e3692092` |

## Utility (not baselines)

| Script | Purpose | SHA-256 (first 16) |
|--------|---------|---------------------|
| `run_exp002.py` | SRA data download (Exp002) | `b99c0eb11d67bc7c` |
| `search_ncbi_datasets.py` | NCBI dataset search | `703553dd76b87243` |
| `fetch_ncbi_phase35.py` | Phase 35 NCBI bulk fetch | `d9dc06889e219dec` |

---

## Verification

Automated (preferred):

```bash
./scripts/verify_baseline_integrity.sh
```

Manual:

```bash
sha256sum scripts/*.py | sort > /tmp/current_hashes.txt
# Compare against this manifest
```

If a script hash changes, the corresponding Rust validation binary must be
re-verified to ensure the hardcoded expected values still match.
