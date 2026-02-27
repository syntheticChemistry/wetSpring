# Python Baseline Manifest

**Generated:** February 27, 2026 (Phase 61 audit — SPDX headers + provenance metadata added)
**Last scripts commit:** Phase 61 audit (hashes updated after adding SPDX-License-Identifier headers)

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
| `validate_exp001.py` | `validate_16s_pipeline` | 001 | QIIME2/DADA2 | `1b9fcc3c6658f48e` |
| `validate_public_16s_python.py` | `validate_public_benchmarks` | 014 | BioPython+NCBI | `e6a5ab3c66d4b7cd` |
| `waters2008_qs_ode.py` | `validate_qs_ode` | 020 | scipy.integrate | `81d532bb914621bd` |
| `gillespie_baseline.py` | `validate_gillespie` | 022 | numpy SSA | `8ed2243125a5f729` |
| `fernandez2020_bistable.py` | `validate_bistable` | 023 | scipy ODE | `afaddbd41db0192f` |
| `srivastava2011_multi_signal.py` | `validate_multi_signal` | 024 | scipy ODE | `67e91e853506274e` |
| `bruger2018_cooperation.py` | `validate_cooperation` | 025 | scipy ODE | `3b32ecc4a6d29753` |
| `mhatre2020_capacitor.py` | `validate_capacitor` | 027 | scipy ODE | `c2494c5b437d1bde` |
| `hsueh2022_phage_defense.py` | `validate_phage_defense` | 030 | scipy ODE | `c8591a143d84a3bb` |
| `algae_timeseries_baseline.py` | `validate_algae_timeseries` | 039 | Cahill proxy | `ec53e6b2458e05ad` |
| `bloom_surveillance_baseline.py` | `validate_bloom_surveillance` | 040 | Smallwood proxy | `ccd0cbdd4f2d7f69` |

## Track 1b: Comparative Genomics & Phylogenetics

| Script | Rust Binary | Exp | Tool | SHA-256 (first 16) |
|--------|-------------|-----|------|---------------------|
| `newick_parse_baseline.py` | `validate_newick_parse` | 019 | dendropy | `9c7214b25760c86e` |
| `rf_distance_baseline.py` | `validate_rf_distance` | 021 | dendropy RF | `fc8f5969c1e04431` |
| `liu2014_hmm_baseline.py` | `validate_hmm` | 026 | numpy HMM | `cd366113f13a7cd0` |
| `smith_waterman_baseline.py` | `validate_alignment` | 028 | pure Python | `c71fd31338975395` |
| `felsenstein_pruning_baseline.py` | `validate_felsenstein` | 029 | pure Python | `cc88aee8c74e5eb1` |
| `wang2021_rawr_bootstrap.py` | `validate_bootstrap` | 031 | pure Python | `3924050f2b396a09` |
| `alamin2024_placement.py` | `validate_placement` | 032 | pure Python | `54de38f7dad4d4d0` |
| `liu2009_neighbor_joining.py` | `validate_neighbor_joining` | 033 | pure Python | `b3012bb9563dfb72` |
| `zheng2023_dtl_reconciliation.py` | `validate_reconciliation` | 034 | pure Python | `5554075856376f1c` |
| `phynetpy_rf_baseline.py` | `validate_phynetpy_rf` | 036 | PhyNetPy | `b3f0f0de914e2d74` |
| `phylohmm_introgression_baseline.py` | `validate_phylohmm` | 037 | PhyloNet-HMM | `25bde4f249085a30` |
| `sate_alignment_baseline.py` | `validate_sate_pipeline` | 038 | SATe pipeline | `5584d52b2f63249c` |

## Track 1c: Deep-Sea Metagenomics (Anderson)

| Script | Rust Binary | Exp | Tool | SHA-256 (first 16) |
|--------|-------------|-----|------|---------------------|
| `anderson2015_rare_biosphere.py` | `validate_rare_biosphere` | 051 | diversity/rarefaction | `15a31f41d7d62526` |
| `anderson2014_viral_metagenomics.py` | `validate_viral_metagenomics` | 052 | dN/dS + diversity | `fb23c9652b58b2d8` |
| `mateos2023_sulfur_phylogenomics.py` | `validate_sulfur_phylogenomics` | 053 | clock/reconciliation | `83d2adec8e22a565` |
| `boden2024_phosphorus_phylogenomics.py` | `validate_phosphorus_phylogenomics` | 054 | clock/reconciliation | `11b8c5a2a8219c30` |
| `anderson2017_population_genomics.py` | `validate_population_genomics` | 055 | ANI/SNP | `5e4a769e3329cb72` |
| `moulana2020_pangenomics.py` | `validate_pangenomics` | 056 | pangenome/enrichment | `dfa4af917716477c` |
| `barracuda_cpu_v4_baseline.py` | `validate_barracuda_cpu_v4` | 057 | 5 Track 1c timing | `98a0eb028fd98338` |

## Track 2: Analytical Chemistry (LC-MS, PFAS)

| Script | Rust Binary | Exp | Tool | SHA-256 (first 16) |
|--------|-------------|-----|------|---------------------|
| `validate_track2.py` | `validate_mzml`, `validate_pfas` | 005-007 | mzML/PFAS | `658134aa4c67b848` |
| `pfas_tree_export.py` | `validate_pfas_decision_tree` | 008 | sklearn DT | `4f816795f6a03b36` |
| `exp008_pfas_ml_baseline.py` | `validate_epa_pfas_ml` | 008/041 | sklearn RF+GBM | `f836ccdc74ceb899` |
| `generate_peak_baselines.py` | `validate_peaks` | 010 | scipy peaks | `e2f88fb9261ad247` |
| `epa_pfas_ml_baseline.py` | `validate_epa_pfas_ml` | 041 | Jones F&T proxy | `b50075efd62d60c2` |
| `massbank_spectral_baseline.py` | `validate_massbank_spectral` | 042 | Jones MS proxy | `9ca49c6fae456c88` |

## Benchmarks

| Script | Rust Binary | Exp | Tool | SHA-256 (first 16) |
|--------|-------------|-----|------|---------------------|
| `benchmark_rust_vs_python.py` | `benchmark_23_domain_timing` | 043/059 | 18-domain timing | `46604acdecd1458f` |
| `benchmark_python_baseline.py` | `benchmark_three_tier` | 069 | Python baseline JSON | `24ef52de7ac56380` |

## Track 3: Drug Repurposing (Fajgenbaum / Every Cure)

| Script | Rust Binary | Exp | Tool | SHA-256 (first 16) |
|--------|-------------|-----|------|---------------------|
| `nmf_drug_disease_pipeline.py` | `validate_nmf_drug_repurposing` | 159 | NMF (Lee & Seung) | `9c9cc26a5cf320b8` |
| `fajgenbaum_pathway_scoring.py` | `validate_fajgenbaum_pathway` | 157 | Pathway scoring | `6e906b0159e0c26a` |
| `transe_knowledge_graph.py` | `validate_knowledge_graph_embedding` | 161 | TransE (KG) | `a8d567aaf686a847` |

## Utility (not baselines)

| Script | Purpose | SHA-256 (first 16) |
|--------|---------|---------------------|
| `run_exp002.py` | SRA data download (Exp002) | `7fbcd4bf2f3ec2b6` |
| `search_ncbi_datasets.py` | NCBI dataset search | `00f1e1133cdd14d4` |
| `fetch_ncbi_phase35.py` | Phase 35 NCBI bulk fetch | `0c55e7fd85ede4e6` |

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
