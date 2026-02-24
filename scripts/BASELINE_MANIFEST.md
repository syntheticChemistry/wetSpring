# Python Baseline Manifest

**Generated:** February 23, 2026
**Last scripts commit:** `48fb787` (Phase 28)

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

---

## Track 1: Microbial Ecology

| Script | Rust Binary | Exp | Tool | SHA-256 (first 16) |
|--------|-------------|-----|------|---------------------|
| `validate_exp001.py` | `validate_16s_pipeline` | 001 | QIIME2/DADA2 | `527b3004bbadac6b` |
| `validate_public_16s_python.py` | `validate_public_benchmarks` | 014 | BioPython+NCBI | `9ceb3e22fd9d683a` |
| `waters2008_qs_ode.py` | `validate_qs_ode` | 020 | scipy.integrate | `486c67baa6bc1d17` |
| `gillespie_baseline.py` | `validate_gillespie` | 022 | numpy SSA | `938fd6f094cbf30a` |
| `fernandez2020_bistable.py` | `validate_bistable` | 023 | scipy ODE | `9aca89bb4083c2f7` |
| `srivastava2011_multi_signal.py` | `validate_multi_signal` | 024 | scipy ODE | `2141570e20713911` |
| `bruger2018_cooperation.py` | `validate_cooperation` | 025 | scipy ODE | `aeca1a95e9835cd9` |
| `mhatre2020_capacitor.py` | `validate_capacitor` | 027 | scipy ODE | `e8de78a9c86ecb6c` |
| `hsueh2022_phage_defense.py` | `validate_phage_defense` | 030 | scipy ODE | `d7782e4bfbf4234d` |
| `algae_timeseries_baseline.py` | `validate_algae_timeseries` | 039 | Cahill proxy | `0bd30bdfd8029799` |
| `bloom_surveillance_baseline.py` | `validate_bloom_surveillance` | 040 | Smallwood proxy | `7b15a2e5af69b05e` |

## Track 1b: Comparative Genomics & Phylogenetics

| Script | Rust Binary | Exp | Tool | SHA-256 (first 16) |
|--------|-------------|-----|------|---------------------|
| `newick_parse_baseline.py` | `validate_newick_parse` | 019 | dendropy | `fc49da36f336a400` |
| `rf_distance_baseline.py` | `validate_rf_distance` | 021 | dendropy RF | `9907cd42a8a90637` |
| `liu2014_hmm_baseline.py` | `validate_hmm` | 026 | numpy HMM | `c814563b6f00ad13` |
| `smith_waterman_baseline.py` | `validate_alignment` | 028 | pure Python | `07c076562311d190` |
| `felsenstein_pruning_baseline.py` | `validate_felsenstein` | 029 | pure Python | `49f2ef090cce21d1` |
| `wang2021_rawr_bootstrap.py` | `validate_bootstrap` | 031 | pure Python | `a9669d9179c192a5` |
| `alamin2024_placement.py` | `validate_placement` | 032 | pure Python | `9859bed2b3b7de28` |
| `liu2009_neighbor_joining.py` | `validate_neighbor_joining` | 033 | pure Python | `84cfc553edf5bc9a` |
| `zheng2023_dtl_reconciliation.py` | `validate_reconciliation` | 034 | pure Python | `00faed6493798f42` |
| `phynetpy_rf_baseline.py` | `validate_phynetpy_rf` | 036 | PhyNetPy | `7c33967ddf5648e6` |
| `phylohmm_introgression_baseline.py` | `validate_phylohmm` | 037 | PhyloNet-HMM | `676d4bfb541058d5` |
| `sate_alignment_baseline.py` | `validate_sate_pipeline` | 038 | SATe pipeline | `4f2d468dd0d8c463` |

## Track 1c: Deep-Sea Metagenomics (Anderson)

| Script | Rust Binary | Exp | Tool | SHA-256 (first 16) |
|--------|-------------|-----|------|---------------------|
| `anderson2015_rare_biosphere.py` | `validate_rare_biosphere` | 051 | diversity/rarefaction | `3d62d1e1c78773dd` |
| `anderson2014_viral_metagenomics.py` | `validate_viral_metagenomics` | 052 | dN/dS + diversity | `376f3810d622cc7e` |
| `mateos2023_sulfur_phylogenomics.py` | `validate_sulfur_phylogenomics` | 053 | clock/reconciliation | `9b9fd60b3cdf13df` |
| `boden2024_phosphorus_phylogenomics.py` | `validate_phosphorus_phylogenomics` | 054 | clock/reconciliation | `7fab34c11cfabf2a` |
| `anderson2017_population_genomics.py` | `validate_population_genomics` | 055 | ANI/SNP | `5536937a3a74e051` |
| `moulana2020_pangenomics.py` | `validate_pangenomics` | 056 | pangenome/enrichment | `c178e9965565ff46` |
| `barracuda_cpu_v4_baseline.py` | `validate_barracuda_cpu_v4` | 057 | 5 Track 1c timing | `53cd6534b0e80795` |

## Track 2: Analytical Chemistry (LC-MS, PFAS)

| Script | Rust Binary | Exp | Tool | SHA-256 (first 16) |
|--------|-------------|-----|------|---------------------|
| `validate_track2.py` | `validate_mzml`, `validate_pfas` | 005-007 | mzML/PFAS | `95849562003f1932` |
| `pfas_tree_export.py` | `validate_pfas_decision_tree` | 008 | sklearn DT | `15b9cf4518dbc4e2` |
| `exp008_pfas_ml_baseline.py` | `validate_epa_pfas_ml` | 008/041 | sklearn RF+GBM | `5752fb23113fbb72` |
| `generate_peak_baselines.py` | `validate_peaks` | 010 | scipy peaks | `a8a15590cd5f04fe` |
| `epa_pfas_ml_baseline.py` | `validate_epa_pfas_ml` | 041 | Jones F&T proxy | `554cbad7750dcbc8` |
| `massbank_spectral_baseline.py` | `validate_massbank_spectral` | 042 | Jones MS proxy | `c0f1abeb489646da` |

## Benchmarks

| Script | Rust Binary | Exp | Tool | SHA-256 (first 16) |
|--------|-------------|-----|------|---------------------|
| `benchmark_rust_vs_python.py` | `benchmark_23_domain_timing` | 043/059 | 18-domain timing | `27eea58f99a84db2` |
| `benchmark_python_baseline.py` | `benchmark_three_tier` | 069 | Python baseline JSON | `6334b6d9a2a69b0f` |

## Track 3: Drug Repurposing (Fajgenbaum / Every Cure)

| Script | Rust Binary | Exp | Tool | SHA-256 (first 16) |
|--------|-------------|-----|------|---------------------|
| `nmf_drug_disease_pipeline.py` | `validate_nmf_drug_repurposing` | 159 | NMF (Lee & Seung) | `292b8186d345b666` |

## Utility (not baselines)

| Script | Purpose | SHA-256 (first 16) |
|--------|---------|---------------------|
| `run_exp002.py` | SRA data download (Exp002) | `b99c0eb11d67bc7c` |
| `search_ncbi_datasets.py` | NCBI dataset search | `703553dd76b87243` |

---

## Verification

To verify no baseline drift, re-hash all scripts:

```bash
sha256sum scripts/*.py | sort > /tmp/current_hashes.txt
# Compare against this manifest
```

If a script hash changes, the corresponding Rust validation binary must be
re-verified to ensure the hardcoded expected values still match.
