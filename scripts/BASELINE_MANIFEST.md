# Python Baseline Manifest

**Generated:** February 27, 2026 (Phase 61 audit — SPDX headers + provenance metadata added)
**Verified:** March 24, 2026 (V137 — SHA-256 hashes reconciled with `provenance_registry.rs`; `exp008_pfas_ml_baseline.py` and `python_anaerobic_biogas_baseline.py` hashes corrected; download script hashes added; numpy version matrix documented.)
**Last scripts commit:** V120 (added `scripts/tolerances.py` — shared tolerance constants mirroring Rust)

## Commit History Clarification

Script `# Commit:` headers reference the commit at which each script's
**mathematical output** was last validated, not the manifest freeze commit:

- `756df26` — original baseline generation (pre-SPDX, Tracks 1-3)
- `wetSpring Phase 66` — Track 4 (soil QS) baselines added in Phase 66
- `48fb787` — manifest freeze: SPDX headers added, SHA-256 hashes updated

The SHA-256 values in this manifest reflect post-SPDX content (commit `48fb787`).
Numerical output is unchanged from the original generation commits.

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

- **Hardware:** Eastgate i9-12900K, 64 GB DDR5, NVIDIA RTX 4070 (Ada)
- **OS:** Pop!_OS 22.04 (Ubuntu 22.04 base, x86_64, kernel 6.17.9)
- **Python:** 3.11+ (CPython) — note: `requirements.txt` pins `>=1.24,<2` for numpy;
  some frozen JSON baselines record `numpy 2.2.6` from a later environment.
  When reproducing, install from `requirements.txt` for Track 1–4 baselines;
  Track 6 (biogas) JSONs may require `numpy>=2.0` to match exact digits.
- **NumPy:** 1.26+ for original baselines; 2.2.6 for Track 6 biogas baselines
- **SciPy:** 1.12+ (where used — ODE, peaks, integration)
- **Galaxy:** quay.io/galaxy/galaxy:latest (Exp001 only, with qiime2-2026 conda env)
- **Baseline freeze:** commit `48fb787` (2026-02-23)

### Standard reproduction command (most scripts)

```bash
cd /path/to/wetSpring
python3 scripts/<script>.py
```

### Per-script command exceptions

| Script | Exact command | Notes |
|--------|--------------|-------|
| `validate_exp001.py` | `docker run ... && conda activate qiime2-2026 && python3 scripts/validate_exp001.py` | Galaxy container + QIIME2 conda env |
| `validate_track2.py` | `python3 -m venv .venv && source .venv/bin/activate && pip install asari findpfas && python3 scripts/validate_track2.py` | Requires asari 1.13.1 + FindPFAS venv |
| `run_exp002.py` | `python3 scripts/run_exp002.py --accession PRJNA488170` | SRA download utility, not a baseline |
| All others | `python3 scripts/<script>.py` | No special environment |

For Exp001 (Galaxy/QIIME2), see `validate_exp001.py` for full container instructions.
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
| `exp008_pfas_ml_baseline.py` | `validate_epa_pfas_ml` | 008/041 | sklearn RF+GBM | `c957d9ec59cf2388` |
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

### Track 6 — Anaerobic Biogas Kinetics (Liao / ADREC)

| Script | Rust Binary | Exp | Tool | SHA-256 (first 16) |
|--------|-------------|-----|------|---------------------|
| `python_anaerobic_biogas_baseline.py` | `validate_barracuda_cpu_v27`, `validate_anaerobic_*` | 336-346 | scipy/numpy | `5429ecaf2827b1d9` |

### Track 4 — No-Till Soil QS & Anderson Geometry

| Script | Rust Binary | Exp | Tool | SHA-256 (first 16) |
|--------|-------------|-----|------|---------------------|
| `martinez2023_pore_geometry.py` | `validate_soil_qs_pore_geometry` | 170 | Anderson + QS ODE | `b75181ca308568de` |
| `feng2024_pore_diversity.py` | `validate_soil_pore_diversity` | 171 | Shannon/BC/Anderson | `5dd58505694e385b` |
| `mukherjee2024_colonization.py` | `validate_soil_distance_colonization` | 172 | AI diffusion/Anderson | `2056f32a550fc155` |
| `islam2014_brandt_farm.py` | `validate_notill_brandt_farm` | 173 | Aggregate/SOM | `72a4703cfbb70293` |
| `zuber2016_meta_analysis.py` | `validate_notill_meta_analysis` | 174 | Meta-analysis CI | `b3112e70f47ae972` |
| `liang2015_longterm_tillage.py` | `validate_notill_longterm_tillage` | 175 | 2×2×2 factorial | `dd447d48edb8593f` |
| `tecon2017_biofilm_aggregate.py` | `validate_soil_biofilm_aggregate` | 176 | Biofilm/aggregate | `c07ccf97f27777d7` |
| `rabot2018_structure_function.py` | `validate_soil_structure_function` | 177 | Structure→function | `dcf5a170ccd6fbc5` |
| `wang2025_tillage_microbiome.py` | `validate_tillage_microbiome_2025` | 178 | Tillage × compartment | `b947317f2aaa8378` |

### NPU Spectral Triage

| Script | Rust Binary | Exp | Tool | SHA-256 (first 16) |
|--------|-------------|-----|------|---------------------|
| `spectral_match_baseline.py` | `validate_npu_spectral_triage` | 124 | Int8 triage/cosine | `e3f6f4ff4071c4fe` |

## Utility (not baselines)

| Script | Purpose | SHA-256 (first 16) |
|--------|---------|---------------------|
| `run_exp002.py` | SRA data download (Exp002) | `7fbcd4bf2f3ec2b6` |
| `search_ncbi_datasets.py` | NCBI dataset search | `00f1e1133cdd14d4` |
| `fetch_ncbi_phase35.py` | Phase 35 NCBI bulk fetch | `0c55e7fd85ede4e6` |
| `download_priority1.py` | PFAS/Vibrio/Campy/SILVA downloads (Track 1/2 data) | `5d9adf95c2bf44df` |
| `download_priority2.py` | NCBI bulk download utility | `c143a332ea7dad9a` |

---

## R Industry Baselines (vegan / dada2 / phyloseq)

**Added:** 2026-03-10 (V106 — R industry parity validation)

These R baselines validate sovereign Rust implementations against the
gold-standard R packages used by the microbial ecology community (QIIME2,
mothur, phyloseq workflows). R baselines are supplementary to the Python
baselines above.

### R Environment

- **R:** 4.1.2 ("Bird Hippie") on x86_64-pc-linux-gnu
- **vegan:** 2.7.3 (Oksanen et al. — CRAN alpha/beta diversity standard)
- **dada2:** 1.22.0 (Callahan et al. 2016 — Bioconductor ASV denoiser)
- **phyloseq:** 1.38.0 (McMurdie & Holmes 2013 — Bioconductor UniFrac/ordination)
- **ape:** 5.8.1 (Paradis et al. — phylogenetic tree operations)
- **jsonlite:** (JSON output for Rust validator consumption)

### R Baseline Scripts

| Script | Rust Binary | Domain | SHA-256 (first 16) |
|--------|-------------|--------|---------------------|
| `r_vegan_diversity_baseline.R` | `validate_r_industry_parity` | Shannon, Simpson, BC, Chao1, rarefaction, Pielou | `49fd0469caaf16aa` |
| `r_dada2_error_baseline.R` | `validate_r_industry_parity` | Error model, Phred, OMEGA_A, consensus Q | `edcfbe026b102d14` |
| `r_phyloseq_unifrac_baseline.R` | `validate_r_industry_parity` | Weighted/unweighted UniFrac, PCoA, cophenetic | `caec562710d8cc3d` |

### R Baseline JSON Outputs

| JSON Output | Source Script | SHA-256 (first 16) |
|-------------|--------------|---------------------|
| `experiments/results/r_baselines/vegan_diversity.json` | `r_vegan_diversity_baseline.R` | `a9387cec33513368` |
| `experiments/results/r_baselines/dada2_error_model.json` | `r_dada2_error_baseline.R` | `1424bf67c6fcdf51` |
| `experiments/results/r_baselines/phyloseq_unifrac.json` | `r_phyloseq_unifrac_baseline.R` | `088354c831db08f8` |

### Weighted UniFrac Normalization Note

Our Rust `weighted_unifrac` uses **max-normalization**: `Σ b_i|pA-pB| / Σ b_i·max(pA,pB)`.
R/phyloseq uses **sum-normalization**: `Σ b_i|pA-pB| / Σ tipAge·(pA+pB)`.
Both are valid weighted UniFrac variants (Lozupone et al. 2007). The validator
checks structural properties (symmetry, self-distance, bounds, ordering) rather
than exact values for weighted UniFrac, as the normalization difference is
well-understood and documented.

Additionally, phyloseq's `fastUniFrac` has a known trifurcation bug: its
`node.desc` matrix assumes `ncol=2` (binary tree), silently dropping the 3rd
child via R's matrix recycling. The R baseline uses a strictly bifurcating tree
to avoid this.

### Reproduction

```bash
Rscript scripts/r_vegan_diversity_baseline.R
Rscript scripts/r_dada2_error_baseline.R
Rscript scripts/r_phyloseq_unifrac_baseline.R
cargo run --release --bin validate_r_industry_parity
```

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
