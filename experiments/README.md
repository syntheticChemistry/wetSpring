# wetSpring Experiments

Experiment protocols and results for BarraCUDA/ToadStool validation against
published tools and open data. Each experiment establishes a baseline using
existing tools (Galaxy, QIIME2, asari, FindPFAS, scipy), then validates the
Rust CPU and Rust GPU implementations against that baseline.

**Updated**: 2026-02-19

---

## Experiment Index

| Exp | Name | Track | Status | Baseline Tool | Rust Modules Validated | Checks |
|-----|------|-------|--------|---------------|----------------------|--------|
| 001 | [Galaxy Bootstrap](001_galaxy_bootstrap.md) | 1 | DONE | Galaxy 24.1 / QIIME2 / DADA2 | io::fastq, bio::quality, bio::merge_pairs, bio::derep, bio::diversity | 28 |
| 002 | Phytoplankton 16S | 1 | DONE | QIIME2 / DADA2 | (extends Exp001 with PRJNA1195978) | — |
| 003 | Phage Assembly | 1 | DONE | SPAdes / Pharokka (conda) | (assembly/annotation, no Rust module yet) | — |
| 004 | Rust FASTQ + Diversity | 1 | DONE | Exp001 baseline | io::fastq, bio::diversity, bio::kmer, bio::derep | 18 |
| 005 | [asari LC-MS](005_asari_bootstrap.md) | 2 | DONE | asari 1.13.1 | io::mzml | 7 |
| 006 | [PFΔScreen Validation](006_pfascreen_validation.md) | 2 | DONE | FindPFAS / pyOpenMS | bio::tolerance_search, bio::spectral_match, bio::kmd | 10 |
| 007 | Rust mzML + PFAS | 2 | DONE | Exp005/006 baselines | io::mzml, io::ms2, bio::kmd, bio::spectral_match | — |
| 008 | PFAS ML Water Monitoring | 2 | NOT STARTED | Michigan DEQ data | — | — |
| 009 | [Feature Pipeline](009_feature_pipeline_validation.md) | 2 | DONE | asari 1.13.1 (MT02) | bio::eic, bio::signal, bio::feature_table | 9 |
| 010 | [Peak Detection](010_peak_detection_validation.md) | cross | DONE | scipy.signal.find_peaks | bio::signal | 17 |
| 011 | 16S Pipeline End-to-End | 1 | DONE | DADA2/UCHIME/RDP/UniFrac | bio::dada2, bio::chimera, bio::taxonomy, bio::unifrac, bio::derep, bio::diversity | 37 |
| 012 | [Algae Pond 16S](012_algae_pond_16s_validation.md) | 1 | DONE | PRJNA488170 (real NCBI data) | io::fastq, bio::quality, bio::derep, bio::dada2, bio::chimera, bio::taxonomy, bio::unifrac, bio::diversity | 29 |
| 013 | [VOC Peak Validation](013_voc_peak_validation.md) | 1/cross | DONE | Reese 2019 Table 1 (PMC6761164) | bio::signal, bio::tolerance_search | 22 |
| 014 | [Public Data Benchmarks](014_public_data_benchmarks.md) | 1 | DONE | 22 samples, 4 BioProjects vs paper ground truth | io::fastq, bio::quality, bio::derep, bio::dada2, bio::diversity | 202 |
| 015 | [Pipeline Benchmark](015_pipeline_benchmark.md) | 1 | DONE | Rust CPU vs Galaxy/QIIME2 DADA2-R | — | Benchmark |
| 016 | [GPU Pipeline Parity](016_gpu_pipeline_parity.md) | 1 | DONE | CPU vs GPU math parity (10 samples, 4 BioProjects) | bio::chimera_gpu, bio::quality_gpu, bio::taxonomy_gpu, bio::diversity_gpu | 68 |

---

## Results Directory Structure

```
experiments/
├── README.md                           ← this file
├── 001_galaxy_bootstrap.md             ← Galaxy/QIIME2 setup + 16S baseline
├── 005_asari_bootstrap.md              ← asari LC-MS setup + MT02 baseline
├── 006_pfascreen_validation.md         ← FindPFAS setup + PFAS baseline
├── 009_feature_pipeline_validation.md  ← Rust feature pipeline vs asari
├── 010_peak_detection_validation.md    ← Rust peak detection vs scipy
│
└── results/
    ├── 001_galaxy_bootstrap/
    │   ├── validation_report.json      ← DADA2/taxonomy summary stats
    │   ├── dada2-stats.tsv             ← per-sample DADA2 stats
    │   └── manifest.tsv                ← sample manifest
    ├── 002_phytoplankton/
    │   ├── exp002_report.json          ← 2,273 ASVs, 10 samples
    │   └── diversity_report.json       ← diversity metrics
    ├── 003_phage/
    │   └── exp003_report.json          ← assembly + annotation stats
    ├── 005_asari/
    │   ├── exp005_report.json          ← 5,951 features, 8 files
    │   ├── preferred_Feature_table.tsv ← asari feature table (5,951 rows)
    │   └── project.json                ← asari project config
    ├── 006_pfascreen/
    │   └── exp006_report.json          ← 25 PFAS precursors
    ├── 010_peak_baselines/
    │   ├── scipy_baselines.json        ← scipy provenance summary
    │   ├── single_gaussian.dat         ← test vector (200 points, 1 peak)
    │   ├── three_chromatographic.dat   ← test vector (500 points, 3 peaks)
    │   ├── noisy_with_spikes.dat       ← test vector (1000 points, 3 peaks)
    │   ├── overlapping_peaks.dat       ← test vector (200 points, 1 peak)
    │   └── monotonic_no_peaks.dat      ← test vector (100 points, 0 peaks)
    ├── 013_voc_baselines/
    │   └── reese2019_table1.tsv        ← 14 VOC compounds from Reese 2019
    ├── paper_benchmarks/
    │   ├── README.md                    ← benchmark strategy
    │   ├── humphrey2023_bacteriome.tsv  ← 18 OTUs, core genera
    │   ├── humphrey2023_metrics.json    ← community profile, validation targets
    │   ├── carney2016_crash_agents.json ← crash agents, detection methods
    │   ├── reese2019_voc_biomarkers.json← 14 VOC compounds, RI values
    │   └── reichardt2020_spectroradiometric.json ← organisms, methods
    ├── ncbi_dataset_search/
    │   └── ncbi_search_results.json     ← NCBI Entrez search results
    └── track2_validation_report.json   ← combined Track 2 validation
```

---

## Validation Binaries

Each validation binary uses the `Validator` harness with provenance tables,
hardcoded expected values from the baseline experiments, and tolerance
thresholds from `src/tolerances.rs`.

| Binary | Experiment | Checks | Command |
|--------|------------|--------|---------|
| `validate_fastq` | 001 | 28 | `cargo run --bin validate_fastq` |
| `validate_diversity` | 001/004 | 27 | `cargo run --bin validate_diversity` |
| `validate_mzml` | 005 | 7 | `cargo run --bin validate_mzml` |
| `validate_pfas` | 006 | 10 | `cargo run --bin validate_pfas` |
| `validate_features` | 009 | 9 | `cargo run --bin validate_features` |
| `validate_peaks` | 010 | 17 | `cargo run --bin validate_peaks` |
| `validate_diversity_gpu` | — | 38 | `cargo run --features gpu --bin validate_diversity_gpu` |
| `validate_16s_pipeline` | 011 | 37 | `cargo run --bin validate_16s_pipeline` |
| `validate_algae_16s` | 012 | 29 | `cargo run --bin validate_algae_16s` |
| `validate_voc_peaks` | 013 | 22 | `cargo run --bin validate_voc_peaks` |
| `validate_public_benchmarks` | 014 | 202 | `cargo run --bin validate_public_benchmarks` |
| `benchmark_pipeline` | 015 | — | `cargo run --release --bin benchmark_pipeline` |
| `validate_16s_pipeline_gpu` | 016 | 68 | `cargo run --features gpu --release --bin validate_16s_pipeline_gpu` |
| `benchmark_cpu_gpu` | — | — | `cargo run --release --features gpu --bin benchmark_cpu_gpu` |

**Total validation checks**: 494 (388 CPU + 106 GPU)

---

## How to Add a New Experiment

1. Create `experiments/NNN_descriptive_name.md` with: date, status, objective,
   baseline tool, dataset, protocol, acceptance criteria.
2. Run the baseline tool and save results to `experiments/results/NNN_name/`.
3. Create `src/bin/validate_NNN.rs` with provenance table and `Validator` checks.
4. Add the `[[bin]]` entry to `Cargo.toml`.
5. Update this README with the new experiment row.
