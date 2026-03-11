# Industry Tool Coverage Matrix & Gap Analysis

**Last Updated:** March 10, 2026
**Version:** V110

---

## 1. Sovereign Replacements — Complete

These industry tools have full sovereign Rust equivalents in wetSpring,
validated against Python baselines with documented tolerances.

| Industry Tool    | Domain                           | wetSpring Sovereign Module(s)                               | Status |
| ---------------- | -------------------------------- | ----------------------------------------------------------- | ------ |
| QIIME2           | 16S amplicon pipeline            | Full pipeline (QC → DADA2 → diversity → taxonomy → PCoA)   | DONE   |
| DADA2            | Amplicon denoising               | `bio::dada2`, `bio::dada2_gpu`                              | DONE   |
| Galaxy           | Pipeline orchestrator            | Sovereign binaries, IPC server                              | DONE   |
| UCHIME           | Chimera detection                | `bio::chimera`, `bio::chimera_gpu`                          | DONE   |
| VSEARCH          | Merge/derep/clustering           | `bio::merge_pairs`, `bio::derep`                            | DONE   |
| RDP classifier   | Naive Bayes taxonomy             | `bio::taxonomy`, `bio::taxonomy_gpu`                        | DONE   |
| scipy.integrate  | ODE solvers                      | `bio::ode` + barraCuda `BatchedOdeRK4F64`                   | DONE   |
| numpy SSA        | Gillespie simulation             | `bio::gillespie` + `GillespieGpu`                           | DONE   |
| dendropy         | Tree distances                   | `bio::robinson_foulds`                                      | DONE   |
| scikit-learn     | RF, decision tree, GBM           | `bio::random_forest`, `bio::decision_tree`, `bio::gbm`      | DONE   |
| skbio            | Diversity metrics                | `bio::diversity` + GPU                                      | DONE   |
| Biopython        | Phred, sequence I/O              | `bio::phred`, `io::fastq`                                   | DONE   |
| asari            | LC-MS feature detection          | `bio::eic`, `bio::signal`, `bio::feature_table`             | DONE   |
| FindPFAS         | PFAS suspect screening           | `bio::tolerance_search`, `bio::spectral_match`, `bio::kmd`  | DONE   |
| pyOpenMS         | Mass spec toolkit                | `bio::kmd`, `bio::spectral_match`                           | DONE   |
| scipy.signal     | Peak detection + integration     | `bio::signal` (find_peaks, integrate_peak)                  | DONE   |
| MAFFT/MUSCLE     | Multiple sequence alignment      | `bio::msa` (progressive NJ-guided MSA)                      | DONE   |
| Calibration CDS  | Standard curves / quantitation   | `bio::calibration` (fit_calibration, quantify_batch)        | DONE   |
| JCAMP-DX spec    | Spectroscopy data exchange       | `io::jcamp` (streaming, SQZ, compound files)                | **V104**|
| Dorado           | Nanopore neural basecalling      | `bio::dorado` (subprocess delegation + FASTQ parse)         | **V104**|

---

## 2. New in V103–V104

| Module                                | What                                 | Lines | Tests | Version |
| ------------------------------------- | ------------------------------------ | ----- | ----- | ------- |
| `bio::signal::integrate_peak`         | Trapezoidal area under detected peak | ~50   | 4     | V103    |
| `bio::signal::find_peaks_with_area`   | Detect + integrate in one pass       | ~10   | 1     | V103    |
| `bio::calibration`                    | Standard curves + quantitation       | ~160  | 8     | V103    |
| `bio::msa`                            | Progressive MSA (NJ-guided)          | ~510  | 11    | V103    |
| `io::mzxml`                           | mzXML streaming parser               | ~400  | 11    | V103    |
| `io::biom` (feature: json)            | BIOM 1.0 JSON parser (QIIME2)       | ~240  | 9     | V103    |
| `visualization::scenarios::chromatography` | TIC/EIC/quantitation petalTongue | ~310  | 3     | V103    |
| `io::jcamp`                           | JCAMP-DX spectroscopy streaming      | ~490  | 10    | V104    |
| `bio::dorado`                         | Dorado basecaller subprocess         | ~270  | 9     | V104    |
| `signal_gpu::find_peaks_with_area_gpu` | GPU peak detect + area integration | ~30   | 0*    | V104    |
| **Total**                             |                                      |       | **66**|         |

*GPU integration tests require hardware; API compiles verified via type check.

---

## 3. I/O Parser Coverage

| Format     | Module        | Streaming | Compression | Status  |
| ---------- | ------------- | --------- | ----------- | ------- |
| FASTQ      | `io::fastq`   | Yes       | gzip        | DONE    |
| mzML       | `io::mzml`    | Yes       | zlib+base64 | DONE    |
| mzXML      | `io::mzxml`   | Yes       | zlib+base64 | **NEW** |
| MS2        | `io::ms2`     | Yes       | —           | DONE    |
| POD5/FAST5 | `io::nanopore`| Yes       | —           | DONE    |
| XML        | `io::xml`     | Yes       | —           | DONE    |
| BIOM 1.0   | `io::biom`    | No (JSON) | —           | **NEW** |
| JCAMP-DX   | `io::jcamp`   | Yes       | —           | **V104**|

---

## 4. Tools Used As Baselines Only (No Replacement Needed)

| Tool                  | Why                                                                  |
| --------------------- | -------------------------------------------------------------------- |
| mothur                | Legacy 16S; QIIME2/DADA2 supersedes; sovereign equivalent exists     |
| Kraken2               | K-mer taxonomy; our `bio::taxonomy` uses RDP-style (paper-validated) |
| FastQC                | QC reporting; `bio::quality` does the actual filtering               |
| Trimmomatic/Cutadapt  | Adapter trimming; `bio::quality`, `bio::adapter` handle this         |
| MG-RAST               | Web-based metagenomics; data source only                             |
| PhyNetPy/PhyloNet-HMM | Specialized phylogenetics; `bio::hmm` covers HMM core               |
| SATé                  | Divide-and-conquer alignment; `bio::msa` + `bio::felsenstein` cover  |

---

## 5. Proprietary / Closed-Source — Strategy

| Tool/Format           | Barrier                          | Strategy                                         |
| --------------------- | -------------------------------- | ------------------------------------------------ |
| Thermo .raw           | Proprietary binary               | mzML via ProteoWizard msconvert                  |
| Agilent .d            | Proprietary directory            | mzML via ProteoWizard msconvert                  |
| SCIEX .wiff           | Proprietary binary               | mzML via ProteoWizard msconvert                  |
| Shimadzu .lcd         | Proprietary binary               | No open converter — out of scope                 |
| Chromeleon CDS        | Commercial (Thermo Fisher)       | Sovereign: mzML + calibration + petalTongue      |
| MassHunter            | Commercial (Agilent)             | Sovereign: MS2 parser + spectral_match           |
| Xcalibur              | Commercial (Thermo)              | Sovereign: mzML parser                           |
| Waters Empower        | Commercial (Waters)              | Sovereign: mzML parser                           |
| SnapGene              | Synthetic biology tool           | Out of scope for wetSpring (different domain)     |

**Approach:** The bioinformatics community solved vendor lock-in with
**mzML** (open standard) + **ProteoWizard msconvert** (open converter).
Our mzML + mzXML parsers are the correct sovereign approach.

---

## 6. Out of Scope for wetSpring

| Tool                          | Domain                      | Belongs              |
| ----------------------------- | --------------------------- | -------------------- |
| SnapGene / cloning simulation | Synthetic biology           | New Spring           |
| SPAdes / genome assembly      | Assembly                    | New Spring           |
| Prokka / annotation           | Genome annotation           | New Spring           |
| BLAST+                        | Sequence database search    | barraCuda primitive  |
| BWA/bowtie/minimap2           | Read mapping (WGS)          | Not in paper set     |
| GATK/samtools/bcftools        | Variant calling             | Not in paper set     |
| FDA 21 CFR Part 11            | Regulatory framework        | Documentation/process|

---

## 7. Remaining Gaps (Prioritized)

### P3 — Build if demand emerges

| Gap                   | Notes                                                    |
| --------------------- | -------------------------------------------------------- |
| JCAMP-DX parser       | Spectroscopy text format; low priority unless IR/Raman   |
| OpenChrom interop     | Read/write OpenChrom projects; only for instrument work  |
| Dorado basecaller     | Subprocess delegation for nanopore; planned in specs     |

### Completed (formerly P1/P2)

| Item                               | Module                              | Status    |
| ---------------------------------- | ----------------------------------- | --------- |
| ~~Peak integration~~               | `bio::signal::integrate_peak`       | **DONE**  |
| ~~Calibration / quantitation~~     | `bio::calibration`                  | **DONE**  |
| ~~Multiple sequence alignment~~    | `bio::msa`                          | **DONE**  |
| ~~mzXML parser~~                   | `io::mzxml`                         | **DONE**  |
| ~~BIOM format parser~~             | `io::biom`                          | **DONE**  |
| ~~Chromatogram visualization~~     | `scenarios::chromatography`         | **DONE**  |

---

## References

- ICH Q2(R1): Validation of Analytical Procedures
- EPA Methods 533 / 537.1: PFAS quantitation
- Pedrioli et al. 2004, *Nat Biotechnol* 22:1459-1466 (mzXML)
- McDonald et al. 2012, *GigaScience* 1:7 (BIOM format)
- Katoh et al. 2002, *NAR* 30:3059-3066 (MAFFT)
- Edgar 2004, *NAR* 32:1792-1797 (MUSCLE)
- Li et al. 2023, *Nat Commun* 14:4113 (asari)
