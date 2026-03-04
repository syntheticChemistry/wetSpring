# Biological Data Type Catalog — NestGate Evolution Primer

**Last Updated:** February 26, 2026
**Purpose:** Profile every biological data type that wetSpring (and the broader
ecoPrimals ecosystem) touches. This catalog drives NestGate's evolution from
generic content-addressed storage to a biology-aware data primal that can
tag, store, distribute, transform, and reason about scientific data.

**Analogy:** The Springs accelerated ToadStool (compute primal) by exercising
primitives across 200+ experiments. This catalog accelerates NestGate (data
primal) by profiling the data that flows through those experiments.

---

## How to Read This Document

Each section profiles a data domain. For each type:

- **What it is** — biological meaning and context
- **Format(s)** — on-disk representation
- **Where used** — wetSpring modules and experiments
- **Rust representation** — current struct (if any)
- **NestGate needs** — what the data primal must support for this type
- **Volume** — typical sizes for planning storage and bandwidth

---

## 1. Sequencing Reads

### 1.1 FASTQ — Short Reads (Illumina)

**What:** Raw sequencing output. Each record: ID, DNA sequence, quality scores
(Phred+33). Paired-end reads come in R1/R2 files. This is the entry point
for 16S amplicon pipelines, metagenomics, and population genomics.

**Format:** Text (4 lines per record) or gzip-compressed (`.fastq.gz`).

**Where used:**
- `io::fastq` — streaming parser (`FastqRecord`, `FastqRefRecord`, `FastqStats`)
- Exp001, 004, 011, 012, 014, 017, 184, 185 — 16S amplicon pipelines
- `scripts/download_data.sh` — SRA download via `fasterq-dump`

**Rust:** `FastqRecord { header, sequence, quality }`, `FastqStats { total_reads, total_bases, mean_quality, gc_content }`

**Volume:** 1-50 GB per MiSeq run (compressed). 16S amplicon: ~1-5 GB.

**NestGate needs:**
- Content-address individual read files (BLAKE3)
- Tag with: sequencing platform (Illumina/Nanopore), read type (single/paired),
  amplicon target (16S V4, 16S full-length, shotgun), sample metadata
- Link paired files (R1 ↔ R2)
- Link to BioProject/SRA accession
- Stream reads to BarraCUDA pipeline without full materialization
- Deduplication across runs that share control samples

### 1.2 FASTQ — Long Reads (Nanopore, PacBio)

**What:** Same format as short reads but 1 kb to >1 Mb per read. Higher error
rate (~Q20 for Nanopore R10.4.1), different quality profile. Critical for
resolving AMR gene cassettes, structural variants, and full-length 16S.

**Format:** Same as Illumina FASTQ but single-end (no R1/R2 pairing).

**Where used:** Planned for Sub-thesis 06 (field genomics). Not yet in codebase.

**NestGate needs:**
- Same as short reads plus: tag with read length distribution, N50
- Link to flow cell ID, run ID, basecalling model version
- Support streaming from MinION in real time (not just batch files)

### 1.3 FAST5 / POD5 — Raw Nanopore Signal

**What:** Raw ionic current signal from nanopore sequencer. FAST5 is HDF5-based
(legacy). POD5 is the newer Apache Arrow-based format. Contains the electrical
signal that gets basecalled into FASTQ.

**Format:** FAST5 (HDF5, binary), POD5 (Arrow IPC, binary).

**Where used:** Planned for Sub-thesis 06 (`io::nanopore` module, Exp196).

**Volume:** 5-50 GB per flow cell. Much larger than basecalled FASTQ.

**NestGate needs:**
- Store raw signal for re-basecalling with improved models
- Tag with: device ID (MinION serial), flow cell ID, pore count, run metadata
- Link raw → basecalled (FAST5/POD5 → FASTQ provenance chain)
- Optionally stream directly to BarraCUDA `bio::basecall` for on-the-fly processing
- PUF fingerprint association (Exp195 S1) — device attestation via NestGate

### 1.4 FASTA — Reference Sequences

**What:** Sequences without quality scores. Used for reference databases (SILVA,
NCBI nt/nr), assembled genomes, and gene sequences.

**Format:** Text (header line starting with `>`, sequence lines).

**Where used:**
- `validation_helpers` — SILVA 16S reference
- `ncbi::efetch` — NCBI nucleotide/protein fetch
- `ncbi::nestgate` — NestGate cache for NCBI FASTA (already implemented)
- Exp121, 125, 184, 185 — genome assemblies, metagenomes

**Rust:** `ReferenceSeq { id, taxonomy, sequence }` (SILVA), raw `String` (NCBI)

**Volume:** SILVA 138.2: ~3 GB. NCBI nt: ~100 GB. Individual genomes: 1-10 MB.

**NestGate needs:**
- Already partially implemented (FASTA via NCBI EFetch + JSON-RPC cache)
- Add: version tracking for reference databases (SILVA 138.1 vs 138.2)
- Tag with: molecule type (DNA/RNA/protein), organism, taxonomy, gene name
- Support incremental reference DB updates (new SILVA release = delta, not full re-download)

---

## 2. Taxonomy and Classification

### 2.1 Lineage Strings

**What:** Hierarchical taxonomic classification. Semicolon-separated ranks:
`Bacteria;Proteobacteria;Gammaproteobacteria;Vibrionales;Vibrionaceae;Vibrio`

**Format:** Text string. Ranks: Domain, Phylum, Class, Order, Family, Genus, Species.

**Where used:**
- `bio::taxonomy` — RDP-style naive Bayesian classifier
- `taxonomy::Lineage` — `Lineage::from_taxonomy_string()`
- SILVA taxonomy TSV, RDP trainset
- All 16S experiments

**Rust:** `Lineage` struct with `from_taxonomy_string()` parser

**NestGate needs:**
- Canonical lineage storage with rank-level indexing
- Cross-reference between naming authorities (SILVA, RDP, NCBI, GTDB)
- Tag sequences with resolved lineage at store time
- Query by any rank: "give me all Vibrio sequences"

### 2.2 SILVA Reference Taxonomy

**What:** Curated 16S/18S/23S/28S rRNA database. Gold standard for amplicon-based
taxonomy. Updated ~annually.

**Format:** FASTA (sequences) + TSV (taxonomy map: accession → lineage).

**Where used:**
- `validation_helpers` — `SILVA_FASTA`, `SILVA_TAX_TSV`, `stream_taxonomy_tsv()`
- All 16S pipeline experiments
- `scripts/download_data.sh` — downloads SILVA 138.2

**NestGate needs:**
- Version-tracked reference database (SILVA release as a named, content-addressed dataset)
- Differential updates between versions
- Serve to BarraCUDA's taxonomy classifier via JSON-RPC without full local materialization

### 2.3 NCBI Taxonomy

**What:** NCBI's taxonomic database. Tax IDs, names, lineages, synonyms. The
universal ID system for biological organisms.

**Format:** JSON (via Entrez E-utilities), flat files (names.dmp, nodes.dmp).

**Where used:**
- `ncbi::esearch`, `ncbi::esummary`, `ncbi::efetch` — all NCBI queries
- `bio::ncbi_data` — `VibrioAssembly`, `CampyAssembly`, `BiomeProject`
- Exp121, 125, 126, 184

**Rust:** `BiomeProject { biome, samples, qs_gene_count, avg_diversity }` and similar

**NestGate needs:**
- Local mirror of NCBI taxonomy (names.dmp + nodes.dmp, ~500 MB)
- Resolve tax IDs to lineages without network round-trip
- Link all stored sequences to their NCBI tax ID
- Cross-reference with SILVA, GTDB lineage strings

### 2.4 OTU/ASV Tables

**What:** Community composition matrices. Rows = taxa (OTUs or ASVs), columns =
samples, values = read counts. The fundamental data structure for microbial
ecology.

**Format:** TSV/CSV (QIIME2 style), BIOM (HDF5-based, not yet implemented).

**Where used:**
- `bio::diversity` — alpha diversity (Shannon, Pielou, rarefaction)
- `bio::bray_curtis` — beta diversity distance matrices
- `bio::unifrac` — phylogenetic beta diversity
- All 16S experiments, Anderson QS framework

**Rust:** `Vec<f64>` abundance vectors, `Vec<Vec<f64>>` community matrices

**NestGate needs:**
- Store OTU/ASV tables with sample metadata links
- Tag with: pipeline version, reference DB version, rarefaction depth
- Support BIOM format (HDF5) for interop with QIIME2 ecosystem
- Provenance: FASTQ → pipeline → OTU table (full lineage)

---

## 3. Assembled Sequences and Genomes

### 3.1 Genome Assemblies

**What:** Assembled nucleotide sequences from sequencing reads. Complete genomes,
draft genomes, scaffold-level assemblies. Represented in FASTA format.

**Where used:**
- `bio::ncbi_data` — `VibrioAssembly` (200 genomes), `CampyAssembly` (158 genomes)
- `bio::pangenome` — core/accessory/unique genes, Heap's law
- `bio::ani` — Average Nucleotide Identity (pairwise genome comparison)
- `bio::snp` — SNP calling from alignments
- Exp055, 056, 110, 121, 125

**Rust:** `VibrioAssembly { accession, organism, genome_size, scaffold_count, gc_content, habitat, completeness, qs_genes }`, `CampyAssembly { ... }`

**NestGate needs:**
- Store assemblies with assembly-level metadata (completeness, N50, contigs)
- Link to source reads (SRA accession → FASTQ → assembly)
- Tag with: organism, assembly level (complete/scaffold/contig), source habitat
- Pangenome annotations (core/accessory/unique gene assignments per genome)

### 3.2 Metagenome-Assembled Genomes (MAGs)

**What:** Genomes reconstructed from metagenomic sequencing of environmental
samples. Lower quality than isolate genomes. Critical for studying
unculturable organisms.

**Where used:**
- Cold seep metagenomes (Exp185), deep-sea vent communities
- `bio::pangenome` — cross-ecosystem pangenome analysis
- Anderson community structure analysis

**NestGate needs:**
- Tag with: source metagenome, binning method, completeness, contamination
- Quality tier: high (>90% complete, <5% contam), medium, low
- Link to source community and environmental metadata

### 3.3 Gene Sequences

**What:** Individual genes extracted from genomes or metagenomes. Protein-coding
(CDS), rRNA, tRNA. Used for functional analysis, QS gene catalogs, dN/dS.

**Where used:**
- `bio::dnds` — Nei-Gojobori pairwise dN/dS (codon-level analysis)
- `bio::pangenome` — gene clustering, enrichment, functional assignment
- QS gene catalogs (299K genes across 170 metagenomes, Exp144-145)
- PFAS-related genes (efflux pumps, degradation pathways)

**Rust:** Gene sequences as strings, codon-level operations in `bio::dnds`

**NestGate needs:**
- Tag with: gene name, functional annotation (COG, KEGG, Pfam), organism, source genome
- QS gene type classification (synthase, receptor, eavesdropper)
- Link gene → genome → community → environment

---

## 4. Phylogenetic Data

### 4.1 Phylogenetic Trees (Newick)

**What:** Tree topology + branch lengths representing evolutionary relationships.
Used for UniFrac, phylogenetic placement, Robinson-Foulds distance.

**Format:** Newick text: `((A:0.1,B:0.2):0.3,C:0.4);`

**Where used:**
- `bio::unifrac::tree` — `PhyloTree`, `TreeNode`
- `bio::phylo_placement` — read placement on reference tree
- `bio::robinson_foulds` — tree distance
- Exp019, 021, 029, 031, 032, 036, 037, 038

**Rust:** `PhyloTree { nodes: Vec<TreeNode> }`, `TreeNode { label, parent, children, branch_length }`

**NestGate needs:**
- Store trees with method metadata (NJ, ML, Bayesian)
- Link trees to their underlying alignment and sequences
- Version track trees as new sequences are added (incremental phylogenetics)

### 4.2 Distance Matrices

**What:** Pairwise distances between samples (Bray-Curtis, UniFrac, ANI) or
between taxa (Robinson-Foulds). Square symmetric or condensed triangular.

**Where used:**
- `bio::bray_curtis` — community dissimilarity
- `bio::unifrac` — phylogenetic community distance
- `bio::ani` — genome-level distance
- `bio::robinson_foulds` — tree topology distance
- PCoA ordination

**Rust:** `Vec<f64>` condensed triangular, full square matrix for ANI

**NestGate needs:**
- Store precomputed distance matrices for large datasets
- Tag with: distance metric, input data version, pipeline version
- Serve to downstream analysis (PCoA, clustering) without recomputation

---

## 5. Mass Spectrometry and Chemistry

### 5.1 mzML — Vendor-Neutral MS Data

**What:** XML format for mass spectrometry data. Contains spectra (m/z arrays +
intensity arrays), chromatograms, and metadata. Base64-encoded binary arrays,
optionally zlib-compressed.

**Format:** XML with base64 binary arrays.

**Where used:**
- `io::mzml` — streaming parser (`MzmlSpectrum`, `MzmlStats`, `MzmlIter`)
- `bio::eic` — extracted ion chromatograms
- `bio::feature_table` — LC-MS feature detection
- Exp005, 006, 007, 009, 010

**Rust:** `MzmlSpectrum { scan_number, ms_level, rt_minutes, mz_array, intensity_array, total_ion_current }`, `MzmlStats { num_spectra, ms1_count, ms2_count }`

**Volume:** 100 MB to 10 GB per LC-MS run.

**NestGate needs:**
- Store raw mzML with instrument metadata (instrument type, ionization, scan range)
- Tag with: sample ID, analytical method, date, operator
- Link to processed features (mzML → features → identifications)

### 5.2 MS2 — Tandem Mass Spectra

**What:** Text format for MS/MS spectra. Simpler than mzML, commonly exported
from vendor software.

**Format:** Text (header block + m/z intensity pairs per spectrum).

**Where used:**
- `io::ms2` — parser (`Ms2Spectrum`, `Ms2Stats`, `Ms2Iter`)
- Exp006 — PFAS fragment matching

**Rust:** `Ms2Spectrum { scan, precursor_mz, charge, peaks }`, `Ms2Stats { total_spectra, ms2_count }`

**NestGate needs:**
- Same as mzML but lighter weight
- Link MS2 spectra to their precursor ions in MS1

### 5.3 PFAS Chemical Data

**What:** Per- and polyfluoroalkyl substance reference data. Exact masses,
molecular formulas, fragment ions, Kendrick mass defects.

**Where used:**
- `bio::pfas` — tolerance search, KMD filtering
- `bio::pfas_ml` — ML classification (fate-and-transport prediction)
- Exp006, 007, 008, 018, 041, 042
- Jones Lab PFAS library (175 compounds, Zenodo 14341321)

**Rust:** `PfasRef { name, formula, exact_mass }`, `PfasFragments`

**NestGate needs:**
- Curated chemical reference library with versioning
- Tag with: compound class (PFCA, PFSA, precursor), regulatory status
- Link to environmental detections (sample → compound → concentration)
- Cross-reference EPA CompTox dashboard identifiers

### 5.4 Spectral Libraries (MassBank)

**What:** Reference tandem mass spectra for compound identification. Community-
curated, open access. Used for spectral matching (cosine similarity).

**Where used:**
- `bio::spectral_match` — cosine similarity scoring
- Exp042, 111, 124 — spectral library matching
- NPU spectral triage (Exp117, 124)

**NestGate needs:**
- Version-tracked spectral library (MassBank releases, NIST, mzCloud)
- Serve spectra to BarraCUDA's spectral matching module via JSON-RPC

---

## 6. Model and Mathematical Data

### 6.1 ODE Parameters and Trajectories

**What:** Parameters and solutions for ordinary differential equation models.
QS signaling (Waters 2008), phage defense (Hsueh 2022), bistable switching
(Fernandez 2020), cooperation (Bruger 2018), soil QS dynamics.

**Where used:**
- `bio::ode` — RK4 solver with generic state types
- 6 ODE models: QS, Gillespie, phage, bistable, multi-signal, capacitor, cooperation
- GPU: `BatchedOdeRK4<S>::generate_shader()` for WGSL

**Rust:** `OdeResult { times, states }` with model-specific parameter structs

**NestGate needs:**
- Store parameter sets and computed trajectories
- Tag with: model name, parameter source (published paper), solver settings
- Enable parameter sweep result caching (Exp049, 108 — 1024-genome landscapes)

### 6.2 Anderson Hamiltonian and Spectral Data

**What:** Disorder Hamiltonians for Anderson localization analysis. Eigenvalue
spectra, level spacing ratios, localization lengths. The core mathematical
framework underlying Sub-theses 01, 03, 04, 06.

**Where used:**
- `barracuda::spectral` — Anderson Hamiltonian construction
- `bio::anderson_qs` — disorder-to-QS mapping
- GPU: Lanczos eigenvalue extraction
- Exp107-156, 170-182, 186-187

**Rust:** Matrix construction via `anderson_hamiltonian()`, `anderson_3d()`, eigenvalue vectors

**NestGate needs:**
- Cache precomputed spectra for large lattices (L=12+ takes hours)
- Tag with: dimension, lattice size, disorder strength, boundary conditions
- Enable incremental computation (add more disorder realizations to existing dataset)

### 6.3 ML Models (Decision Tree, RF, GBM, ESN)

**What:** Trained model parameters. Decision tree splits, random forest ensembles,
gradient boosted machine parameters, ESN reservoir weights.

**Where used:**
- `bio::decision_tree` — JSON-exported sklearn tree
- `ml::random_forest` — ensemble inference
- `ml::gbm` — gradient boosting inference
- `bio::esn` — echo state network (weights: w_in, w_res, w_out)
- Drug repurposing: NMF, KG embeddings (Exp157-165)

**Rust:** `DecisionTree { nodes }`, `Esn { w_in, w_res, w_out, config }`

**NestGate needs:**
- Version-tracked model storage (model v1 → v2 after retraining)
- Tag with: training data provenance, accuracy metrics, target domain
- Serve models to NPU for deployment (ESN weights → AKD1000 DMA)
- Support model evolution tracking: (1+1)-ES weight mutations over time

### 6.4 Knowledge Graph Triples

**What:** (head, relation, tail) triples for drug repurposing knowledge graphs.
TransE embeddings for link prediction.

**Where used:**
- `bio::kg_embed` — TransE training and inference
- Exp161 — ROBOKOP knowledge graph

**NestGate needs:**
- Triple store with entity/relation typing
- Embedding vectors associated with entities
- Query: "what drugs target this pathway?" via embedding similarity

---

## 7. Environmental and Sensor Data

### 7.1 Community Composition Time Series

**What:** OTU/ASV tables sampled over time. The raw data for sentinel monitoring
(Sub-thesis 04), bloom prediction, soil health tracking.

**Where used:**
- Exp039 (algal pond), Exp040 (bloom surveillance), Exp123 (temporal ESN)
- Track 4 soil QS experiments (Exp170-182)
- Anderson W(t) dynamic disorder (Exp186)

**NestGate needs:**
- Time-indexed community profiles with sample metadata
- Link each time point to environmental covariates (temperature, pH, nutrient levels)
- Support streaming ingestion from field sentinels (Sub-thesis 06)

### 7.2 Soil Sensor Data

**What:** Volumetric water content (VWC), temperature, electrical conductivity
from in-field sensors (SoilWatch 10 pattern). airSpring domain but consumed
by wetSpring for integrated sentinel analysis.

**Where used:**
- Sub-thesis 06 (field genomics), Sub-thesis 08 (NPU agricultural IoT)
- Cross-spring integration: airSpring sensor → wetSpring community → Anderson regime

**NestGate needs:**
- Time-series sensor data with calibration metadata
- Link sensor readings to concurrent community profiles (same station, same time)
- Support real-time streaming from field units

### 7.3 Environmental Metadata

**What:** Sample collection context: GPS coordinates, depth, date/time,
temperature, pH, dissolved oxygen, nutrient concentrations, habitat type.
Essential for ecological interpretation.

**Where used:**
- All NCBI BioProject metadata (Exp121, 125, 126)
- 28-biome global QS atlas (Exp126)
- Cold seep catalogs (Exp144-145)
- Michigan EGLE PFAS monitoring (Exp008)

**NestGate needs:**
- Structured environmental metadata schema (MIxS / ENA checklist compatible)
- Geospatial indexing (query by location, depth, habitat)
- Temporal indexing (query by date range, season)
- Link metadata to all associated data objects (reads, assemblies, profiles)

---

## 8. NPU-Specific Data Types

### 8.1 Int8 Quantized Feature Vectors

**What:** Community or sensor features quantized from f64 to i8 for NPU inference.
Affine quantization: `i8_val = round((f64_val - offset) * scale)`.

**Where used:**
- `bio::esn` + `npu::npu_infer_i8` — ESN readout on AKD1000
- `npu::load_reservoir_weights` — f64 → f32 → SRAM
- Exp083, 114-119, 193-195

**NestGate needs:**
- Store quantization parameters (scale, offset) alongside quantized data
- Tag with: source precision (f64), quantization method (affine), NPU target
- Link quantized model to source model (int8 ESN ↔ f64 ESN)

### 8.2 NPU Weight Sets

**What:** ESN reservoir weights (w_in, w_res) and readout weights (w_out) in
int8 format for AKD1000 deployment. Readout weights mutate via (1+1)-ES.

**Where used:**
- `npu::load_readout_weights` — online weight switching
- Exp194 (sim vs live), Exp195 (online evolution, crosstalk)

**NestGate needs:**
- Version-tracked weight sets (generation 0, 1, 2, ... of evolution)
- Tag with: fitness score, target classifier, hardware device PUF
- Serve weights to NPU via DMA-ready format (no transformation at deploy time)

### 8.3 PUF Fingerprints

**What:** Physical Unclonable Function signatures derived from AKD1000 SRAM
response patterns. Hardware-unique, non-clonable device identity.

**Where used:**
- Exp195 S1 — PUF fingerprint (6.34 bits entropy, dual-state alternating)

**NestGate needs:**
- Store PUF fingerprints as device attestation records
- BearDog integration: PUF → device identity → chain of custody
- Link all data collected by a device to its PUF-verified identity

---

## 9. NestGate Gap Analysis

### Currently Implemented

| Capability | Status |
|-----------|:------:|
| BLAKE3 content addressing | yes |
| FASTA via NCBI EFetch | yes |
| JSON-RPC storage API | yes |
| Generic blob storage | yes |

### Needed for wetSpring Evolution

| Capability | Priority | Data Types Served |
|-----------|:--------:|------------------|
| **Format-aware tagging** | P0 | All — every blob needs typed metadata |
| **FASTQ awareness** | P0 | Sequencing reads (Illumina + Nanopore) |
| **Taxonomy resolution** | P0 | Lineage strings, SILVA, NCBI, GTDB cross-ref |
| **Paired file linking** | P1 | FASTQ R1↔R2, raw↔basecalled, reads↔assembly |
| **Reference DB versioning** | P1 | SILVA, MassBank, NCBI taxonomy |
| **OTU/ASV table storage** | P1 | Community matrices with sample metadata |
| **FAST5/POD5 support** | P1 | Nanopore raw signal (field genomics) |
| **Streaming ingestion** | P1 | Real-time sensor + sequencer data |
| **Environmental metadata** | P1 | MIxS-compatible sample context |
| **mzML awareness** | P2 | Mass spectrometry data |
| **Newick tree storage** | P2 | Phylogenetic trees |
| **Model versioning** | P2 | ML models, ESN weights, NPU weight sets |
| **PUF attestation** | P2 | Device identity via BearDog |
| **Distance matrix caching** | P3 | Precomputed Bray-Curtis, UniFrac, ANI |
| **Anderson spectral cache** | P3 | Large lattice eigenvalue results |
| **Triple store** | P3 | Knowledge graph embeddings |
| **BIOM format** | P3 | QIIME2 interop |
| **Geospatial indexing** | P3 | Environmental monitoring stations |

### The NestGate Evolution Pattern

Just as Springs followed Write → Absorb → Lean for ToadStool:

```
wetSpring exercises data types (Write)
  → NestGate builds typed providers (Absorb)
    → Springs lean on NestGate for data access (Lean)
```

Phase 1 (current): wetSpring manages its own data (local files, scripts).
Phase 2 (next): NestGate provides typed storage + retrieval via JSON-RPC.
Phase 3 (target): All Springs route data through NestGate. No local data management.

---

## 10. Data Flow Diagram — Field Genomics Pipeline

```
                                NestGate
                                  │
        ┌─────────────────────────┼──────────────────────┐
        │                         │                      │
   Store + Tag               Serve + Stream         Provenance
        │                         │                      │
  ┌─────┴──────┐           ┌──────┴──────┐        ┌─────┴──────┐
  │ FAST5/POD5 │           │ SILVA ref   │        │ sweetGrass │
  │ FASTQ      │           │ NCBI tax    │        │ PROV-O     │
  │ OTU tables │           │ PFAS lib    │        │ audit trail│
  │ mzML       │           │ ESN weights │        │            │
  │ PUF sigs   │           │ MassBank    │        │            │
  └─────┬──────┘           └──────┬──────┘        └────────────┘
        │                         │
        ▼                         ▼
  MinION → BarraCUDA → NPU → metalForge → Decision/Alert
  (SEQ)    (CPU/GPU)   (AKD)  (dispatch)   (Songbird)
```

---

## References

This catalog was compiled by interviewing:

- 46 Rust CPU modules in `barracuda/src/`
- 45 GPU wrapper modules
- 200 experiment documents in `experiments/`
- 52 reproduced papers in `specs/PAPER_REVIEW_QUEUE.md`
- NestGate v4.1.0-dev codebase (13 crates)
- Oxford Nanopore, Illumina, and MassBank format specifications
- SILVA, RDP, NCBI, and GTDB taxonomy documentation
- Sub-theses 01-06 in `whitePaper/baseCamp/`
