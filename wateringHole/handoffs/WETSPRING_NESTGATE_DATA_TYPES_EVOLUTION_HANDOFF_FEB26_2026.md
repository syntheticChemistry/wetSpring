# NestGate — Data Types Evolution Handoff

**Date:** February 26, 2026
**From:** wetSpring (life science validation spring)
**To:** NestGate data primal team
**Covers:** Biological data type profiling, priority gap analysis, evolution roadmap
**Reference:** `wetSpring/specs/DATA_TYPES.md` (full catalog, 636 lines)
**License:** AGPL-3.0-only

---

## Executive Summary

wetSpring has profiled every biological data type flowing through 200
experiments, 52 reproduced papers, 46 CPU modules, and 42 GPU modules.
The result: NestGate currently stores blobs. It needs to *understand* what
those blobs are. This handoff defines what "understand" means — format-aware
tagging, paired file linking, taxonomy resolution, streaming ingestion — and
prioritizes the work into 4 tiers.

The evolution follows the same Write → Absorb → Lean pattern that worked for
ToadStool: wetSpring exercises data types, NestGate builds typed providers,
Springs lean on NestGate for data access.

---

## Part 1: What NestGate Does Today

| Capability | Status | Notes |
|-----------|:------:|-------|
| BLAKE3 content addressing | **yes** | All blobs get a deterministic hash |
| FASTA via NCBI EFetch | **yes** | `NCBILiveProvider` fetches sequences on demand |
| JSON-RPC storage API | **yes** | `store_blob`, `get_blob`, `list_blobs` |
| Generic blob storage | **yes** | Any bytes in, hash out |
| ZFS integration | **yes** | Copy-on-write, snapshots, checksums |
| 13 crates, 1,474/1,475 tests | **yes** | Production-grade Rust, 0 C deps |

**What's missing:** NestGate doesn't know the difference between a FASTQ file
and a JPEG. Every blob is bytes. No typed metadata, no format validation, no
relationship tracking between related files.

---

## Part 2: What wetSpring Needs NestGate to Understand

### Tier P0 — Must Have (blocks current and planned experiments)

#### 2a. Format-Aware Tagging

Every blob stored in NestGate needs typed metadata. Not a free-form JSON
field — a structured, validated tag system.

**Proposed API extension:**

```rust
pub struct BlobMetadata {
    pub format: DataFormat,       // FASTQ, FASTA, POD5, mzML, Newick, ...
    pub entity: BiologicalEntity,  // 16S_reads, genome, community_profile, ...
    pub source: DataSource,        // experiment ID, NCBI accession, MinION run ID
    pub quality: Option<QualityInfo>,  // Phred encoding, coverage, confidence
    pub tags: Vec<(String, String)>,   // flexible k/v for domain-specific metadata
}

pub enum DataFormat {
    Fastq, FastqPaired,
    Fasta,
    Fast5, Pod5,
    MzMl, Ms2,
    Newick,
    OtuTable, AsvTable,
    EsnWeights, NpuWeightSet,
    PufFingerprint,
    CommunityProfile,
    TaxonomyLineage,
    DistanceMatrix,
    AndersonSpectral,
    SensorTimeSeries,
    MixsMetadata,
}
```

**Why P0:** Without typed metadata, NestGate cannot answer "show me all 16S
FASTQ files from bloom-site experiments." Every downstream capability
depends on this.

#### 2b. FASTQ Awareness

FASTQ is the most common file format in bioinformatics. wetSpring parses
~500K reads per experiment across 200 experiments. NestGate needs to:

1. **Validate on ingest** — confirm 4-line-per-record structure, Phred+33/+64
2. **Extract metadata** — read count, mean quality, sequence length distribution
3. **Tag with provenance** — which BioProject, which instrument, which primer set

**Volume context:**
- Illumina MiSeq: 1-8M reads, 1-4 GB per run, paired (R1+R2)
- Illumina HiSeq: 100M+ reads, 30-100 GB per lane
- Nanopore MinION: variable, 1-20 GB per flow cell, single-end, long reads

**What wetSpring validates:** `io::fastq` (Exp004, 011, 012, 014, 017).
The parser exists. NestGate needs the storage semantics around it.

#### 2c. Taxonomy Resolution

This is the most insidious data problem in microbiology. Four major taxonomy
databases exist, and they disagree:

| Database | Format | Example |
|----------|--------|---------|
| **SILVA** | `d__Bacteria;p__Proteobacteria;c__Gamma...` | Release 138.2 |
| **NCBI** | Integer taxon IDs + lineage | txid562 = *E. coli* |
| **GTDB** | SILVA-style but reclassified | r220 |
| **RDP** | Hierarchical classifier output | Bootstrap confidence per rank |

Every wetSpring diversity experiment (Exp004-017, 039-040, 051-056, 170-178)
produces taxonomy assignments. Currently stored as raw strings. NestGate needs:

1. **Cross-reference layer** — given a SILVA lineage, resolve NCBI taxID and GTDB name
2. **Version tracking** — SILVA 138.1 → 138.2 renames genera; results change
3. **Rank normalization** — standardize domain/phylum/class/order/family/genus/species

**Bench context (from the wetSpring author):** Used RDP in the rumen
microbiology lab (undergrad), SILVA for 16S pipeline work at Sandia, NCBI for
EFetch integrations. The cross-reference problem is real — the same bacterium
has different identifiers in each database, and reclassifications happen with
every release.

---

### Tier P1 — High Priority (needed for next-generation experiments)

#### 2d. Paired File Linking

Bioinformatics data almost never exists as isolated files:

| Pair Type | Relationship | Example |
|-----------|-------------|---------|
| FASTQ R1 ↔ R2 | Paired-end reads | Forward/reverse from same fragment |
| Raw ↔ basecalled | Signal → sequence | POD5 → FASTQ (Nanopore) |
| Reads ↔ assembly | Input → output | FASTQ → contigs FASTA |
| Sample ↔ metadata | Data → context | FASTQ → MIxS sample sheet |
| Community ↔ reference | Assignment → database | OTU table → SILVA 138.2 |

NestGate needs a **relationship graph** alongside the blob store. Not a full
triple store (that's P3) — just directional edges with typed relationships.

```rust
pub enum BlobRelation {
    PairedWith,       // R1 ↔ R2
    DerivedFrom,      // basecalled FASTQ ← raw POD5
    AssembledFrom,    // contigs ← reads
    ClassifiedAgainst, // OTU table ← SILVA 138.2
    CollectedWith,    // FASTQ + metadata
    ProducedBy,       // output ← experiment binary
}
```

#### 2e. Reference Database Versioning

wetSpring leans on external reference databases that change:

| Database | Release Cycle | Impact of Update |
|----------|:------------:|-----------------|
| SILVA | ~annual | Genus renames, new phyla, reclassifications |
| NCBI taxonomy | monthly | New species, merged taxIDs |
| GTDB | ~6 months | Major reclassifications |
| MassBank | continuous | New spectra, mass accuracy updates |
| PFAS library | as-needed | New PFAS compounds discovered |

NestGate needs to version-pin reference databases so that results are
reproducible. If you classify reads against SILVA 138.1, you need to be
able to re-run with the exact same database 2 years later.

#### 2f. POD5 / FAST5 Support (Field Genomics Gate)

Oxford Nanopore's raw signal formats. **This is the gate for Exp196-202.**

- **FAST5**: HDF5-based, one file per read or multi-read batches. Legacy.
- **POD5**: Apache Arrow-based, columnar, streaming-friendly. Current standard.

NestGate needs:
1. Recognize and tag POD5/FAST5 on ingest
2. Extract signal metadata (channel, sample rate, calibration)
3. Link raw → basecalled FASTQ (DerivedFrom relationship)
4. Handle streaming ingestion (MinION produces reads continuously, not as a
   batch file dump)

**Volume:** A single MinION flow cell generates 1-20 GB of POD5. At 450 bp/s
per pore × 512 active pores, that's ~230 Kbp/s sustained. NestGate needs to
keep up.

#### 2g. Streaming Ingestion

Current NestGate API is request/response: store a blob, get a blob. Field
genomics needs real-time data flow:

- MinION produces reads continuously over 24-72 hours
- Environmental sensors produce time-series data every 1-60 seconds
- NPU classifiers produce classifications in real-time (12.9K Hz)

NestGate needs a streaming ingest endpoint — not "here's a complete file"
but "here's a chunk, more is coming, tag it as part of this run."

#### 2h. OTU/ASV Table Storage

Community composition matrices: rows = samples, columns = taxa, values =
counts. This is the core output of every 16S experiment.

**Current state:** wetSpring computes these in-memory and writes them as
part of validation binary output. They're not stored in NestGate.

**Need:** Store OTU/ASV tables with linked metadata (which samples, which
taxonomy database, which pipeline version). Enable queries like "all bloom-site
community profiles from February 2026."

#### 2i. Environmental Metadata (MIxS)

The MIxS standard (Genomic Standards Consortium) defines minimum metadata for
sequencing runs: sample collection date, GPS coordinates, environment type,
sequencing platform, primer sequences, etc.

NestGate should enforce MIxS-compatible metadata on all field genomics data.
This makes wetSpring data interoperable with global databases (ENA, SRA,
MGnify) without reformatting.

---

### Tier P2 — Medium Priority (enriches existing capabilities)

#### 2j. Model Versioning (ESN Weights, NPU Weight Sets)

wetSpring trains ESN classifiers (QS, Bloom, Disorder) and deploys them to
AKD1000 NPU. Exp195 demonstrated online evolution — the classifier gets
*better over time* on the device. NestGate needs:

1. Version-tracked weight sets (generation 0, 1, 2, ... of evolution)
2. Tag with: fitness score, target classifier, hardware device PUF
3. Serve weights in DMA-ready format (int8 affine, no transformation at deploy)

**Bench relevance:** This is analogous to tracking protocol versions in a wet
lab notebook — which version of the staining protocol produced which result.

#### 2k. mzML Awareness

Mass spectrometry interchange format (XML-based). Used in PFAS detection
(Exp005-009, 018, 041-042). wetSpring parses it via `io::mzml`.

NestGate should:
1. Validate mzML structure on ingest
2. Extract metadata (instrument, acquisition time, scan count)
3. Tag with chemistry context (PFAS screening, general metabolomics, etc.)

#### 2l. Newick Tree Storage

Phylogenetic trees in Newick format. Used in Exp019-034, 036-038 (phylogenetics
track). Simple text format but relationships are complex (nested parenthetical).

NestGate should store trees as typed blobs with metadata: number of tips,
method (NJ, ML, Bayesian), associated alignment, bootstrap support.

#### 2m. PUF Attestation (BearDog Integration)

PUF (Physical Unclonable Function) fingerprints from AKD1000 hardware
(Exp195 S1: 6.34 bits entropy). These are device identity markers.

NestGate stores the fingerprint. BearDog verifies it. Together they provide
chain of custody: "this classification was produced by device X, whose identity
is attested by PUF Y."

**Integration path:** NestGate ↔ BearDog via existing JSON-RPC.

---

### Tier P3 — Long-term (infrastructure for future science)

| Capability | Use Case | Notes |
|-----------|---------|-------|
| Distance matrix caching | Bray-Curtis, UniFrac, ANI precomputation | Saves hours of GPU time for large datasets |
| Anderson spectral cache | Large lattice (L=14-20) eigenvalue results | DF64 computations are expensive; cache results |
| Triple store / KG | Knowledge graph embeddings (Exp161) | ROBOKOP-style drug repurposing |
| BIOM format | QIIME2 interop for collaborators | Enables data exchange with QIIME2 users |
| Geospatial indexing | "All data within 50km of Saginaw Bay" | Environmental monitoring station queries |

---

## Part 3: The Evolution Pattern

```
Phase 1 (current):
  wetSpring manages its own data locally.
  NestGate stores blobs via JSON-RPC.
  No typed awareness.

Phase 2 (next):
  NestGate adds format-aware tagging (P0).
  wetSpring stores FASTQ/FASTA/OTU with typed metadata.
  Taxonomy resolution enables cross-database queries.
  Paired file linking tracks R1↔R2, raw↔basecalled.

Phase 3 (target):
  All Springs route data through NestGate.
  Streaming ingestion handles MinION + sensors.
  Reference DB versioning ensures reproducibility.
  NestGate becomes the single source of truth for
  biological data across the ecosystem.
```

This mirrors ToadStool's evolution:

```
Springs exercised compute primitives (Write)
  → ToadStool built universal shaders (Absorb)
    → Springs deleted local math (Lean)

Springs exercise data types (Write)         ← we are here
  → NestGate builds typed providers (Absorb)
    → Springs delete local data management (Lean)
```

---

## Part 4: What wetSpring Has Already Validated

The following data handling is battle-tested across 200 experiments. NestGate
can lean on this validation when building typed providers:

| Data Type | wetSpring Module | Experiments | Checks |
|-----------|-----------------|:-----------:|:------:|
| FASTQ parsing | `io::fastq` | 004, 011-017, 039-040 | 400+ |
| FASTA retrieval | `ncbi::efetch` | 121, 125, 184-185 | 50+ |
| mzML parsing | `io::mzml` | 005-009, 018, 041-042 | 50+ |
| MS2 parsing | `io::ms2` | 007, 042 | 20+ |
| Newick parsing | `bio::unifrac::tree` | 019-034, 036-038 | 200+ |
| OTU table computation | `bio::diversity` + `bio::dada2` | 004-017, 039-040, 051-056 | 500+ |
| Taxonomy classification | `bio::taxonomy` | 011-017, 039-040, 170-178 | 300+ |
| ESN weights | `bio::esn` | 114-119, 123, 193-195 | 100+ |
| NPU int8 quantization | `npu` | 193-195 | 60 |
| Community profiles | `bio::diversity` | 039-040, 112-113, 170-178 | 400+ |
| Anderson spectral | `spectral::*` (via barracuda) | 107-156 | 3,100+ |

---

## Part 5: Suggested NestGate Development Sequence

Based on wetSpring's immediate needs and the field genomics roadmap:

1. **Format-aware tagging** (P0) — add `BlobMetadata` to the storage API.
   Every subsequent capability depends on this.

2. **FASTQ awareness** (P0) — validate, extract metadata, paired linking.
   This is the most-exercised format (400+ validation checks).

3. **Taxonomy resolution** (P0) — cross-reference SILVA/NCBI/GTDB/RDP.
   Blocks every community analysis query.

4. **POD5 support** (P1) — the field genomics gate. Until NestGate can
   ingest Nanopore signal data, Exp196-202 are blocked.

5. **Streaming ingestion** (P1) — enables real-time MinION + sensor data.
   The field sentinel concept requires this.

6. **Reference DB versioning** (P1) — reproducibility requirement. Without
   this, rerunning a pipeline on a new SILVA release produces different
   results with no way to trace back.

7. **Everything else** — OTU tables, mzML, Newick, model versioning, PUF
   attestation. Important but not blocking.

---

## Part 6: Data Volume Budget

For NestGate capacity planning:

| Source | Volume per Run | Frequency | Storage/Year |
|--------|:-------------:|:---------:|:------------:|
| Illumina FASTQ (16S) | 2-4 GB | weekly | ~200 GB |
| Nanopore POD5 (field) | 5-20 GB | daily (field season) | ~2 TB |
| OTU/ASV tables | 1-10 MB | per FASTQ run | ~5 GB |
| mzML (LC-MS) | 0.5-2 GB | monthly | ~25 GB |
| ESN weight sets | 100 KB | per evolution | ~1 GB |
| Environmental sensors | 10 KB/day | continuous | ~4 MB |
| Anderson spectral cache | 10-500 MB | per lattice | ~50 GB |
| Reference DBs (SILVA) | 1-5 GB | annual | ~5 GB |

**Total estimated:** ~2.5 TB/year at full field deployment. ZFS compression
and deduplication should bring this under 1 TB effective.

---

## Appendix: Full Data Type Catalog

See `wetSpring/specs/DATA_TYPES.md` for the complete 636-line catalog with
Rust struct definitions, volume estimates, and per-module provenance for
every data type wetSpring touches.
