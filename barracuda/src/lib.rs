// SPDX-License-Identifier: AGPL-3.0-or-later
#![deny(unsafe_code)]
#![deny(clippy::expect_used, clippy::unwrap_used)]
#![deny(missing_docs)]
#![warn(clippy::pedantic, clippy::nursery)]
#![allow(
    clippy::module_name_repetitions,
    // GPU interop requires pervasive usize↔u32↔f64 conversion (wgpu APIs
    // use u32 for buffer sizes, dispatch counts; f64 for scientific values).
    // Individual modules add targeted #[allow] where precision loss is
    // asserted safe by domain constraints (e.g., n_samples < 2^32).
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! wetSpring — Life Science + PFAS Analytical Chemistry Pipelines (via `barraCuda`)
//!
//! Rust implementations validated against Python/`Galaxy`/`QIIME2`/`asari`/`FindPFAS`
//! baselines. Each module mirrors a pipeline stage from the validated
//! experiments (Exp001–097, 29 reproduced papers across 4 tracks).
//!
//! # Track 1: Life Science (16S amplicon, phage, diversity)
//! - [`io::fastq`] — FASTQ parsing + quality statistics
//! - [`bio::quality`] — Quality filtering + adapter trimming (Trimmomatic/Cutadapt)
//! - [`bio::merge_pairs`] — Paired-end read merging (VSEARCH/FLASH equivalent)
//! - [`bio::derep`] — Dereplication with abundance tracking (VSEARCH equivalent)
//! - [`bio::kmer`] — K-mer counting with 2-bit encoding
//! - [`bio::diversity`] — Alpha/beta diversity (Shannon, Simpson, Chao1, Bray-Curtis,
//!   Pielou evenness, rarefaction curves)
//! - [`bio::pcoa`] — Principal Coordinates Analysis (CPU Jacobi)
//!
//! # Track 2: PFAS Analytical Chemistry (LC-MS, PFAS screening)
//! - [`io::mzml`] — mzML mass spectrometry data parser (streaming)
//! - [`io::ms2`] — MS2 text format parser (streaming)
//!
//! # Field Genomics (Sub-thesis 06)
//! - [`io::nanopore`] — Nanopore raw signal parser (POD5/NRS streaming)
//! - [`bio::tolerance_search`] — ppm-tolerance m/z matching for suspect screening
//! - [`bio::spectral_match`] — MS2 cosine similarity for library matching
//! - [`bio::kmd`] — Kendrick mass defect analysis for PFAS homologue detection
//!
//! # Signal Processing + Feature Extraction
//! - [`bio::signal`] — 1D peak detection (`find_peaks`, `scipy` equivalent)
//! - [`bio::eic`] — Extracted Ion Chromatogram / mass track extraction
//! - [`bio::feature_table`] — End-to-end feature extraction (`asari` pipeline)
//!
//! # GPU acceleration (feature = "gpu")
//! - `gpu` — GPU device wrapper bridging to barraCuda `WgpuDevice` (wgpu v28)
//! - `bio::diversity_gpu` — Shannon, Simpson, observed, evenness, alpha via `FusedMapReduceF64`,
//!   Bray-Curtis via `BrayCurtisF64` (absorbed upstream)
//! - `bio::pcoa_gpu` — `PCoA` ordination via barraCuda's `BatchedEighGpu`
//! - `bio::spectral_match_gpu` — Pairwise cosine via `GemmF64` + `FusedMapReduceF64`
//! - `bio::kriging` — Spatial interpolation via barraCuda's `KrigingF64`
//! - `bio::stats_gpu` — Variance, correlation, covariance, weighted dot via
//!   `VarianceF64`, `CorrelationF64`, `CovarianceF64`, `WeightedDotF64`
//! - [`tolerances`] — Centralized validation tolerances (CPU, GPU vs CPU)
//!
//! # Shared Mathematics
//! - [`special`] — Error function, gamma, regularized gamma (sovereign, no libm)
//!
//! # Infrastructure
//! - [`validation`] — `hotSpring` validation framework (pass/fail checks, exit codes)
//! - [`ncbi`] — NCBI Entrez helpers (API key discovery, HTTP GET, E-search)
//! - [`encoding`] — Sovereign base64 encode/decode (RFC 4648, zero external deps)
//! - [`error`] — Error types for all parsers and algorithms
//! - [`bench`](mod@bench) — Benchmarking utilities and hardware detection
//!
//! # Evolution path
//!
//! ```text
//! Python baseline → Rust validation (here) → GPU acceleration → sovereign pipeline
//! ```

/// This primal's canonical identifier — used for IPC, provenance, and metrics.
pub const PRIMAL_NAME: &str = "wetspring";

/// Key derivation context prefix — versioned for forward compatibility.
pub const VAULT_KEY_CONTEXT: &str = "wetspring-vault-encryption-v1";

pub mod bench;
pub mod bio;
pub mod df64_host;
pub mod encoding;
pub mod error;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod io;
#[cfg(feature = "ipc")]
pub mod ipc;
pub mod ncbi;
#[cfg(feature = "npu")]
pub mod npu;
#[cfg(feature = "gpu")]
pub mod provenance;
pub mod special;
pub mod tolerances;
pub mod validation;
#[cfg(feature = "vault")]
pub mod vault;
