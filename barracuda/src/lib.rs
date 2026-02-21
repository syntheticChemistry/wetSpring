// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(missing_docs, clippy::pedantic, clippy::nursery)]
#![allow(
    clippy::module_name_repetitions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names
)]
//! wetSpring `BarraCUDA` — Life Science + PFAS Analytical Chemistry Pipelines
//!
//! Rust implementations validated against Python/Galaxy/QIIME2/asari/`FindPFAS`
//! baselines. Each module mirrors a pipeline stage from the validated
//! experiments (Exp001–006).
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
//! - [`bio::tolerance_search`] — ppm-tolerance m/z matching for suspect screening
//! - [`bio::spectral_match`] — MS2 cosine similarity for library matching
//! - [`bio::kmd`] — Kendrick mass defect analysis for PFAS homologue detection
//!
//! # Signal Processing + Feature Extraction
//! - [`bio::signal`] — 1D peak detection (`find_peaks`, `scipy` equivalent)
//! - [`bio::eic`] — Extracted Ion Chromatogram / mass track extraction
//! - [`bio::feature_table`] — End-to-end feature extraction (asari pipeline)
//!
//! # GPU acceleration (feature = "gpu")
//! - `gpu` — GPU device wrapper bridging to `ToadStool` `WgpuDevice` (wgpu v22)
//! - `bio::diversity_gpu` — Shannon, Simpson, observed, evenness, alpha via `FusedMapReduceF64`,
//!   Bray-Curtis via `BrayCurtisF64` (absorbed upstream)
//! - `bio::pcoa_gpu` — `PCoA` ordination via `ToadStool`'s `BatchedEighGpu`
//! - `bio::spectral_match_gpu` — Pairwise cosine via `GemmF64` + `FusedMapReduceF64`
//! - `bio::kriging` — Spatial interpolation via `ToadStool`'s `KrigingF64`
//! - `bio::stats_gpu` — Variance, correlation, covariance, weighted dot via
//!   `VarianceF64`, `CorrelationF64`, `CovarianceF64`, `WeightedDotF64`
//! - [`tolerances`] — Centralized validation tolerances (CPU, GPU vs CPU)
//!
//! # Evolution path
//!
//! ```text
//! Python baseline → Rust validation (here) → GPU acceleration → sovereign pipeline
//! ```

pub mod bench;
pub mod bio;
pub mod encoding;
pub mod error;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod io;
pub mod tolerances;
pub mod validation;
