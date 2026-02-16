//! wetSpring BarraCUDA — Life Science + PFAS Analytical Chemistry Pipelines
//!
//! Rust implementations validated against Python/Galaxy/QIIME2/asari/FindPFAS baselines.
//! Each module mirrors a pipeline stage from the validated experiments (Exp001-006).
//!
//! # Track 1: Life Science (16S amplicon, phage, diversity)
//! - `io::fastq` — FASTQ parsing + quality statistics
//! - `bio::kmer` — K-mer counting with 2-bit encoding
//! - `bio::diversity` — Alpha/beta diversity metrics (Shannon, Simpson, Chao1, Bray-Curtis)
//!
//! # Track 2: PFAS Analytical Chemistry (LC-MS, PFAS screening)
//! - `io::mzml` — mzML mass spectrometry data parser
//! - `io::ms2` — MS2 text format parser
//! - `bio::tolerance_search` — ppm-tolerance m/z matching for suspect screening

pub mod io;
pub mod bio;
