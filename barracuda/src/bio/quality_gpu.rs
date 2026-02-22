// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated quality filtering via `ToadStool`.
//!
//! Delegates to `barracuda::ops::bio::quality_filter::QualityFilterGpu` —
//! the absorbed shader from wetSpring handoff v6. wetSpring provides the
//! high-level API that packs FASTQ quality bytes and applies trim results.
//!
//! # Architecture
//!
//! ```text
//! QualityFilterCached::new(device)
//!   └── QualityFilterGpu::new(device)  ← ToadStool pre-compiles shader
//!
//! filter_reads_gpu(gpu, reads, params)
//!   ├── pack quality bytes (4 per u32) → GPU storage buffer
//!   ├── upload offsets + lengths → GPU storage buffers
//!   ├── dispatch: ToadStool handles pipeline + bind group + workgroups
//!   ├── readback: packed (start, end) per read
//!   └── CPU: apply trim results to create filtered FastqRecords
//! ```

use crate::bio::quality::{self, FilterStats, QualityParams};
use crate::error::{Error, Result};
use crate::gpu::GpuF64;
use crate::io::fastq::FastqRecord;
use barracuda::device::WgpuDevice;
use barracuda::ops::bio::quality_filter::{QualityConfig, QualityFilterGpu as ToadStoolQF};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Pre-compiled quality filter pipeline via `ToadStool`.
pub struct QualityFilterCached {
    device: Arc<WgpuDevice>,
    inner: ToadStoolQF,
}

impl QualityFilterCached {
    /// Create a new quality filter GPU instance.
    ///
    /// # Errors
    ///
    /// Returns an error if `ToadStool` shader compilation fails.
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        let inner = ToadStoolQF::new(Arc::clone(&device))
            .map_err(|e| Error::Gpu(format!("QualityFilterGpu: {e}")))?;
        Ok(Self { device, inner })
    }

    /// Run quality filtering on GPU. Returns (start, end) per read, or None for failed reads.
    ///
    /// # Errors
    ///
    /// Returns an error if GPU dispatch or readback fails.
    #[allow(clippy::cast_possible_truncation)]
    pub fn execute(
        &self,
        reads: &[FastqRecord],
        params: &QualityParams,
    ) -> Result<Vec<Option<(usize, usize)>>> {
        let n = reads.len();
        if n == 0 {
            return Ok(vec![]);
        }

        let (packed_quals, offsets, lengths) = pack_quality_data(reads);
        let d = self.device.device();

        let config = QualityConfig {
            leading_min_quality: u32::from(params.leading_min_quality),
            trailing_min_quality: u32::from(params.trailing_min_quality),
            window_min_quality: u32::from(params.window_min_quality),
            window_size: params.window_size as u32,
            min_length: params.min_length as u32,
            phred_offset: u32::from(params.phred_offset),
        };

        let qual_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("QF qual_data"),
            contents: bytemuck::cast_slice(&packed_quals),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let offsets_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("QF offsets"),
            contents: bytemuck::cast_slice(&offsets),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let lengths_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("QF lengths"),
            contents: bytemuck::cast_slice(&lengths),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let results_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("QF results"),
            contents: bytemuck::cast_slice(&vec![0u32; n]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        self.inner
            .dispatch(
                n as u32,
                &config,
                &qual_buf,
                &offsets_buf,
                &lengths_buf,
                &results_buf,
            )
            .map_err(|e| Error::Gpu(format!("QF dispatch: {e}")))?;

        d.poll(wgpu::Maintain::Wait);

        let raw_results = self
            .device
            .read_buffer_u32(&results_buf, n)
            .map_err(|e| Error::Gpu(format!("QF readback: {e}")))?;

        Ok(raw_results
            .iter()
            .map(|&r| {
                if r == 0 {
                    None
                } else {
                    let start = (r >> 16) as usize;
                    let end = (r & 0xFFFF) as usize;
                    Some((start, end))
                }
            })
            .collect())
    }
}

/// Quality-filter reads using GPU-accelerated quality computation.
///
/// # Errors
///
/// Returns an error if the device lacks `SHADER_F64` or GPU dispatch fails.
pub fn filter_reads_gpu(
    gpu: &GpuF64,
    reads: &[FastqRecord],
    params: &QualityParams,
    qf_cached: Option<&QualityFilterCached>,
) -> Result<(Vec<FastqRecord>, FilterStats)> {
    if !gpu.has_f64 {
        return Err(Error::Gpu("SHADER_F64 required for quality GPU".into()));
    }

    if let Some(qf) = qf_cached {
        let trim_results = qf.execute(reads, params)?;
        let mut output = Vec::with_capacity(reads.len());
        let mut stats = FilterStats {
            input_reads: reads.len(),
            output_reads: 0,
            discarded_reads: 0,
            leading_bases_trimmed: 0,
            trailing_bases_trimmed: 0,
            window_bases_trimmed: 0,
            adapter_bases_trimmed: 0,
        };

        for (record, trim) in reads.iter().zip(trim_results.iter()) {
            if let Some((start, end)) = trim {
                stats.leading_bases_trimmed += *start as u64;
                stats.trailing_bases_trimmed += (record.quality.len() - end) as u64;
                output.push(quality::apply_trim(record, *start, *end));
                stats.output_reads += 1;
            } else {
                stats.discarded_reads += 1;
            }
        }

        Ok((output, stats))
    } else {
        let (filtered, stats) = quality::filter_reads(reads, params);
        Ok((filtered, stats))
    }
}

#[allow(clippy::cast_possible_truncation)]
fn pack_quality_data(reads: &[FastqRecord]) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut offsets = Vec::with_capacity(reads.len());
    let mut lengths = Vec::with_capacity(reads.len());
    let mut byte_offset = 0u32;

    for read in reads {
        offsets.push(byte_offset);
        lengths.push(read.quality.len() as u32);
        byte_offset += read.quality.len() as u32;
    }

    let total_bytes = byte_offset as usize;
    let n_words = total_bytes.div_ceil(4);
    let mut packed = vec![0u32; n_words.max(1)];

    let mut pos = 0usize;
    for read in reads {
        for &q in &read.quality {
            let word_idx = pos / 4;
            let byte_pos = pos % 4;
            packed[word_idx] |= u32::from(q) << (byte_pos * 8);
            pos += 1;
        }
    }

    (packed, offsets, lengths)
}
