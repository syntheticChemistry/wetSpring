// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated quality filtering — real per-read parallel trimming.
//!
//! Custom WGSL shader processes all reads in parallel (one GPU thread per read).
//! Replicates CPU logic exactly: leading trim → trailing trim → sliding window → min length.
//!
//! # Architecture
//!
//! ```text
//! QualityFilterCached::new(device, ctx)
//!   ├── compile quality_filter.wgsl (one-time)
//!   ├── create pipeline + bind group layout
//!   └── warm driver with tiny dispatch
//!
//! filter_reads_gpu(gpu, reads, params)
//!   ├── pack quality bytes (4 per u32) → GPU storage buffer
//!   ├── upload offsets + lengths → GPU storage buffers
//!   ├── dispatch: 1 thread per read, all reads in parallel
//!   ├── readback: packed (start, end) per read
//!   └── CPU: apply trim results to create filtered FastqRecords
//! ```
//!
//! # `ToadStool` absorption path
//!
//! - `ParallelFilter<T>` — per-element parallel scan + filter primitive
//! - Pre-compiled pipeline cache (like `GemmCached`)
//! - `BufferPool` integration for quality data buffers

use crate::bio::quality::{self, FilterStats, QualityParams};
use crate::error::{Error, Result};
use crate::gpu::GpuF64;
use crate::io::fastq::FastqRecord;
use barracuda::device::{TensorContext, WgpuDevice};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

const QF_WGSL: &str = include_str!("../shaders/quality_filter.wgsl");

/// Workgroup size — must match `@workgroup_size(N)` in `shaders/quality_filter.wgsl`.
const WORKGROUP_SIZE: u32 = 256;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct QfParams {
    n_reads: u32,
    leading_min_quality: u32,
    trailing_min_quality: u32,
    window_min_quality: u32,
    window_size: u32,
    min_length: u32,
    phred_offset: u32,
    _pad: u32,
}

/// Pre-compiled quality filter pipeline with `BufferPool` integration.
pub struct QualityFilterCached {
    device: Arc<WgpuDevice>,
    ctx: Arc<TensorContext>,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

impl QualityFilterCached {
    /// Create a new quality filter GPU instance.
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>, ctx: Arc<TensorContext>) -> Self {
        let shader = device
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("QualityFilter"),
                source: wgpu::ShaderSource::Wgsl(QF_WGSL.into()),
            });

        let bgl = device
            .device()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("QF BGL"),
                entries: &[
                    bgl_entry(0, wgpu::BufferBindingType::Uniform),
                    bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                    bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                    bgl_entry(3, wgpu::BufferBindingType::Storage { read_only: true }),
                    bgl_entry(4, wgpu::BufferBindingType::Storage { read_only: false }),
                ],
            });

        let pl = device
            .device()
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("QF PL"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });

        let pipeline = device
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("QualityFilter"),
                layout: Some(&pl),
                module: &shader,
                entry_point: "quality_filter",
                cache: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        Self {
            device,
            ctx,
            pipeline,
            bgl,
        }
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

        let pool = self.ctx.buffer_pool();
        let (packed_quals, offsets, lengths) = pack_quality_data(reads);

        let gpu_params = QfParams {
            n_reads: n as u32,
            leading_min_quality: u32::from(params.leading_min_quality),
            trailing_min_quality: u32::from(params.trailing_min_quality),
            window_min_quality: u32::from(params.window_min_quality),
            window_size: params.window_size as u32,
            min_length: params.min_length as u32,
            phred_offset: u32::from(params.phred_offset),
            _pad: 0,
        };

        let params_buf = self.device.create_uniform_buffer("QF Params", &gpu_params);

        let qual_buf = pool.acquire_pooled(packed_quals.len() * 4);
        self.device
            .queue()
            .write_buffer(&qual_buf, 0, bytemuck::cast_slice(&packed_quals));

        let offsets_buf = pool.acquire_pooled(offsets.len() * 4);
        self.device
            .queue()
            .write_buffer(&offsets_buf, 0, bytemuck::cast_slice(&offsets));

        let lengths_buf = pool.acquire_pooled(lengths.len() * 4);
        self.device
            .queue()
            .write_buffer(&lengths_buf, 0, bytemuck::cast_slice(&lengths));

        let results_buf = pool.acquire_pooled(n * 4);

        let bg = self
            .device
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("QF BG"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: qual_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: offsets_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: lengths_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: results_buf.as_entire_binding(),
                    },
                ],
            });

        let wg_x = (n as u32).div_ceil(WORKGROUP_SIZE);
        let mut encoder =
            self.device
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("QF Encoder"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("QF Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(wg_x, 1, 1);
        }
        self.device.queue().submit(Some(encoder.finish()));

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

const fn bgl_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
