// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated position-parallel SNP calling.
//!
//! One thread per alignment column. Each thread counts allele
//! frequencies across all sequences at its position and reports
//! whether the site is polymorphic.
//!
//! Uses a local WGSL shader (`snp_calling_f64.wgsl`) — a `ToadStool`
//! absorption candidate following Write → Absorb → Lean.
//!
//! # GPU Strategy
//!
//! Alignment positions are independent → one thread per column.
//! For L positions × N sequences, dispatch L threads. Each thread
//! reads N values (column-wise). No transcendentals needed.

use barracuda::device::WgpuDevice;
use barracuda::shaders::precision::ShaderTemplate;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

const SNP_WGSL: &str = include_str!("../shaders/snp_calling_f64.wgsl");

/// Workgroup size — must match `@workgroup_size(N)` in `shaders/snp_calling_f64.wgsl`.
const WORKGROUP_SIZE: u32 = 256;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SnpGpuParams {
    alignment_length: u32,
    n_sequences: u32,
    min_depth: u32,
    _pad: u32,
}

/// GPU SNP calling result.
pub struct SnpGpuResult {
    /// 1 if position is a variant, 0 otherwise. Length = `alignment_length`.
    pub is_variant: Vec<u32>,
    /// Reference allele at each position (0=A,1=C,2=G,3=T).
    pub ref_alleles: Vec<u32>,
    /// Depth at each position.
    pub depths: Vec<u32>,
    /// Alt allele frequency at each position (0.0 for non-variants).
    pub alt_frequencies: Vec<f64>,
}

/// GPU-accelerated batch SNP calling.
pub struct SnpGpu {
    device: Arc<WgpuDevice>,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

struct SnpBuffers {
    params: wgpu::Buffer,
    sequences: wgpu::Buffer,
    variant: wgpu::Buffer,
    ref_allele: wgpu::Buffer,
    depth: wgpu::Buffer,
    alt_freq: wgpu::Buffer,
}

fn create_snp_buffers(
    d: &wgpu::Device,
    params: &SnpGpuParams,
    flat_seqs: &[u32],
    aln_len: usize,
) -> SnpBuffers {
    SnpBuffers {
        params: d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SNP params"),
            contents: bytemuck::bytes_of(params),
            usage: wgpu::BufferUsages::UNIFORM,
        }),
        sequences: d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SNP sequences"),
            contents: bytemuck::cast_slice(flat_seqs),
            usage: wgpu::BufferUsages::STORAGE,
        }),
        variant: d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SNP is_variant"),
            contents: bytemuck::cast_slice(&vec![0u32; aln_len]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        }),
        ref_allele: d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SNP ref_allele"),
            contents: bytemuck::cast_slice(&vec![0u32; aln_len]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        }),
        depth: d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SNP depth"),
            contents: bytemuck::cast_slice(&vec![0u32; aln_len]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        }),
        alt_freq: d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SNP alt_freq"),
            contents: bytemuck::cast_slice(&vec![0.0_f64; aln_len]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        }),
    }
}

impl SnpGpu {
    /// Create a new SNP GPU instance.
    #[must_use]
    pub fn new(device: &Arc<WgpuDevice>) -> Self {
        let patched = ShaderTemplate::for_driver_auto(SNP_WGSL, false);
        let module = device.compile_shader(&patched, Some("SnpCallingF64"));
        let pipeline = device
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("SnpCallingF64"),
                layout: None,
                module: &module,
                entry_point: "main",
                cache: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });
        let bgl = pipeline.get_bind_group_layout(0);
        Self {
            device: Arc::clone(device),
            pipeline,
            bgl,
        }
    }

    /// Call SNPs from aligned sequences on the GPU.
    ///
    /// Sequences as byte slices (ASCII A/C/G/T, gaps as `-`/`.`/`N`).
    /// All sequences must have the same length.
    ///
    /// # Errors
    ///
    /// Returns `Err` if GPU buffer creation or readback fails.
    #[allow(clippy::cast_possible_truncation)]
    pub fn call_snps(&self, sequences: &[&[u8]]) -> crate::error::Result<SnpGpuResult> {
        let n_sequences = sequences.len();
        if n_sequences == 0 {
            return Ok(SnpGpuResult {
                is_variant: Vec::new(),
                ref_alleles: Vec::new(),
                depths: Vec::new(),
                alt_frequencies: Vec::new(),
            });
        }

        let aln_len = sequences[0].len();

        let encode_base = |b: u8| -> u32 {
            match b.to_ascii_uppercase() {
                b'A' => 0,
                b'C' => 1,
                b'G' => 2,
                b'T' => 3,
                _ => 4,
            }
        };

        let mut flat_seqs: Vec<u32> = vec![4; n_sequences * aln_len];
        for (s, seq) in sequences.iter().enumerate() {
            for (j, &byte) in seq.iter().enumerate() {
                flat_seqs[s * aln_len + j] = encode_base(byte);
            }
        }

        let dev = &self.device;
        let d = dev.device();

        let params = SnpGpuParams {
            alignment_length: aln_len as u32,
            n_sequences: n_sequences as u32,
            min_depth: 2,
            _pad: 0,
        };

        let bufs = create_snp_buffers(d, &params, &flat_seqs, aln_len);

        let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bufs.params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bufs.sequences.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bufs.variant.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: bufs.ref_allele.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: bufs.depth.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: bufs.alt_freq.as_entire_binding(),
                },
            ],
        });

        let mut encoder = d.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("SNP calling"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((aln_len as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
        }

        dev.queue().submit(Some(encoder.finish()));
        d.poll(wgpu::Maintain::Wait);

        let is_variant = dev
            .read_buffer_u32(&bufs.variant, aln_len)
            .map_err(|e| crate::error::Error::Gpu(format!("{e}")))?;
        let ref_alleles = dev
            .read_buffer_u32(&bufs.ref_allele, aln_len)
            .map_err(|e| crate::error::Error::Gpu(format!("{e}")))?;
        let depths = dev
            .read_buffer_u32(&bufs.depth, aln_len)
            .map_err(|e| crate::error::Error::Gpu(format!("{e}")))?;
        let alt_frequencies = dev
            .read_buffer_f64(&bufs.alt_freq, aln_len)
            .map_err(|e| crate::error::Error::Gpu(format!("{e}")))?;

        Ok(SnpGpuResult {
            is_variant,
            ref_alleles,
            depths,
            alt_frequencies,
        })
    }
}
