// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated position-parallel SNP calling.
//!
//! One thread per alignment column. Each thread counts allele
//! frequencies across all sequences at its position and reports
//! whether the site is polymorphic.
//!
//! Uses a local WGSL shader (`snp_calling_f64.wgsl`) — a ToadStool
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
    /// 1 if position is a variant, 0 otherwise. Length = alignment_length.
    pub is_variant: Vec<u32>,
    /// Reference allele at each position (0=A,1=C,2=G,3=T).
    pub ref_alleles: Vec<u32>,
    /// Depth at each position.
    pub depths: Vec<u32>,
    /// Alt allele frequency at each position (0.0 for non-variants).
    pub alt_frequencies: Vec<f64>,
}

pub struct SnpGpu {
    device: Arc<WgpuDevice>,
}

impl SnpGpu {
    pub fn new(device: &Arc<WgpuDevice>) -> Self {
        Self {
            device: Arc::clone(device),
        }
    }

    /// Call SNPs from aligned sequences on the GPU.
    ///
    /// Sequences as byte slices (ASCII A/C/G/T, gaps as `-`/`.`/`N`).
    /// All sequences must have the same length.
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

        let params_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SNP params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let seqs_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SNP sequences"),
            contents: bytemuck::cast_slice(&flat_seqs),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let variant_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SNP is_variant"),
            contents: bytemuck::cast_slice(&vec![0u32; aln_len]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let ref_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SNP ref_allele"),
            contents: bytemuck::cast_slice(&vec![0u32; aln_len]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let depth_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SNP depth"),
            contents: bytemuck::cast_slice(&vec![0u32; aln_len]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let freq_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SNP alt_freq"),
            contents: bytemuck::cast_slice(&vec![0.0_f64; aln_len]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let patched = ShaderTemplate::for_driver_auto(SNP_WGSL, false);
        let module = dev.compile_shader(&patched, Some("SnpCallingF64"));
        let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SnpCallingF64"),
            layout: None,
            module: &module,
            entry_point: "main",
            cache: None,
            compilation_options: Default::default(),
        });

        let bgl = pipeline.get_bind_group_layout(0);
        let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: seqs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: variant_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: ref_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: depth_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: freq_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = d.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("SNP calling"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((aln_len as u32).div_ceil(256), 1, 1);
        }

        dev.queue().submit(Some(encoder.finish()));
        d.poll(wgpu::Maintain::Wait);

        let is_variant = dev
            .read_buffer_u32(&variant_buf, aln_len)
            .map_err(|e| crate::error::Error::Gpu(format!("{e}")))?;
        let ref_alleles = dev
            .read_buffer_u32(&ref_buf, aln_len)
            .map_err(|e| crate::error::Error::Gpu(format!("{e}")))?;
        let depths = dev
            .read_buffer_u32(&depth_buf, aln_len)
            .map_err(|e| crate::error::Error::Gpu(format!("{e}")))?;
        let alt_frequencies = dev
            .read_buffer_f64(&freq_buf, aln_len)
            .map_err(|e| crate::error::Error::Gpu(format!("{e}")))?;

        Ok(SnpGpuResult {
            is_variant,
            ref_alleles,
            depths,
            alt_frequencies,
        })
    }
}
