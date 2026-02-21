// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated batch pairwise dN/dS (Nei-Gojobori 1986).
//!
//! One thread per coding sequence pair. Each thread walks codons,
//! classifies synonymous/nonsynonymous sites and differences using
//! a GPU-resident genetic code lookup table, then applies
//! Jukes-Cantor correction.
//!
//! Uses a local WGSL shader (`dnds_batch_f64.wgsl`) — a ToadStool
//! absorption candidate following Write → Absorb → Lean.
//!
//! # GPU Strategy
//!
//! dN/dS is codon-sequential but pair-parallel. Each thread runs the
//! full Nei-Gojobori analysis for one pair. Requires `log()` for
//! Jukes-Cantor — uses polyfill on NVVM drivers.

use barracuda::device::WgpuDevice;
use barracuda::shaders::precision::ShaderTemplate;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

const DNDS_WGSL: &str = include_str!("../shaders/dnds_batch_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct DnDsGpuParams {
    n_pairs: u32,
    n_codons: u32,
}

/// GPU dN/dS result.
pub struct DnDsGpuResult {
    /// dN per pair.
    pub dn: Vec<f64>,
    /// dS per pair.
    pub ds: Vec<f64>,
    /// omega (dN/dS) per pair. -1 if dS ≈ 0.
    pub omega: Vec<f64>,
}

/// Standard genetic code lookup table.
///
/// Index = base0*16 + base1*4 + base2 (A=0, C=1, G=2, T=3).
/// Values: amino acid IDs 0-19, stop=20.
///
/// Amino acid encoding:
/// A=0, C=1, D=2, E=3, F=4, G=5, H=6, I=7, K=8, L=9, M=10,
/// N=11, P=12, Q=13, R=14, S=15, T=16, V=17, W=18, Y=19, *=20
#[rustfmt::skip]
const GENETIC_CODE_TABLE: [u32; 64] = [
    8, 11,  8, 11, 16, 16, 16, 16, 14, 15, 14, 15,  7,  7, 10,  7,  // AA*, AC*, AG*, AT*
   13,  6, 13,  6, 12, 12, 12, 12, 14, 14, 14, 14,  9,  9,  9,  9,  // CA*, CC*, CG*, CT*
    3,  2,  3,  2,  0,  0,  0,  0,  5,  5,  5,  5, 17, 17, 17, 17,  // GA*, GC*, GG*, GT*
   20, 19, 20, 19, 15, 15, 15, 15, 20,  1, 18,  1,  9,  4,  9,  4,  // TA*, TC*, TG*, TT*
];

pub struct DnDsGpu {
    device: Arc<WgpuDevice>,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

impl DnDsGpu {
    pub fn new(device: &Arc<WgpuDevice>) -> Self {
        let patched = ShaderTemplate::for_driver_auto(DNDS_WGSL, true);
        let module = device.compile_shader(&patched, Some("DnDsBatchF64"));
        let pipeline = device
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("DnDsBatchF64"),
                layout: None,
                module: &module,
                entry_point: "main",
                cache: None,
                compilation_options: Default::default(),
            });
        let bgl = pipeline.get_bind_group_layout(0);
        Self {
            device: Arc::clone(device),
            pipeline,
            bgl,
        }
    }

    /// Compute dN/dS for a batch of coding sequence pairs on the GPU.
    ///
    /// Sequences as byte slices (ASCII A/C/G/T). Length must be divisible by 3.
    /// Gap codons (containing `-` or `.`) are skipped.
    #[allow(clippy::cast_possible_truncation)]
    pub fn batch_dnds(&self, pairs: &[(&[u8], &[u8])]) -> crate::error::Result<DnDsGpuResult> {
        let n_pairs = pairs.len();
        if n_pairs == 0 {
            return Ok(DnDsGpuResult {
                dn: Vec::new(),
                ds: Vec::new(),
                omega: Vec::new(),
            });
        }

        let n_codons = pairs
            .iter()
            .map(|(a, b)| a.len().max(b.len()) / 3)
            .max()
            .unwrap_or(0);

        let encode_base = |b: u8| -> u32 {
            match b.to_ascii_uppercase() {
                b'A' => 0,
                b'C' => 1,
                b'G' => 2,
                b'T' => 3,
                _ => 4, // gap / N / unknown
            }
        };

        let flat_size = n_pairs * n_codons * 3;
        let mut seq_a_flat: Vec<u32> = vec![4; flat_size];
        let mut seq_b_flat: Vec<u32> = vec![4; flat_size];

        for (i, (a, b)) in pairs.iter().enumerate() {
            let base = i * n_codons * 3;
            for (j, &byte) in a.iter().enumerate() {
                if j < n_codons * 3 {
                    seq_a_flat[base + j] = encode_base(byte);
                }
            }
            for (j, &byte) in b.iter().enumerate() {
                if j < n_codons * 3 {
                    seq_b_flat[base + j] = encode_base(byte);
                }
            }
        }

        let dev = &self.device;
        let d = dev.device();

        let params = DnDsGpuParams {
            n_pairs: n_pairs as u32,
            n_codons: n_codons as u32,
        };

        let params_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dN/dS params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let seq_a_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dN/dS seq_a"),
            contents: bytemuck::cast_slice(&seq_a_flat),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let seq_b_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dN/dS seq_b"),
            contents: bytemuck::cast_slice(&seq_b_flat),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let gc_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dN/dS genetic code"),
            contents: bytemuck::cast_slice(&GENETIC_CODE_TABLE),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let dn_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dN/dS dn"),
            contents: bytemuck::cast_slice(&vec![0.0_f64; n_pairs]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let ds_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dN/dS ds"),
            contents: bytemuck::cast_slice(&vec![0.0_f64; n_pairs]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let omega_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dN/dS omega"),
            contents: bytemuck::cast_slice(&vec![0.0_f64; n_pairs]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: seq_a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: seq_b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: gc_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: dn_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: ds_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: omega_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = d.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dN/dS batch"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((n_pairs as u32).div_ceil(64), 1, 1);
        }

        dev.queue().submit(Some(encoder.finish()));
        d.poll(wgpu::Maintain::Wait);

        let dn = dev
            .read_buffer_f64(&dn_buf, n_pairs)
            .map_err(|e| crate::error::Error::Gpu(format!("{e}")))?;
        let ds = dev
            .read_buffer_f64(&ds_buf, n_pairs)
            .map_err(|e| crate::error::Error::Gpu(format!("{e}")))?;
        let omega = dev
            .read_buffer_f64(&omega_buf, n_pairs)
            .map_err(|e| crate::error::Error::Gpu(format!("{e}")))?;

        Ok(DnDsGpuResult { dn, ds, omega })
    }
}
