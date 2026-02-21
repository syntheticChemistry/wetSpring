// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated pangenome gene classification.
//!
//! One thread per gene cluster. Each thread reads its presence row
//! and classifies: core (all genomes), accessory (2+ but not all),
//! unique (exactly 1).
//!
//! Uses a local WGSL shader (`pangenome_classify.wgsl`) — a ToadStool
//! absorption candidate following Write → Absorb → Lean.
//!
//! # GPU Strategy
//!
//! Gene classification is row-independent → one thread per gene.
//! The presence matrix is uploaded as flat `u32` (0/1). Pure integer
//! arithmetic, no transcendentals.

use barracuda::device::WgpuDevice;
use barracuda::shaders::precision::ShaderTemplate;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

const PAN_WGSL: &str = include_str!("../shaders/pangenome_classify.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PanGpuParams {
    n_genes: u32,
    n_genomes: u32,
}

/// GPU pangenome classification result.
pub struct PangenomeGpuResult {
    /// Classification per gene: 0=absent, 1=unique, 2=accessory, 3=core.
    pub classifications: Vec<u32>,
    /// Genome presence count per gene.
    pub genome_counts: Vec<u32>,
}

impl PangenomeGpuResult {
    /// Count core genes (present in all genomes).
    #[must_use]
    pub fn core_count(&self) -> usize {
        self.classifications.iter().filter(|&&c| c == 3).count()
    }

    /// Count accessory genes (present in 2+ but not all genomes).
    #[must_use]
    pub fn accessory_count(&self) -> usize {
        self.classifications.iter().filter(|&&c| c == 2).count()
    }

    /// Count unique genes (present in exactly 1 genome).
    #[must_use]
    pub fn unique_count(&self) -> usize {
        self.classifications.iter().filter(|&&c| c == 1).count()
    }
}

pub struct PangenomeGpu {
    device: Arc<WgpuDevice>,
}

impl PangenomeGpu {
    pub fn new(device: &Arc<WgpuDevice>) -> Self {
        Self {
            device: Arc::clone(device),
        }
    }

    /// Classify genes from a flat presence matrix on the GPU.
    ///
    /// `presence_flat` is row-major `[n_genes × n_genomes]`, values 0 or 1.
    #[allow(clippy::cast_possible_truncation)]
    pub fn classify(
        &self,
        presence_flat: &[u8],
        n_genes: usize,
        n_genomes: usize,
    ) -> crate::error::Result<PangenomeGpuResult> {
        if n_genes == 0 {
            return Ok(PangenomeGpuResult {
                classifications: Vec::new(),
                genome_counts: Vec::new(),
            });
        }

        let presence_u32: Vec<u32> = presence_flat.iter().map(|&b| u32::from(b)).collect();

        let dev = &self.device;
        let d = dev.device();

        let params = PanGpuParams {
            n_genes: n_genes as u32,
            n_genomes: n_genomes as u32,
        };

        let params_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Pan params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let presence_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Pan presence"),
            contents: bytemuck::cast_slice(&presence_u32),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let class_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Pan class"),
            contents: bytemuck::cast_slice(&vec![0u32; n_genes]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let count_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Pan count"),
            contents: bytemuck::cast_slice(&vec![0u32; n_genes]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let patched = ShaderTemplate::for_driver_auto(PAN_WGSL, false);
        let module = dev.compile_shader(&patched, Some("PangenomeClassify"));
        let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("PangenomeClassify"),
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
                    resource: presence_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: class_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: count_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = d.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Pangenome classify"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((n_genes as u32).div_ceil(256), 1, 1);
        }

        dev.queue().submit(Some(encoder.finish()));
        d.poll(wgpu::Maintain::Wait);

        let classifications = dev
            .read_buffer_u32(&class_buf, n_genes)
            .map_err(|e| crate::error::Error::Gpu(format!("{e}")))?;
        let genome_counts = dev
            .read_buffer_u32(&count_buf, n_genes)
            .map_err(|e| crate::error::Error::Gpu(format!("{e}")))?;

        Ok(PangenomeGpuResult {
            classifications,
            genome_counts,
        })
    }
}
