// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated pangenome gene classification via `ToadStool`.
//!
//! Delegates to `barracuda::ops::bio::pangenome::PangenomeClassifyGpu` —
//! the absorbed shader from wetSpring handoff v6. wetSpring provides the
//! high-level API that converts presence matrices to GPU buffers.
//!
//! # GPU Strategy
//!
//! Gene classification is row-independent — one thread per gene.
//! The presence matrix is uploaded as flat `u32` (0/1). Pure integer
//! arithmetic, no transcendentals.

use barracuda::device::WgpuDevice;
use barracuda::ops::bio::pangenome::PangenomeClassifyGpu as ToadStoolPangenome;
use std::sync::Arc;
use wgpu::util::DeviceExt;

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

/// GPU-accelerated pangenome gene classification.
pub struct PangenomeGpu {
    device: Arc<WgpuDevice>,
    inner: ToadStoolPangenome,
}

impl PangenomeGpu {
    /// Create a new pangenome GPU instance.
    ///
    /// # Errors
    ///
    /// Returns an error if `ToadStool` shader compilation fails.
    pub fn new(device: &Arc<WgpuDevice>) -> crate::error::Result<Self> {
        let inner = ToadStoolPangenome::new(Arc::clone(device))
            .map_err(|e| crate::error::Error::Gpu(format!("PangenomeClassifyGpu: {e}")))?;
        Ok(Self {
            device: Arc::clone(device),
            inner,
        })
    }

    /// Classify genes from a flat presence matrix on the GPU.
    ///
    /// `presence_flat` is row-major `[n_genes × n_genomes]`, values 0 or 1.
    ///
    /// # Errors
    ///
    /// Returns `Err` if GPU buffer creation or readback fails.
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
        let d = self.device.device();

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

        self.inner
            .dispatch(
                n_genes as u32,
                n_genomes as u32,
                &presence_buf,
                &class_buf,
                &count_buf,
            )
            .map_err(|e| crate::error::Error::Gpu(format!("{e}")))?;

        d.poll(wgpu::Maintain::Wait);

        let classifications = self
            .device
            .read_buffer_u32(&class_buf, n_genes)
            .map_err(|e| crate::error::Error::Gpu(format!("{e}")))?;
        let genome_counts = self
            .device
            .read_buffer_u32(&count_buf, n_genes)
            .map_err(|e| crate::error::Error::Gpu(format!("{e}")))?;

        Ok(PangenomeGpuResult {
            classifications,
            genome_counts,
        })
    }
}
