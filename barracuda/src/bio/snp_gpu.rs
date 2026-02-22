// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated position-parallel SNP calling via `ToadStool`.
//!
//! Delegates to `barracuda::ops::bio::snp::SnpCallingF64` — the absorbed
//! shader from wetSpring handoff v6. wetSpring provides the high-level
//! API that encodes aligned sequences to GPU buffers.
//!
//! # GPU Strategy
//!
//! Alignment positions are independent — one thread per column.
//! For L positions × N sequences, dispatch L threads. Each thread
//! reads N values (column-wise). No transcendentals needed.

use barracuda::device::WgpuDevice;
use barracuda::ops::bio::snp::SnpCallingF64;
use std::sync::Arc;
use wgpu::util::DeviceExt;

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
    inner: SnpCallingF64,
}

impl SnpGpu {
    /// Create a new SNP GPU instance.
    ///
    /// # Errors
    ///
    /// Returns an error if `ToadStool` shader compilation fails.
    pub fn new(device: &Arc<WgpuDevice>) -> crate::error::Result<Self> {
        let inner = SnpCallingF64::new(Arc::clone(device))
            .map_err(|e| crate::error::Error::Gpu(format!("SnpCallingF64: {e}")))?;
        Ok(Self {
            device: Arc::clone(device),
            inner,
        })
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

        let d = self.device.device();

        let sequences_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SNP sequences"),
            contents: bytemuck::cast_slice(&flat_seqs),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let variant_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SNP is_variant"),
            contents: bytemuck::cast_slice(&vec![0u32; aln_len]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let ref_allele_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SNP ref_allele"),
            contents: bytemuck::cast_slice(&vec![0u32; aln_len]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let depth_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SNP depth"),
            contents: bytemuck::cast_slice(&vec![0u32; aln_len]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let alt_freq_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SNP alt_freq"),
            contents: bytemuck::cast_slice(&vec![0.0_f64; aln_len]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        self.inner
            .dispatch(
                aln_len as u32,
                n_sequences as u32,
                2, // min_depth
                &sequences_buf,
                &variant_buf,
                &ref_allele_buf,
                &depth_buf,
                &alt_freq_buf,
            )
            .map_err(|e| crate::error::Error::Gpu(format!("{e}")))?;

        d.poll(wgpu::Maintain::Wait);

        let is_variant = self
            .device
            .read_buffer_u32(&variant_buf, aln_len)
            .map_err(|e| crate::error::Error::Gpu(format!("{e}")))?;
        let ref_alleles = self
            .device
            .read_buffer_u32(&ref_allele_buf, aln_len)
            .map_err(|e| crate::error::Error::Gpu(format!("{e}")))?;
        let depths = self
            .device
            .read_buffer_u32(&depth_buf, aln_len)
            .map_err(|e| crate::error::Error::Gpu(format!("{e}")))?;
        let alt_frequencies = self
            .device
            .read_buffer_f64(&alt_freq_buf, aln_len)
            .map_err(|e| crate::error::Error::Gpu(format!("{e}")))?;

        Ok(SnpGpuResult {
            is_variant,
            ref_alleles,
            depths,
            alt_frequencies,
        })
    }
}
