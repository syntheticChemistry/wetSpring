// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated batch pairwise dN/dS (Nei-Gojobori 1986) via `ToadStool`.
//!
//! Delegates to `barracuda::ops::bio::dnds::DnDsBatchF64` — the absorbed
//! shader from wetSpring handoff v6. wetSpring provides the high-level
//! API that encodes codon sequences and uploads the genetic code table.
//!
//! # GPU Strategy
//!
//! dN/dS is codon-sequential but pair-parallel. Each thread runs the
//! full Nei-Gojobori analysis for one pair. Requires `log()` for
//! Jukes-Cantor — `ToadStool` handles polyfill on NVVM drivers.

use barracuda::device::WgpuDevice;
use barracuda::ops::bio::dnds::DnDsBatchF64;
use std::sync::Arc;
use wgpu::util::DeviceExt;

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
#[rustfmt::skip]
const GENETIC_CODE_TABLE: [u32; 64] = [
    8, 11,  8, 11, 16, 16, 16, 16, 14, 15, 14, 15,  7,  7, 10,  7,
   13,  6, 13,  6, 12, 12, 12, 12, 14, 14, 14, 14,  9,  9,  9,  9,
    3,  2,  3,  2,  0,  0,  0,  0,  5,  5,  5,  5, 17, 17, 17, 17,
   20, 19, 20, 19, 15, 15, 15, 15, 20,  1, 18,  1,  9,  4,  9,  4,
];

/// GPU-accelerated batch dN/dS (Nei-Gojobori method).
pub struct DnDsGpu {
    device: Arc<WgpuDevice>,
    inner: DnDsBatchF64,
}

impl DnDsGpu {
    /// Create a new dN/dS GPU compute instance.
    ///
    /// # Errors
    ///
    /// Returns an error if `ToadStool` shader compilation fails.
    pub fn new(device: &Arc<WgpuDevice>) -> crate::error::Result<Self> {
        let inner = DnDsBatchF64::new(Arc::clone(device))
            .map_err(|e| crate::error::Error::Gpu(format!("DnDsBatchF64: {e}")))?;
        Ok(Self {
            device: Arc::clone(device),
            inner,
        })
    }

    /// Compute dN/dS for a batch of coding sequence pairs on the GPU.
    ///
    /// Sequences as byte slices (ASCII A/C/G/T). Length must be divisible by 3.
    /// Gap codons (containing `-` or `.`) are skipped.
    ///
    /// # Errors
    ///
    /// Returns `Err` if GPU dispatch or buffer readback fails.
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
                _ => 4,
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

        let d = self.device.device();

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

        self.inner
            .dispatch(
                n_pairs as u32,
                n_codons as u32,
                &seq_a_buf,
                &seq_b_buf,
                &gc_buf,
                &dn_buf,
                &ds_buf,
                &omega_buf,
            )
            .map_err(|e| crate::error::Error::Gpu(format!("{e}")))?;

        d.poll(wgpu::Maintain::Wait);

        let dn = self
            .device
            .read_buffer_f64(&dn_buf, n_pairs)
            .map_err(|e| crate::error::Error::Gpu(format!("{e}")))?;
        let ds = self
            .device
            .read_buffer_f64(&ds_buf, n_pairs)
            .map_err(|e| crate::error::Error::Gpu(format!("{e}")))?;
        let omega = self
            .device
            .read_buffer_f64(&omega_buf, n_pairs)
            .map_err(|e| crate::error::Error::Gpu(format!("{e}")))?;

        Ok(DnDsGpuResult { dn, ds, omega })
    }
}
