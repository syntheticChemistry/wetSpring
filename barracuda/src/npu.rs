// SPDX-License-Identifier: AGPL-3.0-or-later
//! NPU integration via akida-driver (independent of `barraCuda`).
//!
//! Wraps `akida_driver` for wetSpring's neuromorphic compute pipeline:
//! runtime device discovery, capability-based int8 inference, and
//! substrate routing (NPU vs CPU fallback).
//!
//! # Architecture
//!
//! - **Zero Mocks**: Real hardware only; tests skip when no device present
//! - **Capability-Based**: Device features discovered at runtime
//! - **Primal Self-Knowledge**: wetSpring discovers NPU, never hardcodes
//! - **Fast AND Safe**: Pure Rust driver (sourced from toadStool neuromorphic crates)

#![expect(
    clippy::cast_possible_truncation,
    reason = "NPU interop: u128→u64 timing (valid for durations < 580 years), \
              f64→i8 quantization where clamping guarantees [0,127] range"
)]
#![expect(
    clippy::cast_possible_wrap,
    reason = "NPU DMA: i8↔u8 bit reinterpretation for int8 data path"
)]
#![expect(
    clippy::cast_precision_loss,
    reason = "NPU metrics: usize→f64 for batch counts and timing ratios; \
              values well within f64 53-bit significand"
)]

pub use akida_driver::{
    AkidaDevice, BackendSelection, BackendType, Capabilities, ChipVersion, DeviceManager,
    InferenceConfig, InferenceExecutor, InferenceResult, LoadConfig, ModelLoader, ModelProgram,
    NpuBackend, NpuConfig,
};

use crate::error::Error;

/// NPU handle wrapping an opened `AkidaDevice` with its capabilities.
///
/// Created via [`discover_npu`] — never constructed manually.
pub struct NpuHandle {
    device: AkidaDevice,
    caps: Capabilities,
}

impl NpuHandle {
    /// Device capabilities (discovered at runtime).
    #[must_use]
    pub const fn capabilities(&self) -> &Capabilities {
        &self.caps
    }

    /// Chip version (`AKD1000`, `AKD1500`, etc.).
    #[must_use]
    pub const fn chip_version(&self) -> ChipVersion {
        self.caps.chip_version
    }

    /// Number of neural processing units.
    #[must_use]
    pub const fn npu_count(&self) -> u32 {
        self.caps.npu_count
    }

    /// On-chip SRAM in megabytes.
    #[must_use]
    pub const fn memory_mb(&self) -> u32 {
        self.caps.memory_mb
    }

    /// `PCIe` bandwidth in GB/s.
    #[must_use]
    pub const fn bandwidth_gbps(&self) -> f32 {
        self.caps.pcie.bandwidth_gbps
    }

    /// Load a model program onto the NPU.
    ///
    /// # Errors
    ///
    /// Returns error if the model is too large for device SRAM or the
    /// DMA transfer fails.
    pub fn load_model(&mut self, program: &ModelProgram) -> Result<(), Error> {
        let config = LoadConfig::from_capabilities(&self.caps, self.device.index());
        let loader = ModelLoader::new(config);
        loader
            .load(program, &mut self.device)
            .map_err(|e| Error::Npu(format!("model load failed: {e}")))?;
        Ok(())
    }

    /// Run int8 inference with the currently loaded model.
    ///
    /// # Errors
    ///
    /// Returns error if input size is wrong or inference times out.
    pub fn infer(
        &mut self,
        config: &InferenceConfig,
        input: &[u8],
    ) -> Result<InferenceResult, Error> {
        let executor = InferenceExecutor::new(config.clone());
        executor
            .infer(input, &mut self.device)
            .map_err(|e| Error::Npu(format!("inference failed: {e}")))
    }

    /// Write raw bytes to device SRAM (DMA).
    ///
    /// # Errors
    ///
    /// Returns error if the DMA transfer fails.
    pub fn write_raw(&mut self, data: &[u8]) -> Result<usize, Error> {
        self.device
            .write(data)
            .map_err(|e| Error::Npu(format!("write failed: {e}")))
    }

    /// Read raw bytes from device SRAM (DMA).
    ///
    /// # Errors
    ///
    /// Returns error if the DMA transfer fails.
    pub fn read_raw(&mut self, buffer: &mut [u8]) -> Result<usize, Error> {
        self.device
            .read(buffer)
            .map_err(|e| Error::Npu(format!("read failed: {e}")))
    }
}

/// Discover and open the first available Akida NPU.
///
/// Uses runtime discovery — no hardcoded device paths.
///
/// # Errors
///
/// Returns error if no Akida hardware is detected or the device
/// cannot be opened.
pub fn discover_npu() -> Result<NpuHandle, Error> {
    let manager =
        DeviceManager::discover().map_err(|e| Error::Npu(format!("discovery failed: {e}")))?;

    let info = manager
        .devices()
        .first()
        .ok_or_else(|| Error::Npu("no devices found after discovery".into()))?;

    let caps = info.capabilities().clone();

    let device = AkidaDevice::open(info).map_err(|e| Error::Npu(format!("open failed: {e}")))?;

    Ok(NpuHandle { device, caps })
}

/// Check whether an Akida NPU is present without opening it.
#[must_use]
pub fn npu_available() -> bool {
    DeviceManager::discover()
        .map(|m| m.device_count() > 0)
        .unwrap_or(false)
}

/// Quantize `f64` value to int8 for NPU inference.
///
/// Maps `[lo, hi]` → `[0, 127]` with clamping.
#[must_use]
pub fn quantize_i8(val: f64, lo: f64, hi: f64) -> i8 {
    let normalized = ((val - lo) / (hi - lo)).clamp(0.0, 1.0);
    (normalized * 127.0) as i8
}

/// Dequantize int8 back to `f64`.
///
/// Maps `[0, 127]` → `[lo, hi]`.
#[must_use]
pub fn dequantize_i8(val: i8, lo: f64, hi: f64) -> f64 {
    let normalized = f64::from(val) / 127.0;
    lo + normalized * (hi - lo)
}

/// Summary of discovered NPU hardware for display.
pub struct NpuSummary {
    /// Chip identifier (e.g. `Akd1000`).
    pub chip: String,
    /// `PCIe` address.
    pub pcie_address: String,
    /// Number of NPUs.
    pub npu_count: u32,
    /// SRAM in MB.
    pub memory_mb: u32,
    /// `PCIe` bandwidth in GB/s.
    pub bandwidth_gbps: f32,
}

/// Discover NPU and return a summary without opening the device.
///
/// # Errors
///
/// Returns error if no Akida hardware is detected.
pub fn npu_summary() -> Result<NpuSummary, Error> {
    let manager =
        DeviceManager::discover().map_err(|e| Error::Npu(format!("discovery failed: {e}")))?;

    let info = manager
        .devices()
        .first()
        .ok_or_else(|| Error::Npu("no devices found".into()))?;

    let caps = info.capabilities();

    Ok(NpuSummary {
        chip: format!("{:?}", caps.chip_version),
        pcie_address: info.pcie_address().to_string(),
        npu_count: caps.npu_count,
        memory_mb: caps.memory_mb,
        bandwidth_gbps: caps.pcie.bandwidth_gbps,
    })
}

// ═══════════════════════════════════════════════════════════════════
// ESN → NPU Bridge
// ═══════════════════════════════════════════════════════════════════

/// Result of a single NPU int8 inference via DMA.
#[derive(Debug, Clone)]
pub struct NpuInferResult {
    /// Raw int8 accumulator outputs (one per class).
    pub raw_i8: Vec<i8>,
    /// Argmax class index.
    pub class: usize,
    /// DMA write latency (nanoseconds).
    pub write_ns: u64,
    /// DMA read latency (nanoseconds).
    pub read_ns: u64,
}

/// Run a single int8 inference round-trip on the NPU via DMA.
///
/// Writes `input_i8` to device SRAM, reads back `n_outputs` bytes.
/// This exercises the real AKD1000 DMA path for classification.
///
/// # Errors
///
/// Returns error if DMA transfer fails.
pub fn npu_infer_i8(
    handle: &mut NpuHandle,
    input_i8: &[i8],
    n_outputs: usize,
) -> Result<NpuInferResult, Error> {
    let input_bytes: Vec<u8> = input_i8.iter().map(|&x| x as u8).collect();

    let t = std::time::Instant::now();
    handle.write_raw(&input_bytes)?;
    let write_ns = t.elapsed().as_nanos() as u64;

    let mut out_buf = vec![0u8; n_outputs];
    let t = std::time::Instant::now();
    handle.read_raw(&mut out_buf)?;
    let read_ns = t.elapsed().as_nanos() as u64;

    let raw_i8: Vec<i8> = out_buf.iter().map(|&b| b as i8).collect();
    let class = raw_i8
        .iter()
        .enumerate()
        .max_by_key(|&(_, v)| *v)
        .map_or(0, |(i, _)| i);

    Ok(NpuInferResult {
        raw_i8,
        class,
        write_ns,
        read_ns,
    })
}

/// Load ESN reservoir weights to NPU SRAM via DMA.
///
/// Writes `w_in` (`input_size` × `reservoir_size`) and `w_res`
/// (`reservoir_size` × `reservoir_size`) as contiguous f32 bytes.
/// The AKD1000 stores weights in on-chip SRAM for spiking execution.
///
/// # Errors
///
/// Returns error if the weights exceed SRAM capacity or DMA fails.
#[expect(clippy::cast_possible_truncation)]
pub fn load_reservoir_weights(
    handle: &mut NpuHandle,
    w_in_f64: &[f64],
    w_res_f64: &[f64],
) -> Result<ReservoirLoadResult, Error> {
    let w_in_f32: Vec<f32> = w_in_f64.iter().map(|&x| x as f32).collect();
    let w_res_f32: Vec<f32> = w_res_f64.iter().map(|&x| x as f32).collect();

    let w_in_bytes = bytemuck::cast_slice::<f32, u8>(&w_in_f32);
    let w_res_bytes = bytemuck::cast_slice::<f32, u8>(&w_res_f32);

    let total_bytes = w_in_bytes.len() + w_res_bytes.len();
    let sram_bytes = handle.memory_mb() as usize * 1024 * 1024;
    if total_bytes > sram_bytes {
        return Err(Error::Npu(format!(
            "reservoir weights ({total_bytes} bytes) exceed SRAM ({sram_bytes} bytes)"
        )));
    }

    let t = std::time::Instant::now();
    let written_in = handle.write_raw(w_in_bytes)?;
    let written_res = handle.write_raw(w_res_bytes)?;
    let load_us = t.elapsed().as_micros() as f64;

    let throughput_mbps = if load_us > 0.0 {
        (written_in + written_res) as f64 / load_us
    } else {
        0.0
    };

    Ok(ReservoirLoadResult {
        w_in_bytes: written_in,
        w_res_bytes: written_res,
        load_us,
        throughput_mbps,
    })
}

/// Metrics from loading reservoir weights to NPU.
#[derive(Debug, Clone)]
pub struct ReservoirLoadResult {
    /// Bytes written for `W_in`.
    pub w_in_bytes: usize,
    /// Bytes written for `W_res`.
    pub w_res_bytes: usize,
    /// Total load time in microseconds.
    pub load_us: f64,
    /// Load throughput in MB/s.
    pub throughput_mbps: f64,
}

/// Load ESN readout weights (`W_out`) to NPU SRAM via DMA.
///
/// Enables online readout switching — swap the classifier without
/// reloading the reservoir. This exploits `AKD1000` weight mutation.
///
/// # Errors
///
/// Returns error if DMA transfer fails.
#[expect(clippy::cast_possible_truncation)]
pub fn load_readout_weights(handle: &mut NpuHandle, w_out_i8: &[i8]) -> Result<u64, Error> {
    let bytes: Vec<u8> = w_out_i8.iter().map(|&x| x as u8).collect();
    let t = std::time::Instant::now();
    handle.write_raw(&bytes)?;
    Ok(t.elapsed().as_nanos() as u64)
}

/// Run a batch of int8 inferences and return aggregate metrics.
///
/// # Errors
///
/// Returns error if any DMA transfer fails.
pub fn npu_batch_infer(
    handle: &mut NpuHandle,
    inputs_i8: &[Vec<i8>],
    n_outputs: usize,
) -> Result<NpuBatchResult, Error> {
    let mut classes = Vec::with_capacity(inputs_i8.len());
    let mut total_write_ns = 0u64;
    let mut total_read_ns = 0u64;

    for input in inputs_i8 {
        let r = npu_infer_i8(handle, input, n_outputs)?;
        classes.push(r.class);
        total_write_ns += r.write_ns;
        total_read_ns += r.read_ns;
    }

    let n = inputs_i8.len() as f64;
    Ok(NpuBatchResult {
        classes,
        mean_write_ns: total_write_ns as f64 / n,
        mean_read_ns: total_read_ns as f64 / n,
        total_us: (total_write_ns + total_read_ns) as f64 / 1000.0,
        throughput_hz: if total_write_ns + total_read_ns > 0 {
            n * 1_000_000_000.0 / (total_write_ns + total_read_ns) as f64
        } else {
            0.0
        },
    })
}

/// Aggregate metrics from batch NPU inference.
#[derive(Debug, Clone)]
pub struct NpuBatchResult {
    /// Classified index for each input.
    pub classes: Vec<usize>,
    /// Mean DMA write latency per inference (ns).
    pub mean_write_ns: f64,
    /// Mean DMA read latency per inference (ns).
    pub mean_read_ns: f64,
    /// Total batch time (µs).
    pub total_us: f64,
    /// Throughput (inferences/second).
    pub throughput_hz: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_roundtrip() {
        let val = 7.5_f64;
        let q = quantize_i8(val, 5.0, 10.0);
        let deq = dequantize_i8(q, 5.0, 10.0);
        assert!((val - deq).abs() < 0.1, "roundtrip: {val} -> {q} -> {deq}");
    }

    #[test]
    fn test_quantize_clamp() {
        assert_eq!(quantize_i8(-1.0, 0.0, 10.0), 0);
        assert_eq!(quantize_i8(20.0, 0.0, 10.0), 127);
    }

    #[test]
    fn test_npu_available() {
        let avail = npu_available();
        println!("NPU available: {avail}");
    }

    #[test]
    fn test_discover_npu() {
        let handle = discover_npu();
        match handle {
            Ok(h) => {
                println!(
                    "NPU: {:?}, {} NPUs, {} MB SRAM, {:.1} GB/s",
                    h.chip_version(),
                    h.npu_count(),
                    h.memory_mb(),
                    h.bandwidth_gbps()
                );
            }
            Err(e) => {
                println!("No NPU: {e} (expected on CI)");
            }
        }
    }
}
