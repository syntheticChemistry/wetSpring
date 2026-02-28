// SPDX-License-Identifier: AGPL-3.0-or-later
//! Int8-quantized readout weights for NPU deployment.
//!
//! Exports trained ESN readout weights as int8 for Akida AKD1000 FC layer inference.

/// Int8-quantized readout weights for NPU deployment.
#[derive(Debug, Clone)]
pub struct NpuReadoutWeights {
    /// Quantized readout weights: `output_size × reservoir_size`, row-major.
    pub weights_i8: Vec<i8>,
    /// Quantization scale: `real = quantized * scale + zero_point`.
    pub scale: f64,
    /// Quantization zero point.
    pub zero_point: f64,
    /// Number of output classes.
    pub output_size: usize,
    /// Reservoir dimensionality.
    pub reservoir_size: usize,
}

impl NpuReadoutWeights {
    /// Run int8 inference on a reservoir state vector.
    /// Returns f64 outputs (dequantized from int8 accumulation).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn infer(&self, state: &[f64]) -> Vec<f64> {
        let n_res = self.reservoir_size;
        let n_out = self.output_size;
        let mut output = vec![0.0_f64; n_out];

        // Quantize state to int8
        let s_min = state.iter().copied().fold(f64::INFINITY, f64::min);
        let s_max = state.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let s_range = s_max - s_min;
        let s_scale = if s_range > 0.0 { s_range / 255.0 } else { 1.0 };

        let state_i8: Vec<i8> = state
            .iter()
            .map(|&v| {
                let q = ((v - s_min) / s_scale).round() as i64 - 128;
                q.clamp(-128, 127) as i8
            })
            .collect();

        // Int8 matmul: W_out_i8 · state_i8 → i32 accumulator → dequantize
        for o in 0..n_out {
            let mut acc = 0_i64;
            for r in 0..n_res.min(state_i8.len()) {
                acc += i64::from(self.weights_i8[o * n_res + r]) * i64::from(state_i8[r]);
            }
            // Dequantize: real = (acc * w_scale * s_scale) + corrections
            output[o] = acc as f64 * self.scale * s_scale;
        }

        output
    }

    /// Classify: return argmax of int8 inference output.
    #[must_use]
    pub fn classify(&self, state: &[f64]) -> usize {
        let output = self.infer(state);
        output
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |(i, _)| i)
    }
}
