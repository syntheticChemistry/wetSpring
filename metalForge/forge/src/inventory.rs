// SPDX-License-Identifier: AGPL-3.0-or-later

//! Hardware inventory — assemble probed substrates.
//!
//! Runs GPU probing (via wgpu/barracuda), CPU probing (procfs), and
//! NPU probing (local device nodes). Returns every substrate discovered
//! on this machine. If it's not in the inventory, it doesn't exist to us.

use std::io::{self, Write};

use crate::probe;
use crate::substrate::{Substrate, SubstrateKind};

/// Discover all compute substrates on this machine.
///
/// Always returns at least one entry (CPU). GPUs come from wgpu adapter
/// enumeration. NPUs come from device node probing.
#[must_use]
pub fn discover() -> Vec<Substrate> {
    let mut substrates = Vec::new();
    substrates.push(probe::probe_cpu());
    substrates.extend(probe::probe_gpus());
    substrates.extend(probe::probe_npus());
    substrates
}

/// Write a human-readable inventory to the given writer.
///
/// # Errors
///
/// Returns an I/O error if writing fails.
pub fn write_inventory(substrates: &[Substrate], w: &mut impl Write) -> io::Result<()> {
    writeln!(w, "┌──────────────────────────────────────────────┐")?;
    writeln!(w, "│  wetSpring Forge — Hardware Inventory         │")?;
    writeln!(w, "├──────────────────────────────────────────────┤")?;

    for (i, s) in substrates.iter().enumerate() {
        writeln!(w, "│ {i}: {s}")?;

        if let Some(ref backend) = s.identity.backend {
            writeln!(w, "│    backend: {backend}")?;
        }
        if let Some(idx) = s.identity.adapter_index {
            writeln!(w, "│    adapter: {idx}")?;
        }
        if let Some(ref node) = s.identity.device_node {
            writeln!(w, "│    device:  {node}")?;
        }
        if let Some(cores) = s.properties.core_count {
            let threads = s.properties.thread_count.unwrap_or(cores);
            writeln!(w, "│    cores:   {cores} ({threads} threads)")?;
        }
        if s.properties.has_f64 {
            writeln!(w, "│    SHADER_F64: YES")?;
        }

        writeln!(w, "│    caps:    {}", s.capability_summary())?;
        writeln!(w, "│")?;
    }

    let gpu_count = substrates
        .iter()
        .filter(|s| s.kind == SubstrateKind::Gpu)
        .count();
    let npu_count = substrates
        .iter()
        .filter(|s| s.kind == SubstrateKind::Npu)
        .count();
    writeln!(w, "│  Total: 1 CPU, {gpu_count} GPU(s), {npu_count} NPU(s)")?;
    writeln!(w, "└──────────────────────────────────────────────┘")?;
    Ok(())
}

/// Print a human-readable inventory to stdout.
///
/// Convenience wrapper around [`write_inventory`] for CLI usage.
pub fn print_inventory(substrates: &[Substrate]) {
    let mut stdout = io::stdout().lock();
    let _ = write_inventory(substrates, &mut stdout);
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn discover_always_has_cpu() {
        let subs = discover();
        assert!(subs.iter().any(|s| s.kind == SubstrateKind::Cpu));
    }

    #[test]
    fn discover_finds_gpus_via_wgpu() {
        let subs = discover();
        let gpus: Vec<_> = subs
            .iter()
            .filter(|s| s.kind == SubstrateKind::Gpu)
            .collect();
        for gpu in &gpus {
            assert!(gpu.has(&crate::substrate::Capability::ShaderDispatch));
            assert!(gpu.identity.adapter_index.is_some());
        }
    }

    #[test]
    fn inventory_has_one_cpu() {
        let subs = discover();
        let cpu_count = subs.iter().filter(|s| s.kind == SubstrateKind::Cpu).count();
        assert_eq!(cpu_count, 1);
    }
}
