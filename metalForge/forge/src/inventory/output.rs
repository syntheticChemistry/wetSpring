// SPDX-License-Identifier: AGPL-3.0-or-later

//! Inventory output — human-readable formatting.

use std::io::{self, Write};

use crate::substrate::{Substrate, SubstrateKind, SubstrateOrigin};

/// Write a human-readable inventory to the given writer.
///
/// # Errors
///
/// Returns an I/O error if writing fails.
///
/// # Examples
///
/// ```
/// use wetspring_forge::inventory::write_inventory;
/// use wetspring_forge::substrate::{
///     Capability, Identity, Properties, Substrate, SubstrateKind, SubstrateOrigin,
/// };
///
/// let substrates = vec![Substrate {
///     kind: SubstrateKind::Cpu,
///     identity: Identity::named("test-cpu"),
///     properties: Properties::default(),
///     capabilities: vec![Capability::F64Compute, Capability::F32Compute],
///     origin: SubstrateOrigin::Local,
/// }];
/// let mut buf = Vec::new();
/// write_inventory(&substrates, &mut buf).unwrap();
/// let output = String::from_utf8(buf).unwrap();
/// assert!(output.contains("wetSpring Forge"));
/// assert!(output.contains("test-cpu"));
/// assert!(output.contains("f64"));
/// ```
pub fn write_inventory(substrates: &[Substrate], w: &mut impl Write) -> io::Result<()> {
    writeln!(w, "┌──────────────────────────────────────────────┐")?;
    writeln!(w, "│  wetSpring Forge — Hardware Inventory         │")?;
    writeln!(w, "├──────────────────────────────────────────────┤")?;

    for (i, s) in substrates.iter().enumerate() {
        let origin_tag = match &s.origin {
            SubstrateOrigin::Local => "",
            SubstrateOrigin::Mesh { gate_name } => gate_name.as_str(),
        };
        if origin_tag.is_empty() {
            writeln!(w, "│ {i}: {s}")?;
        } else {
            writeln!(w, "│ {i}: {s} [mesh: {origin_tag}]")?;
        }

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

    let local_count = substrates
        .iter()
        .filter(|s| s.origin == SubstrateOrigin::Local)
        .count();
    let mesh_count = substrates.len() - local_count;
    let gpu_count = substrates
        .iter()
        .filter(|s| s.kind == SubstrateKind::Gpu)
        .count();
    let npu_count = substrates
        .iter()
        .filter(|s| s.kind == SubstrateKind::Npu)
        .count();
    writeln!(
        w,
        "│  Total: {local_count} local, {mesh_count} mesh, {gpu_count} GPU(s), {npu_count} NPU(s)"
    )?;
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
