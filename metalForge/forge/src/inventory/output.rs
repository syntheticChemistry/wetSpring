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

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::substrate::{Capability, Identity, Properties};

    fn cpu_substrate(name: &str) -> Substrate {
        Substrate {
            kind: SubstrateKind::Cpu,
            identity: Identity::named(name),
            properties: Properties::default(),
            capabilities: vec![Capability::F64Compute, Capability::F32Compute],
            origin: SubstrateOrigin::Local,
        }
    }

    fn gpu_substrate(name: &str) -> Substrate {
        let mut props = Properties::default();
        props.has_f64 = true;
        props.core_count = Some(2048);
        props.thread_count = Some(2048);
        Substrate {
            kind: SubstrateKind::Gpu,
            identity: Identity {
                name: name.into(),
                driver: Some("nvidia-580".into()),
                backend: Some("vulkan".into()),
                adapter_index: Some(0),
                device_node: Some("/dev/dri/card0".into()),
                pci_id: Some("10de:2504".into()),
            },
            properties: props,
            capabilities: vec![
                Capability::F64Compute,
                Capability::F32Compute,
                Capability::ShaderDispatch,
            ],
            origin: SubstrateOrigin::Local,
        }
    }

    #[test]
    fn write_inventory_empty() {
        let mut buf = Vec::new();
        write_inventory(&[], &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("wetSpring Forge"));
        assert!(output.contains("Total: 0 local, 0 mesh, 0 GPU(s), 0 NPU(s)"));
    }

    #[test]
    fn write_inventory_cpu_only() {
        let substrates = vec![cpu_substrate("test-cpu")];
        let mut buf = Vec::new();
        write_inventory(&substrates, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("test-cpu"));
        assert!(output.contains("Total: 1 local, 0 mesh, 0 GPU(s)"));
    }

    #[test]
    fn write_inventory_gpu_with_details() {
        let substrates = vec![gpu_substrate("test-gpu")];
        let mut buf = Vec::new();
        write_inventory(&substrates, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("test-gpu"));
        assert!(output.contains("vulkan"));
        assert!(output.contains("adapter: 0"));
        assert!(output.contains("/dev/dri/card0"));
        assert!(output.contains("SHADER_F64: YES"));
        assert!(output.contains("2048"));
        assert!(output.contains("1 GPU(s)"));
    }

    #[test]
    fn write_inventory_mesh_substrate() {
        let substrates = vec![Substrate {
            kind: SubstrateKind::Gpu,
            identity: Identity::named("remote-gpu"),
            properties: Properties::default(),
            capabilities: vec![Capability::F32Compute],
            origin: SubstrateOrigin::Mesh {
                gate_name: "lab-gate".into(),
            },
        }];
        let mut buf = Vec::new();
        write_inventory(&substrates, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("[mesh: lab-gate]"));
        assert!(output.contains("0 local, 1 mesh"));
    }

    #[test]
    fn write_inventory_mixed() {
        let substrates = vec![
            cpu_substrate("cpu-0"),
            gpu_substrate("gpu-0"),
        ];
        let mut buf = Vec::new();
        write_inventory(&substrates, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("cpu-0"));
        assert!(output.contains("gpu-0"));
        assert!(output.contains("2 local, 0 mesh, 1 GPU(s)"));
    }

    #[test]
    fn print_inventory_does_not_panic() {
        let substrates = vec![cpu_substrate("stdout-cpu")];
        print_inventory(&substrates);
    }
}
