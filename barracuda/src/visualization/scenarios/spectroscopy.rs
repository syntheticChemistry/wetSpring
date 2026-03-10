// SPDX-License-Identifier: AGPL-3.0-or-later
//! JCAMP-DX spectroscopy scenario: spectra, peak tables, multi-block.
//!
//! Parses JCAMP-DX files via [`crate::io::jcamp`] and builds scenarios
//! with spectrum channels, peak bar charts, and metadata.

use std::path::Path;

use crate::io::jcamp;
use crate::visualization::types::{EcologyScenario, ScenarioEdge, ScientificRange};

use super::{bar, node, scaffold, timeseries};

/// Build a spectroscopy scenario from a JCAMP-DX file.
///
/// Parses all blocks from the file and creates:
/// - A `TimeSeries` for each XY data block (spectrum/chromatogram)
/// - A `Bar` chart for each peak table (≤ 50 points)
///
/// # Errors
///
/// Returns `Err` if the file cannot be parsed.
pub fn spectroscopy_scenario(
    path: &Path,
) -> crate::error::Result<(EcologyScenario, Vec<ScenarioEdge>)> {
    let blocks = jcamp::parse_jcamp(path)?;

    let title = blocks
        .first()
        .map_or("JCAMP-DX Spectrum", |b| {
            if b.title.is_empty() { "JCAMP-DX Spectrum" } else { &b.title }
        });

    let mut s = scaffold(
        &format!("{title} — Spectroscopy"),
        &format!("{} data blocks from JCAMP-DX", blocks.len()),
    );
    s.domain = "measurement".into();

    let mut spec_node = node(
        "spectroscopy",
        "JCAMP-DX Data",
        "data",
        &["spectroscopy"],
    );

    for (i, block) in blocks.iter().enumerate() {
        let block_label = if block.title.is_empty() {
            format!("Block {}", i + 1)
        } else {
            block.title.clone()
        };

        if !block.x.is_empty() && !block.y.is_empty() {
            let x_unit = if block.x_units.is_empty() { "x" } else { &block.x_units };
            let y_unit = if block.y_units.is_empty() { "y" } else { &block.y_units };

            spec_node.data_channels.push(timeseries(
                &format!("spectrum_{i}"),
                &block_label,
                x_unit,
                y_unit,
                y_unit,
                &block.x,
                &block.y,
            ));
        }

        // Compact peak tables get a bar chart
        if block.x.len() <= 50 && !block.x.is_empty() {
            let labels: Vec<String> = block
                .x
                .iter()
                .map(|x| format!("{x:.2}"))
                .collect();
            let y_unit = if block.y_units.is_empty() { "intensity" } else { &block.y_units };
            spec_node.data_channels.push(bar(
                &format!("peaks_{i}"),
                &format!("{block_label} Peaks"),
                &labels,
                &block.y,
                y_unit,
            ));
        }
    }

    spec_node.scientific_ranges.push(ScientificRange {
        label: "Signal present".into(),
        min: 0.01,
        max: f64::MAX,
        status: "normal".into(),
    });

    s.nodes.push(spec_node);
    Ok((s, vec![]))
}

/// Build a spectroscopy scenario from in-memory data (no file I/O).
#[must_use]
pub fn spectroscopy_scenario_from_data(
    title: &str,
    x_values: &[f64],
    y_values: &[f64],
    x_unit: &str,
    y_unit: &str,
) -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        &format!("{title} — Spectrum"),
        &format!("{} data points", x_values.len()),
    );
    s.domain = "measurement".into();

    let mut spec_node = node("spectrum", "Spectrum Data", "data", &["spectroscopy"]);
    spec_node.data_channels.push(timeseries(
        "spectrum",
        title,
        x_unit,
        y_unit,
        y_unit,
        x_values,
        y_values,
    ));
    s.nodes.push(spec_node);
    (s, vec![])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_data_basic() {
        let (scenario, _) =
            spectroscopy_scenario_from_data("Test IR", &[4000.0, 3500.0, 3000.0], &[0.5, 0.8, 0.3], "cm⁻¹", "T");
        assert_eq!(scenario.nodes.len(), 1);
        assert_eq!(scenario.domain, "measurement");
    }

    #[test]
    fn from_data_empty() {
        let (scenario, _) = spectroscopy_scenario_from_data("Empty", &[], &[], "x", "y");
        assert_eq!(scenario.nodes.len(), 1);
    }
}
