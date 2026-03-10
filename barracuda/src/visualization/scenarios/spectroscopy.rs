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

    let title = blocks.first().map_or("JCAMP-DX Spectrum", |b| {
        if b.title.is_empty() {
            "JCAMP-DX Spectrum"
        } else {
            &b.title
        }
    });

    let mut s = scaffold(
        &format!("{title} — Spectroscopy"),
        &format!("{} data blocks from JCAMP-DX", blocks.len()),
    );
    s.domain = "measurement".into();

    let mut spec_node = node("spectroscopy", "JCAMP-DX Data", "data", &["spectroscopy"]);

    for (i, block) in blocks.iter().enumerate() {
        let block_label = if block.title.is_empty() {
            format!("Block {}", i + 1)
        } else {
            block.title.clone()
        };

        if !block.x.is_empty() && !block.y.is_empty() {
            let x_unit = if block.x_units.is_empty() {
                "x"
            } else {
                &block.x_units
            };
            let y_unit = if block.y_units.is_empty() {
                "y"
            } else {
                &block.y_units
            };

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
            let labels: Vec<String> = block.x.iter().map(|x| format!("{x:.2}")).collect();
            let y_unit = if block.y_units.is_empty() {
                "intensity"
            } else {
                &block.y_units
            };
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
        "spectrum", title, x_unit, y_unit, y_unit, x_values, y_values,
    ));
    s.nodes.push(spec_node);
    (s, vec![])
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn from_data_basic() {
        let (scenario, _) = spectroscopy_scenario_from_data(
            "Test IR",
            &[4000.0, 3500.0, 3000.0],
            &[0.5, 0.8, 0.3],
            "cm⁻¹",
            "T",
        );
        assert_eq!(scenario.nodes.len(), 1);
        assert_eq!(scenario.domain, "measurement");
    }

    #[test]
    fn from_data_empty() {
        let (scenario, _) = spectroscopy_scenario_from_data("Empty", &[], &[], "x", "y");
        assert_eq!(scenario.nodes.len(), 1);
    }

    #[test]
    fn from_data_has_timeseries_channel() {
        let (scenario, edges) = spectroscopy_scenario_from_data(
            "UV-Vis",
            &[200.0, 300.0, 400.0, 500.0],
            &[0.1, 0.9, 0.5, 0.2],
            "nm",
            "Abs",
        );
        assert!(edges.is_empty());
        assert_eq!(scenario.nodes.len(), 1);
        assert!(!scenario.nodes[0].data_channels.is_empty());
    }

    #[test]
    fn scenario_from_jcamp_file() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.jcamp");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "##TITLE= Test Spectrum").unwrap();
        writeln!(f, "##DATA TYPE= INFRARED SPECTRUM").unwrap();
        writeln!(f, "##XUNITS= 1/CM").unwrap();
        writeln!(f, "##YUNITS= ABSORBANCE").unwrap();
        writeln!(f, "##PEAK TABLE= (XY..XY)").unwrap();
        writeln!(f, "4000.0, 0.1").unwrap();
        writeln!(f, "3500.0, 0.9").unwrap();
        writeln!(f, "3000.0, 0.5").unwrap();
        writeln!(f, "##END=").unwrap();

        let (scenario, edges) = spectroscopy_scenario(path.as_path()).unwrap();
        assert!(edges.is_empty());
        assert_eq!(scenario.domain, "measurement");
        assert_eq!(scenario.nodes.len(), 1);
        let node = &scenario.nodes[0];
        assert_eq!(node.data_channels.len(), 2);
        assert!(!node.scientific_ranges.is_empty());
    }

    #[test]
    fn scenario_from_jcamp_multi_block() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("multi.jcamp");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "##TITLE= Block A").unwrap();
        writeln!(f, "##XUNITS= 1/CM").unwrap();
        writeln!(f, "##YUNITS= T").unwrap();
        writeln!(f, "##PEAK TABLE= (XY..XY)").unwrap();
        writeln!(f, "1000.0, 50.0").unwrap();
        writeln!(f, "##END=").unwrap();
        writeln!(f, "##TITLE= Block B").unwrap();
        writeln!(f, "##XUNITS= nm").unwrap();
        writeln!(f, "##YUNITS= Abs").unwrap();
        writeln!(f, "##PEAK TABLE= (XY..XY)").unwrap();
        writeln!(f, "200.0, 0.3").unwrap();
        writeln!(f, "300.0, 0.9").unwrap();
        writeln!(f, "##END=").unwrap();

        let (scenario, _) = spectroscopy_scenario(path.as_path()).unwrap();
        let node = &scenario.nodes[0];
        assert!(node.data_channels.len() >= 3);
    }

    #[test]
    fn scenario_empty_title_fallback() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("notitle.jcamp");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "##TITLE=").unwrap();
        writeln!(f, "##PEAK TABLE= (XY..XY)").unwrap();
        writeln!(f, "100.0, 1.0").unwrap();
        writeln!(f, "##END=").unwrap();

        let (scenario, _) = spectroscopy_scenario(path.as_path()).unwrap();
        assert!(scenario.name.contains("JCAMP-DX Spectrum"));
    }

    #[test]
    fn scenario_large_peak_table_no_bar() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("large.jcamp");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "##TITLE= Large").unwrap();
        writeln!(f, "##XUNITS= cm-1").unwrap();
        writeln!(f, "##YUNITS= T").unwrap();
        writeln!(f, "##PEAK TABLE= (XY..XY)").unwrap();
        for i in 0..60 {
            writeln!(f, "{}.0, {}.0", i * 10, i).unwrap();
        }
        writeln!(f, "##END=").unwrap();

        let (scenario, _) = spectroscopy_scenario(path.as_path()).unwrap();
        let node = &scenario.nodes[0];
        assert_eq!(node.data_channels.len(), 1);
    }

    #[test]
    fn scenario_nonexistent_file() {
        let path = std::path::Path::new("/tmp/nonexistent_wetspring_jcamp_test.jcamp");
        assert!(spectroscopy_scenario(path).is_err());
    }
}
