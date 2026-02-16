# asari Configuration (Track 2: blueFish)

**Tool**: asari — Trackable and scalable LC-MS metabolomics data processing
**Source**: https://github.com/shuzhao-li-lab/asari
**Paper**: Nature Communications 14, 4113 (2023)
**License**: MIT

## Installation

```bash
python3 -m venv ~/envs/wetspring-t2
source ~/envs/wetspring-t2/bin/activate
pip install asari-metabolomics
```

## Usage

```bash
# Process mzML files in a directory
asari process -i /path/to/mzml/files/ -o /path/to/output/

# With specific parameters
asari process -i data/ -o results/ --mode pos --mass_range 50,2000
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mode` | pos | Ionization mode: pos or neg |
| `--mass_range` | 50,2000 | m/z range to process |
| `--min_peak_height` | 1000 | Minimum peak intensity |
| `--mz_tolerance` | 5 | Mass tolerance in ppm |

## Output Files

- `preferred_Feature_table.tsv` — Main feature table (sample × feature)
- `export/full_Feature_table.tsv` — Complete feature table with metadata
- `export/aligned_features.json` — Feature alignment details
- `export/log.txt` — Processing log for reproducibility

## Rust Evolution Notes

Core algorithms to port:
1. mzML binary parser (base64-decoded float arrays)
2. Mass track extraction (histogram binning of m/z)
3. Gaussian peak detection (composite scoring)
4. Cross-sample alignment (reference mapping)
5. Feature quantification (trapezoidal integration)
