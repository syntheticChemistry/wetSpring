# Experiment 005: asari Bootstrap

**Track**: 2 (PFAS / blueFish)
**Date**: 2026-02-12
**Status**: NOT STARTED

## Objective

Install and validate **asari**, the open-source LC-MS metabolomics data
processing tool (Nature Communications 2023). This is the foundational
tool for Track 2 — if we can replicate asari's output, we have a
validated open-source baseline to port to Rust+GPU.

## Background

asari (github.com/shuzhao-li-lab/asari) addresses reproducibility and
provenance issues in metabolomics data processing. It offers:

- Explicit tracking of processing steps
- Improved feature detection vs XCMS and MZmine
- Scalable performance on large datasets
- Vendor-neutral input (mzML format via ProteoWizard msconvert)

Developed by Shuzhao Li's lab, with A. Daniel Jones (MSU) involvement.
Published in Nature Communications 14, 4113 (2023).

## Protocol

### Step 1: Install asari

```bash
# Create Python virtual environment for Track 2
python3 -m venv ~/envs/wetspring-t2
source ~/envs/wetspring-t2/bin/activate

# Install asari
pip install asari-metabolomics

# Verify
asari --version
```

### Step 2: Get Demo Data

```bash
cd "$(git rev-parse --show-toplevel)/data"

# Clone asari demo datasets
git clone https://github.com/shuzhao-li-lab/data.git asari-demo-data
```

### Step 3: Run asari on Demo Data

```bash
# Process demo mzML files
asari process -i data/asari-demo-data/<demo_folder>/ -o results/005_asari_bootstrap/
```

### Step 4: Inspect Output

- Feature table (sample × feature intensity matrix)
- Feature metadata (m/z, RT, quality scores)
- Processing log (for reproducibility tracking)
- Mass tracks and chromatograms

### Step 5: Benchmark

Record:
- Total runtime
- Peak memory usage
- Number of features detected
- Number of samples processed
- Feature quality distribution

## Success Criteria

- [ ] asari installed and runnable
- [ ] Demo mzML data downloaded
- [ ] asari processes demo data without errors
- [ ] Feature table produced (CSV/TSV)
- [ ] Feature count matches expected range from demo data
- [ ] Processing is fully reproducible (same input → same output)
- [ ] Runtime and memory documented

## Notes for Rust Evolution

Key algorithmic components to understand for future porting:

1. **mzML parsing**: XML + base64-encoded binary arrays
2. **Mass track extraction**: Binning m/z values across scans
3. **Peak detection**: Composite scoring of chromatographic peaks
4. **Mass alignment**: Reference-based feature alignment across samples
5. **Feature quantification**: Integration of peak areas

Each of these is a candidate for Rust+GPU acceleration in Phase B2-B3.
