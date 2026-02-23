# Experiment 006: PFΔScreen Validation

**Track**: 2 (PFAS / blueFish)
**Date**: 2026-02-12
**Status**: COMPLETE (10 checks)

## Objective

Install and validate **PFΔScreen**, the open-source Python tool for
automated PFAS feature prioritization in non-target high-resolution
mass spectrometry (HRMS) data. This establishes the PFAS screening
baseline for Track 2.

## Background

PFΔScreen (github.com/JonZwe/PFAScreen) provides:

- Mass defect / carbon number (MD/C-m/C) analysis
- Kendrick mass defect (KMD) for homologous series identification
- Diagnostic fragment ions from MS2 data
- Fragment mass differences (ΔF = CF2 = 49.9968 Da)
- GUI and command-line interfaces
- Vendor-independent mzML input

Published in Analytical and Bioanalytical Chemistry (2023) by Zweigle et al.
Demonstrated identification of >80 PFAS compounds across >15 classes in
contaminated soil samples, including novel transformation products.

## Protocol

### Step 1: Install PFΔScreen

```bash
source ~/envs/wetspring-t2/bin/activate

# Install from GitHub
pip install git+https://github.com/JonZwe/PFAScreen.git

# Or clone and install
git clone https://github.com/JonZwe/PFAScreen.git
cd PFAScreen && pip install .

# Install pyOpenMS dependency
pip install pyopenms
```

### Step 2: Also Install FindPFΔS (predecessor)

```bash
git clone https://github.com/JonZwe/FindPFAS.git
cd FindPFAS && pip install .
```

### Step 3: Obtain Test HRMS Data

Options (in priority order):
1. PFΔScreen paper supplementary data (check journal)
2. NORMAN Digital Sample Freezing Platform
3. MassBank PFAS reference spectra
4. Generate synthetic mzML test data with known PFAS signatures

```bash
cd "$(git rev-parse --show-toplevel)/data"
mkdir -p pfas-hrms-test
# See scripts/download_public_data.sh for data download
```

### Step 4: Run PFΔScreen

```bash
# Command-line mode
pfascreen -i data/pfas-hrms-test/sample.mzML \
          -o results/006_pfascreen/ \
          --mode LC-HRMS
```

### Step 5: Inspect Output

- PFAS candidate list with confidence scores
- KMD plots showing homologous series
- MS2 fragment matches and diagnostic ions
- Compound class assignments (PFCA, PFSA, FTS, etc.)

### Step 6: Validate

Compare PFΔScreen output against:
- Known PFAS in test sample (if ground truth available)
- Published identifications from paper
- Cross-reference with EPA CompTox PFAS list

## Success Criteria

- [ ] PFΔScreen installed and runnable
- [ ] pyOpenMS working
- [ ] Test mzML data obtained
- [ ] PFΔScreen produces PFAS candidate list
- [ ] KMD plots generated
- [ ] MS2 matching functional
- [ ] Results consistent with expected PFAS in test data
- [ ] Runtime and memory documented

## PFAS Detection Chemistry (Reference)

Key m/z values for PFAS fragment ions:
| Fragment | m/z (exact) | Formula |
|----------|------------|---------|
| CF3⁻ | 68.9952 | CF3⁻ |
| C2F5⁻ | 118.9920 | C2F5⁻ |
| C3F7⁻ | 168.9888 | C3F7⁻ |
| SO3⁻ | 79.9574 | SO3⁻ |
| FSO3⁻ | 98.9558 | FSO3⁻ |
| CF2 (Δm) | 49.9968 | Homologous series step |

## Notes for Rust Evolution

Key algorithmic components for future porting:

1. **mzML → feature detection** (pyOpenMS): centroiding, peak picking
2. **Mass defect calculation**: arithmetic on exact masses
3. **KMD analysis**: Kendrick normalization, defect grouping
4. **MS2 matching**: cosine similarity of fragment spectra
5. **Homologous series**: pattern matching in m/z space

The MS2 cosine similarity kernel is especially valuable — it maps
directly to a GPU dot product shader usable in many domains.
