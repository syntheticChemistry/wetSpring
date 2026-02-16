# PFΔScreen Configuration (Track 2: blueFish)

**Tool**: PFΔScreen — Automated PFAS feature prioritization for non-target HRMS
**Source**: https://github.com/JonZwe/PFAScreen
**Paper**: Analytical and Bioanalytical Chemistry (2023)
**License**: LGPL-2.1

## Installation

```bash
source ~/envs/wetspring-t2/bin/activate
pip install pyopenms
pip install git+https://github.com/JonZwe/PFAScreen.git
```

## Also install FindPFΔS (predecessor tool)

```bash
pip install git+https://github.com/JonZwe/FindPFAS.git
```

## Usage

```bash
# Run PFAS screening on mzML file
pfascreen -i sample.mzML -o results/ --mode LC-HRMS
```

## PFAS Detection Methods

PFΔScreen uses multiple complementary approaches:

1. **MD/C-m/C**: Mass defect vs carbon number plot — PFAS cluster in
   distinct regions due to high fluorine content
2. **KMD (Kendrick Mass Defect)**: Identifies homologous series
   (compounds differing by CF2 units, Δm = 49.9968 Da)
3. **Diagnostic fragments**: MS2 fragments characteristic of C-F bonds
   (CF3⁻ m/z 68.9952, C2F5⁻ 118.9920, C3F7⁻ 168.9888)
4. **Fragment mass differences**: Neutral losses specific to PFAS classes

## Key PFAS Fragment Reference

| Fragment | m/z (exact) | Class indicator |
|----------|------------|-----------------|
| CF3⁻ | 68.9952 | General PFAS |
| C2F5⁻ | 118.9920 | General PFAS |
| C3F7⁻ | 168.9888 | General PFAS |
| SO3⁻ | 79.9574 | Sulfonates (PFSA) |
| FSO3⁻ | 98.9558 | Sulfonates |
| HSO4⁻ | 96.9601 | Sulfates |
| CF2=CF-CF3 | 131.0 | FTS class |

## Rust Evolution Notes

Core algorithms to port:
1. pyOpenMS feature detection → Rust centroiding
2. KMD calculation → Rust mass arithmetic (high precision)
3. MS2 cosine similarity → GPU dot product kernel
4. Suspect list matching → GPU hash table lookup
5. Homologous series detection → Rust pattern matching
