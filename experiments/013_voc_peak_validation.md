# Experiment 013: VOC Peak Detection Validation (Reese 2019)

**Track**: 1 (Life Science) / Cross-track (signal processing)
**Date**: 2026-02-19
**Status**: ACTIVE
**Depends on**: Exp010 (Peak detection vs scipy)

---

## Objective

Validate Rust `bio::signal::find_peaks` and retention index matching against
the published VOC peak data from Reese et al. 2019, which identified 14 VOC
biomarkers from Microchloropsis salina cultures using SPME-GC-MS.

This bridges the gap between synthetic peak validation (Exp010) and
paper-specific control validation: we confirm our peak-detection and
retention-index tools can reproduce the discriminating compounds reported in
the paper.

## Data Source

| Field | Value |
|-------|-------|
| Paper | Reese et al. 2019. Sci. Rep. 9:13866 |
| DOI | 10.1038/s41598-019-50125-z |
| PMC | PMC6761164 |
| Data | Table 1 (14 compounds: m/z, RI, NIST ID, match %) |
| Availability | All data included in article body and supplementary |
| Extracted to | `experiments/results/013_voc_baselines/reese2019_table1.tsv` |

### Key Compounds (A+R only = grazer biomarkers)

| # | m/z | Compound | RI | Class |
|---|-----|----------|-----|-------|
| 1 | 82 | 2,2,6-trimethylcyclohexanone | 1021 | Carotenoid |
| 4 | 137 | beta-cyclocitral | 1209 | Carotenoid |
| 5 | 121 | 4-(2,6,6-trimethyl-1-cyclohexen-1-yl)-2-butanone | 1419 | Carotenoid |
| 6 | 177 | trans-beta-ionone | 1495 | Carotenoid |
| 7 | 57 | Alkane (unidentified) | 1691 | Alkane |
| 9 | 96 | 3-Nonenoic acid methyl ester | 1134 | Methyl ester |

## What We Can Validate

The paper provides retention indices and base peak masses, not raw chromatograms.
Therefore, validation targets are:

1. **Retention index calculation**: Given elution times and an alkane standard
   series, compute Kovats retention indices matching the paper's values
2. **RI matching tolerance**: Verify our tolerance search correctly matches
   experimental RI to theoretical RI within the 5% window used in the paper
3. **Peak classification**: Group compounds by structural class (carotenoid,
   alkane, fatty acid) using fragmentation patterns
4. **Biomarker discrimination**: Separate A+R-only compounds from A+R,A compounds

## Protocol

1. Parse `reese2019_table1.tsv` into structured compound records
2. For each compound with both experimental and theoretical RI:
   - Compute RI deviation (should be <5% per paper criteria)
3. Generate a synthetic chromatogram with 7 A+R peaks at their reported RIs
4. Run `bio::signal::find_peaks` on the synthetic chromatogram
5. Verify all 7 peaks are detected with correct indices
6. Test retention index tolerance matching

## Acceptance Criteria

1. TSV baseline parses correctly (14 compounds)
2. All 5 compounds with theoretical RI: deviation <5%
3. Synthetic chromatogram peak detection finds 7 A+R biomarker peaks
4. Peak retention indices match within ±1 of expected positions
5. NIST match percentages parse correctly for identified compounds
6. Biomarker classification separates A+R from A+R,A compounds

## Validation Binary

`cargo run --bin validate_voc_peaks`

## Outputs

- `validate_voc_peaks` checks in Validator harness
- RI deviation analysis
- Peak detection accuracy on synthetic GC-MS chromatogram
