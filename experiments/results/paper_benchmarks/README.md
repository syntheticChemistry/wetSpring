# Paper Benchmark Data

Structured reference data extracted from the 4 source papers. Used as ground
truth to validate pipeline results on public NCBI datasets.

**Strategy**: We cannot validate the exact papers (raw data not publicly
available for 3 of 4). Instead, we use paper findings as validation targets:
run our pipeline on similar public datasets and confirm biologically consistent
results.

## Files

| File | Paper | Contents |
|------|-------|----------|
| `humphrey2023_bacteriome.tsv` | Humphrey 2023 | 18 OTUs, core genera, taxonomy |
| `humphrey2023_metrics.json` | Humphrey 2023 | Community profile, key findings, validation targets |
| `carney2016_crash_agents.json` | Carney 2016 | Crash agents (Brachionus, Chaetonotus), detection methods |
| `reese2019_voc_biomarkers.json` | Reese 2019 | 14 VOC compounds, RI values, NIST IDs |
| `reichardt2020_spectroradiometric.json` | Reichardt 2020 | Organisms detected, method details |

## How Paper Data Validates Public Datasets

```
Paper Reference Data         Public NCBI Data
(ground truth)               (pipeline input)
       |                           |
       |    ┌─────────────────┐    |
       └──> | BENCHMARK CHECK | <──┘
            └────────┬────────┘
                     |
            Biologically consistent?
            Same phyla/genera detected?
            Diversity in expected range?
            Crash agents resolvable?
```

## Validation Criteria

A pipeline run on public Nannochloropsis/microalgae 16S data is considered
**validated** if:

1. **Taxonomy**: Resolves to genus level for marine-associated bacteria
2. **Expected phyla**: Detects Proteobacteria, Bacteroidetes, Planctomycetes
3. **Diversity range**: Shannon 1.0-4.0, Simpson 0.5-1.0
4. **OTU count**: 10-500 OTUs (varies by depth and method)
5. **No hallucination**: Does not fabricate organisms absent from the data

## Data Availability Status

| Paper | Raw Data | Paper Data | Status |
|-------|----------|------------|--------|
| Humphrey 2023 | NOT in SRA | OTUs in figures/tables | Benchmark extracted |
| Carney 2016 | NOT in SRA | Crash agents in text | Benchmark extracted |
| Reese 2019 | Table 1 in paper | 14 VOC compounds | Benchmark extracted |
| Reichardt 2020 | NOT available | Detection results in text | Benchmark extracted, no validation possible |
