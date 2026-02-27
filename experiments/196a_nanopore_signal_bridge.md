# Experiment 196a: Nanopore Signal Bridge — POD5/NRS Parser Validation

**Date:** February 27, 2026
**Phase:** 61
**Track:** Field Genomics (Sub-thesis 06)
**Status:** PASS (28/28 checks)
**Binary:** `validate_nanopore_signal_bridge`

---

## Objective

Validate `io::nanopore` — a sovereign FAST5/POD5 signal parser and NRS (Nanopore
Read Simulation) synthetic read generator. This is the pre-hardware validation
step: all parsing logic is exercised against synthetic signal structures that
match the POD5 spec, without requiring actual MinION hardware.

## Background

Oxford Nanopore's POD5 format stores raw ionic current signal alongside read
metadata (channel, read ID, start time, signal scaling). Our parser must:

1. Parse POD5 headers and extract signal metadata
2. Stream reads via iterator API (no whole-file buffering)
3. Convert raw signal to FASTQ-compatible quality + sequence
4. Handle NRS synthetic format for pre-hardware testing

## Validation Sections

| Section | Checks | What It Validates |
|---------|:------:|-------------------|
| S1: POD5 header parsing | 4 | File signature, version, channel count, run_id extraction |
| S2: Signal extraction | 5 | Raw signal arrays, picoamp conversion, scaling parameters |
| S3: Quality metrics | 4 | Mean quality score, signal-to-noise ratio, read length distribution |
| S4: Read→FASTQ conversion | 5 | Synthetic basecalling, quality string, header formatting |
| S5: Streaming iteration | 4 | `NanoporeIter` lazy evaluation, memory usage, batch retrieval |
| S6: NRS format | 3 | Synthetic read generation from community profiles, round-trip fidelity |
| S7: Error handling | 3 | Truncated files, invalid headers, corrupt signal graceful recovery |

**Total:** 28/28 PASS

## Tolerance Constants

- `NANOPORE_SIGNAL_SNR` — minimum signal-to-noise ratio for valid reads
- `BASECALL_ACCURACY` — synthetic basecaller accuracy threshold

## Connection to Sub-thesis 06

This is the first operational component of the field genomics pipeline. With
`io::nanopore` operational, synthetic reads can flow through the existing 16S
pipeline (Exp196b) and into NPU classification (Exp196c). Real POD5 files from
MinION hardware will exercise the same parsing code path.
