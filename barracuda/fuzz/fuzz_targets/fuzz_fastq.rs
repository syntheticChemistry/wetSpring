// SPDX-License-Identifier: AGPL-3.0-or-later
#![no_main]
use libfuzzer_sys::fuzz_target;
use std::io::Write;
use tempfile::NamedTempFile;

fuzz_target!(|data: &[u8]| {
    // Write fuzz data to a temp file and try to parse it
    let mut f = NamedTempFile::new().unwrap();
    f.write_all(data).unwrap();
    let path = f.path().to_path_buf();

    // Should not panic regardless of input
    let _ = wetspring_barracuda::io::fastq::FastqIter::open(&path)
        .and_then(Iterator::collect::<std::result::Result<Vec<_>, _>>);

    // Also test stats_from_file
    let _ = wetspring_barracuda::io::fastq::stats_from_file(&path);
});
