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
    let _ = wetspring_barracuda::io::mzml::parse_mzml(&path);

    // Also test stats_from_file
    let _ = wetspring_barracuda::io::mzml::stats_from_file(&path);
});
