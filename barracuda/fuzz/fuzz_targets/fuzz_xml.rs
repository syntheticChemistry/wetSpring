#![no_main]
use libfuzzer_sys::fuzz_target;
use std::io::{BufReader, Cursor};
use wetspring_barracuda::io::xml::{XmlEvent, XmlReader};

fuzz_target!(|data: &[u8]| {
    // Create XmlReader from bytes and iterate events until Eof
    let cursor = Cursor::new(data);
    let reader = BufReader::new(cursor);
    let mut xml_reader = XmlReader::new(reader);
    xml_reader.set_trim_text(true);

    loop {
        match xml_reader.next_event() {
            Ok(XmlEvent::Eof) => break,
            Ok(_) => {}
            Err(_) => break,
        }
    }
});
