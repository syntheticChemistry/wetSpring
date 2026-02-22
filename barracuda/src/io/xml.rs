// SPDX-License-Identifier: AGPL-3.0-or-later
//! Minimal XML pull parser — sovereign, zero-dependency.
//!
//! Handles the subset of XML used by mzML: elements with attributes,
//! text content, self-closing elements, comments, and processing
//! instructions (the latter two are silently skipped).
//!
//! # Limitations
//!
//! - No namespace handling (uses local names only).
//! - No CDATA sections.
//! - No DTD entity expansion beyond the 5 predefined XML entities.
//! - Attributes must use `"` or `'` quoting.
//!
//! These limitations are acceptable for machine-generated mzML files.

use crate::error::{Error, Result};
use std::collections::VecDeque;
use std::io::BufRead;

const INITIAL_TEXT_BUF_CAPACITY: usize = 4096;
const INITIAL_TAG_BUF_CAPACITY: usize = 512;

/// An XML event produced by the pull parser.
#[derive(Debug)]
pub enum XmlEvent {
    /// Start of an element.  Also emitted for self-closing elements
    /// (e.g., `<cvParam .../>`); a matching [`EndElement`](XmlEvent::EndElement)
    /// is queued immediately after.
    StartElement {
        /// Element name (local, no namespace prefix).
        name: String,
        /// Attribute `(key, value)` pairs (values are XML-unescaped).
        attrs: Vec<(String, String)>,
    },
    /// End of an element.
    EndElement {
        /// Element name.
        name: String,
    },
    /// Text content between tags (XML-unescaped).
    Text(String),
    /// End of the document.
    Eof,
}

/// Streaming XML pull parser over any [`BufRead`] source.
pub struct XmlReader<R> {
    reader: R,
    trim_text: bool,
    queue: VecDeque<XmlEvent>,
    text_buf: Vec<u8>,
    tag_buf: Vec<u8>,
    eof: bool,
}

impl<R: BufRead> XmlReader<R> {
    /// Create a new parser wrapping a buffered reader.
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            trim_text: false,
            queue: VecDeque::with_capacity(4),
            text_buf: Vec::with_capacity(INITIAL_TEXT_BUF_CAPACITY),
            tag_buf: Vec::with_capacity(INITIAL_TAG_BUF_CAPACITY),
            eof: false,
        }
    }

    /// Enable or disable whitespace trimming of text events.
    pub fn set_trim_text(&mut self, trim: bool) {
        self.trim_text = trim;
    }

    /// Pull the next XML event from the stream.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Xml`] for malformed XML or underlying I/O failures.
    pub fn next_event(&mut self) -> Result<XmlEvent> {
        if let Some(event) = self.queue.pop_front() {
            return Ok(event);
        }
        if self.eof {
            return Ok(XmlEvent::Eof);
        }

        loop {
            // ── Phase 1: read text content until '<' or EOF ──
            self.text_buf.clear();
            let n = self
                .reader
                .read_until(b'<', &mut self.text_buf)
                .map_err(|e| Error::Xml(format!("read error: {e}")))?;

            if n == 0 {
                self.eof = true;
                return Ok(XmlEvent::Eof);
            }

            let found_lt = self.text_buf.last() == Some(&b'<');
            if found_lt {
                self.text_buf.pop();
            }

            // Convert text to owned String (releases borrow on self.text_buf).
            let text_owned = if self.text_buf.is_empty() {
                None
            } else {
                let raw = String::from_utf8_lossy(&self.text_buf).into_owned();
                if self.trim_text {
                    let trimmed = raw.trim().to_owned();
                    if trimmed.is_empty() {
                        None
                    } else {
                        Some(xml_unescape(&trimmed))
                    }
                } else {
                    Some(xml_unescape(&raw))
                }
            };

            if let Some(text) = text_owned {
                self.queue.push_back(XmlEvent::Text(text));
            }

            if !found_lt {
                self.eof = true;
                self.queue.push_back(XmlEvent::Eof);
                return Ok(self.queue.pop_front().unwrap_or(XmlEvent::Eof));
            }

            // ── Phase 2: read tag content until '>' ──────────
            self.tag_buf.clear();
            let tn = self
                .reader
                .read_until(b'>', &mut self.tag_buf)
                .map_err(|e| Error::Xml(format!("read error: {e}")))?;

            if tn == 0 || self.tag_buf.last() != Some(&b'>') {
                return Err(Error::Xml("unexpected EOF inside tag".into()));
            }
            self.tag_buf.pop(); // strip '>'

            // ── Phase 3: classify tag and queue events ───────

            // Comment: <!-- ... -->
            if self.tag_buf.starts_with(b"!--") {
                if !self.tag_buf.ends_with(b"--") {
                    self.skip_to_comment_end()?;
                }
                if let Some(event) = self.queue.pop_front() {
                    return Ok(event);
                }
                continue;
            }

            // Processing instruction (<?...?>) or DOCTYPE (<!...>): skip.
            if self.tag_buf.starts_with(b"?") || self.tag_buf.starts_with(b"!") {
                if let Some(event) = self.queue.pop_front() {
                    return Ok(event);
                }
                continue;
            }

            // End tag: </name>
            if self.tag_buf.starts_with(b"/") {
                let name = std::str::from_utf8(&self.tag_buf[1..])
                    .map_err(|e| Error::Xml(format!("invalid UTF-8 in end tag: {e}")))?
                    .trim()
                    .to_owned();
                self.queue.push_back(XmlEvent::EndElement { name });
                return Ok(self.queue.pop_front().unwrap_or(XmlEvent::Eof));
            }

            // Start or self-closing element.
            let tag_str = std::str::from_utf8(&self.tag_buf)
                .map_err(|e| Error::Xml(format!("invalid UTF-8 in tag: {e}")))?
                .to_owned();
            let is_empty = tag_str.trim_end().ends_with('/');
            let body = if is_empty {
                tag_str
                    .trim_end()
                    .strip_suffix('/')
                    .unwrap_or(&tag_str)
                    .trim()
            } else {
                tag_str.trim()
            };

            let (name_part, attr_part) = body
                .find(|c: char| c.is_ascii_whitespace())
                .map_or((body, ""), |pos| (&body[..pos], body[pos..].trim()));

            let name = name_part.to_owned();
            let attrs = parse_attributes(attr_part);

            self.queue.push_back(XmlEvent::StartElement {
                name: name.clone(),
                attrs,
            });
            if is_empty {
                self.queue.push_back(XmlEvent::EndElement { name });
            }

            return Ok(self.queue.pop_front().unwrap_or(XmlEvent::Eof));
        }
    }

    // ── Private helpers ──────────────────────────────────────────

    /// Consume bytes until `-->` is found (comment end).
    fn skip_to_comment_end(&mut self) -> Result<()> {
        loop {
            self.tag_buf.clear();
            let n = self
                .reader
                .read_until(b'>', &mut self.tag_buf)
                .map_err(|e| Error::Xml(format!("read error: {e}")))?;
            if n == 0 {
                return Err(Error::Xml("unexpected EOF inside comment".into()));
            }
            self.tag_buf.pop(); // strip '>'
            if self.tag_buf.ends_with(b"--") {
                return Ok(());
            }
        }
    }
}

// ── Free functions ───────────────────────────────────────────────

/// Parse XML attribute pairs from the attribute portion of a tag.
///
/// Expects input like `key1="val1" key2="val2"`.
fn parse_attributes(s: &str) -> Vec<(String, String)> {
    let mut attrs = Vec::new();
    let mut rest = s.trim();

    while !rest.is_empty() {
        let Some(eq) = rest.find('=') else {
            break;
        };
        let key = rest[..eq].trim();
        rest = rest[eq + 1..].trim();

        if rest.is_empty() {
            break;
        }
        let quote = rest.as_bytes()[0];
        if quote != b'"' && quote != b'\'' {
            break;
        }
        rest = &rest[1..];
        let Some(end) = rest.find(quote as char) else {
            break;
        };
        let value = &rest[..end];
        rest = rest[end + 1..].trim();

        attrs.push((key.to_owned(), xml_unescape(value)));
    }

    attrs
}

/// Unescape the 5 predefined XML character entities.
#[must_use]
fn xml_unescape(s: &str) -> String {
    if !s.contains('&') {
        return s.to_owned();
    }
    s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::io::{BufReader, Cursor};

    fn events_from(xml: &str) -> Vec<XmlEvent> {
        let cursor = Cursor::new(xml.as_bytes());
        let mut reader = XmlReader::new(BufReader::new(cursor));
        reader.set_trim_text(true);
        let mut events = Vec::new();
        loop {
            let event = reader.next_event().unwrap();
            if matches!(event, XmlEvent::Eof) {
                break;
            }
            events.push(event);
        }
        events
    }

    #[test]
    fn simple_element_with_text() {
        let events = events_from("<root>hello</root>");
        assert_eq!(events.len(), 3);
        assert!(matches!(&events[0], XmlEvent::StartElement { name, .. } if name == "root"));
        assert!(matches!(&events[1], XmlEvent::Text(t) if t == "hello"));
        assert!(matches!(&events[2], XmlEvent::EndElement { name } if name == "root"));
    }

    #[test]
    fn self_closing_with_attr() {
        let events = events_from(r#"<item key="value"/>"#);
        assert_eq!(events.len(), 2);
        match &events[0] {
            XmlEvent::StartElement { name, attrs } => {
                assert_eq!(name, "item");
                assert_eq!(attrs, &[("key".to_owned(), "value".to_owned())]);
            }
            other => panic!("expected StartElement, got {other:?}"),
        }
        assert!(matches!(&events[1], XmlEvent::EndElement { name } if name == "item"));
    }

    #[test]
    fn multiple_attributes() {
        let events = events_from(r#"<cv accession="MS:1000511" value="1"/>"#);
        match &events[0] {
            XmlEvent::StartElement { attrs, .. } => {
                assert_eq!(attrs.len(), 2);
                assert_eq!(attrs[0], ("accession".to_owned(), "MS:1000511".to_owned()));
                assert_eq!(attrs[1], ("value".to_owned(), "1".to_owned()));
            }
            other => panic!("expected StartElement, got {other:?}"),
        }
    }

    #[test]
    fn nested_elements() {
        let events = events_from("<a><b>text</b></a>");
        assert_eq!(events.len(), 5);
        assert!(matches!(&events[0], XmlEvent::StartElement { name, .. } if name == "a"));
        assert!(matches!(&events[1], XmlEvent::StartElement { name, .. } if name == "b"));
        assert!(matches!(&events[2], XmlEvent::Text(t) if t == "text"));
        assert!(matches!(&events[3], XmlEvent::EndElement { name } if name == "b"));
        assert!(matches!(&events[4], XmlEvent::EndElement { name } if name == "a"));
    }

    #[test]
    fn xml_declaration_skipped() {
        let events = events_from(r#"<?xml version="1.0"?><root/>"#);
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], XmlEvent::StartElement { name, .. } if name == "root"));
        assert!(matches!(&events[1], XmlEvent::EndElement { name } if name == "root"));
    }

    #[test]
    fn comment_skipped() {
        let events = events_from("<a><!-- comment --><b/></a>");
        assert_eq!(events.len(), 4);
        assert!(matches!(&events[0], XmlEvent::StartElement { name, .. } if name == "a"));
        assert!(matches!(&events[1], XmlEvent::StartElement { name, .. } if name == "b"));
        assert!(matches!(&events[2], XmlEvent::EndElement { name } if name == "b"));
        assert!(matches!(&events[3], XmlEvent::EndElement { name } if name == "a"));
    }

    #[test]
    fn text_trimming() {
        let events = events_from("<a>  hello world  </a>");
        assert!(matches!(&events[1], XmlEvent::Text(t) if t == "hello world"));
    }

    #[test]
    fn entity_unescape_text() {
        let events = events_from("<a>&amp;&lt;&gt;</a>");
        assert!(matches!(&events[1], XmlEvent::Text(t) if t == "&<>"));
    }

    #[test]
    fn entity_unescape_attr() {
        let events = events_from(r#"<a val="a&amp;b"/>"#);
        match &events[0] {
            XmlEvent::StartElement { attrs, .. } => {
                assert_eq!(attrs[0].1, "a&b");
            }
            other => panic!("expected StartElement, got {other:?}"),
        }
    }

    #[test]
    fn whitespace_only_text_trimmed() {
        let events = events_from("<a>   \n\t  </a>");
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], XmlEvent::StartElement { name, .. } if name == "a"));
        assert!(matches!(&events[1], XmlEvent::EndElement { name } if name == "a"));
    }

    #[test]
    fn empty_element_with_space_before_slash() {
        let events = events_from(r"<br />");
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], XmlEvent::StartElement { name, .. } if name == "br"));
        assert!(matches!(&events[1], XmlEvent::EndElement { name } if name == "br"));
    }

    #[test]
    fn multiline_comment_skipped() {
        let events = events_from("<a><!-- multi\nline\ncomment --><b/></a>");
        assert_eq!(events.len(), 4);
        assert!(matches!(&events[0], XmlEvent::StartElement { name, .. } if name == "a"));
        assert!(matches!(&events[1], XmlEvent::StartElement { name, .. } if name == "b"));
    }

    #[test]
    fn empty_document() {
        let xml = "";
        let mut reader = XmlReader::new(std::io::Cursor::new(xml));
        reader.set_trim_text(true);
        let event = reader.next_event().unwrap();
        assert!(matches!(event, XmlEvent::Eof));
    }

    #[test]
    fn single_quoted_attributes() {
        let events = events_from("<a name='value'/>");
        match &events[0] {
            XmlEvent::StartElement { attrs, .. } => {
                assert_eq!(attrs.len(), 1);
                assert_eq!(attrs[0].0, "name");
                assert_eq!(attrs[0].1, "value");
            }
            other => panic!("expected StartElement, got {other:?}"),
        }
    }

    #[test]
    fn text_with_entities_apos_and_quot() {
        let events = events_from("<a>&apos;&quot;</a>");
        assert!(matches!(&events[1], XmlEvent::Text(t) if t == "'\""));
    }

    #[test]
    fn adjacent_elements_no_whitespace() {
        let events = events_from("<a/><b/><c/>");
        assert_eq!(events.len(), 6);
        assert!(matches!(&events[0], XmlEvent::StartElement { name, .. } if name == "a"));
        assert!(matches!(&events[2], XmlEvent::StartElement { name, .. } if name == "b"));
        assert!(matches!(&events[4], XmlEvent::StartElement { name, .. } if name == "c"));
    }

    #[test]
    fn comment_spanning_multiple_gt_chars() {
        // Comment with '>' inside it — exercises skip_to_comment_end loop
        let events = events_from("<a><!-- a > b > c --><b/></a>");
        assert_eq!(events.len(), 4);
        assert!(matches!(&events[0], XmlEvent::StartElement { name, .. } if name == "a"));
        assert!(matches!(&events[1], XmlEvent::StartElement { name, .. } if name == "b"));
    }

    #[test]
    fn eof_after_text_no_close_tag() {
        // Text with no close tag — hits the !found_lt + EOF path
        let xml = "<a>trailing text";
        let mut reader = XmlReader::new(std::io::Cursor::new(xml));
        reader.set_trim_text(true);
        let e1 = reader.next_event().unwrap();
        assert!(matches!(e1, XmlEvent::StartElement { name, .. } if name == "a"));
        let e2 = reader.next_event().unwrap();
        assert!(matches!(e2, XmlEvent::Text(t) if t == "trailing text"));
    }

    #[test]
    fn eof_returns_eof_on_repeated_calls() {
        let xml = "";
        let mut reader = XmlReader::new(std::io::Cursor::new(xml));
        reader.set_trim_text(true);
        let e1 = reader.next_event().unwrap();
        assert!(matches!(e1, XmlEvent::Eof));
        // Second call should also return Eof (exercises self.eof guard)
        let e2 = reader.next_event().unwrap();
        assert!(matches!(e2, XmlEvent::Eof));
    }

    #[test]
    fn truncated_tag_eof() {
        // Tag that never closes with '>'
        let xml = "<a><unclosed";
        let mut reader = XmlReader::new(std::io::Cursor::new(xml));
        reader.set_trim_text(true);
        let e1 = reader.next_event().unwrap();
        assert!(matches!(e1, XmlEvent::StartElement { name, .. } if name == "a"));
        let e2 = reader.next_event();
        assert!(e2.is_err()); // "unexpected EOF inside tag"
    }

    #[test]
    fn text_without_trim() {
        // Non-trimmed text with entities (exercises raw xml_unescape path)
        let xml = "<a>  &amp;hello  </a>";
        let mut reader = XmlReader::new(std::io::Cursor::new(xml));
        // trim_text defaults to false
        let e1 = reader.next_event().unwrap();
        assert!(matches!(e1, XmlEvent::StartElement { name, .. } if name == "a"));
        let e2 = reader.next_event().unwrap();
        match e2 {
            XmlEvent::Text(t) => assert_eq!(t, "  &hello  "),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn malformed_attribute_no_equals() {
        // Attribute without '=' — triggers the break in parse_attributes
        let events = events_from("<a broken/>");
        match &events[0] {
            XmlEvent::StartElement { name, attrs, .. } => {
                assert_eq!(name, "a");
                assert!(attrs.is_empty());
            }
            other => panic!("expected StartElement, got {other:?}"),
        }
    }

    #[test]
    fn malformed_attribute_no_value() {
        // Attribute with '=' but no value — triggers rest.is_empty() break
        let events = events_from("<a key=/>");
        match &events[0] {
            XmlEvent::StartElement { name, attrs, .. } => {
                assert_eq!(name, "a");
                assert!(attrs.is_empty());
            }
            other => panic!("expected StartElement, got {other:?}"),
        }
    }

    #[test]
    fn malformed_attribute_no_quote() {
        // Attribute with '=' but unquoted value — triggers quote check break
        let events = events_from("<a key=noquote/>");
        match &events[0] {
            XmlEvent::StartElement { name, attrs, .. } => {
                assert_eq!(name, "a");
                assert!(attrs.is_empty());
            }
            other => panic!("expected StartElement, got {other:?}"),
        }
    }

    #[test]
    fn malformed_attribute_unclosed_quote() {
        // Attribute with opening quote but no closing quote
        let events = events_from(r#"<a key="unclosed/>"#);
        match &events[0] {
            XmlEvent::StartElement { name, attrs, .. } => {
                assert_eq!(name, "a");
                assert!(attrs.is_empty());
            }
            other => panic!("expected StartElement, got {other:?}"),
        }
    }

    #[test]
    fn text_before_and_after_comment() {
        // Text before comment, then text after — exercises queue + comment interaction
        let events = events_from("<a>before<!-- comment -->after</a>");
        // StartElement, Text("before"), Text("after"), EndElement
        let texts: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                XmlEvent::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(texts, vec!["before", "after"]);
    }

    #[test]
    fn doctype_skipped() {
        let events = events_from("<!DOCTYPE html><root/>");
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], XmlEvent::StartElement { name, .. } if name == "root"));
    }

    #[test]
    fn text_before_pi_returns_queued_text() {
        // Text queued before a processing instruction is encountered
        let events = events_from("<a>hello<?pi data?></a>");
        let texts: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                XmlEvent::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(texts, vec!["hello"]);
    }

    #[test]
    fn text_before_doctype_returns_queued_text() {
        // Text queued before a DOCTYPE (<!...>) — exercises same return path
        let events = events_from("<a>data<!ENTITY foo></a>");
        let texts: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                XmlEvent::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(texts, vec!["data"]);
    }

    #[test]
    fn eof_inside_comment_is_error() {
        // Comment with an inner `>` but no `-->` closing — the initial
        // tag read sees `!-- x >` (ends with '>') and identifies a comment,
        // then skip_to_comment_end hits EOF.
        let xml = "<a><!-- x > unclosed";
        let mut reader = XmlReader::new(std::io::Cursor::new(xml));
        reader.set_trim_text(true);
        let e1 = reader.next_event().unwrap();
        assert!(matches!(e1, XmlEvent::StartElement { name, .. } if name == "a"));
        let e2 = reader.next_event();
        assert!(e2.is_err());
        let msg = format!("{}", e2.unwrap_err());
        assert!(msg.contains("EOF inside comment"));
    }
}
