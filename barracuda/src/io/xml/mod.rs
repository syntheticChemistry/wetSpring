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
use std::borrow::Cow;
use std::collections::{HashMap, VecDeque};
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
///
/// Interns element names so repeated tags (e.g. mzML's `cvParam`,
/// `spectrum`, `binaryDataArray`) share a single `String` allocation
/// rather than allocating on every open/close tag.
pub struct XmlReader<R> {
    reader: R,
    trim_text: bool,
    queue: VecDeque<XmlEvent>,
    text_buf: Vec<u8>,
    tag_buf: Vec<u8>,
    eof: bool,
    name_pool: HashMap<Box<str>, String>,
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
            name_pool: HashMap::with_capacity(32),
        }
    }

    /// Return a `String` for `name`, reusing a prior allocation if the
    /// same element name was seen before. This avoids per-tag allocation
    /// for the ~6 element names that dominate mzML files.
    fn intern_name(&mut self, name: &str) -> String {
        if let Some(existing) = self.name_pool.get(name) {
            return existing.clone();
        }
        let owned = name.to_owned();
        self.name_pool.insert(name.into(), owned.clone());
        owned
    }

    /// Enable or disable whitespace trimming of text events.
    pub const fn set_trim_text(&mut self, trim: bool) {
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

            let text_owned = if self.text_buf.is_empty() {
                None
            } else {
                let raw = String::from_utf8_lossy(&self.text_buf);
                if self.trim_text {
                    let trimmed = raw.trim();
                    if trimmed.is_empty() {
                        None
                    } else {
                        Some(xml_unescape(trimmed).into_owned())
                    }
                } else {
                    Some(xml_unescape(&raw).into_owned())
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
                let raw = std::str::from_utf8(&self.tag_buf[1..])
                    .map_err(|e| Error::Xml(format!("invalid UTF-8 in end tag: {e}")))?
                    .trim()
                    .to_owned();
                let name = self.intern_name(&raw);
                self.queue.push_back(XmlEvent::EndElement { name });
                return Ok(self.queue.pop_front().unwrap_or(XmlEvent::Eof));
            }

            // Start or self-closing element — parse name and attrs from
            // tag_buf, then release the borrow before interning.
            let tag_str = std::str::from_utf8(&self.tag_buf)
                .map_err(|e| Error::Xml(format!("invalid UTF-8 in tag: {e}")))?;
            let is_empty = tag_str.trim_end().ends_with('/');
            let body = if is_empty {
                tag_str
                    .trim_end()
                    .strip_suffix('/')
                    .unwrap_or(tag_str)
                    .trim()
            } else {
                tag_str.trim()
            };

            let (name_part, attr_part) = body
                .find(|c: char| c.is_ascii_whitespace())
                .map_or((body, ""), |pos| (&body[..pos], body[pos..].trim()));

            let name_owned = name_part.to_owned();
            let attrs = parse_attributes(attr_part);

            let name = self.intern_name(&name_owned);

            if is_empty {
                self.queue.push_back(XmlEvent::StartElement {
                    name: name.clone(),
                    attrs,
                });
                self.queue.push_back(XmlEvent::EndElement { name });
            } else {
                self.queue.push_back(XmlEvent::StartElement { name, attrs });
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
    let mut attrs = Vec::with_capacity(8);
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

        attrs.push((key.to_owned(), xml_unescape(value).into_owned()));
    }

    attrs
}

/// Unescape the 5 predefined XML character entities.
///
/// Returns `Cow::Borrowed` when no `&` is present (common case for
/// machine-generated mzML), avoiding an allocation.
#[must_use]
fn xml_unescape(s: &str) -> Cow<'_, str> {
    if !s.contains('&') {
        return Cow::Borrowed(s);
    }
    Cow::Owned(
        s.replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
            .replace("&apos;", "'"),
    )
}

#[cfg(test)]
mod tests;
