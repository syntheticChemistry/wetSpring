// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(clippy::expect_used, clippy::unwrap_used)]

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
    let XmlEvent::StartElement { name, attrs } = &events[0] else {
        panic!("events[0] should be StartElement");
    };
    assert_eq!(name, "item");
    assert_eq!(attrs, &[("key".to_owned(), "value".to_owned())]);
    assert!(matches!(&events[1], XmlEvent::EndElement { name } if name == "item"));
}

#[test]
fn multiple_attributes() {
    let events = events_from(r#"<cv accession="MS:1000511" value="1"/>"#);
    let XmlEvent::StartElement { attrs, .. } = &events[0] else {
        panic!("events[0] should be StartElement");
    };
    assert_eq!(attrs.len(), 2);
    assert_eq!(attrs[0], ("accession".to_owned(), "MS:1000511".to_owned()));
    assert_eq!(attrs[1], ("value".to_owned(), "1".to_owned()));
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
    let XmlEvent::StartElement { attrs, .. } = &events[0] else {
        panic!("events[0] should be StartElement");
    };
    assert_eq!(attrs[0].1, "a&b");
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
    let XmlEvent::StartElement { attrs, .. } = &events[0] else {
        panic!("events[0] should be StartElement");
    };
    assert_eq!(attrs.len(), 1);
    assert_eq!(attrs[0].0, "name");
    assert_eq!(attrs[0].1, "value");
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
    let XmlEvent::Text(t) = e2 else {
        panic!("e2 should be Text");
    };
    assert_eq!(t, "  &hello  ");
}

#[test]
fn malformed_attribute_no_equals() {
    let events = events_from("<a broken/>");
    let XmlEvent::StartElement { name, attrs } = &events[0] else {
        panic!("events[0] should be StartElement");
    };
    assert_eq!(name, "a");
    assert!(attrs.is_empty());
}

#[test]
fn malformed_attribute_no_value() {
    let events = events_from("<a key=/>");
    let XmlEvent::StartElement { name, attrs } = &events[0] else {
        panic!("events[0] should be StartElement");
    };
    assert_eq!(name, "a");
    assert!(attrs.is_empty());
}

#[test]
fn malformed_attribute_no_quote() {
    let events = events_from("<a key=noquote/>");
    let XmlEvent::StartElement { name, attrs } = &events[0] else {
        panic!("events[0] should be StartElement");
    };
    assert_eq!(name, "a");
    assert!(attrs.is_empty());
}

#[test]
fn malformed_attribute_unclosed_quote() {
    let events = events_from(r#"<a key="unclosed/>"#);
    let XmlEvent::StartElement { name, attrs } = &events[0] else {
        panic!("events[0] should be StartElement");
    };
    assert_eq!(name, "a");
    assert!(attrs.is_empty());
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
    let err = e2.unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("EOF inside comment"));
}
