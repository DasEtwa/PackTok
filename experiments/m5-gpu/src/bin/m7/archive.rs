//! Bounded local intake for the two pinned historical TEI collections.
use crate::{
    corpus,
    source::{self, Catalog, Format, RightsDecision, Source, Split},
};
use packtok_m5::{Result, hash};
use quick_xml::{Reader, events::Event};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    fs::{self, File},
    io::{BufWriter, Read, Write},
    path::{Path, PathBuf},
};
use zip::ZipArchive;

const MAX_XML: usize = 32 << 20;
const DTA_URL: &str =
    "https://registry.text-plus.org/doc/collection/c582ca78-26b4-4e5a-b817-e2d7c04f0a9d";
const EEBO_URL: &str = "https://ora.ox.ac.uk/objects/uuid:ad7da8fc-cd8e-4637-8b7c-99498436dbaa";

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
enum Family {
    Dta,
    Eebo,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Input {
    family: Family,
    path: PathBuf,
    url: String,
    version: String,
    acquired: String,
    bytes: u64,
    sha256: String,
}

#[derive(Default)]
struct Text {
    author: String,
    authority: String,
    title: String,
    category: String,
    body: String,
    license: bool,
    language: Option<bool>,
}

fn decode(input: &[u8], family: Family) -> Result<Text> {
    let input = std::str::from_utf8(input)?;
    let mut reader = Reader::from_str(input.trim_start_matches('\u{feff}'));
    reader.config_mut().expand_empty_elements = true;
    let mut out = Text::default();
    let mut depth = 0_usize;
    let mut body = None;
    let mut ignored = None;
    let mut capture = None;
    let mut in_author = None;
    loop {
        match reader.read_event()? {
            Event::Start(el) => {
                let local = el.local_name();
                let name = local.as_ref();
                depth += 1;
                let attr = |key: &[u8]| -> Result<Option<String>> {
                    for item in el.attributes().with_checks(true) {
                        let item = item?;
                        if item.key.as_ref().eq_ignore_ascii_case(key) {
                            return Ok(Some(
                                item.decode_and_unescape_value(reader.decoder())?
                                    .into_owned(),
                            ));
                        }
                    }
                    Ok(None)
                };
                if name.eq_ignore_ascii_case(b"text") && body.is_none() {
                    if let Family::Eebo = family
                        && let Some(lang) = attr(b"lang")?
                    {
                        out.language = Some(["eng", "en"].contains(&lang.as_str()));
                    }
                    body = Some(depth);
                } else if body.is_some() {
                    if ignored.is_none()
                        && [b"note".as_slice(), b"figure", b"fw"]
                            .iter()
                            .any(|t| name.eq_ignore_ascii_case(t))
                    {
                        ignored = Some(depth);
                    }
                    if ignored.is_none()
                        && [
                            b"p".as_slice(),
                            b"head",
                            b"l",
                            b"lb",
                            b"pb",
                            b"item",
                            b"div",
                            b"ab",
                        ]
                        .iter()
                        .any(|t| name.eq_ignore_ascii_case(t))
                        && !out.body.ends_with('\n')
                    {
                        out.body.push('\n');
                    }
                } else if name.eq_ignore_ascii_case(b"licence")
                    || name.eq_ignore_ascii_case(b"license")
                {
                    out.license |= attr(b"target")?.is_some_and(|s| {
                        s.starts_with("https://creativecommons.org/licenses/by-sa/4.0/")
                            || s.starts_with("http://creativecommons.org/licenses/by-sa/4.0/")
                    });
                } else if name.eq_ignore_ascii_case(b"language")
                    || name.eq_ignore_ascii_case(b"langusage")
                {
                    if let Some(id) = attr(b"ident")?.or(attr(b"id")?) {
                        let good = match family {
                            Family::Dta => ["de", "deu", "ger"].contains(&id.as_str()),
                            Family::Eebo => ["en", "eng"].contains(&id.as_str()),
                        };
                        out.language = Some(out.language.unwrap_or(false) || good);
                    }
                } else if capture.is_none()
                    && out.author.trim().is_empty()
                    && name.eq_ignore_ascii_case(b"author")
                {
                    capture = Some((depth, 0));
                    in_author = Some(depth);
                } else if in_author.is_some()
                    && name.eq_ignore_ascii_case(b"persname")
                    && out.authority.is_empty()
                {
                    out.authority = attr(b"ref")?.unwrap_or_default();
                } else if capture.is_none()
                    && out.title.trim().is_empty()
                    && name.eq_ignore_ascii_case(b"title")
                {
                    capture = Some((depth, 1));
                } else if capture.is_none()
                    && out.category.is_empty()
                    && name.eq_ignore_ascii_case(b"classCode")
                    && attr(b"scheme")?.is_some_and(|s| s.ends_with("#dtamain"))
                {
                    capture = Some((depth, 2));
                }
            }
            Event::End(el) => {
                if capture.is_some_and(|(d, _)| d == depth) {
                    capture = None;
                }
                if in_author == Some(depth) {
                    in_author = None;
                }
                if ignored == Some(depth) {
                    ignored = None;
                }
                if body == Some(depth) && el.local_name().as_ref().eq_ignore_ascii_case(b"text") {
                    break;
                }
                if body.is_some()
                    && ignored.is_none()
                    && [b"p".as_slice(), b"l", b"div", b"head"]
                        .iter()
                        .any(|tag| el.local_name().as_ref().eq_ignore_ascii_case(tag))
                    && !out.body.ends_with('\n')
                {
                    out.body.push('\n');
                }
                depth = depth.checked_sub(1).ok_or("invalid XML nesting")?;
            }
            Event::Text(t) => {
                let s = t.xml_content()?;
                if body.is_some() && ignored.is_none() {
                    out.body.push_str(&s);
                } else if let Some((_, kind)) = capture {
                    match kind {
                        0 => out.author.push_str(&s),
                        1 => out.title.push_str(&s),
                        _ => out.category.push_str(&s),
                    }
                }
            }
            Event::CData(t) if body.is_some() && ignored.is_none() => {
                out.body.push_str(&t.decode()?)
            }
            Event::GeneralRef(r) if body.is_some() && ignored.is_none() => {
                if let Some(c) = r.resolve_char_ref()? {
                    out.body.push(c);
                } else {
                    out.body.push_str(match r.decode()?.as_ref() {
                        "amp" => "&",
                        "lt" => "<",
                        "gt" => ">",
                        "quot" => "\"",
                        "apos" => "'",
                        _ => return Err("unsupported named entity in body".into()),
                    })
                }
            }
            Event::Eof => return Err("missing closed text element".into()),
            _ => {}
        }
        if out.body.len() > corpus::MAX_DOC {
            return Err("oversized extracted document".into());
        }
    }
    Ok(out)
}

fn collapse(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}
fn split(group: &str) -> Split {
    let h = packtok_train::fnv1a64(format!("liv-m7-historical-split-v1/{group}").as_bytes()) % 100;
    match h {
        0..=79 => Split::Train,
        80..=89 => Split::Validation,
        _ => Split::Test,
    }
}

pub fn import(plan: &Path, out: &Path) -> Result<()> {
    let inputs: Vec<Input> = corpus::json(plan)?;
    if inputs.is_empty() {
        return Err("no licensed archives selected".into());
    }
    let work = corpus::stage(out)?;
    let cache = work.join("cache");
    fs::create_dir(&cache)?;
    let mut ledger = BufWriter::new(File::create_new(work.join("members.jsonl"))?);
    let mut sources = Vec::new();
    let mut ids = BTreeSet::new();
    for input in &inputs {
        let approved_url = match input.family {
            Family::Dta => {
                input.url == "https://media.dwds.de/dta/download/dta_kernkorpus_2020-07-20.zip"
            }
            Family::Eebo => input
                .url
                .strip_prefix(EEBO_URL)
                .and_then(|tail| tail.strip_prefix("/files/m"))
                .is_some_and(|id| id.len() == 32 && id.bytes().all(|x| x.is_ascii_hexdigit())),
        };
        if !approved_url || input.version.trim().is_empty() || input.acquired.trim().is_empty() {
            return Err("archive source is outside the reviewed publisher release".into());
        }
        if source::digest(&input.path)? != (input.bytes, input.sha256.clone()) {
            return Err("archive size/hash mismatch".into());
        }
        let mut zip = ZipArchive::new(File::open(&input.path)?)?;
        let mut order: Vec<_> = (0..zip.len())
            .filter_map(|idx| {
                zip.by_index_raw(idx)
                    .ok()
                    .map(|f| (f.name().to_owned(), idx))
            })
            .collect();
        order.sort_unstable();
        for (member, idx) in order {
            if !member.to_ascii_lowercase().ends_with(".xml") {
                continue;
            }
            let mut item = zip.by_index(idx)?;
            let (label, lang, publisher, origin) = match input.family {
                Family::Dta => ("dta", "de", "BBAW Deutsches Textarchiv", DTA_URL),
                Family::Eebo => (
                    "eebo",
                    "en",
                    "Text Creation Partnership / Oxford Research Archive",
                    EEBO_URL,
                ),
            };
            let id = format!("{label}-{}", &hash(member.as_bytes())[..24]);
            let mut raw = Vec::new();
            let mut row = serde_json::json!({"archive_sha256":input.sha256,"member":member,"member_bytes":item.size(),"source":id});
            if item.size() <= MAX_XML as u64 {
                (&mut item).take(MAX_XML as u64 + 1).read_to_end(&mut raw)?;
                if raw.len() > MAX_XML {
                    return Err("archive member exceeded declared limit".into());
                }
                row["member_sha256"] = hash(&raw).into();
                match decode(&raw, input.family) {
                    Ok(t)
                        if (matches!(input.family, Family::Eebo) || t.license)
                            && t.language != Some(false)
                            && !t.body.trim().is_empty() =>
                    {
                        let author = collapse(&t.author);
                        let group_key = if !t.authority.is_empty() {
                            t.authority
                        } else if !author.is_empty() {
                            author.to_lowercase()
                        } else {
                            collapse(&t.title).to_lowercase()
                        };
                        let group = if group_key.is_empty() {
                            id.clone()
                        } else {
                            hash(group_key.as_bytes())
                        };
                        if !ids.insert(id.clone()) {
                            return Err("duplicate source archive member".into());
                        }
                        let split = split(&group);
                        let category = if t.category.is_empty() {
                            format!("{label}-historical-print")
                        } else {
                            collapse(&t.category)
                        };
                        let body = t.body.trim_matches(['\t', '\r', '\n', ' ']);
                        let path = cache.join(&id);
                        packtok_m5::write_new(&path, body.as_bytes())?;
                        let rights = match input.family {
                            Family::Dta => "CC BY-SA 4.0",
                            Family::Eebo => {
                                "Public-domain TCP transcriptions; rights released by the publisher"
                            }
                        };
                        row["decision"] = "imported".into();
                        row["title"] = collapse(&t.title).into();
                        row["author"] = author.clone().into();
                        row["group"] = group.clone().into();
                        row["split"] = split.name().into();
                        row["processed_sha256"] = hash(body.as_bytes()).into();
                        row["processed_bytes"] = body.len().into();
                        sources.push(Source {id,group,split,language:lang.into(),category,format:Format::Text,
                            author:if author.is_empty(){"Unattributed in source".into()}else{author},
                            publisher:publisher.into(),version:input.version.clone(),url:input.url.clone(),download_url:None,
                            acquired:input.acquired.clone(),rights:rights.into(),rights_decision:RightsDecision::RedistributionCleared,
                            evidence:format!("{origin} ; {}",if matches!(input.family,Family::Dta) {
                                "individual TEI licence target recorded as CC BY-SA 4.0"
                            } else {"https://www.textpartnership.net/pages/faq.html"}),
                            attribution:format!("Credit {publisher}, original author and archive member. Extracted, markup removed. Preserve applicable source license and credit."),
                            jurisdiction:"Local German research. Downstream distribution follows the source rights and license.".into(),
                            bytes:body.len() as u64,sha256:hash(body.as_bytes())});
                    }
                    Ok(_) => {
                        row["decision"] = "rejected-rights-language-or-empty".into();
                    }
                    Err(error) => {
                        row["decision"] = "rejected-xml".into();
                        row["error"] = error.to_string().into();
                    }
                }
            } else {
                row["decision"] = "oversized-xml".into();
            }
            serde_json::to_writer(&mut ledger, &row)?;
            ledger.write_all(b"\n")?;
        }
    }
    if sources.is_empty() {
        return Err("no eligible TEI documents".into());
    }
    ledger.flush()?;
    ledger.get_ref().sync_all()?;
    File::open(&cache)?.sync_all()?;
    corpus::write_json(&work.join("archives.json"), &inputs)?;
    corpus::write_json(
        &work.join("catalog.json"),
        &Catalog {
            version: 1,
            sources,
        },
    )?;
    source::load(&work.join("catalog.json"))?;
    corpus::publish(&work, out)
}

#[cfg(test)]
#[test]
fn extracts_body_and_retains_historical_rights_notice() -> Result<()> {
    let xml=br#"<TEI><teiHeader><author><persName ref="gnd">A Author</persName></author><title>Book</title><licence target="https://creativecommons.org/licenses/by-sa/4.0/"/><language ident="deu">Deutsch</language></teiHeader><text><body><p>Eins &amp; zwei<note>editor</note><lb/>drei</p><p>vier</p></body></text></TEI>"#;
    let text = decode(xml, Family::Dta)?;
    assert!(text.license);
    assert_eq!(text.language, Some(true));
    assert_eq!(text.author, "A Author");
    assert_eq!(text.authority, "gnd");
    assert_eq!(text.body.trim(), "Eins & zwei\ndrei\nvier");
    Ok(())
}
