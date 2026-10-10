use crate::{Result, data::Sequence, model::Transformer};
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::BTreeMap, fs, path::Path, time::Instant};

pub const DOMAIN_NAMES: [&str; 6] = [
    "english-literature",
    "german-prose",
    "rust-source",
    "synthetic-json",
    "synthetic-unicode",
    "structured",
];
const BLOCK_BYTES: u64 = 8192;
const BOOTSTRAPS: usize = 1000;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Span {
    pub start: u64,
    pub end: u64,
    pub domain: String,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct DomainMetric {
    pub domain: String,
    pub available: bool,
    pub target_bytes: u64,
    pub tokens: u64,
    pub nll: f64,
    pub bits_per_byte: f64,
    pub blocks: usize,
    pub ci95_bits_per_byte: Option<[f64; 2]>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Report {
    pub split: String,
    pub global_bits_per_byte: f64,
    pub global_target_bytes: u64,
    pub global_tokens: u64,
    pub seconds: f64,
    pub metrics: Vec<DomainMetric>,
    pub aggregation_matches_global: bool,
}
#[derive(Clone, Copy, Debug, Default)]
struct Cell {
    nll: f64,
    bytes: u64,
    tokens: u64,
}
#[derive(Default)]
struct Totals {
    global: Cell,
    domains: BTreeMap<String, Cell>,
    blocks: BTreeMap<(String, u64), Cell>,
}

pub fn domain_for_category(category: &str) -> Option<&'static str> {
    match category {
        "english-drama" | "english-prose" => Some("english-literature"),
        "german-prose" => Some("german-prose"),
        "code" => Some("rust-source"),
        "synthetic-json" => Some("synthetic-json"),
        "synthetic-unicode" => Some("synthetic-unicode"),
        _ => None,
    }
}
pub fn spans_from_manifest(path: &Path, split: &str) -> Result<Vec<Span>> {
    if !["validation", "test"].contains(&split) {
        return Err("domain evaluation is held-out only".into());
    }
    let root: Value = serde_json::from_slice(&fs::read(path)?)?;
    let sources = root["sources"].as_array().ok_or("manifest sources")?;
    let mut category = BTreeMap::new();
    for source in sources {
        let name = source["name"].as_str().ok_or("manifest source name")?;
        let value = source["category"]
            .as_str()
            .ok_or("manifest source category")?;
        category.insert(name.to_owned(), value.to_owned());
    }
    let units = root["units"].as_array().ok_or("manifest units")?;
    let mut spans = Vec::new();
    let mut offset = 0_u64;
    for unit in units {
        if unit["proposed"].as_str() != Some(split) || unit["kept"].as_bool() != Some(true) {
            continue;
        }
        let source = unit["source"].as_str().ok_or("manifest unit source")?;
        let raw_category = category.get(source).ok_or("unknown manifest source")?;
        let domain = domain_for_category(raw_category)
            .ok_or("unexpected retained held-out category (structured is declared unavailable)")?
            .to_owned();
        let start = unit["start"].as_u64().ok_or("manifest unit start")?;
        let end = unit["end"].as_u64().ok_or("manifest unit end")?;
        if end <= start {
            return Err("empty manifest domain unit".into());
        }
        let len = end - start;
        spans.push(Span {
            start: offset,
            end: offset + len,
            domain,
        });
        offset += len;
    }
    let split_row = root["splits"]
        .as_array()
        .ok_or("manifest splits")?
        .iter()
        .find(|row| row["split"].as_str() == Some(split))
        .ok_or("manifest split missing")?;
    let expected = split_row["bytes"]
        .as_u64()
        .ok_or("manifest split byte count")?;
    if offset != expected {
        return Err(format!("domain span coverage {offset} != split {expected}").into());
    }
    Ok(spans)
}
fn category_of_range(spans: &[Span], start: u64, end: u64, cursor: &mut usize) -> Result<String> {
    while *cursor < spans.len() && spans[*cursor].end <= start {
        *cursor += 1;
    }
    let first = spans
        .get(*cursor)
        .filter(|s| s.start <= start && s.end > start)
        .ok_or("unmapped target byte")?;
    let domain = first.domain.as_str();
    let mut i = *cursor;
    let mut covered = first.end.min(end);
    while covered < end {
        i += 1;
        let next = spans
            .get(i)
            .filter(|s| s.start == covered)
            .ok_or("domain span gap inside token")?;
        if next.domain != domain {
            return Ok("mixed-domain".into());
        }
        covered = next.end.min(end);
    }
    Ok(domain.to_owned())
}
fn add(t: &mut Totals, domain: &str, block: u64, nll: f64, bytes: u64) {
    for cell in [
        &mut t.global,
        t.domains.entry(domain.into()).or_default(),
        t.blocks.entry((domain.into(), block)).or_default(),
    ] {
        cell.nll += nll;
        cell.bytes += bytes;
        cell.tokens += 1;
    }
}
fn bits(cell: Cell) -> Option<f64> {
    (cell.bytes > 0 && cell.nll.is_finite())
        .then(|| cell.nll / cell.bytes as f64 / std::f64::consts::LN_2)
}
fn bootstrap_ci(blocks: &[(u64, Cell)], seed: u64) -> Option<[f64; 2]> {
    if blocks.len() < 2 {
        return None;
    }
    let mut state = if seed == 0 { 0x9e3779b97f4a7c15 } else { seed };
    let mut values = Vec::with_capacity(BOOTSTRAPS);
    for _ in 0..BOOTSTRAPS {
        let mut nll = 0.0;
        let mut bytes = 0_u64;
        for _ in 0..blocks.len() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let c = blocks[(state as usize) % blocks.len()].1;
            nll += c.nll;
            bytes += c.bytes;
        }
        if bytes > 0 {
            values.push(nll / bytes as f64 / std::f64::consts::LN_2);
        }
    }
    values.sort_by(f64::total_cmp);
    Some([
        values[(values.len() * 25) / 1000],
        values[((values.len() * 975) / 1000).min(values.len() - 1)],
    ])
}
fn report(split: &str, totals: Totals, seconds: f64) -> Result<Report> {
    let global = totals.global;
    if global.bytes == 0 || global.tokens == 0 {
        return Err("empty domain score".into());
    }
    let mut metrics = Vec::new();
    for domain in DOMAIN_NAMES.into_iter().chain(["mixed-domain"]) {
        let Some(cell) = totals.domains.get(domain).copied() else {
            metrics.push(DomainMetric {
                domain: domain.into(),
                available: false,
                ..DomainMetric::default()
            });
            continue;
        };
        let buckets: Vec<_> = totals
            .blocks
            .iter()
            .filter(|((name, _), _)| name == domain)
            .map(|((_, block), cell)| (*block, *cell))
            .collect();
        let seed = domain
            .bytes()
            .fold(0_u64, |h, b| h.rotate_left(5) ^ u64::from(b));
        metrics.push(DomainMetric {
            domain: domain.into(),
            available: true,
            target_bytes: cell.bytes,
            tokens: cell.tokens,
            nll: cell.nll,
            bits_per_byte: bits(cell).ok_or("domain has no bytes")?,
            blocks: buckets.len(),
            ci95_bits_per_byte: bootstrap_ci(&buckets, seed),
        });
    }
    let cat = totals.domains.values().fold(Cell::default(), |mut a, c| {
        a.nll += c.nll;
        a.bytes += c.bytes;
        a.tokens += c.tokens;
        a
    });
    // The global and per-domain NLL sums visit the same token losses in a
    // different order. Allow a tight relative FP64 accumulation tolerance;
    // byte and token accounting must still reconcile exactly.
    let nll_tolerance = 1e-12 * global.nll.abs().max(cat.nll.abs()).max(1.0);
    let consistent = cat.bytes == global.bytes
        && cat.tokens == global.tokens
        && (cat.nll - global.nll).abs() <= nll_tolerance;
    if !consistent {
        return Err("domain/category aggregation does not reconcile to global score".into());
    }
    Ok(Report {
        split: split.into(),
        global_bits_per_byte: bits(global).ok_or("no global metric")?,
        global_target_bytes: global.bytes,
        global_tokens: global.tokens,
        seconds,
        metrics,
        aggregation_matches_global: consistent,
    })
}
pub fn evaluate(
    model: &Transformer,
    seq: &Sequence,
    spans: &[Span],
    split: &str,
    device: &Device,
) -> Result<Report> {
    if !["validation", "test"].contains(&split) {
        return Err("domain evaluation is held-out only".into());
    }
    seq.validate()?;
    let raw_bytes: u64 = seq.bytes.iter().map(|&b| u64::from(b)).sum();
    if spans.first().map(|s| s.start) != Some(0)
        || spans.last().map(|s| s.end) != Some(raw_bytes)
        || spans.windows(2).any(|w| w[0].end != w[1].start)
    {
        return Err("manifest spans do not exactly cover encoded split".into());
    }
    let mut offsets = Vec::with_capacity(seq.bytes.len() + 1);
    offsets.push(0_u64);
    for &n in &seq.bytes {
        offsets.push(offsets.last().copied().unwrap() + u64::from(n));
    }
    let start_time = Instant::now();
    let mut totals = Totals::default();
    let mut span_cursor = 0;
    let mut start = 0;
    while start + 1 < seq.tokens.len() {
        let context = model.config.context.min(seq.tokens.len() - 1 - start);
        let x = Tensor::from_vec(
            seq.tokens[start..start + context].to_vec(),
            (1, context),
            device,
        )?;
        let logits = model.forward(&x)?.to_vec3::<f32>()?;
        for (j, row) in logits[0].iter().enumerate().take(context) {
            let target_index = start + j + 1;
            let target = seq.tokens[target_index] as usize;
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let log_z = row
                .iter()
                .map(|&z| f64::from(z - max).exp())
                .sum::<f64>()
                .ln()
                + f64::from(max);
            let nll = log_z - f64::from(row[target]);
            if !nll.is_finite() || nll < 0.0 {
                return Err("invalid domain-evaluation token loss".into());
            }
            let a = offsets[target_index];
            let b = offsets[target_index + 1];
            let domain = category_of_range(spans, a, b, &mut span_cursor)?;
            add(&mut totals, &domain, a / BLOCK_BYTES, nll, b - a);
        }
        start += context;
    }
    report(split, totals, start_time.elapsed().as_secs_f64())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn review_evaluation_window_and_batch_boundaries_preserve_bytes_and_scores() -> Result<()> {
        let model = Transformer::new(crate::runner::tiny_config(), 82, &Device::Cpu)?;
        for len in [2, 8, 9, 16, 17, 24, 25] {
            let seq = Sequence {
                tokens: (0..len).map(|i| (i % 31) as u32).collect(),
                bytes: (0..len).map(|i| (i % 4 + 1) as u32).collect(),
            };
            let total = seq.target_bytes(0, len)?;
            let spans = [
                Span {
                    start: 0,
                    end: 3,
                    domain: "english-literature".into(),
                },
                Span {
                    start: 3,
                    end: total,
                    domain: "german-prose".into(),
                },
            ];
            let stratified = evaluate(&model, &seq, &spans, "test", &Device::Cpu)?;
            for batch_size in [1, 3, 8] {
                let global = crate::runner::evaluate(&model, &seq, batch_size, &Device::Cpu)?;
                assert_eq!(global.targets, (len - 1) as u64);
                assert_eq!(global.raw_target_bytes, total - u64::from(seq.bytes[0]));
                assert_eq!(global.targets, stratified.global_tokens);
                assert_eq!(global.raw_target_bytes, stratified.global_target_bytes);
                assert!((global.bits_per_byte - stratified.global_bits_per_byte).abs() < 1e-5);
            }
        }
        Ok(())
    }

    #[test]
    fn tokens_crossing_units_keep_domain_and_crossing_categories_are_mixed() -> Result<()> {
        let spans = vec![
            Span {
                start: 0,
                end: 2,
                domain: "english-literature".into(),
            },
            Span {
                start: 2,
                end: 5,
                domain: "english-literature".into(),
            },
            Span {
                start: 5,
                end: 9,
                domain: "german-prose".into(),
            },
        ];
        let mut cursor = 0;
        assert_eq!(
            category_of_range(&spans, 1, 4, &mut cursor)?,
            "english-literature"
        );
        assert_eq!(
            category_of_range(&spans, 4, 7, &mut cursor)?,
            "mixed-domain"
        );
        Ok(())
    }
    #[test]
    fn heldout_manifest_reports_structured_as_unavailable_and_covers_bytes() -> Result<()> {
        let manifest = PathBuf::from("provenance/corpus-v2-manifest.json");
        let spans = spans_from_manifest(&manifest, "validation")?;
        let bytes = spans.last().ok_or("no spans")?.end;
        assert_eq!(bytes, 1_459_716);
        assert!(!spans.iter().any(|s| s.domain == "structured"));
        assert_eq!(domain_for_category("structured"), None);
        assert_eq!(
            domain_for_category("english-drama"),
            Some("english-literature")
        );
        assert_eq!(domain_for_category("code"), Some("rust-source"));
        assert!(spans_from_manifest(&manifest, "test").is_ok());
        Ok(())
    }
    #[test]
    fn stratified_category_aggregation_reconciles_exactly() -> Result<()> {
        let mut totals = Totals::default();
        add(&mut totals, "english-literature", 0, 1.0, 2);
        add(&mut totals, "german-prose", 1, 2.0, 4);
        add(&mut totals, "mixed-domain", 2, 0.5, 1);
        let report = report("validation", totals, 0.0)?;
        assert!(report.aggregation_matches_global);
        assert_eq!(report.global_tokens, 3);
        assert_eq!(report.global_target_bytes, 7);
        assert!(
            report
                .metrics
                .iter()
                .any(|m| m.domain == "structured" && !m.available && m.tokens == 0)
        );
        assert!(
            report
                .metrics
                .iter()
                .any(|m| m.domain == "mixed-domain" && m.tokens == 1)
        );
        Ok(())
    }
    #[test]
    fn domain_aggregation_rejects_real_nll_and_counter_mismatches() {
        let mut nll_mismatch = Totals::default();
        add(&mut nll_mismatch, "english-literature", 0, 1.0, 2);
        nll_mismatch.global.nll += 0.01;
        assert!(report("validation", nll_mismatch, 0.0).is_err());

        let mut byte_mismatch = Totals::default();
        add(&mut byte_mismatch, "english-literature", 0, 1.0, 2);
        byte_mismatch.global.bytes += 1;
        assert!(report("validation", byte_mismatch, 0.0).is_err());

        let mut token_mismatch = Totals::default();
        add(&mut token_mismatch, "english-literature", 0, 1.0, 2);
        token_mismatch.global.tokens += 1;
        assert!(report("validation", token_mismatch, 0.0).is_err());
    }

    #[test]
    fn large_domain_aggregation_allows_only_fp64_ordering_error() -> Result<()> {
        let mut totals = Totals::default();
        for i in 0_u64..800_000 {
            let jitter = ((i.wrapping_mul(2_654_435_761) & 0xffff) as f64) / 65_535.0 * 1e-6;
            let nll = std::f64::consts::LN_2 * 9.0 + jitter;
            let domain = if i % 7 == 0 {
                "german-prose"
            } else {
                "english-literature"
            };
            add(&mut totals, domain, i / BLOCK_BYTES, nll, 1);
        }
        let result = report("validation", totals, 0.0)?;
        assert!(result.aggregation_matches_global);
        Ok(())
    }

    #[test]
    fn per_domain_forward_scoring_matches_global_byte_score() -> Result<()> {
        use crate::runner::{evaluate as global_evaluate, tiny_config};
        let model = Transformer::new(tiny_config(), 41, &Device::Cpu)?;
        let seq = Sequence {
            tokens: vec![1, 2, 3, 4, 5, 6],
            bytes: vec![1, 2, 3, 1, 1, 1],
        };
        let spans = vec![
            Span {
                start: 0,
                end: 4,
                domain: "english-literature".into(),
            },
            Span {
                start: 4,
                end: 9,
                domain: "german-prose".into(),
            },
        ];
        let actual = evaluate(&model, &seq, &spans, "validation", &Device::Cpu)?;
        let expected = global_evaluate(&model, &seq, 1, &Device::Cpu)?;
        assert!((actual.global_bits_per_byte - expected.bits_per_byte).abs() < 1e-4);
        assert!(actual.aggregation_matches_global);
        assert_eq!(actual.global_tokens, 5);
        assert_eq!(actual.global_target_bytes, 8);
        assert!(
            actual
                .metrics
                .iter()
                .any(|m| m.domain == "mixed-domain" && m.available && m.target_bytes == 3)
        );
        assert!(
            actual
                .metrics
                .iter()
                .any(|m| m.domain == "structured" && !m.available)
        );
        Ok(())
    }
}
