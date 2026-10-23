//! Matched, non-learning Semantic Packs v1 experiment. No gold data trains BPE.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::hint::black_box;
use std::mem::size_of;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

use packtok_format::semantic::{SEMANTIC_SIDECAR_VERSION, SemanticSidecar};
use packtok_packs::semantic::{SemanticAnnotation, SemanticDecision, SemanticId, SemanticRegistry};
use packtok_tokenizer::{BpeTokenizer, Tokenizer, encode_with_semantics};
use packtok_train::{BpeTrainingConfig, fnv1a64, load_corpus, train_bpe_with_provenance};

use crate::metrics::process_peak_memory_bytes;

const EVAL_PATH: &str = "fixtures/semantic-v1/eval.tsv";
const TRAIN_PATH: &str = "fixtures/benchmark/train.txt";
const TRIALS: usize = 7;
const WARMUP: Duration = Duration::from_millis(25);
const MEASURE: Duration = Duration::from_millis(100);
const REPEATS: usize = 64;

#[derive(Clone, Debug, Eq, PartialEq)]
struct Gold {
    start: usize,
    end: usize,
    concept: Option<SemanticId>,
}

#[derive(Debug)]
struct Case {
    id: String,
    category: String,
    text: String,
    gold: Vec<Gold>,
}

#[derive(Default, Debug, Eq, PartialEq)]
struct Accuracy {
    expected_spans: usize,
    actual_spans: usize,
    matching_spans: usize,
    expected_resolved: usize,
    actual_resolved: usize,
    correct_resolved: usize,
    expected_ambiguous: usize,
    correctly_ambiguous: usize,
    false_resolutions_of_ambiguity: usize,
    unknown_cases: usize,
    unknown_cases_with_annotations: usize,
}

#[derive(Debug)]
struct Timing {
    trial: usize,
    variant: &'static str,
    direction: &'static str,
    iterations: u64,
    elapsed_ns: u128,
    bytes: usize,
    tokens: usize,
}

impl Timing {
    fn mib_per_second(&self) -> f64 {
        self.bytes as f64 * self.iterations as f64 * 1e9
            / self.elapsed_ns as f64
            / (1024.0 * 1024.0)
    }
}

/// Runs from repository root and refuses to overwrite an existing result directory.
pub fn run(label: &str) -> Result<(), Box<dyn Error>> {
    if label.is_empty()
        || label.len() > 80
        || !label
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    {
        return Err("semantic run label must contain 1..80 ASCII letters/digits/-/_".into());
    }
    let parent = Path::new("experiments/semantic-packs-v1");
    fs::create_dir_all(parent)?;
    let directory = parent.join(label);
    fs::create_dir(&directory)?;
    // Capture provenance before any measurements. Failed runs retain this record.
    fs::write(directory.join("environment.txt"), environment())?;
    let result = run_in(&directory);
    if let Err(error) = &result {
        fs::write(directory.join("FAILED.txt"), format!("{error}\n"))?;
    }
    result
}

fn run_in(directory: &Path) -> Result<(), Box<dyn Error>> {
    let fixture = fs::read_to_string(EVAL_PATH)?;
    let cases = parse_cases(&fixture)?;
    let registry = SemanticRegistry::builtin();
    let expected_registry_identity = registry.identity();
    let corpus = load_corpus(Path::new(TRAIN_PATH))?;
    let config = BpeTrainingConfig::default();
    let artifact = train_bpe_with_provenance(corpus.bytes(), config, corpus.provenance())?;
    let artifact_bytes = artifact.to_bytes()?;
    fs::write(directory.join("m1.packtok"), &artifact_bytes)?;
    let tokenizer = BpeTokenizer::from_artifact(&artifact)?;
    let mut accuracy = Accuracy::default();
    let mut case_csv = String::from(
        "case_id\tcategory\ttext_bytes\tlexical_tokens\tgold_spans\tactual_spans\tannotations_equal\troundtrip\tdeterministic\tsidecar_bytes\tannotation_capacity_bytes\tregistry_namespace_capacity_bytes\n",
    );
    let mut decisions = String::new();
    let mut sidecar_total = 0;
    let mut annotation_capacity_total = 0;
    let mut registry_namespace_capacity_total = 0;
    let mut lexical_count_total = 0;
    let mut lexical_capacity_total = 0;
    for case in &cases {
        let lexical = tokenizer.encode(&case.text)?;
        let off = encode_with_semantics(&tokenizer, &case.text, None)?;
        let on = encode_with_semantics(&tokenizer, &case.text, Some(registry))?;
        let repeated = encode_with_semantics(&tokenizer, &case.text, Some(registry))?;
        if off.lexical_tokens != lexical
            || off.semantic_annotations.is_some()
            || off.semantic_registry.is_some()
            || on.semantic_registry.as_ref() != Some(&expected_registry_identity)
            || on.lexical_tokens != lexical
        {
            return Err(format!("lexical compatibility failed: {}", case.id).into());
        }
        let annotations = on
            .semantic_annotations
            .as_ref()
            .ok_or("semantic annotations absent")?;
        let roundtrip = tokenizer.decode_bytes(&on.lexical_tokens)? == case.text.as_bytes();
        let deterministic = repeated.lexical_tokens == on.lexical_tokens
            && repeated.semantic_annotations == on.semantic_annotations
            && repeated.semantic_registry == on.semantic_registry;
        if !roundtrip || !deterministic {
            return Err(format!("roundtrip/determinism failed: {}", case.id).into());
        }
        let sidecar = SemanticSidecar::annotate(&case.text, registry);
        if sidecar.annotations() != annotations {
            return Err("sidecar and pipeline annotations differ".into());
        }
        let encoded = sidecar.to_bytes()?;
        let decoded = SemanticSidecar::from_bytes(&encoded, &case.text, registry)?;
        if decoded.annotations() != annotations || decoded.to_bytes()? != encoded {
            return Err("sidecar roundtrip/determinism failed".into());
        }
        let memory = annotation_capacity_bytes(annotations);
        let registry_namespace_capacity = on
            .semantic_registry
            .as_ref()
            .ok_or("semantic registry identity absent")?
            .namespace
            .capacity();
        registry_namespace_capacity_total += registry_namespace_capacity;
        sidecar_total += encoded.len();
        annotation_capacity_total += memory;
        lexical_count_total += lexical.len();
        lexical_capacity_total += lexical.capacity() * size_of::<packtok_core::TokenId>();
        let equal = record_accuracy(&mut accuracy, case, annotations);
        writeln!(
            case_csv,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            case.id,
            case.category,
            case.text.len(),
            lexical.len(),
            case.gold.len(),
            annotations.len(),
            equal,
            roundtrip,
            deterministic,
            encoded.len(),
            memory,
            registry_namespace_capacity
        )?;
        writeln!(
            decisions,
            "{} {:?}\ngold={:?}\nactual={:?}\n",
            case.id, case.text, case.gold, annotations
        )?;
    }
    fs::write(directory.join("cases.tsv"), case_csv)?;
    fs::write(directory.join("decisions.txt"), decisions)?;
    // Newline-separated batch uses exactly the same bytes for both variants. Context
    // cannot cross record newlines under v1's conservative context policy.
    let batch = cases
        .iter()
        .map(|c| c.text.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let batch = format!("{batch}\n").repeat(REPEATS);
    let lexical = tokenizer.encode(&batch)?;
    let semantic = encode_with_semantics(&tokenizer, &batch, Some(registry))?;
    // Existing production instrumentation measures the shared lexical path once,
    // outside timed operations. It does not count semantic annotation allocations.
    let (measured_lexical, lexical_encode_buffers) =
        tokenizer.encode_bytes_with_stats(batch.as_bytes())?;
    let (measured_decoded, lexical_decode_buffers) = tokenizer.decode_bytes_with_stats(&lexical)?;
    if measured_lexical != lexical || measured_decoded != batch.as_bytes() {
        return Err("instrumented lexical encode/decode differential failed".into());
    }
    if semantic.lexical_tokens != lexical || tokenizer.decode_bytes(&lexical)? != batch.as_bytes() {
        return Err("batch lexical differential/roundtrip failed".into());
    }
    let mut timings = Vec::new();
    for trial in 0..TRIALS {
        // Alternate A/B order between trials to limit systematic ordering bias.
        let variants = if trial % 2 == 0 {
            ["A", "B"]
        } else {
            ["B", "A"]
        };
        for variant in variants {
            let encode = || {
                if variant == "A" {
                    black_box(
                        tokenizer
                            .encode(black_box(&batch))
                            .expect("validated encoding"),
                    );
                } else {
                    black_box(
                        encode_with_semantics(&tokenizer, black_box(&batch), Some(registry))
                            .expect("validated encoding"),
                    );
                }
            };
            let decode = || {
                let tokens = if variant == "A" {
                    &lexical
                } else {
                    &semantic.lexical_tokens
                };
                black_box(
                    tokenizer
                        .decode_bytes(black_box(tokens))
                        .expect("validated decoding"),
                );
            };
            timings.push(measure(
                trial,
                variant,
                "encode",
                batch.len(),
                lexical.len(),
                encode,
            ));
            timings.push(measure(
                trial,
                variant,
                "decode",
                batch.len(),
                lexical.len(),
                decode,
            ));
        }
    }
    let mut timing_csv = String::from(
        "trial\tvariant\tdirection\titerations\tinput_bytes\tlexical_tokens\telapsed_ns\tmib_per_second\n",
    );
    for timing in &timings {
        writeln!(
            timing_csv,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6}",
            timing.trial,
            timing.variant,
            timing.direction,
            timing.iterations,
            timing.bytes,
            timing.tokens,
            timing.elapsed_ns,
            timing.mib_per_second()
        )?;
    }
    fs::write(directory.join("timings.tsv"), timing_csv)?;
    let rss = process_peak_memory_bytes();
    let mut report = String::new();
    writeln!(
        report,
        "Semantic Packs v1: same flat BPE tokens, optional semantic sidecar"
    )?;
    writeln!(
        report,
        "train={TRAIN_PATH}; train_fnv1a64={:016x}; train_bytes={}; target_vocab=512; max_merges=256; min_frequency=2",
        corpus.provenance().fnv1a64,
        corpus.bytes().len()
    )?;
    writeln!(
        report,
        "eval={EVAL_PATH}; eval_fnv1a64={:016x}; cases={}; raw_text_bytes={}; no normalization; no model training",
        fnv1a64(fixture.as_bytes()),
        cases.len(),
        cases.iter().map(|c| c.text.len()).sum::<usize>()
    )?;
    writeln!(
        report,
        "artifact_fnv1a64={:016x}; artifact_bytes={}; vocabulary_size={}",
        fnv1a64(&artifact_bytes),
        artifact_bytes.len(),
        artifact
            .flat_bpe()
            .ok_or("missing flat model")?
            .vocabulary_size()
    )?;
    writeln!(
        report,
        "semantic_registry_namespace={}; revision={}; policy={}; fingerprint={:016x}; sidecar_format_version={SEMANTIC_SIDECAR_VERSION}",
        expected_registry_identity.namespace,
        expected_registry_identity.revision,
        expected_registry_identity.policy_id,
        expected_registry_identity.fingerprint
    )?;
    writeln!(
        report,
        "timing: trials={TRIALS}; warmup_ms={}; minimum_measurement_ms={}; repeat_batch={REPEATS}; batch_bytes={}; batch_fnv1a64={:016x}; A/B order alternates; parse/serialization excluded",
        WARMUP.as_millis(),
        MEASURE.as_millis(),
        batch.len(),
        fnv1a64(batch.as_bytes())
    )?;
    for variant in ["A", "B"] {
        for direction in ["encode", "decode"] {
            let mut samples = timings
                .iter()
                .filter(|t| t.variant == variant && t.direction == direction)
                .map(Timing::mib_per_second)
                .collect::<Vec<_>>();
            samples.sort_by(f64::total_cmp);
            writeln!(
                report,
                "{variant} {direction}: median={:.6} MiB/s; min={:.6}; max={:.6}",
                samples[TRIALS / 2],
                samples[0],
                samples[TRIALS - 1]
            )?;
        }
    }
    writeln!(
        report,
        "shared_lexical_batch_encode: allocation_requests={}; peak_buffer_bytes={}; shared_lexical_batch_decode: allocation_requests={}; peak_buffer_bytes={}",
        lexical_encode_buffers.allocation_requests,
        lexical_encode_buffers.peak_buffer_bytes,
        lexical_decode_buffers.allocation_requests,
        lexical_decode_buffers.peak_buffer_bytes
    )?;
    writeln!(
        report,
        "lexical buffer stats: same production BPE operation in A/B, measured once outside timings; vector growth requests and peak simultaneous working/output vector capacities; excludes additional semantic allocations, immutable tokenizer, caller input and allocator overhead"
    )?;
    writeln!(
        report,
        "lexical_tokens_total={lexical_count_total}; lexical_output_capacity_bytes_total={lexical_capacity_total}; A/B lexical IDs and sequence lengths identical"
    )?;
    writeln!(
        report,
        "B extra: serialized_sidecar_bytes_total={sidecar_total}; annotation_output_capacity_bytes_total={annotation_capacity_total}; registry_identity_namespace_capacity_bytes_total={registry_namespace_capacity_total}; empty sidecars included; sidecar bytes exclude lexical stream"
    )?;
    writeln!(
        report,
        "memory estimates include annotation vector capacities, nested candidate vectors and separate owned registry-identity namespace String capacities; exclude outer/identity stack structs, allocator metadata, immutable registry, stack and runtime temporaries; not peak additional RSS"
    )?;
    match rss {
        Ok(value) => writeln!(
            report,
            "process_peak_rss_bytes={value}; scope=whole paired benchmark including training and A/B; cannot isolate semantic overhead"
        )?,
        Err(error) => writeln!(report, "process_peak_rss=unavailable ({error})")?,
    }
    writeln!(report, "accuracy={accuracy:?}")?;
    writeln!(
        report,
        "span_precision={}; span_recall={}; resolved_precision={}; resolved_recall={}",
        ratio(accuracy.matching_spans, accuracy.actual_spans),
        ratio(accuracy.matching_spans, accuracy.expected_spans),
        ratio(accuracy.correct_resolved, accuracy.actual_resolved),
        ratio(accuracy.correct_resolved, accuracy.expected_resolved)
    )?;
    writeln!(
        report,
        "All case/batch roundtrips, optional-off differentials, repeated outputs and sidecar serialization roundtrips passed. Accuracy counts exact byte spans and concept IDs, not approximate meanings."
    )?;
    writeln!(
        report,
        "Limitations: agent-authored synthetic development conformance corpus, no held-out statistical generalization claim; tiny registry, rule heuristics, no compression/learning improvement established; decoder ignores sidecar and shares identical implementation in A/B."
    )?;
    fs::write(directory.join("report.txt"), &report)?;
    print!("{report}");
    println!("saved {}", directory.display());
    Ok(())
}

fn measure<F: FnMut()>(
    trial: usize,
    variant: &'static str,
    direction: &'static str,
    bytes: usize,
    tokens: usize,
    mut operation: F,
) -> Timing {
    let start = Instant::now();
    while start.elapsed() < WARMUP {
        operation();
    }
    let start = Instant::now();
    let mut iterations = 0;
    while start.elapsed() < MEASURE {
        operation();
        iterations += 1;
    }
    Timing {
        trial,
        variant,
        direction,
        iterations,
        elapsed_ns: start.elapsed().as_nanos(),
        bytes,
        tokens,
    }
}

fn ratio(numerator: usize, denominator: usize) -> String {
    if denominator == 0 {
        "undefined".to_owned()
    } else {
        format!("{:.6}", numerator as f64 / denominator as f64)
    }
}

fn annotation_capacity_bytes(annotations: &Vec<SemanticAnnotation>) -> usize {
    annotations.capacity() * size_of::<SemanticAnnotation>()
        + annotations
            .iter()
            .map(|a| a.candidates.capacity() * size_of::<SemanticId>())
            .sum::<usize>()
}

fn record_accuracy(accuracy: &mut Accuracy, case: &Case, actual: &[SemanticAnnotation]) -> bool {
    accuracy.expected_spans += case.gold.len();
    accuracy.actual_spans += actual.len();
    accuracy.expected_resolved += case.gold.iter().filter(|g| g.concept.is_some()).count();
    accuracy.expected_ambiguous += case.gold.iter().filter(|g| g.concept.is_none()).count();
    for annotation in actual {
        let gold = case
            .gold
            .iter()
            .find(|g| g.start == annotation.start && g.end == annotation.end);
        if gold.is_some() {
            accuracy.matching_spans += 1;
        }
        if annotation.decision == SemanticDecision::Resolved {
            accuracy.actual_resolved += 1;
            if let Some(gold) = gold {
                match gold.concept {
                    Some(id) if annotation.candidates == [id] => accuracy.correct_resolved += 1,
                    None => accuracy.false_resolutions_of_ambiguity += 1,
                    _ => {}
                }
            }
        } else if gold.is_some_and(|g| g.concept.is_none()) {
            accuracy.correctly_ambiguous += 1;
        }
    }
    if case.category == "unknown" {
        accuracy.unknown_cases += 1;
        accuracy.unknown_cases_with_annotations += usize::from(!actual.is_empty());
    }
    actual.len() == case.gold.len()
        && actual.iter().zip(&case.gold).all(|(a, g)| {
            a.start == g.start
                && a.end == g.end
                && match g.concept {
                    Some(id) => a.decision == SemanticDecision::Resolved && a.candidates == [id],
                    None => a.decision == SemanticDecision::Ambiguous,
                }
        })
}

fn parse_cases(input: &str) -> Result<Vec<Case>, Box<dyn Error>> {
    let mut cases = Vec::new();
    let mut seen = BTreeSet::new();
    for (number, line) in input.lines().enumerate() {
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let parts = line.splitn(4, '\t').collect::<Vec<_>>();
        if parts.len() != 4 || parts[0].is_empty() || !seen.insert(parts[0]) {
            return Err(format!("invalid/duplicate semantic fixture row {}", number + 1).into());
        }
        let text = unescape(parts[3])?;
        let mut gold = Vec::new();
        if parts[2] != "-" {
            for span in parts[2].split(';') {
                let fields = span.split(':').collect::<Vec<_>>();
                if fields.len() != 3 {
                    return Err("invalid gold span".into());
                }
                let start = fields[0].parse::<usize>()?;
                let end = fields[1].parse::<usize>()?;
                let concept = if fields[2] == "?" {
                    None
                } else {
                    Some(SemanticId(fields[2].parse()?))
                };
                if start >= end
                    || end > text.len()
                    || !text.is_char_boundary(start)
                    || !text.is_char_boundary(end)
                    || gold.last().is_some_and(|g: &Gold| g.end > start)
                {
                    return Err("invalid gold byte span".into());
                }
                if concept.is_some_and(|id| SemanticRegistry::builtin().concept(id).is_none()) {
                    return Err("unknown gold concept ID".into());
                }
                gold.push(Gold {
                    start,
                    end,
                    concept,
                });
            }
        }
        cases.push(Case {
            id: parts[0].to_owned(),
            category: parts[1].to_owned(),
            text,
            gold,
        });
    }
    if cases.is_empty() {
        return Err("empty semantic corpus".into());
    }
    Ok(cases)
}

fn unescape(input: &str) -> Result<String, Box<dyn Error>> {
    let mut result = String::new();
    let mut chars = input.chars();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            result.push(match chars.next() {
                Some('n') => '\n',
                Some('r') => '\r',
                Some('t') => '\t',
                Some('0') => '\0',
                Some('v') => '\u{000b}',
                Some('f') => '\u{000c}',
                Some('\\') => '\\',
                _ => return Err("invalid fixture escape".into()),
            });
        } else {
            result.push(ch);
        }
    }
    Ok(result)
}

fn environment() -> String {
    let mut output = String::from(
        "command: cargo +1.85.0 run --release -p packtok-bench -- semantic <unique-label>\n",
    );
    for (program, args) in [
        ("rustc", vec!["--version", "--verbose"]),
        ("cargo", vec!["--version"]),
        ("git", vec!["rev-parse", "HEAD"]),
        ("git", vec!["status", "--short"]),
        ("uname", vec!["-a"]),
    ] {
        let _ = writeln!(output, "$ {program} {}", args.join(" "));
        match Command::new(program).args(&args).output() {
            Ok(result) => {
                output.push_str(&String::from_utf8_lossy(&result.stdout));
                output.push_str(&String::from_utf8_lossy(&result.stderr));
            }
            Err(error) => {
                let _ = writeln!(output, "unavailable: {error}");
            }
        }
    }
    if let Ok(cpu) = fs::read_to_string("/proc/cpuinfo") {
        if let Some(model) = cpu.lines().find(|line| line.starts_with("model name")) {
            let _ = writeln!(output, "{model}");
        }
    }
    let paths = [
        "Cargo.lock",
        "crates/packtok-bench/src/semantic.rs",
        "crates/packtok-tokenizer/src/semantic.rs",
        "crates/packtok-packs/src/semantic.rs",
        "crates/packtok-format/src/semantic.rs",
        EVAL_PATH,
        TRAIN_PATH,
    ];
    for path in paths {
        if let Ok(bytes) = fs::read(path) {
            let _ = writeln!(output, "source_fnv1a64 {:016x} {path}", fnv1a64(&bytes));
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_is_valid_and_includes_required_classes() {
        let cases = parse_cases(include_str!("../../../fixtures/semantic-v1/eval.tsv")).unwrap();
        assert!(cases.len() >= 28);
        for category in [
            "slang",
            "en",
            "de",
            "ambiguous",
            "context",
            "unknown",
            "unicode",
            "whitespace",
        ] {
            assert!(
                cases.iter().any(|c| c.category == category),
                "missing {category}"
            );
        }
        assert!(cases.iter().any(|c| c.text.is_empty()));
        assert!(cases.iter().any(|c| c.text.contains('\0')));
    }

    #[test]
    fn independent_review_regression_gold_matches_conservative_decisions() {
        let cases = parse_cases(include_str!(
            "../../../fixtures/semantic-v1/review-regressions.tsv"
        ))
        .unwrap();
        assert_eq!(cases.len(), 7);
        for case in cases {
            let annotations = SemanticRegistry::builtin().annotate(&case.text);
            assert!(
                record_accuracy(&mut Accuracy::default(), &case, &annotations),
                "{}: expected {:?}, actual {:?}",
                case.id,
                case.gold,
                annotations
            );
        }
        assert_eq!(
            unescape("fr\\vbro ka\\fwas").unwrap(),
            "fr\u{000b}bro ka\u{000c}was"
        );
    }

    #[test]
    fn fixture_parser_rejects_bad_spans_escapes_and_duplicate_ids() {
        for input in [
            "a\tde\t0:1:1\tä",
            "a\tde\t0:3:9999\tidk",
            "a\tde\t-\t\\x",
            "a\tde\t-\tx\na\tde\t-\ty",
            "a\tde\t3:1:1\tidk",
        ] {
            assert!(parse_cases(input).is_err());
        }
        assert_eq!(unescape("a\\t\\n\\r\\0\\\\").unwrap(), "a\t\n\r\0\\");
    }

    #[test]
    fn accuracy_uses_exact_span_and_does_not_reward_wrong_concept() {
        let case = Case {
            id: "idk".into(),
            category: "en".into(),
            text: "idk".into(),
            gold: vec![Gold {
                start: 0,
                end: 3,
                concept: Some(SemanticId(2)),
            }],
        };
        let mut accuracy = Accuracy::default();
        let annotations = SemanticRegistry::builtin().annotate("idk");
        assert!(!record_accuracy(&mut accuracy, &case, &annotations));
        assert_eq!(accuracy.matching_spans, 1);
        assert_eq!(accuracy.correct_resolved, 0);
        assert_eq!(accuracy.actual_resolved, 1);
    }
}
