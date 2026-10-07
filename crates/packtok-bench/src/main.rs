#![forbid(unsafe_code)]

mod audit;
mod m3;
mod m4;
mod m4_corpus;
mod metrics;
mod scan;

use std::error::Error;
use std::hint::black_box;
use std::path::Path;
use std::time::{Duration, Instant};

use packtok_packs::lexical_pack_name;
use packtok_tokenizer::{BpeTokenizer, ByteFallbackTokenizer, FactorizedTokenizer, Tokenizer};
use packtok_train::{
    BpeTrainingConfig, FactorizedTrainingConfig, fnv1a64, load_corpus, train_bpe_with_provenance,
    train_factorized_bpe_with_provenance_report,
};

const MEASUREMENT_TIME: Duration = Duration::from_millis(200);
const WARMUP_TIME: Duration = Duration::from_millis(25);
const SAMPLE_REPETITIONS: usize = 32;

struct Sample {
    name: &'static str,
    path: &'static str,
}

const TRAIN_PATH: &str = "fixtures/benchmark/train.txt";
const SAMPLES: [Sample; 4] = [
    Sample {
        name: "English prose",
        path: "fixtures/benchmark/eval/english.txt",
    },
    Sample {
        name: "German prose",
        path: "fixtures/benchmark/eval/german.txt",
    },
    Sample {
        name: "Unicode-heavy text",
        path: "fixtures/benchmark/eval/unicode.txt",
    },
    Sample {
        name: "Source-like text",
        path: "fixtures/benchmark/eval/code.txt",
    },
];

struct Measurement {
    bytes_processed: u64,
    tokens_processed: u64,
    iterations: u64,
    elapsed: Duration,
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = std::env::args();
    let _program = arguments.next();
    let command = arguments.next();
    if command.as_deref() == Some("m4-corpus") {
        return m4_corpus::run();
    }
    if command.as_deref() == Some("m4") {
        let corpus = arguments
            .next()
            .ok_or("usage: packtok-bench m4 tiny|large unique-label")?;
        let label = arguments.next().ok_or("missing unique run label")?;
        if arguments.next().is_some() {
            return Err("unexpected argument".into());
        }
        return m4::run(&corpus, &label);
    }
    if command.as_deref() == Some("audit") {
        if arguments.next().is_some() {
            return Err("usage: packtok-bench audit".into());
        }
        return audit::run();
    }
    if command.as_deref() == Some("m3") {
        let label = arguments.next().unwrap_or_else(|| "manual".to_owned());
        if arguments.next().is_some() {
            return Err("usage: packtok-bench m3 [unique-run-label]".into());
        }
        return m3::run(&label);
    }
    let corpus = load_corpus(Path::new(TRAIN_PATH))?;
    let config = BpeTrainingConfig::default();
    let training_started = Instant::now();
    let artifact = train_bpe_with_provenance(corpus.bytes(), config, corpus.provenance())?;
    let training_elapsed = training_started.elapsed();
    let artifact_size = artifact.to_bytes()?.len();
    let model = artifact
        .flat_bpe()
        .ok_or("training produced no BPE model")?;
    let bpe = BpeTokenizer::from_artifact(&artifact)?;
    let scanner = scan::ScanTokenizer {
        model: model.clone(),
        decoder: bpe.clone(),
    };
    let byte_tokenizer = ByteFallbackTokenizer::default();
    let factorized_config = FactorizedTrainingConfig::default();
    let factorized_started = Instant::now();
    let factorized_report = train_factorized_bpe_with_provenance_report(
        corpus.bytes(),
        factorized_config,
        corpus.provenance(),
    )?;
    let factorized_training_elapsed = factorized_started.elapsed();
    let factorized_artifact_size = factorized_report.artifact.to_bytes()?.len();
    let factorized = FactorizedTokenizer::from_artifact(&factorized_report.artifact)?;
    println!("PackTok M0 byte fallback, M1 flat BPE, and M2 factorized BPE benchmark");
    println!(
        "measurement: fixed input; {WARMUP_TIME:?} warm-up, {MEASUREMENT_TIME:?} per direction; release profile recommended"
    );
    println!(
        "split: train={TRAIN_PATH}; train_fnv1a64={:016x}; eval=fixtures/benchmark/eval/*.txt; no normalization",
        corpus.provenance().fnv1a64
    );
    println!(
        "M1 model: {} byte tokens + {} learned merges = {} vocabulary IDs; artifact={} B; fixture={} B; train-once={:.3} ms ({:.2} MiB/s; excludes file loading)",
        256,
        model.merges().len(),
        model.vocabulary_size(),
        artifact_size,
        corpus.provenance().total_bytes,
        training_elapsed.as_secs_f64() * 1000.0,
        mib_per_second(corpus.provenance().total_bytes, training_elapsed),
    );
    println!(
        "M2 model: 256 shared byte tokens + {} learned tokens = {} logical IDs; router=lexical-v1; artifact={} B; train-once={:.3} ms ({:.2} MiB/s; excludes file loading)",
        factorized_report.total_logical_vocab_size - 256,
        factorized_report.total_logical_vocab_size,
        factorized_artifact_size,
        factorized_training_elapsed.as_secs_f64() * 1000.0,
        mib_per_second(corpus.provenance().total_bytes, factorized_training_elapsed),
    );
    for pack_stats in factorized_report.packs {
        println!(
            "  M2 training pack {}: spans={} routed_bytes={} learned_merges={} local_vocab_size={}",
            pack_display_name(pack_stats.pack_id),
            pack_stats.spans,
            pack_stats.routed_bytes,
            pack_stats.learned_merges,
            pack_stats.local_vocab_size
        );
    }

    for sample in SAMPLES {
        let text = std::fs::read_to_string(sample.path)?;
        ensure_held_out(corpus.bytes(), text.as_bytes())?;
        println!(
            "Evaluation {}: path={}; raw_bytes={}; fnv1a64={:016x}",
            sample.name,
            sample.path,
            text.len(),
            fnv1a64(text.as_bytes())
        );
        let input = text.repeat(SAMPLE_REPETITIONS);
        let byte_encoded = byte_tokenizer.encode(&input)?;
        let bpe_encoded = bpe.encode(&input)?;
        let (factorized_encoded, factorized_stats) = factorized.encode_with_stats(&input)?;
        let (measured_tokens, encode_stats) = bpe.encode_bytes_with_stats(input.as_bytes())?;
        let (measured_bytes, decode_stats) = bpe.decode_bytes_with_stats(&bpe_encoded)?;
        if measured_tokens != bpe_encoded || measured_bytes != input.as_bytes() {
            return Err(format!("instrumented result differs for {}", sample.name).into());
        }
        if scanner.encode(&input)? != bpe_encoded {
            return Err(format!("reference IDs differ for {}", sample.name).into());
        }
        if byte_tokenizer.decode_bytes(&byte_encoded)? != input.as_bytes()
            || bpe.decode_bytes(&bpe_encoded)? != input.as_bytes()
            || factorized.decode_bytes(&factorized_encoded)? != input.as_bytes()
        {
            return Err(format!("round-trip failed for {}", sample.name).into());
        }

        let byte_encode = measure_encode(&byte_tokenizer, &input)?;
        let byte_decode = measure_decode(&byte_tokenizer, &byte_encoded)?;
        let bpe_encode = measure_encode(&bpe, &input)?;
        let bpe_decode = measure_decode(&bpe, &bpe_encoded)?;
        let scan_encode = measure_encode(&scanner, &input)?;
        let factorized_encode = measure_encode(&factorized, &input)?;
        let factorized_decode = measure_decode(&factorized, &factorized_encoded)?;
        let bytes_per_token = input.len() as f64 / bpe_encoded.len().max(1) as f64;
        let factorized_bytes_per_token =
            input.len() as f64 / factorized_encoded.len().max(1) as f64;
        let token_reduction = if byte_encoded.is_empty() {
            0.0
        } else {
            (1.0 - bpe_encoded.len() as f64 / byte_encoded.len() as f64) * 100.0
        };
        println!(
            "{}: input={} B; M0 tokens/input={} encode={:.2} MiB/s ({} iters, {} tokens) decode={:.2} MiB/s ({} iters); M1 tokens/input={} encode={:.2} MiB/s ({} iters, {} tokens) decode={:.2} MiB/s ({} iters) bytes/token={:.3} reduction_vs_M0={:.2}%",
            sample.name,
            input.len(),
            byte_encoded.len(),
            mib_per_second(byte_encode.bytes_processed, byte_encode.elapsed),
            byte_encode.iterations,
            byte_encode.tokens_processed,
            mib_per_second(byte_decode.bytes_processed, byte_decode.elapsed),
            byte_decode.iterations,
            bpe_encoded.len(),
            mib_per_second(bpe_encode.bytes_processed, bpe_encode.elapsed),
            bpe_encode.iterations,
            bpe_encode.tokens_processed,
            mib_per_second(bpe_decode.bytes_processed, bpe_decode.elapsed),
            bpe_decode.iterations,
            bytes_per_token,
            token_reduction,
        );
        println!(
            "  matched pre-audit rank scans: encode={:.2} MiB/s; current/scan={:.2}x",
            mib_per_second(scan_encode.bytes_processed, scan_encode.elapsed),
            mib_per_second(bpe_encode.bytes_processed, bpe_encode.elapsed)
                / mib_per_second(scan_encode.bytes_processed, scan_encode.elapsed)
        );
        println!(
            "  M1 owned vector allocation/reallocation requests enc/dec={}/{}; peak vector capacity enc/dec={}/{} B (excludes model, input, allocator overhead and RSS)",
            encode_stats.allocation_requests,
            decode_stats.allocation_requests,
            encode_stats.peak_buffer_bytes,
            decode_stats.peak_buffer_bytes
        );
        let factorized_reduction_vs_m1 = if bpe_encoded.is_empty() {
            0.0
        } else {
            (1.0 - factorized_encoded.len() as f64 / bpe_encoded.len() as f64) * 100.0
        };
        let raw_token_percentage = factorized_stats
            .for_pack(packtok_tokenizer::DEFAULT_BYTE_FALLBACK_PACK_ID)
            .map(|stats| 100.0 * stats.tokens as f64 / factorized_encoded.len().max(1) as f64)
            .unwrap_or(0.0);
        println!(
            "M2 {}: tokens={} bytes/token={:.3}; encode={:.2} MiB/s ({} iters); decode={:.2} MiB/s ({} iters); sequence_change_vs_M1={:+.2}%; raw_byte_token_share={:.2}%",
            sample.name,
            factorized_encoded.len(),
            factorized_bytes_per_token,
            mib_per_second(factorized_encode.bytes_processed, factorized_encode.elapsed),
            factorized_encode.iterations,
            mib_per_second(factorized_decode.bytes_processed, factorized_decode.elapsed),
            factorized_decode.iterations,
            -factorized_reduction_vs_m1,
            raw_token_percentage
        );
        println!(
            "  M2 output vector capacity={} B; temporary router/symbol capacity peak={} B; pack transitions={}",
            factorized_stats.output_capacity_bytes,
            factorized_stats.temporary_peak_bytes,
            factorized_stats.pack_transitions
        );
        for pack_stats in factorized_stats.packs {
            let learned_merges = factorized
                .model()
                .pack(pack_stats.pack_id)
                .map_or(0, |pack| pack.merges().len());
            let average_span = if pack_stats.spans == 0 {
                0.0
            } else {
                pack_stats.routed_bytes as f64 / pack_stats.spans as f64
            };
            println!(
                "  M2 pack {}: output_tokens={} output_bytes={} routed_bytes={} spans={} average_span_bytes={:.2} learned_merges={}",
                pack_display_name(pack_stats.pack_id),
                pack_stats.tokens,
                pack_stats.token_bytes,
                pack_stats.routed_bytes,
                pack_stats.spans,
                average_span,
                learned_merges
            );
        }
        print_sequence_lengths(&text, &bpe)?;
        print_factorized_sequence_lengths(&text, &factorized)?;
    }

    let repeated_model = packtok_train::train_model(&vec![b'a'; 512], config)?;
    let mut larger_merges = repeated_model.merges().to_vec();
    for right in 0..64 {
        larger_merges.push(packtok_format::FlatBpeMerge {
            left: 0,
            right,
            result: 256 + larger_merges.len() as u32,
        });
    }
    for (name, input, stress_model) in [
        ("Dense repeated bytes", "a".repeat(16_384), repeated_model),
        (
            "Dense repetitions, larger model",
            "a".repeat(16_384),
            packtok_format::FlatBpeModel::new(larger_merges.clone())?,
        ),
        (
            "Dense repetitions with one exceptional byte",
            format!("{}b{}", "a".repeat(8_191), "a".repeat(8_192)),
            packtok_format::FlatBpeModel::new(larger_merges)?,
        ),
        (
            "Dense alternating bytes",
            "ab".repeat(8_192),
            packtok_train::train_model("ab".repeat(512).as_bytes(), config)?,
        ),
        (
            "Dense three-byte repetitions",
            "abc".repeat(5_461),
            packtok_train::train_model("abc".repeat(512).as_bytes(), config)?,
        ),
        (
            "Unmatched repeated bytes",
            "Z".repeat(16_384),
            model.clone(),
        ),
        (
            "Training corpus repeated (stress only, excluded from held-out metrics)",
            String::from_utf8(corpus.bytes().repeat(32))?,
            model.clone(),
        ),
    ] {
        let runtime = BpeTokenizer::new(stress_model.clone());
        let scanner = scan::ScanTokenizer {
            model: stress_model,
            decoder: runtime.clone(),
        };
        let tokens = runtime.encode(&input)?;
        if scanner.encode(&input)? != tokens || runtime.decode_bytes(&tokens)? != input.as_bytes() {
            return Err(format!("stress round-trip or reference IDs differ for {name}").into());
        }
        let current = measure_encode(&runtime, &input)?;
        let scan = measure_encode(&scanner, &input)?;
        println!(
            "Stress {name}: input={} B; merges={}; tokens={}; current={:.2} MiB/s; scan={:.2} MiB/s; current/scan={:.2}x",
            input.len(),
            runtime.merges().len(),
            tokens.len(),
            mib_per_second(current.bytes_processed, current.elapsed),
            mib_per_second(scan.bytes_processed, scan.elapsed),
            mib_per_second(current.bytes_processed, current.elapsed)
                / mib_per_second(scan.bytes_processed, scan.elapsed)
        );
    }

    match metrics::process_peak_memory_bytes() {
        Ok(bytes) => println!(
            "OS process peak resident/working-set memory={bytes} B (entire run, includes training/models/M0/M1/M2/reference; not per operation)"
        ),
        Err(error) => println!("OS process peak resident/working-set memory unavailable: {error}"),
    }

    Ok(())
}

fn ensure_held_out(train: &[u8], evaluation: &[u8]) -> Result<(), Box<dyn Error>> {
    // Characters and short syntax naturally overlap; copied passages must not.
    for window in evaluation.windows(32) {
        if train.windows(32).any(|candidate| candidate == window) {
            return Err("evaluation has a 32-byte passage copied from training".into());
        }
    }
    Ok(())
}

fn print_sequence_lengths(text: &str, bpe: &BpeTokenizer) -> Result<(), Box<dyn Error>> {
    let records: Vec<_> = text.split_inclusive('\n').collect();
    let byte_lengths = records.iter().map(|record| record.len()).collect();
    let bpe_lengths = records
        .iter()
        .map(|record| bpe.encode(record).map(|tokens| tokens.len()))
        .collect::<Result<Vec<_>, _>>()?;
    for (name, lengths) in [("M0", byte_lengths), ("M1", bpe_lengths)] {
        if let Some(stats) = metrics::SequenceLengths::new(lengths) {
            println!(
                "  {name} evaluation record lengths: n={} min={} p50={} p95={} max={} mean={:.2} tokens (original lines with newline bytes)",
                stats.count, stats.min, stats.median, stats.p95, stats.max, stats.mean
            );
        }
    }
    Ok(())
}

fn print_factorized_sequence_lengths(
    text: &str,
    factorized: &FactorizedTokenizer,
) -> Result<(), Box<dyn Error>> {
    let records: Vec<_> = text.split_inclusive('\n').collect();
    let lengths = records
        .iter()
        .map(|record| factorized.encode(record).map(|tokens| tokens.len()))
        .collect::<Result<Vec<_>, _>>()?;
    if let Some(stats) = metrics::SequenceLengths::new(lengths) {
        println!(
            "  M2 evaluation record lengths: n={} min={} p50={} p95={} max={} mean={:.2} tokens (original lines with newline bytes)",
            stats.count, stats.min, stats.median, stats.p95, stats.max, stats.mean
        );
    }
    Ok(())
}

fn pack_display_name(pack_id: u16) -> &'static str {
    if pack_id == packtok_tokenizer::DEFAULT_BYTE_FALLBACK_PACK_ID {
        "BYTE_FALLBACK"
    } else {
        lexical_pack_name(pack_id).unwrap_or("UNKNOWN_PACK")
    }
}

fn measure_encode(tokenizer: &impl Tokenizer, input: &str) -> Result<Measurement, Box<dyn Error>> {
    let warmup = Instant::now();
    while warmup.elapsed() < WARMUP_TIME {
        drop(black_box(tokenizer.encode(black_box(input))?));
    }
    let started = Instant::now();
    let mut bytes_processed = 0_u64;
    let mut tokens_processed = 0_u64;
    let mut iterations = 0_u64;

    while started.elapsed() < MEASUREMENT_TIME {
        let tokens = tokenizer.encode(black_box(input))?;
        bytes_processed = bytes_processed
            .checked_add(u64::try_from(input.len())?)
            .ok_or("benchmark byte counter overflow")?;
        tokens_processed = tokens_processed
            .checked_add(u64::try_from(tokens.len())?)
            .ok_or("benchmark token counter overflow")?;
        iterations = iterations
            .checked_add(1)
            .ok_or("benchmark iteration counter overflow")?;
        drop(black_box(tokens));
    }

    Ok(Measurement {
        bytes_processed,
        tokens_processed,
        iterations,
        elapsed: started.elapsed(),
    })
}

fn measure_decode(
    tokenizer: &impl Tokenizer,
    tokens: &[packtok_tokenizer::TokenId],
) -> Result<Measurement, Box<dyn Error>> {
    let warmup = Instant::now();
    while warmup.elapsed() < WARMUP_TIME {
        drop(black_box(tokenizer.decode_bytes(black_box(tokens))?));
    }
    let started = Instant::now();
    let mut bytes_processed = 0_u64;
    let mut tokens_processed = 0_u64;
    let mut iterations = 0_u64;

    while started.elapsed() < MEASUREMENT_TIME {
        let bytes = tokenizer.decode_bytes(black_box(tokens))?;
        bytes_processed = bytes_processed
            .checked_add(u64::try_from(bytes.len())?)
            .ok_or("benchmark byte counter overflow")?;
        tokens_processed = tokens_processed
            .checked_add(u64::try_from(tokens.len())?)
            .ok_or("benchmark token counter overflow")?;
        iterations = iterations
            .checked_add(1)
            .ok_or("benchmark iteration counter overflow")?;
        drop(black_box(bytes));
    }

    Ok(Measurement {
        bytes_processed,
        tokens_processed,
        iterations,
        elapsed: started.elapsed(),
    })
}

fn mib_per_second(bytes: u64, elapsed: Duration) -> f64 {
    let seconds = elapsed.as_secs_f64();
    if seconds == 0.0 {
        return 0.0;
    }
    bytes as f64 / (1024.0 * 1024.0 * seconds)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benchmark_training_and_evaluation_passages_are_separate() {
        let train = include_bytes!("../../../fixtures/benchmark/train.txt");
        for evaluation in [
            include_bytes!("../../../fixtures/benchmark/eval/english.txt").as_slice(),
            include_bytes!("../../../fixtures/benchmark/eval/german.txt").as_slice(),
            include_bytes!("../../../fixtures/benchmark/eval/unicode.txt").as_slice(),
            include_bytes!("../../../fixtures/benchmark/eval/code.txt").as_slice(),
        ] {
            assert!(ensure_held_out(train, evaluation).is_ok());
            assert!(evaluation.split_inclusive(|byte| *byte == b'\n').count() >= 3);
        }
        assert!(ensure_held_out(train, train).is_err());
    }
}
