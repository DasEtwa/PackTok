#![forbid(unsafe_code)]

use std::error::Error;
use std::hint::black_box;
use std::path::Path;
use std::time::{Duration, Instant};

use packtok_tokenizer::{BpeTokenizer, ByteFallbackTokenizer, Tokenizer};
use packtok_train::{BpeTrainingConfig, load_corpus, train_bpe_with_provenance};

const MEASUREMENT_TIME: Duration = Duration::from_millis(200);
const SAMPLE_REPETITIONS: usize = 32;

struct Sample {
    name: &'static str,
    text: &'static str,
}

struct Measurement {
    bytes_processed: u64,
    tokens_processed: u64,
    iterations: u64,
    elapsed: Duration,
}

fn main() -> Result<(), Box<dyn Error>> {
    let samples = [
        Sample {
            name: "English prose",
            text: "A small, repeatable benchmark gives us a baseline before optimization. \
                   PackTok currently maps every UTF-8 byte to one local token.",
        },
        Sample {
            name: "German prose",
            text: "Sämtliche Häuser öffnen ihre Türen, während Grüße über die Straße hallen. \
                   Größere Wörter und ihre Veränderungen bleiben exakt erhalten.",
        },
        Sample {
            name: "Unicode-heavy text",
            text: "a\u{0308} e\u{0301} 漢字かなカナ 中文 ✨🦦🇩🇪👩🏽‍💻 — Unicode bytes stay unchanged. \
                   नमस्ते мир مرحبا",
        },
        Sample {
            name: "Source-like text",
            text: "fn encode(input: &str) -> Vec<TokenId> {\n    input.as_bytes().iter()\n        .map(|byte| TokenId::new(BYTE_PACK, u32::from(*byte)))\n        .collect()\n}\n\n// Keep behavior deterministic.\n",
        },
    ];

    let corpus = load_corpus(Path::new("fixtures/m1_bpe_corpus.txt"))?;
    let config = BpeTrainingConfig::default();
    let training_started = Instant::now();
    let artifact = train_bpe_with_provenance(corpus.bytes(), config, corpus.provenance())?;
    let training_elapsed = training_started.elapsed();
    let artifact_size = artifact.to_bytes()?.len();
    let model = artifact
        .flat_bpe()
        .ok_or("training produced no BPE model")?;
    let bpe = BpeTokenizer::from_artifact(&artifact)?;
    let byte_tokenizer = ByteFallbackTokenizer::default();
    println!("PackTok M0 byte fallback and M1 flat byte-level BPE benchmark");
    println!(
        "measurement: fixed input; {MEASUREMENT_TIME:?} per direction; release profile recommended"
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

    for sample in samples {
        let input = sample.text.repeat(SAMPLE_REPETITIONS);
        let byte_encoded = byte_tokenizer.encode(&input)?;
        let bpe_encoded = bpe.encode(&input)?;
        if byte_tokenizer.decode_bytes(&byte_encoded)? != input.as_bytes()
            || bpe.decode_bytes(&bpe_encoded)? != input.as_bytes()
        {
            return Err(format!("round-trip failed for {}", sample.name).into());
        }

        let byte_encode = measure_encode(&byte_tokenizer, &input)?;
        let byte_decode = measure_decode(&byte_tokenizer, &byte_encoded)?;
        let bpe_encode = measure_encode(&bpe, &input)?;
        let bpe_decode = measure_decode(&bpe, &bpe_encoded)?;
        let bytes_per_token = input.len() as f64 / bpe_encoded.len().max(1) as f64;
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
    }

    Ok(())
}

fn measure_encode(tokenizer: &impl Tokenizer, input: &str) -> Result<Measurement, Box<dyn Error>> {
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
