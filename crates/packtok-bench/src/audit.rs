//! Fixed local workloads for performance/correctness review; no corpus downloads.
use std::error::Error;
use std::hint::black_box;
use std::time::{Duration, Instant};

use packtok_model::{CausalLm, ModelConfig, OptimizerConfig, PackVocabulary, TrainingExample};
use packtok_tokenizer::{FactorizedTokenizer, TokenId, Tokenizer};
use packtok_train::{FactorizedTrainingConfig, train_factorized_bpe};

pub fn run() -> Result<(), Box<dyn Error>> {
    println!(
        "audit-v1: release; warmup=25ms measurement>=200ms per workload; seed=20261007; model hidden=16 context=16; untrained weights; timings include returned buffer destruction"
    );
    let config = ModelConfig {
        hidden_size: 16,
        context_length: 16,
    };
    let flat = CausalLm::new_flat(config, 512, 20_261_007)?;
    let factorized = CausalLm::new_factorized(
        config,
        vec![
            PackVocabulary {
                pack_id: 1,
                token_count: 253,
            },
            PackVocabulary {
                pack_id: 3,
                token_count: 3,
            },
            PackVocabulary {
                pack_id: u16::MAX,
                token_count: 256,
            },
        ],
        20_261_007,
    )?;
    for (name, model, tokens) in [
        (
            "flat",
            flat,
            (0..64)
                .map(|i| TokenId::new(0, i % 512))
                .collect::<Vec<_>>(),
        ),
        (
            "factorized",
            factorized,
            (0..64)
                .map(|i| match i % 3 {
                    0 => TokenId::new(1, i % 253),
                    1 => TokenId::new(3, i % 3),
                    _ => TokenId::new(u16::MAX, i % 256),
                })
                .collect(),
        ),
    ] {
        let examples: Vec<_> = (1..tokens.len())
            .map(|i| TrainingExample {
                inputs: &tokens[i.saturating_sub(16)..i],
                targets: &tokens[i..i + 1],
            })
            .collect();
        let batch: Vec<_> = (0..4)
            .map(|i| TrainingExample {
                inputs: &tokens[i..i + 16],
                targets: &tokens[i + 1..i + 17],
            })
            .collect();
        let bytes = model.to_bytes()?;
        let score = model.evaluate(&examples)?;
        println!(
            "{name}: targets={} loss={:.12} parameter_count={} artifact_bytes={}",
            score.target_tokens,
            score.loss_per_token,
            model.parameter_counts().total,
            bytes.len()
        );
        measure(&format!("{name}-evaluate-63-targets"), || {
            black_box(model.evaluate(black_box(&examples))?);
            Ok(())
        })?;
        measure(&format!("{name}-train-64-targets-including-clone"), || {
            let mut copy = model.clone();
            black_box(copy.train_batch(black_box(&batch), OptimizerConfig::default())?);
            Ok(())
        })?;
        measure(&format!("{name}-load"), || {
            black_box(CausalLm::from_bytes(black_box(&bytes))?);
            Ok(())
        })?;
    }
    let artifact =
        train_factorized_bpe(b"hello hello 12 12!!", FactorizedTrainingConfig::default())?;
    let tokenizer = FactorizedTokenizer::from_artifact(&artifact)?;
    for (name, text) in [
        ("short-spans", "a1!".repeat(16_384)),
        ("long-span", "a".repeat(49_152)),
    ] {
        let (tokens, stats) = tokenizer.encode_with_stats(&text)?;
        assert_eq!(tokenizer.decode_bytes(&tokens)?, text.as_bytes());
        println!(
            "m2-{name}: input_bytes={} tokens={} output_capacity_bytes={} temporary_peak_bytes={}",
            text.len(),
            tokens.len(),
            stats.output_capacity_bytes,
            stats.temporary_peak_bytes
        );
        measure(&format!("m2-{name}-encode"), || {
            black_box(tokenizer.encode(black_box(&text))?);
            Ok(())
        })?;
    }
    Ok(())
}

fn measure(
    name: &str,
    mut operation: impl FnMut() -> Result<(), Box<dyn Error>>,
) -> Result<(), Box<dyn Error>> {
    let warmup = Instant::now();
    while warmup.elapsed() < Duration::from_millis(25) {
        operation()?;
    }
    let started = Instant::now();
    let mut iterations = 0_u64;
    while started.elapsed() < Duration::from_millis(200) {
        operation()?;
        iterations += 1;
    }
    let elapsed = started.elapsed().as_secs_f64();
    println!(
        "{name}: iterations={iterations} elapsed_ms={:.3} us_per_iteration={:.3}",
        elapsed * 1000.0,
        elapsed * 1e6 / iterations as f64
    );
    Ok(())
}
