//! Fixed M4 factorial experiment; uses the M3 sampling/scoring implementation.
use crate::{
    m3::{self, EncodedText, SampleRng},
    m4_corpus::{self, ROOT},
    metrics::process_peak_memory_bytes,
};
use packtok_core::TokenId;

use packtok_model::{CausalLm, IdMapping, ModelConfig, OptimizerConfig, TrainingExample};
use packtok_tokenizer::{BpeTokenizer, FactorizedTokenizer, Tokenizer};
use packtok_train::{
    BpeTrainingConfig, FactorizedTrainingConfig, load_corpus, train_bpe_with_provenance,
    train_factorized_bpe_with_provenance_report,
};
use std::{error::Error, fmt::Write, fs, path::Path, time::Instant};

fn nll_per_byte(
    loss_per_token: f64,
    targets: usize,
    represented_bytes: usize,
) -> Result<f64, Box<dyn Error>> {
    if targets == 0 || represented_bytes == 0 || !loss_per_token.is_finite() {
        return Err("invalid byte-normalized loss inputs".into());
    }
    Ok(loss_per_token * targets as f64 / represented_bytes as f64)
}

const CONFIG: ModelConfig = ModelConfig {
    hidden_size: 16,
    context_length: 16,
};
const SEEDS: [u64; 3] = [20_261_007, 20_261_008, 20_261_009];

struct Representation {
    splits: Vec<EncodedText>,
    mapping: IdMapping,
    synthetic: bool,
}

impl Representation {
    fn tokens_for(
        &self,
        variant: &str,
        tokens: &[TokenId],
    ) -> Result<Vec<TokenId>, Box<dyn Error>> {
        tokens
            .iter()
            .map(|&token| {
                Ok(match variant {
                    "B" => self.mapping.to_packed(token.local)?,
                    "C" => TokenId::new(0, self.mapping.to_global(token)?),
                    _ => token,
                })
            })
            .collect()
    }
    fn original(&self, variant: &str, tokens: &[TokenId]) -> Result<Vec<TokenId>, Box<dyn Error>> {
        tokens
            .iter()
            .map(|&token| {
                Ok(match variant {
                    "B" => TokenId::new(0, self.mapping.to_global(token)?),
                    "C" if token.pack == 0 => self.mapping.to_packed(token.local)?,
                    "C" => return Err("flat model token has nonzero pack".into()),
                    _ => token,
                })
            })
            .collect()
    }
    fn model(&self, variant: &str, seed: u64) -> Result<CausalLm, Box<dyn Error>> {
        Ok(match variant {
            "A" | "C" => CausalLm::new_flat(CONFIG, self.mapping.len() as u32, seed)?,
            "B" => CausalLm::new_factorized_permuted(
                CONFIG,
                self.mapping.packs().to_vec(),
                seed,
                &self.mapping.embedding_permutation(),
            )?,
            "D" => CausalLm::new_factorized(CONFIG, self.mapping.packs().to_vec(), seed)?,
            _ => return Err("unknown variant".into()),
        })
    }
    fn mapped_split(&self, variant: &str, index: usize) -> Result<EncodedText, Box<dyn Error>> {
        let original = &self.splits[index];
        let tokens = self.tokens_for(variant, &original.tokens)?;
        if self.original(variant, &tokens)? != original.tokens {
            return Err("mapping changes tokenizer sequence".into());
        }
        Ok(EncodedText {
            raw_bytes: original.raw_bytes,
            tokens,
            token_byte_lengths: original.token_byte_lengths.clone(),
        })
    }
}

#[derive(Clone)]
struct ResultRow {
    variant: String,
    regime: String,
    seed: u64,
    bits: f64,
    params: usize,
    macs: u64,
    updates: usize,
    train_ms: f64,
    test_ms: f64,
}

pub(super) fn run(corpus_name: &str, label: &str) -> Result<(), Box<dyn Error>> {
    if !["tiny", "large"].contains(&corpus_name)
        || label.is_empty()
        || !label
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-')
    {
        return Err("invalid corpus or run label".into());
    }
    let root = Path::new(ROOT);
    let run_path = root.join("runs").join(format!("{corpus_name}-{label}"));
    fs::create_dir_all(root.join("runs"))?;
    fs::create_dir(&run_path)?; // fails closed: never overwrite results
    let environment = m3::environment_record(&run_path)?;
    m3::write_new(&run_path.join("environment.txt"), environment.as_bytes())?;
    let directory = if corpus_name == "tiny" {
        Path::new("fixtures/m3-model")
    } else {
        Path::new("experiments/m4-ablation/corpus-v2")
    };
    let raw: Vec<Vec<u8>> = ["train", "validation", "test"]
        .iter()
        .map(|name| fs::read(directory.join(format!("{name}.txt"))))
        .collect::<Result<_, _>>()?;
    m4_corpus::guard(&raw)?;
    let text: Vec<&str> = raw
        .iter()
        .map(|b| std::str::from_utf8(b))
        .collect::<Result<_, _>>()?;
    // Artifact provenance preserves the spelling of the input path. Path::join
    // introduces backslashes on Windows, unlike the frozen M3 forward-slash
    // path; canonical spelling keeps descriptive metadata byte-identical too.
    let train_path = format!(
        "{}/train.txt",
        directory.to_string_lossy().replace('\\', "/")
    );
    let corpus = load_corpus(Path::new(&train_path))?;
    let mut report = format!(
        "PackTok M4 v1 corpus={corpus_name} hidden=16 context=16 batch=4 seeds={SEEDS:?} Adam=M3-default budget=\"M3 MAC v1\" no-early-stop\n{environment}"
    );
    let start = Instant::now();
    let m1 = train_bpe_with_provenance(
        corpus.bytes(),
        BpeTrainingConfig::default(),
        corpus.provenance(),
    )?;
    let m1_ms = start.elapsed().as_secs_f64() * 1000.;
    let m1_again = train_bpe_with_provenance(
        corpus.bytes(),
        BpeTrainingConfig::default(),
        corpus.provenance(),
    )?;
    let start = Instant::now();
    let m2 = train_factorized_bpe_with_provenance_report(
        corpus.bytes(),
        FactorizedTrainingConfig::default(),
        corpus.provenance(),
    )?;
    let m2_ms = start.elapsed().as_secs_f64() * 1000.;
    let m2_again = train_factorized_bpe_with_provenance_report(
        corpus.bytes(),
        FactorizedTrainingConfig::default(),
        corpus.provenance(),
    )?;
    let m1_bytes = m1.to_bytes()?;
    let m2_bytes = m2.artifact.to_bytes()?;
    if m1_bytes != m1_again.to_bytes()? || m2_bytes != m2_again.artifact.to_bytes()? {
        return Err("non-deterministic tokenizer artifacts".into());
    }
    if corpus_name == "tiny" {
        for (name, bytes) in [("m1-flat-v2", &m1_bytes), ("m2-factorized-v3", &m2_bytes)] {
            let path = format!("experiments/m3-model/artifacts/{name}.packtok");
            if fs::read(path)? != *bytes {
                return Err("frozen M3 tokenizer differs".into());
            }
        }
    }
    m3::write_new(&run_path.join("m1.packtok"), &m1_bytes)?;
    m3::write_new(&run_path.join("m2.packtok"), &m2_bytes)?;
    writeln!(
        report,
        "tokenizer_training m1_ms={m1_ms:.3} m2_ms={m2_ms:.3} m1_bytes={} m2_bytes={} both_trained_twice_identical=true m2_allocation={:?}",
        m1_bytes.len(),
        m2_bytes.len(),
        m2.packs
    )?;
    let flat = BpeTokenizer::from_artifact(&m1)?;
    let packed = FactorizedTokenizer::from_artifact(&m2.artifact)?;
    let mut m1_splits = Vec::new();
    let mut m2_splits = Vec::new();
    for (i, &input) in text.iter().enumerate() {
        let start = Instant::now();
        let timed_tokens = flat.encode(input)?;
        let elapsed = start.elapsed();
        let a = m3::encode_m1_text(&flat, &m1, input)?;
        if timed_tokens != a.tokens {
            return Err("M1 encode mismatch".into());
        }
        writeln!(
            report,
            "encode tokenizer=M1 split={i} bytes={} tokens={} bytes_per_token={:.8} ms={:.3} mib_per_s={:.6}",
            input.len(),
            a.tokens.len(),
            input.len() as f64 / a.tokens.len() as f64,
            elapsed.as_secs_f64() * 1000.,
            input.len() as f64 / 1048576. / elapsed.as_secs_f64()
        )?;
        let start = Instant::now();
        let timed_tokens = packed.encode(input)?;
        let elapsed = start.elapsed();
        let d = m3::encode_m2_text(&packed, &m2.artifact, input)?;
        if timed_tokens != d.tokens {
            return Err("M2 encode mismatch".into());
        }
        writeln!(
            report,
            "encode tokenizer=M2 split={i} bytes={} tokens={} bytes_per_token={:.8} ms={:.3} mib_per_s={:.6}",
            input.len(),
            d.tokens.len(),
            input.len() as f64 / d.tokens.len() as f64,
            elapsed.as_secs_f64() * 1000.,
            input.len() as f64 / 1048576. / elapsed.as_secs_f64()
        )?;
        if flat.decode_bytes(&a.tokens)? != raw[i] || packed.decode_bytes(&d.tokens)? != raw[i] {
            return Err("tokenizer roundtrip failed".into());
        }
        m1_splits.push(a);
        m2_splits.push(d);
    }
    let c_map = IdMapping::flatten(
        m2.artifact
            .registry()
            .packs()
            .iter()
            .map(|p| packtok_model::PackVocabulary {
                pack_id: p.id(),
                token_count: p.local_token_count(),
            })
            .collect(),
    )?;
    let expansions = (0..m1.flat_bpe().ok_or("no flat vocabulary")?.vocabulary_size())
        .map(|id| flat.decode_bytes(&[TokenId::new(0, id)]))
        .collect::<Result<Vec<_>, _>>()?;
    let b_map = IdMapping::balanced(&expansions, c_map.packs().len())?;
    for (name, map) in [("B", &b_map), ("C", &c_map)] {
        let bytes = map.to_bytes();
        if IdMapping::from_bytes(&bytes)? != *map {
            return Err("mapping serialization failure".into());
        }
        m3::write_new(&run_path.join(format!("{name}.ptmap")), &bytes)?;
        writeln!(
            report,
            "mapping variant={name} rows={} packs={:?}",
            map.len(),
            map.packs()
        )?;
    }
    let m1_repr = Representation {
        splits: m1_splits,
        mapping: b_map,
        synthetic: true,
    };
    let m2_repr = Representation {
        splits: m2_splits,
        mapping: c_map,
        synthetic: false,
    };
    let updates = if corpus_name == "tiny" { 120 } else { 2000 };
    let mut rows = Vec::new();
    for seed in SEEDS {
        let mut budget = 0;
        for regime in ["schedule", "mac"] {
            for variant in ["A", "B", "C", "D"] {
                let representation = if variant == "A" || variant == "B" {
                    &m1_repr
                } else {
                    &m2_repr
                };
                let tokenizer: &dyn Tokenizer = if representation.synthetic {
                    &flat
                } else {
                    &packed
                };
                let row = train_run(
                    &run_path,
                    representation,
                    tokenizer,
                    variant,
                    regime,
                    seed,
                    updates,
                    budget,
                    &mut report,
                )?;
                if regime == "schedule" && variant == "A" {
                    budget = row.macs;
                }
                println!(
                    "completed corpus={corpus_name} regime={regime} variant={variant} seed={seed} updates={} test_bits_per_byte={:.8}",
                    row.updates, row.bits
                );
                rows.push(row);
                fs::write(run_path.join("progress.txt"), &report)?; // retained on failure
            }
        }
    }
    for regime in ["schedule", "mac"] {
        for variant in ["A", "B", "C", "D"] {
            let selected: Vec<_> = rows
                .iter()
                .filter(|r| r.regime == regime && r.variant == variant)
                .collect();
            writeln!(
                report,
                "summary regime={regime} variant={variant} bits_mean_sd={} train_ms_mean_sd={} test_ms_mean_sd={} params={} macs={:?} updates={:?}",
                m3::mean_sd(&selected.iter().map(|r| r.bits).collect::<Vec<_>>()),
                m3::mean_sd(&selected.iter().map(|r| r.train_ms).collect::<Vec<_>>()),
                m3::mean_sd(&selected.iter().map(|r| r.test_ms).collect::<Vec<_>>()),
                selected[0].params,
                selected.iter().map(|r| r.macs).collect::<Vec<_>>(),
                selected.iter().map(|r| r.updates).collect::<Vec<_>>()
            )?;
        }
        let mut diffs: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
        for seed in SEEDS {
            let val = |v: &str| {
                rows.iter()
                    .find(|r| r.seed == seed && r.regime == regime && r.variant == v)
                    .unwrap()
                    .bits
            };
            let [a, b, c, d] = [val("A"), val("B"), val("C"), val("D")];
            let delta = [b - a, c - a, d - a, (d - c) - (b - a)];
            writeln!(
                report,
                "paired regime={regime} seed={seed} B-A={:.8} C-A={:.8} D-A={:.8} interaction={:.8}",
                delta[0], delta[1], delta[2], delta[3]
            )?;
            for i in 0..4 {
                diffs[i].push(delta[i]);
            }
        }
        for (name, values) in ["B-A", "C-A", "D-A", "interaction"].iter().zip(&diffs) {
            writeln!(
                report,
                "paired_summary regime={regime} difference={name} mean_sd={}",
                m3::mean_sd(values)
            )?;
        }
    }
    writeln!(
        report,
        "process_peak_memory_bytes={:?}",
        process_peak_memory_bytes()
    )?;
    m3::write_new(&run_path.join("run-summary.txt"), report.as_bytes())?;
    print!("{report}");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn train_run(
    path: &Path,
    repr: &Representation,
    tokenizer: &dyn Tokenizer,
    variant: &str,
    regime: &str,
    seed: u64,
    fixed: usize,
    budget: u64,
    report: &mut String,
) -> Result<ResultRow, Box<dyn Error>> {
    let training_started = Instant::now();
    let mut model = repr.model(variant, seed)?;
    let train = repr.mapped_split(variant, 0)?;
    let validation = repr.mapped_split(variant, 1)?;
    let test = repr.mapped_split(variant, 2)?;
    let val_examples = m3::evaluation_examples(&validation, CONFIG.context_length);
    let mut rng = SampleRng::new(seed ^ 0xa341_316c_9e37_79b9);
    let mut updates = 0;
    let mut macs = 0_u64;
    let mut train_bytes = 0_usize;
    let mut validation_ms = 0.;
    let mut next_checkpoint = 1;
    loop {
        if (regime == "schedule" && updates >= fixed) || (regime == "mac" && macs >= budget) {
            break;
        }
        if updates >= 20_000 {
            return Err("MAC budget cap exceeded".into());
        }
        let windows =
            m3::sample_windows(&train.tokens, &train.token_byte_lengths, &mut rng, 4, 16)?;
        let batch: Vec<_> = windows
            .iter()
            .map(|w| TrainingExample {
                inputs: &w.inputs,
                targets: &w.targets,
            })
            .collect();
        let step = model.train_batch(&batch, OptimizerConfig::default())?;
        macs += step.estimated_macs;
        updates += 1;
        train_bytes += windows.iter().map(|w| w.target_bytes).sum::<usize>();
        let checkpoint = if regime == "schedule" {
            updates == 1 || updates % (fixed / 4) == 0 || updates == fixed
        } else {
            updates == 1 || macs >= budget * next_checkpoint / 4 || macs >= budget
        };
        if checkpoint {
            if regime == "mac" && macs >= budget * next_checkpoint / 4 {
                next_checkpoint += 1;
            }
            let start = Instant::now();
            let metrics = model.evaluate(&val_examples)?;
            let ms = start.elapsed().as_secs_f64() * 1000.;
            validation_ms += ms;
            let bits = metrics.loss_per_token * metrics.target_tokens as f64
                / validation.target_bytes() as f64
                / std::f64::consts::LN_2;
            writeln!(
                report,
                "curve regime={regime} variant={variant} seed={seed} updates={updates} macs={macs} train_loss_token={:.8} validation_loss_token={:.8} validation_bits_byte={bits:.8} validation_ms={ms:.3}",
                step.loss_per_token, metrics.loss_per_token
            )?;
        }
    }
    let train_ms = training_started.elapsed().as_secs_f64() * 1000.;
    let val = m3::evaluate_split(&model, &validation)?;
    let scored = m3::evaluate_split(&model, &test)?;
    let m = &scored.metrics;
    let nll = nll_per_byte(m.loss_per_token, m.target_tokens, scored.scored_bytes)?;
    let bits = nll / std::f64::consts::LN_2;
    let parameters = model.parameter_counts();
    let bytes = model.to_bytes()?;
    let loaded = CausalLm::from_bytes(&bytes)?;
    if loaded.to_bytes()? != bytes || loaded.evaluate(&m3::evaluation_examples(&test, 16))? != *m {
        return Err("serialized model changes scoring".into());
    }
    let input = "At the harbor";
    let original = tokenizer.encode(input)?;
    let prompt = repr.tokens_for(variant, &original)?;
    let generated = loaded.greedy_generate(&prompt, 8)?;
    let decoded = tokenizer.decode_bytes(&repr.original(variant, &generated)?)?;
    if !decoded.starts_with(input.as_bytes()) {
        return Err("generation changes raw prompt".into());
    }
    m3::write_new(
        &path.join(format!("{regime}-{variant}-seed-{seed}.ptlm")),
        &bytes,
    )?;
    let test_ms = scored.elapsed.as_secs_f64() * 1000.;
    writeln!(
        report,
        "result regime={regime} variant={variant} seed={seed} params={} embedding_params={} backbone_params={} head_params={} updates={updates} train_targets={} train_target_bytes={train_bytes} macs={macs} train_ms={train_ms:.3} curve_validation_ms={validation_ms:.3} final_validation_ms={:.3} validation_bits_byte={:.8} test_ms={test_ms:.3} test_bits_byte={bits:.8} test_nll_byte={nll:.8} test_loss_token={:.8} test_target_bytes={} test_targets={} bytes_token={:.8} model_bytes={} parameter_adam_gradient_bytes={} peak_process_bytes={:?}",
        parameters.total,
        parameters.embeddings,
        parameters.backbone,
        parameters.output_head,
        updates * 64,
        val.elapsed.as_secs_f64() * 1000.,
        val.metrics.loss_per_token * val.metrics.target_tokens as f64
            / val.scored_bytes as f64
            / std::f64::consts::LN_2,
        m.loss_per_token,
        scored.scored_bytes,
        m.target_tokens,
        scored.bytes_per_token,
        bytes.len(),
        parameters.total * 16,
        process_peak_memory_bytes()
    )?;
    if variant == "B" || variant == "D" {
        writeln!(
            report,
            "heads regime={regime} variant={variant} seed={seed} {}",
            m3::factorized_stats(&model, &test, m)?
        )?;
    } else {
        writeln!(
            report,
            "heads regime={regime} variant={variant} seed={seed} output_rows={} logits_per_target={} head_parameters={} measured_scoring_ms={test_ms:.3}",
            repr.mapping.len(),
            repr.mapping.len(),
            parameters.output_head
        )?;
    }
    let mut distribution = std::collections::BTreeMap::<u16, usize>::new();
    for token in &generated[prompt.len()..] {
        *distribution.entry(token.pack).or_default() += 1;
    }
    writeln!(
        report,
        "generation regime={regime} variant={variant} seed={seed} prompt_exact=true valid_ids=true decode_success=true selected_pack_counts={distribution:?} hex={}",
        m3::bytes_hex(&decoded)
    )?;
    Ok(ResultRow {
        variant: variant.to_owned(),
        regime: regime.to_owned(),
        seed,
        bits,
        params: parameters.total,
        macs,
        updates,
        train_ms,
        test_ms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use packtok_format::Artifact;
    #[test]
    fn normalization_uses_target_bytes_not_token_count_or_whole_file() {
        assert_eq!(nll_per_byte(3., 2, 5).unwrap(), 1.2);
        assert!(nll_per_byte(3., 2, 0).is_err());
        assert!(nll_per_byte(3., 0, 5).is_err());
        assert!(nll_per_byte(f64::NAN, 2, 5).is_err());
        let tokens = vec![TokenId::new(0, 0), TokenId::new(0, 1), TokenId::new(0, 2)];
        let split = EncodedText {
            raw_bytes: 12,
            tokens,
            token_byte_lengths: vec![7, 2, 3],
        };
        assert_eq!(split.target_bytes(), 5);
        assert_eq!(m3::evaluation_examples(&split, 16).len(), 2);
    }
    use packtok_train::{train_bpe, train_factorized_bpe};
    fn fixtures() -> (Artifact, Artifact, BpeTokenizer, FactorizedTokenizer) {
        let bytes = b"At the harbor 123! At the harbor 123! At the harbor 123!";
        let a = train_bpe(bytes, BpeTrainingConfig::default()).unwrap();
        let d = train_factorized_bpe(bytes, FactorizedTrainingConfig::default()).unwrap();
        let at = BpeTokenizer::from_artifact(&a).unwrap();
        let dt = FactorizedTokenizer::from_artifact(&d).unwrap();
        (a, d, at, dt)
    }
    #[test]
    fn all_four_mapping_embeddings_reproducibility_overfit_and_generation() {
        let (a, d, at, dt) = fixtures();
        let c_map = IdMapping::flatten(
            d.registry()
                .packs()
                .iter()
                .map(|p| packtok_model::PackVocabulary {
                    pack_id: p.id(),
                    token_count: p.local_token_count(),
                })
                .collect(),
        )
        .unwrap();
        let expansions = (0..a.flat_bpe().unwrap().vocabulary_size())
            .map(|id| at.decode_bytes(&[TokenId::new(0, id)]).unwrap())
            .collect::<Vec<_>>();
        let b_map = IdMapping::balanced(&expansions, c_map.packs().len()).unwrap();
        let ar = Representation {
            splits: vec![],
            mapping: b_map,
            synthetic: true,
        };
        let dr = Representation {
            splits: vec![],
            mapping: c_map,
            synthetic: false,
        };
        let prompt = "At the harbor";
        let ai = at.encode(prompt).unwrap();
        let di = dt.encode(prompt).unwrap();
        let a_model = ar.model("A", 19).unwrap();
        let b_model = ar.model("B", 19).unwrap();
        assert_eq!(
            a_model.hidden_states(&ai).unwrap(),
            b_model
                .hidden_states(&ar.tokens_for("B", &ai).unwrap())
                .unwrap()
        );
        let c_model = dr.model("C", 19).unwrap();
        let d_model = dr.model("D", 19).unwrap();
        assert_eq!(
            c_model
                .hidden_states(&dr.tokens_for("C", &di).unwrap())
                .unwrap(),
            d_model.hidden_states(&di).unwrap()
        );
        for variant in ["A", "B", "C", "D"] {
            let (r, t): (&Representation, &dyn Tokenizer) = if variant == "A" || variant == "B" {
                (&ar, &at)
            } else {
                (&dr, &dt)
            };
            let ids = r.tokens_for(variant, &t.encode(prompt).unwrap()).unwrap();
            assert_eq!(
                r.original(variant, &ids).unwrap(),
                t.encode(prompt).unwrap()
            );
            let inputs = &ids[..ids.len() - 1];
            let targets = &ids[1..];
            let batch = [TrainingExample { inputs, targets }];
            let mut model = r.model(variant, 19).unwrap();
            let mut duplicate = model.clone();
            let before = model.evaluate(&batch).unwrap().loss_per_token;
            for _ in 0..80 {
                model
                    .train_batch(&batch, OptimizerConfig::default())
                    .unwrap();
                duplicate
                    .train_batch(&batch, OptimizerConfig::default())
                    .unwrap();
            }
            assert_eq!(model, duplicate);
            assert!(model.evaluate(&batch).unwrap().loss_per_token < before * 0.4);
            assert_eq!(model.parameter_counts().total, model.parameters().len());
            let n = inputs.len() as u64;
            let dim = 16_u64;
            let head_rows = targets
                .iter()
                .map(|target| {
                    if variant == "A" || variant == "C" {
                        r.mapping.len() as u64
                    } else {
                        r.mapping.packs().len() as u64
                            + r.mapping
                                .packs()
                                .iter()
                                .find(|p| p.pack_id == target.pack)
                                .unwrap()
                                .token_count as u64
                    }
                })
                .sum::<u64>();
            assert_eq!(
                model.estimate_training_macs(&batch).unwrap(),
                (3 * n - 1) * dim * dim + 3 * head_rows * dim
            );
            let loaded = CausalLm::from_bytes(&model.to_bytes().unwrap()).unwrap();
            assert_eq!(
                loaded.hidden_states(inputs).unwrap(),
                model.hidden_states(inputs).unwrap()
            );
            let generated = loaded.greedy_generate(&ids, 8).unwrap();
            assert!(
                t.decode_bytes(&r.original(variant, &generated).unwrap())
                    .unwrap()
                    .starts_with(prompt.as_bytes())
            );
        }
    }
}
