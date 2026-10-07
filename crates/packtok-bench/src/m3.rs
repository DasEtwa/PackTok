use std::collections::BTreeSet;
use std::error::Error;
use std::fmt::Write as FmtWrite;
use std::fs::{self, OpenOptions};
use std::io::Write as IoWrite;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

use packtok_core::DEFAULT_BYTE_FALLBACK_PACK_ID;
use packtok_format::Artifact;
use packtok_model::{
    CausalLm, EvaluationMetrics, ModelConfig, OptimizerConfig, PackVocabulary, TrainingExample,
};
use packtok_tokenizer::{BpeTokenizer, FactorizedTokenizer, TokenId, Tokenizer};
use packtok_train::{
    BpeTrainingConfig, FactorizedTrainingConfig, fnv1a64, load_corpus, train_bpe_with_provenance,
    train_factorized_bpe_with_provenance_report,
};

use crate::metrics::process_peak_memory_bytes;

const TRAIN_PATH: &str = "fixtures/m3-model/train.txt";
const VALIDATION_PATH: &str = "fixtures/m3-model/validation.txt";
const TEST_PATH: &str = "fixtures/m3-model/test.txt";
const ARTIFACT_DIRECTORY: &str = "experiments/m3-model/artifacts";
const RUN_DIRECTORY: &str = "experiments/m3-model/runs";
const MODEL_CONFIG: ModelConfig = ModelConfig {
    hidden_size: 16,
    context_length: 16,
};
const OPTIMIZER: OptimizerConfig = OptimizerConfig {
    learning_rate: 0.01,
    beta1: 0.9,
    beta2: 0.999,
    epsilon: 1.0e-8,
    gradient_clip_norm: 1.0,
};
const TRAINING_BATCH_SIZE: usize = 4;
const SAME_BACKBONE_STEPS: usize = 120;
const MAX_BUDGET_STEPS: usize = 2_000;
const CHECKPOINT_INTERVAL: usize = 20;
const REPETITIONS: usize = 3;
const GENERATION_NEW_TOKENS: usize = 8;

pub(super) struct EncodedText {
    pub(super) raw_bytes: usize,
    pub(super) tokens: Vec<TokenId>,
    pub(super) token_byte_lengths: Vec<usize>,
}

impl EncodedText {
    pub(super) fn token_count(&self) -> usize {
        self.tokens.len()
    }

    fn bytes_per_token(&self) -> f64 {
        self.raw_bytes as f64 / self.tokens.len() as f64
    }

    fn target_count(&self) -> usize {
        self.tokens.len().saturating_sub(1)
    }

    pub(super) fn target_bytes(&self) -> usize {
        self.token_byte_lengths.iter().skip(1).sum()
    }
}

pub(super) struct Window {
    pub(super) inputs: Vec<TokenId>,
    pub(super) targets: Vec<TokenId>,
    pub(super) target_bytes: usize,
}

struct TrainingDataset<'a> {
    tokens: &'a [TokenId],
    token_byte_lengths: &'a [usize],
    packs: &'a [PackVocabulary],
}

struct TrainingPlan {
    seed: u64,
    sample_seed: u64,
    fixed_steps: Option<usize>,
    mac_budget: Option<u64>,
}

struct TrainedRun {
    model: CausalLm,
    steps: usize,
    tokens_processed: u64,
    bytes_processed: u64,
    estimated_macs: u64,
    training_time: Duration,
    final_train_loss: f64,
    validation_curve: Vec<(usize, f64, f64)>,
}

struct RunContext<'a> {
    run_path: &'a Path,
    regime: &'static str,
    tokenizer_name: &'static str,
    seed: u64,
    validation: &'a EncodedText,
    test: &'a EncodedText,
    tokenizer: &'a dyn Tokenizer,
}

pub(super) struct SplitResult {
    pub(super) metrics: EvaluationMetrics,
    pub(super) elapsed: Duration,
    pub(super) raw_bytes: usize,
    pub(super) scored_bytes: usize,
    pub(super) tokens: usize,
    pub(super) bytes_per_token: f64,
}

struct RepetitionResult {
    regime: &'static str,
    tokenizer: &'static str,
    seed: u64,
    steps: usize,
    model_parameters: usize,
    embedding_parameters: usize,
    backbone_parameters: usize,
    output_parameters: usize,
    train_tokens: u64,
    train_bytes: u64,
    estimated_macs: u64,
    train_ms: f64,
    final_train_loss: f64,
    validation: SplitResult,
    test: SplitResult,
    model_artifact_bytes: usize,
    model_parameter_state_bytes: usize,
    generated_hex: String,
    factorized_test_stats: Option<String>,
    validation_curve: Vec<(usize, f64, f64)>,
}

/// Trains both tokenizer artifacts and model variants, preserving every result.
pub fn run(label: &str) -> Result<(), Box<dyn Error>> {
    if label.is_empty()
        || !label
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
    {
        return Err("run label must contain only ASCII letters, digits, '-' or '_'".into());
    }
    let run_path = Path::new(RUN_DIRECTORY).join(label);
    fs::create_dir_all(RUN_DIRECTORY)?;
    fs::create_dir(&run_path).map_err(|error| {
        format!(
            "cannot create {}; use a new run label so earlier evidence is preserved: {error}",
            run_path.display()
        )
    })?;
    fs::create_dir_all(ARTIFACT_DIRECTORY)?;

    let train = load_corpus(Path::new(TRAIN_PATH))?;
    let validation_bytes = fs::read(VALIDATION_PATH)?;
    let test_bytes = fs::read(TEST_PATH)?;
    validate_utf8(&validation_bytes, VALIDATION_PATH)?;
    validate_utf8(&test_bytes, TEST_PATH)?;
    reject_shared_32_byte_passages(
        ("train", train.bytes()),
        ("validation", &validation_bytes),
        ("test", &test_bytes),
    )?;

    let m1_first = train_bpe_with_provenance(
        train.bytes(),
        BpeTrainingConfig::default(),
        train.provenance(),
    )?;
    let m1_second = train_bpe_with_provenance(
        train.bytes(),
        BpeTrainingConfig::default(),
        train.provenance(),
    )?;
    let m1_artifact_bytes = m1_first.to_bytes()?;
    if m1_artifact_bytes != m1_second.to_bytes()? {
        return Err("M1 tokenizer training was not byte deterministic".into());
    }
    let m2_first = train_factorized_bpe_with_provenance_report(
        train.bytes(),
        FactorizedTrainingConfig::default(),
        train.provenance(),
    )?;
    let m2_second = train_factorized_bpe_with_provenance_report(
        train.bytes(),
        FactorizedTrainingConfig::default(),
        train.provenance(),
    )?;
    let m2_artifact_bytes = m2_first.artifact.to_bytes()?;
    if m2_artifact_bytes != m2_second.artifact.to_bytes()? {
        return Err("M2 tokenizer training was not byte deterministic".into());
    }
    let m1_artifact_path = Path::new(ARTIFACT_DIRECTORY).join("m1-flat-v2.packtok");
    let m2_artifact_path = Path::new(ARTIFACT_DIRECTORY).join("m2-factorized-v3.packtok");
    write_or_check(&m1_artifact_path, &m1_artifact_bytes)?;
    write_or_check(&m2_artifact_path, &m2_artifact_bytes)?;
    let m1_artifact = Artifact::from_bytes(&m1_artifact_bytes)?;
    let m2_artifact = Artifact::from_bytes(&m2_artifact_bytes)?;
    let m1 = BpeTokenizer::from_artifact(&m1_artifact)?;
    let m2 = FactorizedTokenizer::from_artifact(&m2_artifact)?;
    let m1_pack_vocabularies = m1_packs(&m1_artifact)?;
    let m2_pack_vocabularies: Vec<PackVocabulary> = m2_artifact
        .registry()
        .packs()
        .iter()
        .map(|pack| PackVocabulary {
            pack_id: pack.id(),
            token_count: pack.local_token_count(),
        })
        .collect();

    let training_text = std::str::from_utf8(train.bytes())?;
    let validation_text = std::str::from_utf8(&validation_bytes)?;
    let test_text = std::str::from_utf8(&test_bytes)?;
    let m1_training = encode_m1_text(&m1, &m1_artifact, training_text)?;
    let m2_training = encode_m2_text(&m2, &m2_artifact, training_text)?;
    let m1_validation = encode_m1_text(&m1, &m1_artifact, validation_text)?;
    let m2_validation = encode_m2_text(&m2, &m2_artifact, validation_text)?;
    let m1_test = encode_m1_text(&m1, &m1_artifact, test_text)?;
    let m2_test = encode_m2_text(&m2, &m2_artifact, test_text)?;
    if m1_training.tokens.len() <= MODEL_CONFIG.context_length
        || m2_training.tokens.len() <= MODEL_CONFIG.context_length
    {
        return Err("training split is too short for the configured context".into());
    }
    let m1_training_data = TrainingDataset {
        tokens: &m1_training.tokens,
        token_byte_lengths: &m1_training.token_byte_lengths,
        packs: &m1_pack_vocabularies,
    };
    let m2_training_data = TrainingDataset {
        tokens: &m2_training.tokens,
        token_byte_lengths: &m2_training.token_byte_lengths,
        packs: &m2_pack_vocabularies,
    };

    let environment = environment_record(&run_path)?;
    let mut report = String::new();
    writeln!(report, "PackTok M3 model comparison")?;
    writeln!(report, "run_label={label}")?;
    writeln!(report, "run_directory={}", run_path.display())?;
    writeln!(report)?;
    writeln!(report, "{environment}")?;
    writeln!(
        report,
        "model_config=causal_tanh_rnn hidden={} context={} optimizer=Adam lr={} beta1={} beta2={} epsilon={} gradient_clip={} batch_size={} same_backbone_steps={} repetitions={} seed_base=20261007 precision=f32 device=CPU thread_count=1 weight_decay=0 warmup=0",
        MODEL_CONFIG.hidden_size,
        MODEL_CONFIG.context_length,
        OPTIMIZER.learning_rate,
        OPTIMIZER.beta1,
        OPTIMIZER.beta2,
        OPTIMIZER.epsilon,
        OPTIMIZER.gradient_clip_norm,
        TRAINING_BATCH_SIZE,
        SAME_BACKBONE_STEPS,
        REPETITIONS
    )?;
    writeln!(
        report,
        "tokenizer_config=M1 target_vocab=512 max_merges=256 min_pair_frequency=2; M2 router=lexical-v1 target_vocab=512 max_learned_tokens=256 min_pair_frequency=2"
    )?;
    writeln!(
        report,
        "tokenizer_determinism=each tokenizer trained twice from identical M3 train bytes; serialized artifact bytes matched"
    )?;
    writeln!(
        report,
        "tokenizer_artifact=M1 bytes={} path={}; M2 bytes={} path={}",
        m1_artifact_bytes.len(),
        m1_artifact_path.display(),
        m2_artifact_bytes.len(),
        m2_artifact_path.display()
    )?;
    writeln!(
        report,
        "dataset_train path={} bytes={} fnv1a64={:016x}; validation path={} bytes={} fnv1a64={:016x}; test path={} bytes={} fnv1a64={:016x}; normalization=none separators=none",
        TRAIN_PATH,
        train.bytes().len(),
        fnv1a64(train.bytes()),
        VALIDATION_PATH,
        validation_bytes.len(),
        fnv1a64(&validation_bytes),
        TEST_PATH,
        test_bytes.len(),
        fnv1a64(&test_bytes)
    )?;
    writeln!(
        report,
        "token_counts train M1={} M2={}; validation M1={} M2={}; test M1={} M2={}",
        m1_training.token_count(),
        m2_training.token_count(),
        m1_validation.token_count(),
        m2_validation.token_count(),
        m1_test.token_count(),
        m2_test.token_count()
    )?;
    writeln!(report, "m2_train_pack_allocation={:?}", m2_first.packs)?;
    writeln!(report)?;
    writeln!(report, "repeat_results_begin")?;

    let mut results = Vec::new();
    for repetition in 0..REPETITIONS {
        let seed = 20_261_007_u64 + repetition as u64;
        let sample_seed = seed ^ 0xa341_316c_9e37_79b9;
        let m1_a = train_model(
            &m1_training_data,
            TrainingPlan {
                seed,
                sample_seed,
                fixed_steps: Some(SAME_BACKBONE_STEPS),
                mac_budget: None,
            },
            &m1_validation,
        )?;
        let mac_budget = m1_a.estimated_macs;
        let m2_a = train_model(
            &m2_training_data,
            TrainingPlan {
                seed,
                sample_seed,
                fixed_steps: Some(SAME_BACKBONE_STEPS),
                mac_budget: None,
            },
            &m2_validation,
        )?;
        let m1_b = train_model(
            &m1_training_data,
            TrainingPlan {
                seed,
                sample_seed,
                fixed_steps: None,
                mac_budget: Some(mac_budget),
            },
            &m1_validation,
        )?;
        let m2_b = train_model(
            &m2_training_data,
            TrainingPlan {
                seed,
                sample_seed,
                fixed_steps: None,
                mac_budget: Some(mac_budget),
            },
            &m2_validation,
        )?;
        for (regime, tokenizer_name, trained, validation, test, tokenizer) in [
            (
                "A-same-backbone",
                "M1-flat",
                m1_a,
                &m1_validation,
                &m1_test,
                &m1 as &dyn Tokenizer,
            ),
            (
                "A-same-backbone",
                "M2-factorized",
                m2_a,
                &m2_validation,
                &m2_test,
                &m2 as &dyn Tokenizer,
            ),
            (
                "B-matched-MAC",
                "M1-flat",
                m1_b,
                &m1_validation,
                &m1_test,
                &m1 as &dyn Tokenizer,
            ),
            (
                "B-matched-MAC",
                "M2-factorized",
                m2_b,
                &m2_validation,
                &m2_test,
                &m2 as &dyn Tokenizer,
            ),
        ] {
            let result = finish_run(
                RunContext {
                    run_path: &run_path,
                    regime,
                    tokenizer_name,
                    seed,
                    validation,
                    test,
                    tokenizer,
                },
                trained,
            )?;
            append_repetition(&mut report, repetition + 1, &result)?;
            results.push(result);
        }
    }
    writeln!(report)?;
    writeln!(report, "summary_begin")?;
    append_summary(&mut report, &results)?;
    writeln!(
        report,
        "process_peak_memory_bytes={}",
        process_peak_memory_bytes()?
    )?;
    writeln!(report, "repeat_results_end")?;
    write_new(&run_path.join("environment.txt"), environment.as_bytes())?;
    write_new(&run_path.join("run-summary.txt"), report.as_bytes())?;
    print!("{report}");
    Ok(())
}

fn train_model(
    dataset: &TrainingDataset<'_>,
    plan: TrainingPlan,
    validation: &EncodedText,
) -> Result<TrainedRun, Box<dyn Error>> {
    let training_started = Instant::now();
    let mut model = if dataset.packs.len() == 1 && dataset.packs[0].pack_id == 0 {
        CausalLm::new_flat(MODEL_CONFIG, dataset.packs[0].token_count, plan.seed)?
    } else {
        CausalLm::new_factorized(MODEL_CONFIG, dataset.packs.to_vec(), plan.seed)?
    };
    let mut rng = SampleRng::new(plan.sample_seed);
    let mut steps = 0_usize;
    let mut tokens_processed = 0_u64;
    let mut bytes_processed = 0_u64;
    let mut estimated_macs = 0_u64;
    let mut final_train_loss = 0.0_f64;
    let mut validation_curve = Vec::new();
    let validation_examples = evaluation_examples(validation, MODEL_CONFIG.context_length);

    loop {
        if plan.fixed_steps.is_some_and(|fixed| steps >= fixed)
            || plan
                .mac_budget
                .is_some_and(|budget| estimated_macs >= budget)
        {
            break;
        }
        if steps >= MAX_BUDGET_STEPS {
            return Err(
                format!("training exceeded {MAX_BUDGET_STEPS} updates before its budget").into(),
            );
        }

        let windows = sample_windows(
            dataset.tokens,
            dataset.token_byte_lengths,
            &mut rng,
            TRAINING_BATCH_SIZE,
            MODEL_CONFIG.context_length,
        )?;
        let examples: Vec<TrainingExample<'_>> = windows
            .iter()
            .map(|window| TrainingExample {
                inputs: &window.inputs,
                targets: &window.targets,
            })
            .collect();
        let step = model.train_batch(&examples, OPTIMIZER)?;
        steps += 1;
        tokens_processed = tokens_processed
            .checked_add(u64::try_from(step.target_tokens)?)
            .ok_or("training token count overflow")?;
        let step_bytes = windows.iter().try_fold(0_u64, |sum, window| {
            sum.checked_add(u64::try_from(window.target_bytes).map_err(|_| "window byte overflow")?)
                .ok_or("training byte count overflow")
        })?;
        bytes_processed = bytes_processed
            .checked_add(step_bytes)
            .ok_or("training byte count overflow")?;
        estimated_macs = estimated_macs
            .checked_add(step.estimated_macs)
            .ok_or("training MAC estimate overflow")?;
        final_train_loss = step.loss_per_token;

        if steps == 1
            || steps % CHECKPOINT_INTERVAL == 0
            || plan.fixed_steps == Some(steps)
            || plan
                .mac_budget
                .is_some_and(|budget| estimated_macs >= budget)
        {
            let validation_metrics = model.evaluate(&validation_examples)?;
            let score = bits_per_byte(&validation_metrics, validation.target_bytes());
            validation_curve.push((steps, validation_metrics.loss_per_token, score));
        }
    }
    Ok(TrainedRun {
        model,
        steps,
        tokens_processed,
        bytes_processed,
        estimated_macs,
        training_time: training_started.elapsed(),
        final_train_loss,
        validation_curve,
    })
}

fn finish_run(
    context: RunContext<'_>,
    trained: TrainedRun,
) -> Result<RepetitionResult, Box<dyn Error>> {
    let validation_result = evaluate_split(&trained.model, context.validation)?;
    let test_result = evaluate_split(&trained.model, context.test)?;
    let input_prompt = "At the harbor";
    let prompt_tokens = context.tokenizer.encode(input_prompt)?;
    let generated_tokens = trained
        .model
        .greedy_generate(&prompt_tokens, GENERATION_NEW_TOKENS)?;
    let generated_bytes = context.tokenizer.decode_bytes(&generated_tokens)?;
    if !generated_bytes.starts_with(input_prompt.as_bytes()) {
        return Err(format!(
            "{} {} generation did not preserve its input prompt",
            context.regime, context.tokenizer_name
        )
        .into());
    }
    let generated_hex = bytes_hex(&generated_bytes);
    let model_bytes = trained.model.to_bytes()?;
    let model_path = context.run_path.join(format!(
        "{}-{}-seed-{}.ptlm",
        context.regime.to_ascii_lowercase(),
        context.tokenizer_name.to_ascii_lowercase(),
        context.seed
    ));
    write_new(&model_path, &model_bytes)?;

    let parameters = trained.model.parameter_counts();
    let model_parameter_state_bytes = parameters
        .total
        .checked_mul(4)
        .and_then(|bytes| bytes.checked_mul(4))
        .ok_or("model state byte estimate overflow")?;
    let factorized_test_stats = if context.tokenizer_name == "M2-factorized" {
        Some(factorized_stats(
            &trained.model,
            context.test,
            &test_result.metrics,
        )?)
    } else {
        None
    };
    Ok(RepetitionResult {
        regime: context.regime,
        tokenizer: context.tokenizer_name,
        seed: context.seed,
        steps: trained.steps,
        model_parameters: parameters.total,
        embedding_parameters: parameters.embeddings,
        backbone_parameters: parameters.backbone,
        output_parameters: parameters.output_head,
        train_tokens: trained.tokens_processed,
        train_bytes: trained.bytes_processed,
        estimated_macs: trained.estimated_macs,
        train_ms: trained.training_time.as_secs_f64() * 1000.0,
        final_train_loss: trained.final_train_loss,
        validation: validation_result,
        test: test_result,
        model_artifact_bytes: model_bytes.len(),
        model_parameter_state_bytes,
        generated_hex,
        factorized_test_stats,
        validation_curve: trained.validation_curve,
    })
}

pub(super) fn evaluate_split(
    model: &CausalLm,
    split: &EncodedText,
) -> Result<SplitResult, Box<dyn Error>> {
    let examples = evaluation_examples(split, MODEL_CONFIG.context_length);
    let started = Instant::now();
    let metrics = model.evaluate(&examples)?;
    let elapsed = started.elapsed();
    Ok(SplitResult {
        metrics,
        elapsed,
        raw_bytes: split.raw_bytes,
        scored_bytes: split.target_bytes(),
        tokens: split.token_count(),
        bytes_per_token: split.bytes_per_token(),
    })
}

pub(super) fn evaluation_examples(
    split: &EncodedText,
    context_length: usize,
) -> Vec<TrainingExample<'_>> {
    let mut examples = Vec::with_capacity(split.target_count());
    for target_index in 1..split.tokens.len() {
        let context_start = target_index.saturating_sub(context_length);
        examples.push(TrainingExample {
            inputs: &split.tokens[context_start..target_index],
            targets: &split.tokens[target_index..target_index + 1],
        });
    }
    examples
}

pub(super) fn sample_windows(
    tokens: &[TokenId],
    token_byte_lengths: &[usize],
    rng: &mut SampleRng,
    batch_size: usize,
    context_length: usize,
) -> Result<Vec<Window>, Box<dyn Error>> {
    if tokens.len() != token_byte_lengths.len() || tokens.len() <= context_length {
        return Err("training token/byte lengths are inconsistent or too short".into());
    }
    let choices = tokens.len() - context_length;
    let mut windows = Vec::with_capacity(batch_size);
    for _ in 0..batch_size {
        let start = rng.next() as usize % choices;
        let target_start = start + 1;
        let end = target_start + context_length;
        let target_bytes = token_byte_lengths[target_start..end]
            .iter()
            .try_fold(0_usize, |sum, length| sum.checked_add(*length))
            .ok_or("sampled target byte count overflow")?;
        windows.push(Window {
            inputs: tokens[start..start + context_length].to_vec(),
            targets: tokens[target_start..end].to_vec(),
            target_bytes,
        });
    }
    Ok(windows)
}

pub(super) fn encode_m1_text(
    tokenizer: &BpeTokenizer,
    artifact: &Artifact,
    text: &str,
) -> Result<EncodedText, Box<dyn Error>> {
    let tokens = tokenizer.encode(text)?;
    let model = artifact.flat_bpe().ok_or("missing flat BPE model")?;
    let lengths = tokens
        .iter()
        .map(|token| {
            model
                .byte_length(token.local)
                .ok_or_else(|| "flat token has no byte length".into())
        })
        .collect::<Result<Vec<usize>, Box<dyn Error>>>()?;
    encoded_text(text, tokens, lengths)
}

pub(super) fn encode_m2_text(
    tokenizer: &FactorizedTokenizer,
    artifact: &Artifact,
    text: &str,
) -> Result<EncodedText, Box<dyn Error>> {
    let tokens = tokenizer.encode(text)?;
    let model = artifact
        .factorized_bpe()
        .ok_or("missing factorized BPE model")?;
    let fallback_id = artifact.registry().byte_fallback().pack_id();
    let mut lengths = Vec::with_capacity(tokens.len());
    for token in &tokens {
        let length = if token.pack == fallback_id {
            1
        } else {
            model
                .pack(token.pack)
                .and_then(|pack| pack.byte_length(token.local))
                .ok_or_else(|| format!("no byte length for token {}/{}", token.pack, token.local))?
        };
        lengths.push(length);
    }
    encoded_text(text, tokens, lengths)
}

fn encoded_text(
    text: &str,
    tokens: Vec<TokenId>,
    token_byte_lengths: Vec<usize>,
) -> Result<EncodedText, Box<dyn Error>> {
    if tokens.len() != token_byte_lengths.len() {
        return Err("token byte-length table does not match token count".into());
    }
    let represented_bytes = token_byte_lengths
        .iter()
        .try_fold(0_usize, |sum, length| sum.checked_add(*length))
        .ok_or("encoded byte total overflow")?;
    if represented_bytes != text.len() {
        return Err("encoded token byte lengths do not cover the raw UTF-8 input".into());
    }
    Ok(EncodedText {
        raw_bytes: text.len(),
        tokens,
        token_byte_lengths,
    })
}

fn m1_packs(artifact: &Artifact) -> Result<Vec<PackVocabulary>, Box<dyn Error>> {
    let model = artifact.flat_bpe().ok_or("missing flat BPE model")?;
    Ok(vec![PackVocabulary {
        pack_id: 0,
        token_count: model.vocabulary_size(),
    }])
}

fn bits_per_byte(metrics: &EvaluationMetrics, target_bytes: usize) -> f64 {
    metrics.loss_per_token * metrics.target_tokens as f64
        / target_bytes as f64
        / std::f64::consts::LN_2
}

fn evaluate_split_bits(split: &SplitResult) -> f64 {
    bits_per_byte(&split.metrics, split.scored_bytes)
}

pub(super) fn factorized_stats(
    model: &CausalLm,
    split: &EncodedText,
    metrics: &EvaluationMetrics,
) -> Result<String, Box<dyn Error>> {
    let packs = model
        .pack_vocabularies()
        .ok_or("missing factorized model pack registry")?;
    let mut target_counts = vec![0_usize; packs.len()];
    let mut transitions = 0_usize;
    for pair in split.tokens.windows(2) {
        if pair[0].pack != pair[1].pack {
            transitions += 1;
        }
    }
    for token in split.tokens.iter().skip(1) {
        let index = packs
            .binary_search_by_key(&token.pack, |pack| pack.pack_id)
            .map_err(|_| format!("test target references absent M2 pack {}", token.pack))?;
        target_counts[index] += 1;
    }
    let pack_fraction = metrics.pack_head_logit_fraction.unwrap_or(0.0);
    let mut output = String::new();
    write!(
        output,
        "pack_accuracy={:.6} local_accuracy_given_gold_pack={:.6} active_local_head_size_mean={:.6} pack_head_logit_fraction={pack_fraction:.6} local_head_logit_fraction={:.6} pack_transitions={transitions}",
        metrics.pack_accuracy.unwrap_or(0.0),
        metrics.local_accuracy_given_pack.unwrap_or(0.0),
        metrics.average_active_local_head_size.unwrap_or(0.0),
        1.0 - pack_fraction
    )?;
    let byte_fallback_count = split
        .tokens
        .iter()
        .skip(1)
        .filter(|token| token.pack == DEFAULT_BYTE_FALLBACK_PACK_ID)
        .count();
    write!(
        output,
        " byte_fallback_targets={byte_fallback_count}/{} byte_fallback_fraction={:.6}",
        metrics.target_tokens,
        byte_fallback_count as f64 / metrics.target_tokens as f64
    )?;
    for (index, pack) in packs.iter().enumerate() {
        let accuracy = metrics.per_pack.get(index).map_or(0.0, |row| {
            if row.targets == 0 {
                0.0
            } else {
                row.correct as f64 / row.targets as f64
            }
        });
        write!(
            output,
            " pack[{}]={{targets:{},accuracy:{accuracy:.6},local_count:{}}}",
            pack.pack_id, target_counts[index], pack.token_count
        )?;
    }
    Ok(output)
}

fn append_repetition(
    report: &mut String,
    repetition: usize,
    result: &RepetitionResult,
) -> Result<(), Box<dyn Error>> {
    let validation = &result.validation;
    let test = &result.test;
    writeln!(
        report,
        "rep={repetition} regime={} tokenizer={} seed={} params={} embedding_params={} backbone_params={} output_params={} train_steps={} train_tokens={} train_target_bytes={} estimated_macs={} train_ms={:.3} final_train_loss_token={:.8}",
        result.regime,
        result.tokenizer,
        result.seed,
        result.model_parameters,
        result.embedding_parameters,
        result.backbone_parameters,
        result.output_parameters,
        result.steps,
        result.train_tokens,
        result.train_bytes,
        result.estimated_macs,
        result.train_ms,
        result.final_train_loss
    )?;
    writeln!(
        report,
        "  validation raw_bytes={} scored_bytes={} tokens={} bytes_per_token={:.6} loss_token={:.8} bits_per_byte={:.8} nll_per_byte={:.8} perplexity_token_specific={:.8} eval_ms={:.3}",
        validation.raw_bytes,
        validation.scored_bytes,
        validation.tokens,
        validation.bytes_per_token,
        validation.metrics.loss_per_token,
        evaluate_split_bits(validation),
        validation.metrics.loss_per_token * validation.metrics.target_tokens as f64
            / validation.scored_bytes as f64,
        validation.metrics.loss_per_token.exp(),
        validation.elapsed.as_secs_f64() * 1000.0
    )?;
    writeln!(
        report,
        "  test raw_bytes={} scored_bytes={} tokens={} bytes_per_token={:.6} loss_token={:.8} bits_per_byte={:.8} nll_per_byte={:.8} perplexity_token_specific={:.8} token_accuracy={:.6} eval_ms={:.3} model_artifact_bytes={} parameter_adam_and_gradient_bytes={} generated_bytes_hex={}",
        test.raw_bytes,
        test.scored_bytes,
        test.tokens,
        test.bytes_per_token,
        test.metrics.loss_per_token,
        evaluate_split_bits(test),
        test.metrics.loss_per_token * test.metrics.target_tokens as f64 / test.scored_bytes as f64,
        test.metrics.loss_per_token.exp(),
        test.metrics.token_accuracy,
        test.elapsed.as_secs_f64() * 1000.0,
        result.model_artifact_bytes,
        result.model_parameter_state_bytes,
        result.generated_hex
    )?;
    if let Some(stats) = &result.factorized_test_stats {
        writeln!(report, "  M2_test_head_stats {stats}")?;
    }
    for (step, loss, bits) in &result.validation_curve {
        writeln!(
            report,
            "  validation_curve step={step} loss_token={loss:.8} bits_per_byte={bits:.8}"
        )?;
    }
    Ok(())
}

fn append_summary(report: &mut String, results: &[RepetitionResult]) -> Result<(), Box<dyn Error>> {
    for regime in ["A-same-backbone", "B-matched-MAC"] {
        for tokenizer in ["M1-flat", "M2-factorized"] {
            let selected: Vec<&RepetitionResult> = results
                .iter()
                .filter(|result| result.regime == regime && result.tokenizer == tokenizer)
                .collect();
            let validation_bits: Vec<f64> = selected
                .iter()
                .map(|result| evaluate_split_bits(&result.validation))
                .collect();
            let test_bits: Vec<f64> = selected
                .iter()
                .map(|result| evaluate_split_bits(&result.test))
                .collect();
            let train_ms: Vec<f64> = selected.iter().map(|result| result.train_ms).collect();
            let evaluation_ms: Vec<f64> = selected
                .iter()
                .map(|result| result.test.elapsed.as_secs_f64() * 1000.0)
                .collect();
            writeln!(
                report,
                "summary regime={regime} tokenizer={tokenizer} repetitions={} validation_bits_per_byte_mean_sd_variance={} test_bits_per_byte_mean_sd_variance={} train_ms_mean_sd_variance={} test_eval_ms_mean_sd_variance={}",
                selected.len(),
                mean_sd(&validation_bits),
                mean_sd(&test_bits),
                mean_sd(&train_ms),
                mean_sd(&evaluation_ms)
            )?;
        }
    }
    Ok(())
}

pub(super) fn mean_sd(values: &[f64]) -> String {
    if values.is_empty() {
        return "unavailable".to_owned();
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f64>()
        / values.len() as f64;
    format!("{mean:.8}+/-{:.8};variance={variance:.12}", variance.sqrt())
}

pub(super) fn environment_record(run_path: &Path) -> Result<String, Box<dyn Error>> {
    let rustc = Command::new("rustc")
        .arg("--version")
        .arg("--verbose")
        .output()?;
    let cargo = Command::new("cargo")
        .arg("--version")
        .arg("--verbose")
        .output()?;
    let commit = Command::new("git").args(["rev-parse", "HEAD"]).output()?;
    let rustc_text = String::from_utf8_lossy(&rustc.stdout);
    let cargo_text = String::from_utf8_lossy(&cargo.stdout);
    let commit_text = String::from_utf8_lossy(&commit.stdout).trim().to_owned();
    let cpu = cpu_name().unwrap_or_else(|| "unavailable".to_owned());
    let os_version = os_version().unwrap_or_else(|| "unavailable".to_owned());
    let logical_cpus = std::thread::available_parallelism().map_or_else(
        |_| "unavailable".to_owned(),
        |count| count.get().to_string(),
    );
    let mut output = format!(
        "source_commit={commit_text} os={} os_version={} arch={} cpu={} logical_cpus={} build_profile={} debug_assertions={}",
        std::env::consts::OS,
        os_version,
        std::env::consts::ARCH,
        cpu,
        logical_cpus,
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        cfg!(debug_assertions)
    );
    writeln!(output)?;
    writeln!(output, "rustc_verbose=")?;
    writeln!(output, "{}", rustc_text.trim())?;
    writeln!(output, "cargo_verbose=")?;
    writeln!(output, "{}", cargo_text.trim())?;
    writeln!(
        output,
        "thread_policy=one explicit training/evaluation thread"
    )?;
    writeln!(output, "run_directory={}", run_path.display())?;
    Ok(output)
}

fn cpu_name() -> Option<String> {
    #[cfg(target_os = "linux")]
    {
        let cpuinfo = fs::read_to_string("/proc/cpuinfo").ok()?;
        return cpuinfo.lines().find_map(|line| {
            line.strip_prefix("model name\t: ")
                .or_else(|| line.strip_prefix("Hardware\t: "))
                .map(str::to_owned)
        });
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        let output = Command::new("powershell.exe")
            .args([
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)",
            ])
            .creation_flags(0x0800_0000)
            .output()
            .ok()?;
        return output
            .status
            .success()
            .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned());
    }
    #[allow(unreachable_code)]
    None
}

fn os_version() -> Option<String> {
    #[cfg(target_os = "linux")]
    {
        let release = fs::read_to_string("/etc/os-release").ok()?;
        return release.lines().find_map(|line| {
            line.strip_prefix("PRETTY_NAME=")
                .map(|value| value.trim_matches('"').to_owned())
        });
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        let output = Command::new("powershell.exe")
            .args([
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "(Get-CimInstance Win32_OperatingSystem | Select-Object -First 1 -ExpandProperty Caption) + ' ' + (Get-CimInstance Win32_OperatingSystem | Select-Object -First 1 -ExpandProperty BuildNumber)",
            ])
            .creation_flags(0x0800_0000)
            .output()
            .ok()?;
        return output
            .status
            .success()
            .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned());
    }
    #[allow(unreachable_code)]
    None
}

fn validate_utf8(bytes: &[u8], path: &str) -> Result<(), Box<dyn Error>> {
    std::str::from_utf8(bytes)
        .map(|_| ())
        .map_err(|error| format!("{path} is not UTF-8 at byte {}", error.valid_up_to()).into())
}

fn reject_shared_32_byte_passages(
    first: (&str, &[u8]),
    second: (&str, &[u8]),
    third: (&str, &[u8]),
) -> Result<(), Box<dyn Error>> {
    let splits = [first, second, third];
    let passage_sets: Vec<BTreeSet<[u8; 32]>> = splits
        .iter()
        .map(|(_, bytes)| {
            bytes
                .windows(32)
                .map(|window| {
                    let mut passage = [0_u8; 32];
                    passage.copy_from_slice(window);
                    passage
                })
                .collect()
        })
        .collect();
    for left in 0..splits.len() {
        for right in left + 1..splits.len() {
            if passage_sets[left]
                .iter()
                .any(|passage| passage_sets[right].contains(passage))
            {
                return Err(format!(
                    "32-byte raw passage overlap between {} and {} split files",
                    splits[left].0, splits[right].0
                )
                .into());
            }
        }
    }
    Ok(())
}

fn write_or_check(path: &Path, bytes: &[u8]) -> Result<(), Box<dyn Error>> {
    if path.exists() {
        if fs::read(path)? != bytes {
            return Err(format!(
                "{} exists with different bytes; preserving it and refusing overwrite",
                path.display()
            )
            .into());
        }
        return Ok(());
    }
    write_new(path, bytes)
}

pub(super) fn write_new(path: &Path, bytes: &[u8]) -> Result<(), Box<dyn Error>> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    if let Err(error) = file.write_all(bytes).and_then(|()| file.sync_all()) {
        drop(file);
        let cleanup = fs::remove_file(path);
        return match cleanup {
            Ok(()) => Err(error.into()),
            Err(cleanup_error) => Err(format!(
                "writing {} failed ({error}) and cleanup failed ({cleanup_error})",
                path.display()
            )
            .into()),
        };
    }
    Ok(())
}

pub(super) fn bytes_hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(output, "{byte:02x}");
    }
    output
}

pub(super) struct SampleRng {
    state: u64,
}

impl SampleRng {
    pub(super) fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9e37_79b9_7f4a_7c15
            } else {
                seed
            },
        }
    }

    pub(super) fn next(&mut self) -> u64 {
        let mut value = self.state;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.state = value;
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use packtok_model::{ModelConfig, PackVocabulary};
    use packtok_train::{train_bpe, train_factorized_bpe};

    #[test]
    fn flat_and_factorized_generation_decode_with_the_prompt_bytes_intact() {
        let prompt = "At the harbor";
        let config = ModelConfig {
            hidden_size: 8,
            context_length: 8,
        };

        let flat_artifact = train_bpe(prompt.as_bytes(), BpeTrainingConfig::default()).unwrap();
        let flat_tokenizer = BpeTokenizer::from_artifact(&flat_artifact).unwrap();
        let flat_model = CausalLm::new_flat(
            config,
            flat_artifact.flat_bpe().unwrap().vocabulary_size(),
            101,
        )
        .unwrap();
        let flat_prompt = flat_tokenizer.encode(prompt).unwrap();
        let flat_generated = flat_model.greedy_generate(&flat_prompt, 3).unwrap();
        let flat_bytes = flat_tokenizer.decode_bytes(&flat_generated).unwrap();
        assert!(flat_bytes.starts_with(prompt.as_bytes()));

        let factored_artifact =
            train_factorized_bpe(prompt.as_bytes(), FactorizedTrainingConfig::default()).unwrap();
        let factorized_tokenizer = FactorizedTokenizer::from_artifact(&factored_artifact).unwrap();
        let packs: Vec<PackVocabulary> = factored_artifact
            .registry()
            .packs()
            .iter()
            .map(|pack| PackVocabulary {
                pack_id: pack.id(),
                token_count: pack.local_token_count(),
            })
            .collect();
        let factorized_model = CausalLm::new_factorized(config, packs, 101).unwrap();
        let factorized_prompt = factorized_tokenizer.encode(prompt).unwrap();
        let factorized_generated = factorized_model
            .greedy_generate(&factorized_prompt, 3)
            .unwrap();
        let factorized_bytes = factorized_tokenizer
            .decode_bytes(&factorized_generated)
            .unwrap();
        assert!(factorized_bytes.starts_with(prompt.as_bytes()));
    }

    #[test]
    fn corpus_split_guard_rejects_shared_raw_passages() {
        let left = b"one distinct manually authored passage for training.";
        let right = b"another separate text sample for held out model evaluation.";
        let test = b"third split contains its own independent sentences.";
        assert!(
            reject_shared_32_byte_passages(("train", left), ("validation", right), ("test", test))
                .is_ok()
        );
        let overlap = b"one distinct manually authored passage for training.";
        assert!(
            reject_shared_32_byte_passages(
                ("train", left),
                ("validation", overlap),
                ("test", test)
            )
            .is_err()
        );
    }
}
