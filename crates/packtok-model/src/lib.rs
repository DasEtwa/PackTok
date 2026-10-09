#![forbid(unsafe_code)]

//! Small CPU-only autoregressive models used by controlled PackTok experiments.
//!
//! The crate implements one causal tanh recurrent cell, a flat vocabulary head,
//! and a genuinely factorized pack/local head. It deliberately has no tensor or
//! tokenizer-training dependency so model-facing contracts can be replaced
//! independently of tokenizer artifacts.

use std::fmt;

use packtok_core::{PackId, TokenId};

const MODEL_MAGIC: &[u8; 8] = b"PACKLM3\0";
const MODEL_FORMAT_VERSION: u16 = 1;
const MAX_HIDDEN_SIZE: usize = 1024;
const MAX_CONTEXT_LENGTH: usize = 4096;
const MAX_LOCAL_VOCABULARY: u32 = 1_000_000;
const MAX_PACK_COUNT: usize = 1024;
const MAX_PARAMETER_COUNT: usize = 16_777_216;
const MAX_MODEL_BYTES: usize = 64 * 1024 * 1024;

/// Dimensions shared by the flat and factorized model variants.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModelConfig {
    /// Width of token embeddings, recurrent state, and output projections.
    pub hidden_size: usize,
    /// Maximum number of input positions processed in one training context.
    pub context_length: usize,
}

impl ModelConfig {
    /// Validates dimensions before any parameter allocation.
    pub fn validate(self) -> Result<Self, ModelError> {
        if !(2..=MAX_HIDDEN_SIZE).contains(&self.hidden_size) {
            return Err(ModelError::InvalidConfig("hidden_size is outside 2..=1024"));
        }
        if !(1..=MAX_CONTEXT_LENGTH).contains(&self.context_length) {
            return Err(ModelError::InvalidConfig(
                "context_length is outside 1..=4096",
            ));
        }
        Ok(self)
    }
}

/// The number of local IDs declared in one pack vocabulary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PackVocabulary {
    /// Stable pack identity; local IDs are interpreted only inside this pack.
    pub pack_id: PackId,
    /// Valid local IDs occupy `0..token_count`.
    pub token_count: u32,
}

/// Adam hyperparameters and global gradient clipping used by one update.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OptimizerConfig {
    /// Positive Adam learning rate.
    pub learning_rate: f32,
    /// First-moment decay in `[0, 1)`.
    pub beta1: f32,
    /// Second-moment decay in `[0, 1)`.
    pub beta2: f32,
    /// Positive denominator stabilizer.
    pub epsilon: f32,
    /// Maximum global L2 norm before gradients are rescaled.
    pub gradient_clip_norm: f32,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1.0e-8,
            gradient_clip_norm: 1.0,
        }
    }
}

impl OptimizerConfig {
    fn validate(self) -> Result<Self, ModelError> {
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err(ModelError::InvalidOptimizer(
                "learning_rate must be finite and positive",
            ));
        }
        if !self.beta1.is_finite() || !(0.0..1.0).contains(&self.beta1) {
            return Err(ModelError::InvalidOptimizer(
                "beta1 must be finite and in [0, 1)",
            ));
        }
        if !self.beta2.is_finite() || !(0.0..1.0).contains(&self.beta2) {
            return Err(ModelError::InvalidOptimizer(
                "beta2 must be finite and in [0, 1)",
            ));
        }
        if !self.epsilon.is_finite() || self.epsilon <= 0.0 {
            return Err(ModelError::InvalidOptimizer(
                "epsilon must be finite and positive",
            ));
        }
        if !self.gradient_clip_norm.is_finite() || self.gradient_clip_norm <= 0.0 {
            return Err(ModelError::InvalidOptimizer(
                "gradient_clip_norm must be finite and positive",
            ));
        }
        Ok(self)
    }
}

/// One causal training/evaluation window and its next-token targets.
///
/// Training normally uses equal-length input and target slices. Evaluation may
/// supply one target, in which case it is predicted after the final input token.
#[derive(Clone, Copy, Debug)]
pub struct TrainingExample<'a> {
    /// Input token context; the model resets its hidden state before each example.
    pub inputs: &'a [TokenId],
    /// Aligned next-token targets, or a single target after the complete context.
    pub targets: &'a [TokenId],
}

/// Parameter partition used to make model-budget differences explicit.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ParameterCounts {
    /// Input token embedding rows, including every local pack namespace.
    pub embeddings: usize,
    /// Position vectors, recurrent matrix, and recurrent bias.
    pub backbone: usize,
    /// Flat head or pack plus all local-token heads, including biases.
    pub output_head: usize,
    /// Sum of the three preceding partitions.
    pub total: usize,
}

/// Loss and accuracy aggregates for a batch or evaluation split.
#[derive(Clone, Debug, PartialEq)]
pub struct EvaluationMetrics {
    /// Number of target tokens included in the result.
    pub target_tokens: usize,
    /// Mean negative log likelihood in nats per token.
    pub loss_per_token: f64,
    /// Flat loss, or pack loss for a factorized model, in nats per token.
    pub primary_head_loss_per_token: Option<f64>,
    /// Conditional local loss for a factorized model, in nats per token.
    pub local_loss_per_token: Option<f64>,
    /// Greedy next-token accuracy. For M2 both the pack and conditional local ID
    /// must match the target.
    pub token_accuracy: f64,
    /// Factorized pack accuracy, absent for a flat model.
    pub pack_accuracy: Option<f64>,
    /// Local-token accuracy evaluated under each ground-truth target pack.
    pub local_accuracy_given_pack: Option<f64>,
    /// Target counts and pack-head accuracy in ascending pack-ID order.
    pub per_pack: Vec<PackAccuracy>,
    /// Mean number of local logits evaluated for factorized target losses.
    pub average_active_local_head_size: Option<f64>,
    /// Aggregate pack-logit count divided by pack plus target-pack local logits.
    pub pack_head_logit_fraction: Option<f64>,
}

/// Factorized pack accuracy row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PackAccuracy {
    /// Stable target pack ID.
    pub pack_id: PackId,
    /// Number of targets from this pack.
    pub targets: usize,
    /// Correct pack predictions for those targets.
    pub correct: usize,
}

/// Metrics returned after one Adam update.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrainStepMetrics {
    /// Mean pre-update target loss in nats per token.
    pub loss_per_token: f64,
    /// Number of supervised targets in the batch.
    pub target_tokens: usize,
    /// Analytical multiply-accumulate estimate for this batch.
    pub estimated_macs: u64,
    /// Global gradient norm before optional clipping.
    pub gradient_norm: f64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PackLayout {
    pack_id: PackId,
    token_count: usize,
    row_start: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum HeadKind {
    Flat { vocabulary_size: usize },
    Factorized { packs: Vec<PackLayout> },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ParameterLayout {
    embedding: usize,
    position: usize,
    recurrent: usize,
    recurrent_bias: usize,
    flat_output: usize,
    flat_output_bias: usize,
    pack_output: usize,
    pack_output_bias: usize,
    local_output: usize,
    local_output_bias: usize,
    parameter_count: usize,
}

/// Safe, deterministic, single-threaded causal language model for M3.
///
/// Flat models use one embedding table and one global head. Factorized models
/// use distinct input rows per `(pack, local)` ID and a pack head followed by
/// pack-local output heads. Parameters are stored in one row-major vector so
/// optimizer accounting and binary serialization use an explicit stable order.
#[derive(Clone, Debug, PartialEq)]
pub struct CausalLm {
    config: ModelConfig,
    kind: HeadKind,
    seed: u64,
    layout: ParameterLayout,
    weights: Vec<f32>,
    first_moment: Vec<f32>,
    second_moment: Vec<f32>,
    optimizer_step: u64,
}

impl CausalLm {
    /// Creates an M1-style model with pack `0` and one flat output head.
    pub fn new_flat(
        config: ModelConfig,
        vocabulary_size: u32,
        seed: u64,
    ) -> Result<Self, ModelError> {
        Self::flat_model(config, vocabulary_size, seed, None)
    }

    fn flat_model(
        config: ModelConfig,
        vocabulary_size: u32,
        seed: u64,
        parameter_bytes: Option<&[u8]>,
    ) -> Result<Self, ModelError> {
        let config = config.validate()?;
        if !(256..=MAX_LOCAL_VOCABULARY).contains(&vocabulary_size) {
            return Err(ModelError::InvalidVocabulary);
        }
        Self::build(
            config,
            HeadKind::Flat {
                vocabulary_size: usize::try_from(vocabulary_size)
                    .map_err(|_| ModelError::LengthOverflow)?,
            },
            seed,
            parameter_bytes,
        )
    }

    /// Creates an M2-style model with independent input and output rows per pack.
    ///
    /// Pack declarations are sorted by numeric ID before parameter assignment;
    /// duplicate IDs, empty packs, or oversized total vocabularies are rejected.
    pub fn new_factorized(
        config: ModelConfig,
        packs: Vec<PackVocabulary>,
        seed: u64,
    ) -> Result<Self, ModelError> {
        Self::factorized_model(config, packs, seed, None)
    }

    fn factorized_model(
        config: ModelConfig,
        mut packs: Vec<PackVocabulary>,
        seed: u64,
        parameter_bytes: Option<&[u8]>,
    ) -> Result<Self, ModelError> {
        let config = config.validate()?;
        if packs.is_empty() || packs.len() > MAX_PACK_COUNT {
            return Err(ModelError::InvalidVocabulary);
        }
        packs.sort_unstable_by_key(|pack| pack.pack_id);
        let mut layouts = Vec::with_capacity(packs.len());
        let mut row_start = 0_usize;
        for (index, pack) in packs.iter().enumerate() {
            if pack.token_count == 0 || pack.token_count > MAX_LOCAL_VOCABULARY {
                return Err(ModelError::InvalidVocabulary);
            }
            if index > 0 && packs[index - 1].pack_id == pack.pack_id {
                return Err(ModelError::DuplicatePackId(pack.pack_id));
            }
            layouts.push(PackLayout {
                pack_id: pack.pack_id,
                token_count: usize::try_from(pack.token_count)
                    .map_err(|_| ModelError::LengthOverflow)?,
                row_start,
            });
            row_start = row_start
                .checked_add(
                    usize::try_from(pack.token_count).map_err(|_| ModelError::LengthOverflow)?,
                )
                .ok_or(ModelError::LengthOverflow)?;
            if row_start > usize::try_from(MAX_LOCAL_VOCABULARY).unwrap_or(usize::MAX) {
                return Err(ModelError::InvalidVocabulary);
            }
        }
        Self::build(
            config,
            HeadKind::Factorized { packs: layouts },
            seed,
            parameter_bytes,
        )
    }

    fn build(
        config: ModelConfig,
        kind: HeadKind,
        seed: u64,
        parameter_bytes: Option<&[u8]>,
    ) -> Result<Self, ModelError> {
        let rows = match &kind {
            HeadKind::Flat { vocabulary_size } => *vocabulary_size,
            HeadKind::Factorized { packs } => packs
                .last()
                .and_then(|pack| pack.row_start.checked_add(pack.token_count))
                .ok_or(ModelError::InvalidVocabulary)?,
        };
        let hidden = config.hidden_size;
        let mut cursor = 0_usize;
        let embedding = reserve_parameters(&mut cursor, rows, hidden)?;
        let position = reserve_parameters(&mut cursor, config.context_length, hidden)?;
        let recurrent = reserve_parameters(&mut cursor, hidden, hidden)?;
        let recurrent_bias = reserve_parameters(&mut cursor, 1, hidden)?;
        let (
            flat_output,
            flat_output_bias,
            pack_output,
            pack_output_bias,
            local_output,
            local_output_bias,
        ) = match &kind {
            HeadKind::Flat { vocabulary_size } => {
                let weight = reserve_parameters(&mut cursor, *vocabulary_size, hidden)?;
                let bias = reserve_parameters(&mut cursor, 1, *vocabulary_size)?;
                (weight, bias, 0, 0, 0, 0)
            }
            HeadKind::Factorized { packs } => {
                let pack_weight = reserve_parameters(&mut cursor, packs.len(), hidden)?;
                let pack_bias = reserve_parameters(&mut cursor, 1, packs.len())?;
                let local_weight = reserve_parameters(&mut cursor, rows, hidden)?;
                let local_bias = reserve_parameters(&mut cursor, 1, rows)?;
                (0, 0, pack_weight, pack_bias, local_weight, local_bias)
            }
        };
        if cursor > MAX_PARAMETER_COUNT {
            return Err(ModelError::ModelTooLarge);
        }
        // The shared layout calculation must agree with the serialized body
        // before weights or Adam moments are allocated.
        if parameter_bytes.is_some_and(|bytes| bytes.len() != cursor * 4) {
            return Err(ModelError::MalformedArtifact(
                "parameter count does not match dimensions",
            ));
        }
        let mut weights = vec![0.0_f32; cursor];
        if let Some(bytes) = parameter_bytes {
            for (weight, bytes) in weights.iter_mut().zip(bytes.chunks_exact(4)) {
                let value = f32::from_le_bytes(bytes.try_into().expect("exact four-byte chunk"));
                if !value.is_finite() {
                    return Err(ModelError::MalformedArtifact("parameter is not finite"));
                }
                *weight = value;
            }
        } else {
            let mut rng = SeededRng::new(seed);
            let scale = 0.15_f32 / (hidden as f32).sqrt();
            fill_uniform(
                &mut weights[embedding..embedding + rows * hidden],
                &mut rng,
                scale,
            );
            fill_uniform(
                &mut weights[position..position + config.context_length * hidden],
                &mut rng,
                scale,
            );
            for row in 0..hidden {
                for col in 0..hidden {
                    let value = if row == col { 0.5 } else { 0.0 };
                    weights[recurrent + row * hidden + col] = value + rng.signed() * 0.01;
                }
            }
            match &kind {
                HeadKind::Flat { vocabulary_size } => {
                    fill_uniform(
                        &mut weights[flat_output..flat_output + *vocabulary_size * hidden],
                        &mut rng,
                        scale,
                    );
                }
                HeadKind::Factorized { packs } => {
                    fill_uniform(
                        &mut weights[pack_output..pack_output + packs.len() * hidden],
                        &mut rng,
                        scale,
                    );
                    fill_uniform(
                        &mut weights[local_output..local_output + rows * hidden],
                        &mut rng,
                        scale,
                    );
                }
            }
        }
        Ok(Self {
            config,
            kind,
            seed,
            layout: ParameterLayout {
                embedding,
                position,
                recurrent,
                recurrent_bias,
                flat_output,
                flat_output_bias,
                pack_output,
                pack_output_bias,
                local_output,
                local_output_bias,
                parameter_count: cursor,
            },
            first_moment: vec![0.0; cursor],
            second_moment: vec![0.0; cursor],
            weights,
            optimizer_step: 0,
        })
    }

    /// Returns the shared model dimensions.
    #[must_use]
    pub const fn config(&self) -> ModelConfig {
        self.config
    }

    /// Returns the initialization seed stored in this model artifact.
    #[must_use]
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// Returns the flat vocabulary size, or `None` for a factorized model.
    #[must_use]
    pub fn flat_vocabulary_size(&self) -> Option<u32> {
        match self.kind {
            HeadKind::Flat { vocabulary_size } => u32::try_from(vocabulary_size).ok(),
            HeadKind::Factorized { .. } => None,
        }
    }

    /// Returns factorized pack declarations in ascending stable pack-ID order.
    #[must_use]
    pub fn pack_vocabularies(&self) -> Option<Vec<PackVocabulary>> {
        match &self.kind {
            HeadKind::Flat { .. } => None,
            HeadKind::Factorized { packs } => Some(
                packs
                    .iter()
                    .map(|pack| PackVocabulary {
                        pack_id: pack.pack_id,
                        token_count: u32::try_from(pack.token_count).unwrap_or(u32::MAX),
                    })
                    .collect(),
            ),
        }
    }

    /// Returns embedding, recurrent-backbone, output-head, and total parameters.
    #[must_use]
    pub fn parameter_counts(&self) -> ParameterCounts {
        let hidden = self.config.hidden_size;
        let rows = self.embedding_rows();
        let embeddings = rows * hidden;
        let backbone = self.config.context_length * hidden + hidden * hidden + hidden;
        let output_head = match &self.kind {
            HeadKind::Flat { vocabulary_size } => vocabulary_size * hidden + vocabulary_size,
            HeadKind::Factorized { packs } => {
                packs.len() * hidden + packs.len() + rows * hidden + rows
            }
        };
        ParameterCounts {
            embeddings,
            backbone,
            output_head,
            total: embeddings + backbone + output_head,
        }
    }

    /// Returns the current parameter values for deterministic audit comparisons.
    #[must_use]
    pub fn parameters(&self) -> &[f32] {
        &self.weights
    }

    /// Returns causal hidden states after each input, including the zero state.
    pub fn hidden_states(&self, inputs: &[TokenId]) -> Result<Vec<f32>, ModelError> {
        self.validate_context(inputs)?;
        let hidden = self.config.hidden_size;
        let capacity = inputs
            .len()
            .checked_add(1)
            .and_then(|length| length.checked_mul(hidden))
            .ok_or(ModelError::LengthOverflow)?;
        let mut states = vec![0.0_f32; capacity];
        for (position, token) in inputs.iter().copied().enumerate() {
            let row = self.embedding_row(token, position)?;
            let previous_start = position * hidden;
            let state_start = (position + 1) * hidden;
            let (past, current) = states.split_at_mut(state_start);
            self.advance_state(
                row,
                position,
                &past[previous_start..previous_start + hidden],
                &mut current[..hidden],
            );
        }
        Ok(states)
    }

    /// Greedily predicts one next token from a non-empty context.
    ///
    /// Factorized generation chooses the highest-scoring pack first, then the
    /// highest-scoring local ID inside that pack. Ties resolve to the first row
    /// in canonical pack/local order.
    pub fn predict_next(&self, context: &[TokenId]) -> Result<TokenId, ModelError> {
        if context.is_empty() {
            return Err(ModelError::EmptyContext);
        }
        let start = context.len().saturating_sub(self.config.context_length);
        let context = &context[start..];
        let states = self.hidden_states(context)?;
        let hidden = self.config.hidden_size;
        let last = &states[context.len() * hidden..(context.len() + 1) * hidden];
        match &self.kind {
            HeadKind::Flat { vocabulary_size } => {
                let logits = self.flat_logits(last, *vocabulary_size);
                validate_logits(&logits)?;
                let local = argmax(&logits);
                Ok(TokenId::new(
                    0,
                    u32::try_from(local).map_err(|_| ModelError::LengthOverflow)?,
                ))
            }
            HeadKind::Factorized { packs } => {
                let pack_logits = self.pack_logits(last, packs.len());
                validate_logits(&pack_logits)?;
                let selected = argmax(&pack_logits);
                let pack = packs.get(selected).ok_or(ModelError::InvalidVocabulary)?;
                let local_logits = self.local_logits(last, pack);
                validate_logits(&local_logits)?;
                let local = argmax(&local_logits);
                Ok(TokenId::new(
                    pack.pack_id,
                    u32::try_from(local).map_err(|_| ModelError::LengthOverflow)?,
                ))
            }
        }
    }

    /// Greedily generates tokens and returns the prompt followed by its continuation.
    pub fn greedy_generate(
        &self,
        prompt: &[TokenId],
        new_token_count: usize,
    ) -> Result<Vec<TokenId>, ModelError> {
        if prompt.is_empty() {
            return Err(ModelError::EmptyContext);
        }
        let capacity = prompt
            .len()
            .checked_add(new_token_count)
            .ok_or(ModelError::LengthOverflow)?;
        if capacity > MAX_GENERATION_TOKENS {
            return Err(ModelError::GenerationTooLong);
        }
        let mut output = Vec::with_capacity(capacity);
        output.extend_from_slice(prompt);
        for _ in 0..new_token_count {
            let start = output.len().saturating_sub(self.config.context_length);
            let next = self.predict_next(&output[start..])?;
            output.push(next);
        }
        Ok(output)
    }

    /// Computes deterministic analytical MACs for the matrix operations used by
    /// forward and backward passes on this batch. Softmax, Adam, and elementwise
    /// operations are intentionally excluded.
    pub fn estimate_training_macs(&self, batch: &[TrainingExample<'_>]) -> Result<u64, ModelError> {
        let hidden =
            u64::try_from(self.config.hidden_size).map_err(|_| ModelError::LengthOverflow)?;
        let hidden_sq = hidden
            .checked_mul(hidden)
            .ok_or(ModelError::LengthOverflow)?;
        let mut macs = 0_u64;
        for (example_index, example) in batch.iter().enumerate() {
            validate_example(example_index, *example, self.config.context_length)?;
            self.validate_context(example.inputs)?;
            let input_len =
                u64::try_from(example.inputs.len()).map_err(|_| ModelError::LengthOverflow)?;
            let recurrent_products = input_len
                .checked_mul(hidden_sq)
                .and_then(|value| value.checked_mul(2))
                .and_then(|value| {
                    value.checked_add(input_len.saturating_sub(1).saturating_mul(hidden_sq))
                })
                .ok_or(ModelError::LengthOverflow)?;
            macs = macs
                .checked_add(recurrent_products)
                .ok_or(ModelError::LengthOverflow)?;
            let target_start = if example.targets.len() == 1 {
                example.inputs.len() - 1
            } else {
                0
            };
            for target_offset in 0..example.targets.len() {
                let target = example.targets[target_offset];
                self.embedding_row(target, target_start + target_offset)
                    .map_err(|_| ModelError::InvalidToken {
                        index: target_offset,
                        token: target,
                    })?;
                let output_rows = match &self.kind {
                    HeadKind::Flat { vocabulary_size } => {
                        u64::try_from(*vocabulary_size).map_err(|_| ModelError::LengthOverflow)?
                    }
                    HeadKind::Factorized { packs } => {
                        let pack_index = self.pack_index(target.pack)?;
                        let pack = &packs[pack_index];
                        u64::try_from(packs.len() + pack.token_count)
                            .map_err(|_| ModelError::LengthOverflow)?
                    }
                };
                let output_macs = output_rows
                    .checked_mul(hidden)
                    .and_then(|value| value.checked_mul(3))
                    .ok_or(ModelError::LengthOverflow)?;
                macs = macs
                    .checked_add(output_macs)
                    .ok_or(ModelError::LengthOverflow)?;
            }
        }
        Ok(macs)
    }

    /// Applies one mean-loss Adam update to the supplied independent windows.
    pub fn train_batch(
        &mut self,
        batch: &[TrainingExample<'_>],
        optimizer: OptimizerConfig,
    ) -> Result<TrainStepMetrics, ModelError> {
        let optimizer = optimizer.validate()?;
        if batch.is_empty() {
            return Err(ModelError::EmptyBatch);
        }
        let target_count =
            batch
                .iter()
                .enumerate()
                .try_fold(0_usize, |sum, (index, example)| {
                    validate_example(index, *example, self.config.context_length)?;
                    sum.checked_add(example.targets.len())
                        .ok_or(ModelError::LengthOverflow)
                })?;
        if target_count == 0 {
            return Err(ModelError::EmptyBatch);
        }
        let macs = self.estimate_training_macs(batch)?;
        let hidden = self.config.hidden_size;
        let mut gradients = vec![0.0_f32; self.layout.parameter_count];
        let mut loss_sum = 0.0_f64;

        for (example_index, example) in batch.iter().enumerate() {
            validate_example(example_index, *example, self.config.context_length)?;
            let states = self.hidden_states(example.inputs)?;
            let mut hidden_gradients = vec![0.0_f32; example.inputs.len() * hidden];
            let target_start = if example.targets.len() == 1 {
                example.inputs.len() - 1
            } else {
                0
            };
            let loss_scale = 1.0_f32 / target_count as f32;

            for (target_offset, target) in example.targets.iter().copied().enumerate() {
                let position = target_start + target_offset;
                let hidden_start = (position + 1) * hidden;
                let state = &states[hidden_start..hidden_start + hidden];
                match &self.kind {
                    HeadKind::Flat { vocabulary_size } => {
                        if target.pack != 0
                            || usize::try_from(target.local).unwrap_or(usize::MAX)
                                >= *vocabulary_size
                        {
                            return Err(ModelError::InvalidToken {
                                index: target_offset,
                                token: target,
                            });
                        }
                        let logits = self.flat_logits(state, *vocabulary_size);
                        let (loss, output_gradient) =
                            softmax_cross_entropy(&logits, target.local as usize, loss_scale)?;
                        loss_sum += loss;
                        self.accumulate_flat_head(
                            state,
                            &output_gradient,
                            &mut gradients,
                            &mut hidden_gradients[position * hidden..(position + 1) * hidden],
                        );
                    }
                    HeadKind::Factorized { packs } => {
                        let pack_index =
                            self.pack_index(target.pack)
                                .map_err(|_| ModelError::InvalidToken {
                                    index: target_offset,
                                    token: target,
                                })?;
                        let pack = &packs[pack_index];
                        if usize::try_from(target.local).unwrap_or(usize::MAX) >= pack.token_count {
                            return Err(ModelError::InvalidToken {
                                index: target_offset,
                                token: target,
                            });
                        }
                        let pack_logits = self.pack_logits(state, packs.len());
                        let (pack_loss, pack_gradient) =
                            softmax_cross_entropy(&pack_logits, pack_index, loss_scale)?;
                        loss_sum += pack_loss;
                        self.accumulate_pack_head(
                            state,
                            &pack_gradient,
                            &mut gradients,
                            &mut hidden_gradients[position * hidden..(position + 1) * hidden],
                        );
                        let local_logits = self.local_logits(state, pack);
                        let (local_loss, local_gradient) = softmax_cross_entropy(
                            &local_logits,
                            target.local as usize,
                            loss_scale,
                        )?;
                        loss_sum += local_loss;
                        self.accumulate_local_head(
                            state,
                            pack,
                            &local_gradient,
                            &mut gradients,
                            &mut hidden_gradients[position * hidden..(position + 1) * hidden],
                        );
                    }
                }
            }
            self.backprop_recurrent(example.inputs, &states, &hidden_gradients, &mut gradients)?;
        }

        let gradient_norm = gradients
            .iter()
            .map(|gradient| f64::from(*gradient) * f64::from(*gradient))
            .sum::<f64>()
            .sqrt();
        if !gradient_norm.is_finite() {
            return Err(ModelError::NonFiniteComputation);
        }
        if gradient_norm > f64::from(optimizer.gradient_clip_norm) {
            let scale = (f64::from(optimizer.gradient_clip_norm) / gradient_norm) as f32;
            for gradient in &mut gradients {
                *gradient *= scale;
            }
        }
        self.apply_adam(&gradients, optimizer)?;
        Ok(TrainStepMetrics {
            loss_per_token: loss_sum / target_count as f64,
            target_tokens: target_count,
            estimated_macs: macs,
            gradient_norm,
        })
    }

    /// Evaluates target loss and greedy accuracies without changing parameters.
    pub fn evaluate(
        &self,
        examples: &[TrainingExample<'_>],
    ) -> Result<EvaluationMetrics, ModelError> {
        if examples.is_empty() {
            return Err(ModelError::EmptyBatch);
        }
        let mut target_count = 0_usize;
        let mut loss_sum = 0.0_f64;
        let mut primary_loss = 0.0_f64;
        let mut local_loss = 0.0_f64;
        let mut correct_tokens = 0_usize;
        let mut correct_packs = 0_usize;
        let mut correct_locals = 0_usize;
        let mut active_local_sum = 0_usize;
        let mut pack_rows = match &self.kind {
            HeadKind::Flat { .. } => Vec::new(),
            HeadKind::Factorized { packs } => packs
                .iter()
                .map(|pack| PackAccuracy {
                    pack_id: pack.pack_id,
                    targets: 0,
                    correct: 0,
                })
                .collect(),
        };
        let hidden = self.config.hidden_size;
        let mut total_pack_logits = 0_usize;
        let mut total_output_logits = 0_usize;

        for (example_index, example) in examples.iter().enumerate() {
            validate_example(example_index, *example, self.config.context_length)?;
            let states = self.hidden_states(example.inputs)?;
            let target_start = if example.targets.len() == 1 {
                example.inputs.len() - 1
            } else {
                0
            };
            for (target_offset, target) in example.targets.iter().copied().enumerate() {
                let position = target_start + target_offset;
                let state = &states[(position + 1) * hidden..(position + 2) * hidden];
                match &self.kind {
                    HeadKind::Flat { vocabulary_size } => {
                        if target.pack != 0
                            || usize::try_from(target.local).unwrap_or(usize::MAX)
                                >= *vocabulary_size
                        {
                            return Err(ModelError::InvalidToken {
                                index: target_offset,
                                token: target,
                            });
                        }
                        let logits = self.flat_logits(state, *vocabulary_size);
                        let loss = softmax_loss(&logits, target.local as usize)?;
                        loss_sum += loss;
                        primary_loss += loss;
                        correct_tokens += usize::from(argmax(&logits) == target.local as usize);
                    }
                    HeadKind::Factorized { packs } => {
                        let pack_index =
                            self.pack_index(target.pack)
                                .map_err(|_| ModelError::InvalidToken {
                                    index: target_offset,
                                    token: target,
                                })?;
                        let pack = &packs[pack_index];
                        if usize::try_from(target.local).unwrap_or(usize::MAX) >= pack.token_count {
                            return Err(ModelError::InvalidToken {
                                index: target_offset,
                                token: target,
                            });
                        }
                        let pack_logits = self.pack_logits(state, packs.len());
                        let pack_ce = softmax_loss(&pack_logits, pack_index)?;
                        let predicted_pack = argmax(&pack_logits);
                        let row = &mut pack_rows[pack_index];
                        row.targets += 1;
                        let pack_correct = predicted_pack == pack_index;
                        if pack_correct {
                            row.correct += 1;
                            correct_packs += 1;
                        }
                        primary_loss += pack_ce;
                        let local_logits = self.local_logits(state, pack);
                        let local_ce = softmax_loss(&local_logits, target.local as usize)?;
                        let local_correct = argmax(&local_logits) == target.local as usize;
                        if local_correct {
                            correct_locals += 1;
                        }
                        correct_tokens += usize::from(pack_correct && local_correct);
                        local_loss += local_ce;
                        loss_sum += pack_ce + local_ce;
                        active_local_sum = active_local_sum
                            .checked_add(pack.token_count)
                            .ok_or(ModelError::LengthOverflow)?;
                        total_pack_logits = total_pack_logits
                            .checked_add(packs.len())
                            .ok_or(ModelError::LengthOverflow)?;
                        total_output_logits = total_output_logits
                            .checked_add(packs.len() + pack.token_count)
                            .ok_or(ModelError::LengthOverflow)?;
                    }
                }
                target_count = target_count
                    .checked_add(1)
                    .ok_or(ModelError::LengthOverflow)?;
            }
        }
        if target_count == 0 {
            return Err(ModelError::EmptyBatch);
        }
        let factorized = matches!(self.kind, HeadKind::Factorized { .. });
        Ok(EvaluationMetrics {
            target_tokens: target_count,
            loss_per_token: loss_sum / target_count as f64,
            primary_head_loss_per_token: Some(primary_loss / target_count as f64),
            local_loss_per_token: factorized.then_some(local_loss / target_count as f64),
            token_accuracy: correct_tokens as f64 / target_count as f64,
            pack_accuracy: factorized.then_some(correct_packs as f64 / target_count as f64),
            local_accuracy_given_pack: factorized
                .then_some(correct_locals as f64 / target_count as f64),
            per_pack: pack_rows,
            average_active_local_head_size: factorized
                .then_some(active_local_sum as f64 / target_count as f64),
            pack_head_logit_fraction: factorized
                .then_some(total_pack_logits as f64 / total_output_logits as f64),
        })
    }

    /// Serializes parameters and architecture using the deterministic model format.
    pub fn to_bytes(&self) -> Result<Vec<u8>, ModelError> {
        let packs = self.serialized_packs();
        let descriptors_bytes = packs
            .len()
            .checked_mul(6)
            .ok_or(ModelError::LengthOverflow)?;
        let parameter_bytes = self
            .weights
            .len()
            .checked_mul(4)
            .ok_or(ModelError::LengthOverflow)?;
        let total = 8_usize
            .checked_add(2 + 1 + 1 + 4 + 4 + 8 + 4)
            .and_then(|value| value.checked_add(descriptors_bytes))
            .and_then(|value| value.checked_add(8))
            .and_then(|value| value.checked_add(parameter_bytes))
            .ok_or(ModelError::LengthOverflow)?;
        if total > MAX_MODEL_BYTES {
            return Err(ModelError::ModelTooLarge);
        }
        let mut bytes = Vec::with_capacity(total);
        bytes.extend_from_slice(MODEL_MAGIC);
        bytes.extend_from_slice(&MODEL_FORMAT_VERSION.to_le_bytes());
        bytes.push(u8::from(matches!(self.kind, HeadKind::Factorized { .. })));
        bytes.push(0);
        push_u32(
            &mut bytes,
            u32::try_from(self.config.hidden_size).map_err(|_| ModelError::LengthOverflow)?,
        );
        push_u32(
            &mut bytes,
            u32::try_from(self.config.context_length).map_err(|_| ModelError::LengthOverflow)?,
        );
        bytes.extend_from_slice(&self.seed.to_le_bytes());
        push_u32(
            &mut bytes,
            u32::try_from(packs.len()).map_err(|_| ModelError::LengthOverflow)?,
        );
        for pack in packs {
            bytes.extend_from_slice(&pack.pack_id.to_le_bytes());
            push_u32(&mut bytes, pack.token_count);
        }
        push_u64(
            &mut bytes,
            u64::try_from(self.weights.len()).map_err(|_| ModelError::LengthOverflow)?,
        );
        for weight in &self.weights {
            if !weight.is_finite() {
                return Err(ModelError::NonFiniteComputation);
            }
            bytes.extend_from_slice(&weight.to_le_bytes());
        }
        debug_assert_eq!(bytes.len(), total);
        Ok(bytes)
    }

    /// Loads and validates one deterministic model parameter artifact.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ModelError> {
        if bytes.len() > MAX_MODEL_BYTES {
            return Err(ModelError::ModelTooLarge);
        }
        let mut cursor = Cursor::new(bytes);
        if cursor.read_exact(8)? != MODEL_MAGIC {
            return Err(ModelError::MalformedArtifact("invalid magic"));
        }
        if cursor.read_u16()? != MODEL_FORMAT_VERSION {
            return Err(ModelError::MalformedArtifact("unsupported model version"));
        }
        let kind = cursor.read_u8()?;
        if cursor.read_u8()? != 0 {
            return Err(ModelError::MalformedArtifact("reserved byte is nonzero"));
        }
        let config = ModelConfig {
            hidden_size: usize::try_from(cursor.read_u32()?)
                .map_err(|_| ModelError::LengthOverflow)?,
            context_length: usize::try_from(cursor.read_u32()?)
                .map_err(|_| ModelError::LengthOverflow)?,
        };
        config.validate()?;
        let seed = cursor.read_u64()?;
        let pack_count =
            usize::try_from(cursor.read_u32()?).map_err(|_| ModelError::LengthOverflow)?;
        if pack_count == 0 || pack_count > MAX_PACK_COUNT {
            return Err(ModelError::MalformedArtifact("invalid pack count"));
        }
        let mut packs = Vec::with_capacity(pack_count);
        for _ in 0..pack_count {
            packs.push(PackVocabulary {
                pack_id: cursor.read_u16()?,
                token_count: cursor.read_u32()?,
            });
        }
        if packs
            .windows(2)
            .any(|pair| pair[0].pack_id >= pair[1].pack_id)
        {
            return Err(ModelError::MalformedArtifact(
                "pack IDs are not strictly sorted",
            ));
        }
        let declared_params =
            usize::try_from(cursor.read_u64()?).map_err(|_| ModelError::LengthOverflow)?;
        if declared_params > MAX_PARAMETER_COUNT {
            return Err(ModelError::ModelTooLarge);
        }
        let expected_remaining = declared_params
            .checked_mul(4)
            .ok_or(ModelError::LengthOverflow)?;
        if cursor.remaining() != expected_remaining {
            return Err(ModelError::MalformedArtifact(
                "parameter byte length does not match header",
            ));
        }
        let parameter_bytes = cursor.read_exact(expected_remaining)?;
        let model = match kind {
            0 if pack_count == 1 && packs[0].pack_id == 0 => {
                Self::flat_model(config, packs[0].token_count, seed, Some(parameter_bytes))?
            }
            1 => Self::factorized_model(config, packs, seed, Some(parameter_bytes))?,
            0 => {
                return Err(ModelError::MalformedArtifact(
                    "flat model must have only pack 0",
                ));
            }
            _ => return Err(ModelError::MalformedArtifact("unknown model kind")),
        };
        if cursor.remaining() != 0 {
            return Err(ModelError::MalformedArtifact("trailing bytes"));
        }
        Ok(model)
    }

    fn serialized_packs(&self) -> Vec<PackVocabulary> {
        match &self.kind {
            HeadKind::Flat { vocabulary_size } => vec![PackVocabulary {
                pack_id: 0,
                token_count: u32::try_from(*vocabulary_size).unwrap_or(u32::MAX),
            }],
            HeadKind::Factorized { packs } => packs
                .iter()
                .map(|pack| PackVocabulary {
                    pack_id: pack.pack_id,
                    token_count: u32::try_from(pack.token_count).unwrap_or(u32::MAX),
                })
                .collect(),
        }
    }

    fn embedding_rows(&self) -> usize {
        match &self.kind {
            HeadKind::Flat { vocabulary_size } => *vocabulary_size,
            HeadKind::Factorized { packs } => packs
                .last()
                .and_then(|pack| pack.row_start.checked_add(pack.token_count))
                .unwrap_or(0),
        }
    }

    fn pack_index(&self, pack_id: PackId) -> Result<usize, ModelError> {
        match &self.kind {
            HeadKind::Factorized { packs } => packs
                .binary_search_by_key(&pack_id, |pack| pack.pack_id)
                .map_err(|_| ModelError::UnknownPack(pack_id)),
            HeadKind::Flat { .. } => Err(ModelError::WrongModelKind),
        }
    }

    fn embedding_row(&self, token: TokenId, position: usize) -> Result<usize, ModelError> {
        match &self.kind {
            HeadKind::Flat { vocabulary_size } => {
                if token.pack != 0
                    || usize::try_from(token.local).unwrap_or(usize::MAX) >= *vocabulary_size
                {
                    return Err(ModelError::InvalidToken {
                        index: position,
                        token,
                    });
                }
                Ok(usize::try_from(token.local).map_err(|_| ModelError::LengthOverflow)?)
            }
            HeadKind::Factorized { packs } => {
                let pack_index = self.pack_index(token.pack)?;
                let pack = &packs[pack_index];
                let local = usize::try_from(token.local).map_err(|_| ModelError::LengthOverflow)?;
                if local >= pack.token_count {
                    return Err(ModelError::InvalidToken {
                        index: position,
                        token,
                    });
                }
                pack.row_start
                    .checked_add(local)
                    .ok_or(ModelError::LengthOverflow)
            }
        }
    }

    fn validate_context(&self, inputs: &[TokenId]) -> Result<(), ModelError> {
        if inputs.is_empty() {
            return Err(ModelError::EmptyContext);
        }
        if inputs.len() > self.config.context_length {
            return Err(ModelError::ContextTooLong {
                actual: inputs.len(),
                maximum: self.config.context_length,
            });
        }
        for (index, token) in inputs.iter().copied().enumerate() {
            self.embedding_row(token, index)?;
        }
        Ok(())
    }

    fn advance_state(
        &self,
        embedding_row: usize,
        position: usize,
        previous: &[f32],
        output: &mut [f32],
    ) {
        let hidden = self.config.hidden_size;
        let position_offset = self.layout.position + position * hidden;
        let embedding_offset = self.layout.embedding + embedding_row * hidden;
        for (row, out) in output.iter_mut().enumerate() {
            let mut sum = self.weights[embedding_offset + row]
                + self.weights[position_offset + row]
                + self.weights[self.layout.recurrent_bias + row];
            let matrix_row = self.layout.recurrent + row * hidden;
            for (column, previous_value) in previous.iter().copied().enumerate() {
                sum += self.weights[matrix_row + column] * previous_value;
            }
            *out = sum.tanh();
        }
    }

    fn flat_logits(&self, hidden: &[f32], vocabulary_size: usize) -> Vec<f32> {
        let dimension = self.config.hidden_size;
        let mut logits = vec![0.0; vocabulary_size];
        for (row, logit) in logits.iter_mut().enumerate() {
            let offset = self.layout.flat_output + row * dimension;
            let mut value = self.weights[self.layout.flat_output_bias + row];
            for (column, hidden_value) in hidden.iter().copied().enumerate() {
                value += self.weights[offset + column] * hidden_value;
            }
            *logit = value;
        }
        logits
    }

    fn pack_logits(&self, hidden: &[f32], pack_count: usize) -> Vec<f32> {
        let dimension = self.config.hidden_size;
        let mut logits = vec![0.0; pack_count];
        for (row, logit) in logits.iter_mut().enumerate() {
            let offset = self.layout.pack_output + row * dimension;
            let mut value = self.weights[self.layout.pack_output_bias + row];
            for (column, hidden_value) in hidden.iter().copied().enumerate() {
                value += self.weights[offset + column] * hidden_value;
            }
            *logit = value;
        }
        logits
    }

    fn local_logits(&self, hidden: &[f32], pack: &PackLayout) -> Vec<f32> {
        let dimension = self.config.hidden_size;
        let mut logits = vec![0.0; pack.token_count];
        for (local, logit) in logits.iter_mut().enumerate() {
            let row = pack.row_start + local;
            let offset = self.layout.local_output + row * dimension;
            let mut value = self.weights[self.layout.local_output_bias + row];
            for (column, hidden_value) in hidden.iter().copied().enumerate() {
                value += self.weights[offset + column] * hidden_value;
            }
            *logit = value;
        }
        logits
    }

    fn accumulate_flat_head(
        &self,
        hidden: &[f32],
        gradient: &[f32],
        gradients: &mut [f32],
        hidden_gradient: &mut [f32],
    ) {
        let dimension = self.config.hidden_size;
        for (row, delta) in gradient.iter().copied().enumerate() {
            gradients[self.layout.flat_output_bias + row] += delta;
            let weight = self.layout.flat_output + row * dimension;
            for column in 0..dimension {
                gradients[weight + column] += delta * hidden[column];
                hidden_gradient[column] += self.weights[weight + column] * delta;
            }
        }
    }

    fn accumulate_pack_head(
        &self,
        hidden: &[f32],
        gradient: &[f32],
        gradients: &mut [f32],
        hidden_gradient: &mut [f32],
    ) {
        let dimension = self.config.hidden_size;
        for (row, delta) in gradient.iter().copied().enumerate() {
            gradients[self.layout.pack_output_bias + row] += delta;
            let weight = self.layout.pack_output + row * dimension;
            for column in 0..dimension {
                gradients[weight + column] += delta * hidden[column];
                hidden_gradient[column] += self.weights[weight + column] * delta;
            }
        }
    }

    fn accumulate_local_head(
        &self,
        hidden: &[f32],
        pack: &PackLayout,
        gradient: &[f32],
        gradients: &mut [f32],
        hidden_gradient: &mut [f32],
    ) {
        let dimension = self.config.hidden_size;
        for (local, delta) in gradient.iter().copied().enumerate() {
            let row = pack.row_start + local;
            gradients[self.layout.local_output_bias + row] += delta;
            let weight = self.layout.local_output + row * dimension;
            for column in 0..dimension {
                gradients[weight + column] += delta * hidden[column];
                hidden_gradient[column] += self.weights[weight + column] * delta;
            }
        }
    }

    fn backprop_recurrent(
        &self,
        inputs: &[TokenId],
        states: &[f32],
        hidden_gradients: &[f32],
        gradients: &mut [f32],
    ) -> Result<(), ModelError> {
        let dimension = self.config.hidden_size;
        let mut carry = vec![0.0_f32; dimension];
        let mut pre_activation = vec![0.0_f32; dimension];
        for position in (0..inputs.len()).rev() {
            let previous_start = position * dimension;
            let state_start = (position + 1) * dimension;
            let gradient_start = position * dimension;
            for row in 0..dimension {
                let state = states[state_start + row];
                pre_activation[row] =
                    (hidden_gradients[gradient_start + row] + carry[row]) * (1.0 - state * state);
                gradients[self.layout.recurrent_bias + row] += pre_activation[row];
            }
            let embedding_row = self.embedding_row(inputs[position], position)?;
            let embedding_start = self.layout.embedding + embedding_row * dimension;
            let position_start = self.layout.position + position * dimension;
            for row in 0..dimension {
                gradients[embedding_start + row] += pre_activation[row];
                gradients[position_start + row] += pre_activation[row];
            }
            carry.fill(0.0);
            for (row, delta) in pre_activation.iter().copied().enumerate() {
                let matrix_start = self.layout.recurrent + row * dimension;
                for column in 0..dimension {
                    gradients[matrix_start + column] += delta * states[previous_start + column];
                    if position > 0 {
                        carry[column] += self.weights[matrix_start + column] * delta;
                    }
                }
            }
        }
        Ok(())
    }

    fn apply_adam(
        &mut self,
        gradients: &[f32],
        optimizer: OptimizerConfig,
    ) -> Result<(), ModelError> {
        let next_step = self
            .optimizer_step
            .checked_add(1)
            .ok_or(ModelError::LengthOverflow)?;
        let step = next_step as f32;
        let correction1 = 1.0 - optimizer.beta1.powf(step);
        let correction2 = 1.0 - optimizer.beta2.powf(step);
        // Preflight every proposed value before committing any state. Recompute
        // on commit to keep the existing arithmetic and avoid three scratch vectors.
        let proposed = |index: usize, gradient: f32| {
            let moment1 =
                optimizer.beta1 * self.first_moment[index] + (1.0 - optimizer.beta1) * gradient;
            let moment2 = optimizer.beta2 * self.second_moment[index]
                + (1.0 - optimizer.beta2) * gradient * gradient;
            let first = moment1 / correction1;
            let second = moment2 / correction2;
            let weight = self.weights[index]
                - optimizer.learning_rate * first / (second.sqrt() + optimizer.epsilon);
            (moment1, moment2, weight)
        };
        for (index, gradient) in gradients.iter().copied().enumerate() {
            let (first, second, weight) = proposed(index, gradient);
            if !first.is_finite() || !second.is_finite() || !weight.is_finite() {
                return Err(ModelError::NonFiniteComputation);
            }
        }
        for (index, gradient) in gradients.iter().copied().enumerate() {
            let moment1 =
                optimizer.beta1 * self.first_moment[index] + (1.0 - optimizer.beta1) * gradient;
            let moment2 = optimizer.beta2 * self.second_moment[index]
                + (1.0 - optimizer.beta2) * gradient * gradient;
            let first = moment1 / correction1;
            let second = moment2 / correction2;
            self.weights[index] -=
                optimizer.learning_rate * first / (second.sqrt() + optimizer.epsilon);
            self.first_moment[index] = moment1;
            self.second_moment[index] = moment2;
        }
        self.optimizer_step = next_step;
        Ok(())
    }
}

/// Returns the analytical training-MAC count for a batch after validating its
/// target shapes and token IDs.
fn reserve_parameters(
    cursor: &mut usize,
    rows: usize,
    columns: usize,
) -> Result<usize, ModelError> {
    let start = *cursor;
    *cursor = cursor
        .checked_add(
            rows.checked_mul(columns)
                .ok_or(ModelError::LengthOverflow)?,
        )
        .ok_or(ModelError::LengthOverflow)?;
    if *cursor > MAX_PARAMETER_COUNT {
        return Err(ModelError::ModelTooLarge);
    }
    Ok(start)
}

fn fill_uniform(values: &mut [f32], rng: &mut SeededRng, scale: f32) {
    for value in values {
        *value = rng.signed() * scale;
    }
}

fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for (index, value) in values.iter().copied().enumerate().skip(1) {
        if value > values[best] {
            best = index;
        }
    }
    best
}

fn softmax_cross_entropy(
    logits: &[f32],
    target: usize,
    scale: f32,
) -> Result<(f64, Vec<f32>), ModelError> {
    let target_logit = logits.get(target).ok_or(ModelError::InvalidVocabulary)?;
    validate_logits(logits)?;
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probabilities: Vec<f64> = logits
        .iter()
        .map(|value| (f64::from(*value) - f64::from(maximum)).exp())
        .collect();
    let denominator = probabilities.iter().sum::<f64>();
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(ModelError::NonFiniteComputation);
    }
    for probability in &mut probabilities {
        *probability /= denominator;
    }
    let loss = (f64::from(maximum) - f64::from(*target_logit)) + denominator.ln();
    let gradient = probabilities
        .into_iter()
        .enumerate()
        .map(|(index, probability)| (probability as f32 - f32::from(index == target)) * scale)
        .collect();
    Ok((loss, gradient))
}

fn validate_logits(logits: &[f32]) -> Result<(), ModelError> {
    if logits.is_empty() || logits.iter().any(|value| !value.is_finite()) {
        return Err(ModelError::NonFiniteComputation);
    }
    Ok(())
}

// Evaluation needs neither normalized probabilities nor output gradients.
fn softmax_loss(logits: &[f32], target: usize) -> Result<f64, ModelError> {
    let target_logit = logits.get(target).ok_or(ModelError::InvalidVocabulary)?;
    validate_logits(logits)?;
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let denominator = logits
        .iter()
        .map(|value| (f64::from(*value) - f64::from(maximum)).exp())
        .sum::<f64>();
    Ok((f64::from(maximum) - f64::from(*target_logit)) + denominator.ln())
}

fn validate_example(
    index: usize,
    example: TrainingExample<'_>,
    context_limit: usize,
) -> Result<(), ModelError> {
    if example.inputs.is_empty() {
        return Err(ModelError::InvalidExample {
            index,
            reason: "input context is empty",
        });
    }
    if example.inputs.len() > context_limit {
        return Err(ModelError::ContextTooLong {
            actual: example.inputs.len(),
            maximum: context_limit,
        });
    }
    if example.targets.is_empty()
        || (example.targets.len() != example.inputs.len() && example.targets.len() != 1)
    {
        return Err(ModelError::InvalidExample {
            index,
            reason: "targets must align with inputs or contain one final target",
        });
    }
    Ok(())
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn read_exact(&mut self, length: usize) -> Result<&'a [u8], ModelError> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or(ModelError::LengthOverflow)?;
        let result = self
            .bytes
            .get(self.offset..end)
            .ok_or(ModelError::MalformedArtifact("truncated input"))?;
        self.offset = end;
        Ok(result)
    }

    fn read_u8(&mut self) -> Result<u8, ModelError> {
        self.read_exact(1)?
            .first()
            .copied()
            .ok_or(ModelError::MalformedArtifact("truncated input"))
    }

    fn read_u16(&mut self) -> Result<u16, ModelError> {
        let bytes = self.read_exact(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_u32(&mut self) -> Result<u32, ModelError> {
        let bytes = self.read_exact(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_u64(&mut self) -> Result<u64, ModelError> {
        let bytes = self.read_exact(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }
}

struct SeededRng {
    state: u64,
}

impl SeededRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9e37_79b9_7f4a_7c15
            } else {
                seed
            },
        }
    }

    fn next(&mut self) -> u64 {
        let mut value = self.state;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.state = value;
        value
    }

    fn signed(&mut self) -> f32 {
        let mantissa = (self.next() >> 40) as u32;
        (mantissa as f32 / 8_388_607.5) - 1.0
    }
}

/// Errors from model construction, training, evaluation, and model-file parsing.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModelError {
    /// Model dimensions or vocabulary sizes are outside the documented bounds.
    InvalidConfig(&'static str),
    /// Optimizer values are invalid or non-finite.
    InvalidOptimizer(&'static str),
    /// Vocabulary declaration is empty or exceeds the model's documented cap.
    InvalidVocabulary,
    /// Two factorized vocabulary descriptors declare one pack ID.
    DuplicatePackId(PackId),
    /// The requested operation does not match the model's head kind.
    WrongModelKind,
    /// A factorized token references a pack absent from this model.
    UnknownPack(PackId),
    /// A token ID is outside its model vocabulary.
    InvalidToken { index: usize, token: TokenId },
    /// Context has no tokens.
    EmptyContext,
    /// Context exceeds the model's configured maximum.
    ContextTooLong { actual: usize, maximum: usize },
    /// Training or evaluation received an invalid sequence shape.
    InvalidExample { index: usize, reason: &'static str },
    /// A batch contains no examples or no target tokens.
    EmptyBatch,
    /// Requested generation exceeds the explicit safety limit.
    GenerationTooLong,
    /// The model or serialized parameter count exceeds documented bounds.
    ModelTooLarge,
    /// An integer size calculation overflowed.
    LengthOverflow,
    /// A forward, backward, or update calculation produced NaN or infinity.
    NonFiniteComputation,
    /// Model-file bytes are malformed, unsupported, or noncanonical.
    MalformedArtifact(&'static str),
}

impl fmt::Display for ModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(reason) => {
                write!(formatter, "invalid model configuration: {reason}")
            }
            Self::InvalidOptimizer(reason) => {
                write!(formatter, "invalid optimizer configuration: {reason}")
            }
            Self::InvalidVocabulary => formatter.write_str("invalid or oversized model vocabulary"),
            Self::DuplicatePackId(id) => {
                write!(formatter, "pack ID {id} is declared more than once")
            }
            Self::WrongModelKind => {
                formatter.write_str("operation is not supported for this model kind")
            }
            Self::UnknownPack(id) => write!(formatter, "pack ID {id} is not present in the model"),
            Self::InvalidToken { index, token } => write!(
                formatter,
                "token {index} ({}/{}) is outside the model vocabulary",
                token.pack, token.local
            ),
            Self::EmptyContext => formatter.write_str("a non-empty token context is required"),
            Self::ContextTooLong { actual, maximum } => write!(
                formatter,
                "context has {actual} tokens; model maximum is {maximum}"
            ),
            Self::InvalidExample { index, reason } => {
                write!(formatter, "invalid training example {index}: {reason}")
            }
            Self::EmptyBatch => formatter.write_str("training/evaluation batch has no targets"),
            Self::GenerationTooLong => {
                formatter.write_str("generated sequence exceeds the model safety limit")
            }
            Self::ModelTooLarge => formatter.write_str("model exceeds the documented size limit"),
            Self::LengthOverflow => formatter.write_str("model size or operation count overflowed"),
            Self::NonFiniteComputation => {
                formatter.write_str("model calculation produced a non-finite value")
            }
            Self::MalformedArtifact(reason) => {
                write!(formatter, "malformed model artifact: {reason}")
            }
        }
    }
}

impl std::error::Error for ModelError {}

const MAX_GENERATION_TOKENS: usize = 4096;

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> ModelConfig {
        ModelConfig {
            hidden_size: 8,
            context_length: 4,
        }
    }

    fn alternating_flat(length: usize) -> Vec<TokenId> {
        (0..length)
            .map(|index| TokenId::new(0, if index % 2 == 0 { 97 } else { 98 }))
            .collect()
    }

    fn windows(tokens: &[TokenId], context: usize) -> Vec<TrainingExample<'_>> {
        (0..tokens.len() - context)
            .map(|start| TrainingExample {
                inputs: &tokens[start..start + context],
                targets: &tokens[start + 1..start + context + 1],
            })
            .collect()
    }

    #[test]
    fn same_seed_initializes_identical_parameters() {
        let first = CausalLm::new_flat(tiny_config(), 260, 77).unwrap();
        let second = CausalLm::new_flat(tiny_config(), 260, 77).unwrap();
        let different = CausalLm::new_flat(tiny_config(), 260, 78).unwrap();
        assert_eq!(first.parameters(), second.parameters());
        assert_ne!(first.parameters(), different.parameters());
    }

    #[test]
    fn parameter_partitions_sum_to_total_for_both_heads() {
        let flat = CausalLm::new_flat(tiny_config(), 260, 7).unwrap();
        let factorized = CausalLm::new_factorized(
            tiny_config(),
            vec![
                PackVocabulary {
                    pack_id: 3,
                    token_count: 2,
                },
                PackVocabulary {
                    pack_id: 1,
                    token_count: 3,
                },
            ],
            7,
        )
        .unwrap();
        for model in [&flat, &factorized] {
            let counts = model.parameter_counts();
            assert_eq!(
                counts.embeddings + counts.backbone + counts.output_head,
                counts.total
            );
            assert_eq!(counts.total, model.parameters().len());
        }
        assert_eq!(factorized.pack_vocabularies().unwrap()[0].pack_id, 1);
    }

    #[test]
    fn same_local_id_in_different_packs_uses_distinct_embedding_rows() {
        let model = CausalLm::new_factorized(
            tiny_config(),
            vec![
                PackVocabulary {
                    pack_id: 1,
                    token_count: 2,
                },
                PackVocabulary {
                    pack_id: 2,
                    token_count: 2,
                },
            ],
            4,
        )
        .unwrap();
        let first = TokenId::new(1, 0);
        let second = TokenId::new(2, 0);
        let first_state = model.hidden_states(&[first]).unwrap();
        let second_state = model.hidden_states(&[second]).unwrap();
        assert_ne!(first_state, second_state);
        assert_ne!(
            model.embedding_row(first, 0),
            model.embedding_row(second, 0)
        );
    }

    #[test]
    fn output_shapes_follow_flat_and_factorized_vocabularies() {
        let flat = CausalLm::new_flat(tiny_config(), 260, 1).unwrap();
        assert_eq!(flat.flat_vocabulary_size(), Some(260));
        let example_tokens = [TokenId::new(0, 97), TokenId::new(0, 98)];
        assert_eq!(flat.hidden_states(&example_tokens).unwrap().len(), 3 * 8);
        let fact = CausalLm::new_factorized(
            tiny_config(),
            vec![
                PackVocabulary {
                    pack_id: 1,
                    token_count: 2,
                },
                PackVocabulary {
                    pack_id: u16::MAX,
                    token_count: 256,
                },
            ],
            1,
        )
        .unwrap();
        let prediction = fact.predict_next(&[TokenId::new(1, 0)]).unwrap();
        assert!(prediction.pack == 1 || prediction.pack == u16::MAX);
    }

    #[test]
    fn changing_future_inputs_does_not_change_prior_hidden_states() {
        let model = CausalLm::new_flat(tiny_config(), 260, 9).unwrap();
        let prefix = [TokenId::new(0, 97), TokenId::new(0, 98)];
        let mut with_future = prefix.to_vec();
        with_future.push(TokenId::new(0, 101));
        let before = model.hidden_states(&prefix).unwrap();
        let after = model.hidden_states(&with_future).unwrap();
        assert_eq!(
            &before[..(prefix.len() + 1) * 8],
            &after[..(prefix.len() + 1) * 8]
        );
    }

    #[test]
    fn flat_and_factorized_cross_entropy_are_finite_and_teacher_forced() {
        let flat = CausalLm::new_flat(tiny_config(), 260, 11).unwrap();
        let mut factorized = CausalLm::new_factorized(
            tiny_config(),
            vec![
                PackVocabulary {
                    pack_id: 0,
                    token_count: 3,
                },
                PackVocabulary {
                    pack_id: u16::MAX,
                    token_count: 256,
                },
            ],
            11,
        )
        .unwrap();
        let flat_tokens = [
            TokenId::new(0, 97),
            TokenId::new(0, 98),
            TokenId::new(0, 97),
        ];
        let fact_tokens = [
            TokenId::new(0, 0),
            TokenId::new(u16::MAX, 97),
            TokenId::new(0, 0),
        ];
        let flat_batch = [TrainingExample {
            inputs: &flat_tokens[..2],
            targets: &flat_tokens[1..],
        }];
        let fact_batch = [TrainingExample {
            inputs: &fact_tokens[..2],
            targets: &fact_tokens[1..],
        }];
        let flat_before = flat.evaluate(&flat_batch).unwrap().loss_per_token;
        let fact_before = factorized.evaluate(&fact_batch).unwrap().loss_per_token;
        assert!(flat_before.is_finite() && fact_before.is_finite());
        let fact_result = factorized
            .train_batch(&fact_batch, OptimizerConfig::default())
            .unwrap();
        assert_eq!(fact_result.target_tokens, 2);
        let fact_after = factorized.evaluate(&fact_batch).unwrap();
        assert!(fact_after.primary_head_loss_per_token.unwrap().is_finite());
        assert!(fact_after.local_loss_per_token.unwrap().is_finite());
        assert!(flat.evaluate(&flat_batch).unwrap().loss_per_token > 0.0);
    }

    #[test]
    fn byte_fallback_rows_are_valid_factorized_targets() {
        let mut model = CausalLm::new_factorized(
            tiny_config(),
            vec![
                PackVocabulary {
                    pack_id: 0,
                    token_count: 2,
                },
                PackVocabulary {
                    pack_id: u16::MAX,
                    token_count: 256,
                },
            ],
            12,
        )
        .unwrap();
        let inputs = [TokenId::new(0, 0), TokenId::new(u16::MAX, 0xff)];
        let batch = [TrainingExample {
            inputs: &inputs[..1],
            targets: &inputs[1..],
        }];
        assert!(
            model
                .train_batch(&batch, OptimizerConfig::default())
                .is_ok()
        );
        let predicted = model.predict_next(&inputs[..1]).unwrap();
        assert!(model.embedding_row(predicted, 0).is_ok());
    }

    #[test]
    fn invalid_pack_and_local_targets_are_rejected() {
        let mut model = CausalLm::new_factorized(
            tiny_config(),
            vec![PackVocabulary {
                pack_id: 1,
                token_count: 2,
            }],
            5,
        )
        .unwrap();
        for target in [TokenId::new(2, 0), TokenId::new(1, 2)] {
            let inputs = [TokenId::new(1, 0)];
            let batch = [TrainingExample {
                inputs: &inputs,
                targets: &[target],
            }];
            assert!(matches!(
                model.train_batch(&batch, OptimizerConfig::default()),
                Err(ModelError::InvalidToken { .. })
            ));
        }
    }

    #[test]
    fn adam_update_changes_parameters() {
        let mut model = CausalLm::new_flat(tiny_config(), 260, 15).unwrap();
        let tokens = alternating_flat(5);
        let batch = [TrainingExample {
            inputs: &tokens[..4],
            targets: &tokens[1..],
        }];
        let before = model.parameters().to_vec();
        model
            .train_batch(&batch, OptimizerConfig::default())
            .unwrap();
        assert_ne!(before, model.parameters());
    }

    #[test]
    fn flat_model_overfits_a_small_repeated_sequence() {
        let mut model = CausalLm::new_flat(tiny_config(), 260, 21).unwrap();
        let tokens: Vec<TokenId> = (0..25)
            .map(|index| {
                TokenId::new(
                    0,
                    if index % 2 == 0 {
                        b'a' as u32
                    } else {
                        b' ' as u32
                    },
                )
            })
            .collect();
        let raw_text: Vec<u8> = tokens.iter().map(|token| token.local as u8).collect();
        let mut raw_fixture = b"a ".repeat(12);
        raw_fixture.push(b'a');
        assert_eq!(raw_text, raw_fixture);
        let batch = windows(&tokens, 4);
        let before = model.evaluate(&batch).unwrap().loss_per_token;
        for _ in 0..100 {
            model
                .train_batch(&batch, OptimizerConfig::default())
                .unwrap();
        }
        let after = model.evaluate(&batch).unwrap().loss_per_token;
        assert!(
            after < before * 0.7,
            "loss did not clearly fall: {before} -> {after}"
        );
    }

    #[test]
    fn factorized_model_overfits_the_same_raw_sequence_and_learns_packs() {
        let mut model = CausalLm::new_factorized(
            tiny_config(),
            vec![
                PackVocabulary {
                    pack_id: 1,
                    token_count: 2,
                },
                PackVocabulary {
                    pack_id: u16::MAX,
                    token_count: 256,
                },
            ],
            22,
        )
        .unwrap();
        let tokens: Vec<TokenId> = (0..25)
            .map(|index| {
                if index % 2 == 0 {
                    TokenId::new(1, 0)
                } else {
                    TokenId::new(u16::MAX, b' ' as u32)
                }
            })
            .collect();
        let raw_text: Vec<u8> = tokens
            .iter()
            .map(|token| match token.pack {
                1 => b'a',
                u16::MAX => token.local as u8,
                _ => unreachable!(),
            })
            .collect();
        let mut raw_fixture = b"a ".repeat(12);
        raw_fixture.push(b'a');
        assert_eq!(raw_text, raw_fixture);
        let batch = windows(&tokens, 4);
        let before = model.evaluate(&batch).unwrap().loss_per_token;
        for _ in 0..120 {
            model
                .train_batch(&batch, OptimizerConfig::default())
                .unwrap();
        }
        let after = model.evaluate(&batch).unwrap();
        assert!(
            after.loss_per_token < before * 0.7,
            "loss did not clearly fall: {before} -> {}",
            after.loss_per_token
        );
        assert!(after.pack_accuracy.unwrap() > 0.8);
    }

    #[test]
    fn factorized_accuracy_reports_pack_and_gold_pack_local_heads() {
        let model = CausalLm::new_factorized(
            tiny_config(),
            vec![
                PackVocabulary {
                    pack_id: 2,
                    token_count: 4,
                },
                PackVocabulary {
                    pack_id: 9,
                    token_count: 7,
                },
            ],
            3,
        )
        .unwrap();
        let tokens = [TokenId::new(2, 1), TokenId::new(9, 4)];
        let batch = [TrainingExample {
            inputs: &tokens[..1],
            targets: &tokens[1..],
        }];
        let metrics = model.evaluate(&batch).unwrap();
        assert_eq!(metrics.target_tokens, 1);
        assert!(metrics.pack_accuracy.is_some());
        assert!(metrics.local_accuracy_given_pack.is_some());
        assert_eq!(metrics.average_active_local_head_size, Some(7.0));
        assert!((metrics.pack_head_logit_fraction.unwrap() - 2.0 / 9.0).abs() < 1.0e-12);
        assert_eq!(metrics.per_pack.len(), 2);
    }

    #[test]
    fn model_serialization_round_trip_preserves_greedy_ids() {
        let mut model = CausalLm::new_factorized(
            tiny_config(),
            vec![
                PackVocabulary {
                    pack_id: 1,
                    token_count: 3,
                },
                PackVocabulary {
                    pack_id: u16::MAX,
                    token_count: 256,
                },
            ],
            31,
        )
        .unwrap();
        let tokens = [TokenId::new(1, 0), TokenId::new(u16::MAX, 97)];
        let batch = [TrainingExample {
            inputs: &tokens[..1],
            targets: &tokens[1..],
        }];
        model
            .train_batch(&batch, OptimizerConfig::default())
            .unwrap();
        let bytes = model.to_bytes().unwrap();
        let loaded = CausalLm::from_bytes(&bytes).unwrap();
        assert_eq!(loaded.parameters(), model.parameters());
        assert_eq!(
            loaded.predict_next(&tokens).unwrap(),
            model.predict_next(&tokens).unwrap()
        );
        assert_eq!(loaded.to_bytes().unwrap(), bytes);
    }

    #[test]
    fn malformed_model_bytes_and_nonfinite_parameters_are_rejected() {
        let model = CausalLm::new_flat(tiny_config(), 260, 33).unwrap();
        let bytes = model.to_bytes().unwrap();
        assert!(CausalLm::from_bytes(&bytes[..bytes.len() - 1]).is_err());
        let mut noncanonical = bytes.clone();
        noncanonical[19] = 1;
        assert!(CausalLm::from_bytes(&noncanonical).is_err());
        let mut with_nan = bytes;
        let last = with_nan.len();
        with_nan[last - 4..].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(CausalLm::from_bytes(&with_nan).is_err());

        let mut oversized_declared_count = model.to_bytes().unwrap();
        oversized_declared_count[38..46].copy_from_slice(&u64::MAX.to_le_bytes());
        assert_eq!(
            CausalLm::from_bytes(&oversized_declared_count),
            Err(ModelError::ModelTooLarge)
        );
    }

    #[test]
    fn greedy_generation_is_bounded_and_keeps_prompt() {
        let model = CausalLm::new_flat(tiny_config(), 260, 42).unwrap();
        let prompt = [TokenId::new(0, 97), TokenId::new(0, 98)];
        let generated = model.greedy_generate(&prompt, 3).unwrap();
        assert_eq!(&generated[..prompt.len()], &prompt);
        assert_eq!(generated.len(), 5);
        assert!(matches!(
            model.greedy_generate(&prompt, MAX_GENERATION_TOKENS),
            Err(ModelError::GenerationTooLong)
        ));
    }

    #[test]
    fn training_mac_estimate_is_positive_and_changes_with_active_head() {
        let flat = CausalLm::new_flat(tiny_config(), 260, 2).unwrap();
        let factorized = CausalLm::new_factorized(
            tiny_config(),
            vec![
                PackVocabulary {
                    pack_id: 1,
                    token_count: 2,
                },
                PackVocabulary {
                    pack_id: u16::MAX,
                    token_count: 256,
                },
            ],
            2,
        )
        .unwrap();
        let input = [TokenId::new(1, 0), TokenId::new(1, 1)];
        let flat_input = [TokenId::new(0, 97), TokenId::new(0, 98)];
        let fact = [TrainingExample {
            inputs: &input[..1],
            targets: &input[1..],
        }];
        let flat_batch = [TrainingExample {
            inputs: &flat_input[..1],
            targets: &flat_input[1..],
        }];
        assert!(factorized.estimate_training_macs(&fact).unwrap() > 0);
        assert!(
            factorized.estimate_training_macs(&fact).unwrap()
                < flat.estimate_training_macs(&flat_batch).unwrap()
        );
    }

    #[test]
    fn cross_entropy_is_invariant_to_large_common_offsets() {
        for offset in [0.0_f32, 1.0e20, -1.0e20, f32::MAX, -f32::MAX] {
            let (loss, gradient) = softmax_cross_entropy(&[offset, offset], 0, 1.0).unwrap();
            assert!(
                (loss - std::f64::consts::LN_2).abs() < 1.0e-12,
                "offset={offset} loss={loss}"
            );
            assert_eq!(gradient, [-0.5, 0.5]);
            assert_eq!(softmax_loss(&[offset, offset], 0).unwrap(), loss);
        }
    }

    #[test]
    fn loss_only_matches_training_loss_on_extreme_and_invalid_logits() {
        for logits in [
            vec![0.0],
            vec![-1000.0, 1000.0, 0.0],
            vec![-f32::MAX, f32::MAX],
            vec![f32::NAN],
            vec![f32::INFINITY],
            vec![f32::NEG_INFINITY],
        ] {
            for target in 0..=logits.len() {
                assert_eq!(
                    softmax_loss(&logits, target),
                    softmax_cross_entropy(&logits, target, 1.0).map(|(loss, _)| loss)
                );
            }
        }
    }

    #[test]
    fn recurrent_and_both_head_gradients_match_finite_differences() {
        let config = ModelConfig {
            hidden_size: 2,
            context_length: 3,
        };
        let flat = CausalLm::new_flat(config, 256, 19).unwrap();
        let factorized = CausalLm::new_factorized(
            config,
            vec![
                PackVocabulary {
                    pack_id: 1,
                    token_count: 2,
                },
                PackVocabulary {
                    pack_id: 7,
                    token_count: 3,
                },
            ],
            19,
        )
        .unwrap();
        for (mut model, tokens) in [
            (
                flat,
                vec![
                    TokenId::new(0, 0),
                    TokenId::new(0, 1),
                    TokenId::new(0, 2),
                    TokenId::new(0, 0),
                ],
            ),
            (
                factorized,
                vec![
                    TokenId::new(1, 0),
                    TokenId::new(7, 1),
                    TokenId::new(1, 1),
                    TokenId::new(7, 0),
                ],
            ),
        ] {
            // One aligned window plus one final-only target tests recurrent
            // carry and normalization by total targets rather than examples.
            let examples = [
                TrainingExample {
                    inputs: &tokens[..3],
                    targets: &tokens[1..],
                },
                TrainingExample {
                    inputs: &tokens[1..3],
                    targets: &tokens[3..],
                },
            ];
            let mut updated = model.clone();
            let optimizer = OptimizerConfig {
                gradient_clip_norm: 1000.0,
                ..OptimizerConfig::default()
            };
            let metrics = updated.train_batch(&examples, optimizer).unwrap();
            assert!(metrics.gradient_norm < f64::from(optimizer.gradient_clip_norm));
            for index in 0..model.weights.len() {
                let original = model.weights[index];
                model.weights[index] = original + 1.0e-3;
                let plus = model.evaluate(&examples).unwrap().loss_per_token;
                let positive_weight = model.weights[index];
                model.weights[index] = original - 1.0e-3;
                let minus = model.evaluate(&examples).unwrap().loss_per_token;
                let negative_weight = model.weights[index];
                model.weights[index] = original;
                let numerical =
                    (plus - minus) / (f64::from(positive_weight) - f64::from(negative_weight));
                let analytical = f64::from(updated.first_moment[index] / (1.0 - optimizer.beta1));
                assert!(
                    (numerical - analytical).abs() < 2.0e-5 + 0.01 * analytical.abs(),
                    "parameter={index} numerical={numerical} analytical={analytical}"
                );
            }
            // The documented MAC formula must hold for both target shapes.
            let rows = match &model.kind {
                HeadKind::Flat { vocabulary_size } => examples
                    .iter()
                    .map(|e| e.targets.len() * vocabulary_size)
                    .sum::<usize>(),
                HeadKind::Factorized { packs } => examples
                    .iter()
                    .flat_map(|e| e.targets)
                    .map(|t| packs.len() + packs[model.pack_index(t.pack).unwrap()].token_count)
                    .sum(),
            };
            assert_eq!(
                model.estimate_training_macs(&examples).unwrap(),
                ((3 * 3 - 1 + 3 * 2 - 1) * 4 + 3 * rows * 2) as u64
            );
        }
    }

    #[test]
    fn compact_header_cannot_allocate_a_large_unrepresented_model() {
        let mut bytes = CausalLm::new_flat(tiny_config(), 256, 1)
            .unwrap()
            .to_bytes()
            .unwrap();
        bytes[12..16].copy_from_slice(&512_u32.to_le_bytes());
        bytes[16..20].copy_from_slice(&16_u32.to_le_bytes());
        bytes[34..38].copy_from_slice(&8192_u32.to_le_bytes());
        bytes[38..46].copy_from_slice(&0_u64.to_le_bytes());
        bytes.truncate(46);
        assert_eq!(
            CausalLm::from_bytes(&bytes),
            Err(ModelError::MalformedArtifact(
                "parameter count does not match dimensions"
            ))
        );
        assert_eq!(
            CausalLm::flat_model(
                ModelConfig {
                    hidden_size: 512,
                    context_length: 16
                },
                8192,
                1,
                Some(&[])
            ),
            Err(ModelError::MalformedArtifact(
                "parameter count does not match dimensions"
            ))
        );
    }

    #[test]
    fn failed_adam_update_preserves_weights_moments_and_step() {
        let mut model = CausalLm::new_flat(tiny_config(), 256, 1).unwrap();
        model.weights[1] = -f32::MAX;
        let before = model.clone();
        let mut gradients = vec![0.0; model.weights.len()];
        gradients[..2].fill(1.0);
        let optimizer = OptimizerConfig {
            learning_rate: f32::MAX,
            ..OptimizerConfig::default()
        };
        assert_eq!(
            model.apply_adam(&gradients, optimizer),
            Err(ModelError::NonFiniteComputation)
        );
        assert_eq!(model, before);
    }

    #[test]
    fn overflowing_adam_moment_is_rejected_without_changing_state() {
        let mut model = CausalLm::new_flat(tiny_config(), 256, 1).unwrap();
        let before = model.clone();
        let gradients = vec![1.0e22; model.weights.len()];
        assert_eq!(
            model.apply_adam(&gradients, OptimizerConfig::default()),
            Err(ModelError::NonFiniteComputation)
        );
        assert_eq!(model, before);
    }

    #[test]
    fn inference_rejects_overflowing_logits_in_both_heads() {
        let mut flat = CausalLm::new_flat(tiny_config(), 256, 1).unwrap();
        flat.weights.fill(10.0);
        flat.weights[flat.layout.flat_output..].fill(f32::MAX);
        assert_eq!(
            flat.predict_next(&[TokenId::new(0, 0)]),
            Err(ModelError::NonFiniteComputation)
        );
        let mut factorized = CausalLm::new_factorized(
            tiny_config(),
            vec![PackVocabulary {
                pack_id: 1,
                token_count: 2,
            }],
            1,
        )
        .unwrap();
        factorized.weights.fill(10.0);
        factorized.weights[factorized.layout.pack_output..].fill(f32::MAX);
        assert_eq!(
            factorized.predict_next(&[TokenId::new(1, 0)]),
            Err(ModelError::NonFiniteComputation)
        );
        factorized.weights[factorized.layout.pack_output..factorized.layout.local_output].fill(0.0);
        assert_eq!(
            factorized.predict_next(&[TokenId::new(1, 0)]),
            Err(ModelError::NonFiniteComputation)
        );
    }

    #[test]
    fn mac_estimate_validates_input_ids_and_reports_example_index() {
        let mut model = CausalLm::new_flat(tiny_config(), 256, 1).unwrap();
        let invalid = [TokenId::new(0, 256)];
        let valid = [TokenId::new(0, 0)];
        assert!(matches!(
            model.estimate_training_macs(&[TrainingExample {
                inputs: &invalid,
                targets: &valid
            }]),
            Err(ModelError::InvalidToken { .. })
        ));
        let batch = [
            TrainingExample {
                inputs: &valid,
                targets: &valid,
            },
            TrainingExample {
                inputs: &[],
                targets: &valid,
            },
        ];
        assert!(matches!(
            model.train_batch(&batch, OptimizerConfig::default()),
            Err(ModelError::InvalidExample { index: 1, .. })
        ));
    }
}
