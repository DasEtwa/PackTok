//! Independent scale experiment, derived from the M5 reference Transformer.
//! M5's frozen model/protocol are unchanged. This module adds BF16 weights,
//! FP32 normalization/attention-softmax/loss, a causal semantic-input channel,
//! and explicit layer-recomputation backpropagation. Dense attention remains
//! quadratic within each layer: no Flash Attention implementation is claimed.

use candle_core::{D, DType, Device, Module, Tensor, Var, backprop::GradStore};
use candle_nn::{Embedding, Linear, VarBuilder, VarMap};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const SEMANTIC_ROWS: usize = 6;
pub const MAX_PARAMETERS: u128 = 2_000_000_000;
const MAX_CPU_PARAMETERS: usize = 5_000_000;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    pub layers: usize,
    pub hidden: usize,
    pub heads: usize,
    pub ffn: usize,
    pub context: usize,
    pub vocab: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            layers: 2,
            hidden: 64,
            heads: 4,
            ffn: 256,
            context: 128,
            vocab: 512,
        }
    }
}

impl ModelConfig {
    pub fn preset(name: &str, context: usize) -> candle_core::Result<Self> {
        let config = match name {
            "tiny" => Self {
                context,
                ..Self::default()
            },
            "100m" => Self {
                layers: 8,
                hidden: 1024,
                heads: 16,
                ffn: 4096,
                context,
                vocab: 512,
            },
            "1b" => Self {
                layers: 36,
                hidden: 1536,
                heads: 24,
                ffn: 6144,
                context,
                vocab: 512,
            },
            _ => candle_core::bail!("unknown model preset {name}; expected tiny, 100m or 1b"),
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> candle_core::Result<()> {
        self.parameter_count().map(|_| ())
    }

    /// Includes the six-row semantic embedding in both experimental variants.
    /// Bounds and u128 accounting precede all model tensor allocations.
    pub fn parameter_count(&self) -> candle_core::Result<usize> {
        if !(1..=64).contains(&self.layers)
            || !(2..=8192).contains(&self.hidden)
            || !(1..=128).contains(&self.heads)
            || self.hidden.checked_rem(self.heads) != Some(0)
            || !(1..=32768).contains(&self.ffn)
            || !(1..=4096).contains(&self.context)
            || self.vocab != 512
        {
            candle_core::bail!("invalid semantic-scale architecture");
        }
        let (v, c, d, l, f) = (
            self.vocab as u128,
            self.context as u128,
            self.hidden as u128,
            self.layers as u128,
            self.ffn as u128,
        );
        let n = v * d
            + c * d
            + SEMANTIC_ROWS as u128 * d
            + l * (4 * (d * d + d) + 2 * d * f + f + d + 2 * d)
            + d
            + v * d
            + v;
        if n > MAX_PARAMETERS {
            candle_core::bail!("semantic-scale parameter cap exceeded");
        }
        usize::try_from(n).map_err(candle_core::Error::wrap)
    }
}

struct RmsNorm {
    scale: Tensor,
}
impl RmsNorm {
    fn new(hidden: usize, vb: VarBuilder<'_>) -> candle_core::Result<Self> {
        Ok(Self {
            scale: vb.get_with_hints(hidden, "weight", candle_nn::Init::Const(1.0))?,
        })
    }
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let float = x.to_dtype(DType::F32)?;
        float
            .broadcast_div(&(float.sqr()?.mean_keepdim(D::Minus1)? + 1e-5)?.sqrt()?)?
            .broadcast_mul(&self.scale.to_dtype(DType::F32)?)?
            .to_dtype(x.dtype())
    }
}

struct Block {
    norm1: RmsNorm,
    norm2: RmsNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    up: Linear,
    down: Linear,
}
impl Block {
    fn new(c: &ModelConfig, vb: VarBuilder<'_>) -> candle_core::Result<Self> {
        Ok(Self {
            norm1: RmsNorm::new(c.hidden, vb.pp("norm1"))?,
            norm2: RmsNorm::new(c.hidden, vb.pp("norm2"))?,
            q: candle_nn::linear(c.hidden, c.hidden, vb.pp("q"))?,
            k: candle_nn::linear(c.hidden, c.hidden, vb.pp("k"))?,
            v: candle_nn::linear(c.hidden, c.hidden, vb.pp("v"))?,
            out: candle_nn::linear(c.hidden, c.hidden, vb.pp("out"))?,
            up: candle_nn::linear(c.hidden, c.ffn, vb.pp("up"))?,
            down: candle_nn::linear(c.ffn, c.hidden, vb.pp("down"))?,
        })
    }
    fn forward(&self, x: &Tensor, mask: &Tensor, heads: usize) -> candle_core::Result<Tensor> {
        let (batch, time, hidden) = x.dims3()?;
        let width = hidden / heads;
        let z = self.norm1.forward(x)?;
        let split = |p: &Linear| -> candle_core::Result<Tensor> {
            p.forward(&z)?
                .reshape((batch, time, heads, width))?
                .transpose(1, 2)?
                .contiguous()
        };
        let (q, k, v) = (split(&self.q)?, split(&self.k)?, split(&self.v)?);
        let scores = (q.matmul(&k.transpose(2, 3)?)?.to_dtype(DType::F32)?
            / (width as f64).sqrt())?
        .broadcast_add(mask)?;
        let probability = candle_nn::ops::softmax(&scores, D::Minus1)?.to_dtype(x.dtype())?;
        let attention = probability
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, time, hidden))?;
        let residual = (x + self.out.forward(&attention)?)?;
        let ff = self.down.forward(
            &self
                .up
                .forward(&self.norm2.forward(&residual)?)?
                .gelu_erf()?,
        )?;
        residual + ff
    }
}

pub struct Transformer {
    pub config: ModelConfig,
    pub vars: VarMap,
    embeddings: Embedding,
    positions: Embedding,
    semantic: Embedding,
    blocks: Vec<Block>,
    norm: RmsNorm,
    head: Linear,
}
impl Transformer {
    /// CPU construction is deliberately restricted to tiny correctness fixtures.
    /// A preset's parameter count can be inspected without constructing a model.
    pub fn new(
        config: ModelConfig,
        seed: u64,
        device: &Device,
        dtype: DType,
    ) -> candle_core::Result<Self> {
        let parameters = config.parameter_count()?;
        if !matches!(dtype, DType::F32 | DType::BF16) {
            candle_core::bail!("model dtype must be F32 or BF16");
        }
        if matches!(device, Device::Cpu) && parameters > MAX_CPU_PARAMETERS {
            candle_core::bail!(
                "CPU model allocation is restricted to <=5M-parameter correctness fixtures"
            );
        }
        let mut shapes = BTreeMap::new();
        shapes.insert(
            "embedding.weight".to_string(),
            vec![config.vocab, config.hidden],
        );
        shapes.insert(
            "position.weight".to_string(),
            vec![config.context, config.hidden],
        );
        shapes.insert(
            "semantic.weight".to_string(),
            vec![SEMANTIC_ROWS, config.hidden],
        );
        shapes.insert("norm.weight".to_string(), vec![config.hidden]);
        shapes.insert("head.weight".to_string(), vec![config.vocab, config.hidden]);
        shapes.insert("head.bias".to_string(), vec![config.vocab]);
        for i in 0..config.layers {
            for name in ["norm1", "norm2"] {
                shapes.insert(format!("block.{i}.{name}.weight"), vec![config.hidden]);
            }
            for (name, input, output) in [
                ("q", config.hidden, config.hidden),
                ("k", config.hidden, config.hidden),
                ("v", config.hidden, config.hidden),
                ("out", config.hidden, config.hidden),
                ("up", config.hidden, config.ffn),
                ("down", config.ffn, config.hidden),
            ] {
                shapes.insert(format!("block.{i}.{name}.weight"), vec![output, input]);
                shapes.insert(format!("block.{i}.{name}.bias"), vec![output]);
            }
        }
        let vars = VarMap::new();
        let mut rng = Rng::new(seed);
        {
            let mut map = vars
                .data()
                .lock()
                .map_err(|_| candle_core::Error::Msg("model variable lock poisoned".into()))?;
            for (name, shape) in shapes {
                let count: usize = shape.iter().product();
                let values = if name.ends_with(".bias") {
                    vec![0.0_f32; count]
                } else if name.contains("norm") {
                    vec![1.0_f32; count]
                } else {
                    (0..count)
                        .map(|_| (rng.unit() * 0.04 - 0.02) as f32)
                        .collect()
                };
                let tensor = Tensor::from_vec(values, shape, device)?.to_dtype(dtype)?;
                map.insert(name, Var::from_tensor(&tensor)?);
            }
        }
        let vb = VarBuilder::from_varmap(&vars, dtype, device);
        let embeddings = candle_nn::embedding(config.vocab, config.hidden, vb.pp("embedding"))?;
        let positions = candle_nn::embedding(config.context, config.hidden, vb.pp("position"))?;
        let semantic = candle_nn::embedding(SEMANTIC_ROWS, config.hidden, vb.pp("semantic"))?;
        let blocks = (0..config.layers)
            .map(|i| Block::new(&config, vb.pp(format!("block.{i}"))))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let norm = RmsNorm::new(config.hidden, vb.pp("norm"))?;
        let head = candle_nn::linear(config.hidden, config.vocab, vb.pp("head"))?;
        let model = Self {
            config,
            vars,
            embeddings,
            positions,
            semantic,
            blocks,
            norm,
            head,
        };
        let actual: usize = model.named_vars().values().map(|v| v.elem_count()).sum();
        if actual != parameters {
            candle_core::bail!("analytical/model parameter mismatch");
        }
        Ok(model)
    }

    pub fn named_vars(&self) -> BTreeMap<String, Var> {
        self.vars
            .data()
            .lock()
            .expect("model variable lock")
            .iter()
            .map(|(name, variable)| (name.clone(), variable.clone()))
            .collect()
    }

    fn embedding_forward(&self, tokens: &Tensor, semantic: &Tensor) -> candle_core::Result<Tensor> {
        let (batch, time) = tokens.dims2()?;
        if batch == 0
            || time == 0
            || time > self.config.context
            || semantic.dims() != [batch, time]
            || tokens.dtype() != DType::U32
            || semantic.dtype() != DType::U32
        {
            candle_core::bail!("invalid lexical/semantic input shape or dtype");
        }
        // Validate IDs before CUDA embedding lookup as well as on CPU. The
        // reductions synchronize, trading a small cost for explicit rejection
        // instead of a device-side indexing fault on malformed model inputs.
        if tokens.max_all()?.to_scalar::<u32>()? as usize >= self.config.vocab
            || semantic.max_all()?.to_scalar::<u32>()? as usize >= SEMANTIC_ROWS
        {
            candle_core::bail!("lexical or semantic ID outside model vocabulary");
        }
        let positions = Tensor::arange(0_u32, time as u32, tokens.device())?;
        let lexical = self
            .embeddings
            .forward(tokens)?
            .broadcast_add(&self.positions.forward(&positions)?)?;
        // Row zero is never a learned no-semantic signal. Masking guarantees
        // neutral zero features and identical architectures in both variants.
        let mask = semantic
            .to_dtype(DType::F32)?
            .clamp(0_f32, 1_f32)?
            .unsqueeze(D::Minus1)?
            .to_dtype(lexical.dtype())?;
        lexical + self.semantic.forward(semantic)?.broadcast_mul(&mask)?
    }

    fn attention_mask(&self, time: usize, device: &Device) -> candle_core::Result<Tensor> {
        let values: Vec<f32> = (0..time)
            .flat_map(|i| (0..time).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
            .collect();
        Tensor::from_vec(values, (1, 1, time, time), device)
    }

    fn output(&self, hidden: &Tensor) -> candle_core::Result<Tensor> {
        self.head
            .forward(&self.norm.forward(hidden)?)?
            .to_dtype(DType::F32)
    }

    /// Produces F32 logits. Semantic inputs must themselves be causally derived;
    /// the data adapter is responsible for delaying a feature until observed.
    pub fn forward(&self, tokens: &Tensor, semantic: &Tensor) -> candle_core::Result<Tensor> {
        let mut x = self.embedding_forward(tokens, semantic)?;
        let mask = self.attention_mask(tokens.dim(1)?, tokens.device())?;
        for block in &self.blocks {
            x = block.forward(&x, &mask, self.config.heads)?;
        }
        self.output(&x)
    }

    /// Inference/evaluation path: retain at most one block's temporary graph.
    /// Detaching only the final logits would retain all parameter-dependent
    /// activation graphs until the complete forward finishes, even without
    /// backward. There is no dropout, so values equal ordinary forward.
    pub fn forward_detached(
        &self,
        tokens: &Tensor,
        semantic: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let mut hidden = self.embedding_forward(tokens, semantic)?.detach();
        let mask = self.attention_mask(tokens.dim(1)?, tokens.device())?;
        for block in &self.blocks {
            hidden = block.forward(&hidden, &mask, self.config.heads)?.detach();
        }
        Ok(self.output(&hidden)?.detach())
    }

    /// Returns mean cross-entropy and detached F32 gradients by stable name.
    /// With checkpointing enabled only detached layer inputs survive the forward
    /// sweep. Backward recomputes one layer at a time, propagating its input VJP.
    /// No optimizer state/update is performed here; callers may accumulate means.
    pub fn gradients(
        &self,
        tokens: &Tensor,
        semantic: &Tensor,
        targets: &Tensor,
        checkpointed: bool,
    ) -> candle_core::Result<(f64, BTreeMap<String, Tensor>)> {
        let mut gradients = BTreeMap::new();
        let value = self.accumulate_gradients(
            tokens,
            semantic,
            targets,
            checkpointed,
            &mut gradients,
            1.0,
        )?;
        Ok((value, gradients))
    }

    /// Accumulates each layer's detached F32 parameter gradient directly into
    /// caller storage. Checkpointed microbatches never create a second complete
    /// full-model gradient map. `scale` normally equals 1 / microbatch count.
    pub fn accumulate_gradients(
        &self,
        tokens: &Tensor,
        semantic: &Tensor,
        targets: &Tensor,
        checkpointed: bool,
        output: &mut BTreeMap<String, Tensor>,
        scale: f64,
    ) -> candle_core::Result<f64> {
        if !scale.is_finite() || scale <= 0.0 {
            candle_core::bail!("gradient accumulation scale must be finite and positive");
        }
        if !output.is_empty() {
            self.validate_gradient_set(output)?;
        }
        if !checkpointed {
            let loss = loss(&self.forward(tokens, semantic)?, targets)?;
            let value = f64::from(loss.to_scalar::<f32>()?);
            let store = loss.backward()?;
            self.collect_gradients(&store, output, scale)?;
            self.validate_gradient_set(output)?;
            return Ok(value);
        }
        let mut hidden = self.embedding_forward(tokens, semantic)?.detach();
        let mask = self.attention_mask(tokens.dim(1)?, tokens.device())?;
        let mut saved = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            saved.push(hidden.clone());
            hidden = block.forward(&hidden, &mask, self.config.heads)?.detach();
        }
        let (value, mut upstream) = {
            let input = Var::from_tensor(&hidden)?;
            let loss = loss(&self.output(&input)?, targets)?;
            let value = f64::from(loss.to_scalar::<f32>()?);
            let store = loss.backward()?;
            self.collect_gradients(&store, output, scale)?;
            let upstream = store
                .get(&input)
                .ok_or_else(|| candle_core::Error::Msg("missing head input gradient".into()))?
                .detach();
            (value, upstream)
        };
        drop(hidden);
        for block in self.blocks.iter().rev() {
            let saved_input = saved.pop().expect("one saved input per block");
            upstream = {
                let input = Var::from_tensor(&saved_input)?;
                let recomputed = block.forward(&input, &mask, self.config.heads)?;
                let surrogate = (recomputed.to_dtype(DType::F32)?
                    * upstream.to_dtype(DType::F32)?)?
                .sum_all()?;
                let store = surrogate.backward()?;
                self.collect_gradients(&store, output, scale)?;
                store
                    .get(&input)
                    .ok_or_else(|| {
                        candle_core::Error::Msg("missing recomputed block input gradient".into())
                    })?
                    .detach()
            };
        }
        {
            let embedding = self.embedding_forward(tokens, semantic)?;
            let surrogate =
                (embedding.to_dtype(DType::F32)? * upstream.to_dtype(DType::F32)?)?.sum_all()?;
            self.collect_gradients(&surrogate.backward()?, output, scale)?;
        }
        self.validate_gradient_set(output)?;
        Ok(value)
    }

    fn collect_gradients(
        &self,
        store: &GradStore,
        output: &mut BTreeMap<String, Tensor>,
        scale: f64,
    ) -> candle_core::Result<()> {
        for (name, variable) in self.named_vars() {
            if let Some(gradient) = store.get(&variable) {
                let gradient = gradient.to_dtype(DType::F32)?.detach();
                let gradient = if scale == 1.0 {
                    gradient
                } else {
                    (gradient * scale)?.detach()
                };
                if let Some(previous) = output.remove(&name) {
                    output.insert(name, (previous + gradient)?.detach());
                } else {
                    output.insert(name, gradient);
                }
            }
        }
        Ok(())
    }

    fn validate_gradient_set(
        &self,
        gradients: &BTreeMap<String, Tensor>,
    ) -> candle_core::Result<()> {
        let variables = self.named_vars();
        if gradients.len() != variables.len()
            || variables.iter().any(|(name, var)| {
                gradients
                    .get(name)
                    .is_none_or(|g| g.dims() != var.dims() || g.dtype() != DType::F32)
            })
        {
            candle_core::bail!("incomplete model gradient set");
        }
        Ok(())
    }
}

pub fn loss(logits: &Tensor, targets: &Tensor) -> candle_core::Result<Tensor> {
    let (batch, time, vocab) = logits.dims3()?;
    if targets.dims() != [batch, time] || targets.dtype() != DType::U32 || batch * time == 0 {
        candle_core::bail!("invalid target shape or dtype");
    }
    if targets.max_all()?.to_scalar::<u32>()? as usize >= vocab {
        candle_core::bail!("target ID outside model vocabulary");
    }
    candle_nn::loss::cross_entropy(
        &logits
            .to_dtype(DType::F32)?
            .reshape((batch * time, vocab))?,
        &targets.flatten_all()?,
    )
}

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0x9e3779b97f4a7c15 } else { seed })
    }
    fn unit(&mut self) -> f64 {
        let mut value = self.0;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.0 = value;
        (value >> 11) as f64 / (1_u64 << 53) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> ModelConfig {
        ModelConfig {
            layers: 2,
            hidden: 8,
            heads: 2,
            ffn: 16,
            context: 8,
            vocab: 512,
        }
    }
    fn sample() -> candle_core::Result<(Tensor, Tensor, Tensor)> {
        Ok((
            Tensor::new(&[[1_u32, 2, 3, 4]], &Device::Cpu)?,
            Tensor::new(&[[0_u32, 1, 0, 2]], &Device::Cpu)?,
            Tensor::new(&[[2_u32, 3, 4, 5]], &Device::Cpu)?,
        ))
    }
    fn assert_close(left: &Tensor, right: &Tensor, tolerance: f32) -> candle_core::Result<()> {
        assert_eq!(left.dims(), right.dims());
        for (a, b) in left
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .zip(right.flatten_all()?.to_vec1::<f32>()?)
        {
            assert!(
                (*a - b).abs() <= tolerance,
                "{a} != {b} (tolerance {tolerance})"
            );
        }
        Ok(())
    }

    #[test]
    fn presets_account_without_allocating_large_models() -> candle_core::Result<()> {
        let tiny = ModelConfig::preset("tiny", 128)?;
        assert!(tiny.parameter_count()? < 1_000_000);
        let medium = ModelConfig::preset("100m", 128)?;
        assert!((100_000_000..105_000_000).contains(&medium.parameter_count()?));
        let billion = ModelConfig::preset("1b", 128)?;
        assert!((1_000_000_000..1_100_000_000).contains(&billion.parameter_count()?));
        assert!(Transformer::new(billion, 1, &Device::Cpu, DType::F32).is_err());
        assert!(ModelConfig::preset("1b", 0).is_err());
        assert!(ModelConfig::preset("other", 8).is_err());
        let invalid = ModelConfig {
            layers: usize::MAX,
            ..tiny
        };
        assert!(invalid.parameter_count().is_err());
        Ok(())
    }

    #[test]
    fn deterministic_initialization_and_exact_parameter_count() -> candle_core::Result<()> {
        let first = Transformer::new(tiny(), 42, &Device::Cpu, DType::F32)?;
        let second = Transformer::new(tiny(), 42, &Device::Cpu, DType::F32)?;
        assert_eq!(
            first.config.parameter_count()?,
            first
                .named_vars()
                .values()
                .map(|v| v.elem_count())
                .sum::<usize>()
        );
        let (tokens, semantic, _) = sample()?;
        assert_close(
            &first.forward(&tokens, &semantic)?,
            &second.forward(&tokens, &semantic)?,
            0.0,
        )
    }

    #[test]
    fn checkpointed_gradients_match_full_graph_for_every_parameter() -> candle_core::Result<()> {
        let model = Transformer::new(tiny(), 123, &Device::Cpu, DType::F32)?;
        let (tokens, semantic, targets) = sample()?;
        let (normal_loss, normal) = model.gradients(&tokens, &semantic, &targets, false)?;
        let (checkpoint_loss, checkpoint) = model.gradients(&tokens, &semantic, &targets, true)?;
        assert!((normal_loss - checkpoint_loss).abs() < 1e-6);
        assert_eq!(normal.len(), model.named_vars().len());
        assert_eq!(normal.len(), checkpoint.len());
        for (name, gradient) in &normal {
            assert_close(gradient, &checkpoint[name], 1e-5)?;
            assert!(
                gradient
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .iter()
                    .all(|value| value.is_finite())
            );
        }
        Ok(())
    }

    #[test]
    fn checkpointed_microbatch_accumulation_matches_full_batch_mean() -> candle_core::Result<()> {
        let model = Transformer::new(tiny(), 19, &Device::Cpu, DType::F32)?;
        let tokens = Tensor::new(&[[1_u32, 2, 3, 4], [5_u32, 6, 7, 8]], &Device::Cpu)?;
        let semantic = Tensor::new(&[[0_u32, 1, 0, 2], [3_u32, 0, 4, 0]], &Device::Cpu)?;
        let targets = Tensor::new(&[[2_u32, 3, 4, 5], [6_u32, 7, 8, 9]], &Device::Cpu)?;
        let (full_loss, full) = model.gradients(&tokens, &semantic, &targets, false)?;
        let mut accumulated = BTreeMap::new();
        let mut micro_loss = 0.0;
        for index in 0..2 {
            micro_loss += 0.5
                * model.accumulate_gradients(
                    &tokens.narrow(0, index, 1)?,
                    &semantic.narrow(0, index, 1)?,
                    &targets.narrow(0, index, 1)?,
                    true,
                    &mut accumulated,
                    0.5,
                )?;
        }
        assert!((full_loss - micro_loss).abs() < 1e-6);
        for (name, gradient) in full {
            assert_close(&gradient, &accumulated[&name], 1e-5)?;
        }
        Ok(())
    }

    #[test]
    fn zero_semantic_feature_is_neutral_and_attention_is_causal() -> candle_core::Result<()> {
        let model = Transformer::new(tiny(), 9, &Device::Cpu, DType::F32)?;
        let (tokens, semantic, _) = sample()?;
        let zero = Tensor::zeros((1, 4), DType::U32, &Device::Cpu)?;
        let before = model.forward(&tokens, &zero)?;
        let variable = &model.named_vars()["semantic.weight"];
        variable.set(&Tensor::ones(variable.shape(), DType::F32, &Device::Cpu)?)?;
        assert_close(&before, &model.forward(&tokens, &zero)?, 0.0)?;
        let changed = Tensor::new(&[[1_u32, 2, 99, 100]], &Device::Cpu)?;
        assert_close(
            &before.narrow(1, 0, 2)?,
            &model.forward(&changed, &zero)?.narrow(1, 0, 2)?,
            1e-7,
        )?;
        let semantic_logits = model.forward(&tokens, &semantic)?;
        assert!(
            before
                .sub(&semantic_logits)?
                .abs()?
                .sum_all()?
                .to_scalar::<f32>()?
                > 0.0
        );
        Ok(())
    }

    #[test]
    fn detached_inference_preserves_logits_and_invalid_inputs_return_errors()
    -> candle_core::Result<()> {
        let model = Transformer::new(tiny(), 77, &Device::Cpu, DType::F32)?;
        let (tokens, semantic, targets) = sample()?;
        assert_close(
            &model.forward(&tokens, &semantic)?,
            &model.forward_detached(&tokens, &semantic)?,
            0.0,
        )?;
        let invalid = Tensor::new(&[[0_u32, 6, 0, 0]], &Device::Cpu)?;
        assert!(model.forward_detached(&tokens, &invalid).is_err());
        let bad_token = Tensor::new(&[[0_u32, 512, 0, 0]], &Device::Cpu)?;
        assert!(model.forward(&bad_token, &semantic).is_err());
        assert!(loss(&model.forward(&tokens, &semantic)?, &bad_token).is_err());
        let mut gradients = BTreeMap::new();
        assert!(
            model
                .accumulate_gradients(&tokens, &semantic, &targets, true, &mut gradients, f64::NAN)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn baseline_zero_features_still_produce_a_complete_checkpoint_gradient_set()
    -> candle_core::Result<()> {
        let model = Transformer::new(tiny(), 7, &Device::Cpu, DType::F32)?;
        let (tokens, _, targets) = sample()?;
        let zero = Tensor::zeros((1, 4), DType::U32, &Device::Cpu)?;
        let (_, grads) = model.gradients(&tokens, &zero, &targets, true)?;
        assert_eq!(
            grads["semantic.weight"]
                .abs()?
                .sum_all()?
                .to_scalar::<f32>()?,
            0.0
        );
        Ok(())
    }
}
