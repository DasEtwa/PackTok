use candle_core::{D, DType, Device, Module, Tensor, Var};
use candle_nn::{Embedding, Linear, VarBuilder, VarMap};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

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
            layers: 8,
            hidden: 384,
            heads: 6,
            ffn: 1536,
            context: 256,
            vocab: 512,
        }
    }
}
impl ModelConfig {
    pub fn validate(&self) -> candle_core::Result<()> {
        self.parameter_count().map(|_| ())
    }
    pub fn parameter_count(&self) -> candle_core::Result<usize> {
        if self.layers == 0
            || self.layers > 16
            || self.hidden < 2
            || self.hidden > 1024
            || self.heads == 0
            || !self.hidden.is_multiple_of(self.heads)
            || self.ffn == 0
            || self.ffn > 4096
            || self.context == 0
            || self.context > 1024
            || self.vocab != 512
        {
            candle_core::bail!("invalid M5 architecture");
        }
        // Do all analytical accounting in u128; never allocate from an unchecked layout.
        let (v, c, d, l, f) = (
            self.vocab as u128,
            self.context as u128,
            self.hidden as u128,
            self.layers as u128,
            self.ffn as u128,
        );
        let n = v * d + c * d + l * (4 * (d * d + d) + 2 * d * f + f + d + 2 * d) + d + v * d + v;
        if n > 128_000_000 {
            candle_core::bail!("M5 parameter limit");
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
        x.broadcast_div(&(x.sqr()?.mean_keepdim(D::Minus1)? + 1e-5)?.sqrt()?)?
            .broadcast_mul(&self.scale)
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
        let scores =
            (q.matmul(&k.transpose(2, 3)?)? / (width as f64).sqrt())?.broadcast_add(mask)?;
        let prob = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let attention = prob
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, time, hidden))?;
        let x = (x + self.out.forward(&attention)?)?;
        let ff = self
            .down
            .forward(&self.up.forward(&self.norm2.forward(&x)?)?.gelu_erf()?)?;
        x + ff
    }
}
pub struct Transformer {
    pub config: ModelConfig,
    pub vars: VarMap,
    embeddings: Embedding,
    positions: Embedding,
    blocks: Vec<Block>,
    norm: RmsNorm,
    head: Linear,
}
impl Transformer {
    pub fn new(config: ModelConfig, seed: u64, device: &Device) -> candle_core::Result<Self> {
        config.validate()?;
        let mut shapes = BTreeMap::new();
        shapes.insert(
            "embedding.weight".to_string(),
            vec![config.vocab, config.hidden],
        );
        shapes.insert(
            "position.weight".to_string(),
            vec![config.context, config.hidden],
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
        let device_vars = VarMap::new();
        let mut rng = Rng::new(seed);
        {
            let mut map = device_vars.data().lock().unwrap();
            for (name, shape) in shapes {
                let count = shape.iter().product();
                let values = if name.ends_with(".bias") {
                    vec![0.0; count]
                } else if name.contains("norm") {
                    vec![1.0; count]
                } else {
                    (0..count)
                        .map(|_| (rng.unit() * 0.04 - 0.02) as f32)
                        .collect()
                };
                map.insert(
                    name,
                    Var::from_tensor(&Tensor::from_vec(values, shape, device)?)?,
                );
            }
        }
        let vb = VarBuilder::from_varmap(&device_vars, DType::F32, device);
        let embeddings = candle_nn::embedding(config.vocab, config.hidden, vb.pp("embedding"))?;
        let positions = candle_nn::embedding(config.context, config.hidden, vb.pp("position"))?;
        let mut blocks = Vec::new();
        for i in 0..config.layers {
            blocks.push(Block::new(&config, vb.pp(format!("block.{i}")))?);
        }
        let norm = RmsNorm::new(config.hidden, vb.pp("norm"))?;
        let head = candle_nn::linear(config.hidden, config.vocab, vb.pp("head"))?;
        let model = Self {
            config,
            vars: device_vars,
            embeddings,
            positions,
            blocks,
            norm,
            head,
        };
        let actual: usize = model.named_vars().values().map(|v| v.elem_count()).sum();
        if actual != model.config.parameter_count()? {
            candle_core::bail!("parameter mismatch");
        }
        Ok(model)
    }
    pub fn named_vars(&self) -> BTreeMap<String, Var> {
        self.vars
            .data()
            .lock()
            .unwrap()
            .iter()
            .map(|(n, v)| (n.clone(), v.clone()))
            .collect()
    }
    pub fn optimizer_vars(&self) -> Vec<Var> {
        self.named_vars().into_values().collect()
    }
    pub fn forward(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        let (batch, time) = input.dims2()?;
        if batch == 0 || time == 0 || time > self.config.context {
            candle_core::bail!("invalid input shape");
        }
        let pos = Tensor::arange(0_u32, time as u32, input.device())?;
        let mut x = self
            .embeddings
            .forward(input)?
            .broadcast_add(&self.positions.forward(&pos)?)?;
        let values: Vec<f32> = (0..time)
            .flat_map(|i| (0..time).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
            .collect();
        let mask = Tensor::from_vec(values, (1, 1, time, time), input.device())?;
        for block in &self.blocks {
            x = block.forward(&x, &mask, self.config.heads)?;
        }
        self.head.forward(&self.norm.forward(&x)?)
    }
    pub fn load(&mut self, path: &std::path::Path) -> candle_core::Result<()> {
        // Safe buffered loading also verifies exact tensor set, dimensions and finite FP32.
        let data = candle_core::safetensors::load(path, &Device::Cpu)?;
        let vars = self.named_vars();
        if data.len() != vars.len() {
            candle_core::bail!("checkpoint tensor count");
        }
        for (name, var) in &vars {
            let t = data
                .get(name)
                .ok_or_else(|| candle_core::Error::Msg(format!("missing {name}")))?;
            if t.shape() != var.shape() || t.dtype() != DType::F32 {
                candle_core::bail!("checkpoint tensor layout");
            }
            if t.flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .any(|x| !x.is_finite())
            {
                candle_core::bail!("nonfinite checkpoint");
            }
        }
        for (name, var) in vars {
            var.set(&data[&name].to_device(var.device())?)?;
        }
        Ok(())
    }
}
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0x9e3779b97f4a7c15 } else { seed })
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1_u64 << 53) as f64
    }
}
pub fn loss(logits: &Tensor, targets: &Tensor) -> candle_core::Result<Tensor> {
    let (b, t, v) = logits.dims3()?;
    if targets.dims() != [b, t] || b * t == 0 {
        candle_core::bail!("invalid target shape");
    }
    candle_nn::loss::cross_entropy(&logits.reshape((b * t, v))?, &targets.flatten_all()?)
}
pub fn finite_gradients(
    model: &Transformer,
    grads: &candle_core::backprop::GradStore,
) -> candle_core::Result<BTreeMap<String, f64>> {
    let mut groups = BTreeMap::new();
    for (name, var) in model.named_vars() {
        let grad = grads
            .get(&var)
            .ok_or_else(|| candle_core::Error::Msg(format!("missing gradient: {name}")))?;
        let squared = grad.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
        if !squared.is_finite() {
            candle_core::bail!("nonfinite gradient: {name}");
        }
        if name.ends_with(".weight") && squared <= 0.0 {
            candle_core::bail!("zero weight gradient: {name}");
        }
        let group = if name.starts_with("embedding") {
            "embedding"
        } else if name.starts_with("position") {
            "position"
        } else if name.contains("norm") {
            "rmsnorm"
        } else if name.starts_with("head") {
            "head"
        } else if name.contains(".up.") || name.contains(".down.") {
            "ffn"
        } else {
            "attention"
        };
        *groups.entry(group.to_string()).or_default() += squared;
    }
    if groups.values().any(|&g| g <= 0.0) {
        candle_core::bail!("zero gradient family: {groups:?}");
    }
    Ok(groups)
}
#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::Optimizer;
    fn tiny() -> ModelConfig {
        ModelConfig {
            layers: 1,
            hidden: 8,
            heads: 2,
            ffn: 16,
            context: 8,
            vocab: 512,
        }
    }
    #[test]
    fn primary_parameters_and_layout() -> candle_core::Result<()> {
        assert_eq!(ModelConfig::default().parameter_count()?, 14_681_984);
        let c = tiny();
        let m = Transformer::new(c.clone(), 19, &Device::Cpu)?;
        assert_eq!(
            m.optimizer_vars()
                .iter()
                .map(|v| v.elem_count())
                .sum::<usize>(),
            c.parameter_count()?
        );
        assert!(
            ModelConfig {
                hidden: usize::MAX,
                ..tiny()
            }
            .validate()
            .is_err()
        );
        Ok(())
    }
    #[test]
    fn causal_mask_and_initialization() -> candle_core::Result<()> {
        let a = Transformer::new(tiny(), 19, &Device::Cpu)?;
        let b = Transformer::new(tiny(), 19, &Device::Cpu)?;
        let x = Tensor::new(&[[1_u32, 2, 3, 4]], &Device::Cpu)?;
        let y = Tensor::new(&[[1_u32, 2, 99, 98]], &Device::Cpu)?;
        let ax = a.forward(&x)?.to_vec3::<f32>()?;
        let bx = b.forward(&x)?.to_vec3::<f32>()?;
        let ay = a.forward(&y)?.to_vec3::<f32>()?;
        assert_eq!(ax, bx);
        assert_eq!(&ax[0][..2], &ay[0][..2]);
        assert_ne!(ax[0][2], ay[0][2]);
        Ok(())
    }
    #[test]
    fn all_parameter_families_get_gradients_and_adam_updates() -> candle_core::Result<()> {
        let m = Transformer::new(tiny(), 19, &Device::Cpu)?;
        let x = Tensor::new(&[[1_u32, 2, 3, 4]], &Device::Cpu)?;
        let y = Tensor::new(&[[2_u32, 3, 4, 1]], &Device::Cpu)?;
        let before = m.named_vars()["embedding.weight"]
            .flatten_all()?
            .to_vec1::<f32>()?;
        let l = loss(&m.forward(&x)?, &y)?;
        assert!(l.to_scalar::<f32>()?.is_finite());
        let g = l.backward()?;
        finite_gradients(&m, &g)?;
        let mut opt = candle_nn::AdamW::new(m.optimizer_vars(), candle_nn::ParamsAdamW::default())?;
        opt.step(&g)?;
        assert_ne!(
            before,
            m.named_vars()["embedding.weight"]
                .flatten_all()?
                .to_vec1::<f32>()?
        );
        Ok(())
    }
    #[test]
    fn gradient_gate_rejects_one_missing_projection_even_when_family_is_active()
    -> candle_core::Result<()> {
        let m = Transformer::new(tiny(), 19, &Device::Cpu)?;
        let up = m.named_vars()["block.0.up.weight"].clone();
        up.set(&Tensor::zeros(up.shape(), DType::F32, &Device::Cpu)?)?;
        let x = Tensor::new(&[[1_u32, 2, 3, 4]], &Device::Cpu)?;
        let y = Tensor::new(&[[2_u32, 3, 4, 1]], &Device::Cpu)?;
        let gradients = loss(&m.forward(&x)?, &y)?.backward()?;
        let down = m.named_vars()["block.0.down.weight"].clone();
        assert_eq!(
            gradients
                .get(&down)
                .unwrap()
                .sqr()?
                .sum_all()?
                .to_scalar::<f32>()?,
            0.0
        );
        let bias = m.named_vars()["block.0.down.bias"].clone();
        assert!(
            gradients
                .get(&bias)
                .unwrap()
                .sqr()?
                .sum_all()?
                .to_scalar::<f32>()?
                > 0.0
        );
        assert!(finite_gradients(&m, &gradients).is_err());
        Ok(())
    }
    #[test]
    fn gradients_match_finite_differences() -> candle_core::Result<()> {
        let m = Transformer::new(tiny(), 19, &Device::Cpu)?;
        let x = Tensor::new(&[[1_u32, 2, 3, 4]], &Device::Cpu)?;
        let y = Tensor::new(&[[2_u32, 3, 4, 1]], &Device::Cpu)?;
        let gradients = loss(&m.forward(&x)?, &y)?.backward()?;
        for name in [
            "embedding.weight",
            "position.weight",
            "block.0.norm1.weight",
            "block.0.q.weight",
            "block.0.k.weight",
            "block.0.v.weight",
            "block.0.out.weight",
            "block.0.up.weight",
            "block.0.down.weight",
            "norm.weight",
            "head.weight",
        ] {
            let vars = m.named_vars();
            let var = vars
                .get(name)
                .ok_or_else(|| candle_core::Error::Msg(format!("missing fixture {name}")))?;
            let grad = gradients
                .get(var)
                .ok_or_else(|| candle_core::Error::Msg("missing gradient".into()))?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let (index, &analytic) = grad
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
                .unwrap();
            let original = var.flatten_all()?.to_vec1::<f32>()?;
            let mut values = original.clone();
            let epsilon = 0.001_f32;
            values[index] += epsilon;
            var.set(&Tensor::from_vec(values, var.shape(), &Device::Cpu)?)?;
            let plus = loss(&m.forward(&x)?, &y)?.to_scalar::<f32>()?;
            let mut values = original.clone();
            values[index] -= epsilon;
            var.set(&Tensor::from_vec(values, var.shape(), &Device::Cpu)?)?;
            let minus = loss(&m.forward(&x)?, &y)?.to_scalar::<f32>()?;
            var.set(&Tensor::from_vec(original, var.shape(), &Device::Cpu)?)?;
            let numerical = (plus - minus) / (2.0 * epsilon);
            assert!(
                (analytic - numerical).abs() <= 0.001 + 0.03 * analytic.abs(),
                "{name}[{index}]: gradient={analytic} numerical={numerical}"
            );
        }
        Ok(())
    }
    #[test]
    fn checkpoint_roundtrip_and_tiny_overfit() -> candle_core::Result<()> {
        let mut m = Transformer::new(tiny(), 19, &Device::Cpu)?;
        let x = Tensor::new(&[[1_u32, 2, 3, 4]], &Device::Cpu)?;
        let y = Tensor::new(&[[2_u32, 3, 4, 1]], &Device::Cpu)?;
        let initial = loss(&m.forward(&x)?, &y)?.to_scalar::<f32>()?;
        let mut opt = candle_nn::AdamW::new(
            m.optimizer_vars(),
            candle_nn::ParamsAdamW {
                lr: 0.01,
                weight_decay: 0.0,
                ..Default::default()
            },
        )?;
        for _ in 0..100 {
            opt.backward_step(&loss(&m.forward(&x)?, &y)?)?;
        }
        let final_loss = loss(&m.forward(&x)?, &y)?.to_scalar::<f32>()?;
        assert!(
            final_loss < 0.25 && final_loss < initial * 0.1,
            "{initial} -> {final_loss}"
        );
        let path = std::env::temp_dir().join(format!(
            "packtok-m5-checkpoint-{}.safetensors",
            std::process::id()
        ));
        m.vars.save(&path)?;
        let expected = m.forward(&x)?.to_vec3::<f32>()?;
        let fresh = Transformer::new(tiny(), 100, &Device::Cpu)?;
        m = fresh;
        m.load(&path)?;
        assert_eq!(expected, m.forward(&x)?.to_vec3::<f32>()?);
        std::fs::remove_file(path).map_err(candle_core::Error::wrap)?;
        Ok(())
    }
}
