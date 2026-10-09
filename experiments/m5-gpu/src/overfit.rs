use crate::{
    Result,
    data::Sequence,
    model::{ModelConfig, Rng, Transformer, loss},
    runner,
};
use candle_core::{Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use serde::Serialize;
use std::{fs, io::Write, path::Path, time::Instant};

pub const UPDATES: usize = 500;
pub const LOSS_STEPS: [usize; 9] = [0, 1, 10, 25, 50, 100, 200, 350, UPDATES];
pub const FIXED_INPUT: [u32; 8] = [1, 2, 3, 4, 1, 2, 3, 4];
pub const FIXED_TARGET: [u32; 8] = [2, 3, 4, 1, 2, 3, 4, 1];
pub const PRIMARY_SEED: u64 = 20261008;
pub const SAMPLER_XOR: u64 = 0xa341316c9e3779b9;
pub const PRIMARY_UPDATES: usize = 18;
const DIVERGENCE_FACTOR: f32 = 10.0;
const DIVERGENCE_ADDEND: f32 = 10.0;
const STAGNATION_WINDOW: usize = 100;
const STAGNATION_MIN_DELTA: f32 = 1e-4;

#[derive(Clone, Debug, Serialize)]
pub struct Sample {
    pub state: String,
    pub learning_rate: f64,
    pub step: usize,
    pub loss: Option<f32>,
    pub gradients_finite: Option<bool>,
    pub parameters_finite: bool,
    pub stagnated: bool,
    pub event: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Outcome {
    pub state: String,
    pub learning_rate: f64,
    pub starting_weights_hash: String,
    pub initial_loss: f32,
    pub final_loss: Option<f32>,
    pub completed_updates: usize,
    pub seconds: f64,
    pub passed_original_threshold: bool,
    pub divergence_step: Option<usize>,
    pub nonfinite_step: Option<usize>,
    pub stagnation_step: Option<usize>,
    pub samples: Vec<Sample>,
    pub first_update_gradient_l2: Option<f64>,
    pub first_update_selected_gradient_l2: std::collections::BTreeMap<String, f64>,
    pub first_update_selected_weight_delta_l2: std::collections::BTreeMap<String, f64>,
}

const SELECTED_TENSORS: [&str; 4] = [
    "embedding.weight",
    "block.0.q.weight",
    "block.0.up.weight",
    "head.weight",
];

fn selected_gradient_diagnostics(
    model: &Transformer,
    grads: &candle_core::backprop::GradStore,
) -> Result<(f64, std::collections::BTreeMap<String, f64>)> {
    let mut total = 0.0_f64;
    let mut selected = std::collections::BTreeMap::new();
    for (name, var) in model.named_vars() {
        let grad = grads
            .get(&var)
            .ok_or("missing gradient during diagnostics")?;
        let squared = f64::from(grad.sqr()?.sum_all()?.to_scalar::<f32>()?);
        if !squared.is_finite() {
            return Err("nonfinite gradient norm during diagnostics".into());
        }
        total += squared;
        if SELECTED_TENSORS.contains(&name.as_str()) {
            selected.insert(name, squared.sqrt());
        }
    }
    if selected.len() != SELECTED_TENSORS.len() || !total.is_finite() {
        return Err("selected gradient diagnostic tensor missing".into());
    }
    Ok((total.sqrt(), selected))
}

fn fixed_batch(device: &Device) -> Result<(Tensor, Tensor)> {
    Ok((
        Tensor::new(&[FIXED_INPUT], device)?,
        Tensor::new(&[FIXED_TARGET], device)?,
    ))
}

pub(crate) fn fixed_batch_for_checkpoint(device: &Device) -> Result<(Tensor, Tensor)> {
    fixed_batch(device)
}

fn finite_norms(model: &Transformer, grads: &candle_core::backprop::GradStore) -> Result<bool> {
    for var in model.named_vars().values() {
        let Some(grad) = grads.get(var) else {
            return Ok(false);
        };
        let squared = grad.sqr()?.sum_all()?.to_scalar::<f32>()?;
        if !squared.is_finite() {
            return Ok(false);
        }
    }
    Ok(true)
}

fn parameters_finite(model: &Transformer) -> Result<bool> {
    for var in model.named_vars().values() {
        let squared = var.as_tensor().sqr()?.sum_all()?.to_scalar::<f32>()?;
        if !squared.is_finite() {
            return Ok(false);
        }
    }
    Ok(true)
}

fn divergence_limit(initial: f32) -> f32 {
    (initial * DIVERGENCE_FACTOR).max(initial + DIVERGENCE_ADDEND)
}

pub fn run(
    model: &Transformer,
    device: &Device,
    state: &str,
    learning_rate: f64,
) -> Result<Outcome> {
    if !(state == "fresh" || state == "post-18" || state.ends_with("-lr-comparison"))
        || !learning_rate.is_finite()
        || learning_rate <= 0.0
    {
        return Err("invalid overfit diagnostic state or learning rate".into());
    }
    if model.config.context < FIXED_INPUT.len() {
        return Err("diagnostic fixture exceeds model context".into());
    }
    let (x, y) = fixed_batch(device)?;
    let starting_weights_hash = runner::init_hash(model)?;
    let initial = loss(&model.forward(&x)?, &y)?.to_scalar::<f32>()?;
    let mut samples = vec![Sample {
        state: state.into(),
        learning_rate,
        step: 0,
        loss: Some(initial),
        gradients_finite: None,
        parameters_finite: parameters_finite(model)?,
        stagnated: false,
        event: (!initial.is_finite()).then(|| "nonfinite-loss".into()),
    }];
    if !initial.is_finite() || !samples[0].parameters_finite {
        return Ok(Outcome {
            state: state.into(),
            learning_rate,
            starting_weights_hash,
            initial_loss: initial,
            final_loss: Some(initial),
            completed_updates: 0,
            seconds: 0.0,
            passed_original_threshold: false,
            divergence_step: None,
            nonfinite_step: Some(0),
            stagnation_step: None,
            samples,
            first_update_gradient_l2: None,
            first_update_selected_gradient_l2: std::collections::BTreeMap::new(),
            first_update_selected_weight_delta_l2: std::collections::BTreeMap::new(),
        });
    }

    let mut optimizer = AdamW::new(
        model.optimizer_vars(),
        ParamsAdamW {
            lr: learning_rate,
            weight_decay: 0.0,
            ..runner::adam()
        },
    )?;
    let mut best_loss = initial;
    let mut last_improvement = 0;
    let mut stagnation_step = None;
    let mut divergence_step = None;
    let mut nonfinite_step = None;
    let mut first_update_gradient_l2 = None;
    let mut first_update_selected_gradient_l2 = std::collections::BTreeMap::new();
    let mut first_update_selected_weight_delta_l2 = std::collections::BTreeMap::new();
    let start = Instant::now();
    let mut completed = 0;
    let mut final_loss = Some(initial);

    for update in 1..=UPDATES {
        let step_loss = loss(&model.forward(&x)?, &y)?;
        let value = step_loss.to_scalar::<f32>()?;
        let observed_step = update - 1;
        if !value.is_finite() {
            nonfinite_step = Some(observed_step);
            samples.push(Sample {
                state: state.into(),
                learning_rate,
                step: observed_step,
                loss: None,
                gradients_finite: None,
                parameters_finite: false,
                stagnated: stagnation_step.is_some(),
                event: Some("nonfinite-loss".into()),
            });
            final_loss = None;
            break;
        }
        if value > divergence_limit(initial) {
            divergence_step = Some(observed_step);
            samples.push(Sample {
                state: state.into(),
                learning_rate,
                step: observed_step,
                loss: Some(value),
                gradients_finite: None,
                parameters_finite: parameters_finite(model)?,
                stagnated: stagnation_step.is_some(),
                event: Some("diverged-loss-threshold".into()),
            });
            final_loss = Some(value);
            break;
        }
        if best_loss - value >= STAGNATION_MIN_DELTA {
            best_loss = value;
            last_improvement = observed_step;
        } else if stagnation_step.is_none()
            && observed_step.saturating_sub(last_improvement) >= STAGNATION_WINDOW
            && value >= 0.25
        {
            stagnation_step = Some(observed_step);
        }

        let gradients = step_loss.backward()?;
        let gradients_finite = finite_norms(model, &gradients)?;
        if !gradients_finite {
            nonfinite_step = Some(observed_step);
            samples.push(Sample {
                state: state.into(),
                learning_rate,
                step: observed_step,
                loss: Some(value),
                gradients_finite: Some(false),
                parameters_finite: parameters_finite(model)?,
                stagnated: stagnation_step.is_some(),
                event: Some("nonfinite-gradient".into()),
            });
            final_loss = Some(value);
            break;
        }
        let selected_before = if update == 1 {
            let (total, selected) = selected_gradient_diagnostics(model, &gradients)?;
            first_update_gradient_l2 = Some(total);
            first_update_selected_gradient_l2 = selected;
            let vars = model.named_vars();
            SELECTED_TENSORS
                .iter()
                .map(|name| {
                    Ok((
                        (*name).to_owned(),
                        vars.get(*name)
                            .ok_or("selected parameter missing")?
                            .as_tensor()
                            .copy()?,
                    ))
                })
                .collect::<Result<std::collections::BTreeMap<_, _>>>()?
        } else {
            std::collections::BTreeMap::new()
        };
        optimizer.step(&gradients)?;
        device.synchronize()?;
        if update == 1 {
            let vars = model.named_vars();
            for (name, before) in selected_before {
                let after = vars
                    .get(&name)
                    .ok_or("selected parameter missing after update")?;
                let delta = (after.as_tensor() - &before)?;
                let squared = f64::from(delta.sqr()?.sum_all()?.to_scalar::<f32>()?);
                if !squared.is_finite() || squared == 0.0 {
                    return Err("selected first AdamW update was zero or nonfinite".into());
                }
                first_update_selected_weight_delta_l2.insert(name, squared.sqrt());
            }
        }
        completed = update;

        if LOSS_STEPS.contains(&completed) {
            let measured = loss(&model.forward(&x)?, &y)?.to_scalar::<f32>()?;
            let parameters_are_finite = parameters_finite(model)?;
            let mut event = None;
            if !measured.is_finite() || !parameters_are_finite {
                nonfinite_step = Some(completed);
                event = Some(if !measured.is_finite() {
                    "nonfinite-loss".into()
                } else {
                    "nonfinite-parameters".into()
                });
            } else if measured > divergence_limit(initial) {
                divergence_step = Some(completed);
                event = Some("diverged-loss-threshold".into());
            }
            samples.push(Sample {
                state: state.into(),
                learning_rate,
                step: completed,
                loss: measured.is_finite().then_some(measured),
                gradients_finite: Some(true),
                parameters_finite: parameters_are_finite,
                stagnated: stagnation_step.is_some(),
                event,
            });
            final_loss = measured.is_finite().then_some(measured);
            if nonfinite_step.is_some() || divergence_step.is_some() {
                break;
            }
        }
    }

    let final_value = match final_loss {
        Some(v) if completed == UPDATES => v,
        Some(_) => loss(&model.forward(&x)?, &y)?.to_scalar::<f32>()?,
        None => f32::NAN,
    };
    let threshold_pass = final_value.is_finite()
        && final_value < 0.25
        && final_value < initial * 0.1
        && divergence_step.is_none()
        && nonfinite_step.is_none();
    if completed == UPDATES && samples.last().map(|s| s.step) != Some(UPDATES) {
        samples.push(Sample {
            state: state.into(),
            learning_rate,
            step: UPDATES,
            loss: final_value.is_finite().then_some(final_value),
            gradients_finite: Some(true),
            parameters_finite: parameters_finite(model)?,
            stagnated: stagnation_step.is_some(),
            event: None,
        });
    }
    Ok(Outcome {
        state: state.into(),
        learning_rate,
        starting_weights_hash,
        initial_loss: initial,
        final_loss: final_value.is_finite().then_some(final_value),
        completed_updates: completed,
        seconds: start.elapsed().as_secs_f64(),
        passed_original_threshold: threshold_pass,
        divergence_step,
        nonfinite_step,
        stagnation_step,
        samples,
        first_update_gradient_l2,
        first_update_selected_gradient_l2,
        first_update_selected_weight_delta_l2,
    })
}

fn post_primary_state(train: &Sequence, device: &Device) -> Result<(Transformer, String)> {
    let model = Transformer::new(ModelConfig::default(), PRIMARY_SEED, device)?;
    let initial_hash = runner::init_hash(&model)?;
    let mut optimizer = AdamW::new(model.optimizer_vars(), runner::adam())?;
    let mut sampler = Rng::new(PRIMARY_SEED ^ SAMPLER_XOR);
    for _ in 0..PRIMARY_UPDATES {
        let starts: Vec<_> = (0..8)
            .map(|_| (sampler.next_u64() as usize) % (train.tokens.len() - 256))
            .collect();
        let (x, y, _) = runner::batch(train, &starts, 256, device)?;
        optimizer.backward_step(&loss(&model.forward(&x)?, &y)?)?;
        device.synchronize()?;
    }
    Ok((model, initial_hash))
}

pub fn diagnose_cpu(
    data: &Path,
    output: &Path,
    compare_learning_rate: Option<f64>,
) -> Result<Vec<Outcome>> {
    let train = Sequence::load(&data.join("A-train.seq"))?;
    if train.tokens.len() <= 256 {
        return Err("post-primary diagnostic requires the frozen A TRAIN sequence".into());
    }
    let mut log = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    let device = Device::Cpu;
    let mut outcomes = Vec::new();
    for state in ["fresh", "post-18"] {
        let model = if state == "fresh" {
            Transformer::new(ModelConfig::default(), PRIMARY_SEED, &device)?
        } else {
            post_primary_state(&train, &device)?.0
        };
        let outcome = run(&model, &device, state, 0.005)?;
        for row in &outcome.samples {
            writeln!(log, "{}", serde_json::to_string(row)?)?;
            println!("{}", serde_json::to_string(row)?);
        }
        writeln!(log, "{}", serde_json::to_string(&outcome)?)?;
        let needs_lr_comparison =
            !outcome.passed_original_threshold && compare_learning_rate.is_some();
        outcomes.push(outcome);
        if needs_lr_comparison {
            let lr = compare_learning_rate.expect("checked above");
            let model = if state == "fresh" {
                Transformer::new(ModelConfig::default(), PRIMARY_SEED, &device)?
            } else {
                post_primary_state(&train, &device)?.0
            };
            let outcome = run(&model, &device, &format!("{state}-lr-comparison"), lr)?;
            for row in &outcome.samples {
                writeln!(log, "{}", serde_json::to_string(row)?)?;
                println!("{}", serde_json::to_string(row)?);
            }
            writeln!(log, "{}", serde_json::to_string(&outcome)?)?;
            outcomes.push(outcome);
        }
    }
    log.sync_all()?;
    Ok(outcomes)
}

pub fn write_outcome(out: &mut fs::File, outcome: &Outcome) -> Result<()> {
    for row in &outcome.samples {
        writeln!(out, "{}", serde_json::to_string(row)?)?;
        println!("{}", serde_json::to_string(row)?);
    }
    out.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::tiny_config;

    #[test]
    fn fixed_batch_diagnostic_reports_all_required_steps_and_learns() -> Result<()> {
        let model = Transformer::new(tiny_config(), 19, &Device::Cpu)?;
        let result = run(&model, &Device::Cpu, "fresh", 0.005)?;
        assert_eq!(
            result.samples.iter().map(|s| s.step).collect::<Vec<_>>(),
            LOSS_STEPS
        );
        assert!(result.final_loss.unwrap() < result.initial_loss);
        assert!(result.first_update_gradient_l2.unwrap().is_finite());
        assert_eq!(
            result.first_update_selected_gradient_l2.len(),
            SELECTED_TENSORS.len()
        );
        assert_eq!(
            result.first_update_selected_weight_delta_l2.len(),
            SELECTED_TENSORS.len()
        );
        assert!(
            result
                .first_update_selected_weight_delta_l2
                .values()
                .all(|norm| norm.is_finite() && *norm > 0.0)
        );
        assert!(result.samples.iter().all(|s| {
            s.loss.is_some_and(f32::is_finite)
                && s.gradients_finite.is_none_or(|v| v)
                && s.parameters_finite
        }));
        Ok(())
    }

    #[test]
    fn diagnostic_threshold_keeps_the_historical_value_and_detects_divergence() {
        assert_eq!(divergence_limit(7.5), 75.0);
        assert!(divergence_limit(7.5) < 75.1);
        assert!(20.0_f32 > divergence_limit(1.0));
        assert_eq!(STAGNATION_WINDOW, 100);
        assert_eq!(STAGNATION_MIN_DELTA, 1e-4);
        assert_eq!(LOSS_STEPS, [0, 1, 10, 25, 50, 100, 200, 350, 500]);
    }
}
