//! CPU-only weight recovery example. The frozen Transformer is unchanged.
use candle_core::{Device, Tensor};
use packtok_m5::{
    Result,
    model::{ModelConfig, Transformer},
};
use std::{env, path::Path};

fn run() -> Result<()> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: recover_weights ORIGINAL RECOVERED".into());
    }
    let mut original = Transformer::new(ModelConfig::default(), 20261009, &Device::Cpu)?;
    original.load(Path::new(&args[0]))?;
    let mut recovered = Transformer::new(ModelConfig::default(), 20261010, &Device::Cpu)?;
    recovered.load(Path::new(&args[1]))?;
    let input = Tensor::new(&[[1_u32, 2, 3, 4, 1, 2, 3, 4]], &Device::Cpu)?;
    let a = original.forward(&input)?.flatten_all()?.to_vec1::<f32>()?;
    let b = recovered.forward(&input)?.flatten_all()?.to_vec1::<f32>()?;
    let mut maximum = 0_f64;
    for (&x, &y) in a.iter().zip(&b) {
        let delta = f64::from((x - y).abs());
        if !x.is_finite() || !y.is_finite() || delta > 1e-5 + 0.001 * f64::from(x.abs()) {
            return Err("recovered weight reference mismatch".into());
        }
        maximum = maximum.max(delta);
    }
    println!(
        "{}",
        serde_json::json!({"status":"PASS", "backend":"CPU", "parameters":original.config.parameter_count()?, "reference_input":[1,2,3,4,1,2,3,4], "logit_count":a.len(), "maximum_absolute_delta":maximum, "absolute_tolerance":1e-5, "relative_tolerance":0.001, "exact_training_resume":false})
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("Weight recovery failed: {error}");
        std::process::exit(1);
    }
}
