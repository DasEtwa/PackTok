use packtok_m5::{Result, corpus, data, extended, overfit, runner};
use std::path::Path;
fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let root = Path::new("experiments/m5-gpu");
    match args.first().map(String::as_str) {
        Some("reference") => runner::reference(&root.join("provenance/cpu-reference.json")),
        Some("plan") => {
            let config = runner::TrainConfig::default();
            packtok_m5::write_new(
                &root.join("configs/primary.json"),
                &serde_json::to_vec_pretty(&config)?,
            )?;
            runner::plan(
                &config,
                &root.join("data/prepared-v2"),
                &root.join("provenance/schedule-plan.json"),
            )
        }
        Some("preflight") if args.len() == 4 => runner::preflight(
            Path::new(&args[1]),
            Path::new(&args[2]),
            Path::new(&args[3]),
        ),
        Some("plan-extended") if args.len() == 4 => extended::plan(
            Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3])),
        Some("initialization-hashes") if args.len() >= 4 => {
            let seeds = args[3..].iter().map(|v| v.parse()).collect::<std::result::Result<Vec<u64>, _>>()?;
            let report = extended::initialization_hashes(Path::new(&args[1]), Path::new(&args[2]), &seeds)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
        Some("train-extended") if args.len() == 8 => extended::train(
            Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),&args[4],&args[5],
            args[6].parse()?,&args[7]),
        Some("diagnose-overfit") if (args.len() == 3 || args.len() == 4) => {
            let comparison_lr = args.get(3).map(|v| v.parse()).transpose()?;
            let outcomes = overfit::diagnose_cpu(
                Path::new(&args[1]),
                Path::new(&args[2]),
                comparison_lr,
            )?;
            for outcome in outcomes {
                println!("{}", serde_json::to_string(&outcome)?);
            }
            Ok(())
        },
        Some("train") if args.len() == 7 => {
            let config = serde_json::from_slice(&std::fs::read(&args[1])?)?;
            runner::train(
                &config,
                Path::new(&args[2]),
                Path::new(&args[3]),
                &args[4],
                &args[5],
                args[6].parse()?,
            )
        }
        Some("corpus") => corpus::build(root),
        Some("prepare") => data::prepare(root),
        Some("verify-prepared") => data::verify_prepared(root),
        Some("cuda-components") if args.len()==2 => {
            let json: serde_json::Value = serde_json::from_slice(&std::fs::read(&args[1])?)?;
            for name in [
                "cuda_nvcc",
                "cuda_cudart",
                "cuda_cccl",
                "cuda_nvrtc",
                "libcublas",
                "libcurand",
            ] {
                let c = &json[name]["linux-x86_64"];
                println!(
                    "{name}\t{}\t{}\t{}",
                    c["relative_path"].as_str().ok_or("CUDA path")?,
                    c["sha256"].as_str().ok_or("CUDA hash")?,
                    c["size"].as_str().ok_or("CUDA size")?
                );
            }
            Ok(())
        }
        _ => Err("usage: packtok-m5 corpus | prepare | verify-prepared | reference | plan | cuda-components JSON | preflight DATA REFERENCE OUT | diagnose-overfit DATA OUT [COMPARISON_LR] | plan-extended CONFIG DATA OUT | initialization-hashes CONFIG DATA SEED... | train-extended CONFIG DATA OUT VARIANT REGIME SEED APPROVAL_REFERENCE".into()),
    }
}
fn main() {
    if let Err(e) = run() {
        eprintln!("M5 failed: {e}");
        std::process::exit(1);
    }
}
