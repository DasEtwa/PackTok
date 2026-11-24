use packtok_m5::Result;
use std::path::Path;

#[path = "m7/archive.rs"]
mod archive;

#[path = "m7/corpus.rs"]
mod corpus;
#[path = "m7/dedup.rs"]
mod dedup;
#[path = "m7/source.rs"]
mod source;
#[cfg(test)]
#[path = "m7/tests.rs"]
mod tests;
#[path = "m7/tokens.rs"]
mod tokens;
#[path = "m7/train.rs"]
mod train;

fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        ["import-archives", plan, output] => archive::import(Path::new(plan),Path::new(output)),
        ["fetch", catalog, cache, id, budget] =>
            source::fetch(Path::new(catalog), Path::new(cache), id, budget.parse()?),
        ["build", catalog, cache, output] =>
            corpus::build(Path::new(catalog), Path::new(cache), Path::new(output)),
        ["verify", dataset] => { corpus::verify(Path::new(dataset))?; Ok(()) }
        ["tokenize", dataset, output, sample_bytes] =>
            tokens::prepare(Path::new(dataset), Path::new(output), sample_bytes.parse()?),
        ["verify-tokens", dataset, tokenized] =>
            tokens::verify(Path::new(dataset), Path::new(tokenized)),
        ["train-cpu", dataset, tokenized, run, variant, stop] =>
            train::run(Path::new(dataset), Path::new(tokenized), Path::new(run), variant, stop.parse()?, &candle_core::Device::Cpu),
        ["train-l4", dataset, tokenized, run, variant, stop, approval] => {
            if approval.is_empty() || std::env::var("PACKTOK_M7_GPU_APPROVAL").as_deref()!=Ok(*approval) {
                return Err("matching approved M7 L4 run reference is required".into());
            }
            if std::env::var("NVIDIA_TF32_OVERRIDE").as_deref() != Ok("0") {
                return Err("FP32 probe requires NVIDIA_TF32_OVERRIDE=0 before CUDA initialization".into());
            }
            packtok_m5::runner::require_l4()?;
            train::run(Path::new(dataset), Path::new(tokenized), Path::new(run), variant, stop.parse()?, &packtok_m5::runner::cuda()?)
        }
        _ => Err("usage: packtok-m7 import-archives PLAN NEW_IMPORT | fetch CATALOG CACHE SOURCE_ID MAX_BYTES | build CATALOG CACHE NEW_DATASET | verify DATASET | tokenize DATASET NEW_TOKENIZED TRAIN_SAMPLE_BYTES | verify-tokens DATASET TOKENIZED | train-cpu DATASET TOKENIZED RUN A_OR_C STOP_STEP | train-l4 DATASET TOKENIZED RUN A_OR_C STOP_STEP APPROVAL".into())
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("M7: {error}");
        std::process::exit(1);
    }
}
