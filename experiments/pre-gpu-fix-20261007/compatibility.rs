//! Fixed read-only compatibility probe, run before and after validation fixes.
use packtok_core::TokenId;
use packtok_format::Artifact;
use packtok_model::{CausalLm, ModelConfig, PackVocabulary};
use std::{error::Error, fs, path::Path};

fn main() -> Result<(), Box<dyn Error>> {
    let config = ModelConfig {
        hidden_size: 2,
        context_length: 2,
    };
    for (name, model, pack) in [
        ("flat", CausalLm::new_flat(config, 256, 19)?, 0),
        (
            "factorized",
            CausalLm::new_factorized(
                config,
                vec![PackVocabulary {
                    pack_id: 7,
                    token_count: 256,
                }],
                19,
            )?,
            7,
        ),
    ] {
        let prompt = [
            TokenId::new(pack, 97),
            TokenId::new(pack, 98),
            TokenId::new(pack, 99),
        ];
        println!(
            "golden {name} seed=19 hidden=2 context=2 prompt={prompt:?} generated={:?}",
            model.greedy_generate(&prompt, 8)?
        );
    }
    let mut models = 0;
    let mut tokenizers = 0;
    for directory in [
        "experiments/m3-model/runs/final-20261007",
        "experiments/m3-model/runs/audit-20261007",
        "experiments/m4-ablation/runs/tiny-final-20261007-v2",
        "experiments/m4-ablation/runs/large-final-20261007",
        "experiments/m3-model/artifacts",
    ] {
        let mut paths = fs::read_dir(directory)?
            .map(|e| e.map(|e| e.path()))
            .collect::<Result<Vec<_>, _>>()?;
        paths.sort();
        for path in paths {
            let bytes = fs::read(&path)?;
            match path.extension().and_then(|s| s.to_str()) {
                Some("ptlm") => {
                    let model = CausalLm::from_bytes(&bytes)?;
                    assert_eq!(
                        model.to_bytes()?,
                        bytes,
                        "model wire changed: {}",
                        path.display()
                    );
                    let pack = model.pack_vocabularies().map_or(0, |p| p[0].pack_id);
                    let prompt = vec![TokenId::new(pack, 0); model.config().context_length + 1];
                    println!(
                        "model {} bytes={} identity=true generation={:?}",
                        path.display(),
                        bytes.len(),
                        model.greedy_generate(&prompt, 8)?
                    );
                    models += 1;
                }
                Some("packtok") => {
                    assert_eq!(
                        Artifact::from_bytes(&bytes)?.to_bytes()?,
                        bytes,
                        "tokenizer wire changed: {}",
                        path.display()
                    );
                    println!(
                        "tokenizer {} bytes={} identity=true",
                        path.display(),
                        bytes.len()
                    );
                    tokenizers += 1;
                }
                _ => {}
            }
        }
    }
    // Exact historical M4 A/D versus M3 model identity checks used at delivery.
    for regime in ["schedule", "mac"] {
        for (variant, tokenizer) in [("A", "m1-flat"), ("D", "m2-factorized")] {
            for seed in [20_261_007, 20_261_008, 20_261_009] {
                let m3_regime = if regime == "schedule" {
                    "a-same-backbone"
                } else {
                    "b-matched-mac"
                };
                let old = Path::new("experiments/m3-model/runs/final-20261007")
                    .join(format!("{m3_regime}-{tokenizer}-seed-{seed}.ptlm"));
                let m4 = Path::new("experiments/m4-ablation/runs/tiny-final-20261007-v2")
                    .join(format!("{regime}-{variant}-seed-{seed}.ptlm"));
                assert_eq!(
                    fs::read(&old)?,
                    fs::read(&m4)?,
                    "historical identity differs"
                );
                println!("historical {regime} {variant} {seed} identical=true");
            }
        }
    }
    println!("complete models={models} tokenizers={tokenizers} historical_pairs=12");
    Ok(())
}
