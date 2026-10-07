#![forbid(unsafe_code)]

use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use packtok_core::TokenId;
use packtok_format::{Artifact, FlatBpeModel};
use packtok_tokenizer::{BpeTokenizer, ByteFallbackTokenizer, Tokenizer};
use packtok_train::{BpeTrainingConfig, load_corpus, train_bpe_with_provenance};

const USAGE: &str = "Usage:
  packtok encode <text>
  packtok encode --artifact <artifact-path> <text>
  packtok decode <pack_id:local_id ...>
  packtok decode --artifact <artifact-path> <local_id ...>
  packtok init <artifact-path>
  packtok train-bpe <corpus-file-or-directory> <artifact-path> [--target-vocab N] [--max-merges N] [--min-frequency N]
  packtok validate <artifact-path>
  packtok inspect <artifact-path>
  packtok inspect-token <artifact-path> <local-id>
  packtok inspect-merges <artifact-path> [limit]";

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let command = args.next().ok_or_else(|| USAGE.to_owned())?;
    let arguments: Vec<String> = args.collect();

    match command.as_str() {
        "encode" => encode_command(&arguments),
        "decode" => decode_command(&arguments),
        "init" => {
            let path = one_path(&arguments, "artifact path")?;
            write_new_artifact(&path, &Artifact::byte_fallback_only())?;
            println!("created PackTok artifact at {}", path.display());
            Ok(())
        }
        "train-bpe" => train_command(&arguments),
        "validate" => {
            let artifact = read_artifact(&one_path(&arguments, "artifact path")?)?;
            println!(
                "valid PackTok artifact v{}: {} packs, {} special tokens, {} metadata entries",
                artifact.format_version(),
                artifact.registry().packs().len(),
                artifact.registry().special_tokens().len(),
                artifact.metadata().len()
            );
            Ok(())
        }
        "inspect" => inspect_command(&arguments),
        "inspect-token" => inspect_token_command(&arguments),
        "inspect-merges" => inspect_merges_command(&arguments),
        "help" | "--help" | "-h" => {
            println!("{USAGE}");
            Ok(())
        }
        _ => Err(USAGE.to_owned()),
    }
}

fn encode_command(arguments: &[String]) -> Result<(), String> {
    if arguments
        .first()
        .is_some_and(|argument| argument == "--artifact")
    {
        if arguments.len() < 3 {
            return Err(format!("missing BPE artifact or text\n{USAGE}"));
        }
        let artifact = read_artifact(Path::new(&arguments[1]))?;
        let tokenizer =
            BpeTokenizer::from_artifact(&artifact).map_err(|error| error.to_string())?;
        let input = arguments[2..].join(" ");
        let tokens = tokenizer
            .encode(&input)
            .map_err(|error| error.to_string())?;
        println!(
            "{}",
            tokens
                .iter()
                .map(|token| token.local.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        );
        return Ok(());
    }

    let input = arguments.join(" ");
    let tokenizer = ByteFallbackTokenizer::default();
    let tokens = tokenizer
        .encode(&input)
        .map_err(|error| error.to_string())?;
    println!(
        "{}",
        tokens
            .iter()
            .map(|token| format!("{}:{}", token.pack, token.local))
            .collect::<Vec<_>>()
            .join(" ")
    );
    Ok(())
}

fn decode_command(arguments: &[String]) -> Result<(), String> {
    if arguments
        .first()
        .is_some_and(|argument| argument == "--artifact")
    {
        if arguments.len() < 2 {
            return Err(format!("missing BPE artifact path\n{USAGE}"));
        }
        let artifact = read_artifact(Path::new(&arguments[1]))?;
        let tokenizer =
            BpeTokenizer::from_artifact(&artifact).map_err(|error| error.to_string())?;
        let ids = arguments[2..]
            .iter()
            .flat_map(|argument| argument.split_whitespace())
            .map(|value| {
                value
                    .parse::<u32>()
                    .map(|local| TokenId::new(0, local))
                    .map_err(|error| format!("invalid flat token ID {value:?}: {error}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let bytes = tokenizer
            .decode_bytes(&ids)
            .map_err(|error| error.to_string())?;
        print_decoded_bytes(bytes);
        return Ok(());
    }

    let specifications = arguments
        .iter()
        .flat_map(|argument| argument.split_whitespace())
        .collect::<Vec<_>>();
    let tokens = specifications
        .iter()
        .map(|specification| parse_token_id(specification))
        .collect::<Result<Vec<_>, _>>()?;
    let bytes = ByteFallbackTokenizer::default()
        .decode_bytes(&tokens)
        .map_err(|error| error.to_string())?;
    print_decoded_bytes(bytes);
    Ok(())
}

fn train_command(arguments: &[String]) -> Result<(), String> {
    if arguments.len() < 2 {
        return Err(format!("missing corpus or output artifact path\n{USAGE}"));
    }
    let corpus_path = Path::new(&arguments[0]);
    let output_path = Path::new(&arguments[1]);
    let mut config = BpeTrainingConfig::default();
    let mut index = 2;
    while index < arguments.len() {
        let option = arguments[index].as_str();
        let value = arguments
            .get(index + 1)
            .ok_or_else(|| format!("option {option} requires a value\n{USAGE}"))?;
        match option {
            "--target-vocab" => {
                config.target_vocab_size = value.parse().map_err(|error| {
                    format!("invalid target vocabulary size {value:?}: {error}")
                })?;
            }
            "--max-merges" => {
                config.max_merges = value
                    .parse()
                    .map_err(|error| format!("invalid maximum merge count {value:?}: {error}"))?;
            }
            "--min-frequency" => {
                config.min_pair_frequency = value.parse().map_err(|error| {
                    format!("invalid minimum pair frequency {value:?}: {error}")
                })?;
            }
            _ => return Err(format!("unknown training option {option:?}\n{USAGE}")),
        }
        index += 2;
    }
    config.validate().map_err(|error| error.to_string())?;
    let corpus = load_corpus(corpus_path).map_err(|error| error.to_string())?;
    let artifact = train_bpe_with_provenance(corpus.bytes(), config, corpus.provenance())
        .map_err(|error| error.to_string())?;
    write_new_artifact(output_path, &artifact)?;
    let model = artifact.flat_bpe().ok_or("trainer returned no BPE model")?;
    println!(
        "trained {} learned merges (vocabulary {}); corpus {} bytes across {} files; wrote {}",
        model.merges().len(),
        model.vocabulary_size(),
        corpus.provenance().total_bytes,
        corpus.provenance().files.len(),
        output_path.display()
    );
    Ok(())
}

fn inspect_command(arguments: &[String]) -> Result<(), String> {
    let artifact = read_artifact(&one_path(arguments, "artifact path")?)?;
    println!("PackTok artifact v{}", artifact.format_version());
    println!(
        "byte fallback pack: {}",
        artifact.registry().byte_fallback().pack_id()
    );
    for pack in artifact.registry().packs() {
        println!(
            "pack {}: {} ({} local IDs)",
            pack.id(),
            pack.name(),
            pack.local_token_count()
        );
    }
    for special in artifact.registry().special_tokens() {
        println!(
            "special token {}:{}: {}",
            special.id().pack,
            special.id().local,
            special.name()
        );
    }
    if let Some(model) = artifact.flat_bpe() {
        println!(
            "flat byte-level BPE vocabulary: {} tokens",
            model.vocabulary_size()
        );
        println!("learned merges: {}", model.merges().len());
    }
    for (key, value) in artifact.metadata() {
        println!("metadata {key}={value}");
    }
    Ok(())
}

fn inspect_token_command(arguments: &[String]) -> Result<(), String> {
    if arguments.len() != 2 {
        return Err(format!(
            "expected an artifact path and local token ID\n{USAGE}"
        ));
    }
    let artifact = read_artifact(Path::new(&arguments[0]))?;
    let model = artifact
        .flat_bpe()
        .ok_or("artifact has no flat BPE model")?;
    let id = arguments[1]
        .parse::<u32>()
        .map_err(|error| format!("invalid local token ID {:?}: {error}", arguments[1]))?;
    print_token(model, id)
}

fn print_token(model: &FlatBpeModel, id: u32) -> Result<(), String> {
    let tokenizer = BpeTokenizer::new(model.clone());
    let bytes = tokenizer
        .decode_bytes(&[TokenId::new(0, id)])
        .map_err(|error| error.to_string())?;
    println!("Token ID: {id}");
    println!("Bytes: {bytes:?}");
    println!("Escaped bytes: {}", escape_bytes(&bytes));
    if let Ok(text) = std::str::from_utf8(&bytes) {
        println!("UTF-8 display: {:?}", text);
    } else {
        println!("UTF-8 display: <invalid UTF-8>");
    }
    if id >= 256 {
        let rank = id - 256;
        let merge = model
            .merges()
            .get(usize::try_from(rank).map_err(|_| "merge rank exceeds host limits")?)
            .ok_or_else(|| format!("token ID {id} is outside the model vocabulary"))?;
        println!("Created by merge rank: {rank}");
        println!("Parents: {} + {}", merge.left, merge.right);
    } else if id >= model.vocabulary_size() {
        return Err(format!("token ID {id} is outside the model vocabulary"));
    } else {
        println!("Base byte token: {id}");
    }
    Ok(())
}

fn inspect_merges_command(arguments: &[String]) -> Result<(), String> {
    if arguments.is_empty() || arguments.len() > 2 {
        return Err(format!(
            "expected an artifact path and optional merge limit\n{USAGE}"
        ));
    }
    let artifact = read_artifact(Path::new(&arguments[0]))?;
    let model = artifact
        .flat_bpe()
        .ok_or("artifact has no flat BPE model")?;
    let limit = arguments
        .get(1)
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|error| format!("invalid merge limit {value:?}: {error}"))
        })
        .transpose()?
        .unwrap_or(50);
    for (rank, merge) in model.merges().iter().take(limit).enumerate() {
        println!(
            "rank {rank}: {} + {} -> {}",
            merge.left, merge.right, merge.result
        );
    }
    if limit < model.merges().len() {
        println!("showing {limit} of {} merges", model.merges().len());
    }
    Ok(())
}

fn escape_bytes(bytes: &[u8]) -> String {
    let mut escaped = String::from("\"");
    for byte in bytes {
        match byte {
            b'\\' => escaped.push_str("\\\\"),
            b'\"' => escaped.push_str("\\\""),
            b'\n' => escaped.push_str("\\n"),
            b'\r' => escaped.push_str("\\r"),
            b'\t' => escaped.push_str("\\t"),
            0x20..=0x7e => escaped.push(char::from(*byte)),
            _ => escaped.push_str(&format!("\\x{byte:02X}")),
        }
    }
    escaped.push('\"');
    escaped
}

fn print_decoded_bytes(bytes: Vec<u8>) {
    match String::from_utf8(bytes) {
        Ok(text) => println!("{text}"),
        Err(error) => println!("non-UTF-8 bytes: {}", escape_bytes(error.as_bytes())),
    }
}

fn one_path(arguments: &[String], description: &str) -> Result<PathBuf, String> {
    if arguments.len() != 1 {
        return Err(format!("expected one {description}\n{USAGE}"));
    }
    Ok(PathBuf::from(&arguments[0]))
}

fn write_new_artifact(path: &Path, artifact: &Artifact) -> Result<(), String> {
    let bytes = artifact.to_bytes().map_err(|error| error.to_string())?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("cannot create {}: {error}", path.display()))?;
    file.write_all(&bytes)
        .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    Ok(())
}

fn read_artifact(path: &Path) -> Result<Artifact, String> {
    let bytes =
        fs::read(path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    Artifact::from_bytes(&bytes).map_err(|error| format!("invalid {}: {error}", path.display()))
}

fn parse_token_id(specification: &str) -> Result<TokenId, String> {
    let (pack, local) = specification
        .split_once(':')
        .ok_or_else(|| format!("token {specification:?} must use pack_id:local_id syntax"))?;
    if local.contains(':') {
        return Err(format!(
            "token {specification:?} contains more than one ':'"
        ));
    }
    let pack = pack
        .parse::<u16>()
        .map_err(|error| format!("invalid pack ID in {specification:?}: {error}"))?;
    let local = local
        .parse::<u32>()
        .map_err(|error| format!("invalid local ID in {specification:?}: {error}"))?;
    Ok(TokenId::new(pack, local))
}

#[cfg(test)]
mod tests {
    use super::*;
    use packtok_core::DEFAULT_BYTE_FALLBACK_PACK_ID;

    #[test]
    fn parses_full_width_pack_and_local_ids() {
        assert_eq!(
            parse_token_id("65535:4294967295"),
            Ok(TokenId::new(DEFAULT_BYTE_FALLBACK_PACK_ID, u32::MAX))
        );
    }

    #[test]
    fn rejects_malformed_token_ids() {
        assert!(parse_token_id("5").is_err());
        assert!(parse_token_id("65536:1").is_err());
        assert!(parse_token_id("5:4294967296").is_err());
        assert!(parse_token_id("5:1:2").is_err());
    }

    #[test]
    fn byte_inspection_escapes_controls_and_invalid_utf8() {
        assert_eq!(escape_bytes(b"a\n\xff"), "\"a\\n\\xFF\"");
    }
}
