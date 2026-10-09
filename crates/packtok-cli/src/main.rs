#![forbid(unsafe_code)]

use std::env;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use packtok_core::TokenId;
use packtok_format::{Artifact, FlatBpeModel, MAX_ARTIFACT_BYTES, SymbolRef};
use packtok_packs::{LEXICAL_V1_ROUTER_ID, LexicalV1Router, PackRouter, lexical_pack_name};
use packtok_tokenizer::{BpeTokenizer, ByteFallbackTokenizer, FactorizedTokenizer, Tokenizer};
use packtok_train::{
    BpeTrainingConfig, FactorizedTrainingConfig, load_corpus, train_bpe_with_provenance,
    train_factorized_bpe_with_provenance_report,
};

const USAGE: &str = "Usage:
  packtok encode <text>
  packtok encode --artifact <artifact-path> <text>
  packtok decode <pack_id:local_id ...>
  packtok decode --artifact <artifact-path> <local_id ...>
  packtok init <artifact-path>
  packtok train-bpe <corpus-file-or-directory> <artifact-path> [--target-vocab N] [--max-merges N] [--min-frequency N]
  packtok train-packs <corpus-file-or-directory> <artifact-path> [--target-vocab N] [--max-merges N] [--min-frequency N]
  packtok route <text>
  packtok validate <artifact-path>
  packtok inspect <artifact-path>
  packtok inspect-pack <artifact-path> <pack-name-or-id>
  packtok inspect-token <artifact-path> <local-id | pack:local>
  packtok inspect-merges <artifact-path> [pack-name-or-id] [limit]
  packtok stats <artifact-path>";

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
    let mut args = env::args_os().skip(1).map(|argument| {
        argument
            .into_string()
            .map_err(|argument| format!("argument {argument:?} is not valid Unicode"))
    });
    let command = args.next().transpose()?.ok_or_else(|| USAGE.to_owned())?;
    let arguments = args.collect::<Result<Vec<_>, _>>()?;

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
        "train-packs" => train_packs_command(&arguments),
        "route" => route_command(&arguments),
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
        "inspect-pack" => inspect_pack_command(&arguments),
        "stats" => stats_command(&arguments),
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
        if artifact.factorized_bpe().is_some() {
            let tokenizer =
                FactorizedTokenizer::from_artifact(&artifact).map_err(|error| error.to_string())?;
            let input = arguments[2..].join(" ");
            let tokens = tokenizer
                .encode(&input)
                .map_err(|error| error.to_string())?;
            println!("{}", format_token_ids(&artifact, &tokens));
            return Ok(());
        }
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
        if artifact.factorized_bpe().is_some() {
            let tokenizer =
                FactorizedTokenizer::from_artifact(&artifact).map_err(|error| error.to_string())?;
            let ids = arguments[2..]
                .iter()
                .flat_map(|argument| argument.split_whitespace())
                .map(|value| parse_artifact_token_id(&artifact, value))
                .collect::<Result<Vec<_>, _>>()?;
            let bytes = tokenizer
                .decode_bytes(&ids)
                .map_err(|error| error.to_string())?;
            return write_decoded_bytes(&mut std::io::stdout().lock(), &bytes);
        }
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
        return write_decoded_bytes(&mut std::io::stdout().lock(), &bytes);
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
    write_decoded_bytes(&mut std::io::stdout().lock(), &bytes)
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

fn train_packs_command(arguments: &[String]) -> Result<(), String> {
    if arguments.len() < 2 {
        return Err(format!("missing corpus or output artifact path\n{USAGE}"));
    }
    let corpus_path = Path::new(&arguments[0]);
    let output_path = Path::new(&arguments[1]);
    let mut config = FactorizedTrainingConfig::default();
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
                config.max_learned_tokens = value.parse().map_err(|error| {
                    format!("invalid global learned-token limit {value:?}: {error}")
                })?;
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
    let report =
        train_factorized_bpe_with_provenance_report(corpus.bytes(), config, corpus.provenance())
            .map_err(|error| error.to_string())?;
    write_new_artifact(output_path, &report.artifact)?;
    println!(
        "trained factorized BPE with router {}; shared byte tokens=256; total vocabulary={}; corpus {} bytes across {} files; wrote {}",
        LEXICAL_V1_ROUTER_ID,
        report.total_logical_vocab_size,
        corpus.provenance().total_bytes,
        corpus.provenance().files.len(),
        output_path.display()
    );
    for stats in report.packs {
        println!(
            "  pack {}: {} spans, {} routed bytes, {} learned tokens",
            pack_name(stats.pack_id),
            stats.spans,
            stats.routed_bytes,
            stats.learned_merges
        );
    }
    Ok(())
}

fn route_command(arguments: &[String]) -> Result<(), String> {
    if arguments.is_empty() {
        return Err(format!("missing text to route\n{USAGE}"));
    }
    let text = arguments.join(" ");
    let router = LexicalV1Router;
    println!("router: {}", router.policy_id());
    for span in router.route(&text) {
        println!(
            "{:<10} {}",
            pack_name(span.pack_id),
            escape_bytes(&text.as_bytes()[span.start..span.end])
        );
    }
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
    if let Some(model) = artifact.factorized_bpe() {
        println!("router policy: {}", model.router_policy_id());
        println!("shared byte fallback: 256 local IDs");
        println!(
            "learned tokens across packs: {}",
            model.learned_token_count()
        );
        for pack in model.packs() {
            println!(
                "{} (pack {}): {} learned local IDs / merges",
                pack_name(pack.pack_id()),
                pack.pack_id(),
                pack.local_token_count()
            );
        }
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
    if artifact.factorized_bpe().is_some() {
        let id = parse_artifact_token_id(&artifact, &arguments[1])?;
        return print_factorized_token(&artifact, id);
    }
    let model = artifact
        .flat_bpe()
        .ok_or("artifact has no flat BPE model")?;
    let id = arguments[1]
        .parse::<u32>()
        .map_err(|error| format!("invalid local token ID {:?}: {error}", arguments[1]))?;
    print_token(model, id)
}

fn print_factorized_token(artifact: &Artifact, id: TokenId) -> Result<(), String> {
    let tokenizer =
        FactorizedTokenizer::from_artifact(artifact).map_err(|error| error.to_string())?;
    let bytes = tokenizer
        .decode_bytes(&[id])
        .map_err(|error| error.to_string())?;
    println!("Pack: {}", pack_name(id.pack));
    println!("Pack ID: {}", id.pack);
    println!("Local ID: {}", id.local);
    println!("Bytes: {bytes:?}");
    println!("Escaped bytes: {}", escape_bytes(&bytes));
    if let Ok(text) = std::str::from_utf8(&bytes) {
        println!("UTF-8 display: {text:?}");
    } else {
        println!("UTF-8 display: <invalid UTF-8>");
    }
    if id.pack == artifact.registry().byte_fallback().pack_id() {
        println!("Shared raw byte token: {}", id.local);
    } else {
        let model = artifact
            .factorized_bpe()
            .ok_or("artifact has no factorized BPE model")?;
        let pack = model
            .pack(id.pack)
            .ok_or("token pack has no learned model")?;
        let rank = usize::try_from(id.local).map_err(|_| "local ID exceeds host limits")?;
        let merge = pack
            .merges()
            .get(rank)
            .ok_or("token local ID is outside its pack model")?;
        println!("Created by local merge rank: {}", id.local);
        println!(
            "Parents: {} + {}",
            format_symbol(merge.left),
            format_symbol(merge.right)
        );
    }
    Ok(())
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
    if arguments.is_empty() || arguments.len() > 3 {
        return Err(format!(
            "expected an artifact path and optional merge limit\n{USAGE}"
        ));
    }
    let artifact = read_artifact(Path::new(&arguments[0]))?;
    if let Some(model) = artifact.factorized_bpe() {
        if arguments.len() < 2 {
            return Err(format!("M2 artifacts require a pack name or ID\n{USAGE}"));
        }
        let pack_id = parse_pack_id(&artifact, &arguments[1])?;
        let pack = model
            .pack(pack_id)
            .ok_or_else(|| format!("pack {} has no learned merge table", pack_name(pack_id)))?;
        let limit = arguments
            .get(2)
            .map(|value| {
                value
                    .parse::<usize>()
                    .map_err(|error| format!("invalid merge limit {value:?}: {error}"))
            })
            .transpose()?
            .unwrap_or(50);
        for (rank, merge) in pack.merges().iter().take(limit).enumerate() {
            println!(
                "{} rank {rank}: {} + {} -> {}:{rank}",
                pack_name(pack_id),
                format_symbol(merge.left),
                format_symbol(merge.right),
                pack_name(pack_id)
            );
        }
        if limit < pack.merges().len() {
            println!("showing {limit} of {} merges", pack.merges().len());
        }
        return Ok(());
    }
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

fn inspect_pack_command(arguments: &[String]) -> Result<(), String> {
    if arguments.len() != 2 {
        return Err(format!(
            "expected an artifact path and pack name or ID\n{USAGE}"
        ));
    }
    let artifact = read_artifact(Path::new(&arguments[0]))?;
    let pack_id = parse_pack_id(&artifact, &arguments[1])?;
    let descriptor = artifact
        .registry()
        .pack(pack_id)
        .ok_or_else(|| format!("artifact has no pack {}", arguments[1]))?;
    println!("Pack: {}", descriptor.name());
    println!("Pack ID: {}", descriptor.id());
    println!("Local token count: {}", descriptor.local_token_count());
    if let Some(model) = artifact.factorized_bpe() {
        if let Some(pack) = model.pack(pack_id) {
            println!("Learned merges: {}", pack.merges().len());
        } else if pack_id == artifact.registry().byte_fallback().pack_id() {
            println!("Kind: shared raw byte fallback (0..=255)");
        } else {
            println!("Learned merges: 0 (all routed symbols use shared byte fallback)");
        }
    }
    Ok(())
}

fn stats_command(arguments: &[String]) -> Result<(), String> {
    let artifact = read_artifact(&one_path(arguments, "artifact path")?)?;
    println!("PackTok artifact v{} statistics", artifact.format_version());
    for (key, value) in artifact.metadata() {
        if key.starts_with("m2.") || key.starts_with("corpus.") || key.starts_with("experiment.") {
            println!("{key}={value}");
        }
    }
    Ok(())
}

fn format_symbol(symbol: SymbolRef) -> String {
    match symbol {
        SymbolRef::Byte(byte) => format!("BYTE({byte})"),
        SymbolRef::Local(local) => format!("LOCAL({local})"),
    }
}

fn pack_name(pack_id: u16) -> String {
    if pack_id == packtok_core::DEFAULT_BYTE_FALLBACK_PACK_ID {
        return "BYTE_FALLBACK".to_owned();
    }
    lexical_pack_name(pack_id)
        .map(str::to_owned)
        .unwrap_or_else(|| pack_id.to_string())
}

fn format_token_ids(artifact: &Artifact, tokens: &[TokenId]) -> String {
    tokens
        .iter()
        .map(|token| {
            let name = artifact
                .registry()
                .pack(token.pack)
                .map(|pack| pack.name())
                .unwrap_or_else(|| "UNKNOWN_PACK");
            format!("{name}:{}", token.local)
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn parse_pack_id(artifact: &Artifact, specification: &str) -> Result<u16, String> {
    if let Some(pack) = artifact
        .registry()
        .packs()
        .iter()
        .find(|pack| pack.name() == specification)
    {
        return Ok(pack.id());
    }
    let pack_id = specification
        .parse::<u16>()
        .map_err(|error| format!("unknown pack {specification:?}: {error}"))?;
    if artifact.registry().pack(pack_id).is_none() {
        return Err(format!("artifact has no pack ID {pack_id}"));
    }
    Ok(pack_id)
}

fn parse_artifact_token_id(artifact: &Artifact, specification: &str) -> Result<TokenId, String> {
    let (pack, local) = specification
        .split_once(':')
        .ok_or_else(|| format!("M2 token {specification:?} must use PACK:LOCAL syntax"))?;
    if local.contains(':') {
        return Err(format!(
            "token {specification:?} contains more than one ':'"
        ));
    }
    let pack_id = parse_pack_id(artifact, pack)?;
    let local_id = local
        .parse::<u32>()
        .map_err(|error| format!("invalid local ID in {specification:?}: {error}"))?;
    Ok(TokenId::new(pack_id, local_id))
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

fn write_decoded_bytes(writer: &mut impl Write, bytes: &[u8]) -> Result<(), String> {
    writer
        .write_all(bytes)
        .and_then(|()| writer.flush())
        .map_err(|error| format!("cannot write decoded bytes: {error}"))
}

fn one_path(arguments: &[String], description: &str) -> Result<PathBuf, String> {
    if arguments.len() != 1 {
        return Err(format!("expected one {description}\n{USAGE}"));
    }
    Ok(PathBuf::from(&arguments[0]))
}

fn write_new_artifact(path: &Path, artifact: &Artifact) -> Result<(), String> {
    write_new_artifact_with(path, artifact, |file, bytes| {
        file.write_all(bytes)?;
        file.sync_all()
    })
}

fn write_new_artifact_with(
    path: &Path,
    artifact: &Artifact,
    write: impl FnOnce(&mut File, &[u8]) -> std::io::Result<()>,
) -> Result<(), String> {
    let bytes = artifact.to_bytes().map_err(|error| error.to_string())?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("cannot create {}: {error}", path.display()))?;
    if let Err(error) = write(&mut file, &bytes) {
        // Only this create_new call created the destination. Close it before
        // removal on Windows, and let a subsequent training attempt retry.
        drop(file);
        let cleanup = std::fs::remove_file(path);
        return Err(match cleanup {
            Ok(()) => format!(
                "cannot write {}: {error}; removed incomplete artifact",
                path.display()
            ),
            Err(cleanup_error) => format!(
                "cannot write {}: {error}; cannot remove incomplete artifact: {cleanup_error}",
                path.display()
            ),
        });
    }
    Ok(())
}

fn read_artifact(path: &Path) -> Result<Artifact, String> {
    let file =
        File::open(path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    read_bounded_artifact(file).map_err(|error| format!("invalid {}: {error}", path.display()))
}

fn read_bounded_artifact(reader: impl Read) -> Result<Artifact, String> {
    let mut bytes = Vec::new();
    reader
        .take(MAX_ARTIFACT_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("cannot read artifact: {error}"))?;
    Artifact::from_bytes(&bytes).map_err(|error| error.to_string())
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
    fn parses_m2_pack_names_and_local_ids() {
        let artifact = train_factorized_bpe_with_provenance_report(
            b"abab 1212",
            FactorizedTrainingConfig {
                target_vocab_size: 258,
                max_learned_tokens: 2,
                min_pair_frequency: 1,
            },
            &packtok_train::CorpusProvenance {
                input: "test".to_owned(),
                files: vec![],
                total_bytes: 9,
                fnv1a64: packtok_train::fnv1a64(b"abab 1212"),
            },
        )
        .expect("train test artifact")
        .artifact;
        assert_eq!(
            parse_artifact_token_id(&artifact, "TEXT:0"),
            Ok(TokenId::new(0, 0))
        );
        assert_eq!(
            parse_artifact_token_id(&artifact, "NUMBER:0"),
            Ok(TokenId::new(1, 0))
        );
        assert_eq!(
            parse_artifact_token_id(&artifact, "BYTE_FALLBACK:32"),
            Ok(TokenId::new(DEFAULT_BYTE_FALLBACK_PACK_ID, 32))
        );
        assert_eq!(
            format_token_ids(&artifact, &[TokenId::new(0, 0), TokenId::new(1, 0)]),
            "TEXT:0 NUMBER:0"
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

    #[test]
    fn artifact_reads_stop_at_size_limit() {
        // An endless source must be rejected after at most limit + 1 bytes.
        let mut reader = std::io::repeat(0).take(u64::MAX);
        let error = read_bounded_artifact(&mut reader).expect_err("oversized artifact rejected");
        assert!(error.contains("exceeds limit"));
        assert_eq!(reader.limit(), u64::MAX - MAX_ARTIFACT_BYTES as u64 - 1);
    }

    #[test]
    fn bounded_artifact_reader_accepts_legacy_and_rejects_truncation() {
        let artifact = Artifact::byte_fallback_only();
        let bytes = artifact.to_bytes().expect("serialize");
        assert_eq!(read_bounded_artifact(bytes.as_slice()), Ok(artifact));
        assert!(read_bounded_artifact(&bytes[..bytes.len() - 1]).is_err());
    }

    #[test]
    fn decoded_output_is_exact_for_text_empty_and_invalid_utf8() {
        for input in [
            b"abc".as_slice(),
            b"",
            b"\n\r\n",
            &[0xff, 0, 0xc3, 0x28],
            "äöüß 👩🏽‍💻".as_bytes(),
        ] {
            let mut output = Vec::new();
            write_decoded_bytes(&mut output, input).expect("write bytes");
            assert_eq!(output, input);
        }
        assert!(
            write_decoded_bytes(&mut std::io::Cursor::new(&mut [0_u8; 1][..]), b"abc").is_err()
        );
    }

    #[test]
    fn failed_artifact_write_is_removed_and_existing_files_are_preserved() {
        let path = std::env::temp_dir().join(format!(
            "packtok-write-failure-{}.packtok",
            std::process::id()
        ));
        let artifact = Artifact::byte_fallback_only();
        let error = write_new_artifact_with(&path, &artifact, |file, bytes| {
            file.write_all(&bytes[..3])?;
            Err(std::io::Error::new(
                std::io::ErrorKind::StorageFull,
                "injected write failure",
            ))
        })
        .expect_err("write must fail");
        assert!(error.contains("removed incomplete artifact"));
        assert!(!path.exists());
        write_new_artifact(&path, &artifact).expect("retry succeeds");
        let expected = std::fs::read(&path).expect("read complete artifact");
        assert_eq!(Artifact::from_bytes(&expected), Ok(artifact.clone()));
        assert!(write_new_artifact(&path, &artifact).is_err());
        assert_eq!(
            std::fs::read(&path).expect("read preserved artifact"),
            expected
        );
        std::fs::remove_file(path).expect("remove test artifact");
    }
}
