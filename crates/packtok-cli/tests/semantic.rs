use std::process::Command;

use packtok_format::Artifact;
use packtok_packs::semantic::SemanticRegistry;
use packtok_tokenizer::{
    BpeTokenizer, ByteFallbackTokenizer, FactorizedTokenizer, Tokenizer, encode_with_semantics,
};

fn semantic(arguments: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_packtok"))
        .arg("semantic")
        .args(arguments)
        .output()
        .expect("run semantic CLI")
}

#[test]
fn semantic_debug_explains_original_spans_candidates_and_decisions() {
    let output = semantic(&["ong idk bro; fr; KA"]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.starts_with("Lexical tokens: 65535:111 65535:110 65535:103"));
    assert!(
        text.contains("Semantic registry: packtok.semantic revision=1 policy=semantic-rules-v1")
    );
    assert!(text.contains("4..7 \"idk\" -> Resolved [1:UNCERTAINTY]"));
    assert!(text.contains("\"fr\" -> Ambiguous"));
    assert!(text.contains("\"KA\" -> Ambiguous"));
    assert!(text.contains("reason="));
    assert_eq!(text.as_bytes(), semantic(&["ong idk bro; fr; KA"]).stdout);
}

#[test]
fn semantic_debug_handles_empty_unknown_and_unicode_text_and_bad_arguments() {
    for input in ["", "unbekannt xyzzy 👩🏽‍💻 äöüß", "\t\r\n"] {
        let output = semantic(&[input]);
        assert!(output.status.success());
        assert!(String::from_utf8_lossy(&output.stdout).contains("Semantic annotations: 0"));
    }
    for arguments in [
        &[][..],
        &["--artifact"][..],
        &["--artifact", "missing.packtok"][..],
    ] {
        let output = semantic(arguments);
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
    }
}

fn assert_differential(tokenizer: &dyn Tokenizer) {
    let registry = SemanticRegistry::builtin();
    for input in [
        "ong idk bro",
        "ka; keine Ahnung; dunno; I don’t know",
        "fr bro\nlanguage: fr\nKA",
        "\0 \t\r\n 👩🏽‍💻 äöüß e\u{301} 漢字",
        "unknown-idk-suffix identifier_idk unknownword",
        "",
    ] {
        let baseline = tokenizer.encode(input).unwrap();
        for mode in [None, Some(registry)] {
            let encoded = encode_with_semantics(tokenizer, input, mode).unwrap();
            assert_eq!(encoded.lexical_tokens, baseline);
            assert_eq!(
                tokenizer.decode_bytes(&encoded.lexical_tokens).unwrap(),
                input.as_bytes()
            );
            assert_eq!(
                encoded,
                encode_with_semantics(tokenizer, input, mode).unwrap()
            );
        }
    }
}

#[test]
fn semantic_pipeline_preserves_real_m1_and_m2_models_and_artifacts() {
    let corpus = include_bytes!("../../../fixtures/benchmark/train.txt");
    let m1 = packtok_train::train_bpe(corpus, packtok_train::BpeTrainingConfig::default()).unwrap();
    assert_eq!(m1.flat_bpe().unwrap().vocabulary_size(), 512);
    let m2 = packtok_train::train_factorized_bpe(
        corpus,
        packtok_train::FactorizedTrainingConfig::default(),
    )
    .unwrap();
    let m1_tokenizer = BpeTokenizer::from_artifact(&m1).unwrap();
    let m2_tokenizer = FactorizedTokenizer::from_artifact(&m2).unwrap();
    assert_differential(&m1_tokenizer);
    assert_differential(&m2_tokenizer);
    // Semantic annotation accepts text; the pre-existing raw-byte APIs still
    // preserve bytes that cannot be passed through a UTF-8 text interface.
    let invalid_utf8 = b"idk\xff\0\xc3\x28 fr";
    assert_eq!(
        m1_tokenizer
            .decode_bytes(&m1_tokenizer.encode_bytes(invalid_utf8).unwrap())
            .unwrap(),
        invalid_utf8
    );
    assert_eq!(
        m2_tokenizer
            .decode_bytes(&m2_tokenizer.encode_bytes(invalid_utf8).unwrap())
            .unwrap(),
        invalid_utf8
    );
    for (label, artifact) in [
        ("m1", m1),
        ("m2", m2),
        ("m0", Artifact::byte_fallback_only()),
    ] {
        let bytes = artifact.to_bytes().unwrap();
        let path = std::env::temp_dir().join(format!(
            "packtok-semantic-{label}-{}.packtok",
            std::process::id()
        ));
        std::fs::write(&path, &bytes).unwrap();
        let output = semantic(&["--artifact", path.to_str().unwrap(), "idk fr"]);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let text = String::from_utf8(output.stdout).unwrap();
        let tokenizer: Box<dyn Tokenizer> = if label == "m1" {
            Box::new(BpeTokenizer::from_artifact(&artifact).unwrap())
        } else if label == "m2" {
            Box::new(FactorizedTokenizer::from_artifact(&artifact).unwrap())
        } else {
            Box::new(ByteFallbackTokenizer::from_registry(artifact.registry()))
        };
        let expected_ids = tokenizer
            .encode("idk fr")
            .unwrap()
            .iter()
            .map(|token| format!("{}:{}", token.pack, token.local))
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(
            text.lines().next().unwrap(),
            format!("Lexical tokens: {expected_ids}")
        );
        assert!(text.contains("UNCERTAINTY"));
        assert!(text.contains("\"fr\" -> Ambiguous"));
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        std::fs::remove_file(path).unwrap();
    }
}
