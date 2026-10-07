use std::process::Command;

fn decode(arguments: &[String]) -> Vec<u8> {
    let output = Command::new(env!("CARGO_BIN_EXE_packtok"))
        .arg("decode")
        .args(arguments)
        .output()
        .expect("run decoder");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

#[test]
fn byte_fallback_stdout_preserves_exact_bytes_and_encode_round_trips() {
    for input in [
        b"abc".as_slice(),
        b"",
        b"\n\r\n\t",
        &[0xff, 0, 0xc3, 0x28],
        "äöüß 👩🏽‍💻".as_bytes(),
    ] {
        let ids: Vec<_> = input.iter().map(|byte| format!("65535:{byte}")).collect();
        assert_eq!(decode(&ids), input);
    }
    let encoded = Command::new(env!("CARGO_BIN_EXE_packtok"))
        .args(["encode", "abc"])
        .output()
        .expect("encode text");
    assert!(encoded.status.success());
    assert_eq!(
        decode(&[String::from_utf8(encoded.stdout).expect("text IDs")]),
        b"abc"
    );
}

#[test]
fn bpe_stdout_preserves_exact_text_raw_bytes_and_empty_output() {
    let path = std::env::temp_dir().join(format!(
        "packtok-stdout-test-{}.packtok",
        std::process::id()
    ));
    let artifact =
        packtok_train::train_bpe(b"abababab", packtok_train::BpeTrainingConfig::default())
            .expect("train");
    std::fs::write(&path, artifact.to_bytes().expect("serialize")).expect("write artifact");
    let options = vec![
        "--artifact".to_owned(),
        path.to_str().expect("Unicode path").to_owned(),
    ];
    assert_eq!(decode(&options), b"");
    let mut raw = options.clone();
    raw.push("255 0 195 40".to_owned());
    assert_eq!(decode(&raw), [255, 0, 195, 40]);
    for input in ["abc", "abababab", "äöüß 👩🏽‍💻", "\n\r\n\t"] {
        let output = Command::new(env!("CARGO_BIN_EXE_packtok"))
            .arg("encode")
            .args(&options)
            .arg(input)
            .output()
            .expect("BPE encode");
        assert!(output.status.success());
        let mut arguments = options.clone();
        arguments.push(String::from_utf8(output.stdout).expect("text IDs"));
        assert_eq!(decode(&arguments), input.as_bytes());
    }
    std::fs::remove_file(path).expect("remove artifact");
}
