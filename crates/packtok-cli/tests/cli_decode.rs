use std::process::Command;

#[cfg(any(unix, windows))]
#[test]
fn non_unicode_arguments_return_a_clear_error_instead_of_panicking() {
    #[cfg(unix)]
    let invalid = {
        use std::os::unix::ffi::OsStringExt;
        std::ffi::OsString::from_vec(vec![0xff])
    };
    #[cfg(windows)]
    let invalid = {
        use std::os::windows::ffi::OsStringExt;
        std::ffi::OsString::from_wide(&[0xd800])
    };
    let output = Command::new(env!("CARGO_BIN_EXE_packtok"))
        .arg("validate")
        .arg(invalid)
        .output()
        .expect("invoke CLI");
    assert_eq!(
        output.status.code(),
        Some(1),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stderr).contains("not valid Unicode"));
    assert!(output.stdout.is_empty());
}

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

#[test]
fn factorized_cli_routes_encodes_decodes_and_inspects_pack_tokens() {
    let path = std::env::temp_dir().join(format!(
        "packtok-factorized-cli-{}.packtok",
        std::process::id()
    ));
    let artifact = packtok_train::train_factorized_bpe(
        b"hello 123! hello 123! hello 123!",
        packtok_train::FactorizedTrainingConfig {
            target_vocab_size: 272,
            max_learned_tokens: 16,
            min_pair_frequency: 1,
        },
    )
    .expect("train factorized artifact");
    std::fs::write(&path, artifact.to_bytes().expect("serialize artifact"))
        .expect("write artifact");
    let path_string = path.to_str().expect("Unicode path");

    let route = Command::new(env!("CARGO_BIN_EXE_packtok"))
        .args(["route", "Hello", "123!"])
        .output()
        .expect("route input");
    assert!(route.status.success());
    let route_output = String::from_utf8(route.stdout).expect("route output");
    assert!(route_output.contains("TEXT"));
    assert!(route_output.contains("NUMBER"));
    assert!(route_output.contains("STRUCTURE"));

    let input = "hello 123!";
    let encoded = Command::new(env!("CARGO_BIN_EXE_packtok"))
        .args(["encode", "--artifact", path_string, input])
        .output()
        .expect("encode input");
    assert!(
        encoded.status.success(),
        "{}",
        String::from_utf8_lossy(&encoded.stderr)
    );
    let ids = String::from_utf8(encoded.stdout).expect("token IDs");
    assert!(ids.contains("BYTE_FALLBACK:"));
    let decoded = Command::new(env!("CARGO_BIN_EXE_packtok"))
        .args(["decode", "--artifact", path_string, ids.trim()])
        .output()
        .expect("decode token IDs");
    assert!(
        decoded.status.success(),
        "{}",
        String::from_utf8_lossy(&decoded.stderr)
    );
    assert_eq!(decoded.stdout, input.as_bytes());

    let inspected = Command::new(env!("CARGO_BIN_EXE_packtok"))
        .args(["inspect-token", path_string, "TEXT:0"])
        .output()
        .expect("inspect token");
    assert!(
        inspected.status.success(),
        "{}",
        String::from_utf8_lossy(&inspected.stderr)
    );
    let inspection = String::from_utf8(inspected.stdout).expect("inspection text");
    assert!(inspection.contains("Pack: TEXT"));
    assert!(inspection.contains("Created by local merge rank"));

    let stats = Command::new(env!("CARGO_BIN_EXE_packtok"))
        .args(["stats", path_string])
        .output()
        .expect("stats");
    assert!(stats.status.success());
    assert!(String::from_utf8_lossy(&stats.stdout).contains("m2.router_policy_id=lexical-v1"));
    std::fs::remove_file(path).expect("remove artifact");
}
