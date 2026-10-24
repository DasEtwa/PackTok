//! Tiny numerical/operational reference only; no GPU or large CPU model.
use std::{
    fs,
    path::Path,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

fn command(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_packtok-semantic-scale"))
        .args(args)
        .output()
        .unwrap()
}
fn success(args: &[&str]) -> serde_json::Value {
    let result = command(args);
    assert!(
        result.status.success(),
        "{args:?}: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let output = String::from_utf8(result.stdout).unwrap();
    serde_json::from_str(output.lines().last().unwrap_or("null")).unwrap_or(serde_json::Value::Null)
}
fn path(p: &Path) -> &str {
    p.to_str().unwrap()
}

#[test]
fn tiny_cli_prepare_plan_preflight_train_resume_evaluate() {
    let root = std::env::temp_dir().join(format!(
        "packtok-scale-cli-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let input = root.join("input");
    let data = root.join("data");
    let out = root.join("run");
    let plan = root.join("plan.json");
    fs::create_dir_all(&input).unwrap();
    for (split, text) in [
        ("train", "ong idk bro; keine Ahnung! ngl, I don't know. "),
        (
            "validation",
            "tbh: tomorrow remains unknown. dunno, vielleicht. ",
        ),
        (
            "test",
            "Heute scheint die Sonne. for real, another example. ",
        ),
    ] {
        fs::write(
            input.join(format!("{split}.txt")),
            format!("{text}\n").repeat(12),
        )
        .unwrap();
    }
    let r = command(&["prepare", "--input", path(&input), "--output", path(&data)]);
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    success(&[
        "plan",
        "--data",
        path(&data),
        "--out",
        path(&plan),
        "--preset",
        "tiny",
        "--context",
        "8",
        "--dtype",
        "f32",
        "--batch",
        "1",
        "--accum",
        "2",
        "--steps",
        "4",
        "--max-seconds",
        "60",
        "--checkpoint-every",
        "2",
    ]);
    let invalid_plan = root.join("one-step.json");
    let invalid = command(&[
        "plan",
        "--data",
        path(&data),
        "--out",
        path(&invalid_plan),
        "--preset",
        "tiny",
        "--context",
        "8",
        "--dtype",
        "f32",
        "--steps",
        "1",
    ]);
    assert!(!invalid.status.success());
    assert!(!invalid_plan.exists());
    let denied = command(&[
        "train",
        "--config",
        path(&plan),
        "--output",
        path(&out),
        "--cpu",
    ]);
    assert!(!denied.status.success());
    let result = success(&[
        "preflight",
        "--config",
        path(&plan),
        "--output",
        path(&out),
        "--cpu",
    ]);
    assert_eq!(result["passed"], true);
    assert_eq!(result["cuda_runtime_tested"], false);
    let summary = success(&[
        "train",
        "--config",
        path(&plan),
        "--output",
        path(&out),
        "--cpu",
    ]);
    assert_eq!(summary["completed_steps"], 4);
    assert_eq!(summary["tokens_seen"], 64);
    let pointer = fs::read(out.join("checkpoints/latest.json")).unwrap();
    let denied = command(&[
        "train",
        "--config",
        path(&plan),
        "--output",
        path(&out),
        "--cpu",
    ]);
    assert!(!denied.status.success());
    success(&[
        "train",
        "--config",
        path(&plan),
        "--output",
        path(&out),
        "--cpu",
        "--resume",
    ]);
    assert_eq!(
        pointer,
        fs::read(out.join("checkpoints/latest.json")).unwrap()
    );
    let r = command(&[
        "evaluate",
        "--config",
        path(&plan),
        "--output",
        path(&out),
        "--cpu",
    ]);
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    let report: serde_json::Value =
        serde_json::from_slice(&fs::read(out.join("evaluation.json")).unwrap()).unwrap();
    assert_eq!(report["step"], 4);
    assert_eq!(report["reports"]["validation"]["full_corpus"], false);
    assert!(
        report["reports"]["test"]["bits_per_raw_byte"]
            .as_f64()
            .unwrap()
            .is_finite()
    );
    let training_file = data.join("train.ptks");
    let mut damaged = fs::read(&training_file).unwrap();
    damaged[0] ^= 1;
    fs::write(&training_file, damaged).unwrap();
    let failed = command(&[
        "preflight",
        "--config",
        path(&plan),
        "--output",
        path(&out),
        "--cpu",
    ]);
    assert!(!failed.status.success());
    assert!(!out.join("preflight.json").exists());
    assert_eq!(
        pointer,
        fs::read(out.join("checkpoints/latest.json")).unwrap()
    );
    fs::remove_dir_all(root).unwrap();
}
