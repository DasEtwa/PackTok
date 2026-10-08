# Verification gates

Root CPU workspace, from repository root:

```sh
cargo +1.85.0 fmt --all --check
cargo +1.85.0 clippy --workspace --all-targets --all-features -- -D warnings
cargo +1.85.0 test --workspace
cargo +1.85.0 test --workspace --release
cargo +1.85.0 build --release --workspace
```

The maintenance record also verifies Windows with its installed Rust/Cargo 1.98.1.
Tests include arbitrary-byte/Unicode/German/emoji round-trips, malformed artifact
rejection, deterministic IDs and serialization, baseline differentials, mappings,
model contracts and historical compatibility. Compiler output and counts are
retained; an interrupted invocation is not a pass.

M5 on local Ubuntu-24.04, recorded Rust/Cargo 1.99.0 (`+stable` for that installed
toolchain), CPU only:

```sh
cargo +stable fmt --manifest-path experiments/m5-gpu/Cargo.toml --all --check
cargo +stable clippy --locked --manifest-path experiments/m5-gpu/Cargo.toml --all-targets -- -D warnings
cargo +stable test --locked --manifest-path experiments/m5-gpu/Cargo.toml
cargo +stable test --locked --release --manifest-path experiments/m5-gpu/Cargo.toml
cargo +stable build --locked --release --manifest-path experiments/m5-gpu/Cargo.toml --example recover_weights
python3 experiments/m5-gpu/scripts/test-recovery.py
bash experiments/m5-gpu/scripts/test-lifecycle.sh
bash experiments/m5-gpu/scripts/test-transfer.sh
```

[WSL supervision](gpu-colab.md) documents the longer actual-entrypoint tests.
[Storage](storage-and-recovery.md) describes independent Drive readback tests.
[Maintenance evidence](../maintenance/2026-10-08/REPORT.md) records links,
whitespace/security, artifact hashes and scoped self-review. CPU compilation
of CUDA features is not evidence of GPU execution.
