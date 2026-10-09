# Building and CLI use

The root workspace declares Rust 1.85 and edition 2024. Install from
[rustup](https://rustup.rs/), then from the repository root:

```sh
rustup toolchain install 1.85.0 --profile minimal --component rustfmt --component clippy
cargo +1.85.0 build --release --workspace
cargo +1.85.0 run --release -p packtok-cli -- help
cargo +1.85.0 run --release -p packtok-cli -- encode "Hi"
cargo +1.85.0 run --release -p packtok-cli -- decode "65535:72 65535:105"
```

The last command writes exactly `Hi`, without a newline. CLI tokens are supplied
as one quoted whitespace-separated argument. Artifact paths must be fresh:

```sh
cargo +1.85.0 run --release -p packtok-cli -- train-bpe fixtures/m1_bpe_corpus.txt target/example.packtok
cargo +1.85.0 run --release -p packtok-cli -- inspect target/example.packtok
cargo +1.85.0 run --release -p packtok-cli -- encode --artifact target/example.packtok "Sämtliche Häuser"
cargo +1.85.0 run --release -p packtok-cli -- validate target/example.packtok
cargo +1.85.0 run --release -p packtok-cli -- route "Hello 123!"
cargo +1.85.0 run --release -p packtok-cli -- train-packs fixtures/benchmark/train.txt target/packs-example.packtok
cargo +1.85.0 run --release -p packtok-cli -- inspect-pack target/packs-example.packtok TEXT
cargo +1.85.0 run --release -p packtok-cli -- inspect-token target/packs-example.packtok TEXT:0
cargo +1.85.0 run --release -p packtok-cli -- inspect-merges target/packs-example.packtok TEXT
cargo +1.85.0 run --release -p packtok-cli -- stats target/packs-example.packtok
```

Use `decode --artifact PATH "<numeric IDs from encode>"` for trained artifacts.
Examples are validated against the actual release CLI in the
[maintenance record](../maintenance/2026-10-08/REPORT.md).

M5 runs only in the dedicated Ubuntu-24.04 checkout
`/home/dasetwa/projects/PackTok` (host VHD under `D:\PackTok-WSL`, user dasetwa).
It is a separate manifest, not a root workspace member. Its recorded toolchain is
Rust/Cargo 1.99.0 and pinned Cargo.lock; see [testing](testing.md). No Rust toolchain
file is currently present. The root remains compatible with its declared MSRV.
