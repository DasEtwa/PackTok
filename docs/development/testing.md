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

## M5 Windows checkpoint and report synchronization

On 2026-11-04, seven checkpoint/report tests initially returned Windows OS error
5 (`PermissionDenied`), while 36 passed and the historical test was ignored.
Windows requires a directory handle opened with `FILE_FLAG_BACKUP_SEMANTICS`
and requires write access for the observed flush. M5 now uses those flags,
flushes the directory, and propagates errors; it opens the temporary checkpoint
file with write access before flushing. Existing Unix synchronization is
unchanged. These operations passed on the local Windows filesystem, but the
documented Windows directory-handle operation list does not establish
POSIX-equivalent namespace durability after power failure; these tests are not a
power-loss experiment and filesystem behavior can differ. See Microsoft's
[directory handle](https://learn.microsoft.com/en-us/windows/win32/fileio/obtaining-a-handle-to-a-directory)
and [file flush](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-flushfilebuffers)
documentation. With Rust 1.99.0, the 44 regular M5 tests passed in debug and
release on native Windows and Ubuntu-24.04; the historical test remains a
separate explicit invocation.

## Frozen M5 sequence integration test

The test `frozen_configs_recheck_scientific_inputs_and_schedules` is ignored in
the ordinary suite because it requires six large, untracked original M5
`prepared-v2` sequences. Run it separately on Ubuntu-24.04 with those exact
files, either in `experiments/m5-gpu/data/prepared-v2` or in an external
directory supplied as an absolute path:

```sh
PACKTOK_M5_PREPARED_DIR=/absolute/path/to/prepared-v2 \
  cargo +stable test --locked --manifest-path experiments/m5-gpu/Cargo.toml \
  extended::tests::frozen_configs_recheck_scientific_inputs_and_schedules -- --ignored --exact
```

The test checks the SHA-256 identities in the frozen configurations, including
all six sequences; synthetic replacements cannot pass. Expected hashes are in
`experiments/m5-gpu/configs/pilot-2k-v1.json`. Restore the original files from a
verified backup, or reproduce the frozen corpus and preparation in an isolated
checkout using `scripts/acquire.sh` and the M5 `corpus` and `prepare` commands.
Check source availability, exact source hashes and licenses first: acquisition
downloads data and preparation writes artifacts and provenance. Do not run
those commands over an existing research directory. Verify the six outputs
against the frozen hashes before using them. No GPU is needed.

For reconstruction in a disposable checkout of the same revision, run from its
root (the acquisition script verifies the tracked source hash list):

```sh
bash experiments/m5-gpu/scripts/acquire.sh
cargo +stable run --release --locked --manifest-path experiments/m5-gpu/Cargo.toml -- corpus
cargo +stable run --release --locked --manifest-path experiments/m5-gpu/Cargo.toml -- prepare
cargo +stable run --release --locked --manifest-path experiments/m5-gpu/Cargo.toml -- verify-prepared
cargo +stable test --locked --manifest-path experiments/m5-gpu/Cargo.toml \
  extended::tests::frozen_configs_recheck_scientific_inputs_and_schedules -- --ignored --exact
```

The historical corpus contains approximately 37.5 MB of split text; source
downloads and generated sequence files require additional disk space. Source
changes or unavailability may prevent reconstruction; never relax recorded
hashes to make it pass.

On 2026-11-04 the six sequences were found in the local Ubuntu-24.04 M6
continuation bundle at `PackTok-m6-research/experiments/m5-gpu/provenance/
m6-five-seed-cont-20261010T022723Z/packtok-m5/data/prepared-v2`. All six
SHA-256 values matched `pilot-2k-v1.json`. The exact ignored-test command above,
with `PACKTOK_M5_PREPARED_DIR` set to that directory, passed on CPU (1 test,
70.69 seconds). The existing bundle was read without modification; no GPU or
new model training was used. The local path is machine-specific and is not a
substitute for a separately verified backup. After the Windows synchronization
change, the same explicit historical test passed again on Ubuntu-24.04 in
89.51 seconds, using the same original sequence directory.

Shell scripts use LF in Git checkouts. Frozen M5 configurations and M6 analysis
evidence are checked out byte-for-byte; `SHA256SUMS-v2.txt` can be checked from
the repository root. The older `SHA256SUMS.txt` refers to the revision-1 script,
not the current script; retrieve that script from commit `6138c06` when checking
the historical pair.
