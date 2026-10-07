# M3 synthetic model split

These three files are manually authored synthetic PackTok examples distributed
under the repository Apache-2.0 license. They are not downloaded, externally
annotated, or selected from the M1/M2 tokenizer benchmark evaluation files.
The split method is file assignment: train.txt is training-only,
validation.txt is used for observational loss curves, and test.txt is held out
for final reporting.

Files contain the raw checked-in UTF-8 bytes, including their line feeds. There
is no normalization, whitespace cleanup, inserted boundary text, or file
separator. Each split is read independently. Training artifacts are learned
only from train.txt. The harness rejects any exact contiguous 32-byte sequence
shared across two split files.

The corpus is intentionally synthetic and small; it supports a controlled
implementation check and does not represent a natural-language population.
SHA-256 fingerprints and raw byte totals are recorded in
M3_MODEL_BENCHMARK.md.
