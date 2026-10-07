# Synthetic benchmark split

All texts here are manually authored PackTok fixtures under the repository's
Apache-2.0 license. No dataset is downloaded or externally annotated.

- `train.txt` is the only input used to learn the benchmark's BPE model.
  Repeated lines are intentional training frequency observations.
- `eval/*.txt` are held-out evaluation documents. Their first passages preserve
  the historical benchmark samples; additional short and long records permit
  sequence-length distributions. Training never reads this directory.
- No validation set is defined because this benchmark performs no model fitting,
  parameter tuning or model-quality evaluation beyond tokenizer training.

Files use raw checked-in UTF-8 bytes, with no normalization or preprocessing.
Each evaluation document is repeated 32 times for timing; length distributions
use its original newline-delimited records, including the newline bytes.
Evaluation is synthetic and small, not representative of a production corpus.
The benchmark records sizes and FNV-1a checksums and rejects any shared contiguous
32-byte substring between evaluation and training. Shared characters,
short words and generic syntax remain expected across the two splits.
