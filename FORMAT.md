# PackTok artifact format

This file is normative for the standalone implementation. M0 version 1 remains
readable and writable. M1 adds version 2 for flat byte-level BPE. M2 adds
version 3 for factorized pack-local BPE.

## Shared encoding rules

- Eight-byte magic: `PACKTOK\0`.
- Integers are little-endian.
- Strings are UTF-8 preceded by a `u32` byte length.
- The fixed 20-byte header contains magic, `u16` version, zero `u16` flags,
  `u16` reserved byte-fallback pack ID, zero `u16` reserved field, and `u32` pack
  count.
- Pack descriptors are sorted by ascending pack ID. Each has `u16` ID, `u32`
  local-token count, then a length-prefixed name.
- Special-token declarations are sorted by `(pack ID, local ID)`. Each contains
  `u16` pack ID, `u32` local ID, then a length-prefixed name.
- Metadata is a sorted map of length-prefixed UTF-8 key/value pairs. It is
  descriptive and does not define runtime behavior.
- The parser rejects unknown versions, nonzero reserved fields, invalid UTF-8,
  noncanonical ordering, invalid references, malformed lengths, trailing bytes,
  and artifacts larger than 16 MiB.

## Version 1 — M0 contracts

After the shared header come pack descriptors, the special-token count and
records, and the metadata count and entries. The artifact ends after metadata.

The byte-fallback pack contains exactly 256 local IDs, `0..=255`, one per byte.
Version 1 contains no learned vocabulary or BPE section. The default byte-fallback
pack ID is `65535`, though a version-1 artifact may declare another ID.

`FORMAT_VERSION` remains a source-compatible constant for version 1. Each parsed
or constructed artifact reports its actual version through `format_version()`.

## Version 2 — M1 flat byte-level BPE

Version 2 keeps the exact shared header and common sections, then appends:

1. `u32` merge count.
2. One 12-byte record per rank, in rank order: `u32 left ID`, `u32 right ID`,
   `u32 result ID`.

The 256 base byte tokens have local IDs `0..=255`. Rank `r` must create result ID
`256 + r`. Both parents must be lower than the result ID, so references are
backward-only and cannot form cycles. A pair may appear only once. The parser
checks all IDs, merge order, expanded byte lengths, and the artifact-size limit.
Each token may expand to at most 1 MiB (`MAX_BPE_TOKEN_BYTES`), including tokens
constructed directly through the Rust API. Lengths are checked from the merge
parents before allocating decoded output. This rejects compact self-doubling
merge tables that would otherwise demand enormous allocations.
No textual display form is stored: each token's bytes are defined by the base
byte or its ordered parent merges.

An M1 registry contains exactly two descriptors: the flat BPE vocabulary at pack
ID `0`, with local count `256 + merge_count`, and a separate 256-ID byte-fallback
descriptor at a different pack ID. The registry has no special tokens. Pack names
are descriptive. M1 runtime IDs use pack `0`; the byte-fallback descriptor keeps
the M0 core contract present, while raw byte representability in M1 comes from
the base IDs `0..=255` of the flat vocabulary.

The BPE merge section is normative. Metadata is not: changing a corpus path or
description can change artifact bytes without changing the vocabulary or runtime
encoding. Trained provenance currently uses `bpe.*` and `corpus.*` keys described
in [M1_BPE.md](M1_BPE.md). The corpus checksum is FNV-1a 64-bit, for identification
and reproducibility only; it is not cryptographic and is not used to validate the
artifact.

Version 2 is the M1 wire contract. Tokenizer training policy and the metadata key
set remain experimental. Version 2 does not add normalization, special-token
semantics, or any factorized-pack behavior.

## Version 3 — M2 factorized pack BPE

Version 3 keeps the exact 20-byte header and shared registry, special-token, and
metadata sections used by versions 1 and 2. After common metadata it appends:

1. A UTF-8 length-prefixed router policy ID. The only accepted ID is
   lexical-v1.
2. A u32 count of specialized merge graphs, bounded at three.
3. For each graph in ascending u16 pack-ID order: pack ID (u16), merge count
   (u32), then ordered merge records.
4. Each merge record is 10 bytes: left SymbolRef, then right SymbolRef.
   A symbol reference is a one-byte tag followed by a little-endian u32 value.
   Tag 0 means Byte(value) and requires value <= 255; tag 1 means
   Local(value) and requires that value be smaller than the current merge
   rank. Other tags are invalid.

The pack ID list may contain only non-empty lexical-v1 specialized graphs:
TEXT (0), NUMBER (1), and STRUCTURE (2). Graphs are serialized in that
order when present. Rank r creates local token ID r; references to Local
symbols resolve only in that same graph. A merge can therefore combine shared
bytes and prior tokens from its own pack, but can never refer to another pack's
local namespace. Duplicate ordered parent pairs in one graph are invalid. The
byte expansion of every learned token is bounded by MAX_BPE_TOKEN_BYTES.

The version-3 registry must contain exactly the declared specialized graphs
plus one shared byte-fallback descriptor. Its ID is the reserved default
65535, its name is BYTE_FALLBACK, and its local IDs are the unchanged byte
values 0..=255. Specialized descriptor names and local counts must match the
router registry and graph merge counts. Special-token declarations are rejected
for this version. Identical byte strings in different packs remain distinct IDs
because a token address is the pair (pack ID, local ID).

Version 3's router and merge graph are normative runtime data. Metadata remains
descriptive and does not affect encoding. M2 training records the target and
realized budget, minimum frequency, tie-break rule, corpus identity, no
normalization, and per-pack corpus/allocation counts as metadata. The training
report and exact metadata vocabulary are described in M2_PACKS.md.

The reader rejects unknown routers, unsupported pack IDs, noncanonical graph
ordering, empty graphs, invalid symbol tags or byte values, forward or
cross-namespace local references, repeated pairs, descriptor/count mismatches,
special tokens, malformed or trailing data, and tokens that exceed the
expansion bound. The artifact-size cap is 16 MiB for both reading and writing;
graph count is at most three. The parser validates all parent ranks and
expansion lengths before runtime use.

Version 3 does not reinterpret versions 1 or 2. The version-1 M0 and version-2
M1 reader/writer paths retain their prior canonical bytes and IDs. The format
reader supports all three versions without implicit migration. The router
registry, wire records, token bounds, and metadata conventions in version 3
are the current M2 experiment contract; changing their runtime meaning requires
a new artifact version.

## Limits and compatibility

All three versions are limited to 16 MiB. Pack, special-token, and metadata collections
are bounded to 65,536 entries. Version 1 serialization remains byte-for-byte
unchanged. The reader supports versions 1, 2, and 3; it does not migrate between them.
Constructors and writers enforce the reader's collection and total-size limits
in every version, so accepted artifacts cannot serialize into an oversized or
over-counted file that this reader would reject. The CLI also bounds bytes read
before parsing, including files that grow while being read.
The 1 MiB per-token bound tightens acceptance of previously loadable oversized
tokens; such artifacts now fail validation. It does not change wire records,
normalization, ID assignment, or the bytes of artifacts within the bound.
Future incompatible model sections require a new version or an explicitly
specified feature extension.
