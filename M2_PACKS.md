# M2 — Factorized pack architecture

## Research question

M2 asks whether one fixed logical token budget can be split among independent
learned vocabularies while preserving exact byte representation and deterministic
training/runtime behavior, and what that factorization costs against the frozen
M1 flat byte-level BPE baseline.

M2 is an architectural control experiment. TEXT, NUMBER, and STRUCTURE are
mechanical routing labels for this experiment. They are not POS classes,
language annotations, semantic claims, or a claim that these categories match
natural-language truth. No result in M2 establishes that factorized tokenization
is better than flat BPE.

## Token identity and shared bytes

A token address is the pair (PackId, LocalTokenId). Equal local IDs in two
packs are distinct tokens. The shared byte fallback uses reserved pack ID 65535
and local IDs 0 through 255, with one local ID per byte value. This namespace is
shared by all specialized packs; byte tokens are not copied into TEXT, NUMBER,
or STRUCTURE.

The M1 control uses 256 byte IDs and up to 256 learned IDs for a target of 512
logical slots. M2 uses the same total budget: 256 shared byte IDs and up to 256
learned local IDs allocated globally among the specialized packs. A learned
token in pack P has ID (P, rank-within-P). Its merge rank and local ID are the
same zero-based number. Two packs may both have local ID zero.

## Routing policy: lexical-v1

packtok-packs defines the generic PackRouter contract and one built-in,
versioned policy named lexical-v1. The policy is context-free and classifies
Unicode scalar values without Unicode property tables:

| Input scalar | Pack |
| --- | --- |
| ASCII A–Z or a–z | TEXT |
| Any complete non-ASCII UTF-8 scalar, including umlauts, CJK, emoji, and non-ASCII punctuation | TEXT |
| ASCII 0–9 | NUMBER |
| Every other ASCII byte, including spaces, tabs, line breaks, punctuation, symbols, and controls | STRUCTURE |

Adjacent scalars assigned to the same pack are coalesced into a span. Span
offsets are byte offsets at UTF-8 character boundaries. Empty input produces no
spans. For a valid Rust string the router covers every input byte exactly once.
The policy is intentionally crude; it does not detect words, numbers with
punctuation, language, or syntax.

Raw byte APIs accept arbitrary byte slices. Valid UTF-8 is routed as above.
Invalid UTF-8 is encoded entirely through the shared byte fallback, so arbitrary
bytes remain representable. M2 training accepts only valid UTF-8 corpora and
reports the first invalid offset. No API normalizes text.

## Span boundaries and exact representation

Training and runtime only merge adjacent symbols inside one routed span. A merge
cannot cross a routing boundary or refer to another pack's local symbol. A
merge parent is either Byte(u8) in the one shared byte namespace or Local(id)
in the current pack.

For example, hello123 routes to TEXT hello followed by NUMBER 123. The pair o1
is not eligible for an M2 merge. hello world has TEXT, STRUCTURE-space, and
TEXT spans, so a flat-BPE token such as a leading-space word token is
unavailable. These restrictions may reduce compression relative to M1; that is
an intended measured consequence of factorization, not a reason to alter the
evaluation set or routing after observing results.

The encoder applies each pack's ordered merge ranks independently within each
span. Unmerged symbols emit (BYTE_FALLBACK, byte). Learned symbols emit their
pack-local ID. The decoder does not re-route: it expands each token's byte
sequence and concatenates bytes in order. Thus decode is independent of span
decisions after IDs have been produced, and decode_bytes(encode(text)) equals
the original UTF-8 bytes exactly.

## Training and deterministic allocation

Training consumes the supplied corpus bytes as valid UTF-8, routes the corpus,
and keeps one symbol sequence per routed span. It does not insert separators.
Directory inputs use the existing deterministic corpus loader: paths are sorted
lexicographically and file bytes are concatenated without added separators.
The loader records the input path, file list, byte total, and FNV-1a 64-bit
corpus identifier. The benchmark report separately records SHA-256 checksums.

Each training step counts adjacent pairs separately inside every span. Counts
include overlapping windows; when a chosen pair is applied, occurrences are
replaced left-to-right and non-overlapping within each span. A pack's best
candidate is its most frequent eligible pair. Candidates whose combined
expansion would exceed 1 MiB are skipped. Selection across pack candidates is:

1. higher frequency;
2. lower numeric PackId;
3. lexicographically lower (left SymbolRef, right SymbolRef), where Byte sorts
   before Local and values sort numerically within each variant.

The selected pack assigns its next contiguous local ID and only its own symbol
sequences are rewritten. Other pack states and their candidate frequencies are
unchanged. The procedure stops at the configured global budget or when no
candidate meets minimum frequency and size bounds. A small corpus may therefore
finish below target vocabulary size.

Defaults are target vocabulary 512 logical IDs, at most 256 learned tokens
globally, and minimum pair frequency 2. The target must be at least 256, the
learned-token maximum cannot exceed target minus 256, and minimum frequency
must be positive. These values match M1's default logical budget and minimum
frequency.

The training implementation recomputes pair counts for the affected pack from
its current spans after each selected merge. With K selected merges and N
current symbols in that pack, this simple referenceable path performs roughly
O(KN) symbol visits plus ordered-map counting work. M2 has not added an
incremental pair-update structure; its measured training throughput and known
limits are recorded in M2_PACKS_BENCHMARK.md. No optimization is justified until
the benchmark identifies a meaningful bottleneck.

An independent, deliberately slower reference path separately classifies
characters, linearly counts candidate pairs, fully rebuilds affected spans, and
provides reference encoding. Differential tests compare routing, merge graphs,
and runtime IDs on deterministic fixtures.

## Version-3 artifact and metadata

Version 3 preserves the common M0/M1 artifact sections, then stores the exact
router ID and sorted pack-local ordered merge graphs. Each merge stores two
tagged SymbolRefs; rank supplies the result local ID. Runtime-defining router
and merge information is normative. Corpus, configuration, and statistics
metadata are descriptive. Full wire details and validation rules are in
FORMAT.md.

M2 training metadata includes:

- algorithm ID and router policy ID;
- requested target size, maximum learned tokens, minimum pair frequency;
- realized learned-token and total logical vocabulary counts;
- deterministic tie-break, local merge ordering, and no-normalization policy;
- corpus input, file count, file list, raw byte count, FNV-1a identifier, and
  preprocessing statement;
- per-pack span count, routed bytes, learned merge count, and local vocabulary
  size, including the shared fallback row.

The benchmark record includes stronger SHA-256 source/corpus/artifact hashes and
the held-out train/evaluation distinction. Metadata never affects runtime
encoding.

## Runtime errors and bounds

Version 3 rejects an unknown router, unsupported or duplicate pack ID, empty
graph, noncanonical graph ordering, duplicate ordered merge pair, non-prior
local parent, invalid byte/tag, registry mismatch, special-token declaration,
trailing bytes, and expanded tokens larger than 1 MiB. A local reference can
only address an earlier rank in its own graph; the representation has no field
for a cross-pack reference. Artifacts are bounded to 16 MiB on both read and
write. The CLI and format limits are described in FORMAT.md.

The encoder checks router spans for exact coverage, ordered contiguity, valid
pack IDs, and UTF-8 boundaries. The decoder validates pack and local IDs before
building output and checks total decoded length. A byte sequence can be
represented by byte tokens or by a learned token; if identical bytes happen to
exist in multiple local namespaces, their token IDs are still distinct and
decode to the same bytes without ambiguity.

## Inspection and commands

The CLI exposes:

    packtok route "Hello 123!"
    packtok train-packs fixtures/benchmark/train.txt target/m2.packtok
    packtok encode --artifact target/m2.packtok "Sämtliche Häuser"
    packtok decode --artifact target/m2.packtok <pack:local IDs>
    packtok inspect-pack target/m2.packtok TEXT
    packtok inspect-token target/m2.packtok TEXT:0
    packtok inspect-merges target/m2.packtok TEXT
    packtok stats target/m2.packtok
    packtok validate target/m2.packtok

Token inspection prints bytes and escaped bytes even if the expansion is not
valid UTF-8, then displays the rank and parent symbols for learned tokens.

## Known limitations and non-goals

- lexical-v1 sends every non-ASCII scalar to TEXT and all ASCII spacing and
  punctuation to STRUCTURE. It may create short spans and prevent useful
  cross-boundary merges.
- Each pack learns only from its routed spans; an individual pack may receive
  few or no learned tokens under the global competition.
- Training repeatedly rescans the selected pack's sequences; this favors a
  small inspectable implementation over incremental update complexity.
- Temporary allocation counters describe vector capacities, not allocator
  metadata or process RSS. Whole-process memory is a separate benchmark
  observation.
- The corpus is a small synthetic fixture and is not a representative language
  dataset. Results do not establish general tokenizer quality.

M2 does not implement POS or morphology routing, language detection, learned or
neural routing, adaptive vocabularies, priority tokens, model embeddings, model
heads, transformer code, telemetry, or user/domain overlays.

## Next milestone

M3 is the first matched tiny-model comparison. It should consume the frozen M1
and M2 tokenizer artifacts and compare model-facing cost and validation results
under a documented, matched compute and parameter budget. M2's benchmark and
any negative results remain historical inputs to that decision.
