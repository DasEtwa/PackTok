PackTok

Experimental factorized tokenization for language models.

PackTok explores a simple idea:

Instead of placing every token inside one flat vocabulary, split the vocabulary into specialized packs and let the model predict both the pack and the token inside that pack.

A traditional tokenizer may use a vocabulary like:

2649  -> "beautiful"
3651  -> "house"
7821  -> "running"

PackTok can represent the same idea more like:

ADJ:2649   -> "beautiful"
NOUN:3651  -> "house"
VERB:7821  -> "running"

Internally, a token can be treated as:

(pack_id, local_token_id)

Instead of one large global token space.

⸻

Why?

Modern subword tokenizers such as BPE are very good at compression and general-purpose text handling.

But they usually treat the vocabulary as one large shared pool.

That means:

* nouns compete with punctuation
* German morphology competes with English words
* programming tokens compete with normal language
* numbers compete with everything else
* rare but structurally useful fragments may lose vocabulary space

PackTok asks whether splitting that space into specialized vocabularies can help.

For example:

NOUN
VERB
ADJECTIVE
FUNCTION
NUMBER
PUNCTUATION
CODE
OTHER
BYTE

Each pack can maintain its own local vocabulary.

⸻

Prediction

A normal language model roughly predicts:

context
   ↓
logits over entire vocabulary
   ↓
next token

PackTok can instead use:

context
   ↓
pack prediction
   ↓
local token prediction
   ↓
next token

For example:

"The building is very ..."
ADJECTIVE   0.81
NOUN        0.08
ADVERB      0.05
OTHER       0.06

Then:

ADJECTIVE pack
large       0.21
beautiful   0.17
old         0.11
...

Another option is to combine both scores:

token_score =
    pack_score
    +
    local_token_score

This avoids turning pack prediction into a hard routing decision.

⸻

Packs are not limited to parts of speech

POS-based packs are only the first experiment.

Possible pack strategies include:

NOUN
VERB
ADJECTIVE
ADVERB

or:

ROOT
PREFIX
SUFFIX
INFLECTION

or:

GERMAN
ENGLISH
CODE
MATH
RAW_BYTES

or combinations of several factors.

Eventually, packs could also be learned automatically instead of being defined manually.

⸻

Goals

PackTok is currently a research experiment.

The first goal is not to prove that packed vocabularies are better.

The goal is to test them properly.

Initial experiments should compare PackTok against a normal BPE baseline using:

* the same corpus
* the same model architecture
* comparable parameter budgets
* comparable training compute
* identical evaluation sets

Metrics should include:

* validation loss
* bytes per token
* sequence length
* training efficiency
* inference cost
* vocabulary memory
* unknown-word/generalization behavior
* multilingual performance
* morphology
* code
* numbers and structured data

⸻

Planned first experiment

Baseline

16K byte-level BPE vocabulary

PackTok v0.1

Example:

NOUN
VERB
ADJECTIVE
FUNCTION
NUMBER
PUNCTUATION
CODE
OTHER
BYTE_FALLBACK

Both variants will be trained and evaluated on the same dataset with the same small language model.

If PackTok loses, that result is useful.

If it wins somewhere, we investigate why.

⸻

Important

PackTok is not intended to claim that linguistic categories are always the correct way to tokenize language.

Natural language is messy.

Words can change function depending on context, token boundaries do not always match morphemes, and different languages behave very differently.

That is exactly why this project exists:

test the idea instead of assuming it works.

⸻

Status

Very early experimental stage.

Expect:

* breaking changes
* strange experiments
* failed ideas
* benchmark scripts
* tokenizer prototypes
* tiny language models
* lots of comparisons

⸻

License

PackTok is licensed under the Apache License 2.0.

See LICENSE for details.
