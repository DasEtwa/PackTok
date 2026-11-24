# M7 corpus expansion and L4 follow-up

## Licensed historical sources

| Source | Release and acquisition | Rights and handling |
|---|---|---|
| Deutsches Textarchiv Kernkorpus | BBAW archive dated 2020-07-20, acquired 2026-11-24. The transfer was 376,206,131 bytes. The ZIP lists 1,472 XML members. | The collection lists CC BY-SA 4.0. The importer admits a member only if its TEI header carries that license. Preserve the title, original author, archive member ID, BBAW credit, license link and changes. If sharing adapted text, apply the license's ShareAlike condition. [Collection and license](https://registry.text-plus.org/doc/collection/c582ca78-26b4-4e5a-b817-e2d7c04f0a9d), [CC BY-SA 4.0 legal code](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en) |
| EEBO-TCP, Phase I | University of Oxford's Version 1, TEI P3 2003 tranche, acquired 2026-11-24. The transfer was 321,132,665 bytes. The ZIP lists 3,119 XML members. | The publisher released the Phase I transcriptions to public-domain reuse in 2015. This intake covers the transcriptions and preserves credit to the original authors, Michigan/Oxford TCP and the archived edition. Page images have separate rights and are outside this source. [Oxford release](https://ora.ox.ac.uk/objects/uuid%3Aad7da8fc-cd8e-4637-8b7c-99498436dbaa), [TCP rights statement](https://www.textpartnership.net/pages/faq.html) |

Use the exact artifact lengths and observed SHA-256 values in [acquisition.json](acquisition.json). The shipped files from both institutions contained 2,244,257,355 uncompressed bytes of XML. That number includes markup and metadata. The raw network payloads and extracted data stay outside Git. WSL fetched the DTA archive directly. Windows curl fetched Oxford's archive after WSL was unable to resolve its hostname. Integrity, import, dataset construction and verification run in Ubuntu-24.04. The machine-readable source plan requires known publisher file URLs and SHA-256 before it opens an archive.

The importer reads bounded ZIP members, decodes TEI body text, removes editorial notes, figures and running heads, and preserves structural newlines. Unsupported entities, invalid XML, over-limit members and missing DTA rights are rejected with reasons. Source title, author, original archive/member SHA-256, output SHA-256 and byte lengths go into the generated member ledger. No files are written to archive-supplied paths. The existing pipeline then applies fixed language/quality rules and exact, near and passage deduplication. Output is UTF-8 text in the existing verified JSONL shards. Ingest uses quick-xml 0.38.4 and ZIP/Deflate in the isolated workspace. XML and archive processing could not be covered by the pre-existing plain-text and JSONL readers.

Assignment uses a fixed hash of the authority ID or normalized author string, with an unattributed-title fallback. The declared buckets are 80/10/10 for train, validation and test before filtering. The same group always selects the same split. This is intended to increase historical-language coverage across authors and sources. It excludes modern text and has no inference claim about the broader LIV domains. Source authorship variants may need further authority linking. No model outcomes entered the split assignment.

## Local reproduction

Run from the directory that contains the selected archives. Use a fresh native output path.

```sh
packtok-m7 import-archives /path/to/acquisition.json prepared
packtok-m7 build prepared/catalog.json prepared/cache dataset
packtok-m7 verify dataset
```

The original test and license state of the LIV Math dataset remains recorded in the [source ledger](../SOURCES.md). Harvard Institutional Books early access is excluded due to its noncommercial, redistribution and jurisdiction terms. [Publisher terms](https://huggingface.co/datasets/institutional/institutional-books-hl/blob/main/README.md).

## L4 data clearance and budget

The authorized follow-up has one NVIDIA L4, up to 1,200 seconds and a 0.60 CU envelope. The remote package uses six purpose-written bilingual fixtures under PackTok's Apache-2.0 license, solely to exercise the training and resume path. The two train texts are repeated within their documents to satisfy the existing 1 KiB tokenizer sample minimum. Author-generated fixture rights cover the remote use. The historical corpora remain local. The fixture catalog and exact executable/package hashes are retained. Local training and checkpoint restore passed in the CPU rehearsal. [Fixture](../gpu-fixture/documents.tsv), [license](../../../LICENSE), [package identity](l4-ready.json), [source catalog](l4-source-catalog.json).

The first pre-allocation Colab inventory check encountered an OAuth authorization prompt. There were zero allocation requests. An authorization prompt is an account-access blocker, and the preflight output is recorded by SHA-256 with sensitive fields omitted. [Preflight record](l4-auth-preflight.json).

The next inventory check succeeded. The single authorized allocation then failed with `Service Unavailable` before the runtime became ready. The allocation command took 31.563 seconds, and the final inventory and usage checks finished 36.031 seconds after the request. Colab reported zero active assignments and 235.37 CU before and after. No remote worker ran. The displayed balance delta is zero at CLI precision. [Attempt receipt](l4-attempt.json), [lifecycle accounting](l4-accounting.json), [stages](l4-stages.jsonl).

[Storage audit](STORAGE.md)
