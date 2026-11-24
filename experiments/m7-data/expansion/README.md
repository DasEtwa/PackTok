# M7 corpus expansion and L4 follow-up

## Licensed historical sources

| Source | Release and acquisition | Rights and handling |
|---|---|---|
| Deutsches Textarchiv Kernkorpus | BBAW archive dated 2020-07-20, acquired 2026-11-24. The transfer was 376,206,131 bytes. The ZIP lists 1,472 XML members. | The collection lists CC BY-SA 4.0. The importer admits a member only if its TEI header carries that license. Preserve the title, original author, archive member ID, BBAW credit, license link and changes. If sharing adapted text, apply the license's ShareAlike condition. [Collection and license](https://registry.text-plus.org/doc/collection/c582ca78-26b4-4e5a-b817-e2d7c04f0a9d), [CC BY-SA 4.0 legal code](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en) |
| EEBO-TCP, Phase I | University of Oxford's Version 1, TEI P3 2003 tranche, acquired 2026-11-24. The transfer was 321,132,665 bytes. The ZIP lists 3,119 XML members. | The publisher released the Phase I transcriptions to public-domain reuse in 2015. This intake covers the transcriptions and preserves credit to the original authors, Michigan/Oxford TCP and the archived edition. Page images have separate rights and are outside this source. [Oxford release](https://ora.ox.ac.uk/objects/uuid%3Aad7da8fc-cd8e-4637-8b7c-99498436dbaa), [TCP rights statement](https://www.textpartnership.net/pages/faq.html) |

Use the exact artifact lengths and observed SHA-256 values in [acquisition.json](acquisition.json). The shipped files from both institutions contained 2,244,257,355 uncompressed bytes of XML. That number includes markup and metadata. The raw network payloads and extracted data stay outside Git. WSL fetched the DTA archive directly. Windows curl fetched Oxford's archive after WSL was unable to resolve its hostname. Integrity, import, dataset construction and verification run in Ubuntu-24.04. The machine-readable source plan requires known publisher file URLs and SHA-256 before it opens an archive.

The importer reads bounded ZIP members and extracts the first TEI `text` outside the header. Grouped editions can contain subsequent text elements. Extraction removes editorial notes, figures and running heads and inserts structural newlines, including at line and page breaks. It handles XML character references and the five predefined named entities. Unknown names, invalid XML encountered during extraction, over-limit members and missing DTA rights are rejected. The metadata gate recognizes `de`, `deu` and `ger` for DTA and `en` and `eng` for EEBO. The ledger records exclusions. A failed ZIP index read aborts publication, and each archive must match the pinned length and SHA-256 before and after extraction. Source title, author, original archive and member SHA-256, output SHA-256 and byte lengths go into the generated member ledger. Output paths use generated IDs. The existing pipeline applies fixed language and quality rules and exact, near and passage deduplication. Output is UTF-8 text in the existing verified JSONL shards. Ingest uses quick-xml 0.38.4 and ZIP/Deflate in the isolated workspace to process structured releases. [Independent review](REVIEW.md).

Assignment uses a fixed hash of the authority ID or normalized author string, with an unattributed-title fallback. The target buckets are 80/10/10 for train, validation and test before filtering. The same group always selects the same split. The design covers historical German and English across authors and publishers. Edition conventions and authority-name variation can still limit author and work linkage. Assignment uses source metadata and no model scores.

## Local reproduction

Run from the directory that contains the selected archives. On the benchmark machine the original archives are in `~/datasets/packtok/m7-historical-en-de-20261124/raw`. Use fresh native output paths.

```sh
packtok-m7 import-archives /path/to/acquisition.json prepared
packtok-m7 build prepared/catalog.json prepared/cache dataset
packtok-m7 verify dataset
```

The original test and license state of the LIV Math dataset remains recorded in the [source ledger](../SOURCES.md). Harvard Institutional Books early access is excluded due to its noncommercial, redistribution and jurisdiction terms. [Publisher terms](https://huggingface.co/datasets/institutional/institutional-books-hl/blob/main/README.md).

## Published corpus and held-out split

The verified dataset is stored on the Ubuntu-24.04 filesystem under `~/datasets/packtok/m7-historical-en-de-20261124/dataset`. The Ubuntu-24.04 VHDX is on D:. The member ledger beside the dataset preserves archive-level and document-level provenance. The checked-in receipts describe the release without putting corpus text in Git. [Dataset receipt](dataset-receipt.json), [published verification](published-verify.log).

| Split | Documents | UTF-8 text bytes |
|---|---:|---:|
| Train | 3,743 | 1,319,820,159 |
| Validation | 269 | 59,568,609 |
| Test | 225 | 71,397,630 |
| Total | 4,237 | 1,450,786,398 |

The importer extracted 4,569 documents totaling 1,614,606,812 UTF-8 bytes. The build excluded 332 documents. Matched cross-split passages accounted for 318 exclusions and 152,923,107 bytes. Passage matching can also detect common material such as long quotations. Validation retained 39,119,266 German bytes and 20,449,343 English bytes. The older M7 validation used 613,903 bytes from one author. Reported bytes count text before JSONL escaping. The 24 published shard files occupy 1,491,465,873 bytes. [Import summary](import-summary.json), [before-filter profile](pre-filter-split.json), [final split profile](split-profile.json), [validation index](validation-index.jsonl).

This is an evaluation set for historical print. It covers both languages and both collections, and results obtained on it apply to this mixture. DTA's source genre labels are preserved where available. The English archive gets the broader historical-print label. The initial M7 training probe still scores prefixes from at most eight documents. Use a separately fixed scoring protocol to measure the enlarged set. The test split stays sealed for model evaluation.

## Author separation for evaluation

The follow-up independent review detected that authority IDs and name fallbacks can represent the same person with different group IDs. The profile checks exact normalized author strings separately from group IDs. Unicode NFKC, case folding and punctuation delimiters found one candidate shared by train and validation and one shared by validation and test. The separate `dataset-eval-v1` excludes the two corresponding validation records. This view has 267 documents with 58,502,073 UTF-8 bytes (108 German and 159 English documents), about 95 times the validation text of the older M7 probe. It reuses the immutable train and test shards by hard link on D: and has its own manifest and verification. Author aliasing and unnamed or secondary authors still limit the check. [Author audit](author-overlap.json), [evaluation manifest receipt](evaluation-receipt.json).

To derive the evaluation view, execute the following commands from a native Ubuntu-24.04 directory with enough space for the validation shard. Keep the verified base and view immutable.

```sh
python3 derive-evaluation.py "$base" author-overlap.json dataset-eval-v1.pending
packtok-m7 verify dataset-eval-v1.pending
mv -T -n dataset-eval-v1.pending dataset-eval-v1
```

## Local disk bottleneck

The historical build took 37 min 37 s with 86,576 KiB peak RSS on the i7-2600 under Ubuntu-24.04. Its process was CPU-active for 49% of wall time. The disk-backed deduplication index reached 1,077,940,224 bytes. A sample during validation recorded 24,348,430,336 bytes of cumulative writes attributed to the storage layer for the process and roughly 90% I/O pressure on WSL over the preceding minute. The storage-layer count is an OS measurement and includes writes from the build beyond its final output. Final verification took 21.37 s with 40,536 KiB peak RSS. [Build log](dataset-build.log), [I/O sample](build-io-sample.json), [published verification](published-verify.log).

Windows identified C: as the SanDisk SSD PLUS 120 GB and D: as the WDC WD20EZAZ HDD. C: reported 19,895,902,208 available bytes at inspection, below the 20,000,000,000-byte reserve. The benchmark used the native WSL filesystem backed by D: for all inputs and outputs. Only the deduplication index moved to `/dev/shm` in the RAM runs. [Storage pilot and reproducibility](STORAGE.md#scratch-index-pilot).

After verifying the original archives, both corpus views and all four pilot outputs, the run removed its extracted cache, incomplete re-import and pilot outputs. The durable archives and corpus views are under the same release directory on D:. [Cleanup record](temp-cleanup.txt).

## L4 data clearance and budget

The authorized follow-up has one NVIDIA L4, up to 1,200 seconds and a 0.60 CU envelope. The remote package uses six purpose-written bilingual fixtures under PackTok's Apache-2.0 license, solely to exercise the training and resume path. The two train texts are repeated within their documents to satisfy the existing 1 KiB tokenizer sample minimum. Author-generated fixture rights cover the remote use. The historical corpora remain local. The fixture catalog and exact executable/package hashes are retained. Local training and checkpoint restore passed in the CPU rehearsal. [Fixture](../gpu-fixture/documents.tsv), [license](../../../LICENSE), [package identity](l4-ready.json), [source catalog](l4-source-catalog.json).

The first pre-allocation Colab inventory check encountered an OAuth authorization prompt. There were zero allocation requests. An authorization prompt is an account-access blocker, and the preflight output is recorded by SHA-256 with sensitive fields omitted. [Preflight record](l4-auth-preflight.json).

The next inventory check succeeded. The single authorized allocation then failed with `Service Unavailable` before the runtime became ready. The allocation command took 31.563 seconds, and the final inventory and usage checks finished 36.031 seconds after the request. Colab reported zero active assignments and 235.37 CU before and after. No remote worker ran. The displayed balance delta is zero at CLI precision. [Attempt receipt](l4-attempt.json), [lifecycle accounting](l4-accounting.json), [stages](l4-stages.jsonl).

[Storage audit](STORAGE.md)
