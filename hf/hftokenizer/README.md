# Hugging Face byte-level BPE

`hftokenizer.New` reads `tokenizer.json` from an `io.Reader`. The tokenizer is
immutable and supports concurrent encoding and decoding. It has no runtime
module dependencies and does not fetch model files.

Supported pipelines use deterministic byte-level BPE with the GPT-2, Qwen2 or
Llama3 split pattern, no normalization or NFC, literal added tokens, and no
post-processor, ByteLevel processing, or a single-input TemplateProcessing
sequence. The file controls BOS insertion. Unknown pipeline options, padding,
truncation, stochastic merges, byte fallback, and added-token normalization,
word-boundary or stripping policies return errors.

`Decode` omits added tokens marked special and performs lossy UTF-8 decoding.
`DecodeBytes` returns the bytes before that conversion: a streaming caller must
retain incomplete UTF-8 across token boundaries. Unknown IDs return errors.
`MaxTokenID` permits checking the tokenizer against an embedding table; it is
not a count of contiguous vocabulary entries.

Unicode letter and number categories are pinned to version 16.0.0. NFC uses
version 9.0.0, as used by `unicode-normalization-alignments` 0.1.12 in Hugging Face
`tokenizers` 0.22.2. This intentionally differs from using the installed Go
version's Unicode data. The generated data comes from the Unicode Consortium;
input URLs and SHA-256 hashes are in `internal/genunicode/sources.json`.
`go generate` verifies those hashes before regenerating the tables.

Portable tests compare six synthetic pipelines with `tokenizers` 0.22.2.
Their generator and provenance are in `testdata`. Optional reference tests use
`HF_TOKENIZER_REFERENCE_DIR`, `HF_SPLITTER_REFERENCE`,
`HF_NORMALIZATION_REFERENCE_DIR`, and `HF_NORMALIZATION_TEST` for larger local
corpora. A pinned Qwen2.5-0.5B tokenizer has been checked with the first test;
the GPT-2 and Llama3 pipeline tests use synthetic vocabularies. This package
does not render chat templates or provide pair encodings, offsets, padding,
truncation or SentencePiece tokenization.
