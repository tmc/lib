// Package hftokenizer reads deterministic byte-level BPE tokenizer.json files.
// It supports GPT-2, Qwen2 and Llama3 pre-tokenization, optional NFC normalization,
// literal added tokens and single-input template processing. Unicode categories
// are pinned to 16.0.0 and NFC data to 9.0.0, matching tokenizers 0.22.2. Unsupported pipeline
// options return errors. It does not render chat templates or tokenize input pairs.
//
// Encode applies the file's post-processor, including any declared BOS tokens.
// Decode and DecodeBytes omit added tokens marked special. No token IDs or
// vocabulary conventions are inferred from model names.
package hftokenizer
