package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestForwardPass(t *testing.T) {
	cfg := DefaultModelConfig()
	weights := GenerateDeterministicWeights(&cfg)
	headDim := cfg.HeadDim()
	kvDim := cfg.NumKeyValueHeads * headDim
	kvCache := NewKVCacheState(cfg.NumHiddenLayers, cfg.MaxSeqLen, kvDim)

	// Step 0 forward pass
	logits0 := Forward(weights, &cfg, kvCache, 42, 0)
	if len(logits0) != cfg.VocabSize {
		t.Fatalf("expected %d logits, got %d", cfg.VocabSize, len(logits0))
	}

	tok0 := sampleArgmax(logits0)
	if tok0 < 0 || int(tok0) >= cfg.VocabSize {
		t.Errorf("sampled token out of bounds: %d", tok0)
	}

	// Step 1 forward pass
	logits1 := Forward(weights, &cfg, kvCache, tok0, 1)
	if len(logits1) != cfg.VocabSize {
		t.Fatalf("expected %d logits, got %d", cfg.VocabSize, len(logits1))
	}

	tok1 := sampleArgmax(logits1)
	if tok1 < 0 || int(tok1) >= cfg.VocabSize {
		t.Errorf("sampled token out of bounds: %d", tok1)
	}
}

func TestByteTokenizer(t *testing.T) {
	tok := NewByteTokenizer()
	prompt := "Hello LLaMA 3.2"

	tokens, err := tok.Encode(prompt)
	if err != nil {
		t.Fatalf("Encode failed: %v", err)
	}

	decoded, err := tok.Decode(tokens)
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}

	if decoded != prompt {
		t.Errorf("expected %q, got %q", prompt, decoded)
	}
}

func TestHFTokenizerParsing(t *testing.T) {
	tmpDir := t.TempDir()
	tokFile := filepath.Join(tmpDir, "tokenizer.json")

	jsonContent := `{
		"model": {
			"vocab": {
				"Hello": 100,
				" world": 101,
				"!": 102
			},
			"merges": ["H ello"]
		},
		"added_tokens": [
			{"id": 0, "content": "<unk>"},
			{"id": 1, "content": "<s>"}
		]
	}`
	if err := os.WriteFile(tokFile, []byte(jsonContent), 0644); err != nil {
		t.Fatalf("write tokenizer.json: %v", err)
	}

	tok, err := LoadHFTokenizer(tokFile)
	if err != nil {
		t.Fatalf("LoadHFTokenizer failed: %v", err)
	}

	if tok.VocabSize() < 5 {
		t.Errorf("expected at least 5 tokens in vocab, got %d", tok.VocabSize())
	}

	decoded, err := tok.Decode([]int32{100, 101, 102})
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}
	if decoded != "Hello world!" {
		t.Errorf("expected 'Hello world!', got %q", decoded)
	}
}

func BenchmarkByteTokenizerEncode(b *testing.B) {
	tok := NewByteTokenizer()
	prompt := "The quick brown fox jumps over the lazy dog. Antigravity inference engine in pure Go."
	b.SetBytes(int64(len(prompt)))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, _ = tok.Encode(prompt)
	}
}

func BenchmarkByteTokenizerDecode(b *testing.B) {
	tok := NewByteTokenizer()
	tokens := []int32{84, 104, 101, 32, 113, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110}
	b.SetBytes(int64(len(tokens)))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, _ = tok.Decode(tokens)
	}
}

func BenchmarkModelForwardStep(b *testing.B) {
	cfg := DefaultModelConfig()
	weights := GenerateDeterministicWeights(&cfg)
	headDim := cfg.HeadDim()
	kvDim := cfg.NumKeyValueHeads * headDim
	kvCache := NewKVCacheState(cfg.NumHiddenLayers, cfg.MaxSeqLen, kvDim)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Forward(weights, &cfg, kvCache, 42, i%cfg.MaxSeqLen)
	}
}
