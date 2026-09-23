//go:build darwin

package main

import (
	"testing"
	"time"
)

// benchPromptTokens is a fixed seven-token prompt. Prefill cost depends on the
// token count, not on which tokens they are.
var benchPromptTokens = []int32{128000, 578, 1401, 311, 69311, 15592, 78335}

// benchDecodePos is the KV cache position the decode benchmark measures at.
// Decode cost grows with the length of the cached prefix, so the position is
// held fixed rather than tied to the iteration counter: b.N is unbounded and
// would run the attention kernel past the end of the KV cache.
const benchDecodePos = 128

func BenchmarkLlama3_2_1B_EndToEnd_Metal(b *testing.B) {
	path := findDefaultLlama3_2_1BPath()
	if path == "" {
		b.Skip("Llama 3.2 1B weights not found in local cache")
	}

	cfg := Llama3_2_1BConfig(path)
	cfg.FuseEGraph = true

	b.Run("Prompt_Prefill", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		start := time.Now()
		for i := 0; i < b.N; i++ {
			if _, err := PrefillStep(cfg, benchPromptTokens); err != nil {
				b.Fatalf("prefill failed: %v", err)
			}
		}
		if d := time.Since(start); d > 0 {
			b.ReportMetric(float64(b.N*len(benchPromptTokens))/d.Seconds(), "tok/s_prefill")
		}
	})

	b.Run("Autoregressive_Decode", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		start := time.Now()
		for i := 0; i < b.N; i++ {
			// One forward pass produces one token.
			if _, err := DecodeStep(cfg, 10, benchDecodePos); err != nil {
				b.Fatalf("decode failed: %v", err)
			}
		}
		if d := time.Since(start); d > 0 {
			b.ReportMetric(float64(b.N)/d.Seconds(), "tok/s_decode")
		}
	})
}
