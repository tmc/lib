package main

import (
	"testing"
	"time"
)

func BenchmarkLlama3_2_1B_EndToEnd_Metal(b *testing.B) {
	path := findDefaultLlama3_2_1BPath()
	if path == "" {
		b.Skip("Llama 3.2 1B weights not found in local cache")
	}

	cfg := Llama3_2_1BConfig(path)
	cfg.FuseEGraph = true

	b.Run("Prompt_Prefill_7_Tokens", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		start := time.Now()
		for i := 0; i < b.N; i++ {
			dur, err := PrefillStep(cfg, 7)
			if err != nil {
				b.Fatalf("prefill failed: %v", err)
			}
			_ = dur
		}
		totalDur := time.Since(start)
		if totalDur > 0 {
			tokPerSec := float64(b.N*7) / totalDur.Seconds()
			b.ReportMetric(tokPerSec, "tok/s_prefill")
		}
	})

	b.Run("Autoregressive_Decode_10_Tokens", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		start := time.Now()
		for i := 0; i < b.N; i++ {
			dur, err := DecodeStep(cfg, 10)
			if err != nil {
				b.Fatalf("decode failed: %v", err)
			}
			_ = dur
		}
		totalDur := time.Since(start)
		if totalDur > 0 {
			tokPerSec := float64(b.N*10) / totalDur.Seconds()
			b.ReportMetric(tokPerSec, "tok/s_decode")
		}
	})
}
