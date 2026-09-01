package main

import (
	"testing"
)

func BenchmarkLlamaPrefill(b *testing.B) {
	cfg := DefaultTinyLlamaConfig()
	promptTokens := 32

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		dur, err := PrefillStep(cfg, promptTokens)
		if err != nil {
			b.Fatalf("prefill failed: %v", err)
		}
		if dur > 0 {
			tokPerSec := float64(promptTokens) / dur.Seconds()
			b.ReportMetric(tokPerSec, "tok/s_prefill")
		}
	}
}

func BenchmarkLlamaDecode(b *testing.B) {
	cfg := DefaultTinyLlamaConfig()
	decodeTokens := 16

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		dur, err := DecodeStep(cfg, decodeTokens)
		if err != nil {
			b.Fatalf("decode failed: %v", err)
		}
		if dur > 0 {
			tokPerSec := float64(decodeTokens) / dur.Seconds()
			b.ReportMetric(tokPerSec, "tok/s_decode")
		}
	}
}

func BenchmarkLlamaInference(b *testing.B) {
	cfg := DefaultTinyLlamaConfig()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := runPlatform(cfg); err != nil {
			b.Fatalf("run failed: %v", err)
		}
	}
}

func BenchmarkLlama3_8B_Decode(b *testing.B) {
	cfg := Llama3_8BConfig()
	decodeTokens := 8

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		dur, err := DecodeStep(cfg, decodeTokens)
		if err != nil {
			b.Fatalf("decode failed: %v", err)
		}
		if dur > 0 {
			tokPerSec := float64(decodeTokens) / dur.Seconds()
			b.ReportMetric(tokPerSec, "tok/s_8b_decode")
		}
	}
}

func BenchmarkTinyLlama1B_Decode(b *testing.B) {
	cfg := TinyLlama_1_1BConfig()
	decodeTokens := 16

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		dur, err := DecodeStep(cfg, decodeTokens)
		if err != nil {
			b.Fatalf("decode failed: %v", err)
		}
		if dur > 0 {
			tokPerSec := float64(decodeTokens) / dur.Seconds()
			b.ReportMetric(tokPerSec, "tok/s_1b_decode")
		}
	}
}
