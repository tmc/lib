package main

import (
	"testing"
)

func BenchmarkLlamaInference(b *testing.B) {
	cfg := DefaultTinyLlamaConfig()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := runPlatform(cfg); err != nil {
			b.Fatalf("run failed: %v", err)
		}
	}
}
