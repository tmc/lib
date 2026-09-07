//go:build linux && !tpu

package main

import "testing"

func TestCUDAEngineCacheKey(t *testing.T) {
	base := Llama3_2_1BConfig("model")
	tests := []struct {
		name string
		edit func(*Config)
	}{
		{"maximum sequence length", func(cfg *Config) { cfg.MaxSeqLen /= 2 }},
		{"model path", func(cfg *Config) { cfg.ModelPath = "other" }},
		{"precision", func(cfg *Config) { cfg.Precision = "f16" }},
		{"fusion", func(cfg *Config) { cfg.FuseEGraph = true }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			changed := base
			test.edit(&changed)
			if cudaEngineCacheKey(base) == cudaEngineCacheKey(changed) {
				t.Fatal("cache key aliases distinct engine configurations")
			}
		})
	}
}

func TestCUDAEngineCacheKeyIgnoresRequestSettings(t *testing.T) {
	a := Llama3_2_1BConfig("model")
	b := a
	b.Prompt = "different request"
	b.GenTokens++
	if cudaEngineCacheKey(a) != cudaEngineCacheKey(b) {
		t.Fatal("cache key includes request-only settings")
	}
}
