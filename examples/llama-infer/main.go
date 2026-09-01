// Command llama-infer demonstrates an end-to-end lightweight AOT static-graph tensor
// inference runtime for LLaMA architectures, automatically selecting either the Apple
// Silicon Metal backend (on darwin/arm64) or the NVIDIA CUDA backend with CPU oracle fallback.
package main

import (
	"flag"
	"fmt"
	"log"
	"runtime"
)

// Config defines hyperparameters for a compact LLaMA model.
type Config struct {
	VocabSize  int
	Dim        int
	HiddenDim  int
	NumLayers  int
	NumHeads   int
	NumKVHeads int
	HeadDim    int
	MaxSeqLen  int
	Eps        float32
	RopeTheta  float32
}

// DefaultTinyLlamaConfig returns configuration for a tiny testing model.
func DefaultTinyLlamaConfig() Config {
	return Config{
		VocabSize:  256,
		Dim:        64,
		HiddenDim:  128,
		NumLayers:  2,
		NumHeads:   4,
		NumKVHeads: 2,
		HeadDim:    16,
		MaxSeqLen:  32,
		Eps:        1e-5,
		RopeTheta:  10000.0,
	}
}

func main() {
	flag.Parse()
	cfg := DefaultTinyLlamaConfig()

	fmt.Printf("=== Cross-Platform LLaMA Tensor Inference (%s/%s) ===\n", runtime.GOOS, runtime.GOARCH)
	fmt.Printf("Config: Dim=%d, HiddenDim=%d, Layers=%d, Heads=%d, KVHeads=%d, Vocab=%d\n\n",
		cfg.Dim, cfg.HiddenDim, cfg.NumLayers, cfg.NumHeads, cfg.NumKVHeads, cfg.VocabSize)

	if err := runPlatform(cfg); err != nil {
		log.Fatalf("llama-infer: %v", err)
	}
}
