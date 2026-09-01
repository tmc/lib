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

// Config defines hyperparameters for a LLaMA model.
type Config struct {
	Name       string
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
		Name:       "TinyLlama-Testing",
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

// Llama3_8BConfig returns configuration for standard LLaMA-3-8B.
func Llama3_8BConfig() Config {
	return Config{
		Name:       "LLaMA-3-8B",
		VocabSize:  128256,
		Dim:        4096,
		HiddenDim:  14336,
		NumLayers:  32,
		NumHeads:   32,
		NumKVHeads: 8,
		HeadDim:    128,
		MaxSeqLen:  8192,
		Eps:        1e-5,
		RopeTheta:  500000.0,
	}
}

// TinyLlama_1_1BConfig returns configuration for TinyLlama-1.1B.
func TinyLlama_1_1BConfig() Config {
	return Config{
		Name:       "TinyLlama-1.1B",
		VocabSize:  32000,
		Dim:        2048,
		HiddenDim:  5632,
		NumLayers:  22,
		NumHeads:   32,
		NumKVHeads: 4,
		HeadDim:    64,
		MaxSeqLen:  2048,
		Eps:        1e-5,
		RopeTheta:  10000.0,
	}
}

func main() {
	modelSize := flag.String("model", "tiny", "model size (tiny, 1b, 8b)")
	flag.Parse()

	var cfg Config
	switch *modelSize {
	case "1b", "tinyllama":
		cfg = TinyLlama_1_1BConfig()
	case "8b", "llama3":
		cfg = Llama3_8BConfig()
	default:
		cfg = DefaultTinyLlamaConfig()
	}

	fmt.Printf("=== Cross-Platform LLaMA Tensor Inference (%s on %s/%s) ===\n", cfg.Name, runtime.GOOS, runtime.GOARCH)
	fmt.Printf("Config: Dim=%d, HiddenDim=%d, Layers=%d, Heads=%d, KVHeads=%d, Vocab=%d\n\n",
		cfg.Dim, cfg.HiddenDim, cfg.NumLayers, cfg.NumHeads, cfg.NumKVHeads, cfg.VocabSize)

	if err := runPlatform(cfg); err != nil {
		log.Fatalf("llama-infer: %v", err)
	}
}
