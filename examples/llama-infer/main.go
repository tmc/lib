// Command llama-infer demonstrates an end-to-end lightweight AOT static-graph tensor
// inference runtime for LLaMA architectures, automatically selecting either the Apple
// Silicon Metal backend (on darwin/arm64) or the NVIDIA CUDA backend with CPU oracle fallback.
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"
)

// Config defines hyperparameters for a LLaMA model.
type Config struct {
	Name       string
	ModelPath  string
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

// Llama3_2_1BConfig returns configuration for official LLaMA-3.2-1B.
func Llama3_2_1BConfig(modelPath string) Config {
	return Config{
		Name:       "LLaMA-3.2-1B-Instruct",
		ModelPath:  modelPath,
		VocabSize:  128256,
		Dim:        2048,
		HiddenDim:  8192,
		NumLayers:  16,
		NumHeads:   32,
		NumKVHeads: 8,
		HeadDim:    64,
		MaxSeqLen:  8192,
		Eps:        1e-5,
		RopeTheta:  500000.0,
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

func findDefaultLlama3_2_1BPath() string {
	home, _ := os.UserHomeDir()
	pattern := filepath.Join(home, ".cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct/snapshots/*")
	matches, _ := filepath.Glob(pattern)
	if len(matches) > 0 {
		return matches[0]
	}
	return ""
}

func main() {
	modelSize := flag.String("model", "llama3.2-1b", "model size (tiny, llama3.2-1b, 1b, 8b)")
	modelPath := flag.String("path", "", "path to model directory containing model.safetensors")
	prompt := flag.String("prompt", "The key to scalable AI architectures is", "prompt text for inference")
	flag.Parse()

	var cfg Config
	switch *modelSize {
	case "llama3.2-1b", "llama3.2", "3.2-1b":
		path := *modelPath
		if path == "" {
			path = findDefaultLlama3_2_1BPath()
		}
		cfg = Llama3_2_1BConfig(path)
	case "1b", "tinyllama":
		cfg = TinyLlama_1_1BConfig()
	case "8b", "llama3":
		cfg = Llama3_8BConfig()
	default:
		cfg = DefaultTinyLlamaConfig()
	}

	fmt.Printf("=== Cross-Platform LLaMA Tensor Inference (%s on %s/%s) ===\n", cfg.Name, runtime.GOOS, runtime.GOARCH)
	fmt.Printf("Architecture: Dim=%d, HiddenDim=%d, Layers=%d, Heads=%d, KVHeads=%d, Vocab=%d\n",
		cfg.Dim, cfg.HiddenDim, cfg.NumLayers, cfg.NumHeads, cfg.NumKVHeads, cfg.VocabSize)
	if cfg.ModelPath != "" {
		fmt.Printf("Safetensors Path: %s\n", cfg.ModelPath)
	}
	fmt.Printf("Prompt: %q\n\n", *prompt)

	if err := runPlatform(cfg); err != nil {
		log.Fatalf("llama-infer: %v", err)
	}
}
