package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
)

func findDefaultQwenModel() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}

	patterns := []string{
		// 1. Qwen3.8-27B
		filepath.Join(home, ".cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/*"),
		// 2. Qwen3.8-27B-FP8
		filepath.Join(home, ".cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/*"),
		// 3. Qwen3-0.6B / Qwen3-1.7B / Qwen3-4B / Qwen3-8B
		filepath.Join(home, ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*"),
		filepath.Join(home, ".cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/*"),
		filepath.Join(home, ".cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/*"),
		filepath.Join(home, ".cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*"),
		// 4. Qwen2.5-0.5B-Instruct
		filepath.Join(home, ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/*"),
		// 5. MLX / local models directory
		filepath.Join(home, "ml-explore/models/Qwen3.8-27B-4bit"),
	}

	for _, p := range patterns {
		matches, _ := filepath.Glob(p)
		for _, m := range matches {
			if _, err := os.Stat(filepath.Join(m, "config.json")); err == nil {
				return m
			}
		}
	}
	return ""
}

func main() {
	modelDirFlag := flag.String("model", "", "Path to Qwen model directory containing config.json & safetensors")
	promptFlag := flag.String("prompt", "The capital of France is", "Input prompt text")
	stepsFlag := flag.Int("steps", 5, "Number of tokens to generate")
	flag.Parse()

	modelDir := *modelDirFlag
	if modelDir == "" {
		modelDir = findDefaultQwenModel()
	}

	fmt.Println("=== Pure-Go E-Graph Qwen Inference Engine ===")

	// 1. Load Architecture Config
	var cfg *QwenConfig
	var err error
	if modelDir != "" {
		cfg, err = LoadConfig(filepath.Join(modelDir, "config.json"))
		if err != nil {
			fmt.Printf("[Config Notice] Failed reading %s (%v), using default Qwen config\n", modelDir, err)
		} else {
			fmt.Printf("[Model Config] Loaded architecture from %s (layers: %d, hidden: %d, heads: %d, kv_heads: %d)\n",
				modelDir, cfg.NumHiddenLayers, cfg.HiddenSize, cfg.NumAttentionHeads, cfg.NumKeyValueHeads)
		}
	}
	if cfg == nil {
		defaultCfg := DefaultQwenConfig()
		cfg = &defaultCfg
		fmt.Printf("[Model Config] Using synthetic Qwen config (layers: %d, hidden: %d, heads: %d)\n",
			cfg.NumHiddenLayers, cfg.HiddenSize, cfg.NumAttentionHeads)
	}

	// 2. Load Tokenizer
	var tokenizer Tokenizer
	if modelDir != "" {
		tokFile := filepath.Join(modelDir, "tokenizer.json")
		if tok, err := LoadHFTokenizer(tokFile); err == nil {
			tokenizer = tok
			fmt.Printf("[Tokenizer] Loaded HuggingFace BPE tokenizer (vocab: %d)\n", tok.VocabSize())
		}
	}
	if tokenizer == nil {
		tokenizer = NewByteTokenizer()
		fmt.Println("[Tokenizer] Initialized Byte-level fallback tokenizer")
	}


	// 3. Initialize CUDA Device and E-Graph Optimization
	if err := driver.Init(0); err != nil {
		fmt.Printf("[GPU Error] Failed initializing CUDA driver: %v\n", err)
		return
	}

	dev, err := cuda.NewDevice(0)
	if err != nil {
		fmt.Printf("[GPU Error] Failed acquiring CUDA device: %v\n", err)
		return
	}

	name, _ := dev.Name()
	fmt.Printf("[Execution Target] Initialized %s (100%% GPU-Resident)\n", name)

	// Run E-Graph Equality Saturation & JIT Compilation
	engine, err := OptimizeAndCompileQwen(dev, cfg)
	if err != nil {
		fmt.Printf("[E-Graph Error] Failed optimizing & compiling megakernels: %v\n", err)
		return
	}
	fmt.Printf("[E-Graph Optimizer] Synthesized optimal Megakernel plan:\n  %s\n\n", engine.planSummary)

	// 4. Load Weights onto GPU
	var weights *GPUModelWeights
	if modelDir != "" {
		w, err := LoadGPUModel(dev, modelDir, cfg)
		if err != nil {
			fmt.Printf("[Model Notice] Failed loading safetensors from %s (%v)\n", modelDir, err)
		} else {
			weights = w
			fmt.Println("[Model Weights] Real weights successfully loaded into GPU VRAM")
		}
	}
	if weights == nil {
		weights, _ = LoadGPUModel(dev, "", cfg)
	}
	defer weights.Free()

	// 5. Initialize Runtime
	runtime, err := NewQwenInferenceRuntime(dev, cfg, engine, weights)
	if err != nil {
		fmt.Printf("[Runtime Error] Failed initializing runtime: %v\n", err)
		return
	}
	defer runtime.Close()

	// 6. Tokenize Prompt
	var promptTokens []int32
	if tokenizer != nil {
		promptTokens, _ = tokenizer.Encode(*promptFlag)
	}
	if len(promptTokens) == 0 {
		promptTokens = []int32{151643, 84, 104, 101}
	}

	fmt.Printf("Prompt    : %q\n", *promptFlag)
	fmt.Printf("Tokens    : %v\n", promptTokens)
	fmt.Printf("Generating: %d tokens\n\n", *stepsFlag)

	// 7. Prefill Phase
	t0 := time.Now()
	var nextTok int32
	for pos, tokID := range promptTokens {
		nextTok, err = runtime.ForwardToken(tokID, pos)
		if err != nil {
			fmt.Printf("[Forward Error] pos %d: %v\n", pos, err)
			return
		}
	}
	prefillElapsed := time.Since(t0)

	// 8. Decode Phase
	fmt.Printf("Streaming : %s", *promptFlag)
	t1 := time.Now()
	var generatedTokens []int32
	pos := len(promptTokens)

	for s := 0; s < *stepsFlag; s++ {
		generatedTokens = append(generatedTokens, nextTok)
		if tokenizer != nil {
			str, _ := tokenizer.Decode([]int32{nextTok})
			fmt.Print(str)
		} else {
			fmt.Printf(" [token:%d]", nextTok)
		}

		nextTok, err = runtime.ForwardToken(nextTok, pos)
		if err != nil {
			fmt.Printf("\n[Decode Error] step %d: %v\n", s, err)
			break
		}
		pos++
	}
	decodeElapsed := time.Since(t1)
	fmt.Println()

	// Performance Summary
	fmt.Println("\n--- GPU Performance Metrics ---")
	fmt.Printf("Prefill Time : %v (%d prompt tokens)\n", prefillElapsed, len(promptTokens))
	fmt.Printf("Decode Time  : %v (%d generated tokens)\n", decodeElapsed, len(generatedTokens))
	if decodeElapsed.Seconds() > 0 {
		fmt.Printf("Throughput   : %.2f tokens/sec\n", float64(len(generatedTokens))/decodeElapsed.Seconds())
	}
}
