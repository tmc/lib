// Command llama-infer demonstrates an end-to-end tensor LLM inference runtime
// executing full Hugging Face BPE tokenization, transformer decoder forward passes,
// KV-cache management, GQA causal attention, SwiGLU MLP, and token generation.
package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"github.com/tmc/lib/nvidia/x/cudatrace"
)

var (
	modelPath   = flag.String("model", "", "path to model directory containing config.json, tokenizer.json, and model.safetensors")
	promptFlag  = flag.String("prompt", "The capital of France is", "text prompt for model generation")
	stepsFlag   = flag.Int("steps", 10, "number of autoregressive tokens to generate")
	tempFlag    = flag.Float64("temp", 0.0, "sampling temperature (0.0 = greedy argmax)")
	tracePath   = flag.String("trace", "", "write Perfetto protobuf timeline trace to file (.pftrace)")
	profileMode = flag.Bool("profile", false, "enable fine-grained kernel latency profiling")
)

func main() {
	flag.Parse()
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	var tracer *cudatrace.Tracer
	if *tracePath != "" {
		f, err := os.Create(*tracePath)
		if err != nil {
			return fmt.Errorf("create trace file: %w", err)
		}
		defer f.Close()
		tracer = cudatrace.New(f)
		fmt.Printf("[Observability] Timeline trace enabled -> %s\n", *tracePath)
	}

	// 1. Load or initialize Tokenizer
	var tokenizer Tokenizer
	var tokType string

	if *modelPath != "" {
		tokFile := filepath.Join(*modelPath, "tokenizer.json")
		if hfTok, err := LoadHFTokenizer(tokFile); err == nil {
			tokenizer = hfTok
			tokType = fmt.Sprintf("HuggingFace BPE (vocab: %d)", hfTok.VocabSize())
		}
	}
	if tokenizer == nil {
		tokenizer = NewByteTokenizer()
		tokType = "Byte-Level BPE Fallback"
	}

	// 2. Load or generate Model Weights
	var weights *ModelWeights
	var cfg *ModelConfig

	if *modelPath != "" {
		w, c, err := LoadModel(*modelPath)
		if err != nil {
			fmt.Printf("[Model Notice] Failed loading weights from %s (%v). Using initialized model.\n", *modelPath, err)
		} else {
			weights = w
			cfg = c
			fmt.Printf("[Model Loader] Loaded real weights from %s (layers: %d, hidden: %d)\n",
				*modelPath, cfg.NumHiddenLayers, cfg.HiddenSize)
		}
	}

	if weights == nil {
		defaultCfg := DefaultModelConfig()
		cfg = &defaultCfg
		weights = GenerateDeterministicWeights(cfg)
		fmt.Printf("[Model Loader] Initialized testing LLaMA model (layers: %d, hidden: %d, heads: %d, kv_heads: %d)\n",
			cfg.NumHiddenLayers, cfg.HiddenSize, cfg.NumAttentionHeads, cfg.NumKeyValueHeads)
	}

	// 3. Tokenize Prompt
	var promptTokens []int32
	var err error
	if tracer != nil {
		_ = tracer.Span(nil, "Tokenize Prompt", "tokenizer", func() {
			promptTokens, err = tokenizer.Encode(*promptFlag)
		})
	} else {
		promptTokens, err = tokenizer.Encode(*promptFlag)
	}
	if err != nil {
		return fmt.Errorf("tokenize prompt: %w", err)
	}
	if len(promptTokens) == 0 {
		promptTokens = []int32{1}
	}

	fmt.Printf("\n=== Tensor LLM Inference Engine ===\n")
	fmt.Printf("Tokenizer : %s\n", tokType)
	fmt.Printf("Prompt    : %q\n", *promptFlag)
	fmt.Printf("Tokens    : %v\n", promptTokens)
	fmt.Printf("Generating: %d tokens (temp=%.2f)\n\n", *stepsFlag, *tempFlag)

	// 4. Initialize KV Cache
	headDim := cfg.HeadDim()
	kvDim := cfg.NumKeyValueHeads * headDim
	kvCache := NewKVCacheState(cfg.NumHiddenLayers, cfg.MaxSeqLen, kvDim)

	// 5. Prompt Prefill Phase
	fmt.Print("Streaming : ", *promptFlag)
	var lastLogits []float32

	tPrefill0 := time.Now()
	for pos, tokID := range promptTokens {
		lastLogits = Forward(weights, cfg, kvCache, tokID, pos)
	}
	prefillDuration := time.Since(tPrefill0)

	// 6. Autoregressive Generation Loop
	curPos := len(promptTokens)
	generatedTokens := make([]int32, 0, *stepsFlag)

	tGen0 := time.Now()
	for step := 0; step < *stepsFlag; step++ {
		if curPos >= cfg.MaxSeqLen {
			break
		}

		var nextTokID int32
		if *tempFlag <= 0 {
			nextTokID = sampleArgmax(lastLogits)
		} else {
			nextTokID = sampleTemperature(lastLogits, float32(*tempFlag))
		}

		generatedTokens = append(generatedTokens, nextTokID)
		chunk, _ := tokenizer.Decode([]int32{nextTokID})
		fmt.Print(chunk)

		// Forward next token
		if tracer != nil {
			_ = tracer.Span(nil, fmt.Sprintf("Decode Step %d", step+1), "forward_pass", func() {
				lastLogits = Forward(weights, cfg, kvCache, nextTokID, curPos)
			})
		} else {
			lastLogits = Forward(weights, cfg, kvCache, nextTokID, curPos)
		}

		curPos++
	}
	genDuration := time.Since(tGen0)

	// 7. Performance Reporting
	tokensPerSec := float64(len(generatedTokens)) / genDuration.Seconds()
	fmt.Printf("\n\n--- Performance Metrics ---\n")
	fmt.Printf("Prefill Time : %v (%d prompt tokens)\n", prefillDuration, len(promptTokens))
	fmt.Printf("Decode Time  : %v (%d generated tokens)\n", genDuration, len(generatedTokens))
	fmt.Printf("Throughput   : %.2f tokens/sec\n", tokensPerSec)

	return nil
}

func sampleArgmax(logits []float32) int32 {
	if len(logits) == 0 {
		return 0
	}
	maxIdx := int32(0)
	maxVal := logits[0]
	for i, v := range logits {
		if v > maxVal {
			maxVal = v
			maxIdx = int32(i)
		}
	}
	return maxIdx
}

func sampleTemperature(logits []float32, temp float32) int32 {
	if len(logits) == 0 {
		return 0
	}
	if temp <= 0 {
		return sampleArgmax(logits)
	}

	maxVal := logits[0]
	for _, v := range logits {
		if v > maxVal {
			maxVal = v
		}
	}

	var sumExp float32
	probs := make([]float32, len(logits))
	for i, v := range logits {
		e := float32(math.Exp(float64((v - maxVal) / temp)))
		probs[i] = e
		sumExp += e
	}

	r := rand.Float32() * sumExp
	var acc float32
	for i, p := range probs {
		acc += p
		if r <= acc {
			return int32(i)
		}
	}
	return int32(len(logits) - 1)
}
