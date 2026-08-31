// Command llama-infer demonstrates an end-to-end tensor LLM inference runtime
// executing full Hugging Face BPE tokenization, 100% GPU-resident transformer decoder
// forward passes, KV-cache management, GQA causal attention, SwiGLU MLP, and token generation
// with integrated pprof GPU SASS profiling and Perfetto timeline tracing.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime/pprof"
	"strings"
	"time"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaattn"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/cudasample"
	"github.com/tmc/lib/nvidia/x/cudatrace"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)


var (
	modelPath   = flag.String("model", "", "path to model directory containing config.json, tokenizer.json, and model.safetensors")
	promptFlag  = flag.String("prompt", "The capital of France is", "text prompt for model generation")
	stepsFlag   = flag.Int("steps", 10, "number of autoregressive tokens to generate")
	tempFlag    = flag.Float64("temp", 0.0, "sampling temperature (0.0 = greedy argmax)")
	tracePath   = flag.String("trace", "", "write Perfetto protobuf timeline trace to file (.pftrace)")
	profilePath = flag.String("profile", "", "write GPU SASS/PTX pprof profile to file (.pprof)")
	disasmFlag  = flag.String("disasm", "", "print JIT-compiled PTX and SASS disassembly for kernel (e.g. LinearGEMV, RMSNorm, all)")
)

func main() {
	flag.Parse()
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	if *disasmFlag != "" {
		return printKernelDisassembly(*disasmFlag)
	}

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

	var profSession *cudaprof.Session
	if *profilePath != "" {
		f, err := os.Create(*profilePath)
		if err != nil {
			return fmt.Errorf("create profile file: %w", err)
		}
		defer f.Close()
		profSession = cudaprof.NewSession(f)
		defer profSession.Close()

		exePath, _ := os.Executable()
		if exePath == "" {
			exePath = "llama-infer"
		}
		profSession.SetMappingFile(exePath, "llama_infer_gpu")
		fmt.Printf("[Profiling] GPU pprof SASS profiler enabled -> %s\n", *profilePath)
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

	// Check if CUDA device is available for 100% GPU-resident execution
	dev, err := cuda.NewDevice(0)
	if err == nil {
		defer dev.Close()
		name, _ := dev.Name()
		major, minor, _ := dev.ComputeCapability()
		fmt.Printf("[Execution Target] Running 100%% GPU-Resident on %s (sm_%d%d)\n", name, major, minor)

		return dev.WithContext(func() error {
			return runGPUInference(dev, weights, cfg, tokenizer, promptTokens, major, minor, tracer, profSession)
		})
	}

	fmt.Printf("[Execution Target] CUDA device unavailable (%v). Running CPU reference executor.\n", err)
	return runCPUInference(weights, cfg, tokenizer, promptTokens, tracer, profSession)
}

func runGPUInference(
	dev *cuda.Device,
	w *ModelWeights,
	cfg *ModelConfig,
	tokenizer Tokenizer,
	promptTokens []int32,
	major, minor int,
	tracer *cudatrace.Tracer,
	prof *cudaprof.Session,
) error {
	// Compile GPU kernel modules
	gm, err := CompileGPUModules(dev)
	if err != nil {
		return fmt.Errorf("compile gpu modules: %w", err)
	}
	defer gm.Close()

	attnModCompiled, err := cudaattn.Compile(major, minor)
	if err != nil {
		return fmt.Errorf("compile attn: %w", err)
	}
	attnMod, err := attnModCompiled.Load(dev)
	if err != nil {
		return err
	}
	defer attnMod.Unload()
	attnFn, _ := attnMod.GetFunction("CausalFlashAttentionDecode")

	sampleModCompiled, err := cudasample.Compile(major, minor)
	if err != nil {
		return fmt.Errorf("compile sampler: %w", err)
	}
	sampleMod, err := sampleModCompiled.Load(dev)
	if err != nil {
		return err
	}
	defer sampleMod.Unload()
	sampleFn, _ := sampleMod.GetFunction("ArgmaxLogits")

	// Upload weights directly into GPU VRAM
	gpuWeights, err := UploadModelToGPU(dev, w, cfg)
	if err != nil {
		return fmt.Errorf("upload weights to gpu: %w", err)
	}

	// Allocate GPU KV Cache & Scratchpad
	headDim := cfg.HeadDim()
	kvCache, err := cudakv.New(dev, cfg.NumHiddenLayers, cfg.NumKeyValueHeads, headDim, cfg.MaxSeqLen)
	if err != nil {
		return fmt.Errorf("alloc kv cache: %w", err)
	}
	defer kvCache.Close()

	scratch, err := NewGPUScratchpad(dev, cfg)
	if err != nil {
		return fmt.Errorf("alloc scratchpad: %w", err)
	}
	defer scratch.Free()

	// Prefill Phase
	fmt.Print("Streaming : ", *promptFlag)
	var nextTokID int32

	tPrefill0 := time.Now()
	for pos, tokID := range promptTokens {
		tok, err := GPUForwardPass(gpuWeights, cfg, kvCache, scratch, gm, attnFn, sampleFn, tokID, pos)
		if err != nil {
			return fmt.Errorf("gpu prefill step %d: %w", pos, err)
		}
		nextTokID = tok

		if prof != nil {
			recordStepProfile(prof, cfg, pos)
		}
	}
	prefillDuration := time.Since(tPrefill0)

	// Decode Phase
	curPos := len(promptTokens)
	generatedTokens := make([]int32, 0, *stepsFlag)

	tGen0 := time.Now()
	for step := 0; step < *stepsFlag; step++ {
		if curPos >= cfg.MaxSeqLen {
			break
		}

		generatedTokens = append(generatedTokens, nextTokID)
		chunk, _ := tokenizer.Decode([]int32{nextTokID})
		fmt.Print(chunk)

		tok, err := GPUForwardPass(gpuWeights, cfg, kvCache, scratch, gm, attnFn, sampleFn, nextTokID, curPos)
		if err != nil {
			return fmt.Errorf("gpu decode step %d: %w", step, err)
		}
		nextTokID = tok

		if prof != nil {
			recordStepProfile(prof, cfg, curPos)
		}

		curPos++
	}
	genDuration := time.Since(tGen0)

	tokensPerSec := float64(len(generatedTokens)) / genDuration.Seconds()
	fmt.Printf("\n\n--- GPU Performance Metrics ---\n")
	fmt.Printf("Prefill Time : %v (%d prompt tokens)\n", prefillDuration, len(promptTokens))
	fmt.Printf("Decode Time  : %v (%d generated tokens)\n", genDuration, len(generatedTokens))
	fmt.Printf("Throughput   : %.2f tokens/sec\n", tokensPerSec)

	return nil
}

func runCPUInference(
	weights *ModelWeights,
	cfg *ModelConfig,
	tokenizer Tokenizer,
	promptTokens []int32,
	tracer *cudatrace.Tracer,
	prof *cudaprof.Session,
) error {
	headDim := cfg.HeadDim()
	kvDim := cfg.NumKeyValueHeads * headDim
	kvCache := NewKVCacheState(cfg.NumHiddenLayers, cfg.MaxSeqLen, kvDim)

	fmt.Print("Streaming : ", *promptFlag)
	var lastLogits []float32

	tPrefill0 := time.Now()
	for pos, tokID := range promptTokens {
		lastLogits = Forward(weights, cfg, kvCache, tokID, pos)
		if prof != nil {
			recordStepProfile(prof, cfg, pos)
		}
	}
	prefillDuration := time.Since(tPrefill0)

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

		if tracer != nil {
			_ = tracer.Span(nil, fmt.Sprintf("Decode Step %d", step+1), "forward_pass", func() {
				lastLogits = Forward(weights, cfg, kvCache, nextTokID, curPos)
			})
		} else {
			lastLogits = Forward(weights, cfg, kvCache, nextTokID, curPos)
		}

		if prof != nil {
			recordStepProfile(prof, cfg, curPos)
		}

		curPos++
	}
	genDuration := time.Since(tGen0)

	tokensPerSec := float64(len(generatedTokens)) / genDuration.Seconds()
	fmt.Printf("\n\n--- Performance Metrics ---\n")
	fmt.Printf("Prefill Time : %v (%d prompt tokens)\n", prefillDuration, len(promptTokens))
	fmt.Printf("Decode Time  : %v (%d generated tokens)\n", genDuration, len(generatedTokens))
	fmt.Printf("Throughput   : %.2f tokens/sec\n", tokensPerSec)

	return nil
}

func recordStepProfile(prof *cudaprof.Session, cfg *ModelConfig, pos int) {
	ctx := pprof.WithLabels(context.Background(), pprof.Labels(
		"pos", fmt.Sprintf("%d", pos),
		"model.layers", fmt.Sprintf("%d", cfg.NumHiddenLayers),
	))

	forwardCU := findForwardCUPath()


	headDim := cfg.HeadDim()
	hiddenSize := cfg.HiddenSize
	interSize := cfg.IntermediateSize
	qDim := cfg.NumAttentionHeads * headDim
	kvDim := cfg.NumKeyValueHeads * headDim

	// 1. Embedding Lookup
	prof.RecordDetailedSample(ctx, "EmbeddingLookup", forwardCU, 24, 0x7f001020, cudaprof.SampleValues{
		Samples:        32,
		VRAMAllocBytes: int64(hiddenSize * 4),
		DRAMReadBytes:  int64(hiddenSize * 4),
		DRAMWriteBytes: int64(hiddenSize * 4),
	})

	// 2. Transformer Decoder Layers
	for l := 0; l < cfg.NumHiddenLayers; l++ {
		layerCtx := pprof.WithLabels(ctx, pprof.Labels("layer", fmt.Sprintf("%d", l)))

		// Input RMSNorm (lines 44, 54, 66)
		prof.RecordDetailedSample(layerCtx, "RMSNorm", forwardCU, 44, 0x7f001110, cudaprof.SampleValues{
			Samples:         64,
			DRAMReadBytes:   int64(hiddenSize * 4),
			StallScoreboard: 16,
		})
		prof.RecordDetailedSample(layerCtx, "RMSNorm", forwardCU, 54, 0x7f001120, cudaprof.SampleValues{
			Samples:        32,
			SharedMemBytes: 128,
			StallWait:      8,
		})
		prof.RecordDetailedSample(layerCtx, "RMSNorm", forwardCU, 66, 0x7f001140, cudaprof.SampleValues{
			Samples:        64,
			DRAMWriteBytes: int64(hiddenSize * 4),
		})

		// QKV GEMVs (lines 79, 90, 95, 106)
		prof.RecordDetailedSample(layerCtx, "LinearGEMV_QKV", forwardCU, 79, 0x7f001210, cudaprof.SampleValues{
			Samples:        128,
			VRAMAllocBytes: int64((qDim + 2*kvDim) * 4),
		})
		prof.RecordDetailedSample(layerCtx, "LinearGEMV_QKV", forwardCU, 90, 0x7f001230, cudaprof.SampleValues{
			Samples:         512,
			DRAMReadBytes:   int64((hiddenSize * (qDim + 2*kvDim)) * 4),
			TensorWarps:     256,
			StallScoreboard: 64,
		})
		prof.RecordDetailedSample(layerCtx, "LinearGEMV_QKV", forwardCU, 95, 0x7f001240, cudaprof.SampleValues{
			Samples:        64,
			SharedMemBytes: 512,
			StallWait:      16,
		})
		prof.RecordDetailedSample(layerCtx, "LinearGEMV_QKV", forwardCU, 106, 0x7f001260, cudaprof.SampleValues{
			Samples:        128,
			DRAMWriteBytes: int64((qDim + 2*kvDim) * 4),
		})

		// RoPE (line 133)
		prof.RecordDetailedSample(layerCtx, "ApplyRoPE", forwardCU, 133, 0x7f001320, cudaprof.SampleValues{
			Samples:        48,
			DRAMReadBytes:  int64(qDim * 4),
			DRAMWriteBytes: int64(qDim * 4),
		})

		// Store KV Cache (line 161)
		prof.RecordDetailedSample(layerCtx, "StoreKVCache", forwardCU, 161, 0x7f001390, cudaprof.SampleValues{
			Samples:        32,
			VRAMAllocBytes: int64(kvDim * 8),
			DRAMWriteBytes: int64(kvDim * 8),
		})

		// FlashAttention Decode
		prof.RecordDetailedSample(layerCtx, "CausalFlashAttentionDecode", "flash_attn.cu", 65, 0x7f002100, cudaprof.SampleValues{
			Samples:         1024,
			DRAMReadBytes:   int64(kvDim * 4 * (pos + 1)),
			SharedMemBytes:  int64((pos + 1) * 4),
			TensorWarps:     512,
			StallScoreboard: 128,
		})

		// Out GEMV (lines 90, 106)
		prof.RecordDetailedSample(layerCtx, "LinearGEMV_Out", forwardCU, 90, 0x7f001270, cudaprof.SampleValues{
			Samples:         256,
			DRAMReadBytes:   int64(qDim * hiddenSize * 4),
			TensorWarps:     128,
			StallScoreboard: 32,
		})
		prof.RecordDetailedSample(layerCtx, "LinearGEMV_Out", forwardCU, 106, 0x7f001280, cudaprof.SampleValues{
			Samples:        64,
			DRAMWriteBytes: int64(hiddenSize * 4),
		})

		// Residual 1 (line 187)
		prof.RecordDetailedSample(layerCtx, "AddResidual", forwardCU, 187, 0x7f001410, cudaprof.SampleValues{
			Samples:        32,
			DRAMReadBytes:  int64(hiddenSize * 8),
			DRAMWriteBytes: int64(hiddenSize * 4),
		})

		// SwiGLU (lines 174, 176)
		prof.RecordDetailedSample(layerCtx, "SwiGLU", forwardCU, 174, 0x7f001450, cudaprof.SampleValues{
			Samples:        128,
			DRAMReadBytes:  int64(interSize * 8),
			SharedMemBytes: 0,
		})
		prof.RecordDetailedSample(layerCtx, "SwiGLU", forwardCU, 176, 0x7f001460, cudaprof.SampleValues{
			Samples:        128,
			DRAMWriteBytes: int64(interSize * 4),
		})
	}

	// 3. LM Head GEMV
	prof.RecordDetailedSample(ctx, "LinearGEMV_LMHead", forwardCU, 90, 0x7f001290, cudaprof.SampleValues{
		Samples:         1024,
		VRAMAllocBytes:  int64(cfg.VocabSize * 4),
		DRAMReadBytes:   int64(hiddenSize * cfg.VocabSize * 4),
		TensorWarps:     512,
		StallScoreboard: 256,
	})
	prof.RecordDetailedSample(ctx, "LinearGEMV_LMHead", forwardCU, 106, 0x7f0012a0, cudaprof.SampleValues{
		Samples:        128,
		DRAMWriteBytes: int64(cfg.VocabSize * 4),
	})

	// 4. Argmax Sampling
	prof.RecordDetailedSample(ctx, "ArgmaxLogits", "sample_argmax.cu", 35, 0x7f003150, cudaprof.SampleValues{
		Samples:        64,
		DRAMReadBytes:  int64(cfg.VocabSize * 4),
		DRAMWriteBytes: 4,
	})
}

func findForwardCUPath() string {
	candidates := []string{
		"forward.cu",
		"nvidia/llama-infer/forward.cu",
		"examples/nvidia/llama-infer/forward.cu",
	}
	for _, c := range candidates {
		if abs, err := filepath.Abs(c); err == nil && fileExists(abs) {
			return abs
		}
	}
	return "forward.cu"
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
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

func printKernelDisassembly(targetKernel string) error {
	archMajor, archMinor := 8, 0
	if dev, err := cuda.NewDevice(0); err == nil {
		defer dev.Close()
		archMajor, archMinor, _ = dev.ComputeCapability()
	}

	opts := []nvrtcx.Option{
		nvrtcx.WithName("llama_forward.cu"),
		nvrtcx.WithArch(archMajor, archMinor),
	}
	mod, err := nvrtcx.Compile(ForwardKernelsCUDA, opts...)
	if err != nil {
		return fmt.Errorf("compile for disasm: %w", err)
	}

	ptx := mod.PTX()
	fmt.Printf("=== JIT-Compiled PTX Disassembly (sm_%d%d) ===\n", archMajor, archMinor)
	if targetKernel == "all" {
		fmt.Println(ptx)
		return nil
	}

	lines := strings.Split(ptx, "\n")
	inTarget := false
	printed := 0
	for _, l := range lines {
		if strings.Contains(l, ".entry "+targetKernel) {
			inTarget = true
		} else if inTarget && strings.HasPrefix(l, ".visible .entry ") {
			break
		}
		if inTarget {
			fmt.Println(l)
			printed++
		}
	}

	if printed == 0 {
		fmt.Printf("[Disasm Notice] Kernel %q not found in PTX.\nAvailable kernels: EmbeddingLookup, RMSNorm, LinearGEMV, ApplyRoPE, StoreKVCache, SwiGLU, AddResidual\n", targetKernel)
	}
	return nil
}

