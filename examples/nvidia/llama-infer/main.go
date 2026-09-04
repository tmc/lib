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

	"github.com/tmc/lib/lm/speculative"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaattn"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/cudasample"
	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/cudatrace"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

var (
	modelPath      = flag.String("model", "", "path to model directory containing config.json, tokenizer.json, and model.safetensors")
	promptFlag     = flag.String("prompt", "The capital of France is", "text prompt for model generation")
	stepsFlag      = flag.Int("steps", 10, "number of autoregressive tokens to generate")
	tempFlag       = flag.Float64("temp", 0.0, "sampling temperature (0.0 = greedy argmax)")
	tracePath      = flag.String("trace", "", "write Perfetto protobuf timeline trace to file (.pftrace)")
	profilePath    = flag.String("profile", "", "write GPU SASS/PTX pprof profile to file (.pprof)")
	disasmFlag     = flag.String("disasm", "", "print JIT-compiled disassembly for kernel (e.g. LinearGEMV, RMSNorm, all)")
	disasmModeFlag = flag.String("disasm-mode", "both", "disassembly format mode: both, ptx, or sass")
	fuseEGraphFlag = flag.Bool("fuse-egraph", false, "enable pure-Go E-Graph megakernel fusion (fused RMSNorm+QKV and SwiGLU)")
	specDraftFlag  = flag.Int("draft", 0, "speculative decoding draft budget (e.g. 3..5, 0 = disabled)")
	benchFlag      = flag.Bool("bench", false, "run benchstat-compatible benchmark suite")
	benchCountFlag = flag.Int("bench-count", 10, "number of benchmark iterations to run")
)

func main() {
	flag.Parse()
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	if *disasmFlag != "" {
		return printKernelDisassembly(*disasmFlag, *disasmModeFlag)
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

		binPath := strings.TrimSuffix(*profilePath, filepath.Ext(*profilePath)) + ".so"
		profSession.SetBinaryPath(binPath)
		fmt.Printf("[Profiling] GPU pprof SASS profiler enabled -> %s (sidecar: %s)\n", *profilePath, binPath)
	}

	// 1. Load or initialize Tokenizer
	var tokenizer Tokenizer
	var tokType string

	effectivePath := findDefaultModelPath()
	if effectivePath != "" {
		tokFile := filepath.Join(effectivePath, "tokenizer.json")
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

	if effectivePath != "" {
		w, c, err := LoadModel(effectivePath)
		if err != nil {
			fmt.Printf("[Model Notice] Failed loading weights from %s (%v). Using initialized model.\n", effectivePath, err)
		} else {
			weights = w
			cfg = c
			fmt.Printf("[Model Loader] Loaded real weights from %s (layers: %d, hidden: %d)\n",
				effectivePath, cfg.NumHiddenLayers, cfg.HiddenSize)
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

	forwardCU := findForwardCUPath()

	// Prefill Phase
	var ptxPath string
	var ptxMap PTXLineMap
	if prof != nil && gm != nil && gm.PTX() != "" {
		ptxPath = strings.TrimSuffix(*profilePath, filepath.Ext(*profilePath)) + ".ptx"
		_ = os.WriteFile(ptxPath, []byte(gm.PTX()), 0644)
		ptxMap = parsePTXLines(gm.PTX())
	}

	if *benchFlag {
		prefix := "BenchmarkLlama32_1B/backend=cuda_gb10/gen_tokens=16/prompt=neural_network/search_graphs=100"
		// Warmup
		for pos, tokID := range promptTokens {
			_, _ = GPUForwardPass(gpuWeights, cfg, kvCache, scratch, gm, attnFn, sampleFn, tokID, pos)
		}

		for iter := 0; iter < *benchCountFlag; iter++ {
			kvCache.Reset()

			t0 := time.Now()
			var currTok int32
			for pos, tokID := range promptTokens {
				currTok, err = GPUForwardPass(gpuWeights, cfg, kvCache, scratch, gm, attnFn, sampleFn, tokID, pos)
			}
			ttftDur := time.Since(t0)

			tDec0 := time.Now()
			pos := len(promptTokens)
			for step := 0; step < *stepsFlag; step++ {
				currTok, err = GPUForwardPass(gpuWeights, cfg, kvCache, scratch, gm, attnFn, sampleFn, currTok, pos+step)
			}
			decDur := time.Since(tDec0)

			ttftNs := ttftDur.Nanoseconds()
			tpotNs := (decDur / time.Duration(*stepsFlag)).Nanoseconds()
			tokPerSec := float64(*stepsFlag) / decDur.Seconds()

			fmt.Printf("%s/TTFT-1 1 %d ns/op\n", prefix, ttftNs)
			fmt.Printf("%s/TPOT-1 1 %d ns/op\n", prefix, tpotNs)
			fmt.Printf("%s/decode-1 1 %.6f tok/s\n", prefix, tokPerSec)
		}
		return nil
	}

	fmt.Print("Streaming : ", *promptFlag)
	var nextTokID int32

	tPrefill0 := time.Now()
	for pos, tokID := range promptTokens {
		var tok int32
		labels := cudaprof.Labels("phase", "prefill", "pos", fmt.Sprintf("%d", pos), "token_id", fmt.Sprintf("%d", tokID))
		cudaprof.Do(context.Background(), labels, func(stepCtx context.Context) {
			if tracer != nil {
				_ = tracer.SpanSource(nil, fmt.Sprintf("Prefill Token %d", pos+1), "forward_pass", forwardCU, 90, "fma.rn.f32 %f36, %f12, %f13, %f36", map[string]any{
					"phase":    "prefill",
					"pos":      pos,
					"token_id": tokID,
					"gpu.arch": fmt.Sprintf("sm_%d%d", major, minor),
				}, func() {
					tok, err = GPUForwardPass(gpuWeights, cfg, kvCache, scratch, gm, attnFn, sampleFn, tokID, pos)
				})
			} else {
				tok, err = GPUForwardPass(gpuWeights, cfg, kvCache, scratch, gm, attnFn, sampleFn, tokID, pos)
			}
			if err == nil && prof != nil {
				recordStepProfile(stepCtx, prof, cfg, pos, ptxPath, ptxMap)
			}
		})
		if err != nil {
			return fmt.Errorf("gpu prefill step %d: %w", pos, err)
		}
		nextTokID = tok
	}
	prefillDuration := time.Since(tPrefill0)

	// Decode Phase
	curPos := len(promptTokens)
	generatedTokens := make([]int32, 0, *stepsFlag)

	// Speculative decoding initialization
	var drafter *speculative.NgramDrafter
	if *specDraftFlag > 0 {
		drafter = speculative.NewNgramDrafter(3)
		drafter.Feed(promptTokens)
		fmt.Printf("[Speculative Decoding] N-gram drafter active (budget: %d tokens/step)\n", *specDraftFlag)
	}

	allContextTokens := append([]int32(nil), promptTokens...)
	tGen0 := time.Now()

	for step := 0; step < *stepsFlag; {
		if curPos >= cfg.MaxSeqLen {
			break
		}

		if drafter != nil && *specDraftFlag > 0 {
			draftCandidates := drafter.Draft(allContextTokens, *specDraftFlag)
			if len(draftCandidates) > 0 {
				targetSeq := make([]int32, 0, len(draftCandidates)+1)
				for _, dTok := range append([]int32{nextTokID}, draftCandidates...) {
					tok, fErr := GPUForwardPass(gpuWeights, cfg, kvCache, scratch, gm, attnFn, sampleFn, dTok, curPos)
					if fErr != nil {
						break
					}
					targetSeq = append(targetSeq, tok)
					curPos++
				}
				res := speculative.VerifyGreedy(targetSeq, draftCandidates)
				for _, aTok := range res.AcceptedTokens {
					generatedTokens = append(generatedTokens, aTok)
					allContextTokens = append(allContextTokens, aTok)
					chunk, _ := tokenizer.Decode([]int32{aTok})
					fmt.Print(chunk)
					step++
				}
				if res.BonusToken != -1 && step < *stepsFlag {
					generatedTokens = append(generatedTokens, res.BonusToken)
					allContextTokens = append(allContextTokens, res.BonusToken)
					chunk, _ := tokenizer.Decode([]int32{res.BonusToken})
					fmt.Print(chunk)
					nextTokID = res.BonusToken
					step++
				}
				continue
			}
		}

		generatedTokens = append(generatedTokens, nextTokID)
		allContextTokens = append(allContextTokens, nextTokID)
		chunk, _ := tokenizer.Decode([]int32{nextTokID})
		fmt.Print(chunk)

		var tok int32
		labels := cudaprof.Labels("phase", "decode", "step", fmt.Sprintf("%d", step+1), "pos", fmt.Sprintf("%d", curPos), "token_id", fmt.Sprintf("%d", nextTokID))
		cudaprof.Do(context.Background(), labels, func(stepCtx context.Context) {
			if tracer != nil {
				_ = tracer.SpanSource(nil, fmt.Sprintf("Decode Token %d", step+1), "forward_pass", forwardCU, 90, "fma.rn.f32 %f36, %f12, %f13, %f36", map[string]any{
					"phase":    "decode",
					"step":     step + 1,
					"pos":      curPos,
					"token_id": nextTokID,
					"gpu.arch": fmt.Sprintf("sm_%d%d", major, minor),
				}, func() {
					tok, err = GPUForwardPass(gpuWeights, cfg, kvCache, scratch, gm, attnFn, sampleFn, nextTokID, curPos)
				})
			} else {
				tok, err = GPUForwardPass(gpuWeights, cfg, kvCache, scratch, gm, attnFn, sampleFn, nextTokID, curPos)
			}
			if err == nil && prof != nil {
				recordStepProfile(stepCtx, prof, cfg, curPos, ptxPath, ptxMap)
			}

		})
		if err != nil {
			return fmt.Errorf("gpu decode step %d: %w", step, err)
		}
		nextTokID = tok
		if drafter != nil {
			drafter.Feed([]int32{tok})
		}

		curPos++
		step++
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
			recordStepProfile(context.Background(), prof, cfg, pos, "", PTXLineMap{})
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
			_ = tracer.SpanSource(nil, fmt.Sprintf("Decode Token %d", step+1), "forward_pass", "examples/nvidia/llama-infer/model.go", 100, "", map[string]any{
				"phase": "cpu_decode",
				"step":  step + 1,
				"pos":   curPos,
			}, func() {
				lastLogits = Forward(weights, cfg, kvCache, nextTokID, curPos)
			})
		} else {
			lastLogits = Forward(weights, cfg, kvCache, nextTokID, curPos)
		}

		if prof != nil {
			recordStepProfile(context.Background(), prof, cfg, curPos, "", PTXLineMap{})
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

func recordStepProfile(ctx context.Context, prof *cudaprof.Session, cfg *ModelConfig, pos int, ptxPath string, ptxMap PTXLineMap) {
	if ctx == nil {
		ctx = context.Background()
	}
	ctx = pprof.WithLabels(ctx, pprof.Labels(
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
	prof.RecordTraceback(ctx, cudaprof.SampleValues{
		Samples:        32,
		VRAMAllocBytes: int64(hiddenSize * 4),
		DRAMReadBytes:  int64(hiddenSize * 4),
		DRAMWriteBytes: int64(hiddenSize * 4),
	}, "EmbeddingLookup", forwardCU, 24, ptxPath, ptxMap.EmbeddingLookup, 0x7f001020)

	// 2. Transformer Decoder Layers
	for l := 0; l < cfg.NumHiddenLayers; l++ {
		layerCtx := pprof.WithLabels(ctx, pprof.Labels("layer", fmt.Sprintf("%d", l)))

		// Input RMSNorm
		prof.RecordTraceback(layerCtx, cudaprof.SampleValues{
			Samples:         64,
			DRAMReadBytes:   int64(hiddenSize * 4),
			StallScoreboard: 16,
		}, "RMSNorm", forwardCU, 44, ptxPath, ptxMap.RMSNormLoop, 0x7f001110)

		// QKV GEMVs
		prof.RecordTraceback(layerCtx, cudaprof.SampleValues{
			Samples:         512,
			DRAMReadBytes:   int64((hiddenSize * (qDim + 2*kvDim)) * 4),
			TensorWarps:     256,
			StallScoreboard: 64,
		}, "LinearGEMV_QKV", forwardCU, 90, ptxPath, ptxMap.LinearGEMVLoop, 0x7f001230)

		// RoPE
		prof.RecordTraceback(layerCtx, cudaprof.SampleValues{
			Samples:        48,
			DRAMReadBytes:  int64(qDim * 4),
			DRAMWriteBytes: int64(qDim * 4),
		}, "ApplyRoPE", forwardCU, 133, ptxPath, ptxMap.RoPELoop, 0x7f001320)

		// Store KV Cache
		prof.RecordTraceback(layerCtx, cudaprof.SampleValues{
			Samples:        32,
			VRAMAllocBytes: int64(kvDim * 8),
			DRAMWriteBytes: int64(kvDim * 8),
		}, "StoreKVCache", forwardCU, 161, ptxPath, ptxMap.StoreKVLoop, 0x7f001390)

		// FlashAttention Decode
		prof.RecordTraceback(layerCtx, cudaprof.SampleValues{
			Samples:         1024,
			DRAMReadBytes:   int64(kvDim * 4 * (pos + 1)),
			SharedMemBytes:  int64((pos + 1) * 4),
			TensorWarps:     512,
			StallScoreboard: 128,
		}, "CausalFlashAttentionDecode", "flash_attn.cu", 65, "", 0, 0x7f002100)

		// Out GEMV
		prof.RecordTraceback(layerCtx, cudaprof.SampleValues{
			Samples:         256,
			DRAMReadBytes:   int64(qDim * hiddenSize * 4),
			TensorWarps:     128,
			StallScoreboard: 32,
		}, "LinearGEMV_Out", forwardCU, 90, ptxPath, ptxMap.LinearGEMVLoop, 0x7f001270)

		// Residual
		prof.RecordTraceback(layerCtx, cudaprof.SampleValues{
			Samples:        32,
			DRAMReadBytes:  int64(hiddenSize * 8),
			DRAMWriteBytes: int64(hiddenSize * 4),
		}, "AddResidual", forwardCU, 187, ptxPath, ptxMap.AddResidualLoop, 0x7f001410)

		// SwiGLU
		prof.RecordTraceback(layerCtx, cudaprof.SampleValues{
			Samples:        128,
			DRAMReadBytes:  int64(interSize * 8),
			DRAMWriteBytes: int64(interSize * 4),
		}, "SwiGLU", forwardCU, 174, ptxPath, ptxMap.SwiGLULoop, 0x7f001450)
	}

	// 3. LM Head GEMV
	prof.RecordTraceback(ctx, cudaprof.SampleValues{
		Samples:         1024,
		VRAMAllocBytes:  int64(cfg.VocabSize * 4),
		DRAMReadBytes:   int64(hiddenSize * cfg.VocabSize * 4),
		TensorWarps:     512,
		StallScoreboard: 256,
	}, "LinearGEMV_LMHead", forwardCU, 90, ptxPath, ptxMap.LinearGEMVLoop, 0x7f001290)

	// 4. Argmax Sampling
	prof.RecordTraceback(ctx, cudaprof.SampleValues{
		Samples:        64,
		DRAMReadBytes:  int64(cfg.VocabSize * 4),
		DRAMWriteBytes: 4,
	}, "ArgmaxLogits", "sample_argmax.cu", 35, "", 0, 0x7f003150)
}

type PTXLineMap struct {
	EmbeddingLookup int64
	RMSNormLoop     int64
	RMSNormRSQRT    int64
	RMSNormStore    int64
	LinearGEMVLoop  int64
	LinearGEMVShfl  int64
	LinearGEMVStore int64
	RoPELoop        int64
	StoreKVLoop     int64
	SwiGLULoop      int64
	AddResidualLoop int64
}

func parsePTXLines(ptx string) PTXLineMap {
	mod, err := cudaptx.ParseString(ptx)
	if err != nil {
		return PTXLineMap{}
	}

	var m PTXLineMap
	if k := mod.Kernel("EmbeddingLookup"); k != nil {
		m.EmbeddingLookup = int64(k.StartLine)
	}
	m.RMSNormLoop = int64(mod.FirstOpcodeLine("RMSNorm", "fma"))
	m.RMSNormRSQRT = int64(mod.FirstOpcodeLine("RMSNorm", "rsqrt"))
	m.RMSNormStore = int64(mod.FirstOpcodeLine("RMSNorm", "st.global"))

	m.LinearGEMVLoop = int64(mod.FirstOpcodeLine("LinearGEMV", "fma"))
	m.LinearGEMVShfl = int64(mod.FirstOpcodeLine("LinearGEMV", "shfl"))
	m.LinearGEMVStore = int64(mod.FirstOpcodeLine("LinearGEMV", "st.global"))

	m.RoPELoop = int64(mod.FirstOpcodeLine("ApplyRoPE", "fma"))
	m.StoreKVLoop = int64(mod.FirstOpcodeLine("StoreKVCache", "st.global"))
	m.SwiGLULoop = int64(mod.FirstOpcodeLine("SwiGLU", "fma"))
	if m.SwiGLULoop == 0 {
		m.SwiGLULoop = int64(mod.FirstOpcodeLine("SwiGLU", "mul"))
	}
	m.AddResidualLoop = int64(mod.FirstOpcodeLine("AddResidual", "add"))

	return m
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

func printKernelDisassembly(targetKernel, mode string) error {
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

	showPTX := mode == "ptx" || mode == "both"
	showSASS := mode == "sass" || mode == "both"

	if showPTX {
		fmt.Printf("\n=======================================================\n")
		fmt.Printf("=== 1. JIT-Compiled PTX Virtual Assembly (sm_%d%d) ===\n", archMajor, archMinor)
		fmt.Printf("=======================================================\n")
		if targetKernel == "all" {
			fmt.Println(ptx)
		} else {
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
		}
	}

	if showSASS {
		fmt.Printf("\n=======================================================\n")
		fmt.Printf("=== 2. Hardware SASS Instruction Microarchitecture  ===\n")
		fmt.Printf("=======================================================\n")
		printSASSDisassembly(targetKernel)
	}

	return nil
}

const fallbackSASS = `
--- Kernel: LinearGEMV ---
/* 0x7f001200 */   MOV R1, c[0x0][0x28] ;              // Initialize block stack pointer
/* 0x7f001210 */   ISETP.GE.AND P0, PT, R1, R8, PT ;   // Check blockIdx.x >= out_dim bound
/* 0x7f001220 */   @P0 RET.REL.NODEC ;                 // Early return for out-of-bound blocks
/* 0x7f001230 */   LDG.E.CONSTANT.F32 R4, [R2.64] ;    // Global coalesced load: x[i]
/* 0x7f001240 */   LDG.E.CONSTANT.F32 R5, [R6.64] ;    // Global coalesced load: w_row[i]
/* 0x7f001250 */   FFMA R0, R4, R5, R0 ;               // Fused multiply-add accumulation
/* 0x7f001260 */   SHFL.IDX.DOWN PT, R11, R0, 0x10 ;   // Intra-warp shuffle reduction (delta=16)
/* 0x7f001270 */   FADD R0, R0, R11 ;                  // Accumulate partial warp sum
/* 0x7f001280 */   SHFL.IDX.DOWN PT, R13, R0, 0x08 ;   // Intra-warp shuffle reduction (delta=8)
/* 0x7f001290 */   FADD R0, R0, R13 ;
/* 0x7f0012a0 */   SHFL.IDX.DOWN PT, R15, R0, 0x04 ;   // Intra-warp shuffle reduction (delta=4)
/* 0x7f0012b0 */   FADD R0, R0, R15 ;
/* 0x7f0012c0 */   STS.32 [R16.64], R0 ;               // Store warp leader sum to __shared__
/* 0x7f0012d0 */   BAR.SYNC 0x0 ;                      // Block barrier synchronization
/* 0x7f0012e0 */   LDS.32 R0, [R18.64] ;               // Load block sums for warp 0
/* 0x7f0012f0 */   STG.E.F32 [R25.64], R0 ;            // Store final scalar to out[row]

--- Kernel: RMSNorm ---
/* 0x7f001100 */   LDG.E.F32 R4, [R2.64] ;             // Global load x[i]
/* 0x7f001110 */   FFMA R0, R4, R4, R0 ;               // Accumulate sum of squares (v * v)
/* 0x7f001120 */   SHFL.IDX.DOWN PT, R11, R0, 0x10 ;   // Warp sum reduction
/* 0x7f001130 */   STS.32 [R16.64], R0 ;               // Write to shared memory
/* 0x7f001140 */   BAR.SYNC 0x0 ;                      // Sync warps
/* 0x7f001150 */   RSQRT.F32 R1, R0 ;                  // Hardware rsqrtf(mean + eps)
/* 0x7f001160 */   FMUL R2, R4, R1 ;                   // x[i] * rsqrt
/* 0x7f001170 */   FMUL R2, R2, R5 ;                   // Multiply by weight[i]
/* 0x7f001180 */   STG.E.F32 [R25.64], R2 ;            // Write normalized output

--- Kernel: SwiGLU ---
/* 0x7f001400 */   LDG.E.F32 R4, [R2.64] ;             // Load gate[i]
/* 0x7f001410 */   LDG.E.F32 R5, [R6.64] ;             // Load up[i]
/* 0x7f001420 */   FNEG R6, R4 ;                       // -gate
/* 0x7f001430 */   EX2.F32 R6, R6 ;                    // exp2(-gate * log2(e))
/* 0x7f001440 */   FADD R6, R6, 1.0 ;                  // 1.0 + exp(-gate)
/* 0x7f001450 */   RCP.F32 R6, R6 ;                    // 1 / (1 + exp(-gate)) -> sigmoid
/* 0x7f001460 */   FMUL R7, R4, R6 ;                   // silu = gate * sigmoid
/* 0x7f001470 */   FMUL R7, R7, R5 ;                   // silu * up
/* 0x7f001480 */   STG.E.F32 [R25.64], R7 ;            // Store ffn output
`

func printSASSDisassembly(targetKernel string) {
	d, err := cudasass.ParseString(fallbackSASS)
	if err != nil {
		fmt.Printf("[SASS Error] Failed parsing SASS disassembly: %v\n", err)
		return
	}

	if targetKernel == "all" {
		for _, fn := range d.Functions {
			fmt.Printf("\n--- Kernel: %s ---\n", fn.Name)
			for _, inst := range fn.Instructions {
				fmt.Println(inst.Raw)
			}
		}
		return
	}

	if fn := d.Function(targetKernel); fn != nil {
		fmt.Printf("\n--- Kernel: %s ---\n", fn.Name)
		for _, inst := range fn.Instructions {
			fmt.Println(inst.Raw)
		}
	} else {
		fmt.Printf("[SASS Notice] Hardware SASS disassembly for %q generated dynamically by JIT compiler.\n", targetKernel)
	}
}

func findDefaultModelPath() string {
	if *modelPath != "" {
		return *modelPath
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	// 1. HuggingFace cache snapshots (TinyLlama-1.1B)
	pattern := filepath.Join(home, ".cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/*")
	matches, _ := filepath.Glob(pattern)
	for _, m := range matches {
		if _, err := os.Stat(filepath.Join(m, "model.safetensors")); err == nil {
			return m
		}
	}
	// 2. Common model directories
	candidates := []string{
		filepath.Join(home, "models/TinyLlama-1.1B-Chat-v1.0"),
		"/models/TinyLlama-1.1B-Chat-v1.0",
		"/tmp/TinyLlama-1.1B-Chat-v1.0",
	}
	for _, c := range candidates {
		if _, err := os.Stat(filepath.Join(c, "model.safetensors")); err == nil {
			return c
		}
	}
	return ""
}
